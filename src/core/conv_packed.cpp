#include "gemini/core/prg_party.h"
#include "core/conv_packed.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gemini/cheetah/hom_conv2d_ss.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"
#include "gemini/core/util/ThreadPool.h"
#include "emp-tool/utils/prg.h"

namespace gemini { // output post-processing of HomConv2DSS (hom_conv2d_ss.cc)
void flood_ciphertext(seal::Ciphertext& ct, std::shared_ptr<seal::UniformRandomGenerator> prng,
                      const seal::SEALContext& context, const seal::PublicKey& pk,
                      const seal::Evaluator& evaluator);
void truncate_for_decryption(seal::Ciphertext& ct, const seal::Evaluator& evaluator,
                             const seal::SEALContext& context);
void remove_unused_coeffs(seal::Ciphertext& ct, const seal::Evaluator& evaluator,
                          std::vector<size_t> used_indices);
} // namespace gemini

namespace Iface {

namespace {

constexpr size_t kMaxBatch = 16; // images per tiling, as on the GPU
// repacking: the tiling's price of one automorphism in bytes (CONV_REPACK_KS_BYTES)
size_t repack_ks_bytes() {
    static const size_t n = getenv("CONV_REPACK_KS_BYTES") ? size_t(atol(getenv("CONV_REPACK_KS_BYTES"))) : 2048;
    return n;
}
// AB2 weight holder: chunks evaluated at once (conv_pipelined); CONV_PIPE_EVALUATORS overrides it
size_t weight_holder_evaluators() {
    static const size_t n = getenv("CONV_PIPE_EVALUATORS") ? std::max(1, atoi(getenv("CONV_PIPE_EVALUATORS"))) : 3;
    return n;
}

size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

size_t live_threads(size_t threads) {
    const size_t n = conv_threads_now().load(std::memory_order_relaxed);
    return n ? std::min(n, threads) : threads;
}

void send(IO::NetIO** ios, const std::string& s) {
    size_t n = s.size();
    ios[0]->send_data(&n, sizeof(n));
    if (n)
        ios[0]->send_data(s.data(), n);
    ios[0]->flush();
}

std::string recv(IO::NetIO** ios) {
    size_t n = 0;
    ios[0]->recv_data(&n, sizeof(n));
    std::string s(n, '\0');
    if (n)
        ios[0]->recv_data(s.data(), n);
    return s;
}

// f(0), ..., f(n - 1) on up to `threads` workers of a pool kept across calls (a layer makes several
// calls, and a new pool per call started all its threads each time), handing out indices one at a time
// (the work per index varies)
void parallel(size_t threads, size_t n, const std::function<void(size_t)>& f) {
    size_t k = std::min(threads, n);
    if (k <= 1) {
        for (size_t i = 0; i < n; i++) f(i);
        return;
    }
    static std::mutex mutex;
    static std::map<size_t, std::unique_ptr<gemini::ThreadPool>> pools;
    gemini::ThreadPool* pool;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& p = pools[k];
        if (!p)
            p = std::make_unique<gemini::ThreadPool>(k);
        pool = p.get();
    }
    std::atomic<size_t> next{0};
    gemini::LaunchWorks(*pool, k, [&](long, size_t, size_t) {
        for (size_t i; (i = next++) < n;) f(i);
        return Code::OK;
    });
}

// SEAL generator on AES-128 in counter mode (emp's PRG, AES-NI, seeded from the OS): the flooding
// noise, the output masks and the zero encryptions draw several words per output coefficient, which
// made SEAL's Blake2 generator a visible cost. Output only, never stored in a seeded ciphertext.
class AesPrng final : public seal::UniformRandomGenerator {
  public:
    AesPrng() : seal::UniformRandomGenerator(seal::prng_seed_type{}) {}
    // the stream (a, b) of this party (seeded runs: PRG_SEED != -1)
    AesPrng(uint64_t a, uint64_t b) : seal::UniformRandomGenerator(seal::prng_seed_type{}), prg_(&block_of(a, b)) {}
    // the same stream on both parties, for the public polynomial of a seeded ciphertext
    explicit AesPrng(const emp::block& seed) : seal::UniformRandomGenerator(seal::prng_seed_type{}), prg_(&seed) {}

  protected:
    seal::prng_type type() const noexcept override { return seal::prng_type::unknown; }
    void refill_buffer() override { prg_.random_data(buffer_begin_, int(buffer_size_)); }

  private:
    static const emp::block& block_of(uint64_t a, uint64_t b) {
        thread_local emp::block seed;
        uint64_t s[2];
        gemini::party_seed(a, b, s);
        seed = emp::makeBlock(s[1], s[0]);
        return seed;
    }
    emp::PRG prg_;
};

// a fresh generator for task k of the n-th call of a stage (random seeds when PRG_SEED == -1)
std::shared_ptr<AesPrng> task_prng(uint64_t stage, uint64_t call, uint64_t k) {
#if PRG_SEED != -1
    return std::make_shared<AesPrng>((stage << 56) | call, k);
#else
    return std::make_shared<AesPrng>();
#endif
}

// FIFO between the stages of conv_pipelined
template <class T>
class Pipe {
  public:
    void push(T v) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(v));
        }
        cv_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        T v = std::move(queue_.front());
        queue_.pop_front();
        return v;
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> queue_;
};

// n values below 2^bits as a little-endian bit stream of ceil(n * bits / 8) bytes
size_t packed_size(size_t n, int bits) { return (n * bits + 7) / 8; }

void pack_bits(const uint64_t* v, size_t n, int bits, uint8_t* dst) {
    unsigned __int128 acc = 0;
    int have = 0;
    for (size_t i = 0; i < n; i++) {
        acc |= static_cast<unsigned __int128>(v[i]) << have;
        for (have += bits; have >= 64; have -= 64, acc >>= 64, dst += 8) std::memcpy(dst, &acc, 8);
    }
    for (; have > 0; have -= 8, acc >>= 8) *dst++ = uint8_t(acc);
}

void unpack_bits(const uint8_t* src, size_t n, int bits, uint64_t* v) {
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    const uint8_t* end  = src + packed_size(n, bits);
    unsigned __int128 acc = 0;
    int have = 0;
    for (size_t i = 0; i < n; i++) {
        while (have < bits) {
            uint64_t word = 0;
            size_t take   = std::min<size_t>(8, end - src);
            std::memcpy(&word, src, take);
            acc |= static_cast<unsigned __int128>(word) << have;
            src += take, have += 8 * int(take);
        }
        v[i] = uint64_t(acc) & mask;
        acc >>= bits, have -= bits;
    }
}

} // namespace

// Ciphertexts on the wire. An input ciphertext (c0, c1) at both primes has c1 = a expanded from a public
// 16-byte seed (a is uniform, taken as its NTT form), so only the seed and c0 travel, at the primes'
// widths. An output ciphertext is truncated for decryption like HomConv2DSS's (one prime; c0 keeps its high
// `c0_bits`, c1 its high `c1_bits`), and only the kept bits of c1 and of c0's used coefficients travel.
struct PackedConv2D::Wire {
    size_t N;
    std::vector<int> q_bits; // widths of the primes of an input ciphertext
    size_t in_bytes;         // one input ciphertext
    int c0_shift, c0_bits, c1_shift, c1_bits;
    size_t out_bytes(size_t used) const { return packed_size(used, c0_bits) + packed_size(N, c1_bits); }
};

// The block layout of troy's Conv2dHelper for a stride-1 convolution of bs H x W images. A polynomial
// has b images x ci input channels x co output-channel slots of h x w tiles: an input tile at
// [image][channel][i][j], a weight polynomial with the co filters flipped at [filter][ci-1-channel][ki][kj],
// so that their product holds output (image, filter, i, j) at slot ci-1 of the (image, filter) group.
//
// Repacked (rp, conv_repack()): b images x h x w positions x C channel slots, channels fastest (C a power of
// two, ci = min(C, ic) of them used), and one filter per weight polynomial, with the weights of channel c at
// X^(C * position - c): the product holds output (image, i, j) at the multiple of C of its position, and
// every other coefficient is a sum over mismatched channels (never at a multiple of C, also where it wraps
// around). The products of C filters of a tile are merged into one ciphertext (pack): filter r at residue r.
struct PackedConv2D::Tiling {
    size_t bs, ic, oc, H, W, kh, kw, N;
    bool rp;
    size_t b = 0, h = 0, w = 0, ci = 0, co = 0, C = 1;
    // eval_groups: products per tile (weight polynomials per input group); out_groups: output ciphertexts per
    // tile (rp: dense ones of C filters each)
    size_t sh, sw, tiles, in_groups, out_groups, eval_groups, yh, yw, oh, ow;
    std::vector<size_t> required; // output coefficients of one ciphertext, (image, filter, i, j)-major

    Tiling(size_t bs, size_t ic, size_t oc, size_t H, size_t W, size_t kh, size_t kw, const Wire& wire, bool rp = false)
        : bs(bs), ic(ic), oc(oc), H(H), W(W), kh(kh), kw(kw), N(wire.N), rp(rp) {
        size_t best = SIZE_MAX;
        if (!rp) {
            // Conv2dHelper::determine_block's search, for the fewest bytes on the wire: an input ciphertext
            // costs about twice an output one, so the layout leans towards fewer inputs
            for (size_t b_ = bs; b_ >= 1; b_--)
                for (size_t h_ = std::min(H, N / b_); h_ >= kh; h_--)
                    for (size_t w_ = std::min(W, N / b_ / h_); w_ >= kw; w_--)
                        for (size_t co_ = std::min(oc, N / b_ / h_ / w_); co_ >= 1; co_--) {
                            size_t ci_ = std::min(N / b_ / h_ / w_ / co_, ic);
                            if (ci_ == 0)
                                continue;
                            size_t spatial = ceil_div(bs, b_) * ceil_div(H - kh + 1, h_ - kh + 1)
                                             * ceil_div(W - kw + 1, w_ - kw + 1);
                            size_t used = b_ * co_ * (h_ - kh + 1) * (w_ - kw + 1);
                            size_t cost = spatial * (ceil_div(ic, ci_) * wire.in_bytes + ceil_div(oc, co_) * wire.out_bytes(used));
                            if (cost < best)
                                best = cost, b = b_, h = h_, w = w_, ci = ci_, co = co_;
                        }
        } else {
            // the fewest bytes, inputs and dense outputs, with every automorphism of the packing counted as
            // repack_ks_bytes() more: their time is what repacking costs (pure bytes take many small tiles with
            // many channel slots, one automorphism per product)
            const size_t ks_bytes = repack_ks_bytes();
            for (size_t b_ = bs; b_ >= 1; b_--)
                for (size_t C_ = 1; b_ * C_ * kh * kw <= N; C_ *= 2)
                    for (size_t h_ = std::min(H, N / b_ / C_); h_ >= kh; h_--)
                        for (size_t w_ = std::min(W, N / b_ / C_ / h_); w_ >= kw; w_--) {
                            size_t spatial = ceil_div(bs, b_) * ceil_div(H - kh + 1, h_ - kh + 1)
                                             * ceil_div(W - kw + 1, w_ - kw + 1);
                            size_t used = b_ * std::min(C_, oc) * (h_ - kh + 1) * (w_ - kw + 1);
                            size_t cost = spatial * (ceil_div(ic, std::min(C_, ic)) * wire.in_bytes
                                                     + ceil_div(oc, C_) * (wire.out_bytes(used) + (C_ - 1) * ks_bytes));
                            if (cost < best)
                                best = cost, b = b_, h = h_, w = w_, C = C_;
                        }
            ci = std::min(C, ic), co = 1;
        }
        yh = h - kh + 1, yw = w - kw + 1, oh = H - kh + 1, ow = W - kw + 1;
        sh = ceil_div(oh, yh), sw = ceil_div(ow, yw);
        tiles     = ceil_div(bs, b) * sh * sw;
        in_groups = ceil_div(ic, ci);
        out_groups = rp ? ceil_div(oc, C) : ceil_div(oc, co);
        eval_groups = rp ? oc : out_groups;
        const size_t g = rp ? std::min(C, oc) : co; // filters of an output ciphertext
        for (size_t bb = 0; bb < b; bb++)
            for (size_t o = 0; o < g; o++)
                for (size_t i = 0; i < yh; i++)
                    for (size_t j = 0; j < yw; j++) required.push_back(out_index(bb, o, i, j));
    }

    size_t out_index(size_t bb, size_t o, size_t i, size_t j) const {
        if (rp)
            return ((bb * h + h - yh + i) * w + (w - yw + j)) * C + o;
        return (bb * ci * co + o * ci + ci - 1) * h * w + (h - yh + i) * w + (w - yw + j);
    }

    // first image, first output row and column of tile t
    void tile(size_t t, size_t& lb, size_t& r0, size_t& c0) const {
        lb = t / (sh * sw) * b, r0 = t % (sh * sw) / sw * yh, c0 = t % sw * yw;
    }

    void input_poly(const Word* x, size_t t, size_t g, seal::Plaintext& pt) const {
        size_t lb, r0, c0;
        tile(t, lb, r0, c0);
        pt.resize(N);
        std::fill_n(pt.data(), N, 0);
        size_t ub = std::min(lb + b, bs), lc = g * ci, uc = std::min(lc + ci, ic);
        size_t ur = std::min(r0 + h, H), uw = std::min(c0 + w, W);
        for (size_t bb = lb; bb < ub; bb++)
            for (size_t c = lc; c < uc; c++)
                for (size_t r = r0; r < ur; r++)
                    for (size_t s = c0; s < uw; s++)
                        pt[rp ? (((bb - lb) * h + r - r0) * w + (s - c0)) * C + (c - lc)
                              : ((bb - lb) * ci * co + (c - lc)) * h * w + (r - r0) * w + (s - c0)]
                            = x[((bb * ic + c) * H + r) * W + s];
    }

    // coefficients (index, value) of weight polynomial (og, g) that hold a weight (og: an output group, rp: a
    // filter). Zero weights are kept: the work must not depend on the weights' values (with the weights known in
    // preprocessing, skipping zeros would leak their number through the response time, and zero dummy weights
    // would look fast).
    void weight_terms(const Word* wt, size_t og, size_t g, std::vector<std::pair<uint32_t, uint64_t>>& terms) const {
        terms.clear();
        size_t lo = rp ? og : og * co, uo = rp ? og + 1 : std::min(lo + co, oc), lc = g * ci, uc = std::min(lc + ci, ic);
        for (size_t o = lo; o < uo; o++)
            for (size_t c = lc; c < uc; c++)
                for (size_t a = 0; a < kh; a++)
                    for (size_t d = 0; d < kw; d++) {
                        Word v = wt[((o * ic + c) * kh + kh - 1 - a) * kw + kw - 1 - d];
                        if (!rp) {
                            terms.emplace_back(((o - lo) * ci + ci - 1 - (c - lc)) * h * w + a * w + d, v);
                            continue;
                        }
                        // X^(C (a w + d) - c): below X^0 it wraps to X^(N + ...) with the sign flipped
                        size_t e = (a * w + d) * C;
                        if (e >= c - lc)
                            terms.emplace_back(uint32_t(e - (c - lc)), v);
                        else
                            terms.emplace_back(uint32_t(N + e - (c - lc)), Word(Word(0) - v));
                    }
    }

    // visit(output index in the NCHW result, coefficient index) for output ciphertext (t, og)
    template <class F>
    void outputs(size_t t, size_t og, F&& visit) const {
        size_t lb, r0, c0;
        tile(t, lb, r0, c0);
        const size_t g = rp ? C : co;
        size_t ub = std::min(lb + b, bs), lo = og * g, uo = std::min(lo + g, oc);
        for (size_t bb = lb; bb < ub; bb++)
            for (size_t o = lo; o < uo; o++)
                for (size_t i = 0; i < yh && r0 + i < oh; i++)
                    for (size_t j = 0; j < yw && c0 + j < ow; j++)
                        visit(((bb * oc + o) * oh + r0 + i) * ow + c0 + j, out_index(bb - lb, o - lo, i, j));
    }
};

struct PackedConv2D::Ntt {
    struct Prime {
        seal::Modulus q;
        const seal::util::NTTTables* tables;
        uint64_t increment;             // SEAL's centered lift: plaintext coefficients >= threshold get it added
        std::vector<uint64_t> power;    // psi^m, m < 2N
        std::vector<uint32_t> exponent; // NTT slot j holds the evaluation at psi^exponent[j]
    };
    // Negacyclic NTTs of size N >> a (a = 1 .. log N - 1) over the same primes, and which of their slots
    // holds a given evaluation point
    struct Small {
        std::unique_ptr<seal::util::NTTTables> tables;
        std::unordered_map<uint64_t, uint32_t> slot_of;
    };
    size_t N;
    uint64_t threshold;
    std::vector<Prime> primes;
    std::vector<std::vector<Small>> small; // [a][prime]
    size_t fold; // products a 128-bit accumulator takes before it has to be reduced

    // A 1x1 weight polynomial is V(X^B) (one weight per slot of B coefficients). For B = 2^a m, m odd,
    // X^B maps the N evaluation points onto the N >> a points of the size-(N >> a) negacyclic NTT:
    // NTT(W)[j] = NTT_small(V)[gather[j]]. Returns a = 0 when that does not save anything.
    int one_by_one(size_t B, std::vector<uint32_t>& gather) const {
        int a = __builtin_ctzll(B);
        if (a == 0 || (N >> a) < 2)
            return 0;
        gather.resize(primes.size() * N);
        for (size_t l = 0; l < primes.size(); l++)
            for (size_t j = 0; j < N; j++)
                gather[l * N + j] = small[a][l].slot_of.at(primes[l].power[(primes[l].exponent[j] * B) & (2 * N - 1)]);
        return a;
    }

    uint64_t lift(uint64_t v, const Prime& p) const { return v >= threshold ? v + p.increment : v; }

    // NTT of a weight polynomial modulo prime l into o. 1x1 weights go through the small NTT
    // (see one_by_one); a polynomial with few terms is evaluated from the power table, sum_k v_k psi^(e_j k)
    // accumulated in 128 bits; the rest through SEAL's NTT.
    void weight(const std::vector<std::pair<uint32_t, uint64_t>>& terms, size_t B, int a,
                const std::vector<uint32_t>& gather, size_t l, uint64_t* o) const {
        {
            const Prime& p = primes[l];
            if (a > 0) {
                thread_local std::vector<uint64_t> v;
                v.assign(N >> a, 0);
                for (auto [k, w] : terms) v[k / B] = lift(w, p);
                seal::util::ntt_negacyclic_harvey(v.data(), *small[a][l].tables);
                const uint32_t* gl = gather.data() + l * N;
                for (size_t j = 0; j < N; j++) o[j] = v[gl[j]];
            } else if (terms.size() <= sparse_terms) {
                using u128 = unsigned __int128;
                thread_local std::vector<u128> acc;
                acc.assign(N, 0);
                const size_t mask = 2 * N - 1;
                for (auto [k, v] : terms) {
                    u128 lv = lift(v, p);
                    for (size_t j = 0; j < N; j++) acc[j] += lv * p.power[(p.exponent[j] * k) & mask];
                }
                for (size_t j = 0; j < N; j++) o[j] = seal::util::barrett_reduce_128(reinterpret_cast<const uint64_t*>(&acc[j]), p.q);
            } else {
                std::fill_n(o, N, 0);
                for (auto [k, v] : terms) o[k] = lift(v, p);
                seal::util::ntt_negacyclic_harvey(o, *p.tables);
            }
        }
    }
    static constexpr size_t sparse_terms = 5; // above, a full NTT is cheaper
};

void PackedConv2D::setUp(const seal::SEALContext& context, const seal::SecretKey& sk,
                         std::shared_ptr<seal::PublicKey> other_pk) {
    context_   = std::make_shared<seal::SEALContext>(context);
    encryptor_ = std::make_shared<seal::Encryptor>(*context_, sk);
    evaluator_ = std::make_shared<seal::Evaluator>(*context_);
    decryptor_ = std::make_shared<seal::Decryptor>(*context_, sk);
    other_pk_  = std::move(other_pk);
    sk_        = std::make_shared<seal::SecretKey>(sk);

    // truncate_for_decryption's widths (hom_conv2d_ss.cc): c0 drops n_delta_bits - 1 low bits (one more
    // when t is not a power of two), c1 n_delta_bits - n_var_bits
    auto wire = std::make_shared<Wire>();
    {
        const auto& first = context_->first_context_data()->parms();
        const auto& last  = context_->last_context_data()->parms();
        wire->N           = first.poly_modulus_degree();
        wire->in_bytes    = sizeof(emp::block);
        for (const auto& q : first.coeff_modulus()) {
            wire->q_bits.push_back(q.bit_count());
            wire->in_bytes += packed_size(wire->N, q.bit_count());
        }
        const int q_bits       = last.coeff_modulus()[0].bit_count();
        const uint64_t t       = last.plain_modulus().value();
        const int n_delta_bits = q_bits - last.plain_modulus().bit_count();
        const int one_more_bit = (t & (t - 1)) == 0 ? 0 : 1;
        const int n_var_bits   = int(std::log2(12. * double(wire->N) * std::sqrt(1 / 18.)));
        wire->c0_shift         = std::clamp(n_delta_bits - 1 - one_more_bit, 0, 63);
        wire->c1_shift         = std::clamp(n_delta_bits - n_var_bits, 0, 63);
        wire->c0_bits          = q_bits - wire->c0_shift;
        wire->c1_bits          = q_bits - wire->c1_shift;
    }
    wire_ = wire;

    auto cd        = context_->first_context_data();
    auto ntt       = std::make_shared<Ntt>();
    ntt->N         = cd->parms().poly_modulus_degree();
    ntt->threshold = cd->plain_upper_half_threshold();
    ntt->fold      = SIZE_MAX;
    size_t N       = ntt->N;
    for (size_t l = 0; l < cd->parms().coeff_modulus().size(); l++) {
        Ntt::Prime p{cd->parms().coeff_modulus()[l], &cd->small_ntt_tables()[l], cd->plain_upper_half_increment()[l], {}, {}};
        p.power.resize(2 * N);
        p.power[0] = 1;
        for (size_t m = 1; m < 2 * N; m++) p.power[m] = seal::util::multiply_uint_mod(p.power[m - 1], p.tables->get_root(), p.q);
        std::unordered_map<uint64_t, uint32_t> exponent_of;
        for (size_t m = 0; m < 2 * N; m++) exponent_of[p.power[m]] = m;
        std::vector<uint64_t> probe(N, 0); // NTT(X) lists the evaluation points in SEAL's slot order
        probe[1] = 1;
        seal::util::ntt_negacyclic_harvey(probe.data(), *p.tables);
        p.exponent.resize(N);
        for (size_t j = 0; j < N; j++) p.exponent[j] = exponent_of.at(probe[j]);
        int bits  = p.q.bit_count();
        ntt->fold = std::min(ntt->fold, 2 * bits >= 127 ? size_t(1) : (size_t(1) << std::min(30, 127 - 2 * bits)) - 1);
        ntt->primes.push_back(std::move(p));
    }
    int logN = __builtin_ctzll(N);
    ntt->small.resize(logN);
    for (int a = 1; a < logN; a++)
        for (auto& p : ntt->primes) {
            Ntt::Small sm;
            sm.tables = std::make_unique<seal::util::NTTTables>(logN - a, p.q);
            std::vector<uint64_t> probe(N >> a, 0);
            probe[1] = 1;
            seal::util::ntt_negacyclic_harvey(probe.data(), *sm.tables);
            for (size_t j = 0; j < probe.size(); j++) sm.slot_of[probe[j]] = j;
            ntt->small[a].push_back(std::move(sm));
        }
    ntt_ = ntt;
}

void PackedConv2D::setUpRepack(IO::NetIO** ios, int party, bool both_evaluate) {
    constexpr size_t N = 8192;
    seal::EncryptionParameters params(seal::scheme_type::bfv);
    params.set_poly_modulus_degree(N);
    params.set_n_special_primes(1); // the last prime is only for key switching
    params.set_coeff_modulus(seal::CoeffModulus::Create(N, {60, 49, 60}));
    params.set_plain_modulus(PLAIN_MOD);
#if PRG_SEED != -1
    params.set_random_generator(std::make_shared<seal::Blake2xbPRNGFactory>(seal::prng_seed_type{gemini::party_seed64(0x8192)}));
#endif
    seal::SEALContext ctx(params, true, seal::sec_level_type::tc128);
    if (!ctx.using_keyswitching())
        throw std::runtime_error("PackedConv2D: the repacking context has no key switching");
    seal::KeyGenerator keygen(ctx);
    auto pk = std::make_shared<seal::PublicKey>(), other_pk = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pk);
    auto put = [&](const auto& obj) {
        std::stringstream ss;
        obj.save(ss);
        send(ios, ss.str());
    };
    auto get = [&](auto& obj) {
        std::stringstream ss(recv(ios));
        obj.load(ctx, ss);
    };
    // ALICE first, as Keys::exchange_keys: two blocking sends of MBs could deadlock on full socket buffers
    if (party == emp::ALICE)
        put(*pk), get(*other_pk);
    else
        get(*other_pk), put(*pk);
    // Galois keys for X -> X^(N/m + 1), m = 1, 2, ..., N/2: the evaluator packs the other party's products
    std::vector<uint32_t> elts;
    for (size_t m = 1; m < N; m *= 2) elts.push_back(uint32_t(N / m + 1));
    auto gk = std::make_shared<seal::GaloisKeys>();
    const bool evaluates = both_evaluate || party == emp::ALICE, encrypts = both_evaluate || party == emp::BOB;
    for (int turn : {emp::ALICE, emp::BOB}) {
        if (party == turn && encrypts)
            put(keygen.create_galois_keys(elts));
        if (party != turn && evaluates)
            get(*gk);
    }
    const seal::SecretKey sk = keygen.secret_key();
    setUp(ctx, sk, other_pk);
    if (evaluates)
        other_gk_ = gk;
    repack_ = true;
}

void PackedConv2D::encrypt(const Tiling& t, const Word* x, std::string& out, size_t threads) const {
    threads = live_threads(threads);
    const Wire& wire = *wire_;
    const auto& cd   = *context_->first_context_data();
    const auto& q    = cd.parms().coeff_modulus();
    const size_t N = wire.N, L = q.size();
    out.assign(t.tiles * t.in_groups * wire.in_bytes, '\0');
    const uint64_t call = enc_calls_++;
    parallel(threads, t.tiles * t.in_groups, [&](size_t k) {
        seal::Plaintext pt;
        t.input_poly(x, k / t.in_groups, k % t.in_groups, pt);
        // c0 = -a*s + e + Delta*m with a uniform from a fresh public seed and e from a secret generator
        emp::block seed;
        task_prng(1, call, k)->generate(sizeof(seed), reinterpret_cast<seal::seal_byte*>(&seed));
        thread_local std::vector<uint64_t> c0, e;
        c0.resize(L * N), e.resize(L * N);
        seal::util::sample_poly_uniform(std::make_shared<AesPrng>(seed), cd.parms(), c0.data());
        seal::util::SEAL_NOISE_SAMPLER(task_prng(2, call, k), cd.parms(), e.data());
        for (size_t l = 0; l < L; l++) {
            uint64_t* c = c0.data() + l * N;
            seal::util::dyadic_product_coeffmod(c, sk_->data().data() + l * N, N, q[l], c);
            seal::util::negate_poly_coeffmod(c, N, q[l], c);
            seal::util::inverse_ntt_negacyclic_harvey(c, cd.small_ntt_tables()[l]);
            seal::util::add_poly_coeffmod(c, e.data() + l * N, N, q[l], c);
        }
        seal::util::multiply_add_plain_with_scaling_variant(pt, cd, seal::util::RNSIter(c0.data(), N));
        auto* dst = reinterpret_cast<uint8_t*>(out.data()) + k * wire.in_bytes;
        std::memcpy(dst, &seed, sizeof(seed));
        dst += sizeof(seed);
        for (size_t l = 0; l < L; l++) {
            pack_bits(c0.data() + l * N, N, wire.q_bits[l], dst);
            dst += packed_size(N, wire.q_bits[l]);
        }
    });
}

void PackedConv2D::evaluate(const Tiling& t, const std::string& in, const Word* x_own, const Word* w,
                            Word* r, std::string& out, size_t threads, uint64_t call_arg) const {
    threads = live_threads(threads);
    const Wire& wire = *wire_;
    std::vector<seal::Ciphertext> x(t.tiles * t.in_groups);
    if (in.size() != x.size() * wire.in_bytes)
        throw std::runtime_error("PackedConv2D: unexpected size of the encrypted input");
    parallel(threads, x.size(), [&](size_t k) {
        // c0 as sent, c1 = a straight in NTT form from the seed
        const auto& cd  = *context_->first_context_data();
        const size_t N = wire.N, L = cd.parms().coeff_modulus().size();
        const auto* src = reinterpret_cast<const uint8_t*>(in.data()) + k * wire.in_bytes;
        emp::block seed;
        std::memcpy(&seed, src, sizeof(seed));
        src += sizeof(seed);
        x[k].resize(*context_, context_->first_parms_id(), 2);
        uint64_t* c0 = x[k].data(0);
        for (size_t l = 0; l < L; l++) {
            unpack_bits(src, N, wire.q_bits[l], c0 + l * N);
            src += packed_size(N, wire.q_bits[l]);
        }
        if (x_own) { // conv(x_other + x_own, w): the own-share term belongs to the triple too
            seal::Plaintext pt;
            t.input_poly(x_own, k / t.in_groups, k % t.in_groups, pt);
            seal::util::multiply_add_plain_with_scaling_variant(pt, cd, seal::util::RNSIter(c0, N));
        }
        for (size_t l = 0; l < L; l++) seal::util::ntt_negacyclic_harvey(c0 + l * N, cd.small_ntt_tables()[l]);
        seal::util::sample_poly_uniform(std::make_shared<AesPrng>(seed), cd.parms(), x[k].data(1));
        x[k].is_ntt_form() = true;
    });

    // y[t][o] = sum_g x[t][g] * w[o][g], per output group: its weight polynomials are NTT-transformed
    // right before use (each is used once per tile, and on large images there is one tile) and the
    // products are summed in 128 bits, reduced once at the end instead of after every product.
    std::vector<seal::Ciphertext> y(t.tiles * t.eval_groups);
    const Ntt& ntt = *ntt_;
    const size_t L = ntt.primes.size(), B = t.h * t.w;
    std::vector<uint32_t> gather;
    const int a = t.kh == 1 && t.kw == 1 && !t.rp ? ntt.one_by_one(B, gather) : 0;
    // Tasks of kBlock output groups and one prime: each input slice is read once for kBlock output
    // groups (the inputs of a layer did not stay in cache, and every output group read all of them), and
    // the accumulators of the block (kBlock * tiles * 2 * N * 16 bytes), its weights and the slice being
    // read fit in a core's L2. The inverse NTT follows with the masking below.
    constexpr size_t kBlock = 2;
    const size_t blocks = (t.eval_groups + kBlock - 1) / kBlock;
    parallel(threads, y.size(), [&](size_t k) {
        y[k].resize(*context_, context_->first_parms_id(), 2);
        y[k].is_ntt_form() = true;
    });
    parallel(threads, blocks * L, [&](size_t task) {
        using u128 = unsigned __int128;
        const size_t N = t.N, l = task % L, o0 = task / L * kBlock, nb = std::min(kBlock, t.eval_groups - o0);
        const seal::Modulus& q = ntt.primes[l].q;
        thread_local std::vector<u128> acc;
        thread_local std::vector<uint64_t> wn;
        thread_local std::vector<std::pair<uint32_t, uint64_t>> terms;
        acc.assign(nb * t.tiles * 2 * N, 0);
        wn.resize(nb * N);
        auto reduce = [&](size_t i) { return seal::util::barrett_reduce_128(reinterpret_cast<const uint64_t*>(&acc[i]), q); };
        for (size_t g = 0, pending = 0; g < t.in_groups; g++) {
            size_t used = 0; // bit ob: output group o0 + ob has weights in input group g
            for (size_t ob = 0; ob < nb; ob++) {
                t.weight_terms(w, o0 + ob, g, terms);
                if (terms.empty())
                    continue;
                ntt.weight(terms, B, a, gather, l, wn.data() + ob * N);
                used |= size_t(1) << ob;
            }
            if (!used)
                continue;
            for (size_t tile = 0; tile < t.tiles; tile++) {
                const seal::Ciphertext& ct = x[tile * t.in_groups + g];
                const uint64_t *a0 = ct.data(0) + l * N, *a1 = ct.data(1) + l * N;
                if (nb == 2 && used == 3) {
                    u128 *s0 = acc.data() + tile * 2 * N, *s1 = s0 + N;
                    u128 *r0 = acc.data() + (t.tiles + tile) * 2 * N, *r1 = r0 + N;
                    const uint64_t *w0 = wn.data(), *w1 = wn.data() + N;
                    for (size_t i = 0; i < N; i++) {
                        u128 x0 = a0[i], x1 = a1[i];
                        s0[i] += x0 * w0[i];
                        s1[i] += x1 * w0[i];
                        r0[i] += x0 * w1[i];
                        r1[i] += x1 * w1[i];
                    }
                } else
                    for (size_t ob = 0; ob < nb; ob++) {
                        if (!(used >> ob & 1))
                            continue;
                        u128 *s0 = acc.data() + (ob * t.tiles + tile) * 2 * N, *s1 = s0 + N;
                        const uint64_t* wv = wn.data() + ob * N;
                        for (size_t i = 0; i < N; i++) {
                            u128 v = wv[i];
                            s0[i] += a0[i] * v;
                            s1[i] += a1[i] * v;
                        }
                    }
            }
            if (++pending == ntt.fold) {
                for (size_t i = 0; i < acc.size(); i++) acc[i] = reduce(i);
                pending = 0;
            }
        }
        for (size_t ob = 0; ob < nb; ob++)
            for (size_t tile = 0; tile < t.tiles; tile++) {
                seal::Ciphertext& ct = y[tile * t.eval_groups + o0 + ob];
                for (size_t p = 0; p < 2; p++)
                    for (size_t i = 0; i < N; i++) ct.data(p)[l * N + i] = reduce(((ob * t.tiles + tile) * 2 + p) * N + i);
            }
    });

    // Mask with a random polynomial (its output coefficients are this party's share), flood, truncate
    // and drop the coefficients the other party does not need
    const uint64_t call = call_arg == UINT64_MAX ? eval_calls_++ : call_arg;
    const uint64_t t_mask = context_->first_context_data()->parms().plain_modulus().value() - 1;
    const size_t out_bytes = wire.out_bytes(t.required.size());
    std::vector<seal::Ciphertext*> z(t.tiles * t.out_groups); // the output ciphertexts
    if (t.rp)
        pack(t, y, z, threads);
    else
        for (size_t k = 0; k < z.size(); k++) z[k] = &y[k];
    out.assign(z.size() * out_bytes, '\0');
    parallel(threads, z.size(), [&](size_t k) {
        seal::Ciphertext& ct = *z[k];
        if (ct.is_ntt_form())
            evaluator_->transform_from_ntt_inplace(ct);
        auto prng = task_prng(3, call, k);
        gemini::flood_ciphertext(ct, prng, *context_, *other_pk_, *evaluator_);
        seal::Plaintext mask(t.N);
        prng->generate(t.N * sizeof(uint64_t), reinterpret_cast<seal::seal_byte*>(mask.data()));
        std::for_each(mask.data(), mask.data() + t.N, [t_mask](uint64_t& u) { u &= t_mask; });
        evaluator_->sub_plain_inplace(ct, mask);
        t.outputs(k / t.out_groups, k % t.out_groups, [&](size_t out, size_t coeff) { r[out] = Word(mask[coeff]); });
        gemini::truncate_for_decryption(ct, *evaluator_, *context_);
        // the kept high bits of c0's used coefficients, then of c1
        thread_local std::vector<uint64_t> v;
        v.resize(std::max(t.required.size(), t.N));
        const uint64_t *c0 = ct.data(0), *c1 = ct.data(1);
        uint64_t low = 0;
        for (size_t j = 0; j < t.required.size(); j++) {
            low |= c0[t.required[j]] & ((uint64_t(1) << wire.c0_shift) - 1);
            v[j] = c0[t.required[j]] >> wire.c0_shift;
        }
        auto* dst = reinterpret_cast<uint8_t*>(out.data()) + k * out_bytes;
        pack_bits(v.data(), t.required.size(), wire.c0_bits, dst);
        dst += packed_size(t.required.size(), wire.c0_bits);
        for (size_t i = 0; i < t.N; i++) {
            low |= c1[i] & ((uint64_t(1) << wire.c1_shift) - 1);
            v[i] = c1[i] >> wire.c1_shift;
        }
        pack_bits(v.data(), t.N, wire.c1_bits, dst);
        if (low || ct.coeff_modulus_size() != 1)
            throw std::runtime_error("PackedConv2D: output not truncated as expected");
    });
}

// PackLWEs' merge tree (Chen, Dai, Kim, Song 2021) on RLWE ciphertexts whose used coefficients sit at the
// multiples of C: node(s, m) packs the filters r = s mod m of a group, used at the multiples of m, as
// node(s, 2m) + X^m node(s + m, 2m) + sigma(node(s, 2m) - X^m node(s + m, 2m)), sigma: X -> X^(N/m + 1). sigma
// fixes X^(2mk) and negates X^(m(2k+1)), so the used coefficients add up twice and the others of the two halves
// cancel, while coefficients off the multiples of m stay off them: after m = C/2, ..., 1, filter r of the group
// is at residue r, doubled log2(C) times. The products are multiplied by C^-1 mod q first (q is odd), which the
// doublings cancel exactly; only the key-switching noise of the automorphisms adds up (far below the flooding).
// A missing half (the group's filters past oc) counts as zero. C - 1 automorphisms per output ciphertext.
void PackedConv2D::pack(const Tiling& t, std::vector<seal::Ciphertext>& y, std::vector<seal::Ciphertext*>& dense,
                        size_t threads) const {
    const auto& cd = *context_->first_context_data();
    const auto& q  = cd.parms().coeff_modulus();
    const size_t N = t.N, L = q.size(), C = t.C, G = t.out_groups, trees = t.tiles * G;
    parallel(threads, y.size(), [&](size_t k) {
        evaluator_->transform_from_ntt_inplace(y[k]);
        for (size_t l = 0; l < L; l++) {
            uint64_t inv = 1;
            if (!seal::util::try_invert_uint_mod(C, q[l], inv))
                throw std::runtime_error("PackedConv2D: C is not invertible");
            for (size_t p = 0; p < 2; p++)
                seal::util::multiply_poly_scalar_coeffmod(y[k].data(p) + l * N, N, inv, q[l], y[k].data(p) + l * N);
        }
    });
    std::vector<seal::Ciphertext*> node(trees * C, nullptr); // node[tree * C + s]
    for (size_t tile = 0; tile < t.tiles; tile++)
        for (size_t og = 0; og < G; og++)
            for (size_t r = 0; r < C && og * C + r < t.oc; r++)
                node[(tile * G + og) * C + r] = &y[tile * t.oc + og * C + r];
    // X^m ct in place
    auto shift = [&](seal::Ciphertext& ct, size_t m) {
        thread_local std::vector<uint64_t> tmp;
        tmp.resize(N);
        for (size_t p = 0; p < 2; p++)
            for (size_t l = 0; l < L; l++) {
                uint64_t* c = ct.data(p) + l * N;
                seal::util::negacyclic_shift_poly_coeffmod(c, N, m, q[l], tmp.data());
                std::copy(tmp.begin(), tmp.end(), c);
            }
    };
    for (size_t m = C / 2; m >= 1; m /= 2) {
        const uint32_t elt = uint32_t(N / m + 1);
        parallel(threads, trees * m, [&](size_t task) {
            seal::Ciphertext*& a = node[task / m * C + task % m];
            seal::Ciphertext* b  = node[task / m * C + task % m + m];
            if (!a && !b)
                return;
            seal::Ciphertext v;
            if (!b) { // a + sigma(a)
                v = *a;
                evaluator_->apply_galois_inplace(v, elt, *other_gk_);
                evaluator_->add_inplace(*a, v);
                return;
            }
            shift(*b, m);
            if (!a) { // X^m b - sigma(X^m b)
                v = *b;
                evaluator_->apply_galois_inplace(v, elt, *other_gk_);
                evaluator_->sub_inplace(*b, v);
                a = b;
                return;
            }
            v = *a;
            evaluator_->sub_inplace(v, *b);
            evaluator_->add_inplace(*a, *b);
            evaluator_->apply_galois_inplace(v, elt, *other_gk_);
            evaluator_->add_inplace(*a, v);
        });
        if (m == 1)
            break;
    }
    for (size_t k = 0; k < trees; k++) dense[k] = node[k * C];
}

void PackedConv2D::decrypt(const Tiling& t, const std::string& in, Word* c, bool accumulate,
                           size_t threads) const {
    threads = live_threads(threads);
    const Wire& wire       = *wire_;
    const size_t out_bytes = wire.out_bytes(t.required.size()), cts = t.tiles * t.out_groups;
    if (in.size() != cts * out_bytes)
        throw std::runtime_error("PackedConv2D: unexpected size of the encrypted output");
    parallel(threads, cts, [&](size_t k) {
        seal::Ciphertext ct(*context_, context_->last_parms_id());
        ct.resize(2);
        uint64_t *c0 = ct.data(0), *c1 = ct.data(1);
        std::fill_n(c0, t.N, 0);
        thread_local std::vector<uint64_t> v;
        v.resize(std::max(t.required.size(), t.N));
        const auto* src = reinterpret_cast<const uint8_t*>(in.data()) + k * out_bytes;
        unpack_bits(src, t.required.size(), wire.c0_bits, v.data());
        for (size_t j = 0; j < t.required.size(); j++) c0[t.required[j]] = v[j] << wire.c0_shift;
        unpack_bits(src + packed_size(t.required.size(), wire.c0_bits), t.N, wire.c1_bits, v.data());
        for (size_t i = 0; i < t.N; i++) c1[i] = v[i] << wire.c1_shift;
        seal::Plaintext pt;
        decryptor_->decrypt(ct, pt);
        t.outputs(k / t.out_groups, k % t.out_groups, [&](size_t out, size_t coeff) {
            Word m = coeff < pt.coeff_count() ? Word(pt[coeff]) : 0;
            c[out] = accumulate ? Word(c[out] + m) : m;
        });
    });
}

void PackedConv2D::conv(IO::NetIO** ios, int party, const Word* x, const Word* w, Word* c, size_t bs,
                        size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, bool is_ab,
                        size_t threads) const {
    size_t oh = ih - kh + 1, ow = iw - kw + 1;
    for (size_t cur = 0; cur < bs; cur += kMaxBatch) {
        Tiling t(std::min(kMaxBatch, bs - cur), ic, oc, ih, iw, kh, kw, *wire_, repack_);
        const Word* xc = x ? x + cur * ic * ih * iw : nullptr;
        Word* cc       = c + cur * oc * oh * ow;
        std::string mine, theirs, y;
        if (!is_ab && !w) { // AB2 input holder: c = conv(x, w) - r
            encrypt(t, xc, mine, threads);
            send(ios, mine);
            decrypt(t, recv(ios), cc, false, threads);
        } else if (!is_ab) { // AB2 weight holder: c = r
            evaluate(t, recv(ios), xc, w, cc, y, threads);
            send(ios, y);
        } else { // AB: c = r_own + (conv(x, w_other) - r_other)
            encrypt(t, xc, mine, threads);
            if (party == 1)
                send(ios, mine), theirs = recv(ios);
            else
                theirs = recv(ios), send(ios, mine);
            evaluate(t, theirs, xc, w, cc, y, threads);
            if (party == 1)
                send(ios, y), theirs = recv(ios);
            else
                theirs = recv(ios), send(ios, y);
            decrypt(t, theirs, cc, true, threads);
        }
    }
}

void PackedConv2D::conv_pipelined(IO::NetIO** ios, int party, const std::vector<size_t>& batch,
                                  const std::function<Job(size_t)>& prepare, bool is_ab, size_t threads) const {
    if (batch.empty())
        return;
    // chunks of at most kMaxBatch images of a convolution, as in conv()
    struct Chunk {
        std::shared_ptr<Job> job;
        std::shared_ptr<const Tiling> t;
        const Word* x;
        Word* c;
        bool last;
    };
    size_t total = 0;
    for (size_t b : batch) total += ceil_div(b, kMaxBatch);
    auto first = std::make_shared<Job>(prepare(0));
    // f(chunk) for all chunks in order, preparing each convolution when its first chunk is reached
    auto chunks = [&](auto&& f) {
        for (size_t i = 0; i < batch.size(); i++) {
            auto job  = i == 0 ? first : std::make_shared<Job>(prepare(i));
            size_t oh = job->ih - job->kh + 1, ow = job->iw - job->kw + 1;
            for (size_t cur = 0; cur < job->bs; cur += kMaxBatch) {
                auto t = std::make_shared<const Tiling>(std::min(kMaxBatch, job->bs - cur), job->ic, job->oc, job->ih,
                                                        job->iw, job->kh, job->kw, *wire_, repack_);
                f(Chunk{job, t, job->x ? job->x + cur * job->ic * job->ih * job->iw : nullptr,
                        job->c + cur * job->oc * oh * ow, cur + kMaxBatch >= job->bs});
            }
        }
    };
    auto finish = [](const Chunk& ch) {
        if (ch.last && ch.job->finish)
            ch.job->finish();
    };
    // each channel carries the messages of one stage in one direction
    IO::NetIO** enc_out = ios + (party == 1 ? 0 : 1);
    IO::NetIO** enc_in  = ios + (party == 1 ? 1 : 0);
    IO::NetIO** res_out = ios + (party == 1 ? 2 : 3);
    IO::NetIO** res_in  = ios + (party == 1 ? 3 : 2);
    Pipe<Chunk> to_evaluate, to_decrypt;
    Pipe<std::string> inbox;
    auto encrypt_all = [&](Pipe<Chunk>& next) {
        chunks([&](Chunk ch) {
            std::string s;
            encrypt(*ch.t, ch.x, s, threads);
            send(enc_out, s);
            next.push(std::move(ch));
        });
    };
    auto receive_all = [&] {
        for (size_t k = 0; k < total; k++) inbox.push(recv(enc_in));
    };

    if (!is_ab && !first->w) { // AB2 input holder: encrypts ahead, decrypts behind
        std::thread enc(encrypt_all, std::ref(to_decrypt));
        for (size_t k = 0; k < total; k++) {
            Chunk ch = to_decrypt.pop();
            decrypt(*ch.t, recv(res_in), ch.c, false, threads);
            finish(ch);
        }
        enc.join();
    } else if (!is_ab) { // AB2 weight holder: evaluates as the encryptions arrive
        // Consecutive chunks are evaluated concurrently, as an AB party overlaps its encryption, evaluation
        // and decryption: an evaluation's parallel regions have fewer tasks than threads on small layers, and
        // one chunk after another left the other threads idle (CIFAR convs: AB2 slower than AB, which does
        // twice the work). The outputs are sent in chunk order, and chunk k uses PRNG call call0 + k, the call
        // the serial loop gave it, so the triples are the same bits.
        std::thread rx(receive_all);
        const uint64_t call0 = eval_calls_;
        eval_calls_ += total;
        struct Work {
            size_t k;
            Chunk ch;
            std::string in;
        };
        Pipe<std::shared_ptr<Work>> work; // nullptr ends an evaluator
        std::mutex m;
        std::condition_variable cv;
        std::vector<std::string> out(total);
        std::vector<char> done(total, 0);
        std::map<const Job*, size_t> left; // chunks of a convolution still to evaluate (finish() after the last)
        std::thread sender([&] {
            for (size_t k = 0; k < total; k++) {
                std::string y;
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [&] { return done[k] != 0; });
                    y = std::move(out[k]);
                }
                send(res_out, y);
            }
        });
        std::vector<std::thread> evaluators;
        for (size_t e = 0; e < weight_holder_evaluators(); e++)
            evaluators.emplace_back([&] {
                while (auto wk = work.pop()) {
                    std::string y;
                    evaluate(*wk->ch.t, wk->in, wk->ch.x, wk->ch.job->w, wk->ch.c, y, threads, call0 + wk->k);
                    bool last;
                    {
                        std::lock_guard<std::mutex> lock(m);
                        out[wk->k] = std::move(y);
                        done[wk->k] = 1;
                        auto it = left.find(wk->ch.job.get());
                        last = --it->second == 0;
                        if (last)
                            left.erase(it); // a later job may be allocated at the same address
                    }
                    cv.notify_all();
                    if (last && wk->ch.job->finish)
                        wk->ch.job->finish();
                }
            });
        size_t k = 0;
        chunks([&](Chunk ch) {
            {
                std::lock_guard<std::mutex> lock(m);
                left.emplace(ch.job.get(), ceil_div(ch.job->bs, kMaxBatch)); // set at the job's first chunk
            }
            work.push(std::make_shared<Work>(Work{k++, std::move(ch), inbox.pop()}));
        });
        for (size_t e = 0; e < weight_holder_evaluators(); e++) work.push(nullptr);
        for (auto& t : evaluators) t.join();
        sender.join();
        rx.join();
    } else { // AB: both roles, c = r_own + (conv(x, w_other) - r_other)
        // As the AB2 weight holder: several chunks are evaluated at once (chunk k with PRNG call call0 + k),
        // the results go out in chunk order.
        std::thread enc(encrypt_all, std::ref(to_evaluate));
        std::thread rx(receive_all);
        const uint64_t call0 = eval_calls_;
        eval_calls_ += total;
        std::mutex m;
        std::condition_variable cv;
        std::vector<std::string> out(total);
        std::vector<char> done(total, 0);
        std::vector<Chunk> evaluated(total);
        std::mutex take;  // the next chunk and its input are taken together, in order
        size_t next = 0;
        std::vector<std::thread> evaluators;
        for (size_t e = 0; e < weight_holder_evaluators(); e++)
            evaluators.emplace_back([&] {
                for (;;) {
                    size_t k;
                    Chunk ch;
                    std::string in;
                    {
                        std::lock_guard<std::mutex> lock(take);
                        if (next == total)
                            return;
                        k  = next++;
                        ch = to_evaluate.pop();
                        in = inbox.pop();
                    }
                    std::string y;
                    evaluate(*ch.t, in, ch.x, ch.job->w, ch.c, y, threads, call0 + k);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        out[k]       = std::move(y);
                        evaluated[k] = std::move(ch);
                        done[k]      = 1;
                    }
                    cv.notify_all();
                }
            });
        std::thread sender([&] {
            for (size_t k = 0; k < total; k++) {
                std::string y;
                Chunk ch;
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [&] { return done[k] != 0; });
                    y  = std::move(out[k]);
                    ch = std::move(evaluated[k]);
                }
                // before sending: the other party reads y only once its own evaluation of this chunk is
                // sent, so both sends would block on full socket buffers if decrypting waited for them
                to_decrypt.push(std::move(ch));
                send(res_out, y);
            }
        });
        for (size_t k = 0; k < total; k++) {
            Chunk ch = to_decrypt.pop();
            decrypt(*ch.t, recv(res_in), ch.c, true, threads);
            finish(ch);
        }
        enc.join();
        rx.join();
        for (auto& t : evaluators) t.join();
        sender.join();
    }
}

} // namespace Iface
