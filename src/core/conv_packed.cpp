#include "core/conv_packed.hpp"

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gemini/cheetah/hom_conv2d_ss.h"
#include "gemini/core/util/ThreadPool.h"

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

size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

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

void parallel(size_t threads, size_t n, const std::function<void(size_t)>& f) {
    gemini::ThreadPool pool(std::max<size_t>(1, std::min(threads, n)));
    gemini::LaunchWorks(pool, n, [&](long, size_t start, size_t end) {
        for (size_t i = start; i < end; i++) f(i);
        return Code::OK;
    });
}

std::string concat(const std::vector<std::string>& parts) {
    size_t n = 0;
    for (auto& p : parts) n += p.size();
    std::string s;
    s.reserve(n);
    for (auto& p : parts) s += p;
    return s;
}

} // namespace

// The block layout of troy's Conv2dHelper for a stride-1 convolution of bs H x W images. A polynomial
// has b images x ci input channels x co output-channel slots of h x w tiles: an input tile at
// [image][channel][i][j], a weight polynomial with the co filters flipped at [filter][ci-1-channel][ki][kj],
// so that their product holds output (image, filter, i, j) at slot ci-1 of the (image, filter) group.
struct PackedConv2D::Tiling {
    size_t bs, ic, oc, H, W, kh, kw, N;
    size_t b = 0, h = 0, w = 0, ci = 0, co = 0;
    size_t sh, sw, tiles, in_groups, out_groups, yh, yw, oh, ow;
    std::vector<size_t> required; // output coefficients of one ciphertext, (image, filter, i, j)-major

    Tiling(size_t bs, size_t ic, size_t oc, size_t H, size_t W, size_t kh, size_t kw, size_t N)
        : bs(bs), ic(ic), oc(oc), H(H), W(W), kh(kh), kw(kw), N(N) {
        // Conv2dHelper::determine_block, EncryptLeft: fewest input plus output ciphertexts
        size_t best = SIZE_MAX;
        for (size_t b_ = bs; b_ >= 1; b_--)
            for (size_t h_ = std::min(H, N / b_); h_ >= kh; h_--)
                for (size_t w_ = std::min(W, N / b_ / h_); w_ >= kw; w_--)
                    for (size_t co_ = std::min(oc, N / b_ / h_ / w_); co_ >= 1; co_--) {
                        size_t ci_ = std::min(N / b_ / h_ / w_ / co_, ic);
                        if (ci_ == 0)
                            continue;
                        size_t spatial = ceil_div(bs, b_) * ceil_div(H - kh + 1, h_ - kh + 1)
                                         * ceil_div(W - kw + 1, w_ - kw + 1);
                        size_t cost = spatial * (ceil_div(ic, ci_) + ceil_div(oc, co_));
                        if (cost < best)
                            best = cost, b = b_, h = h_, w = w_, ci = ci_, co = co_;
                    }
        yh = h - kh + 1, yw = w - kw + 1, oh = H - kh + 1, ow = W - kw + 1;
        sh = ceil_div(oh, yh), sw = ceil_div(ow, yw);
        tiles     = ceil_div(bs, b) * sh * sw;
        in_groups = ceil_div(ic, ci), out_groups = ceil_div(oc, co);
        for (size_t bb = 0; bb < b; bb++)
            for (size_t o = 0; o < co; o++)
                for (size_t i = 0; i < yh; i++)
                    for (size_t j = 0; j < yw; j++) required.push_back(out_index(bb, o, i, j));
    }

    size_t out_index(size_t bb, size_t o, size_t i, size_t j) const {
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
                        pt[((bb - lb) * ci * co + (c - lc)) * h * w + (r - r0) * w + (s - c0)]
                            = x[((bb * ic + c) * H + r) * W + s];
    }

    // nonzero coefficients (index, value) of weight polynomial (og, g)
    void weight_terms(const Word* wt, size_t og, size_t g, std::vector<std::pair<uint32_t, uint64_t>>& terms) const {
        terms.clear();
        size_t lo = og * co, uo = std::min(lo + co, oc), lc = g * ci, uc = std::min(lc + ci, ic);
        for (size_t o = lo; o < uo; o++)
            for (size_t c = lc; c < uc; c++)
                for (size_t a = 0; a < kh; a++)
                    for (size_t d = 0; d < kw; d++)
                        if (Word v = wt[((o * ic + c) * kh + kh - 1 - a) * kw + kw - 1 - d])
                            terms.emplace_back(((o - lo) * ci + ci - 1 - (c - lc)) * h * w + a * w + d, v);
    }

    // visit(output index in the NCHW result, coefficient index) for output ciphertext (t, og)
    template <class F>
    void outputs(size_t t, size_t og, F&& visit) const {
        size_t lb, r0, c0;
        tile(t, lb, r0, c0);
        size_t ub = std::min(lb + b, bs), lo = og * co, uo = std::min(lo + co, oc);
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

    // NTT of a weight polynomial into out (one N-block per prime). 1x1 weights go through the small NTT
    // (see one_by_one); a polynomial with few terms is evaluated from the power table, sum_k v_k psi^(e_j k)
    // accumulated in 128 bits; the rest through SEAL's NTT.
    void weight(const std::vector<std::pair<uint32_t, uint64_t>>& terms, size_t B, int a,
                const std::vector<uint32_t>& gather, uint64_t* out) const {
        for (size_t l = 0; l < primes.size(); l++) {
            const Prime& p = primes[l];
            uint64_t* o    = out + l * N;
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

void PackedConv2D::encrypt(const Tiling& t, const Word* x, std::string& out, size_t threads) const {
    std::vector<std::string> parts(t.tiles * t.in_groups);
    parallel(threads, parts.size(), [&](size_t k) {
        seal::Plaintext pt;
        t.input_poly(x, k / t.in_groups, k % t.in_groups, pt);
        std::stringstream ss;
        encryptor_->encrypt_symmetric(pt).save(ss); // seeded: half a ciphertext on the wire
        parts[k] = ss.str();
    });
    out = concat(parts);
}

void PackedConv2D::evaluate(const Tiling& t, const std::string& in, const Word* x_own, const Word* w,
                            Word* r, std::string& out, size_t threads) const {
    std::vector<seal::Ciphertext> x(t.tiles * t.in_groups);
    std::stringstream ss(in);
    for (auto& ct : x) ct.load(*context_, ss);
    parallel(threads, x.size(), [&](size_t k) {
        if (x_own) { // conv(x_other + x_own, w): the own-share term belongs to the triple too
            seal::Plaintext pt;
            t.input_poly(x_own, k / t.in_groups, k % t.in_groups, pt);
            evaluator_->add_plain_inplace(x[k], pt);
        }
        evaluator_->transform_to_ntt_inplace(x[k]);
    });

    // y[t][o] = sum_g x[t][g] * w[o][g], per output group: its weight polynomials are NTT-transformed
    // right before use (each is used once per tile, and on large images there is one tile) and the
    // products are summed in 128 bits, reduced once at the end instead of after every product.
    std::vector<seal::Ciphertext> y(t.tiles * t.out_groups);
    const Ntt& ntt = *ntt_;
    const size_t L = ntt.primes.size(), stride = L * t.N, B = t.h * t.w;
    std::vector<uint32_t> gather;
    const int a = t.kh == 1 && t.kw == 1 ? ntt.one_by_one(B, gather) : 0;
    parallel(threads, t.out_groups, [&](size_t o) {
        using u128 = unsigned __int128;
        std::vector<u128> acc(t.tiles * 2 * stride, 0);
        std::vector<uint64_t> wn(stride);
        std::vector<std::pair<uint32_t, uint64_t>> terms;
        auto reduce = [&](size_t i, size_t l) {
            return seal::util::barrett_reduce_128(reinterpret_cast<const uint64_t*>(&acc[i]), ntt.primes[l].q);
        };
        for (size_t g = 0, pending = 0; g < t.in_groups; g++) {
            t.weight_terms(w, o, g, terms);
            if (terms.empty())
                continue;
            ntt.weight(terms, B, a, gather, wn.data());
            for (size_t tile = 0; tile < t.tiles; tile++) {
                const seal::Ciphertext& ct = x[tile * t.in_groups + g];
                const uint64_t *a0 = ct.data(0), *a1 = ct.data(1);
                u128 *s0 = acc.data() + tile * 2 * stride, *s1 = s0 + stride;
                for (size_t i = 0; i < stride; i++) {
                    u128 v = wn[i];
                    s0[i] += a0[i] * v;
                    s1[i] += a1[i] * v;
                }
            }
            if (++pending == ntt.fold) {
                for (size_t i = 0; i < acc.size(); i++) acc[i] = reduce(i, i % stride / t.N);
                pending = 0;
            }
        }
        for (size_t tile = 0; tile < t.tiles; tile++) {
            seal::Ciphertext& ct = y[tile * t.out_groups + o];
            ct.resize(*context_, context_->first_parms_id(), 2);
            ct.is_ntt_form() = true;
            for (size_t p = 0; p < 2; p++)
                for (size_t i = 0; i < stride; i++) ct.data(p)[i] = reduce((tile * 2 + p) * stride + i, i / t.N);
            evaluator_->transform_from_ntt_inplace(ct);
        }
    });

    // Mask with a random polynomial (its output coefficients are this party's share), flood, truncate
    // and drop the coefficients the other party does not need
    std::vector<std::string> parts(y.size());
    const uint64_t t_mask = context_->first_context_data()->parms().plain_modulus().value() - 1;
    parallel(threads, y.size(), [&](size_t k) {
        auto prng = seal::UniformRandomGeneratorFactory::DefaultFactory()->create();
        gemini::flood_ciphertext(y[k], prng, *context_, *other_pk_, *evaluator_);
        seal::Plaintext mask(t.N);
        prng->generate(t.N * sizeof(uint64_t), reinterpret_cast<seal::seal_byte*>(mask.data()));
        std::for_each(mask.data(), mask.data() + t.N, [t_mask](uint64_t& u) { u &= t_mask; });
        evaluator_->sub_plain_inplace(y[k], mask);
        t.outputs(k / t.out_groups, k % t.out_groups, [&](size_t out, size_t coeff) { r[out] = Word(mask[coeff]); });
        gemini::truncate_for_decryption(y[k], *evaluator_, *context_);
        gemini::remove_unused_coeffs(y[k], *evaluator_, t.required);
        std::stringstream os;
        y[k].save(os);
        parts[k] = os.str();
    });
    out = concat(parts);
}

void PackedConv2D::decrypt(const Tiling& t, const std::string& in, Word* c, bool accumulate,
                           size_t threads) const {
    std::vector<seal::Ciphertext> y(t.tiles * t.out_groups);
    std::stringstream ss(in);
    for (auto& ct : y) ct.load(*context_, ss);
    parallel(threads, y.size(), [&](size_t k) {
        seal::Plaintext pt;
        decryptor_->decrypt(y[k], pt);
        t.outputs(k / t.out_groups, k % t.out_groups, [&](size_t out, size_t coeff) {
            Word v = coeff < pt.coeff_count() ? Word(pt[coeff]) : 0;
            c[out] = accumulate ? Word(c[out] + v) : v;
        });
    });
}

void PackedConv2D::conv(IO::NetIO** ios, int party, const Word* x, const Word* w, Word* c, size_t bs,
                        size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, bool is_ab,
                        size_t threads) const {
    size_t oh = ih - kh + 1, ow = iw - kw + 1;
    for (size_t cur = 0; cur < bs; cur += kMaxBatch) {
        Tiling t(std::min(kMaxBatch, bs - cur), ic, oc, ih, iw, kh, kw, POLY_MOD);
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

} // namespace Iface
