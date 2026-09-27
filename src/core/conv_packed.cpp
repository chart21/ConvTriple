#include "core/conv_packed.hpp"

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
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

    void weight_poly(const Word* wt, size_t og, size_t g, seal::Plaintext& pt) const {
        pt.resize(N);
        std::fill_n(pt.data(), N, 0);
        size_t lo = og * co, uo = std::min(lo + co, oc), lc = g * ci, uc = std::min(lc + ci, ic);
        for (size_t o = lo; o < uo; o++)
            for (size_t c = lc; c < uc; c++)
                for (size_t a = 0; a < kh; a++)
                    for (size_t d = 0; d < kw; d++)
                        pt[((o - lo) * ci + ci - 1 - (c - lc)) * h * w + a * w + d]
                            = wt[((o * ic + c) * kh + kh - 1 - a) * kw + kw - 1 - d];
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

void PackedConv2D::setUp(const seal::SEALContext& context, const seal::SecretKey& sk,
                         std::shared_ptr<seal::PublicKey> other_pk) {
    context_   = std::make_shared<seal::SEALContext>(context);
    encryptor_ = std::make_shared<seal::Encryptor>(*context_, sk);
    evaluator_ = std::make_shared<seal::Evaluator>(*context_);
    decryptor_ = std::make_shared<seal::Decryptor>(*context_, sk);
    other_pk_  = std::move(other_pk);
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

    // Weight plaintexts in NTT form, a bounded number of output groups at a time
    std::vector<seal::Ciphertext> y(t.tiles * t.out_groups);
    size_t chunk = std::max<size_t>(1, 4096 / t.in_groups);
    std::vector<seal::Plaintext> wp;
    for (size_t o0 = 0; o0 < t.out_groups; o0 += chunk) {
        size_t n = std::min(chunk, t.out_groups - o0);
        wp.assign(n * t.in_groups, seal::Plaintext());
        parallel(threads, wp.size(), [&](size_t k) {
            t.weight_poly(w, o0 + k / t.in_groups, k % t.in_groups, wp[k]);
            evaluator_->transform_to_ntt_inplace(wp[k], context_->first_parms_id());
        });
        parallel(threads, t.tiles * n, [&](size_t k) {
            size_t tile = k / n, o = k % n;
            seal::Ciphertext& acc = y[tile * t.out_groups + o0 + o];
            seal::Ciphertext prod;
            for (size_t g = 0; g < t.in_groups; g++) {
                const auto& ct = x[tile * t.in_groups + g];
                const auto& pt = wp[o * t.in_groups + g];
                if (g == 0)
                    evaluator_->multiply_plain(ct, pt, acc);
                else {
                    evaluator_->multiply_plain(ct, pt, prod);
                    evaluator_->add_inplace(acc, prod);
                }
            }
            evaluator_->transform_from_ntt_inplace(acc);
        });
    }

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
