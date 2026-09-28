#ifndef CONV_LAYOUT_HPP_
#define CONV_LAYOUT_HPP_

// Reduces a padded, strided convolution to stride-1 convolutions over unpadded inputs, for the HE
// conv routines that only implement those (GPU/troy and the packed CPU path).

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "constants.hpp"

namespace ConvLayout {

// Polyphase split of a stride-s convolution into a stride-1 one over R*T phase channels
// (R = min(s, kh), T = min(s, kw)): phase (r, t) of the input, x[c][s*u + r][s*v + t], becomes channel
// (c*R + r)*T + t, and the taps w[o][c][s*a + r][s*b + t] form a ceil(k/s) kernel (zero past k). The
// stride-1 convolution then yields exactly the strided outputs, instead of computing every output
// position and discarding all but 1/s^2 of them. A 1x1 kernel keeps only phase (0, 0): plain subsampling.
template <class T>
void polyphase_input(const T* x, T* dst, size_t bs, size_t ic, size_t ih, size_t iw, size_t s, size_t R,
                     size_t T_) {
    size_t H = (ih + s - 1) / s, W = (iw + s - 1) / s;
    std::fill(dst, dst + bs * ic * R * T_ * H * W, T(0));
    for (size_t b = 0; b < bs; b++)
        for (size_t c = 0; c < ic; c++)
            for (size_t r = 0; r < R; r++)
                for (size_t t = 0; t < T_; t++)
                    for (size_t u = 0; u < H && s * u + r < ih; u++)
                        for (size_t v = 0; v < W && s * v + t < iw; v++)
                            dst[(((b * ic + c) * R + r) * T_ + t) * H * W + u * W + v]
                                = x[((b * ic + c) * ih + s * u + r) * iw + s * v + t];
}

template <class T>
void polyphase_weights(const T* w, T* dst, size_t oc, size_t ic, size_t kh, size_t kw, size_t s, size_t R,
                       size_t T_) {
    size_t KH = (kh + s - 1) / s, KW = (kw + s - 1) / s;
    std::fill(dst, dst + oc * ic * R * T_ * KH * KW, T(0));
    for (size_t o = 0; o < oc; o++)
        for (size_t c = 0; c < ic; c++)
            for (size_t r = 0; r < R; r++)
                for (size_t t = 0; t < T_; t++)
                    for (size_t a = 0; a < KH && s * a + r < kh; a++)
                        for (size_t b = 0; b < KW && s * b + t < kw; b++)
                            dst[(((o * ic + c) * R + r) * T_ + t) * KH * KW + a * KW + b]
                                = w[((o * ic + c) * kh + s * a + r) * kw + s * b + t];
}

// A padded, strided convolution of bs images reduced to a stride-1, unpadded one: x, w and c are the
// reduced convolution's operands (x or w null when not given; c its output buffer), finish() moves its
// result into the caller's output c (bs x oc x nh x nw).
template <class T>
struct Reduced {
    const T* x = nullptr;
    const T* w = nullptr;
    T* c       = nullptr;
    size_t bs, ic, ih, iw, kh, kw, oc;

    Reduced(const T* x_, const T* w_, T* out, size_t bs_, size_t ic_, size_t ih_, size_t iw_, size_t kh_,
            size_t kw_, size_t oc_, size_t stride, size_t padding)
        : bs(bs_), oc(oc_), out_(out) {
        if (padding) {
            auto dim = Utils::pad_zero(x_, padded_, ic_, ih_, iw_, padding, bs_);
            ih_      = std::get<0>(dim);
            iw_      = std::get<1>(dim);
            if (x_)
                x_ = padded_.data();
        }
        nh_ = (ih_ - kh_) / stride + 1, nw_ = (iw_ - kw_) / stride + 1;
        if (stride == 1) {
            x = x_, w = w_, c = out;
            ic = ic_, ih = ih_, iw = iw_, kh = kh_, kw = kw_;
            return;
        }
        size_t R = std::min(stride, kh_), T_ = std::min(stride, kw_);
        ic = ic_ * R * T_, ih = (ih_ + stride - 1) / stride, iw = (iw_ + stride - 1) / stride;
        kh = (kh_ + stride - 1) / stride, kw = (kw_ + stride - 1) / stride;
        if (x_) {
            x1_.resize(bs * ic * ih * iw);
            polyphase_input(x_, x1_.data(), bs, ic_, ih_, iw_, stride, R, T_);
            x = x1_.data();
        }
        if (w_) {
            w1_.resize(oc * ic * kh * kw);
            polyphase_weights(w_, w1_.data(), oc, ic_, kh_, kw_, stride, R, T_);
            w = w1_.data();
        }
        padded_ = {};
        c1_.resize(bs * oc * (ih - kh + 1) * (iw - kw + 1));
        c = c1_.data();
    }

    Reduced(const Reduced&)            = delete; // x, w and c may point into the object
    Reduced& operator=(const Reduced&) = delete;

    void finish() {
        if (c == out_)
            return;
        // the phase split can leave one extra output row/column: crop to nh x nw
        size_t oh1 = ih - kh + 1, ow1 = iw - kw + 1;
        for (size_t n = 0; n < bs * oc; n++)
            for (size_t h = 0; h < nh_; h++)
                std::copy_n(c1_.data() + (n * oh1 + h) * ow1, nw_, out_ + (n * nh_ + h) * nw_);
    }

  private:
    T* out_;
    size_t nh_, nw_;
    std::vector<T> padded_, x1_, w1_, c1_;
};

// Shares of conv(x, w) for `factor` lanes of bs/factor images each (lane i at x + i*|x|, w + i*|w|,
// c + i*|c|), strided and zero-padded. run(x, w, c, bs, ic, ih, iw, kh, kw) must compute the stride-1,
// unpadded convolution of one lane into c (bs x oc x (ih-kh+1) x (iw-kw+1)); x or w may be null.
template <class T, class Run>
void strided_conv(const T* x, const T* w, T* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                  size_t kw, size_t oc, size_t stride, size_t padding, int factor, Run&& run) {
    size_t batch = bs / factor;
    size_t ph = ih + 2 * padding, pw = iw + 2 * padding;
    size_t x_size = batch * ic * ih * iw, w_size = oc * ic * kh * kw;
    size_t c_size = batch * oc * ((ph - kh) / stride + 1) * ((pw - kw) / stride + 1);
    for (int i = 0; i < factor; ++i) {
        Reduced<T> r(x ? x + x_size * i : nullptr, w ? w + w_size * i : nullptr, c + c_size * i, batch, ic, ih, iw,
                     kh, kw, oc, stride, padding);
        run(r.x, r.w, r.c, r.bs, r.ic, r.ih, r.iw, r.kh, r.kw);
        r.finish();
    }
}

} // namespace ConvLayout

#endif
