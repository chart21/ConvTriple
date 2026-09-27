#ifndef CONV_LAYOUT_HPP_
#define CONV_LAYOUT_HPP_

// Reduces a padded, strided convolution to stride-1 convolutions over unpadded inputs, for the HE
// conv routines that only implement those (GPU/troy and the packed CPU path).

#include <algorithm>
#include <cstddef>
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

// Shares of conv(x, w) for `factor` lanes of bs/factor images each (lane i at x + i*|x|, w + i*|w|,
// c + i*|c|), strided and zero-padded. run(x, w, c, bs, ic, ih, iw, kh, kw) must compute the stride-1,
// unpadded convolution of one lane into c (bs x oc x (ih-kh+1) x (iw-kw+1)); x or w may be null.
template <class T, class Run>
void strided_conv(const T* x, const T* w, T* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                  size_t kw, size_t oc, size_t stride, size_t padding, int factor, Run&& run) {
    std::vector<T> padded;
    if (padding) {
        auto dim = Utils::pad_zero(x, padded, ic, ih, iw, padding, bs);
        ih       = std::get<0>(dim);
        iw       = std::get<1>(dim);
        if (x)
            x = padded.data();
    }
    size_t batch = bs / factor;
    size_t nh = (ih - kh) / stride + 1, nw = (iw - kw) / stride + 1;
    size_t x_size = batch * ic * ih * iw, w_size = oc * ic * kh * kw, c_size = batch * oc * nh * nw;

    size_t R = std::min(stride, kh), T_ = std::min(stride, kw);
    size_t ic1 = ic * R * T_, ih1 = (ih + stride - 1) / stride, iw1 = (iw + stride - 1) / stride;
    size_t kh1 = (kh + stride - 1) / stride, kw1 = (kw + stride - 1) / stride;
    size_t oh1 = ih1 - kh1 + 1, ow1 = iw1 - kw1 + 1;
    std::vector<T> x1, w1, c1;

    for (int i = 0; i < factor; ++i) {
        const T* xi = x ? x + x_size * i : nullptr;
        const T* wi = w ? w + w_size * i : nullptr;
        T* ci       = c + c_size * i;
        if (stride == 1) {
            run(xi, wi, ci, batch, ic, ih, iw, kh, kw);
            continue;
        }
        if (xi) {
            x1.resize(batch * ic1 * ih1 * iw1);
            polyphase_input(xi, x1.data(), batch, ic, ih, iw, stride, R, T_);
            xi = x1.data();
        }
        if (wi) {
            w1.resize(oc * ic1 * kh1 * kw1);
            polyphase_weights(wi, w1.data(), oc, ic, kh, kw, stride, R, T_);
            wi = w1.data();
        }
        c1.resize(batch * oc * oh1 * ow1);
        run(xi, wi, c1.data(), batch, ic1, ih1, iw1, kh1, kw1);
        // the phase split can leave one extra output row/column: crop to nh x nw
        for (size_t n = 0; n < batch * oc; n++)
            for (size_t h = 0; h < nh; h++)
                std::copy_n(c1.data() + (n * oh1 + h) * ow1, nw, ci + (n * nh + h) * nw);
    }
}

} // namespace ConvLayout

#endif
