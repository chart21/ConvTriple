#ifndef CONV_GPU_CUH_
#define CONV_GPU_CUH_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "io/net_io_channel.hpp"

using std::vector;

namespace troy {
class HeContext;
}

namespace TROY {

using INT_TYPE = uint32_t;

std::shared_ptr<troy::HeContext> setup();

void conv2d(IO::NetIO** ios, int party, const INT_TYPE* a, const INT_TYPE* b, INT_TYPE* c,
            size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
            size_t stride, size_t padding, bool mod_switch = false, int factor = 1, bool is_ab = false);

void conv2d_dummy(IO::NetIO** ios, int party, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                  size_t kw, size_t oc, size_t stride, size_t padding, bool mod_switch = false);

void conv2d_ab2(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                size_t stride, bool mod_switch);

void conv2d_ab2_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w,
                        INT_TYPE* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                        size_t kw, size_t oc, size_t stride, bool mod_switch);

void conv2d_ab(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
               size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
               size_t stride, bool mod_switch);

template <class T>
bool vector_equal(const vector<T>& a, const vector<T>& b);
vector<INT_TYPE> ideal_conv(const INT_TYPE* x, const INT_TYPE* w, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                            size_t stride = 1);
vector<INT_TYPE> random_polynomial(size_t size, uint64_t max_value = (1UL << 32));
void random_ring(INT_TYPE* dst, size_t n); // uniform words from a CSPRNG

void add_inplace(std::vector<INT_TYPE>& a, const INT_TYPE* b, size_t t);
size_t apply_stride(INT_TYPE* dest, const INT_TYPE* x, const size_t& stride, const size_t& bs,
                    const size_t& ic, const size_t& ih, const size_t& iw, const size_t& kh,
                    const size_t& kw, const size_t& oc, const size_t batch_offset = 0);

template <class T>
void send(T** ios, const std::stringstream& ss);

template <class T>
std::stringstream recv(T** ios);

inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
    __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
}

inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) { return (a + b) % t; }

template <class T>
inline void add_mod_inplace(T& a, uint64_t b, uint64_t t) {
    a = add_mod(a, b, t);
}

} // namespace TROY

template <class T>
void TROY::send(T** ios, const std::stringstream& ss) {
    auto tmp = ss.str();
    size_t n = tmp.size();
    ios[0]->send_data(&n, sizeof(size_t));
    if (n > 0) {
        ios[0]->send_data(tmp.c_str(), n);
    }
    ios[0]->flush();
}

template <class T>
std::stringstream TROY::recv(T** ios) {
    std::stringstream res;
    size_t n{0};
    ios[0]->recv_data(&n, sizeof(size_t));
    if (n > 0) {
        char* s = new char[n];
        ios[0]->recv_data(s, n);
        res = std::stringstream(std::string(s, n));
        delete[] s;
    }
    return res;
}

template <class T>
bool TROY::vector_equal(const vector<T>& a, const vector<T>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

inline size_t dim(size_t dim_i, size_t dim_k, size_t stride, size_t padding) {
    return ((dim_i + 2 * padding - dim_k) / stride) + 1;
}

#endif
