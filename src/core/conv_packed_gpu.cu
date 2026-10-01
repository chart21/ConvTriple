#include "core/conv_packed_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace Iface::packed_gpu {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("PackedConv2D GPU: ") + what + ": " + cudaGetErrorString(e));
}

struct DevPrime {
    uint64_t q, ratio0, ratio1, inv_n, inv_n_quot, last, last_quot, increment;
    const uint64_t *root, *root_quot, *inv_root, *inv_root_quot;
};

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b;
    return s >= q ? s - q : s;
}
__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
// x * w mod q with w's Shoup quotient (x < q)
__device__ __forceinline__ uint64_t mul_shoup(uint64_t x, uint64_t w, uint64_t wq, uint64_t q) {
    uint64_t r = x * w - __umul64hi(x, wq) * q;
    return r >= q ? r - q : r;
}
// SEAL's barrett_reduce_128 of hi * 2^64 + lo
__device__ __forceinline__ uint64_t barrett128(uint64_t lo, uint64_t hi, const DevPrime& p) {
    uint64_t carry = __umul64hi(lo, p.ratio0);
    uint64_t t_lo = lo * p.ratio1, t_hi = __umul64hi(lo, p.ratio1);
    uint64_t tmp1 = t_lo + carry;
    uint64_t tmp3 = t_hi + (tmp1 < t_lo);
    uint64_t u_lo = hi * p.ratio0, u_hi = __umul64hi(hi, p.ratio0);
    uint64_t s = tmp1 + u_lo;
    carry = u_hi + (s < tmp1);
    tmp1 = hi * p.ratio1 + tmp3 + carry;
    uint64_t r = lo - tmp1 * p.q;
    return r >= p.q ? r - p.q : r;
}

constexpr int kNttThreads = 512;

// SEAL's negacyclic NTT (DWTHandler::transform_to_rev, same root order) of polynomial k of `count`: polynomials come in
// groups of L (one per prime) and group j starts at polynomial j * stride; fully reduced in and out
__global__ void ntt_forward(uint64_t* data, size_t stride, int L, const DevPrime* primes, int N) {
    extern __shared__ uint64_t s[];
    const size_t k = blockIdx.x;
    const int l    = int(k % L);
    uint64_t* poly = data + ((k / L) * stride + l) * N;
    const DevPrime p = primes[l];
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = poly[i];
    __syncthreads();
    for (int m = 1, logg = __ffs(N) - 2; m < N; m <<= 1, logg--) {
        for (int b = threadIdx.x; b < N / 2; b += blockDim.x) {
            const int i = b >> logg, j = b & ((1 << logg) - 1);
            const int x = (i << (logg + 1)) + j, y = x + (1 << logg);
            const uint64_t u = s[x], v = mul_shoup(s[y], __ldg(p.root + m + i), __ldg(p.root_quot + m + i), p.q);
            s[x] = add_mod(u, v, p.q);
            s[y] = sub_mod(u, v, p.q);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x) poly[i] = s[i];
}

// SEAL's inverse (DWTHandler::transform_from_rev with the scalar 1/N)
__global__ void ntt_inverse(uint64_t* data, size_t stride, int L, const DevPrime* primes, int N) {
    extern __shared__ uint64_t s[];
    const size_t k = blockIdx.x;
    const int l    = int(k % L);
    uint64_t* poly = data + ((k / L) * stride + l) * N;
    const DevPrime p = primes[l];
    for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] = poly[i];
    __syncthreads();
    int logg = 0;
    for (int m = N >> 1; m > 1; m >>= 1, logg++) {
        const int base = N - 2 * m + 1; // the roots this stage takes: base + i
        for (int b = threadIdx.x; b < N / 2; b += blockDim.x) {
            const int i = b >> logg, j = b & ((1 << logg) - 1);
            const int x = (i << (logg + 1)) + j, y = x + (1 << logg);
            const uint64_t u = s[x], v = s[y];
            s[x] = add_mod(u, v, p.q);
            s[y] = mul_shoup(sub_mod(u, v, p.q), __ldg(p.inv_root + base + i), __ldg(p.inv_root_quot + base + i), p.q);
        }
        __syncthreads();
    }
    for (int j = threadIdx.x; j < N / 2; j += blockDim.x) { // m = 1: scaled by 1/N
        const uint64_t u = s[j], v = s[j + N / 2];
        s[j]         = mul_shoup(add_mod(u, v, p.q), p.inv_n, p.inv_n_quot, p.q);
        s[j + N / 2] = mul_shoup(sub_mod(u, v, p.q), p.last, p.last_quot, p.q);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) poly[i] = s[i];
}

// The NTT of the weight polynomials of output groups [og0, og0 + nog) (see Tiling::weight_terms, non-repacked): block
// (og * G + g) * L + l computes every coefficient of its polynomial from the weights (coefficient (oo * ci + ci - 1 - cc)
// * h * w + a * w + d holds filter oo, channel cc at (kh - 1 - a, kw - 1 - d), lifted to prime l; the rest is zero)
// straight into shared memory and transforms it there
template <class Word>
__global__ void weights_ntt(const Word* wt, uint64_t* W, size_t og0, int G, int L, int N, int oc, int ic, int co,
                            int ci, int h, int w, int kh, int kw, uint64_t threshold, const DevPrime* primes) {
    extern __shared__ uint64_t s[];
    const int l = int(blockIdx.x % L), g = int(blockIdx.x / L % G), og = int(blockIdx.x / L / G);
    const DevPrime p = primes[l];
    const int lo = int(og0 + og) * co, lc = g * ci, hw = h * w;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        uint64_t v = 0;
        const int q = i / hw, rem = i - q * hw, a = rem / w, d = rem - a * w;
        if (q < co * ci && a < kh && d < kw) {
            const int oo = q / ci, cc = ci - 1 - (q - oo * ci), o = lo + oo, c = lc + cc;
            if (o < oc && c < ic) {
                v = uint64_t(wt[((size_t(o) * ic + c) * kh + kh - 1 - a) * kw + kw - 1 - d]);
                v = v >= threshold ? v + p.increment : v;
            }
        }
        s[i] = v;
    }
    __syncthreads();
    for (int m = 1, logg = __ffs(N) - 2; m < N; m <<= 1, logg--) {
        for (int b = threadIdx.x; b < N / 2; b += blockDim.x) {
            const int i = b >> logg, j = b & ((1 << logg) - 1);
            const int x = (i << (logg + 1)) + j, y = x + (1 << logg);
            const uint64_t u = s[x], v = mul_shoup(s[y], __ldg(p.root + m + i), __ldg(p.root_quot + m + i), p.q);
            s[x] = add_mod(u, v, p.q);
            s[y] = sub_mod(u, v, p.q);
        }
        __syncthreads();
    }
    uint64_t* poly = W + ((size_t(og) * G + g) * L + l) * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) poly[i] = s[i];
}

// y[tile][og0 + og][p][l][i] = sum_g x[tile][g][p][l][i] * W[og][g][l][i] (NTT values) for p = 0, 1 (one load of the
// weight for both), 128-bit sums reduced every `fold` products
__global__ void mac(const uint64_t* X, const uint64_t* W, uint64_t* Y, size_t og0, int nog, int OG, int G, int L,
                    int N, const DevPrime* primes, int fold) {
    const int i    = blockIdx.y * blockDim.x + threadIdx.x;
    const int tile = int(blockIdx.x) / nog, og = int(blockIdx.x) % nog;
    const int l    = int(blockIdx.z);
    if (i >= N)
        return;
    const DevPrime& pr = primes[l];
    uint64_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
    int pending = 0;
    for (int g = 0; g < G; g++) {
        const size_t xo  = ((size_t(tile) * G + g) * 2 * L + l) * N + i;
        const uint64_t b = W[((size_t(og) * G + g) * L + l) * N + i];
        const uint64_t a0 = X[xo], a1 = X[xo + size_t(L) * N];
        uint64_t p = a0 * b;
        lo0 += p;
        hi0 += __umul64hi(a0, b) + (lo0 < p);
        p = a1 * b;
        lo1 += p;
        hi1 += __umul64hi(a1, b) + (lo1 < p);
        if (++pending == fold) {
            lo0 = barrett128(lo0, hi0, pr), hi0 = 0;
            lo1 = barrett128(lo1, hi1, pr), hi1 = 0;
            pending = 0;
        }
    }
    const size_t yo = ((size_t(tile) * OG + og0 + og) * 2 * L + l) * N + i;
    Y[yo]                 = barrett128(lo0, hi0, pr);
    Y[yo + size_t(L) * N] = barrett128(lo1, hi1, pr);
}

} // namespace

struct Engine::Impl {
    size_t N;
    int L;
    uint64_t threshold;
    int fold;
    DevPrime* primes = nullptr; // device
    std::vector<uint64_t*> tables;
    size_t weight_budget;        // bytes of weight polynomials per pass
};

bool available() {
    if (getenv("CONV_GPU") && atoi(getenv("CONV_GPU")) == 0)
        return false;
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

Engine::Engine(size_t N, uint64_t threshold, const std::vector<Prime>& primes, size_t fold) : impl_(new Impl) {
    if (N != 4096)
        throw std::runtime_error("PackedConv2D GPU: N = 4096 only");
    impl_->N = N, impl_->L = int(primes.size()), impl_->threshold = threshold, impl_->fold = int(std::min<size_t>(fold, 1 << 20));
    impl_->weight_budget = getenv("CONV_GPU_WEIGHT_MB") ? size_t(atol(getenv("CONV_GPU_WEIGHT_MB"))) << 20 : size_t(256) << 20;
    std::vector<DevPrime> host(primes.size());
    for (size_t l = 0; l < primes.size(); l++) {
        const Prime& p = primes[l];
        DevPrime& d    = host[l];
        d.q = p.q, d.ratio0 = p.ratio0, d.ratio1 = p.ratio1, d.inv_n = p.inv_n, d.inv_n_quot = p.inv_n_quot;
        d.last = p.last, d.last_quot = p.last_quot, d.increment = p.increment;
        const std::vector<uint64_t>* src[4] = {&p.root, &p.root_quot, &p.inv_root, &p.inv_root_quot};
        const uint64_t** dst[4]             = {&d.root, &d.root_quot, &d.inv_root, &d.inv_root_quot};
        for (int t = 0; t < 4; t++) {
            uint64_t* buf;
            check(cudaMalloc(&buf, N * sizeof(uint64_t)), "tables");
            check(cudaMemcpy(buf, src[t]->data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice), "tables");
            impl_->tables.push_back(buf);
            *dst[t] = buf;
        }
    }
    check(cudaMalloc(&impl_->primes, host.size() * sizeof(DevPrime)), "primes");
    check(cudaMemcpy(impl_->primes, host.data(), host.size() * sizeof(DevPrime), cudaMemcpyHostToDevice), "primes");
}

Engine::~Engine() {
    if (!impl_)
        return;
    for (auto* b : impl_->tables) cudaFree(b);
    cudaFree(impl_->primes);
}

namespace {
template <class Word>
void run(const Engine::Impl& e, const Geometry& g, const uint64_t* x, const Word* w, uint64_t* y) {
    thread_local cudaStream_t stream = [] {
        cudaStream_t s;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        return s;
    }();
    const size_t N = e.N, L = size_t(e.L), G = g.in_groups, OG = g.out_groups;
    const size_t ct_words = 2 * L * N;
    const size_t nx = g.tiles * G * ct_words, ny = g.tiles * OG * ct_words, nwt = g.oc * g.ic * g.kh * g.kw;
    // output groups per pass, so that the weight polynomials stay within the budget
    const size_t per_og = G * L * N * sizeof(uint64_t);
    const size_t chunk  = std::max<size_t>(1, std::min(OG, e.weight_budget / per_og));
    uint64_t *dx, *dy, *dw;
    Word* dwt;
    check(cudaMallocAsync(&dx, nx * sizeof(uint64_t), stream), "alloc x");
    check(cudaMallocAsync(&dy, ny * sizeof(uint64_t), stream), "alloc y");
    check(cudaMallocAsync(&dw, chunk * per_og, stream), "alloc w");
    check(cudaMallocAsync(&dwt, nwt * sizeof(Word), stream), "alloc weights");
    check(cudaMemcpyAsync(dx, x, nx * sizeof(uint64_t), cudaMemcpyHostToDevice, stream), "copy x");
    check(cudaMemcpyAsync(dwt, w, nwt * sizeof(Word), cudaMemcpyHostToDevice, stream), "copy weights");
    const size_t shmem = N * sizeof(uint64_t);
    // c0 of every input ciphertext to NTT form (c1 arrives in it)
    if (g.tiles * G)
        ntt_forward<<<unsigned(g.tiles * G * L), kNttThreads, shmem, stream>>>(dx, 2 * L, int(L), e.primes, int(N));
    for (size_t og0 = 0; og0 < OG; og0 += chunk) {
        const size_t nog = std::min(chunk, OG - og0);
        weights_ntt<Word><<<unsigned(nog * G * L), kNttThreads, shmem, stream>>>(
            dwt, dw, og0, int(G), int(L), int(N), int(g.oc), int(g.ic), int(g.co), int(g.ci), int(g.h), int(g.w),
            int(g.kh), int(g.kw), e.threshold, e.primes);
        dim3 grid(unsigned(g.tiles * nog), unsigned((N + 255) / 256), unsigned(L));
        mac<<<grid, 256, 0, stream>>>(dx, dw, dy, og0, int(nog), int(OG), int(G), int(L), int(N), e.primes, e.fold);
    }
    if (g.tiles * OG)
        ntt_inverse<<<unsigned(g.tiles * OG * 2 * L), kNttThreads, shmem, stream>>>(dy, L, int(L), e.primes, int(N));
    check(cudaGetLastError(), "launch");
    check(cudaMemcpyAsync(y, dy, ny * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream), "copy y");
    cudaFreeAsync(dx, stream);
    cudaFreeAsync(dy, stream);
    cudaFreeAsync(dw, stream);
    cudaFreeAsync(dwt, stream);
    check(cudaStreamSynchronize(stream), "evaluate");
}
} // namespace

uint64_t* Engine::staging(size_t words, int which) const {
    struct Buf {
        uint64_t* p = nullptr;
        size_t n    = 0;
        ~Buf() {
            if (p)
                cudaFreeHost(p);
        }
    };
    thread_local Buf bufs[2];
    Buf& b = bufs[which & 1];
    if (b.n < words) {
        if (b.p)
            cudaFreeHost(b.p);
        b.p = nullptr;
        check(cudaMallocHost(&b.p, words * sizeof(uint64_t)), "pinned staging");
        b.n = words;
    }
    return b.p;
}

void Engine::evaluate(const Geometry& g, const uint64_t* x, const uint32_t* w, uint64_t* y) const { run(*impl_, g, x, w, y); }
void Engine::evaluate(const Geometry& g, const uint64_t* x, const uint64_t* w, uint64_t* y) const { run(*impl_, g, x, w, y); }

} // namespace Iface::packed_gpu
