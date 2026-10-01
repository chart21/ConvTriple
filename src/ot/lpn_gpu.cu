// Ferret's LPN step (emp-ot LpnF2<IO, 10>::compute with one task range, as ConvTriple's packs run it) on the GPU:
// nn[i] ^= XOR_j kk[index(i, j)] with the indices from AES-128 under the PRP key (emp's AES_ecb_encrypt_blks of
// makeBlock(i, m)): groups of 4 outputs take 10 AES blocks (40 indices), the at most 7 trailing outputs 3 blocks each
// (__compute1). Same outputs as the CPU's, bit for bit.
#include "ot/lpn_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cheetah::lpn_gpu {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("LPN GPU: ") + what + ": " + cudaGetErrorString(e));
}

// AES-128 encryption tables (T-tables of the forward round, the S-box for the last round)
__constant__ uint32_t c_te0[256];

uint8_t xtime(uint8_t x) { return uint8_t((x << 1) ^ ((x & 0x80) ? 0x1b : 0)); }

void make_tables(uint32_t* te0, uint8_t* sbox) {
    // S-box from the multiplicative inverse in GF(2^8) and the affine map
    uint8_t p = 1, q = 1;
    do {
        p = uint8_t(p ^ (p << 1) ^ ((p & 0x80) ? 0x1b : 0));
        q ^= uint8_t(q << 1);
        q ^= uint8_t(q << 2);
        q ^= uint8_t(q << 4);
        if (q & 0x80)
            q ^= 0x09;
        uint8_t x = uint8_t(q ^ (q << 1 | q >> 7) ^ (q << 2 | q >> 6) ^ (q << 3 | q >> 5) ^ (q << 4 | q >> 4));
        sbox[p] = uint8_t(x ^ 0x63);
    } while (p != 1);
    sbox[0] = 0x63;
    for (int i = 0; i < 256; i++) {
        uint8_t s = sbox[i], s2 = xtime(s), s3 = uint8_t(s2 ^ s);
        // little-endian word: bytes (2s, s, s, 3s) of column 0
        te0[i] = uint32_t(s2) | uint32_t(s) << 8 | uint32_t(s) << 16 | uint32_t(s3) << 24;
    }
}

__device__ __forceinline__ uint32_t rotl8(uint32_t x) { return (x << 8) | (x >> 24); }

// Te0 replicated once per lane (entry x of lane l at x * 32 + l): every lane reads its own bank, so the random table
// lookups of a warp never conflict. The S-box is byte 1 of Te0.
#define TE(x) te[((x) << 5) | lane]
#define SB(x) ((TE(x) >> 8) & 0xff)

// one AES-128 block, state as four little-endian column words; rk: 44 round-key words in the same layout
__device__ __forceinline__ void aes128(uint32_t s[4], const uint32_t* rk, const uint32_t* te, int lane) {
    uint32_t a0 = s[0] ^ rk[0], a1 = s[1] ^ rk[1], a2 = s[2] ^ rk[2], a3 = s[3] ^ rk[3];
#pragma unroll
    for (int r = 1; r < 10; r++) {
        // column c of the result takes row 0 from column c, row 1 from c + 1, row 2 from c + 2, row 3 from c + 3
        uint32_t b0 = TE(a0 & 0xff) ^ rotl8(TE((a1 >> 8) & 0xff)) ^ rotl8(rotl8(TE((a2 >> 16) & 0xff))) ^ rotl8(rotl8(rotl8(TE(a3 >> 24))));
        uint32_t b1 = TE(a1 & 0xff) ^ rotl8(TE((a2 >> 8) & 0xff)) ^ rotl8(rotl8(TE((a3 >> 16) & 0xff))) ^ rotl8(rotl8(rotl8(TE(a0 >> 24))));
        uint32_t b2 = TE(a2 & 0xff) ^ rotl8(TE((a3 >> 8) & 0xff)) ^ rotl8(rotl8(TE((a0 >> 16) & 0xff))) ^ rotl8(rotl8(rotl8(TE(a1 >> 24))));
        uint32_t b3 = TE(a3 & 0xff) ^ rotl8(TE((a0 >> 8) & 0xff)) ^ rotl8(rotl8(TE((a1 >> 16) & 0xff))) ^ rotl8(rotl8(rotl8(TE(a2 >> 24))));
        a0 = b0 ^ rk[4 * r], a1 = b1 ^ rk[4 * r + 1], a2 = b2 ^ rk[4 * r + 2], a3 = b3 ^ rk[4 * r + 3];
    }
    s[0] = (SB(a0 & 0xff) | SB((a1 >> 8) & 0xff) << 8 | SB((a2 >> 16) & 0xff) << 16 | SB(a3 >> 24) << 24) ^ rk[40];
    s[1] = (SB(a1 & 0xff) | SB((a2 >> 8) & 0xff) << 8 | SB((a3 >> 16) & 0xff) << 16 | SB(a0 >> 24) << 24) ^ rk[41];
    s[2] = (SB(a2 & 0xff) | SB((a3 >> 8) & 0xff) << 8 | SB((a0 >> 16) & 0xff) << 16 | SB(a1 >> 24) << 24) ^ rk[42];
    s[3] = (SB(a3 & 0xff) | SB((a0 >> 8) & 0xff) << 8 | SB((a1 >> 16) & 0xff) << 16 | SB(a2 >> 24) << 24) ^ rk[43];
}
#undef TE
#undef SB

// the replicated table into shared memory (32 KiB, dynamic)
__device__ __forceinline__ void load_table(uint32_t* te) {
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) te[i] = c_te0[i >> 5];
}
constexpr size_t kTableBytes = 256 * 32 * sizeof(uint32_t);

struct RoundKeys {
    uint32_t w[44];
};

// groups g0 .. g0 + count - 1 of 4 outputs (group g: outputs start + 4g .. + 3, LpnF2::__compute4): G[4 (g - g0) + m]
// = the XOR of output start + 4g + m's ten gathers (the host XORs it into nn)
__global__ void lpn_gather(uint4* G, const uint4* __restrict__ kk, int64_t start, int64_t g0, int64_t count, uint32_t mask,
                           uint32_t k, RoundKeys rkeys) {
    extern __shared__ uint32_t te[];
    __shared__ uint32_t rk[44];
    load_table(te);
    for (int i = threadIdx.x; i < 44; i += blockDim.x) rk[i] = rkeys.w[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    const int64_t i = start + 4 * (g0 + t);
    uint4 acc[4] = {};
    // block m = makeBlock(i, m): low 64 bits m, high 64 bits i; r = its 40 32-bit words in order
#pragma unroll
    for (int m = 0; m < 10; m++) {
        uint32_t s[4] = {uint32_t(m), 0u, uint32_t(uint64_t(i)), uint32_t(uint64_t(i) >> 32)};
        aes128(s, rk, te, lane);
#pragma unroll
        for (int w = 0; w < 4; w++) {
            const int idx_pos = m * 4 + w;  // r[idx_pos]: output idx_pos / 10, gather idx_pos % 10
            uint32_t index = s[w] & mask;
            index -= index >= k ? k : 0;
            const uint4 v = __ldg(kk + index);
            uint4& a = acc[idx_pos / 10];
            a.x ^= v.x, a.y ^= v.y, a.z ^= v.z, a.w ^= v.w;
        }
    }
#pragma unroll
    for (int m = 0; m < 4; m++) G[4 * t + m] = acc[m];
}

__global__ void aes_test(uint32_t* out, RoundKeys rkeys, int n) {
    extern __shared__ uint32_t te[];
    load_table(te);
    __syncthreads();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n)
        return;
    uint32_t s[4] = {uint32_t(t), 0u, 7u, 0u};
    aes128(s, rkeys.w, te, threadIdx.x & 31);
    for (int w = 0; w < 4; w++) out[4 * t + w] = s[w];
}

struct Device {
    bool ready = false;
    Device() {
        uint32_t te[256];
        uint8_t sb[256];
        make_tables(te, sb);
        check(cudaMemcpyToSymbol(c_te0, te, sizeof(te)), "tables");
        ready = true;
    }
};
Device& device() {
    static Device d;
    return d;
}

// per calling thread: a stream, the table on the device, and a ring of chunk buffers (device and pinned host)
constexpr int kRing = 3;
constexpr int64_t kChunkGroups = int64_t(1) << 18;  // 1M outputs, 16 MiB per chunk
struct Ctx {
    cudaStream_t stream = nullptr;
    uint4* dkk = nullptr;
    size_t cap_kk = 0;
    uint4* dG[kRing] = {};
    uint4* hG[kRing] = {};
    cudaEvent_t done[kRing] = {};
    ~Ctx() {
        if (dkk)
            cudaFree(dkk);
        for (int b = 0; b < kRing; b++) {
            if (dG[b])
                cudaFree(dG[b]);
            if (hG[b])
                cudaFreeHost(hG[b]);
            if (done[b])
                cudaEventDestroy(done[b]);
        }
    }
};
// Contexts are kept in a pool, not per thread: ConvTriple's OT workers are new threads at every call, and pinning the
// buffers again each time cost more than the LPN step
std::mutex g_pool_mutex;
std::vector<Ctx*> g_pool;
Ctx* acquire() {
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (!g_pool.empty()) {
            Ctx* c = g_pool.back();
            g_pool.pop_back();
            return c;
        }
    }
    Ctx* c = new Ctx;
    check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream");
    for (int b = 0; b < kRing; b++) {
        check(cudaMalloc(&c->dG[b], size_t(4 * kChunkGroups) * sizeof(uint4)), "alloc chunk");
        check(cudaMallocHost(&c->hG[b], size_t(4 * kChunkGroups) * sizeof(uint4)), "pinned chunk");
        check(cudaEventCreateWithFlags(&c->done[b], cudaEventDisableTiming), "event");
    }
    return c;
}
void release(Ctx* c) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.push_back(c);
}

} // namespace

bool available() {
    static const bool on = [] {
        if (getenv("LPN_GPU") && atoi(getenv("LPN_GPU")) == 0)
            return false;
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }();
    return on;
}

void compute(void* nn_, const void* kk_, int64_t n, int64_t k, uint32_t mask, const uint32_t round_keys[44],
             int64_t start, int64_t groups) {
    (void) n;
    device();
    struct Lease {
        Ctx* c = acquire();
        ~Lease() { release(c); }
    } lease;
    Ctx& c = *lease.c;
    uint4* nn       = static_cast<uint4*>(nn_);
    const uint4* kk = static_cast<const uint4*>(kk_);
    if (c.cap_kk < size_t(k)) {
        if (c.dkk)
            cudaFree(c.dkk);
        check(cudaMalloc(&c.dkk, size_t(k) * sizeof(uint4)), "alloc kk");
        c.cap_kk = size_t(k);
    }
    RoundKeys rk;
    std::memcpy(rk.w, round_keys, sizeof(rk.w));
    check(cudaMemcpyAsync(c.dkk, kk, size_t(k) * sizeof(uint4), cudaMemcpyHostToDevice, c.stream), "copy kk");
    // chunk ch: gathers on the device, back into its pinned buffer; the host XORs chunk ch - kRing + 1 meanwhile
    const int64_t chunks = (groups + kChunkGroups - 1) / kChunkGroups;
    auto xor_chunk = [&](int64_t ch) {
        const int b = int(ch % kRing);
        check(cudaEventSynchronize(c.done[b]), "lpn chunk");
        const int64_t g0 = ch * kChunkGroups, cnt = std::min(kChunkGroups, groups - g0);
        uint64_t* dst       = reinterpret_cast<uint64_t*>(nn + start + 4 * g0);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(c.hG[b]);
        for (int64_t w = 0; w < 8 * cnt; w++) dst[w] ^= src[w];
    };
    for (int64_t ch = 0; ch < chunks; ch++) {
        if (ch >= kRing)
            xor_chunk(ch - kRing);
        const int b = int(ch % kRing);
        const int64_t g0 = ch * kChunkGroups, cnt = std::min(kChunkGroups, groups - g0);
        lpn_gather<<<unsigned((cnt + 255) / 256), 256, kTableBytes, c.stream>>>(c.dG[b], c.dkk, start, g0, cnt, mask, uint32_t(k), rk);
        check(cudaGetLastError(), "launch");
        check(cudaMemcpyAsync(c.hG[b], c.dG[b], size_t(4 * cnt) * sizeof(uint4), cudaMemcpyDeviceToHost, c.stream), "copy back");
        check(cudaEventRecord(c.done[b], c.stream), "event");
    }
    for (int64_t ch = std::max<int64_t>(0, chunks - kRing); ch < chunks; ch++) xor_chunk(ch);
}

void aes_reference_check(const uint32_t round_keys[44], const uint32_t* expected, int n) {
    device();
    RoundKeys rk;
    std::memcpy(rk.w, round_keys, sizeof(rk.w));
    uint32_t* d;
    check(cudaMalloc(&d, size_t(n) * 16), "alloc");
    aes_test<<<(n + 127) / 128, 128, kTableBytes>>>(d, rk, n);
    std::vector<uint32_t> h(size_t(n) * 4);
    check(cudaMemcpy(h.data(), d, size_t(n) * 16, cudaMemcpyDeviceToHost), "copy");
    cudaFree(d);
    for (int i = 0; i < 4 * n; i++)
        if (h[i] != expected[i])
            throw std::runtime_error("LPN GPU: AES differs from AES-NI at word " + std::to_string(i));
}

} // namespace cheetah::lpn_gpu
