// Ferret's LPN step (emp-ot LpnF2<IO, 10>::compute with one task range, as ConvTriple's packs run it) on the GPU:
// nn[i] ^= XOR_j kk[index(i, j)] with the indices from AES-128 under the PRP key (emp's AES_ecb_encrypt_blks of
// makeBlock(i, m)): groups of 4 outputs take 10 AES blocks (40 indices), the at most 7 trailing outputs 3 blocks each
// (__compute1). Same outputs as the CPU's, bit for bit.
#include "ot/lpn_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

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

// the gathers of the 4 outputs i .. i + 3 (LpnF2::__compute4): block m = makeBlock(i, m) (low 64 bits m, high 64 bits
// i); r = their 40 32-bit words in order, r[10 o + j] the index of output o's gather j
__device__ __forceinline__ void lpn_group(uint4 acc[4], const uint4* __restrict__ kk, int64_t i, uint32_t mask, uint32_t k,
                                          const uint32_t* rk, const uint32_t* te, int lane) {
#pragma unroll
    for (int m = 0; m < 10; m++) {
        uint32_t s[4] = {uint32_t(m), 0u, uint32_t(uint64_t(i)), uint32_t(uint64_t(i) >> 32)};
        aes128(s, rk, te, lane);
#pragma unroll
        for (int w = 0; w < 4; w++) {
            const int idx_pos = m * 4 + w;
            uint32_t index = s[w] & mask;
            index -= index >= k ? k : 0;
            const uint4 v = __ldg(kk + index);
            uint4& a = acc[idx_pos / 10];
            a.x ^= v.x, a.y ^= v.y, a.z ^= v.z, a.w ^= v.w;
        }
    }
}

// groups g0 .. g0 + count - 1 of 4 outputs (group g: outputs start + 4g .. + 3): G[4 (g - g0) + m] = the XOR of output
// start + 4g + m's ten gathers (the host XORs it into nn)
__global__ void lpn_gather(uint4* G, const uint4* __restrict__ kk, int64_t start, int64_t g0, int64_t count, uint32_t mask,
                           uint32_t k, RoundKeys rkeys) {
    extern __shared__ uint32_t te[];
    __shared__ uint32_t rk[44];
    load_table(te);
    for (int i = threadIdx.x; i < 44; i += blockDim.x) rk[i] = rkeys.w[i];
    __syncthreads();
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    uint4 acc[4] = {};
    lpn_group(acc, kk, start + 4 * (g0 + t), mask, k, rk, te, threadIdx.x & 31);
#pragma unroll
    for (int m = 0; m < 4; m++) G[4 * t + m] = acc[m];
}

// the same on the sparse vector on the device: nn[start + 4g + m] ^= the gathers, g < groups
__global__ void lpn_inplace(uint4* nn, const uint4* __restrict__ kk, int64_t start, int64_t groups, uint32_t mask,
                            uint32_t k, RoundKeys rkeys) {
    extern __shared__ uint32_t te[];
    __shared__ uint32_t rk[44];
    load_table(te);
    for (int i = threadIdx.x; i < 44; i += blockDim.x) rk[i] = rkeys.w[i];
    __syncthreads();
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= groups)
        return;
    const int64_t i = start + 4 * t;
    uint4 acc[4] = {};
    lpn_group(acc, kk, i, mask, k, rk, te, threadIdx.x & 31);
#pragma unroll
    for (int m = 0; m < 4; m++) {
        uint4 v = nn[i + m];
        v.x ^= acc[m].x, v.y ^= acc[m].y, v.z ^= acc[m].z, v.w ^= acc[m].w;
        nn[i + m] = v;
    }
}

// MPCOT's GGM trees (emp's SPCOT_Sender::ggm_tree_gen, SPCOT_Recver::ggm_tree_reconstruction), all trees at once, level
// by level; trees of the same level are contiguous (tree i's level-L nodes at i 2^L .. + 2^L - 1)
struct TwoKeys {
    uint32_t w[88];  // TwoKeyPRP's round keys: key 0 (zero_block), key 1 (makeBlock(0, 1))
};

__device__ __forceinline__ uint4 xor4(uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
// & makeBlock(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE): the leaves' bit 0 cleared
__device__ __forceinline__ uint4 clear_lsb(uint4 a) { return make_uint4(a.x & ~1u, a.y, a.z, a.w); }
__device__ __forceinline__ uint4 shfl_xor4(uint4 a, int off) {
    return make_uint4(__shfl_xor_sync(0xffffffffu, a.x, off), __shfl_xor_sync(0xffffffffu, a.y, off),
                      __shfl_xor_sync(0xffffffffu, a.z, off), __shfl_xor_sync(0xffffffffu, a.w, off));
}
__device__ __forceinline__ void atomic_xor4(uint4* dst, uint4 v) {
    unsigned* d = reinterpret_cast<unsigned*>(dst);
    atomicXor(d, v.x), atomicXor(d + 1, v.y), atomicXor(d + 2, v.z), atomicXor(d + 3, v.w);
}

// level L = log_p + 1 from level L - 1 (2^log_p nodes per tree in `in`, the seeds for L = 1): children 2j = p ^
// AES_k0(p), 2j + 1 = p ^ AES_k1(p) (TwoKeyPRP::node_expand); sums[tree][L - 1] ^= {the XOR of the even children, of
// the odd children}, before the last level's bit 0 is cleared. The receiver's punctured node (punct[tree]; the root at
// level 0) has zero children: its sums leave out the punctured pair.
__global__ void ggm_level(uint4* out, const uint4* __restrict__ in, int64_t trees, int log_p, int levels, uint4* sums,
                          const int32_t* __restrict__ punct, bool last, TwoKeys keys) {
    extern __shared__ uint32_t te[];
    __shared__ uint32_t rk[88];
    load_table(te);
    for (int i = threadIdx.x; i < 88; i += blockDim.x) rk[i] = keys.w[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t tree = t >> log_p, j = t & ((int64_t(1) << log_p) - 1);
    uint4 c0 = {}, c1 = {};
    if (tree < trees) {
        if (!(punct && (log_p == 0 || punct[tree] == j))) {
            const uint4 p = in[t];
            uint32_t s0[4] = {p.x, p.y, p.z, p.w}, s1[4] = {p.x, p.y, p.z, p.w};
            aes128(s0, rk, te, lane);
            aes128(s1, rk + 44, te, lane);
            c0 = xor4(p, make_uint4(s0[0], s0[1], s0[2], s0[3]));
            c1 = xor4(p, make_uint4(s1[0], s1[1], s1[2], s1[3]));
        }
        out[2 * t]     = last ? clear_lsb(c0) : c0;
        out[2 * t + 1] = last ? clear_lsb(c1) : c1;
    }
    // the sums over a tree's lanes (2^log_p consecutive threads, aligned: whole warps, or segments of a warp), then one
    // atomic per segment
    const int width = log_p >= 5 ? 32 : 1 << log_p;
    for (int off = width >> 1; off > 0; off >>= 1) {
        c0 = xor4(c0, shfl_xor4(c0, off));
        c1 = xor4(c1, shfl_xor4(c1, off));
    }
    if (tree < trees && (lane & (width - 1)) == 0) {
        uint4* s = sums + 2 * (tree * levels + log_p);
        atomic_xor4(s, c0);
        atomic_xor4(s + 1, c1);
    }
}

// the receiver after level L (one thread per tree): with p the punctured node of level L - 1 and b its choice bit, the
// known node of the pair 2p, 2p + 1 is 2p + b = the XOR of its parity's other nodes ^ the message; the other one is the
// new punctured node. At the leaves: the known leaf with bit 0 cleared, the punctured leaf = the XOR of all leaves (bit 0
// cleared) ^ the secret sum.
__global__ void ggm_fix(uint4* out, int64_t trees, int L, int levels, const uint4* __restrict__ sums,
                        const uint4* __restrict__ msg, const uint8_t* __restrict__ bits, const uint4* __restrict__ secret,
                        int32_t* punct, bool last) {
    const int64_t tree = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tree >= trees)
        return;
    const int64_t p = L == 1 ? 0 : punct[tree];
    const int b = bits[tree * levels + L - 1] ? 1 : 0;
    const uint4* s = sums + 2 * (tree * levels + L - 1);
    uint4* lvl = out + (tree << L);
    uint4 known = xor4(s[b], msg[tree * levels + L - 1]);
    if (!last) {
        lvl[2 * p + b] = known;
        punct[tree] = int32_t(2 * p + 1 - b);
    } else {
        known = clear_lsb(known);
        lvl[2 * p + b] = known;
        lvl[2 * p + 1 - b] = xor4(xor4(clear_lsb(xor4(s[0], s[1])), known), secret[tree]);
    }
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

// a context: a stream, the table on the device, a ring of chunk buffers (device and pinned host), and the trees' buffers
constexpr int kRing = 3;
constexpr int64_t kChunkGroups = int64_t(1) << 18;  // 1M outputs, 16 MiB per chunk
template <typename T>
void grow(T*& p, size_t& cap, size_t need, const char* what) {
    if (cap >= need)
        return;
    if (p)
        cudaFree(p);
    p   = nullptr;
    cap = 0;
    check(cudaMalloc(&p, need * sizeof(T)), what);
    cap = need;
}
struct Ctx {
    cudaStream_t stream = nullptr;
    uint4* dkk = nullptr;
    size_t cap_kk = 0;
    uint4* dG[kRing] = {};
    uint4* hG[kRing] = {};
    cudaEvent_t done[kRing] = {};
    // the trees: the leaves (later the LPN's outputs) in D, the level above in T, both alternately; per tree: the seed,
    // the level sums, the receiver's messages, choice bits, secret sum and punctured node
    uint4 *dD = nullptr, *dT = nullptr, *dSeed = nullptr, *dSums = nullptr, *dMsg = nullptr, *dSecret = nullptr;
    uint8_t* dBits = nullptr;
    int32_t* dPunct = nullptr;
    size_t cap_D = 0, cap_T = 0, cap_seed = 0, cap_sums = 0, cap_msg = 0, cap_secret = 0, cap_bits = 0, cap_punct = 0;
    ~Ctx() {
        for (void* p : {(void*) dkk, (void*) dD, (void*) dT, (void*) dSeed, (void*) dSums, (void*) dMsg, (void*) dSecret,
                        (void*) dBits, (void*) dPunct})
            if (p)
                cudaFree(p);
        for (int b = 0; b < kRing; b++) {
            if (dG[b])
                cudaFree(dG[b]);
            if (hG[b])
                cudaFreeHost(hG[b]);
            if (done[b])
                cudaEventDestroy(done[b]);
        }
    }
    void ring(bool device) {
        for (int b = 0; b < kRing; b++) {
            if (device && !dG[b])
                check(cudaMalloc(&dG[b], size_t(4 * kChunkGroups) * sizeof(uint4)), "alloc chunk");
            if (!hG[b])
                check(cudaMallocHost(&hG[b], size_t(4 * kChunkGroups) * sizeof(uint4)), "pinned chunk");
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
    for (int b = 0; b < kRing; b++) check(cudaEventCreateWithFlags(&c->done[b], cudaEventDisableTiming), "event");
    return c;
}
void release(Ctx* c) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.push_back(c);
}

TwoKeys two_keys(const uint32_t keys[88]) {
    TwoKeys k;
    std::memcpy(k.w, keys, sizeof(k.w));
    return k;
}
// level L's buffer: the leaves (L = depth - 1) in D, then alternately T, D, ...
uint4* level_buf(Ctx& c, int L, int depth) { return (depth - 1 - L) % 2 == 0 ? c.dD : c.dT; }

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

bool mpcot_available() {
    static const bool on = available() && !(getenv("MPCOT_GPU") && atoi(getenv("MPCOT_GPU")) == 0);
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
    c.ring(true);
    uint4* nn       = static_cast<uint4*>(nn_);
    const uint4* kk = static_cast<const uint4*>(kk_);
    grow(c.dkk, c.cap_kk, size_t(k), "alloc kk");
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

namespace {
std::mutex g_pin_mutex;
std::unordered_map<const void*, size_t> g_pinned;  // pinned host buffers and their sizes
bool pinned(const void* p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pin_mutex);
    auto it = g_pinned.find(p);
    return it != g_pinned.end() && it->second >= bytes;
}
} // namespace

bool pin(void* p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pin_mutex);
    auto it = g_pinned.find(p);
    if (it != g_pinned.end())
        return it->second >= bytes;
    device();
    if (cudaHostRegister(p, bytes, cudaHostRegisterDefault) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    g_pinned[p] = bytes;
    return true;
}

void unpin(void* p) {
    std::lock_guard<std::mutex> lock(g_pin_mutex);
    auto it = g_pinned.find(p);
    if (it == g_pinned.end())
        return;
    cudaHostUnregister(p);
    g_pinned.erase(it);
}

Extension::Extension() : ctx_((device(), acquire())) {}
Extension::~Extension() {
    Ctx* c = static_cast<Ctx*>(ctx_);
    cudaStreamSynchronize(c->stream);
    release(c);
}

void Extension::reserve(int64_t trees, int depth, int64_t k) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    const size_t leaves = size_t(trees) << (depth - 1), levels = size_t(depth - 1);
    grow(c.dD, c.cap_D, leaves, "alloc leaves");
    grow(c.dT, c.cap_T, leaves / 2, "alloc tree level");
    grow(c.dSeed, c.cap_seed, size_t(trees), "alloc seeds");
    grow(c.dSums, c.cap_sums, 2 * levels * size_t(trees), "alloc sums");
    grow(c.dMsg, c.cap_msg, levels * size_t(trees), "alloc messages");
    grow(c.dSecret, c.cap_secret, size_t(trees), "alloc secret sums");
    grow(c.dBits, c.cap_bits, levels * size_t(trees), "alloc choice bits");
    grow(c.dPunct, c.cap_punct, size_t(trees), "alloc punctured nodes");
    grow(c.dkk, c.cap_kk, size_t(k), "alloc kk");
    c.ring(false);
}

void Extension::upload_kk(const void* kk, int64_t k) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    check(cudaMemcpyAsync(c.dkk, kk, size_t(k) * sizeof(uint4), cudaMemcpyHostToDevice, c.stream), "copy kk");
}

void Extension::sender_trees(const void* seeds, int64_t trees, int depth, const uint32_t keys[88], void* msg) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    const int levels = depth - 1;
    const TwoKeys tk = two_keys(keys);
    check(cudaMemcpyAsync(c.dSeed, seeds, size_t(trees) * sizeof(uint4), cudaMemcpyHostToDevice, c.stream), "copy seeds");
    check(cudaMemsetAsync(c.dSums, 0, 2 * size_t(levels) * size_t(trees) * sizeof(uint4), c.stream), "zero sums");
    for (int L = 1; L <= levels; L++) {
        const int64_t threads = trees << (L - 1);
        const uint4* in = L == 1 ? c.dSeed : level_buf(c, L - 1, depth);
        ggm_level<<<unsigned((threads + 255) / 256), 256, kTableBytes, c.stream>>>(level_buf(c, L, depth), in, trees, L - 1, levels,
                                                                                    c.dSums, nullptr, L == levels, tk);
        check(cudaGetLastError(), "launch tree level");
    }
    check(cudaMemcpyAsync(msg, c.dSums, 2 * size_t(levels) * size_t(trees) * sizeof(uint4), cudaMemcpyDeviceToHost, c.stream),
          "copy sums");
    check(cudaStreamSynchronize(c.stream), "sender trees");
}

void Extension::recver_trees(const void* msg, const uint8_t* bits, const void* secret, int64_t trees, int depth,
                             const uint32_t keys[88]) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    const int levels = depth - 1;
    const TwoKeys tk = two_keys(keys);
    check(cudaMemcpyAsync(c.dMsg, msg, size_t(levels) * size_t(trees) * sizeof(uint4), cudaMemcpyHostToDevice, c.stream),
          "copy messages");
    check(cudaMemcpyAsync(c.dBits, bits, size_t(levels) * size_t(trees), cudaMemcpyHostToDevice, c.stream), "copy bits");
    check(cudaMemcpyAsync(c.dSecret, secret, size_t(trees) * sizeof(uint4), cudaMemcpyHostToDevice, c.stream), "copy secret");
    check(cudaMemsetAsync(c.dSums, 0, 2 * size_t(levels) * size_t(trees) * sizeof(uint4), c.stream), "zero sums");
    for (int L = 1; L <= levels; L++) {
        const int64_t threads = trees << (L - 1);
        const uint4* in = L == 1 ? c.dSeed : level_buf(c, L - 1, depth);
        uint4* out = level_buf(c, L, depth);
        ggm_level<<<unsigned((threads + 255) / 256), 256, kTableBytes, c.stream>>>(out, in, trees, L - 1, levels, c.dSums,
                                                                                    c.dPunct, L == levels, tk);
        check(cudaGetLastError(), "launch tree level");
        ggm_fix<<<unsigned((trees + 127) / 128), 128, 0, c.stream>>>(out, trees, L, levels, c.dSums, c.dMsg, c.dBits, c.dSecret,
                                                                      c.dPunct, L == levels);
        check(cudaGetLastError(), "launch tree fix");
    }
}

void Extension::lpn(int64_t k, uint32_t mask, const uint32_t round_keys[44], int64_t start, int64_t groups) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    if (groups <= 0)
        return;
    RoundKeys rk;
    std::memcpy(rk.w, round_keys, sizeof(rk.w));
    lpn_inplace<<<unsigned((groups + 255) / 256), 256, kTableBytes, c.stream>>>(c.dD, c.dkk, start, groups, mask, uint32_t(k), rk);
    check(cudaGetLastError(), "launch lpn");
}

void Extension::download(void* out_, int64_t n) {
    Ctx& c = *static_cast<Ctx*>(ctx_);
    uint4* out = static_cast<uint4*>(out_);
    if (pinned(out_, size_t(n) * sizeof(uint4))) {
        check(cudaMemcpyAsync(out, c.dD, size_t(n) * sizeof(uint4), cudaMemcpyDeviceToHost, c.stream), "copy out");
        check(cudaStreamSynchronize(c.stream), "download");
        return;
    }
    const int64_t chunk = 4 * kChunkGroups, chunks = (n + chunk - 1) / chunk;
    auto copy_chunk = [&](int64_t ch) {
        const int b = int(ch % kRing);
        check(cudaEventSynchronize(c.done[b]), "download chunk");
        const int64_t o = ch * chunk, cnt = std::min(chunk, n - o);
        std::memcpy(out + o, c.hG[b], size_t(cnt) * sizeof(uint4));
    };
    for (int64_t ch = 0; ch < chunks; ch++) {
        if (ch >= kRing)
            copy_chunk(ch - kRing);
        const int b = int(ch % kRing);
        const int64_t o = ch * chunk, cnt = std::min(chunk, n - o);
        check(cudaMemcpyAsync(c.hG[b], c.dD + o, size_t(cnt) * sizeof(uint4), cudaMemcpyDeviceToHost, c.stream), "copy out");
        check(cudaEventRecord(c.done[b], c.stream), "event");
    }
    for (int64_t ch = std::max<int64_t>(0, chunks - kRing); ch < chunks; ch++) copy_chunk(ch);
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
