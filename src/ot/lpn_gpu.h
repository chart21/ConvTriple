#ifndef CHEETAH_LPN_GPU_H
#define CHEETAH_LPN_GPU_H
// Ferret's LPN step and MPCOT trees on the GPU (lpn_gpu.cu), bit for bit emp's. Plain C++ interface.
#include <cstddef>
#include <cstdint>

namespace cheetah::lpn_gpu {
// a CUDA device is present (LPN_GPU=0 says no)
bool available();
// and MPCOT's trees go there too (MPCOT_GPU=0 says no: the LPN step alone)
bool mpcot_available();
// nn[start + 4g .. + 3] ^= the gathers of group g < groups (indices from AES-128 with these round keys, as 44 words in
// AES-NI's byte order); kk: the k-block table. Thread-safe (a stream and buffers per calling thread).
void compute(void* nn, const void* kk, int64_t n, int64_t k, uint32_t mask, const uint32_t round_keys[44], int64_t start,
             int64_t groups);

// One ferret extension on the GPU (semi-honest MpcotReg::mpcot, then LpnF2<IO, 10>::compute's groups of 4; the same
// outputs bit for bit): MPCOT's GGM trees are built on the device from the sender's seeds or the receiver's messages,
// the LPN step runs on the leaves in place, and one copy brings the outputs back. The OTs of the tree messages stay on
// the host (emp's objects). Leases a context (stream, buffers) for its lifetime; calls run in order on its stream.
class Extension {
  public:
    Extension();
    ~Extension();
    Extension(const Extension&)            = delete;
    Extension& operator=(const Extension&) = delete;
    // device memory for `trees` trees of `depth` levels (2^(depth - 1) leaves each) and a k-block LPN table; throws if
    // the device has too little memory left (call it before any communication, to fall back to the CPU)
    void reserve(int64_t trees, int depth, int64_t k);
    // the LPN step's table (kk, k blocks)
    void upload_kk(const void* kk, int64_t k);
    // the sender's trees from their seeds (trees blocks; keys: TwoKeyPRP's two AES-128 key schedules, 44 words each);
    // msg: trees x (depth - 1) x {XOR of the even, of the odd nodes} of the levels 1 .. depth - 1, the leaves' before
    // bit 0 is cleared (the OT messages; the secret sum is ((even ^ odd) of the leaves with bit 0 cleared) ^ Delta).
    // Returns when msg is filled.
    void sender_trees(const void* seeds, int64_t trees, int depth, const uint32_t keys[88], void* msg);
    // the receiver's trees: msg (trees x (depth - 1)), the received message of each level; bits (same shape), the
    // choice bits (SPCOT_Recver::b); secret (trees), the secret sums
    void recver_trees(const void* msg, const uint8_t* bits, const void* secret, int64_t trees, int depth,
                      const uint32_t keys[88]);
    // the LPN step's groups of 4 outputs start + 4g, g < groups, on the leaves (the sparse vector)
    void lpn(int64_t k, uint32_t mask, const uint32_t round_keys[44], int64_t start, int64_t groups);
    // the first n outputs into host memory
    void download(void* out, int64_t n);

  private:
    void* ctx_;
};

// host memory the device copies into directly (cudaHostRegister): pin() once per buffer (false if that failed), unpin()
// before the buffer is freed; download() into a pinned buffer skips the bounce buffers
bool pin(void* p, size_t bytes);
void unpin(void* p);

// throws unless the GPU's AES of the blocks (t, 0, 7, 0), t < n, equals `expected` (4 words each)
void aes_reference_check(const uint32_t round_keys[44], const uint32_t* expected, int n);
} // namespace cheetah::lpn_gpu
#endif
