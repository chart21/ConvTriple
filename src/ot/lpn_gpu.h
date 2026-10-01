#ifndef CHEETAH_LPN_GPU_H
#define CHEETAH_LPN_GPU_H
// Ferret's LPN step on the GPU (lpn_gpu.cu): bit for bit LpnF2<IO, 10>::task's groups of 4 outputs. Plain C++ interface.
#include <cstdint>

namespace cheetah::lpn_gpu {
// a CUDA device is present (LPN_GPU=0 says no)
bool available();
// nn[start + 4g .. + 3] ^= the gathers of group g < groups (indices from AES-128 with these round keys, as 44 words in
// AES-NI's byte order); kk: the k-block table. Thread-safe (a stream and buffers per calling thread).
void compute(void* nn, const void* kk, int64_t n, int64_t k, uint32_t mask, const uint32_t round_keys[44], int64_t start,
             int64_t groups);
// throws unless the GPU's AES of the blocks (t, 0, 7, 0), t < n, equals `expected` (4 words each)
void aes_reference_check(const uint32_t round_keys[44], const uint32_t* expected, int n);
} // namespace cheetah::lpn_gpu
#endif
