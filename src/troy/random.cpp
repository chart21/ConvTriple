#include <openssl/rand.h>

#include <stdexcept>

#include "troy/conv2d_gpu.cuh"

void TROY::random_ring(INT_TYPE* dst, size_t n) {
    constexpr size_t chunk = size_t(1) << 28; // RAND_bytes takes an int byte count
    for (size_t off = 0; off < n; off += chunk) {
        int bytes = static_cast<int>(std::min(chunk, n - off) * sizeof(INT_TYPE));
        if (RAND_bytes(reinterpret_cast<unsigned char*>(dst + off), bytes) != 1)
            throw std::runtime_error("RAND_bytes failed");
    }
}
