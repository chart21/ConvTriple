#ifndef CONV_PACKED_GPU_HPP_
#define CONV_PACKED_GPU_HPP_

// The evaluator of PackedConv2D (conv_packed.cpp) on a GPU: the weight polynomials are built and NTT-transformed on
// the device, the products summed there, and the input NTT and output inverse NTT run there too. The transforms are
// SEAL's (ntt_negacyclic_harvey with SEAL's tables, same slot order), so the inputs (c1 arrives in SEAL's NTT form)
// and the outputs (flooded, masked and truncated with SEAL on the host) are the same values as on the CPU, and so are
// the triples. Plain C++ interface: conv_packed.cpp needs no CUDA headers.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace Iface::packed_gpu {

// One prime of the ring, as SEAL's NTTTables and Modulus hold it
struct Prime {
    uint64_t q;
    uint64_t ratio0, ratio1;                       // const_ratio: floor(2^128 / q)
    std::vector<uint64_t> root, root_quot;         // get_from_root_powers(): operand, Shoup quotient (N each)
    std::vector<uint64_t> inv_root, inv_root_quot; // get_from_inv_root_powers()
    uint64_t inv_n, inv_n_quot;                    // inv_degree_modulo()
    uint64_t last, last_quot;                      // inv_root[N - 1] * inv_n, the inverse NTT's last stage
    uint64_t increment;                            // the centered lift of plaintext coefficients >= threshold
};

// A tiling (see PackedConv2D::Tiling): tiles x in_groups input ciphertexts, out_groups weight polynomials per input
// group, co filters x ci channels of kh x kw per weight polynomial in an h x w tile
struct Geometry {
    size_t tiles, in_groups, out_groups, ic, oc, ci, co, h, w, kh, kw;
};

class Engine {
  public:
    Engine(size_t N, uint64_t threshold, const std::vector<Prime>& primes, size_t fold);
    ~Engine();
    // y[tile][og] = sum_g x[tile][g] * W[og][g]. x: tiles * in_groups ciphertexts of 2 x L x N words each, c0 in
    // coefficient form and c1 in NTT form; w: the OIHW weights (oc x ic x kh x kw words); y: tiles * out_groups
    // ciphertexts of 2 x L x N words in coefficient form. Thread-safe: each call runs on a stream of its own.
    void evaluate(const Geometry& g, const uint64_t* x, const uint32_t* w, uint64_t* y) const;
    void evaluate(const Geometry& g, const uint64_t* x, const uint64_t* w, uint64_t* y) const;
    // pinned host memory of at least `words` words for the calling thread (which: 0 inputs, 1 outputs), kept across
    // calls: evaluate's copies from and to it run at full speed
    uint64_t* staging(size_t words, int which) const;

    struct Impl; // opaque (conv_packed_gpu.cu)

  private:
    std::unique_ptr<Impl> impl_;
};

// a CUDA device is present (CONV_GPU=0 says no)
bool available();

} // namespace Iface::packed_gpu

#endif
