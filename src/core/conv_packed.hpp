#ifndef CONV_PACKED_HPP_
#define CONV_PACKED_HPP_

// Conv triples on the CPU (SEAL) with the packing of the GPU path (troy's Conv2dHelper): an input
// ciphertext holds a tile of several images and input channels, and one weight polynomial folds several
// output channels into the product, so small feature maps no longer cost one output ciphertext per
// output channel as in Cheetah's HomConv2DSS. Output ciphertexts are masked, noise-flooded and truncated
// like HomConv2DSS's. On the wire, both directions carry exactly the bits decryption needs: an input
// ciphertext is a seed and c0, an output ciphertext the kept high bits of c1 and of its used c0 coefficients.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include <seal/seal.h>

#include "constants.hpp"
#include "io/net_io_channel.hpp"

namespace Iface {

namespace packed_gpu {
class Engine;
}

// Output repacking (hpmpc CHEETAH_CONV_REPACK, set before the first Keys::instance; the environment variable
// CONV_REPACK=0/1 overrides it): the communication-optimized variant. The convolutions run in their own ring of
// N = 8192 with a special prime, since packing needs key switching and the 109-bit data modulus already takes
// the 128-bit budget of N = 4096. An input ciphertext holds channels interleaved (dense, one filter per product),
// and the evaluator merges the sparse products of C filters into one dense output ciphertext with Galois
// automorphisms before masking, flooding and sending it. conv_repack_ab(): both parties evaluate (AB triples), so
// both send their Galois keys; otherwise (AB2) only the input holder does.
inline bool& conv_repack() {
    static bool on = false;
    return on;
}
inline bool& conv_repack_ab() {
    static bool ab = false;
    return ab;
}
// A limit on the threads of the packed convolutions, read at every encryption, evaluation and decryption (0: the
// caller's count). hpmpc's CHEETAH_CONV_EARLY lowers it while the OT phase runs next to the conv triples and lifts it
// afterwards, so that the conv triples take the idle cores first and all of them once the OT phase is done.
inline std::atomic<size_t>& conv_threads_now() {
    static std::atomic<size_t> n{0};
    return n;
}

class PackedConv2D {
  public:
    using Word = std::conditional_t<BIT_LEN == 32, uint32_t, uint64_t>;

    void setUp(const seal::SEALContext& context, const seal::SecretKey& sk,
               std::shared_ptr<seal::PublicKey> other_pk);
    // conv_repack(): replaces setUp's context by the N = 8192 one (own keys, public keys exchanged on ios[0]) and
    // exchanges the Galois keys the evaluation needs; party is emp's (ALICE evaluates in AB2)
    void setUpRepack(IO::NetIO** ios, int party, bool both_evaluate);

    // Shares c of the stride-1, unpadded conv(x, w) (NCHW input, OIHW weights) of bs images.
    // AB2: the party without w encrypts its x, the other one evaluates (and adds its own x if given);
    // AB: both parties hold shares of x and w and play both roles.
    void conv(IO::NetIO** ios, int party, const Word* x, const Word* w, Word* c, size_t bs, size_t ic,
              size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, bool is_ab, size_t threads) const;

    // One convolution of conv_pipelined, with conv()'s operands; finish() runs once its shares are in c
    struct Job {
        const Word* x = nullptr;
        const Word* w = nullptr;
        Word* c       = nullptr;
        size_t bs, ic, ih, iw, kh, kw, oc;
        std::function<void()> finish;
    };
    // conv() of several convolutions, pipelined across them: the parties encrypt the next convolutions
    // and decrypt the previous ones while the evaluator works on the current one, instead of waiting for
    // each other at every layer. prepare(i) gives convolution i (called once, in order), batch[i] its
    // number of images. Takes 2 channels (AB2) or 4 (AB).
    void conv_pipelined(IO::NetIO** ios, int party, const std::vector<size_t>& batch,
                        const std::function<Job(size_t)>& prepare, bool is_ab, size_t threads) const;

  private:
    // calls of encrypt and evaluate so far (each runs on one thread): the streams of a seeded run
    mutable uint64_t enc_calls_ = 0, eval_calls_ = 0;
    struct Tiling;
    struct Ntt;  // the primes' NTT tables, for NTTs of weight polynomials outside SEAL
    struct Wire; // bit widths and sizes of the ciphertexts on the wire

    bool repack_ = false;
    std::shared_ptr<const seal::GaloisKeys> other_gk_; // the other party's, to pack the products of its ciphertexts
    // repacking: merges the sparse products y (tile-major, oc per tile) into t.tiles * t.out_groups dense ones
    void pack(const Tiling& t, std::vector<seal::Ciphertext>& y, std::vector<seal::Ciphertext*>& dense,
              size_t threads) const;

    void encrypt(const Tiling& t, const Word* x, std::string& out, size_t threads) const;
    // call: the PRNG stream of the masks and flooding (default: the next of eval_calls_); concurrent evaluations
    // pass the index the serial order would have given them, so the outputs do not depend on scheduling
    void evaluate(const Tiling& t, const std::string& in, const Word* x_own, const Word* w, Word* r,
                  std::string& out, size_t threads, uint64_t call = UINT64_MAX) const;
    // evaluate's products y[tile][og] = sum_g x[tile][g] * w[og][g] (y sized by the caller): on the CPU, or on the GPU
    // (gpu_, built with TRIPLE_GPU; not for repacking)
    void multiply(const Tiling& t, const std::string& in, const Word* x_own, const Word* w,
                  std::vector<seal::Ciphertext>& y, size_t threads) const;
    void multiply_gpu(const Tiling& t, const std::string& in, const Word* x_own, const Word* w,
                      std::vector<seal::Ciphertext>& y, size_t threads) const;
    void decrypt(const Tiling& t, const std::string& in, Word* c, bool accumulate, size_t threads) const;

    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<seal::Encryptor> encryptor_;
    std::shared_ptr<seal::Evaluator> evaluator_;
    std::shared_ptr<seal::Decryptor> decryptor_;
    std::shared_ptr<seal::PublicKey> other_pk_;
    std::shared_ptr<const seal::SecretKey> sk_;
    std::shared_ptr<const Ntt> ntt_;
    std::shared_ptr<const Wire> wire_;
    std::shared_ptr<const packed_gpu::Engine> gpu_; // set by setUp when a GPU is there (TRIPLE_GPU builds)
};

} // namespace Iface

#endif
