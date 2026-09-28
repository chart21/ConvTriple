#ifndef CONV_PACKED_HPP_
#define CONV_PACKED_HPP_

// Conv triples on the CPU (SEAL) with the packing of the GPU path (troy's Conv2dHelper): an input
// ciphertext holds a tile of several images and input channels, and one weight polynomial folds several
// output channels into the product, so small feature maps no longer cost one output ciphertext per
// output channel as in Cheetah's HomConv2DSS. Output ciphertexts are masked, noise-flooded, truncated
// and stripped of unused coefficients like HomConv2DSS's.

#include <cstdint>
#include <memory>
#include <type_traits>

#include <seal/seal.h>

#include "constants.hpp"
#include "io/net_io_channel.hpp"

namespace Iface {

class PackedConv2D {
  public:
    using Word = std::conditional_t<BIT_LEN == 32, uint32_t, uint64_t>;

    void setUp(const seal::SEALContext& context, const seal::SecretKey& sk,
               std::shared_ptr<seal::PublicKey> other_pk);

    // Shares c of the stride-1, unpadded conv(x, w) (NCHW input, OIHW weights) of bs images.
    // AB2: the party without w encrypts its x, the other one evaluates (and adds its own x if given);
    // AB: both parties hold shares of x and w and play both roles.
    void conv(IO::NetIO** ios, int party, const Word* x, const Word* w, Word* c, size_t bs, size_t ic,
              size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, bool is_ab, size_t threads) const;

  private:
    struct Tiling;
    struct Ntt; // the primes' NTT tables, for NTTs of weight polynomials outside SEAL

    void encrypt(const Tiling& t, const Word* x, std::string& out, size_t threads) const;
    void evaluate(const Tiling& t, const std::string& in, const Word* x_own, const Word* w, Word* r,
                  std::string& out, size_t threads) const;
    void decrypt(const Tiling& t, const std::string& in, Word* c, bool accumulate, size_t threads) const;

    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<seal::Encryptor> encryptor_;
    std::shared_ptr<seal::Evaluator> evaluator_;
    std::shared_ptr<seal::Decryptor> decryptor_;
    std::shared_ptr<seal::PublicKey> other_pk_;
    std::shared_ptr<const Ntt> ntt_;
};

} // namespace Iface

#endif
