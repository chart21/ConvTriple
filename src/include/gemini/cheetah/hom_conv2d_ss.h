//  Authors: Wen-jie Lu on 2021/9/11.
#ifndef GEMINI_CHEETAH_HOMCONVSS_H_
#define GEMINI_CHEETAH_HOMCONVSS_H_
#include <seal/secretkey.h>
#include <seal/serializable.h>

#include "gemini/cheetah/tensor.h"
#include <optional>
#include <vector>

#include "gemini/cheetah/tensor_shape.h"
#include "gemini/core/util/ThreadPool.h"

// Forward
namespace seal {
class SEALContext;
class PublicKey;

class Plaintext;
class Ciphertext;
class Evaluator;
} // namespace seal

namespace gemini {

Code LaunchWorks(ThreadPool& tpool, size_t num_works,
                 std::function<Code(long wid, size_t start, size_t end)> program);

class TensorEncoder;

// template <typename T> class Tensor;

class HomConv2DSS {
  public:
// #if CONV_USE_CUDA
//   static constexpr size_t kMaxThreads = 1;
// #else
#ifdef HOM_CONV2D_SS_MAX_THREADS
    static constexpr size_t kMaxThreads = HOM_CONV2D_SS_MAX_THREADS;
#else
    static constexpr size_t kMaxThreads = 32;
#endif
    // #endif

    struct Meta {
        TensorShape ishape;
        TensorShape fshape;
        size_t n_filters;
        Padding padding;
        size_t stride;
        bool is_shared_input;
    };

    explicit HomConv2DSS() = default;

    ~HomConv2DSS() = default;

    inline std::string get_str() const { return "conv"; }

    Code setUp(const seal::SEALContext& context, std::optional<seal::SecretKey> sk = std::nullopt,
               std::shared_ptr<seal::PublicKey> pk = nullptr);

    [[nodiscard]] seal::scheme_type scheme() const;

    [[nodiscard]] size_t poly_degree() const;

    uint64_t plain_modulus() const;

    Code encryptImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                      std::vector<seal::Serializable<seal::Ciphertext>>& encrypted_share,
                      std::vector<seal::Plaintext>& encoded_share,
                      size_t nthreads = 1) const; // added

    Code encryptImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                      std::vector<seal::Serializable<seal::Ciphertext>>& encrypted_share,
                      size_t nthreads = 1) const;

    Code encodeImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                     std::vector<seal::Plaintext>& encoded_share, size_t nthreads = 1) const;

    Code encodeFilters(const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                       std::vector<std::vector<seal::Plaintext>>& encoded_filters,
                       size_t nthreads = 1) const;

    Code filtersToNtt(std::vector<std::vector<seal::Plaintext>>& encoded_filters,
                      size_t nthreads = 1) const;

    // Prescribed-share ("flipped") variant: the FILTER owner encrypts its filters ...
    Code encryptFilters(const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                        std::vector<std::vector<seal::Serializable<seal::Ciphertext>>>& enc_filters,
                        size_t nthreads = 1) const;

    // ... and the IMAGE owner evaluates conv(pt_image, ct_filters) and masks the result with
    // CALLER-PRESCRIBED output-share coefficients, so the filter owner decrypts (conv - prescribed)
    // and no share-fixing communication is needed. `prescribed` must have GetConv2DOutShape(meta).
    // Used for MODELWEIGHTS_KNOWN_DURING_PREPROCESSING where the image owner's triple share must
    // equal a PRNG-derived value. Filter cts are NTT-transformed in place on first use (cacheable
    // across batches); `image` polys are consumed (NTT-transformed in place).
    Code conv2DSSPrescribed(std::vector<std::vector<seal::Ciphertext>>& enc_filters,
                            std::vector<seal::Plaintext>& image, const Meta& meta,
                            const Tensor<uint64_t>& prescribed,
                            std::vector<seal::Ciphertext>& out_ct, size_t nthreads = 1) const;

    Code conv2DSS(const std::vector<seal::Ciphertext>& img_share0,
                  const std::vector<seal::Plaintext>& img_share1,
                  const std::vector<std::vector<seal::Plaintext>>& filters, const Meta& meta,
                  std::vector<seal::Ciphertext>& out_share0, Tensor<uint64_t>& out_share1,
                  size_t nthreads = 1, bool in_ntt = true, bool fil_ntt = true,
                  bool out_ntt = true) const;

    Code decryptToTensor(const std::vector<seal::Ciphertext>& enc_tensor, const Meta& meta,
                         Tensor<uint64_t>& out, size_t nthreads = 1) const;

    Code idealFunctionality(const Tensor<uint64_t>& in_tensor,
                            const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                            Tensor<uint64_t>& out_tensor) const;

    seal::SEALContext getContext() const { return *context_; }

  protected:
    size_t conv2DOneFilter(const std::vector<seal::Ciphertext>& enc_tensor,
                           const std::vector<seal::Plaintext>& filter, const Meta& meta,
                           seal::Ciphertext* out_buff, size_t out_buff_sze,
                           bool to_ntt = false) const;

    size_t conv2DOneFilterPrescribed(const std::vector<seal::Plaintext>& image,
                                     const std::vector<seal::Ciphertext>& filter,
                                     seal::Ciphertext* out_buff, size_t out_buff_sze) const;

    Code maskWithPrescribed(std::vector<seal::Ciphertext>& enc_tensor,
                            const Tensor<uint64_t>& prescribed, const Meta& meta,
                            size_t nthreads = 1) const;

    Code sampleRandomMask(const std::vector<size_t>& targets, uint64_t* coeffs_buff,
                          size_t buff_size, seal::Plaintext& mask, seal::parms_id_type pid,
                          std::shared_ptr<seal::UniformRandomGenerator> prng, bool is_ntt) const;

    Code addRandomMask(std::vector<seal::Ciphertext>& enc_tensor, Tensor<uint64_t>& mask_tensor,
                       const Meta& meta, size_t nthreads = 1) const;

    Code removeUnusedCoeffs(std::vector<seal::Ciphertext>& ct, const Meta& meta,
                            double* density = nullptr) const;

    Code postProcessInplace(seal::Plaintext& pt, std::vector<size_t>& targets, uint64_t* out_poly,
                            size_t out_buff_size) const;

    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<TensorEncoder> tencoder_{nullptr};
    std::shared_ptr<seal::Evaluator> evaluator_{nullptr};
    std::shared_ptr<seal::Encryptor> encryptor_{nullptr};
    std::shared_ptr<seal::PublicKey> pk_{nullptr};
    std::optional<seal::SecretKey> sk_{std::nullopt};

#ifdef CONV_USE_CUDA
    troy::SEALContextCuda* contextCu_;
    troy::EvaluatorCuda* evaluatorCu_;
#endif
};

TensorShape GetConv2DOutShape(const HomConv2DSS::Meta& meta);

}; // namespace gemini
#endif // GEMINI_CHEETAH_HOMCONVSS_H_