#ifndef BENCHING_HPP_
#define BENCHING_HPP_

#include <memory>

#include <seal/seal.h>

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>

#include "io/net_io_channel.hpp"
#include "io/send.hpp"

#include <ot/cheetah-ot_pack.h>

#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "utils.hpp"

using Utils::PROTO;
using Utils::Result;

namespace HE_OT {

template <class Channel>
class HE {
  public:
    /**
     * Initializes communication channels/seal contexts and exchanges public
     * keys between Alice and Bob
     *
     * @param party 1 Alice / 2 Bob
     * @param addr nullptr server / IP-addr client
     * @param port First port to use
     * @param threads
     * @param samples Number of repeats (benchmark)
     * @param setup_ot True if OT required (takes some time) otherwise false
     * @param setup_bn Whether to setup encryption contexts for BN
     */
    explicit HE(const int& party, const char* addr, const int& port, const size_t& threads,
                size_t& samples, bool setup_ot = true, bool setup_bn = true);

    HE(const HE& other) = delete;
    HE(HE&& other)      = delete;

    HE& operator=(const HE& other) = delete;
    HE& operator=(HE&& other)      = delete;

    ~HE() {
        for (size_t i = 0; i < threads_; ++i) delete ios_[i];

        delete[] ios_;
    }

    const gemini::HomConv2DSS& get_conv() const { return conv_; }
    const gemini::HomFCSS& get_fc() const { return fc_; }
    const gemini::HomBNSS& get_bn() const { return bn_; }

    /**
     * Uses Random random values for inference.
     */
    template <class T>
    void test_he(std::vector<class T::Meta>& layers, const T& cheetah, const size_t& batchSize = 1,
                 PROTO proto = PROTO::AB);

    /**
     * Computes 2D-Convolution
     * @param A input data
     * @param B input filter
     * @param C result Tensor
     * @param stride
     * @param padding
     * @param batchSize
     */
    void run_conv(const Tensor<uint64_t>& A, const std::vector<Tensor<uint64_t>>& B,
                  Tensor<uint64_t>& C, const size_t& stride, const size_t& padding,
                  const size_t& batchSize = 1);

    /**
     * Performs Scaling/BatchNorm
     * @param A input data
     * @param B scales
     * @param C result Tensor
     * @param batchSize
     */
    void run_bn(const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                const size_t& batchSize);

    /**
     * Elementwise Multiplication
     * @param A input data
     * @param B scales
     * @param C result Tensor
     * @param batchSize
     */
    void run_elem_mult(const vector<Tensor<uint64_t>>& A, const vector<Tensor<uint64_t>>& B,
                       vector<Tensor<uint64_t>>& C, const size_t& batchSize,
                       PROTO proto = PROTO::AB);

    /**
     * Wrapper function to perform Conv/Batchnorm/FC
     */
    template <class T, class Bs>
    void run_he(const T& cheetah, const class T::Meta& meta, const std::vector<Tensor<uint64_t>>& A,
                const std::vector<Bs>& B, std::vector<Tensor<uint64_t>>& C,
                const size_t& batchSize = 1, PROTO proto = PROTO::AB);

    /**
     * Generates beaver boolean-triple
     */
    void run_ot(const size_t& numTriples, bool packed = false);

    /**
     * Do BatchNorm via FC
     */
    double alt_bn(const gemini::HomBNSS::Meta& meta);

  private:
    gemini::HomBNSS bn_;       // cheetah BatchNorm
    gemini::HomConv2DSS conv_; // cheetah 2d-conv
    gemini::HomFCSS fc_;       // cheetah-fc

    std::unique_ptr<sci::OTPack<Channel>> ot_pack_; // cheeath triple-gen
    std::unique_ptr<TripleGenerator<Channel>> triple_gen_;

    seal::SEALContext context_ = Utils::init_he_context();
    std::vector<std::shared_ptr<seal::SEALContext>> bn_contexts_;
    std::vector<std::shared_ptr<seal::PublicKey>> bn_pks_; // public keys (BN)
    std::vector<std::shared_ptr<seal::SecretKey>> bn_sks_; // secret keys (BN)

    size_t threads_;
    size_t samples_; // for benchmarking

    Channel** ios_; // <thread>xChannels

    int party_; // 1 (Alice)/2 (Bob)

    template <class SerKey>
    void exchange_keys_(const SerKey& pkey, seal::PublicKey& o_pkey, const seal::SEALContext& ctx);

    void setup_OT();

    /**
     * Creates/exchanges keys required for BN
     */
    void setup_BN(const std::optional<seal::SecretKey>& skey,
                  const std::shared_ptr<seal::PublicKey>& o_pkey);

    /**
     * @return Total number of bytes sent
     */
    inline size_t counter_() {
        size_t counter = 0;
        for (size_t i = 0; i < threads_; ++i) counter += ios_[i]->counter;
        return counter;
    }

    /**
     * @return Total number of bytes received
     */
    inline size_t recv_counter_() {
        size_t counter = 0;
        for (size_t i = 0; i < threads_; ++i) counter += ios_[i]->recv_counter;
        return counter;
    }

    /**
     * Sets the byte counters back to zero (for all channels)
     */
    inline void reset_counter_() {
        for (size_t i = 0; i < threads_; ++i) {
            ios_[i]->counter = 0;
            ios_[i]->recv_counter = 0;
        }
    }
};

template <class Channel>
HE<Channel>::HE(const int& party, const char* addr, const int& port, const size_t& threads,
                size_t& samples, bool setup_ot, bool setup_bn)
    : threads_(threads), samples_(samples), party_(party) {
    Code code;

    ios_ = Utils::init_ios<Channel>(addr, port, threads);

    if (setup_ot)
        setup_OT();

    seal::KeyGenerator keygen(context_);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys_(*pkey, *o_pkey, context_);

    code = conv_.setUp(context_, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
    code = fc_.setUp(context_, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));

    if (setup_bn)
        setup_BN(skey, o_pkey);

    reset_counter_();
}

template <class Channel>
void HE<Channel>::setup_BN(const std::optional<seal::SecretKey>& skey,
                           const std::shared_ptr<seal::PublicKey>& o_pkey) {
    using namespace seal;
    size_t ntarget_bits = std::ceil(std::log2(PLAIN_MOD));
    size_t crt_bits     = 2 * ntarget_bits + 1 + gemini::HomBNSS::kStatBits;

    const size_t nbits_per_crt_plain = [](size_t crt_bits) {
        constexpr size_t kMaxCRTPrime = 50;
        for (size_t nCRT = 1;; ++nCRT) {
            size_t np = gemini::CeilDiv(crt_bits, nCRT);
            if (np <= kMaxCRTPrime)
                return np;
        }
    }(crt_bits + 1);

    const size_t nCRT = gemini::CeilDiv<size_t>(crt_bits, nbits_per_crt_plain);
    std::vector<int> crt_primes_bits(nCRT, nbits_per_crt_plain);

    const size_t N  = POLY_MOD;
    auto plain_crts = CoeffModulus::Create(N, crt_primes_bits);
    EncryptionParameters seal_parms(scheme_type::bfv);
    seal_parms.set_n_special_primes(0);
    // We are not exporting the pk/ct with more than 109-bit.
    std::vector<int> cipher_moduli_bits{60, 49};
    seal_parms.set_poly_modulus_degree(N);
    seal_parms.set_coeff_modulus(CoeffModulus::Create(N, cipher_moduli_bits));

    bn_contexts_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        seal_parms.set_plain_modulus(plain_crts[i]);
        bn_contexts_[i] = std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);
    }

    std::vector<seal::SEALContext> contexts;
    std::vector<std::optional<SecretKey>> opt_sks;
    bn_sks_.resize(nCRT);
    bn_pks_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        KeyGenerator keygen(*bn_contexts_[i]);
        bn_sks_[i]                   = std::make_shared<SecretKey>(keygen.secret_key());
        bn_pks_[i]                   = std::make_shared<PublicKey>();
        Serializable<PublicKey> s_pk = keygen.create_public_key();

        exchange_keys_(s_pk, *bn_pks_[i], *bn_contexts_[i]);
        contexts.emplace_back(*bn_contexts_[i]);
        opt_sks.emplace_back(*bn_sks_[i]);
    }

    Code code;
    code = bn_.setUp(PLAIN_MOD, context_, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
    code = bn_.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
}

template <class Channel>
void HE<Channel>::setup_OT() {
    std::string unit;
    auto start  = measure::now();
    ot_pack_    = std::make_unique<sci::OTPack<Channel>>(ios_, threads_, party_);
    triple_gen_ = std::make_unique<TripleGenerator<Channel>>(party_, ios_[0], ot_pack_.get());
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT startup time: ", Utils::to_sec(Utils::time_diff(start)));
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT startup data: ", Utils::to_MB(counter_(), unit), unit);
}

template <class Channel>
template <class SerKey>
void HE<Channel>::exchange_keys_(const SerKey& pkey, seal::PublicKey& o_pkey,
                                 const seal::SEALContext& ctx) {
    switch (party_) {
    case emp::ALICE:
        IO::send_pkey(*(ios_[0]), pkey);
        IO::recv_pkey(*(ios_[0]), ctx, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(*(ios_[0]), ctx, o_pkey);
        IO::send_pkey(*(ios_[0]), pkey);
        break;
    }
}

template <class Channel>
void HE<Channel>::run_ot(const size_t& numTriples, bool packed) {
    size_t len = numTriples / (packed ? 8 : 1);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    Utils::log(Utils::Level::DEBUG, "Num Triples: ", numTriples);

    auto start = measure::now();
    for (size_t i = 0; i < samples_; ++i) {
        switch (party_) {
        case emp::ALICE:
            Server::triple_gen(*triple_gen_, a, b, c, numTriples, packed);
            break;
        case emp::BOB:
            Client::triple_gen(*triple_gen_, a, b, c, numTriples, packed);
            break;
        }
    }
    std::string unit;
    auto data = Utils::to_MB(counter_(), unit);
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT TIME[s]: ", Utils::to_sec(Utils::time_diff(start)) / samples_);
    Utils::log(Utils::Level::INFO, "P", party_, ": OT data[", unit, "]: ", (data / samples_));

    delete[] a;
    delete[] b;
    delete[] c;
    reset_counter_();
}

template <class Channel>
template <class T>
void HE<Channel>::test_he(std::vector<class T::Meta>& layers, const T& cheetah,
                          const size_t& batchSize, PROTO proto) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads_ / batch_threads;

    double total      = 0;
    double total_data = 0;
    std::string proto_str("AB");

    std::vector<Result> results(samples_);          // all samples
    std::vector<Result> all_results(layers.size()); // averaged samples

    for (size_t i = 0; i < layers.size(); ++i) {
        ios_[0]->sync();
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);
        double tmp_total = 0;

        for (size_t round = 0; round < samples_; ++round) {
            std::vector<Result> batches_results(batch_threads);
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if ((proto == PROTO::AB2 && party_ == emp::ALICE)
                        || (proto == PROTO::AB && (cur + party_ - 1) % 2 == 0)) {
                        result = Server::perform_proto(layers[i], ios_ + wid * threads_per_thread,
                                                       cheetah, threads_per_thread, proto);
                    } else {
                        result = Client::perform_proto(layers[i], ios_ + wid * threads_per_thread,
                                                       cheetah, threads_per_thread, proto);
                    }

                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            gemini::ThreadPool tpool(batch_threads);

            auto start = measure::now();
            auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
            total += Utils::to_sec(Utils::time_diff(start));
            if (code != Code::OK)
                Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), " ",
                           CodeMessage(code));

            results[round] = Utils::average(batches_results, false);
        }

        total += tmp_total / samples_;

        all_results[i] = Utils::average(results, true);
        total_data += all_results[i].bytes;
    }

    switch (party_) {
    case emp::ALICE:
        Utils::make_csv(all_results, batchSize, threads_,
                        "party" + std::to_string(party_) + "_" + cheetah.get_str() + "_" + proto_str
                            + ".csv");
        break;
    case emp::BOB:
        Utils::make_csv(all_results, batchSize, threads_,
                        "party" + std::to_string(party_) + "_" + cheetah.get_str() + "_" + proto_str
                            + ".csv");
        break;
    }

    std::string unit;
    total_data = Utils::to_MB(total_data, unit);
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [" << unit << "]: " << total_data << "\n";

    reset_counter_();
}

template <class Channel>
double HE<Channel>::alt_bn(const gemini::HomBNSS::Meta& meta_bn) {
    auto meta = Utils::init_meta_fc(1, meta_bn.ishape.height());

    auto start = measure::now();

    Result res;
    switch (party_) {
    case emp::ALICE: {
        std::cerr << meta_bn.ishape.height() << " x " << meta_bn.vec_shape.num_elements() << "\n";
        res = Server::perform_proto(meta, ios_, fc_, threads_, meta_bn.vec_shape.num_elements());
        break;
    }
    case emp::BOB: {
        res = Client::perform_proto(meta, ios_, fc_, threads_, meta_bn.vec_shape.num_elements());
        break;
    }
    }

    double time = Utils::to_sec(Utils::time_diff(start));

    return time;
}

template <class Channel>
template <class T, class Bs>
void HE<Channel>::run_he(const T& cheetah, const class T::Meta& meta,
                         const std::vector<Tensor<uint64_t>>& A, const std::vector<Bs>& B,
                         std::vector<Tensor<uint64_t>>& C, const size_t& batchSize, PROTO proto) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads_ / batch_threads;

    double total      = 0;
    double total_data = 0;
    std::string proto_str("AB");
    proto_str += proto == PROTO::AB2 ? "2" : "";

    std::vector<Result> batches_results(batch_threads);
    auto batch = [&](long wid, size_t start, size_t end) -> Code {
        for (size_t cur = start; cur < end; ++cur) {
            Result result;
            if ((PROTO::AB2 == proto && party_ == emp::ALICE)
                || (PROTO::AB == proto && (cur + party_ - 1) % 2 == 0)) {
                result = Server::perform_proto(meta, ios_ + wid * threads_per_thread, cheetah,
                                               A[cur], B[cur], C[cur], threads_per_thread, proto);
            } else {
                result = Client::perform_proto(meta, ios_ + wid * threads_per_thread, cheetah,
                                               A[cur], B[cur], C[cur], threads_per_thread, proto);
            }

            if (result.ret != Code::OK)
                return result.ret;

            Utils::add_result(batches_results[wid], result);
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(batch_threads);

    auto start = measure::now();
    auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
    total += Utils::to_sec(Utils::time_diff(start));
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), " ", CodeMessage(code));

    std::string unit;
    total_data = Utils::to_MB(total_data, unit);
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [" << unit << "]: " << total_data << "\n";

    reset_counter_();
}

template <class Channel>
void HE<Channel>::run_conv(const Tensor<uint64_t>& A, const std::vector<Tensor<uint64_t>>& B,
                           Tensor<uint64_t>& C, const size_t& stride, const size_t& padding,
                           const size_t& batchSize) {
    gemini::HomConv2DSS::Meta meta
        = Utils::init_meta_conv(A.channels(), A.height(), A.width(), B[0].channels(), B[0].height(),
                                B[0].width(), B.size(), stride, padding);
    run_he(conv_, meta, A, B, C, batchSize);
}

template <class Channel>
void HE<Channel>::run_bn(const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                         const size_t& batchSize) {
    gemini::HomBNSS::Meta meta
        = Utils::init_meta_bn(B.NumElements(), A.shape().rows() * A.shape().rows());

    run_he(bn_, meta, A, B, C, batchSize);
}

template <class Channel>
void HE<Channel>::run_elem_mult(const vector<Tensor<uint64_t>>& A,
                                const vector<Tensor<uint64_t>>& B, vector<Tensor<uint64_t>>& C,
                                const size_t& batchSize, PROTO proto) {
    gemini::HomBNSS::Meta meta;
    meta.vec_shape       = {A[0].NumElements()};
    meta.is_shared_input = true;
    meta.target_base_mod = PLAIN_MOD;

    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads_ / batch_threads;

    double total      = 0;
    double total_data = 0;
    std::string proto_str("AB");
    proto_str += proto == PROTO::AB2 ? "2" : "";

    std::vector<Result> batches_results(batch_threads);
    auto batch = [&](long wid, size_t start, size_t end) -> Code {
        for (size_t cur = start; cur < end; ++cur) {
            Result result;
            if ((PROTO::AB2 == proto && party_ == emp::ALICE)
                || (PROTO::AB == proto
                    && (cur + party_ - 1) % 2 == 0)) { // for AB alternate parties
                result = Server::perform_elem(meta, ios_ + wid * threads_per_thread, bn_, A[cur],
                                              B[cur], C[cur], threads_per_thread, proto);
            } else {
                result = Client::perform_elem(meta, ios_ + wid * threads_per_thread, bn_, A[cur],
                                              B[cur], C[cur], threads_per_thread, proto);
            }

            if (result.ret != Code::OK)
                return result.ret;

            Utils::add_result(batches_results[wid], result);
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(batch_threads);

    auto start = measure::now();
    auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
    total += Utils::to_sec(Utils::time_diff(start));
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), " ", CodeMessage(code));

    std::string unit;
    total_data = Utils::to_MB(total_data, unit);
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [" << unit << "]: " << total_data << "\n";

    reset_counter_();
}

} // namespace HE_OT

#endif