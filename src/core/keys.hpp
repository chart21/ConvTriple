
#ifndef KEYS_HPP_
#define KEYS_HPP_

#include "core/utils.hpp"
#include "io/send.hpp"
#include "ot/cheetah-ot_pack.h"

namespace Iface {

// not thread safe
template <class Channel>
class Keys {
  public:
    static Keys& instance(int party, const std::string& ip, unsigned port, unsigned threads,
                          unsigned io_offset) {
        static Keys k(party, ip, port, threads, io_offset);
        k._party     = party;
        k._ip        = ip;
        k._port      = port;
        k._io_offset = io_offset;
        // k.connect(party, ip, port, threads, io_offset);
        return k;
    }

    const gemini::HomFCSS& get_fc() const { return _fc; }
    const gemini::HomBNSS& get_bn() const { return _bn; }
    const gemini::HomConv2DSS& get_conv() const { return _hom_conv; }
    Channel** get_ios(unsigned threads) {
        connect(_party, _ip, _port, threads, _io_offset);
        return _ios;
    }
    const sci::OTPack<Channel>* get_otpack(int idx) const { return _ot_packs[idx]; }
    unsigned get_io_offset() const { return _io_offset; }

    void disconnect();

  private:
    int _party = 0;
    std::string _ip;
    unsigned _port      = 0;
    unsigned _io_offset = 0;
    gemini::HomFCSS _fc;
    gemini::HomConv2DSS _hom_conv;
    gemini::HomBNSS _bn;
    Channel** _ios;
    unsigned _threads;
    std::vector<sci::OTPack<Channel>*> _ot_packs;
    bool _connected = false;

    Keys(int party, const std::string& ip, unsigned port, unsigned threads, unsigned io_offset)
        : _threads(threads) {
        auto start       = measure::now();
        const char* addr = ip.c_str();
        if (party == emp::ALICE)
            addr = nullptr;
        _ios = Utils::init_ios<Channel>(addr, port, threads, io_offset);

        seal::SEALContext ctx = Utils::init_he_context();

        seal::KeyGenerator keygen(ctx);
        seal::SecretKey skey = keygen.secret_key();
        auto pkey            = std::make_shared<seal::PublicKey>();
        auto o_pkey          = std::make_shared<seal::PublicKey>();
        keygen.create_public_key(*pkey);
        exchange_keys(_ios, *pkey, *o_pkey, ctx, party);

        _fc.setUp(ctx, skey, o_pkey);
        _hom_conv.setUp(ctx, skey, o_pkey);
        _bn.setUp(PLAIN_MOD, ctx, skey, o_pkey);
        setupBn(_ios, ctx, party);

        _ot_packs.resize(threads);
        auto init_ot = [&](int wid, size_t start, size_t end) -> Code {
            for (size_t i = start; i < end; ++i) {
                int cur_party = wid & 1 ? (3 - party) : party;
                _ot_packs[i]  = new sci::OTPack<Channel>(_ios + wid, 1, cur_party, true, false);
            }
            return Code::OK;
        };

        gemini::ThreadPool tpool(threads);
        gemini::LaunchWorks(tpool, threads, init_ot);
        _connected = true;

        auto time = Utils::to_sec(Utils::time_diff(start));
        Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Key exchange   s PRE: ", time, " (threads: ", threads, ")");
        std::string unit;
        double data_sent = 0, data_recv = 0;
        for (size_t i = 0; i < threads; ++i) {
            data_sent += Utils::to_MB(_ios[i]->counter, unit);
            data_recv += Utils::to_MB(_ios[i]->recv_counter, unit);
            _ios[i]->counter = 0;
            _ios[i]->recv_counter = 0;
        }
        Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Key exchange   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    }

    ~Keys() noexcept {
        for (unsigned i = 0; i < _threads; ++i) {
            delete _ot_packs[i];
            delete _ios[i];
        }
        delete[] _ios;
    }

    template <class SerKey>
    void exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
                       const seal::SEALContext& ctx, int party);

    void setupBn(Channel** ios, const seal::SEALContext& ctx, const int& party);

    void connect(int party, const std::string& ip, int port, int threads, int io_offset);

  public:
    Keys(Keys& copy)             = delete;
    Keys(Keys&& copy)            = delete;
    Keys& operator=(Keys& copy)  = delete;
    Keys& operator=(Keys&& copy) = delete;
};

template <class Channel>
template <class SerKey>
void Keys<Channel>::exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
                                  const seal::SEALContext& ctx, int party) {
    switch (party) {
    case emp::ALICE:
        IO::send_pkey(*(ios[0]), pkey);
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        IO::send_pkey(*(ios[0]), pkey);
        break;
    }
}

template <class Channel>
void Keys<Channel>::setupBn(Channel** ios, const seal::SEALContext& ctx, const int& party) {
    using namespace seal;
    KeyGenerator keygen(ctx);

    size_t ntarget_bits = BIT_LEN;
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

#if PRG_SEED != -1
    seal::prng_seed_type seed = {PRG_SEED};
    seal_parms.set_random_generator(std::make_shared<seal::Blake2xbPRNGFactory>(seed));
#endif

    std::vector<std::shared_ptr<seal::SEALContext>> bn_contexts_(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        seal_parms.set_plain_modulus(plain_crts[i]);
        bn_contexts_[i] = std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);
    }

    std::vector<seal::SEALContext> contexts;
    std::vector<std::optional<SecretKey>> opt_sks;

    std::vector<std::shared_ptr<seal::PublicKey>> bn_pks_(nCRT); // public keys (BN)
    std::vector<std::shared_ptr<seal::SecretKey>> bn_sks_(nCRT); // secret keys (BN)

    for (size_t i = 0; i < nCRT; ++i) {
        KeyGenerator keygen(*bn_contexts_[i]);
        bn_sks_[i]                   = std::make_shared<SecretKey>(keygen.secret_key());
        bn_pks_[i]                   = std::make_shared<PublicKey>();
        Serializable<PublicKey> s_pk = keygen.create_public_key();

        exchange_keys(ios, s_pk, *bn_pks_[i], *bn_contexts_[i], party);
        contexts.emplace_back(*bn_contexts_[i]);
        opt_sks.emplace_back(*bn_sks_[i]);
    }

    auto code = _bn.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", party - 1, ": ", CodeMessage(code));
}

template <class Channel>
void Keys<Channel>::connect(int party, const std::string& ip, int port, int threads,
                            int io_offset) {
    if (_connected)
        return;
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    auto build = [&](int wid, size_t start, size_t end) -> Code {
        for (size_t i = start; i < end; ++i) {
            _ios[i]->init_connection(addr, port + i * io_offset);
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, threads, build);
    _connected = true;
}

template <class Channel>
void Keys<Channel>::disconnect() {
    if (!_connected)
        return;
    for (unsigned i = 0; i < _threads; ++i) {
        _ios[i]->disconnect();
        _ios[i]->counter = 0;
    }
    _connected = false;
}
} // namespace Iface

#endif
