
#ifndef KEYS_HPP_
#define KEYS_HPP_

#include "gemini/core/prg_party.h"
#include "core/conv_packed.hpp"

#ifndef TRIPLE_OT_GROUP
#define TRIPLE_OT_GROUP 0 // channels (and ferret threads) per OT pack, 0: from the first demand
#endif
#include <mutex>
#include <stdexcept>

#include "core/utils.hpp"
#include "io/send.hpp"
#include "ot/cheetah-ot_pack.h"

namespace Iface {

// Total OT demand (COTs per direction) of the whole preprocessing, set by the caller before the first
// generation: the OT packs are sized once, at the first request, which is not always the largest one
// (PPA4's Beaver tuples, the multiplexer and COT triples come after the boolean triples).
inline uint64_t& ot_demand_hint() {
    static uint64_t h = 0;
    return h;
}

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
    const PackedConv2D& get_packed_conv() const { return _packed_conv; }
    Channel** get_ios(unsigned threads) {
        connect(_party, _ip, _port, threads, _io_offset);
        return _ios;
    }
    const sci::OTPack<Channel>* get_otpack(int idx) const { return _ot_packs[idx]; }
    // The OT packs: one per TRIPLE_OT_GROUP channels, whose ferret instances extend on that many
    // threads and channels. An OT consumer runs one worker per pack, on the pack's first channel.
    int ot_workers() const { return int(_ot_packs.size()); }
    // per OT consumer call: the seed stream of worker wid's PRGs (emp::DetSeedScope) for reproducible runs
    uint64_t next_ot_call() { return _ot_calls++; }
    static uint64_t ot_seed_tag(uint64_t call, int wid) { return gemini::party_seed64(0x07c0 + call, uint64_t(wid)); }

    // The OT packs, made at the first request. Every ferret instance extends ~10^7 COTs at once
    // (the first time in its setup), so one pack per channel made 2 * threads extensions, several
    // times what a small network consumes: take as many packs as `cots` (per direction, spread
    // evenly) need for one extension each, and give each pack's ferret the other threads.
    void ensure_ot(uint64_t cots) {
        if (!_ot_packs.empty())
            return;
        auto start = measure::now();
        cots = std::max(cots, ot_demand_hint());
        unsigned group = TRIPLE_OT_GROUP;
        if (const char* e = std::getenv("CHEETAH_OT_GROUP"))  // experiments: channels per pack
            group = unsigned(std::max(1, std::atoi(e)));
        if (group == 0) {
            const double per_ext = 0.8 * double(cheetah::ferret_param().n); // headroom for COT / MUX
            group = _threads;
            while (group > 1 && double(cots) > per_ext * double(_threads / group)) group /= 2;
        }
        _ot_group = std::max(1u, std::min(group, _threads));
        _ot_packs.resize(_threads / _ot_group);
        auto init_ot = [&](int, size_t start, size_t end) -> Code {
            for (size_t i = start; i < end; ++i) {
                int cur_party = i & 1 ? (3 - _party) : _party;
                emp::DetSeedScope det(gemini::party_seed64(0x07ac5, i), gemini::kSeeded);  // reproducible OT (PRG_SEED)
                _ot_packs[i]  = new sci::OTPack<Channel>(_ios + i * _ot_group, int(_ot_group), cur_party, true, false);
            }
            return Code::OK;
        };
        gemini::ThreadPool tpool(_ot_packs.size());
        gemini::LaunchWorks(tpool, _ot_packs.size(), init_ot);
        Utils::log(Utils::Level::INFO, "P", _party - 1, ", PID", _io_offset, ": OT packs   s PRE: ",
                   Utils::to_sec(Utils::time_diff(start)), " (packs: ", _ot_packs.size(), ", ferret threads: ", _ot_group,
                   ", unseeded PRGs: ", emp::det_seed_misses().load(), ")");
    }
    Channel* ot_io(int idx) const { return _ios[size_t(idx) * _ot_group]; }
    // n channels of their own, after the regular ones (port + (threads + k) * io_offset), made at the first call: for a
    // generator that runs alongside the OT packs on the regular channels (hpmpc CHEETAH_CONV_EARLY)
    Channel** get_side_ios(unsigned n) {
        std::lock_guard<std::mutex> lock(_side_mutex);
        if (!_side_ios) {
            const char* addr = _party == emp::ALICE ? nullptr : _ip.c_str();
            _side_ios = Utils::init_ios<Channel>(addr, _port + _threads * _io_offset, n, _io_offset);
            _side_n = n;
        } else if (n > _side_n)
            throw std::runtime_error("Keys::get_side_ios: more channels than made at the first call");
        return _side_ios;
    }
    unsigned get_io_offset() const { return _io_offset; }

    void disconnect();

  private:
    int _party = 0;
    std::string _ip;
    unsigned _port      = 0;
    unsigned _io_offset = 0;
    gemini::HomFCSS _fc;
    gemini::HomConv2DSS _hom_conv;
    PackedConv2D _packed_conv;
    gemini::HomBNSS _bn;
    Channel** _ios;
    unsigned _threads;
    unsigned _ot_group = 1;
    uint64_t _ot_calls = 0;
    std::vector<sci::OTPack<Channel>*> _ot_packs;
    bool _connected = false;
    Channel** _side_ios = nullptr;
    unsigned _side_n    = 0;
    std::mutex _side_mutex;

    Keys(int party, const std::string& ip, unsigned port, unsigned threads, unsigned io_offset)
        : _threads(threads) {
        gemini::prg_party() = party;
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
        _packed_conv.setUp(ctx, skey, o_pkey);
        if (const char* e = std::getenv("CONV_REPACK"))
            conv_repack() = std::atoi(e) != 0;
        if (conv_repack())
            _packed_conv.setUpRepack(_ios, party, conv_repack_ab());
        _bn.setUp(PLAIN_MOD, ctx, skey, o_pkey);
        setupBn(_ios, ctx, party);

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
        for (unsigned i = 0; _side_ios && i < _side_n; ++i) delete _side_ios[i];
        delete[] _side_ios;
        for (auto* pack : _ot_packs) delete pack;
        for (unsigned i = 0; i < _threads; ++i) delete _ios[i];
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
    seal::prng_seed_type seed = {gemini::party_seed64()};
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
    if (!_ot_packs.empty()) {
        int64_t ext = 0, cots = 0;
        double sec = 0;
        for (auto* pack : _ot_packs)
            for (auto* ot : {pack->silent_ot, pack->silent_ot_reversed}) {
                ext += ot->extensions(), sec += ot->rcot_seconds(), cots += ot->get_rcot_count();
            }
        Utils::log(Utils::Level::INFO, "P", _party - 1, ", PID", _io_offset, ": OT packs: ", _ot_packs.size(),
                   " x ", _ot_group, " threads, ferret extensions after setup: ", ext, ", time in rcot (sum over packs): ",
                   sec, " s, in the LPN step (incl. setup): ", cheetah::lpn_ns().load() * 1e-9, " s");
    }
    if (gemini::kSeeded && emp::det_seed_misses().load())
        Utils::log(Utils::Level::INFO, "P", _party - 1, ", PID", _io_offset, ": PRGs seeded outside a DetSeedScope: ",
                   emp::det_seed_misses().load(), " (the OT outputs may differ between runs)");
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
