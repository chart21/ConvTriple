#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include <seal/ciphertext.h>
#include <seal/encryptor.h>
#include <seal/keygenerator.h>
#include <seal/serializable.h>
#include <string>

#include <io/net_io_channel.hpp>

#include <gemini/cheetah/hom_bn_ss.h>
#include <type_traits>

#include "core/keys.hpp"
#include "ot/bit-triple-generator.h"
#include "protocols/ot_proto.hpp"
#include "utils.hpp"

namespace Iface {

using UINT_TYPE = std::conditional_t<BIT_LEN == 32, uint32_t, uint64_t>;

template <class Channel, class SerKey>
void exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
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

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[], int bitlength,
                                uint64_t num_triples, const std::string& ip, int port, int party,
                                int threads = 1, TripleGenMethod method = _16KKOT_to_4OT,
                                unsigned io_offset = 1);

void generateBool3TupleCheetah(Beaver3Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party);

void generateBool4TupleCheetah(Beaver4Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party);

void generateArithTriplesCheetah(const UINT_TYPE a[], const UINT_TYPE b[], UINT_TYPE c[],
                                 int bitlength, uint64_t num_triples, const std::string& ip,
                                 int port, int party, int threads = 1,
                                 Utils::PROTO proto = Utils::PROTO::AB, unsigned io_offset = 1);

void generateFCTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, uint64_t com_dim, uint64_t dim2, int party,
                              int threads, Utils::PROTO proto, int factor = 1);

void generateConvTriplesCheetahWrapper(Keys<IO::NetIO>& keys, const UINT_TYPE* a,
                                       const UINT_TYPE* b, UINT_TYPE* c, Utils::ConvParm parm,
                                       int party, int threads, Utils::PROTO proto, int factor = 1,
                                       bool is_shared_input = false);

void generateConvTriplesCheetah(Keys<IO::NetIO>& keys, size_t total_batches,
                                std::vector<Utils::ConvParm>& parms, UINT_TYPE** a, UINT_TYPE** b,
                                UINT_TYPE* c, Utils::PROTO proto, int party, int threads,
                                int factor, bool is_shared_input = false);

void generateConvTriplesCheetah2(Keys<IO::NetIO>& keys, size_t total_batches,
                                 std::vector<Utils::ConvParm>& parms, UINT_TYPE** a, UINT_TYPE** b,
                                 UINT_TYPE* c, Utils::PROTO proto, int party, int threads,
                                 int factor, bool is_shared_input = false);

void generateConvTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                                UINT_TYPE* c, const gemini::HomConv2DSS::Meta& meta, int batch,
                                int party, int threads, Utils::PROTO proto, int factor);

void generateBNTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, size_t num_ele, size_t h, size_t w,
                              int party, int threads, Utils::PROTO proto, int factor = 1);

void do_multiplex(int num_input, UINT_TYPE* x32, uint8_t* sel_packed, UINT_TYPE* y32, int party,
                  const std::string& ip, int port, int io_offset, int threads);

void generateOT(int party, const std::string& ip, int port, int threads, int io_offset);

void generateCOT(int party, UINT_TYPE* a, uint8_t* b, UINT_TYPE* c, const unsigned& num_triples,
                 const std::string& ip, int port, int threads, int io_offset);

void tmp(int party, int threads);

template <class T, bool LSB = true>
inline uint8_t get_nth(const T* a, const size_t& idx) {
    constexpr size_t bits = sizeof(T) * 8;
    size_t block          = idx / bits;
    size_t bit            = idx % bits;

    if constexpr (LSB) {
        return (a[block] >> bit) & 1;
    } else {
        return (a[block] >> (bits - (bit + 1))) & 1;
    }
}

template <class Channel, class Serial>
void send_vec(Channel** ios, std::vector<std::vector<Serial>>& vec, int threads) {
    uint64_t n = vec.size();
    ios[0]->send_data(&n, sizeof(n));

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto* io = ios[wid];
        std::vector<uint64_t> idxs(end - start + 1);

        std::stringstream stream;
        for (size_t i = start; i < end; ++i) {
            idxs[i - start] = vec[i].size();
            for (auto& ele : vec[i]) {
                ele.save(stream);
            }
        }

        auto data   = stream.str();
        idxs.back() = data.size();
        io->send_data(idxs.data(), idxs.size() * sizeof(decltype(idxs)::value_type));
        if (idxs.back() == 0) {
            io->flush();
            return Code::OK;
        }
        io->send_data(data.c_str(), idxs.back());

        io->flush();

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, vec.size(), func);
}

template <class Channel, class Serial>
void recv_vec(Channel** ios, const seal::SEALContext& ctx, std::vector<std::vector<Serial>>& vec,
              int threads) {
    uint64_t n = 0;
    ios[0]->recv_data(&n, sizeof(n));
    vec.resize(n);

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;
        auto* io = ios[wid];

        std::vector<uint64_t> idxs(end - start + 1);
        io->recv_data(idxs.data(), idxs.size() * sizeof(decltype(idxs)::value_type));

        if (idxs.back() == 0)
            return Code::OK;

        char* data = new char[idxs.back()];
        io->recv_data((void*)data, idxs.back());

        std::stringstream stream(std::string(data, idxs.back()));
        for (size_t i = 0; i < idxs.size() - 1; ++i) {
            vec[start + i].resize(idxs[i]);

            for (auto& ct : vec[start + i]) {
                ct.load(ctx, stream);
            }
        }

        delete[] data;
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, vec.size(), func);
}

} // namespace Iface

#endif
