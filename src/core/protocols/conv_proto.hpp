#ifndef CONV_PROTO_HPP_
#define CONV_PROTO_HPP_

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "core/utils.hpp"

using gemini::Tensor;
using Utils::Result;

namespace Server {

template <class Channel>
Result send(const gemini::HomConv2DSS::Meta& meta, Channel** server,
            const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1, const size_t& threads,
            bool flush = true);

template <class Channel>
Result recv(const gemini::HomConv2DSS::Meta& meta, Channel** server,
            const gemini::HomConv2DSS& conv, Tensor<uint64_t>& C1, const size_t& threads);

template <class Channel>
Result Protocol2(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                 const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1,
                 const size_t& threads = 1);

template <class Channel>
Result Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                 const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                 const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                 const size_t& threads = 1);

template <class Channel>
Result Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                 const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                 const std::vector<std::vector<seal::Plaintext>>& enc_B1, Tensor<uint64_t>& C1,
                 const size_t& threads = 1, bool fil_ntt = true);

template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                     const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                     const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                     const size_t& threads = 1, Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                     const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                     const std::vector<std::vector<seal::Plaintext>>& enc_B1,
                     const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                     const size_t& threads = 1, Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                     const gemini::HomConv2DSS& conv, const size_t& threads = 1,
                     Utils::PROTO proto = Utils::PROTO::AB);

/**
 * AB2P: flipped-role AB2 with a PRESCRIBED output share (image owner side, emp::BOB).
 * Receives the encrypted filters (cached across batches in enc_filters_cache when
 * fresh_filters == false), evaluates conv(pt_image, ct_filters) - prescribed and sends the
 * result back; this party's output share IS `prescribed` (not written here).
 */
template <class Channel>
Result Protocol2Prescribed(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                           const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                           const Tensor<uint64_t>& prescribed,
                           std::vector<std::vector<seal::Ciphertext>>& enc_filters_cache,
                           const size_t& threads, bool fresh_filters);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const gemini::HomConv2DSS::Meta& meta,
                 const gemini::HomConv2DSS& conv, const Tensor<T>& A1,
                 const std::vector<Tensor<T>>& B1, const Tensor<T>& C1, Utils::PROTO proto);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result recv(Channel** client, const gemini::HomConv2DSS& conv,
            const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
            const vector<Tensor<uint64_t>>& B2, vector<seal::Plaintext>& enc_A2,
            vector<vector<seal::Plaintext>>& enc_B2, vector<seal::Ciphertext>& enc_A1,
            const size_t& threads);

template <class Channel>
Result send(Channel** client, const gemini::HomConv2DSS& conv, const vector<seal::Ciphertext>& M2,
            const size_t& threads, bool flush = true);

/**
 * AB Protocol
 */
template <class Channel>
Result Protocol1(Channel** client, const gemini::HomConv2DSS& conv,
                 const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1);

/**
 * AB Protocol but weights are encoded with NTT
 */
template <class Channel>
Result Protocol1(Channel** client, const gemini::HomConv2DSS& conv,
                 const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<std::vector<seal::Plaintext>>& enc_B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1, bool fil_ntt = true);

/**
 * AB2 Protocol
 */
template <class Channel>
Result Protocol2(Channel** client, const gemini::HomConv2DSS& conv,
                 const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1);

/**
 * AB2 Protocol but weights are encoded with NTT
 */
template <class Channel>
Result Protocol2(Channel** client, const gemini::HomConv2DSS& conv,
                 const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<std::vector<seal::Plaintext>>& enc_B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1, bool fil_ntt = true);

/**
 * Performs Conv for a given Protocol and verifies the results
 */
template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                     const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                     const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                     const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

/**
 * Performs Conv for a given Protocol and verifies the results but weights are already encoded with
 * NTT
 */
template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                     const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                     const std::vector<Tensor<uint64_t>>& B1,
                     const std::vector<std::vector<seal::Plaintext>>& enc_B1, Tensor<uint64_t>& C1,
                     const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

/**
 * Performs Conv for a given Protocol and verifies the results
 * NOTE: No input (uses predefined values), for testing only
 */
template <class Channel>
Result perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                     const gemini::HomConv2DSS& conv, const size_t& threads,
                     Utils::PROTO proto = Utils::PROTO::AB);

/**
 * AB2P: flipped-role AB2 with a PRESCRIBED output share (filter owner side, emp::ALICE).
 * Encrypts and sends the filters (only when fresh_filters), then receives and decrypts the
 * evaluated result: C2 = conv - peer's prescribed share.
 */
template <class Channel>
Result Protocol2Prescribed(Channel** client, const gemini::HomConv2DSS& conv,
                           const gemini::HomConv2DSS::Meta& meta,
                           const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                           const size_t& threads, bool fresh_filters);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const std::vector<Tensor<T>>& B1,
                 const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::recv(Channel** client, const gemini::HomConv2DSS& conv,
                    const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                    const vector<Tensor<uint64_t>>& B2, vector<seal::Plaintext>& enc_A2,
                    vector<vector<seal::Plaintext>>& enc_B2, vector<seal::Ciphertext>& enc_A1,
                    const size_t& threads) {
    Result measures;
    auto start = measure::now();

    measures.ret = conv.encodeImage(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.ret = conv.encodeFilters(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    // conv.filtersToNtt(enc_B2, threads);

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::recv_encrypted_vector(client, conv.getContext(), enc_A1, threads);

    measures.send_recv += Utils::time_diff(start);
    return measures;
}

template <class Channel>
Result Client::send(Channel** client, const gemini::HomConv2DSS& conv,
                    const vector<seal::Ciphertext>& M2, const size_t& threads, bool flush) {
    Result measures;
    auto start = measure::now();

    IO::send_encrypted_vector(client, M2, threads, flush);

    measures.send_recv += Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol2(Channel** client, const gemini::HomConv2DSS& conv,
                         const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const std::vector<std::vector<seal::Plaintext>>& enc_B2,
                         Tensor<uint64_t>& C2, const size_t& threads, bool fil_ntt) {
    Result measures;

    vector<seal::Plaintext> enc_A2;
    vector<seal::Ciphertext> enc_A1;

    auto start = measure::now();

    if (meta.is_shared_input)
        measures.ret = conv.encodeImage(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::recv_encrypted_vector(client, conv.getContext(), enc_A1, threads);

    measures.send_recv += Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret
        = conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, C2, threads, true, !fil_ntt, true);
    enc_A1.clear();
    enc_A2.clear();
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    Result tmp;
    if ((tmp = send(client, conv, M2, threads)).ret != Code::OK) {
        return tmp;
    }

    measures.serial += tmp.send_recv;
    measures.bytes = tmp.bytes;

    return measures;
}

template <class Channel>
Result Client::Protocol2(Channel** client, const gemini::HomConv2DSS& conv,
                         const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                         const size_t& threads) {
    Result measures;

    vector<seal::Plaintext> enc_A2;
    vector<vector<seal::Plaintext>> enc_B2;
    vector<seal::Ciphertext> enc_A1;

    if ((measures = recv(client, conv, meta, A2, B2, enc_A2, enc_B2, enc_A1, threads)).ret
        != Code::OK) {
        return measures;
    }

    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret = conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, C2, threads);
    enc_A1.clear();
    enc_A2.clear();
    enc_B2.clear();
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    Result tmp;
    if ((tmp = send(client, conv, M2, threads)).ret != Code::OK) {
        return tmp;
    }

    measures.send_recv += tmp.send_recv;
    measures.bytes = tmp.bytes;

    return measures;
}

template <class Channel>
Result Client::Protocol2Prescribed(Channel** client, const gemini::HomConv2DSS& conv,
                                   const gemini::HomConv2DSS::Meta& meta,
                                   const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                                   const size_t& threads, bool fresh_filters) {
    Result measures;

    if (fresh_filters) {
        ////////////////////////////////////////////////////////////////////////////
        // Enc(B2) + send
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();

        std::vector<std::vector<seal::Serializable<seal::Ciphertext>>> enc_B2;
        measures.ret = conv.encryptFilters(B2, meta, enc_B2, threads);
        if (measures.ret != Code::OK)
            return measures;

        measures.encryption = Utils::time_diff(start);

        start = measure::now();
        IO::send_encrypted_filters(*(client[0]), enc_B2);
        client[0]->flush();
        measures.send_recv = Utils::time_diff(start);
    }

    ////////////////////////////////////////////////////////////////////////////
    // recv M2' = conv(A1, B2) - prescribed, decrypt
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Ciphertext> M2;
    IO::recv_encrypted_vector(client, conv.getContext(), M2, threads);

    measures.serial += Utils::time_diff(start);
    start = measure::now();

    measures.ret        = conv.decryptToTensor(M2, meta, C2, threads);
    measures.decryption = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol2Prescribed(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                                   const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                                   const Tensor<uint64_t>& prescribed,
                                   std::vector<std::vector<seal::Ciphertext>>& enc_filters_cache,
                                   const size_t& threads, bool fresh_filters) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // encode A1 (plaintext image)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> enc_A1;
    measures.ret = conv.encodeImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv Enc(B2) (cached across batches while the weights stay the same)
    ////////////////////////////////////////////////////////////////////////////
    if (fresh_filters) {
        start = measure::now();
        enc_filters_cache.clear();
        IO::recv_encrypted_filters(*(server[0]), conv.getContext(), enc_filters_cache,
                                   /*is_truncated*/ false);
        measures.send_recv += Utils::time_diff(start);
    }

    ////////////////////////////////////////////////////////////////////////////
    // M2' = conv(A1, Enc(B2)) - prescribed
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret = conv.conv2DSSPrescribed(enc_filters_cache, enc_A1, meta, prescribed, M2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    IO::send_encrypted_vector(server, M2, threads);
    measures.serial += Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel** client, const gemini::HomConv2DSS& conv,
                         const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                         const size_t& threads) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = conv.encodeFilters(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    auto encoding_time = Utils::time_diff(start);

    measures = Protocol1(client, conv, meta, A2, enc_B2, C2, threads, false);
    measures.encryption += encoding_time;

    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel** client, const gemini::HomConv2DSS& conv,
                         const gemini::HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const std::vector<std::vector<seal::Plaintext>>& enc_B2,
                         Tensor<uint64_t>& C2, const size_t& threads, bool fil_ntt) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A2;
    measures.ret = conv.encryptImage(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);
    start               = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = IO::recv_send(conv.getContext(), client, enc_A2, enc_A1, threads);
    enc_A2.clear();
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2;
    measures.ret = conv.conv2DSS(enc_A1, encoded_A2, enc_B2, meta, enc_M2, R2, threads, true,
                                 !fil_ntt, true);
    enc_A1.clear();
    encoded_A2.clear();
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // send M2' + recv M1'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    measures.ret = IO::recv_send(conv.getContext(), client, enc_M2, enc_M1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // dec(M1') + R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    conv.decryptToTensor(enc_M1, meta, C2, threads);

    measures.decryption = Utils::time_diff(start);

    start = measure::now();

    Utils::op_inplace<uint64_t>(
        C2, R2, [](uint64_t a, uint64_t b) -> uint64_t { return Utils::add(a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    measures.ret = Code::OK;

    return measures;
}

template <class Channel>
Result Server::send(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                    const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                    const size_t& threads, bool flush) {
    Result measures;
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1, threads, flush);
    enc_A1.clear();

    measures.send_recv = Utils::time_diff(start);
    return measures;
}

template <class Channel>
Result Server::recv(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                    const gemini::HomConv2DSS& conv, Tensor<uint64_t>& C1, const size_t& threads) {
    Result measures;
    auto start = measure::now();

    std::vector<seal::Ciphertext> enc_C1;
    IO::recv_encrypted_vector(server, conv.getContext(), enc_C1, threads);

    measures.send_recv += Utils::time_diff(start);
    start = measure::now();

    measures.ret        = conv.decryptToTensor(enc_C1, meta, C1, threads);
    measures.decryption = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol2(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                         const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                         Tensor<uint64_t>& C1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    if ((measures = send(meta, server, conv, A1, threads)).ret != Code::OK) {
        return measures;
    }

    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    Result tmp;
    if ((tmp = recv(meta, server, conv, C1, threads)).ret != Code::OK) {
        return tmp;
    }

    measures.serial += tmp.send_recv;
    measures.decryption = tmp.decryption;
    measures.bytes      = tmp.bytes;
    return measures;
}

template <class Channel>
Result Server::Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                         const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                         const std::vector<std::vector<seal::Plaintext>>& enc_B1,
                         Tensor<uint64_t>& C1, const size_t& threads, bool fil_ntt) {
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A2;
    IO::send_recv(conv.getContext(), server, enc_A1, enc_A2, threads);
    enc_A1.clear();

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1;
    measures.ret
        = conv.conv2DSS(enc_A2, encoded_A1, enc_B1, meta, M1, R1, threads, true, !fil_ntt, true);
    enc_A2.clear();
    encoded_A1.clear();
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    IO::send_recv(conv.getContext(), server, M1, enc_M2, threads);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    measures.ret = conv.decryptToTensor(enc_M2, meta, C1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = Utils::time_diff(start);
    start               = measure::now();

    Utils::op_inplace<uint64_t>(C1, R1, [](uint64_t a, uint64_t b) { return Utils::add(a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    measures.ret = Code::OK;
    return measures;
}

template <class Channel>
Result Server::Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                         const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                         const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                         const size_t& threads) {
    Result measures;
    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<std::vector<seal::Plaintext>> enc_B1;
    measures.ret = conv.encodeFilters(B1, meta, enc_B1, threads);
    if (measures.ret != Code::OK)
        return measures;

    auto encoding_time = Utils::time_diff(start);

    measures = Protocol1(meta, server, conv, A1, enc_B1, C1, threads, false);
    measures.encryption += encoding_time;

    return measures;
}

template <class Channel>
Result Server::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                             const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                             const std::vector<std::vector<seal::Plaintext>>& enc_B1,
                             const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                             const size_t& threads, Utils::PROTO proto) {
    Result measures;

    switch (proto) {
    case Utils::PROTO::AB:
        measures = Server::Protocol1(meta, server, conv, A1, enc_B1, C1, threads);
        break;
    case Utils::PROTO::AB2:
        measures = Server::Protocol2(meta, server, conv, A1, C1, threads);
        break;
    }

#ifdef VERIFY
    Verify_Conv(*(server[0]), meta, conv, A1, B1, C1, proto);
#endif
    return measures;
}

template <class Channel>
Result Server::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                             const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                             const std::vector<Tensor<uint64_t>>& B1, Tensor<uint64_t>& C1,
                             const size_t& threads, Utils::PROTO proto) {
    Result measures;

    switch (proto) {
    case Utils::PROTO::AB:
        measures = Server::Protocol1(meta, server, conv, A1, B1, C1, threads);
        break;
    case Utils::PROTO::AB2:
        measures = Server::Protocol2(meta, server, conv, A1, C1, threads);
        break;
    }

#ifdef VERIFY
    Verify_Conv(*(server[0]), meta, conv, A1, B1, C1, proto);
#endif
    return measures;
}

template <class Channel>
Result Server::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                             const gemini::HomConv2DSS& conv, const size_t& threads,
                             Utils::PROTO proto) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> C1;
    return perform_proto(meta, server, conv, A1, B1, C1, threads, proto);
}

template <class Channel>
Result Client::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                             const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A2,
                             const std::vector<Tensor<uint64_t>>& B2, Tensor<uint64_t>& C2,
                             const size_t& threads, Utils::PROTO proto) {
    Result measures;

    switch (proto) {
    case Utils::PROTO::AB:
        measures = Client::Protocol1(client, conv, meta, A2, B2, C2, threads);
        break;
    case Utils::PROTO::AB2:
        measures = Client::Protocol2(client, conv, meta, A2, B2, C2, threads);
        break;
    }

#ifdef VERIFY
    Verify_Conv(*(client[0]), A2, B2, C2);
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                             const gemini::HomConv2DSS& conv, const Tensor<uint64_t>& A2,
                             const std::vector<Tensor<uint64_t>>& B2,
                             const std::vector<std::vector<seal::Plaintext>>& enc_B2,
                             Tensor<uint64_t>& C2, const size_t& threads, Utils::PROTO proto) {
    Result measures;

    switch (proto) {
    case Utils::PROTO::AB:
        measures = Client::Protocol1(client, conv, meta, A2, enc_B2, C2, threads);
        break;
    case Utils::PROTO::AB2:
        measures = Client::Protocol2(client, conv, meta, A2, enc_B2, C2, threads);
        break;
    }

#ifdef VERIFY
    Verify_Conv(*(client[0]), A2, B2, C2);
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(const gemini::HomConv2DSS::Meta& meta, Channel** client,
                             const gemini::HomConv2DSS& conv, const size_t& threads,
                             Utils::PROTO proto) {
    auto A2 = Utils::init_image(meta, 5);
    auto B2 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> C2;

    return perform_proto(meta, client, conv, A2, B2, C2, threads, proto);
}

#ifdef VERIFY
template <class T>
void Server::Verify_Conv(IO::NetIO& io, const gemini::HomConv2DSS::Meta& meta,
                         const gemini::HomConv2DSS& conv, const Tensor<T>& A1,
                         const std::vector<Tensor<T>>& B1, const Tensor<T>& C1,
                         Utils::PROTO proto) {
    Utils::log(Utils::Level::DEBUG, "VERIFYING CONV with ",
               proto == Utils::PROTO::AB ? "AB" : "AB2");
    Utils::log(Utils::Level::DEBUG, A1.shape(), " x ", meta.fshape, " = ", C1.shape());

    Tensor<T> A2(meta.ishape);
    std::vector<Tensor<T>> B2(meta.n_filters, Tensor<T>(meta.fshape));
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    for (auto& filter : B2) io.recv_data(filter.data(), filter.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // A1 + A2

    if (proto == Utils::PROTO::AB)
        for (size_t i = 0; i < B1.size(); ++i) // B1 + B2
            Utils::op_inplace<T>(B2[i], B1[i],
                                 [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; });

    Tensor<T> test;                              // (A1 + A2) (B1 + B2)
    conv.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();
    for (long c = 0; c < C2.channels(); ++c)
        for (long h = 0; h < C2.height(); ++h)
            for (long w = 0; w < C2.width(); ++w)
                if (!same || test(c, h, w) != C2(c, h, w)) {
                    Utils::log(Utils::Level::FAILED, test(c, h, w), ", ", C2(c, h, w));
                    same = false;
                    goto end;
                }
end:
    if (same)
        Utils::log(Utils::Level::PASSED, "CONV: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "CONV: FAILED");
}

template <class T>
void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const std::vector<Tensor<T>>& B2,
                         const Tensor<T>& C2) {
    log(Utils::Level::DEBUG, "SENDING");
    io.send_data(A2.data(), A2.NumElements() * sizeof(T), false);
    for (auto& filter : B2) io.send_data(filter.data(), filter.NumElements() * sizeof(T), false);
    io.send_data(C2.data(), C2.NumElements() * sizeof(T), false);
    io.flush();
}
#endif

#endif
