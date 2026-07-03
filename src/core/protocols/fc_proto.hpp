#ifndef FC_PROTO_HPP_
#define FC_PROTO_HPP_

#include <gemini/cheetah/hom_fc_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "core/utils.hpp"

using gemini::HomFCSS;
using gemini::Tensor;
using std::vector;
using Utils::Result;

namespace Server {

template <class Channel>
Result Protocol2(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                 const vector<Tensor<uint64_t>>& A1, vector<Tensor<uint64_t>>& C1,
                 const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result Protocol1(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                 const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                 vector<Tensor<uint64_t>>& C1, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result perform_proto(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                     const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                     vector<Tensor<uint64_t>>& C1, const size_t& threads = 1,
                     const size_t& batch = 1, Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_proto(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                     const size_t& threads = 1, const size_t& batch = 1,
                     Utils::PROTO proto = Utils::PROTO::AB2);

/**
 * AB2P: flipped-role AB2 with a PRESCRIBED output share (vector owner side, emp::BOB).
 * Receives the encrypted weight matrix (re-received every fresh_period batch elements),
 * evaluates W*x - prescribed[i] per element and sends the results back; this party's output
 * shares ARE `prescribed` (not written here).
 */
template <class Channel>
Result Protocol2Prescribed(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                           const vector<Tensor<uint64_t>>& A1,
                           const vector<Tensor<uint64_t>>& prescribed, const size_t& threads,
                           const size_t& batch, const size_t& fresh_period);

#ifdef VERIFY
template <class T>
void Verify_FC(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& fc, const Tensor<T>& A1,
               const Tensor<T>& B1, const Tensor<T>& C1, Utils::PROTO proto);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1(Channel** client, const HomFCSS& fc, const HomFCSS::Meta& meta,
                 const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                 vector<Tensor<uint64_t>>& C2, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result Protocol2(Channel** client, const HomFCSS& fc, const HomFCSS::Meta& meta,
                 const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                 vector<Tensor<uint64_t>>& C2, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result perform_proto(const HomFCSS::Meta& meta, Channel** client, const HomFCSS& fc,
                     const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                     vector<Tensor<uint64_t>>& C1, const size_t& threads, const size_t& batch = 1,
                     Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_proto(const HomFCSS::Meta& meta, Channel** client, const HomFCSS& fc,
                     const size_t& threads, const size_t& batch = 1,
                     Utils::PROTO proto = Utils::PROTO::AB2);

/**
 * AB2P: flipped-role AB2 with a PRESCRIBED output share (weight owner side, emp::ALICE).
 * Encrypts and sends the weight matrix (every fresh_period batch elements), then receives and
 * decrypts the evaluated results: C2[i] = W*x_i - peer's prescribed share.
 */
template <class Channel>
Result Protocol2Prescribed(Channel** client, const HomFCSS& fc, const HomFCSS::Meta& meta,
                           const vector<Tensor<uint64_t>>& B2, vector<Tensor<uint64_t>>& C2,
                           const size_t& threads, const size_t& batch, const size_t& fresh_period);

#ifdef VERIFY
template <class T>
void Verify_FC(IO::NetIO& io, const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol2(Channel** client, const HomFCSS& hom_fc, const HomFCSS::Meta& meta,
                         const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                         vector<Tensor<uint64_t>>& C2, const size_t& threads, const size_t& batch) {
    size_t in_para            = 1;
    size_t threads_per_thread = threads / in_para;

    auto prog = [&](long wid, size_t first, size_t end) {
        if (first >= end)
            return Code::OK;

        size_t ele = end - first;
        Result measures;

        auto start = measure::now();

        vector<vector<seal::Plaintext>> enc_A2(ele);
        for (size_t i = 0; i < ele; ++i) {
            measures.ret
                = hom_fc.encodeInputVector(A2[first + i], meta, enc_A2[i], threads_per_thread);
        }
        if (measures.ret != Code::OK)
            return measures.ret;

        vector<vector<vector<seal::Plaintext>>> enc_B2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret
                = hom_fc.encodeWeightMatrix(B2[first + i], meta, enc_B2[i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(start);

        ////////////////////////////////////////////////////////////////////////////
        // recv A' + deserialize
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_A1(ele);
        for (size_t i = 0; i < ele; ++i)
            IO::recv_encrypted_vector(client + wid, hom_fc.getContext(), enc_A1[i], threads);

        measures.send_recv += Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // M2' = (A1' + A2) ⊙ B2 - R
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> M2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = hom_fc.MatVecMul(enc_A1[i], enc_A2[i], enc_B2[i], meta, M2[i],
                                            C2[first + i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.cipher_op = Utils::time_diff(start);

        ////////////////////////////////////////////////////////////////////////////
        // serialize + send M2'
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        for (size_t i = 0; i < ele; ++i) IO::send_encrypted_vector(client + wid, M2[i], threads);

        measures.send_recv += Utils::time_diff(start);
        return Code::OK;
    };

    gemini::ThreadPool tpool(in_para);
    gemini::LaunchWorks(tpool, batch, prog);

    Result final;
    for (size_t i = 0; i < threads; ++i) final.bytes += client[i]->counter;
    return final;
}

template <class Channel>
Result Client::Protocol2Prescribed(Channel** client, const HomFCSS& fc, const HomFCSS::Meta& meta,
                                   const vector<Tensor<uint64_t>>& B2, vector<Tensor<uint64_t>>& C2,
                                   const size_t& threads, const size_t& batch,
                                   const size_t& fresh_period) {
    Result measures;

    for (size_t i = 0; i < batch; ++i) {
        if (i % fresh_period == 0) {
            ////////////////////////////////////////////////////////////////////////////
            // Enc(B2) + send
            ////////////////////////////////////////////////////////////////////////////
            auto start = measure::now();

            vector<vector<seal::Serializable<seal::Ciphertext>>> enc_B2;
            if (Code::OK != (measures.ret = fc.encryptWeightMatrix(B2[i], meta, enc_B2, threads)))
                return measures;

            measures.encryption += Utils::time_diff(start);
            start = measure::now();

            IO::send_encrypted_filters(*(client[0]), enc_B2);
            client[0]->flush();

            measures.send_recv += Utils::time_diff(start);
        }

        ////////////////////////////////////////////////////////////////////////////
        // recv M2' = W*x - prescribed, decrypt
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();

        vector<seal::Ciphertext> M2;
        IO::recv_encrypted_vector(client, fc.getContext(), M2, threads);

        measures.serial += Utils::time_diff(start);
        start = measure::now();

        if (Code::OK != (measures.ret = fc.decryptToVector(M2, meta, C2[i], threads)))
            return measures;

        measures.decryption += Utils::time_diff(start);
    }

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol2Prescribed(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                                   const vector<Tensor<uint64_t>>& A1,
                                   const vector<Tensor<uint64_t>>& prescribed,
                                   const size_t& threads, const size_t& batch,
                                   const size_t& fresh_period) {
    Result measures;
    vector<vector<seal::Ciphertext>> enc_B2;  // encrypted weight matrix, cached across the batch

    for (size_t i = 0; i < batch; ++i) {
        ////////////////////////////////////////////////////////////////////////////
        // encode A1[i] (plaintext vector)
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();

        vector<seal::Plaintext> enc_A1;
        if (Code::OK != (measures.ret = fc.encodeInputVector(A1[i], meta, enc_A1, threads)))
            return measures;

        measures.encryption += Utils::time_diff(start);

        if (i % fresh_period == 0) {
            ////////////////////////////////////////////////////////////////////////////
            // recv Enc(B2)
            ////////////////////////////////////////////////////////////////////////////
            start = measure::now();
            enc_B2.clear();
            IO::recv_encrypted_filters(*(server[0]), fc.getContext(), enc_B2,
                                       /*is_truncated*/ false);
            measures.send_recv += Utils::time_diff(start);
        }

        ////////////////////////////////////////////////////////////////////////////
        // M2' = W*x - prescribed[i]
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<seal::Ciphertext> M2;
        if (Code::OK
            != (measures.ret
                = fc.MatVecMulPrescribed(enc_B2, enc_A1, meta, prescribed[i], M2, threads)))
            return measures;

        measures.cipher_op += Utils::time_diff(start);

        ////////////////////////////////////////////////////////////////////////////
        // serialize + send M2'
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();
        IO::send_encrypted_vector(server, M2, threads);
        measures.serial += Utils::time_diff(start);
    }

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel** client, const HomFCSS& fc, const HomFCSS::Meta& meta,
                         const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                         vector<Tensor<uint64_t>>& C2, const size_t& threads, const size_t& batch) {
    size_t in_para            = 1;
    size_t threads_per_thread = threads / in_para;
    gemini::ThreadPool tpool(in_para);
    Result fin_measures;
    auto prog = [&](long wid, size_t start, size_t end) {
        if (start >= end)
            return Code::OK;
        size_t ele = end - start;
        Result measures;

        ////////////////////////////////////////////////////////////////////////////
        // Receive A1' and enc/send A2'
        ////////////////////////////////////////////////////////////////////////////
        auto mess = measure::now();

        vector<vector<seal::Plaintext>> encoded_A2(ele);
        vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.encryptInputVector(A2[start + i], meta, enc_A2[i], encoded_A2[i],
                                                 threads_per_thread);

        if (measures.ret != Code::OK)
            return measures.ret;

        vector<vector<vector<seal::Plaintext>>> enc_B2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret
                = fc.encodeWeightMatrix(B2[start + i], meta, enc_B2[i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(mess);
        mess                = measure::now();

        vector<vector<seal::Ciphertext>> enc_A1;
        IO::recv_send(fc.getContext(), client + wid, enc_A2, enc_A1, threads_per_thread);

        measures.send_recv += Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // M2' = (A1' + A2) ⊙ B2 - R2
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        vector<vector<seal::Ciphertext>> enc_M2(ele);
        vector<Tensor<uint64_t>> R2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.MatVecMul(enc_A1[i], encoded_A2[i], enc_B2[i], meta, enc_M2[i], R2[i],
                                        threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.cipher_op = Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // send M2' + recv M1'
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        vector<vector<seal::Ciphertext>> enc_M1(ele);
        measures.ret
            = IO::recv_send(fc.getContext(), client + wid, enc_M2, enc_M1, threads_per_thread);

        measures.send_recv += Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // dec(M1') + R2
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        for (size_t i = 0; i < ele; ++i)
            fc.decryptToVector(enc_M1[i], meta, C2[start + i], threads_per_thread);

        measures.decryption = Utils::time_diff(mess);

        mess = measure::now();

        for (size_t i = 0; i < ele; ++i)
            Utils::op_inplace<uint64_t>(
                C2[start + i], R2[i],
                [](uint64_t a, uint64_t b) -> uint64_t { return Utils::add(a, b); });

        measures.plain_op = Utils::time_diff(mess);
        return Code::OK;
    };

    fin_measures.ret = gemini::LaunchWorks(tpool, batch, prog);
    for (size_t i = 0; i < threads; ++i) fin_measures.bytes += client[i]->counter;

    return fin_measures;
}

template <class Channel>
Result Server::Protocol2(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                         const vector<Tensor<uint64_t>>& A1, vector<Tensor<uint64_t>>& C1,
                         const size_t& threads, const size_t& batch) {
    size_t in_para            = 1;
    size_t threads_per_thread = threads / in_para;

    auto prog = [&](long wid, size_t first, size_t end) {
        if (first >= end)
            return Code::OK;
        size_t ele = end - first;
        Result measures;

        ////////////////////////////////////////////////////////////////////////////
        // send Enc(A1)
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();

        vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A1(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret
                = fc.encryptInputVector(A1[first + i], meta, enc_A1[i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(start);

        start = measure::now();

        for (size_t i = 0; i < ele; ++i)
            IO::send_encrypted_vector(server + wid, enc_A1[i], threads);

        measures.send_recv = Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // recv C1 = dec(M2)
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_C1(ele);
        for (size_t i = 0; i < ele; ++i)
            IO::recv_encrypted_vector(server + wid, fc.getContext(), enc_C1[i], threads);

        measures.send_recv += Utils::time_diff(start);
        start = measure::now();

        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.decryptToVector(enc_C1[i], meta, C1[first + i], threads_per_thread);
        measures.decryption = Utils::time_diff(start);
        return Code::OK;
    };

    gemini::ThreadPool tpool(in_para);
    gemini::LaunchWorks(tpool, batch, prog);

    Result final;
    for (size_t i = 0; i < threads; ++i) final.bytes += server[i]->counter;
    return final;
}

template <class Channel>
Result Server::Protocol1(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                         const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                         vector<Tensor<uint64_t>>& C1, const size_t& threads, const size_t& batch) {
    size_t in_para            = 1;
    size_t threads_per_thread = threads / in_para;
    gemini::ThreadPool tpool(in_para);
    auto prog = [&](long wid, size_t first, size_t end) {
        if (first >= end)
            return Code::OK;

        Result measures;
        measures.send_recv = 0;
        size_t ele         = end - first;

        ////////////////////////////////////////////////////////////////////////////
        // Enc(A1), enc(B1), send(A1), recv(A2)
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();
        vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A1(ele);
        vector<vector<seal::Plaintext>> encoded_A1(ele);

        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.encryptInputVector(A1[first + i], meta, enc_A1[i], encoded_A1[i],
                                                 threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        vector<vector<vector<seal::Plaintext>>> enc_B1(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret
                = fc.encodeWeightMatrix(B1[first + i], meta, enc_B1[i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(start);

        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_A2(ele);
        IO::send_recv(fc.getContext(), server + wid, enc_A1, enc_A2, threads_per_thread);

        measures.send_recv = Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // M1 = (A1 + A2') ⊙ B1 - R1
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> M1(ele);
        vector<Tensor<uint64_t>> R1(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.MatVecMul(enc_A2[i], encoded_A1[i], enc_B1[i], meta, M1[i], R1[i],
                                        threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.cipher_op = Utils::time_diff(start);

        ////////////////////////////////////////////////////////////////////////////
        // Send(M1), Recv(M2), Dec(M2)
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_M2(ele);
        IO::send_recv(fc.getContext(), server + wid, M1, enc_M2, threads_per_thread);

        measures.send_recv += Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // Dec(M2) + R1
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        for (size_t i = 0; i < ele; ++i)
            measures.ret = fc.decryptToVector(enc_M2[i], meta, C1[first + i], threads_per_thread);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.decryption = Utils::time_diff(start);
        start               = measure::now();

        for (size_t i = 0; i < ele; ++i)
            Utils::op_inplace<uint64_t>(C1[first + i], R1[i],
                                        [](uint64_t a, uint64_t b) { return Utils::add(a, b); });

        measures.plain_op = Utils::time_diff(start);
        return Code::OK;
    };

    Result final;
    final.ret = gemini::LaunchWorks(tpool, batch, prog);

    for (size_t i = 0; i < threads; ++i) final.bytes += server[i]->counter;
    return final;
}

template <class Channel>
Result Server::perform_proto(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                             const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                             vector<Tensor<uint64_t>>& C1, const size_t& threads,
                             const size_t& batch, Utils::PROTO proto) {

    assert(A1.size() == batch && B1.size() == batch && C1.size() == batch);

    Result measures;
    switch (proto) {
    case Utils::PROTO::AB:
        measures = Server::Protocol1(meta, server, fc, A1, B1, C1, threads, batch);
        break;
    case Utils::PROTO::AB2:
        measures = Server::Protocol2(meta, server, fc, A1, C1, threads, batch);
        break;
    }

#ifdef VERIFY
    for (size_t i = 0; i < batch; ++i) {
        Verify_FC(*(server[0]), meta, fc, A1[i], B1[i], C1[i], proto);
    }
#endif
    return measures;
}

template <class Channel>
Result Server::perform_proto(const HomFCSS::Meta& meta, Channel** server, const HomFCSS& fc,
                             const size_t& threads, const size_t& batch, Utils::PROTO proto) {
    vector<Tensor<uint64_t>> vecs(batch, Tensor<uint64_t>(meta.input_shape));
    for (auto& vec : vecs)
        for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    vector<Tensor<uint64_t>> weights(batch, Tensor<uint64_t>(meta.weight_shape));
    for (auto& weight : weights)
        for (long i = 0; i < weight.rows(); i++)
            for (long j = 0; j < weight.cols(); j++) weight(i, j) = 2;

    vector<Tensor<uint64_t>> C1(batch);

    return perform_proto(meta, server, fc, vecs, weights, C1, threads, batch, proto);
}

template <class Channel>
Result Client::perform_proto(const HomFCSS::Meta& meta, Channel** client, const HomFCSS& fc,
                             const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                             vector<Tensor<uint64_t>>& C2, const size_t& threads,
                             const size_t& batch, Utils::PROTO proto) {
    Result measures;
    switch (proto) {
    case Utils::PROTO::AB:
        measures = Client::Protocol1(client, fc, meta, A2, B2, C2, threads, batch);
        break;
    case Utils::PROTO::AB2:
        measures = Client::Protocol2(client, fc, meta, A2, B2, C2, threads, batch);
        break;
    }

#ifdef VERIFY
    for (size_t i = 0; i < batch; ++i) {
        Verify_FC(*(client[0]), A2[i], B2[i], C2[i]);
    }
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(const HomFCSS::Meta& meta, Channel** client, const HomFCSS& fc,
                             const size_t& threads, const size_t& batch, Utils::PROTO proto) {
    vector<Tensor<uint64_t>> vecs(batch, Tensor<uint64_t>(meta.input_shape));
    for (auto& vec : vecs)
        for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    vector<Tensor<uint64_t>> weights(batch, Tensor<uint64_t>(meta.weight_shape));
    for (auto& weight : weights)
        for (long i = 0; i < weight.rows(); i++)
            for (long j = 0; j < weight.cols(); j++) weight(i, j) = 2;

    std::vector<Tensor<uint64_t>> C2(batch);

    auto measures = perform_proto(meta, client, fc, vecs, weights, C2, threads, batch);

    return measures;
}

#ifdef VERIFY
template <class T>
void Server::Verify_FC(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& fc,
                       const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1,
                       Utils::PROTO proto) {
    Utils::log(Utils::Level::DEBUG, "VERIFYING FC with ", proto == Utils::PROTO::AB ? "AB" : "AB2");
    Utils::log(Utils::Level::DEBUG, meta.weight_shape, " x ", meta.input_shape, " = ", C1.shape());

    Tensor<T> A2(A1.shape());
    Tensor<T> B2(meta.weight_shape);
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    io.recv_data(B2.data(), B2.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&fc](T a, T b) -> T { return Utils::add(a, b); }); // C
    Utils::op_inplace<T>(A2, A1, [&fc](T a, T b) -> T { return Utils::add(a, b); }); // A1 + A2

    if (proto == Utils::PROTO::AB)
        Utils::op_inplace<T>(B2, B1, [&fc](T a, T b) -> T { return Utils::add(a, b); });

    Tensor<T> test;                            // (A1 + A2) (B1 + B2)
    fc.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();

    for (long i = 0; i < C2.length(); ++i) {
        if (!same || test(i) != C2(i)) {
            Utils::log(Utils::Level::FAILED, i, ", ", test(i), ", ", C2(i));
            same = false;
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "FC: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "FC: FAILED");
}

template <class T>
void Client::Verify_FC(IO::NetIO& io, const Tensor<T>& A2, const Tensor<T>& B2,
                       const Tensor<T>& C2) {
    log(Utils::Level::DEBUG, "SENDING");
    io.send_data(A2.data(), A2.NumElements() * sizeof(T), false);
    io.send_data(B2.data(), B2.NumElements() * sizeof(T), false);
    io.send_data(C2.data(), C2.NumElements() * sizeof(T), false);
    io.flush();
}
#endif

#endif