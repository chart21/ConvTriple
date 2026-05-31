#include "hpmpc_interface.hpp"

#include <algorithm>
#include <seal/ciphertext.h>
#include <seal/serializable.h>
#include <sstream>
#include <thread>

#include "emp-tool/utils/constants.h"
#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/multiplexer.hpp"
#include "protocols/ot_proto.hpp"

#include "ot/bit-triple-generator.h"
#include "ot/cheetah-ot_pack.h"

#include "queue.hpp"
#include "send.hpp"

#if USE_CONV_CUDA
#include "troy/conv2d_gpu.cuh"
#endif

#include "elem.hpp"

constexpr uint64_t MAX_BOOL  = 1ULL << 20;
constexpr uint64_t MAX_ARITH = 20'000'000;

#define OTHER_PARTY(party) (3 - party)

namespace Iface {

class PROF : public seal::MMProf {
    std::unique_ptr<seal::MemoryPoolHandle> handle;
    std::shared_ptr<seal::util::MemoryPoolMT> pool;

  public:
    PROF() {
        pool   = std::make_shared<seal::util::MemoryPoolMT>(true);
        handle = std::make_unique<seal::MemoryPoolHandle>(pool);
    }

    ~PROF() noexcept {
        handle.reset();
        if (pool.unique()) {
            std::cout << "UNIQUE\n";
        } else {
            std::cout << "NOT UNIQUE: " << pool.use_count() << "\n";
        }
    }

    seal::MemoryPoolHandle get_pool(uint64_t) { return *handle; }
};

void generateBoolCOTMultTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[],
                                int bitlength [[maybe_unused]], uint64_t num_triples,
                                const std::string& ip, int port, int party, int threads,
                                unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_triples (BOOL COT MULT): ", num_triples);
    require_tuple_count_multiple_of_8(num_triples);
    uint64_t num_bytes = (num_triples + 7) / 8;
    // std::atomic<int> setup = 0;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        // auto start_setup = measure::now();

        // sci::OTPack<IO::NetIO> pack(ios + wid, 1, cur_party, true, false);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], keys.get_otpack(wid), false);

        // setup += Utils::time_diff(start_setup);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / threads));
            switch (cur_party) {
            case emp::ALICE:
                cot_multiply_shares(emp::ALICE, triple_gen.otpack, a + total, b + total, c + total, current);
                break;
            case emp::BOB:
                cot_multiply_shares(emp::BOB, triple_gen.otpack, a + total, b + total, c + total, current);
                break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool COT Mult triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool COT Mult triple data[", unit, "]: ", data);

    // Utils::log(Utils::Level::INFO, "P", party - 1, ": Setup time [s]: ",
    //            Utils::to_sec(setup.load())
    //                / (num_triples > static_cast<size_t>(threads) ? threads : num_triples));

    // keys.disconnect();
}

void generateRandomMultiplicationsCheetah(uint8_t a[], uint8_t b[], uint64_t num_muls,
                                          const std::string& ip, int port, int party,
                                          int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_muls (BOOL MUL): ", num_muls);
    require_tuple_count_multiple_of_8(num_muls);
    uint64_t num_bytes = (num_muls + 7) / 8;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();
    auto** ios = keys.get_ios(threads);


    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        auto* otpack = keys.get_otpack(wid);

        uint64_t total_bytes = end - start;
        uint64_t current_muls = total_bytes * 8;

        std::memset(a + start, 0, total_bytes);
        std::memset(b + start, 0, total_bytes);

        switch (cur_party) {
        case emp::ALICE:
            Server::mul_gen(otpack, a + start, b + start, current_muls);
            break;
        case emp::BOB:
            Client::mul_gen(otpack, a + start, b + start, current_muls);
            break;
        default:
            Utils::log(Utils::Level::ERROR, "Unknown party: P", party - 1);
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool mul time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool mul data[", unit, "]: ", data);
}

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[],
                                int bitlength [[maybe_unused]], uint64_t num_triples,
                                const std::string& ip, int port, int party, int threads,
                                TripleGenMethod method, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_triples (BOOL): ", num_triples);
    require_tuple_count_multiple_of_8(num_triples);
    uint64_t num_bytes = (num_triples + 7) / 8;
    // std::atomic<int> setup = 0;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        // auto start_setup = measure::now();

        // sci::OTPack<IO::NetIO> pack(ios + wid, 1, cur_party, true, false);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], keys.get_otpack(wid), false);

        // setup += Utils::time_diff(start_setup);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / threads));
            switch (cur_party) {
            case emp::ALICE:
                Server::triple_gen(triple_gen, a + total, b + total, c + total, current, true,
                                   method);
                break;
            case emp::BOB:
                Client::triple_gen(triple_gen, a + total, b + total, c + total, current, true,
                                   method);
                break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool triple data[", unit, "]: ", data);

    // Utils::log(Utils::Level::INFO, "P", party - 1, ": Setup time [s]: ",
    //            Utils::to_sec(setup.load())
    //                / (num_triples > static_cast<size_t>(threads) ? threads : num_triples));

    // keys.disconnect();
}

void generateBool3TupleCheetah(Beaver3Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party, int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_tuples (BOOL3): ", num_tuples);
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], keys.get_otpack(wid), false);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / threads));
            Beaver3Tuples sub{
                tuples.a + total, tuples.b + total, tuples.c + total,
                tuples.ab + total, tuples.ac + total, tuples.bc + total,
                tuples.abc + total
            };
            switch (cur_party) {
                case emp::ALICE:
                    Server::tuple3_gen(triple_gen, sub, current * 8);
                    break;
                case emp::BOB:
                    Client::tuple3_gen(triple_gen, sub, current * 8);
                    break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_tuples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool3 tuple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool3 tuple data[", unit, "]: ", data);
}

void generateBool4TupleCheetah(Beaver4Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party, int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_tuples (BOOL4): ", num_tuples);
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], keys.get_otpack(wid), false);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / threads));
            Beaver4Tuples sub{
                tuples.a + total, tuples.b + total, tuples.c + total, tuples.d + total,
                tuples.ab + total, tuples.ac + total, tuples.ad + total,
                tuples.bc + total, tuples.bd + total, tuples.cd + total,
                tuples.abc + total, tuples.abd + total, tuples.acd + total, tuples.bcd + total,
                tuples.abcd + total
            };
            switch (cur_party) {
                case emp::ALICE:
                    Server::tuple4_gen(triple_gen, sub, current * 8);
                    break;
                case emp::BOB:
                    Client::tuple4_gen(triple_gen, sub, current * 8);
                    break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_tuples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool4 tuple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool4 tuple data[", unit, "]: ", data);
}

void generateArithTriplesCheetah(const UINT_TYPE a[], const UINT_TYPE b[], UINT_TYPE c[],
                                 int bitlength, uint64_t num_triples, const std::string& ip,
                                 int port, int party, int threads, Utils::PROTO proto,
                                 unsigned io_offset) {
    assert(bitlength == 32 && "[arith. triples] Unsupported bitlength");
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_triples (ARITH): ", num_triples,
               " " + Utils::proto_str(proto));
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto& bn   = keys.get_bn();
    auto** ios = keys.get_ios(threads);

    auto pool = seal::MemoryPoolHandle::New();
    auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

    Tensor<uint64_t> A({static_cast<long>(num_triples)});
    Tensor<uint64_t> B({static_cast<long>(num_triples)});

    for (uint64_t i = 0; i < num_triples; ++i) {
        if (a)
            A(i) = static_cast<uint64_t>(a[i]);
        if (b)
            B(i) = static_cast<uint64_t>(b[i]);
    }

    gemini::HomBNSS::Meta meta;
    meta.is_shared_input = proto == Utils::PROTO::AB;
    meta.target_base_mod = PLAIN_MOD;

    auto func = [&](size_t wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;
        for (int total = start; total < end;) {
            size_t current = std::min(static_cast<int>(MAX_ARITH / threads), end - total);

            gemini::HomBNSS::Meta m = meta;
            m.vec_shape             = gemini::TensorShape({static_cast<long>(current)});

            Tensor<uint64_t> tmp_A = Tensor<uint64_t>::Wrap(A.data() + total, m.vec_shape);
            Tensor<uint64_t> tmp_B = Tensor<uint64_t>::Wrap(B.data() + total, m.vec_shape);
            Tensor<uint64_t> tmp_C(m.vec_shape);

            Result res;
            switch (party) {
            case emp::ALICE: {
                res = Server::perform_elem(ios + wid, bn, m, tmp_A, tmp_B, tmp_C, 1, proto);
                break;
            }
            case emp::BOB: {
                res = Client::perform_elem(ios + wid, bn, m, tmp_A, tmp_B, tmp_C, 1, proto);
                break;
            }
            default: {
                Utils::log(Utils::Level::ERROR, "Unknown party: P", party - 1);
            }
            }

            for (uint64_t i = 0; i < current; ++i) c[i + total] = static_cast<UINT_TYPE>(tmp_C(i));
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Arith triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Arith triple data[", unit, "]: ", data);

    keys.disconnect();
}

void generateFCTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, uint64_t com_dim, uint64_t dim2, int party,
                              int threads, Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_fc(com_dim, dim2);
    Utils::log(Utils::Level::INFO, "P", party - 1, " FC: ", meta.input_shape, " x ",
               meta.weight_shape, " ", Utils::proto_str(proto));

    auto start = measure::now();

    // meta.is_shared_input = proto == Utils::PROTO::AB;
    auto& fc   = keys.get_fc();
    auto** ios = keys.get_ios(threads);

    uint64_t* ai = new uint64_t[meta.input_shape.num_elements() * batch];
    for (uint i = 0; i < meta.input_shape.num_elements() * batch; ++i)
        ai[i] = a != nullptr ? a[i] : 0;
    std::vector<Tensor<uint64_t>> A(batch);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = Tensor<uint64_t>::Wrap(ai + meta.input_shape.num_elements() * i, meta.input_shape);

    uint64_t* bi = new uint64_t[meta.weight_shape.num_elements() * factor];
    for (uint i = 0; i < meta.weight_shape.num_elements() * factor; ++i)
        bi[i] = b == nullptr ? 0 : b[i];

    size_t tmp = batch / factor;
    std::vector<Tensor<uint64_t>> B(batch);
    for (int i = 0; i < factor; ++i)
        for (size_t j = 0; j < tmp; ++j)
            B[i * tmp + j] = Tensor<uint64_t>::Wrap(
                bi + meta.weight_shape.num_elements() * (i % factor), meta.weight_shape);

    std::vector<Tensor<uint64_t>> C(batch);

    switch (party) {
    case emp::ALICE: {
        Client::perform_proto(meta, ios, fc, A, B, C, threads, batch, proto);
        break;
    }
    case emp::BOB: {
        Server::perform_proto(meta, ios, fc, A, B, C, threads, batch, proto);
        break;
    }
    }

    for (size_t i = 0; i < C.size(); ++i)
        for (size_t j = 0; j < dim2; ++j) {
            c[i * dim2 + j] = C[i](j);
        }

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": FC triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) {
        data += Utils::to_MB(ios[i]->counter, unit);
        ios[i]->counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ": FC triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;
}

void generateConvTriplesCheetahWrapper(Keys<IO::NetIO>& keys, const UINT_TYPE* a,
                                       const UINT_TYPE* b, UINT_TYPE* c, Utils::ConvParm parm,
                                       int party, int threads, Utils::PROTO proto, int factor,
                                       bool is_shared_input) {
#if USE_CONV_CUDA
    if (proto == Utils::PROTO::AB2) {
        TROY::conv2d(keys.get_ios(threads), OTHER_PARTY(party), a, b, c, parm.batchsize, parm.ic,
                     parm.ih, parm.iw, parm.fh, parm.fw, parm.n_filters, parm.stride, parm.padding,
                     true, factor);
        return;
    }
#endif
    auto meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                      parm.n_filters, parm.stride, parm.padding, is_shared_input);

    Utils::log(Utils::Level::INFO, "P", party - 1, " CONV: ", meta.ishape, " x ", meta.fshape,
               " x ", parm.n_filters, ", ", parm.stride, ", ", parm.padding, ", ",
               Utils::proto_str(proto));

    if (Utils::getOutDim(parm) == gemini::GetConv2DOutShape(meta)) {
        generateConvTriplesCheetah(keys, a, b, c, meta, parm.batchsize, party, threads, proto,
                                   factor);
    } else {
        Utils::log(Utils::Level::INFO, "Adding padding manually");

        std::vector<UINT_TYPE> ai;
        std::tuple<int, int> dim;

        dim = Utils::pad_zero(a, ai, parm.ic, parm.ih, parm.iw, parm.padding, parm.batchsize);

        parm.ih      = std::get<0>(dim);
        parm.iw      = std::get<1>(dim);
        parm.padding = 0;

        meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                     parm.n_filters, parm.stride, parm.padding);
        generateConvTriplesCheetah(keys, ai.data(), b, c, meta, parm.batchsize, party, threads,
                                   proto, factor);
    }
}

void generateConvTriplesCheetah(Keys<IO::NetIO>& keys, size_t total_batches,
                                std::vector<Utils::ConvParm>& parms, UINT_TYPE** a, UINT_TYPE** b,
                                UINT_TYPE* c, Utils::PROTO proto, int party, int threads,
                                int factor, bool is_shared_input) {
    auto start = measure::now();

    vector<vector<seal::Plaintext>> enc_a(total_batches);
    vector<vector<vector<seal::Plaintext>>> enc_b(parms.size());
    vector<vector<seal::Ciphertext>> enc_a2(total_batches);
    vector<vector<seal::Serializable<seal::Ciphertext>>> enc_a1(total_batches);

    auto& hom_conv = keys.get_conv();
    auto** ios     = keys.get_ios(threads);

    auto pool = seal::MemoryPoolHandle::New();
    auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

    size_t offset = 0;

    Result result;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta
            = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                    parm.n_filters, parm.stride, parm.padding, is_shared_input);

        uint64_t* ai = new uint64_t[meta.ishape.num_elements() * parm.batchsize];
        if (party == emp::BOB || meta.is_shared_input)
            for (long i = 0; i < meta.ishape.num_elements() * parm.batchsize; ++i) ai[i] = a[n][i];

        uint64_t* bi = new uint64_t[meta.fshape.num_elements() * meta.n_filters * factor];
        if (b)
            for (size_t i = 0; i < meta.fshape.num_elements() * meta.n_filters * factor; ++i)
                bi[i] = b[n][i];

        int ac_batch_size = parm.batchsize / factor;
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            Tensor<uint64_t> A
                = Tensor<uint64_t>::Wrap(ai + meta.ishape.num_elements() * cur_batch, meta.ishape);

            std::vector<Tensor<uint64_t>> B(meta.n_filters);
            for (size_t i = 0; i < meta.n_filters; ++i)
                B[i] = Tensor<uint64_t>::Wrap(
                    bi + meta.fshape.num_elements() * meta.n_filters * (cur_batch / ac_batch_size)
                        + meta.fshape.num_elements() * i,
                    meta.fshape);

            switch (party) {
            case emp::ALICE: {
                if (meta.is_shared_input)
                    hom_conv.encodeImage(A, meta, enc_a[cur_batch + offset], threads);
                if (cur_batch == 0) {
                    hom_conv.encodeFilters(B, meta, enc_b[n], threads);
                    hom_conv.filtersToNtt(enc_b[n], threads);
                }
                break;
            }
            case emp::BOB: {
                hom_conv.encryptImage(A, meta, enc_a1[cur_batch + offset], threads);
                break;
            }
            }
        }
        delete[] ai;
        delete[] bi;
        offset += parm.batchsize;
    }

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": CONV NTT preprocessing time[s]:", Utils::to_sec(Utils::time_diff(start)));

    auto tmp = measure::now();

    // switch (party) {
    // case emp::ALICE: {
    //     recv_vec(ios, hom_conv.getContext(), enc_a2, threads);
    //     break;
    // }
    // case emp::BOB: {
    //     send_vec(ios, enc_a1, threads);
    //     break;
    // }
    // }
    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        for (int batch = 0; batch < parms[n].batchsize; ++batch) {
            switch (party) {
            case emp::BOB: {
                IO::send_encrypted_vector(ios, enc_a1[batch + offset], threads, true);
                break;
            }
            case emp::ALICE: {
                IO::recv_encrypted_vector(ios, hom_conv.getContext(), enc_a2[batch + offset],
                                          threads);
                break;
            }
            }
        }
        offset += parms[n].batchsize;
    }
    // if (party == emp::BOB)
    //     for (int i = 0; i < threads; ++i) ios[i]->flush();

    Utils::log(Utils::Level::DEBUG, "P", party - 1,
               ": send/recv[s]:", Utils::to_sec(Utils::time_diff(tmp)));

    tmp = measure::now();

    vector<vector<seal::Ciphertext>> M(total_batches);
    vector<Tensor<uint64_t>> C(total_batches);
    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta
            = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                    parm.n_filters, parm.stride, parm.padding, is_shared_input);
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::ALICE: {
                result.ret = hom_conv.conv2DSS(
                    enc_a2[cur_batch + offset], enc_a[cur_batch + offset], enc_b[n], meta,
                    M[cur_batch + offset], C[cur_batch + offset], threads, true, false, true);
                break;
            }
            }
        }
        enc_a[n].clear();
        enc_b[n].clear();
        enc_a2[n].clear();
        offset += parm.batchsize;
    }
    enc_a.clear();
    enc_b.clear();
    enc_a2.clear();

    Utils::log(Utils::Level::DEBUG, "P", party - 1,
               ": computation[s]:", Utils::to_sec(Utils::time_diff(tmp)));

    tmp = measure::now();

    // switch (party) {
    // case emp::ALICE: {
    //     send_vec(ios, M, threads);
    //     break;
    // }
    // case emp::BOB: {
    //     recv_vec(ios, hom_conv.getContext(), M, threads);
    //     break;
    // }
    // }

    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        for (int cur_batch = 0; cur_batch < parms[n].batchsize; ++cur_batch) {
            switch (party) {
            case emp::ALICE: { // send
                IO::send_encrypted_vector(ios, M[cur_batch + offset], threads, true);
                break;
            }
            case emp::BOB: { // recv
                IO::recv_encrypted_vector(ios, hom_conv.getContext(), M[cur_batch + offset],
                                          threads);
                break;
            }
            }
        }
        offset += parms[n].batchsize;
    }
    // if (party == emp::ALICE)
    //     for (int i = 0; i < threads; ++i) ios[i]->flush();

    Utils::log(Utils::Level::DEBUG, "P", party - 1,
               ": recv/send[s]:", Utils::to_sec(Utils::time_diff(tmp)));

    tmp = measure::now();

    offset          = 0;
    size_t c_offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta
            = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                    parm.n_filters, parm.stride, parm.padding, is_shared_input);

        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::BOB: {
                result.ret = hom_conv.decryptToTensor(M[cur_batch + offset], meta,
                                                      C[cur_batch + offset], threads);
                break;
            }
            }

            for (long i = 0; i < C[cur_batch + offset].NumElements(); ++i)
                c[c_offset + i] = C[cur_batch + offset].data()[i];
            c_offset += C[cur_batch + offset].NumElements();
        }
        offset += parm.batchsize;
    }

    Utils::log(Utils::Level::DEBUG, "P", party - 1,
               ": decryption[s]:", Utils::to_sec(Utils::time_diff(tmp)));

    auto time = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple time + NTT[s]: ", time);
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) {
        data += Utils::to_MB(ios[i]->counter, unit);
        ios[i]->counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple data[", unit, "]: ", data);
}

void generateConvTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                                UINT_TYPE* c, const gemini::HomConv2DSS::Meta& meta, int batch,
                                int party, int threads, Utils::PROTO proto, int factor) {
    auto start = measure::now();
    auto& conv = keys.get_conv();
    auto** ios = keys.get_ios(threads);

    double time_ntt  = 0;
    double send_recv = 0;
    double compute   = 0;
    double recv_send = 0;
    double decode    = 0;

    std::vector<std::vector<seal::Plaintext>> enc_B;

    uint64_t* ai = new uint64_t[meta.ishape.num_elements() * batch];
    for (long i = 0; i < meta.ishape.num_elements() * batch; ++i) ai[i] = a != nullptr ? a[i] : 0;

    uint64_t* bi = new uint64_t[meta.fshape.num_elements() * meta.n_filters * factor];
    if (b)
        for (size_t i = 0; i < meta.fshape.num_elements() * meta.n_filters * factor; ++i)
            bi[i] = b[i];

    int ac_batch_size = batch / factor;

    for (int cur_batch = 0; cur_batch < batch; ++cur_batch) {
        auto start_ntt = measure::now();
        Tensor<uint64_t> A
            = Tensor<uint64_t>::Wrap(ai + meta.ishape.num_elements() * cur_batch, meta.ishape);

        std::vector<Tensor<uint64_t>> B(meta.n_filters);
        for (size_t i = 0; i < meta.n_filters; ++i)
            B[i] = Tensor<uint64_t>::Wrap(
                bi + meta.fshape.num_elements() * meta.n_filters * (cur_batch / ac_batch_size)
                    + meta.fshape.num_elements() * i,
                meta.fshape);

        Tensor<uint64_t> C;

        Result result;
        switch (party) {
        case emp::ALICE: {
            Code c;
            if (cur_batch % ac_batch_size == 0) {
                enc_B.clear();
                if ((c = conv.encodeFilters(B, meta, enc_B, threads)) != Code::OK) {
                    Utils::log(Utils::Level::ERROR, "Filters encoding failed: ", CodeMessage(c));
                }
                if ((c = conv.filtersToNtt(enc_B, threads)) != Code::OK) {
                    Utils::log(Utils::Level::ERROR, "Filters to NTT failed: ", CodeMessage(c));
                }
            }
            time_ntt += Utils::to_sec(Utils::time_diff(start_ntt));
            result = Client::perform_proto(meta, ios, conv, A, B, enc_B, C, threads, proto);
            time_ntt += Utils::to_sec(result.encryption);
            break;
        }
        case emp::BOB: {
            if (proto == Utils::PROTO::AB) {
                Code c;
                if (cur_batch % ac_batch_size == 0) {
                    enc_B.clear();
                    if ((c = conv.encodeFilters(B, meta, enc_B, threads)) != Code::OK) {
                        Utils::log(Utils::Level::ERROR,
                                   "Filters encoding failed: ", CodeMessage(c));
                    }
                    if ((c = conv.filtersToNtt(enc_B, threads)) != Code::OK) {
                        Utils::log(Utils::Level::ERROR, "Filters to NTT failed: ", CodeMessage(c));
                    }
                }
                time_ntt += Utils::to_sec(Utils::time_diff(start_ntt));
            }
            result = Server::perform_proto(meta, ios, conv, A, enc_B, B, C, threads, proto);
            time_ntt += Utils::to_sec(result.encryption);
            break;
        }
        }

        send_recv += Utils::to_sec(result.send_recv);
        compute += Utils::to_sec(result.cipher_op);
        recv_send += Utils::to_sec(result.serial);
        decode += Utils::to_sec(result.decryption);
        decode += Utils::to_sec(result.plain_op);

        if (result.ret != Code::OK) {
            Utils::log(Utils::Level::ERROR, "CONV failed: ", CodeMessage(result.ret));
        }

        // Utils::print_results(result, 0, batch, threads);
        for (long i = 0; i < C.NumElements(); ++i) c[i + C.NumElements() * cur_batch] = C.data()[i];
    }

    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV NTT preprocessing time[s]: ", time_ntt);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": send/recv[s]: ", send_recv);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": compute[s]: ", compute);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": recv/send[s]: ", recv_send);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": decode[s]: ", decode);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": CONV triple time + NTT[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) {
        data += Utils::to_MB(ios[i]->counter, unit);
        ios[i]->counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;
}

void generateBNTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, size_t num_ele, size_t h, size_t w,
                              int party, int threads, Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_bn(num_ele, h, w);
    Utils::log(Utils::Level::INFO, "P", party - 1, " BN: ", meta.ishape, " x ", meta.vec_shape,
               ", ", Utils::proto_str(proto));

    auto start = measure::now();

    meta.is_shared_input = proto == Utils::PROTO::AB;
    auto& bn             = keys.get_bn();
    auto** ios           = keys.get_ios(threads);

    size_t ac_batch_size = batch / factor;
    for (int cur_batch = 0; cur_batch < batch; ++cur_batch) {
        Tensor<uint64_t> A(meta.ishape);
        for (long i = 0; i < A.channels(); i++)
            for (long j = 0; j < A.height(); j++)
                for (long k = 0; k < A.width(); k++)
                    A(i, j, k) = a != nullptr ? a[meta.ishape.num_elements() * cur_batch
                                                  + i * A.height() * A.width() + j * A.width() + k]
                                              : 0;

        Tensor<uint64_t> B(meta.vec_shape);
        for (long i = 0; i < B.NumElements(); i++)
            B(i) = b != nullptr ? b[i + B.NumElements() * (cur_batch / ac_batch_size)] : 0;

        Tensor<uint64_t> C;

        switch (party) {
        case emp::ALICE: {
            Client::perform_proto(meta, ios, bn, A, B, C, threads, proto);
            break;
        }
        case emp::BOB: {
            Server::perform_proto(meta, ios, bn, A, B, C, threads, proto);
            break;
        }
        }

        for (long i = 0; i < C.channels(); i++)
            for (long j = 0; j < C.height(); j++)
                for (long k = 0; k < C.width(); k++)
                    c[C.NumElements() * cur_batch + i * C.height() * C.width() + j * C.width() + k]
                        = C(i, j, k);
    }

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": BN triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) {
        data += Utils::to_MB(ios[i]->counter, unit);
        ios[i]->counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ": BN triple data[", unit, "]: ", data);
}

void tmp(int party, int threads) {
    // auto context = Utils::init_he_context();
    auto start = measure::now();
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(POLY_MOD, {60, 49}));
    parms.set_n_special_primes(0);
    // size_t prime_mod = seal::PlainModulus::Batching(POLY_MOD, 32).value();
    size_t prime_mod = PLAIN_MOD;
    // std::cout << prime_mod << "\n";
    parms.set_plain_modulus(prime_mod);
    seal::SEALContext context(parms, true, seal::sec_level_type::tc128);

    auto io
        = Utils::init_ios<IO::NetIO>(party == emp::ALICE ? nullptr : "127.0.0.1", 6969, threads);

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(io, *pkey, *o_pkey, context, party);

    seal::Encryptor enc(context, *o_pkey);
    enc.set_secret_key(skey);
    seal::Decryptor dec(context, skey);

    uint64_t num_triples = 9'006'592;
    std::vector<uint64_t> A(num_triples);
    std::vector<uint64_t> B(num_triples);
    std::vector<uint64_t> C(num_triples);

    auto func = [&](int wid, size_t start, size_t end) {
        if (start >= end)
            return Code::OK;
        size_t triple = end - start;

        for (size_t i = start; i < end; ++i) {
            A[i] = 2;
            B[i] = 3;
        }

        elemwise_product_ab(&context, io[wid], &enc, &dec, triple, A.data() + start,
                            B.data() + start, C.data() + start, prime_mod, party, *o_pkey);
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    size_t data = 0;
    for (int i = 0; i < threads; ++i) data += io[i]->counter;
    string st;
    std::cout << "P" << party - 1 << ": time[s]: " << Utils::to_sec(Utils::time_diff(start))
              << "\n";
    std::cout << "P" << party - 1 << ": data: " << Utils::to_MB(data, st) << st << "\n";

    for (int i = 0; i < threads; ++i) {
        delete io[i];
    }
    delete[] io;
}

void do_multiplex(int num_input, const UINT_TYPE* x32, const uint8_t* sel_packed, UINT_TYPE* y32,
                  int party, const std::string& ip, int port, int io_offset, int threads) {
    int bitlen = BIT_LEN;

    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(1);

    uint8_t* sel = new uint8_t[num_input];
    uint64_t* x  = new uint64_t[num_input];
    uint64_t* y  = new uint64_t[num_input];

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        for (size_t i = start; i < end; ++i) {
            sel[i] = get_nth(sel_packed, i);
            // if (party == emp::ALICE)
            //     sel[i] = sel[i] ^ 1;
            x[i] = x32[i];
        }

        if (wid & 1)
            Aux::multiplexer(keys.get_otpack(wid), OTHER_PARTY(party), sel + start, x + start,
                             y + start, end - start, bitlen, bitlen);
        else
            Aux::multiplexer(keys.get_otpack(wid), party, sel + start, x + start, y + start,
                             end - start, bitlen, bitlen);

        for (size_t i = start; i < end; ++i) {
            y32[i] = y[i];
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(1);
    gemini::LaunchWorks(tpool, num_input, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": multiplex time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": multiplex data[", unit, "]: ", data);

#ifdef VERIFY
    if (party == emp::BOB) {
        ios[0]->send_data(sel, sizeof(*sel) * num_input);
        ios[0]->send_data(x32, sizeof(*x32) * num_input);
        ios[0]->send_data(y32, sizeof(*y32) * num_input);
        ios[0]->flush();
    } else {
        Utils::log(Utils::Level::DEBUG, "Verifying MULTIPLEX: ", num_input);
        std::vector<uint8_t> sel_b(num_input);
        std::vector<UINT_TYPE> x_b(num_input);
        std::vector<UINT_TYPE> y_b(num_input);

        ios[0]->recv_data(sel_b.data(), sizeof(decltype(sel_b)::value_type) * num_input);
        ios[0]->recv_data(x_b.data(), sizeof(decltype(x_b)::value_type) * num_input);
        ios[0]->recv_data(y_b.data(), sizeof(decltype(y_b)::value_type) * num_input);

        bool passed = true;
        for (int i = 0; i < num_input; ++i) {
            if (((y32[i] + y_b[i]) & moduloMask)
                != ((x32[i] + x_b[i]) & moduloMask) * ((get_nth(sel_packed, i) ^ sel_b[i]))) {
                passed = false;
                Utils::log(Utils::Level::FAILED, "(", y32[i], " + ", y_b[i], ") = (", x32[i], " + ",
                           x_b[i], ") (", (uint32_t)get_nth(sel_packed, i), " ^ ",
                           (uint32_t)sel_b[i], ")");
                break;
            }
        }

        if (passed)
            Utils::log(Utils::Level::PASSED, "MULTIPLEX: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "MULTIPLEX: FAILED");
    }
#endif

    delete[] sel;
    delete[] x;
    delete[] y;

    keys.disconnect();
}

void generateOT(int party, const std::string& ip, int port, int threads, int io_offset) {
    unsigned num_triples = 9'000'000;
    uint64_t* a          = new uint64_t[num_triples];
    uint8_t* b           = new uint8_t[num_triples];

    for (unsigned i = 0; i < num_triples; ++i) {
        a[i] = 1;
        b[i] = party == emp::ALICE ? 0 : 0;
    }

    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        size_t n = end - start;
        auto* ot = keys.get_otpack(wid);

        switch (party) {
        case emp::ALICE: {
            uint64_t** ot_message = new uint64_t*[n];

            for (unsigned i = 0; i < n; ++i) {
                ot_message[i]    = new uint64_t[2];
                ot_message[i][0] = a[i];
                ot_message[i][1] = b[i];
            }

            ot->silent_ot->send(ot_message, n, 32);
            ot->silent_ot->flush();

            if (party == emp::ALICE) {
                for (unsigned i = 0; i < n; ++i) delete ot_message[i];
            }
            delete[] ot_message;
            break;
        }
        case emp::BOB: {
            ot->silent_ot->recv(a, b, n, 32);
            break;
        }
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    delete[] a;
    delete[] b;

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": OT time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": OT data[", unit, "]: ", data);

    keys.disconnect();
}

void generateCOT(int party, const UINT_TYPE* a, const uint8_t* b, UINT_TYPE* c,
                 const unsigned& num_triples, const std::string& ip, int port, int threads,
                 int io_offset) {
    Utils::log(Utils::Level::DEBUG, "COT: ", num_triples);
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(1);

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        size_t n = end - start;
        auto* ot = keys.get_otpack(wid);

        switch (party) {
        case emp::ALICE: {
            ot->silent_ot->send_cot(c + start, a + start, n, 32);
            for (size_t i = 0; i < n; ++i) {
                c[i + start] = -c[i + start] & moduloMask;
            }
            break;
        }
        case emp::BOB: {
            uint8_t* sel = new uint8_t[n];
            for (size_t i = 0; i < n; ++i) sel[i] = get_nth(b, start + i);

            ot->silent_ot->recv_cot(c + start, (bool*)sel, n, 32);
            delete[] sel;
            break;
        }
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(1);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": COT time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": COT data[", unit, "]: ", data);

#ifdef VERIFY
    if (party == emp::BOB) {
        ios[0]->send_data(b, sizeof(*b) * num_triples / 8);
        ios[0]->send_data(c, sizeof(*c) * num_triples);
    } else {
        std::vector<uint8_t> b_bob(num_triples / 8);
        std::vector<UINT_TYPE> c_bob(num_triples);

        ios[0]->recv_data(b_bob.data(), b_bob.size());
        ios[0]->recv_data(c_bob.data(), c_bob.size() * sizeof(*c));

        bool passed = true;
        for (size_t i = 0; i < num_triples; ++i) {
            if (((c[i] + c_bob[i]) & moduloMask) != a[i] * get_nth(b_bob.data(), i)) {
                passed = false;
                break;
            }
        }
        if (passed)
            Utils::log(Utils::Level::PASSED, "COT: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "COT: FAILED");
    }
#endif

    keys.disconnect();
}

void generateConvTriplesCheetah2(Keys<IO::NetIO>& keys, size_t total_batches,
                                 std::vector<Utils::ConvParm>& parms, UINT_TYPE** a, UINT_TYPE** b,
                                 UINT_TYPE* c, Utils::PROTO proto, int party, int threads,
                                 int factor, bool is_shared_input) {
    auto start = measure::now();
    threads -= 2;

    auto& hom_conv = keys.get_conv();
    auto** ios     = keys.get_ios(2);

    size_t rounds = proto == Utils::PROTO::AB ? total_batches * 2 : total_batches;

    Thread::Queue<std::tuple<std::stringstream, size_t>> send_queue;
    auto send_thread = std::thread([&]() {
        for (size_t i = 0; i < rounds; ++i) {
            if (auto l = send_queue.pop()) {
                auto s = std::move(l.value());
                IO::send_encrypted_vector(*ios[party - 1], std::get<0>(s),
                                          uint32_t(std::get<1>(s)));
            } else
                break;
        }
    });

    Thread::Queue<vector<seal::Ciphertext>> recv_queue;
    auto recv_thread = std::thread([&]() {
        for (size_t i = 0; i < rounds; ++i) {
            std::vector<seal::Ciphertext> l;
            IO::recv_encrypted_vector(*ios[(OTHER_PARTY(party) - 1)], hom_conv.getContext(), l);
            recv_queue.push(l);
        }
    });

    vector<vector<seal::Plaintext>> enc_a(total_batches);
    vector<vector<vector<seal::Plaintext>>> enc_b(parms.size());
    vector<vector<seal::Ciphertext>> enc_a2(total_batches);
    vector<vector<seal::Serializable<seal::Ciphertext>>> enc_a1(total_batches);

    auto pool = seal::MemoryPoolHandle::New();
    auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

    size_t offset = 0;

    Result result;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta  = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                           parm.n_filters, parm.stride, parm.padding);

        meta.is_shared_input = is_shared_input;
        uint64_t* ai         = new uint64_t[meta.ishape.num_elements() * parm.batchsize];
        if (party == emp::BOB || is_shared_input)
            for (long i = 0; i < meta.ishape.num_elements() * parm.batchsize; ++i) ai[i] = a[n][i];

        uint64_t* bi = new uint64_t[meta.fshape.num_elements() * meta.n_filters * factor];
        if (b)
            for (size_t i = 0; i < meta.fshape.num_elements() * meta.n_filters * factor; ++i)
                bi[i] = b[n][i];

        int ac_batch_size = parm.batchsize / factor;
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            Tensor<uint64_t> A
                = Tensor<uint64_t>::Wrap(ai + meta.ishape.num_elements() * cur_batch, meta.ishape);

            std::vector<Tensor<uint64_t>> B(meta.n_filters);
            for (size_t i = 0; i < meta.n_filters; ++i)
                B[i] = Tensor<uint64_t>::Wrap(
                    bi + meta.fshape.num_elements() * meta.n_filters * (cur_batch / ac_batch_size)
                        + meta.fshape.num_elements() * i,
                    meta.fshape);

            switch (party) {
            case emp::ALICE: {
                if (proto == Utils::PROTO::AB) {
                    hom_conv.encryptImage(A, meta, enc_a1[cur_batch + offset],
                                          enc_a[cur_batch + offset], threads);
                    send_queue.push(enc_a1[cur_batch + offset]);
                } else {
                    if (meta.is_shared_input)
                        hom_conv.encodeImage(A, meta, enc_a[cur_batch + offset], threads);
                }
                if (cur_batch == 0) {
                    hom_conv.encodeFilters(B, meta, enc_b[n], threads);
                    hom_conv.filtersToNtt(enc_b[n], threads);
                }
                break;
            }
            case emp::BOB: {
                if (proto == Utils::PROTO::AB) {
                    if (cur_batch == 0) {
                        hom_conv.encodeFilters(B, meta, enc_b[n], threads);
                        hom_conv.filtersToNtt(enc_b[n], threads);
                    }
                    hom_conv.encryptImage(A, meta, enc_a1[cur_batch + offset],
                                          enc_a[cur_batch + offset], threads);
                } else {
                    hom_conv.encryptImage(A, meta, enc_a1[cur_batch + offset], threads);
                }
                send_queue.push(enc_a1[cur_batch + offset]);
                break;
            }
            }
        }
        delete[] ai;
        delete[] bi;
        offset += parm.batchsize;
    }

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": CONV NTT preprocessing time[s]:", Utils::to_sec(Utils::time_diff(start)));

    vector<vector<seal::Ciphertext>> M(total_batches);
    vector<Tensor<uint64_t>> C(total_batches);
    offset = 0;
    for (size_t n = 0; (proto == Utils::PROTO::AB || party == emp::ALICE) && n < parms.size();
         ++n) {
        auto& parm = parms[n];
        auto meta
            = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                    parm.n_filters, parm.stride, parm.padding, is_shared_input);
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::ALICE: {
                enc_a2[cur_batch + offset] = recv_queue.pop().value();
                result.ret                 = hom_conv.conv2DSS(
                    enc_a2[cur_batch + offset], enc_a[cur_batch + offset], enc_b[n], meta,
                    M[cur_batch + offset], C[cur_batch + offset], threads, true, false, true);
                send_queue.push(M[cur_batch + offset]);
                break;
            }
            case emp::BOB: {
                enc_a2[cur_batch + offset] = recv_queue.pop().value();
                result.ret                 = hom_conv.conv2DSS(
                    enc_a2[cur_batch + offset], enc_a[cur_batch + offset], enc_b[n], meta,
                    M[cur_batch + offset], C[cur_batch + offset], threads, true, false, true);
                send_queue.push(M[cur_batch + offset]);
                break;
            }
            }
        }
        enc_a[n].clear();
        enc_b[n].clear();
        enc_a2[n].clear();
        offset += parm.batchsize;
    }
    enc_a.clear();
    enc_b.clear();
    enc_a2.clear();

    offset          = 0;
    size_t c_offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta
            = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                    parm.n_filters, parm.stride, parm.padding, is_shared_input);

        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::ALICE: {
                if (proto == Utils::PROTO::AB2)
                    break;

                Tensor<uint64_t> tmp;
                M[cur_batch + offset] = recv_queue.pop().value();
                result.ret = hom_conv.decryptToTensor(M[cur_batch + offset], meta, tmp, threads);
                Utils::op_inplace<uint64_t>(
                    C[cur_batch + offset], tmp,
                    [](uint64_t a, uint64_t b) -> uint64_t { return Utils::add(a, b); });
                break;
            }
            case emp::BOB: {
                M[cur_batch + offset] = recv_queue.pop().value();
                if (proto == Utils::PROTO::AB) {
                    Tensor<uint64_t> tmp;
                    result.ret
                        = hom_conv.decryptToTensor(M[cur_batch + offset], meta, tmp, threads);
                    Utils::op_inplace<uint64_t>(
                        C[cur_batch + offset], tmp,
                        [](uint64_t a, uint64_t b) -> uint64_t { return Utils::add(a, b); });
                } else {
                    result.ret = hom_conv.decryptToTensor(M[cur_batch + offset], meta,
                                                          C[cur_batch + offset], threads);
                }
                break;
            }
            }

            for (long i = 0; i < C[cur_batch + offset].NumElements(); ++i)
                c[c_offset + i] = C[cur_batch + offset].data()[i];
            c_offset += C[cur_batch + offset].NumElements();
        }
        offset += parm.batchsize;
    }

    send_thread.join();
    recv_thread.join();

    auto time = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple time + NTT[s]: ", time);
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) {
        data += Utils::to_MB(ios[i]->counter, unit);
        ios[i]->counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple data[", unit, "]: ", data);
}

} // namespace Iface
