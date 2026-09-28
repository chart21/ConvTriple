#include "gemini/core/prg_party.h"
#include <fstream>

#include "utils.hpp"

void Utils::add_result(Result& res, const Result& res2) {
    res.encryption += res2.encryption;
    res.cipher_op += res2.cipher_op;
    res.plain_op += res2.plain_op;
    res.decryption += res2.decryption;
    res.send_recv += res2.send_recv;
    res.serial += res2.serial;
    res.bytes += res2.bytes;
}

void Utils::print_info(const gemini::HomConv2DSS::Meta& meta, const size_t& padding) {
    log(Level::DEBUG, "n_threads: ", N_THREADS);
    log(Level::DEBUG, "Padding: ", padding);
    log(Level::DEBUG, "Stride: ", meta.stride);
    log(Level::DEBUG, "i_channel: ", meta.ishape.channels());
    log(Level::DEBUG, "i_width: ", meta.ishape.width());
    log(Level::DEBUG, "i_height: ", meta.ishape.height());
    log(Level::DEBUG, "f_channel: ", meta.fshape.channels());
    log(Level::DEBUG, "f_width: ", meta.fshape.width());
    log(Level::DEBUG, "f_height: ", meta.fshape.height());
    log(Level::DEBUG, "n_filters: ", meta.n_filters);
}

seal::SEALContext Utils::init_he_context() {
    seal::EncryptionParameters params(seal::scheme_type::bfv);
    params.set_poly_modulus_degree(POLY_MOD);
    params.set_n_special_primes(0);
    params.set_coeff_modulus(seal::CoeffModulus::Create(POLY_MOD, {60, 49}));
    params.set_plain_modulus(PLAIN_MOD);

#if PRG_SEED != -1
    seal::prng_seed_type seed = {gemini::party_seed64()};
    params.set_random_generator(std::make_shared<seal::Blake2xbPRNGFactory>(seed));
#endif

    return seal::SEALContext(params, true, SEC_LEVEL);
}

gemini::HomBNSS::Meta Utils::init_meta_bn(const long& rows, const long& h, const long& w) {
    gemini::HomBNSS::Meta meta;
    meta.ishape          = {rows, h, w};
    meta.vec_shape       = {rows};
    meta.target_base_mod = PLAIN_MOD;
    meta.is_shared_input = true;
    return meta;
}

gemini::HomBNSS::Meta Utils::init_meta_bn(const long& rows, const long& cols) {
    gemini::HomBNSS::Meta meta;
    long tmp_w = sqrt(cols);
    long tmp_h = tmp_w;

    while (tmp_h * tmp_w != cols) {
        tmp_w += 1;
        tmp_h = cols / tmp_w;
    }

    meta.ishape          = {rows, tmp_h, tmp_w};
    meta.vec_shape       = {rows};
    meta.target_base_mod = PLAIN_MOD;
    meta.is_shared_input = true;
    return meta;
}

gemini::HomFCSS::Meta Utils::init_meta_fc(const long& common, const long& filter_h) {
    gemini::HomFCSS::Meta meta;

    meta.input_shape     = {common};
    meta.weight_shape    = {filter_h, common};
    meta.is_shared_input = true;

    return meta;
}

gemini::HomConv2DSS::Meta Utils::init_meta_conv(const long& ic, const long& ih, const long& iw,
                                                const long& fc, const long& fh, const long& fw,
                                                const size_t& n_filter, const size_t& stride,
                                                const size_t& padding, bool is_shared) {
    if (ic != fc) {
        Utils::log(Level::ERROR, "Filter Channel and Image Channel must match");
    }
    gemini::HomConv2DSS::Meta meta;

    meta.ishape          = {ic, ih, iw};
    meta.fshape          = {fc, fh, fw};
    meta.is_shared_input = is_shared;
    meta.n_filters       = n_filter;
    meta.padding         = padding == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
    meta.stride          = stride;

    return meta;
}

double Utils::convert(uint64_t v, int nbits) {
    int64_t sv = getSignedVal(getRingElt(v));
    return sv / (1. * std::pow(2, nbits));
}

Utils::Result Utils::average(const std::vector<Result>& res, bool average_bytes) {
    Result avg = {.encryption = 0,
                  .cipher_op  = 0,
                  .plain_op   = 0,
                  .decryption = 0,
                  .send_recv  = 0,
                  .serial     = 0,
                  .bytes      = 0,
                  .ret        = Code::OK};

    if (!res.size())
        return avg;

    size_t len = 0;
    for (size_t i = 0; i < res.size(); ++i) {
        auto& cur = res[i];
        if (!cur.bytes)
            continue;
        avg.send_recv += cur.send_recv;
        avg.encryption += cur.encryption;
        avg.decryption += cur.decryption;
        avg.cipher_op += cur.cipher_op;
        avg.plain_op += cur.plain_op;
        avg.serial += cur.serial;
        avg.bytes += cur.bytes;
        len++;
    }

    avg.send_recv /= len;
    avg.encryption /= len;
    avg.decryption /= len;
    avg.cipher_op /= len;
    avg.plain_op /= len;
    avg.serial /= len;

    if (average_bytes)
        avg.bytes /= len;

    return avg;
}

void Utils::make_csv(const std::vector<Result>& results, const size_t& batchSize,
                     const size_t& threads, const std::string& path) {
    std::ofstream os;

    if (!path.empty()) {
        log(Level::INFO, "writing results to: ", path);
        os.open(path, std::ios::out | std::ios::trunc);
    }

    double total = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        total += print_results(results[i], i, batchSize, threads, path.empty() ? std::cout : os);
    }

    os << "Total time[s]: " << total << "\n";
    double data = 0;
    std::string unit;
    for (auto& res : results) data += Utils::to_MB(res.bytes, unit);
    os << "Total data[" << unit << "]: " << data << "\n";

    if (os.is_open())
        os.close();
}

double Utils::print_results(const Result& res, const size_t& layer, const size_t& batchSize,
                            const size_t& threads, std::ostream& out) {
    std::string unit;
    auto data = to_MB(res.bytes, unit);
    if (!layer)
        out << "Encryption [ms],Cipher Calculations [s],Serialization [s],Decryption [ms],Plain "
               "Calculations [ms], "
               "Sending and Receiving [ms],Total [s],Bytes Send ["
            << unit << "],batchSize,threads\n";

    double total = to_msec(res.encryption) + to_msec(res.cipher_op) + to_msec(res.send_recv)
                   + to_msec(res.decryption) + to_msec(res.plain_op) + to_msec(res.serial);
    total /= 1'000.0;

    out << to_msec(res.encryption) << ", " << to_sec(res.cipher_op) << ", " << to_sec(res.serial)
        << ", " << to_msec(res.decryption) << ", " << to_msec(res.plain_op) << ", "
        << to_msec(res.send_recv) << ", " << total << ", " << data << ", " << batchSize << ", "
        << threads << "\n";

    return total;
}

std::vector<gemini::HomFCSS::Meta> Utils::init_layers_fc() {
    std::vector<gemini::HomFCSS::Meta> layers;
    layers.push_back(Utils::init_meta_fc(1000, 2048));
    return layers;
}

std::vector<gemini::Tensor<uint64_t>> Utils::init_filter(const gemini::HomConv2DSS::Meta& meta,
                                                         const double& num) {
    std::vector<gemini::Tensor<uint64_t>> filters(meta.n_filters);

    for (auto& filter : filters) {
        gemini::Tensor<double> tmp(meta.fshape);
        // tmp.Randomize(0.3);

        for (int c = 0; c < tmp.channels(); ++c) {
            for (int i = 0; i < tmp.height(); ++i) {
                for (int j = 0; j < tmp.width(); ++j) {
                    tmp(c, i, j) = num;
                }
            }
        }
        filter = Utils::convert_fix_point(tmp);
    }

    return filters;
}

gemini::Tensor<double> Utils::convert_double(const gemini::Tensor<uint64_t>& in) {
    gemini::Tensor<double> out(in.shape());
    out.tensor()
        = in.tensor().unaryExpr([&](uint64_t v) { return Utils::convert(v, filter_prec); });

    return out;
}

gemini::TensorShape Utils::getOutDim(const ConvParm& parm) {
    long w = ((parm.iw + 2 * parm.padding - parm.fw) / parm.stride) + 1;
    long h = ((parm.ih + 2 * parm.padding - parm.fh) / parm.stride) + 1;
    return {static_cast<long>(parm.n_filters), h, w};
}
