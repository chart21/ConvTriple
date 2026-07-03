#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <ostream>
#include <thread>

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>
#include <gemini/cheetah/tensor.h>

#include <seal/seal.h>

#include "constants.hpp"

constexpr seal::sec_level_type SEC_LEVEL = seal::sec_level_type::tc128;

namespace Utils {

struct ConvParm {
    int batchsize;
    int ic;
    int iw;
    int ih;

    int fc;
    int fw;
    int fh;
    int n_filters;

    int stride;
    int padding;
};

enum class PROTO {
    AB,
    AB2,
    AB2P,  // AB2 with the HE roles flipped and the evaluating party PRESCRIBING its output share
           // (MODELWEIGHTS_KNOWN_DURING_PREPROCESSING; no share-fixing communication needed)
};

inline std::string proto_str(const PROTO& proto) {
    switch (proto) {
    case PROTO::AB:
        return "AB";
    case PROTO::AB2:
        return "AB2";
    case PROTO::AB2P:
        return "AB2P";
    default:
        return "UNKNOWN";
    }
}

enum class Level {
    DEBUG,
    INFO,
    PASSED,
    FAILED,
    ERROR,
};

template <class... Args>
void log(const Level& l, const Args&... args) {
#if LOGLEVEL == 1
    if (l == Level::INFO)
        return;
#elif LOGLEVEL == 2
    if (l != Level::ERROR && l != Level::FAILED)
        return;
#endif

    auto* stream = &std::cerr;
    switch (l) {
    case Level::DEBUG:
        *stream << PURPLE;
        break;
    case Level::PASSED:
        *stream << GREEN;
        break;
    case Level::FAILED:
        *stream << RED;
        break;
    case Level::ERROR:
        stream = &std::cout;
        *stream << RED;
        break;
    default:
        stream = &std::cerr;
    }

    (*stream << ... << args) << (l != Level::INFO ? NC : "") << std::endl;

    if (l == Level::ERROR)
        exit(EXEC_FAILED);
}

template <class T>
inline double to_sec(const T& num) {
    if constexpr (std::is_same<Unit, std::chrono::microseconds>::value) {
        return num / 1'000'000.0;
    }

    log(Level::ERROR, "Unknown <Unit>");
    return 0.0;
}

template <class T>
inline double to_msec(const T& num) {
    if constexpr (std::is_same<Unit, std::chrono::microseconds>::value) {
        return num / 1'000.0;
    }

    log(Level::ERROR, "Unknown <Unit>");
    return 0.0;
}

template <class T>
inline double to_MB(const T& bytes, std::string& unit) {
    // return bytes / 1'000'000.0;
    unit = "MiB";
    return static_cast<double>(bytes) / (1.0 * (1 << 20));
}

template <class Time>
inline size_t time_diff(const Time& start) {
    return std::chrono::duration_cast<Unit>(measure::now() - start).count();
}

struct Result {
    double encryption = 0;
    double cipher_op  = 0;
    double plain_op   = 0;
    double decryption = 0;
    double send_recv  = 0;
    double serial     = 0;
    double bytes      = 0;
    Code ret          = Code::OK;
};

seal::SEALContext init_he_context();
void print_info(const gemini::HomConv2DSS::Meta& meta, const size_t& padding);

gemini::HomFCSS::Meta init_meta_fc(const long& image_h, const long& filter_h);
gemini::HomBNSS::Meta init_meta_bn(const long& image_h, const long& filter_h);
gemini::HomBNSS::Meta init_meta_bn(const long& image_h, const long& filter_h, const long& filter_w);

gemini::HomConv2DSS::Meta init_meta_conv(const long& ic, const long& ih, const long& iw,
                                         const long& fc, const long& fh, const long& fw,
                                         const size_t& n_filter, const size_t& stride,
                                         const size_t& padding, bool is_shared = true);

std::vector<gemini::HomFCSS::Meta> init_layers_fc();

/**
 * Assigns a port to each thread
 * @param addr NULL for server otherwise IP-addr
 * @param port [Port..(port + threads)] to use
 * @param threads number of ports to listen
 * @return The created channels
 */
template <class Channel>
Channel** init_ios(const char* addr, const int& port, const size_t& threads, const int& offset = 1);

template <class T>
gemini::Tensor<uint64_t> convert_fix_point(const gemini::Tensor<T>& in);

template <class T>
void print_tensor(const gemini::Tensor<T>& t, const long& channel = 0);

double convert(uint64_t v, int nbits);
gemini::Tensor<double> convert_double(const gemini::Tensor<uint64_t>& in);
template <class Meta>
gemini::Tensor<uint64_t> init_image(const Meta& meta, const double& num);
std::vector<gemini::Tensor<uint64_t>> init_filter(const gemini::HomConv2DSS::Meta& meta,
                                                  const double& num);

template <class T>
void op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B, std::function<T(T, T)> op);

template <class T>
static inline uint64_t getRingElt(T x) {
    return ((uint64_t)x) & moduloMask;
}

template <class T>
inline uint64_t add(T a, T b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return getRingElt(sum);
}

static inline int64_t getSignedVal(uint64_t x) {
    assert(x < MOD);
    if (x > moduloMidPt)
        return static_cast<int64_t>(x - MOD);
    else
        return static_cast<int64_t>(x);
}

void add_result(Result& res, const Result& res2);

Result average(const std::vector<Result>& res, bool average_bytes);

double print_results(const Result& res, const size_t& layer, const size_t& batchSize,
                     const size_t& threads, std::ostream& out = std::cout);

void make_csv(const std::vector<Result>& results, const size_t& batchSize, const size_t& threads,
              const std::string& path = "");

template <class T>
std::vector<gemini::Tensor<uint64_t>> to_tensor64(T* buf, const gemini::TensorShape& shape,
                                                  const size_t& batch = 1) {
    std::vector<gemini::Tensor<uint64_t>> res(batch, gemini::Tensor<uint64_t>(shape));

    for (size_t cur = 0; cur < batch; ++cur) {
        uint64_t* data = res[cur].data();
        for (ssize_t i = 0; i < shape.num_elements(); ++i) {
            data[i] = static_cast<uint64_t>(buf[i + cur * shape.num_elements()]);
        }
    }

    return res;
}

gemini::TensorShape getOutDim(const ConvParm& parm);

template <class T>
void transpose(T* mat, uint64_t* mat_t, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            mat_t[j * h + i] = mat[i * w + j];
        }
    }
}

} // namespace Utils

template <class Meta>
gemini::Tensor<uint64_t> Utils::init_image(const Meta& meta, const double& num) {
    gemini::Tensor<uint64_t> image(meta.ishape);

    for (int c = 0; c < image.channels(); ++c) {
        for (int i = 0; i < image.height(); ++i) {
            for (int j = 0; j < image.width(); ++j) {
                image(c, i, j) = num + j;
            }
        }
    }

    return image;
}

template <class T>
void Utils::op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B,
                       std::function<T(T, T)> op) {
    assert(A.shape() == B.shape());

    switch (A.dims()) {
    case 3:
        for (int i = 0; i < A.channels(); ++i) {
            for (int j = 0; j < A.height(); ++j) {
                for (int k = 0; k < A.width(); ++k) {
                    A(i, j, k) = op(A(i, j, k), B(i, j, k));
                }
            }
        }
        break;
    case 2:
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                A(i, j) = op(A(i, j), B(i, j));
            }
        }
        break;
    case 1:
        for (int i = 0; i < A.length(); ++i) {
            A(i) = op(A(i), B(i));
        }
        break;
    }
}

template <class Channel>
Channel** Utils::init_ios(const char* addr, const int& port, const size_t& threads,
                          const int& offset) {
    Channel** res = new Channel*[threads];
    for (size_t wid = 0; wid < threads; ++wid) {
        res[wid] = new Channel(addr, port + wid * offset, false);
    }
    return res;
}

template <class T>
gemini::Tensor<uint64_t> Utils::convert_fix_point(const gemini::Tensor<T>& in) {
    gemini::Tensor<uint64_t> out(in.shape());
    out.tensor() = in.tensor().unaryExpr([](T num) {
        double u    = static_cast<double>(num);
        auto sign   = std::signbit(u);
        uint64_t su = std::floor(std::abs(u * (1 << filter_prec)));
        return getRingElt(sign ? -su : su);
    });

    return out;
}

template <class T>
void Utils::print_tensor(const gemini::Tensor<T>& t, const long& channel) {
    switch (t.dims()) {
    case 3:
        for (long i = 0; i < t.height(); ++i) {
            for (long j = 0; j < t.width(); ++j) {
                std::cerr << static_cast<T>(t(channel, i, j)) << " ";
            }
            std::cerr << "\n";
        }
        break;
    case 2:
        for (long i = 0; i < t.rows(); ++i) {
            for (long j = 0; j < t.cols(); ++j) {
                std::cerr << static_cast<T>(t(i, j)) << " ";
            }
            std::cerr << "\n";
        }
        break;
    case 1:
        for (long i = 0; i < t.length(); ++i) {
            std::cerr << static_cast<T>(t(i)) << "\n";
        }
        break;
    }
}

#endif // DEFS_HPP_
