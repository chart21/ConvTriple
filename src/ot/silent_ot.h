// Author: Zhicong Huang
#ifndef CHEETAH_SILENT_OT_H
#define CHEETAH_SILENT_OT_H

#include "gemini/core/prg_party.h"
#include <emp-ot/cot.h>
#include <emp-ot/ferret/ferret_cot.h>
#include <math.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "io/net_io_channel.hpp"
#include "ot/mitccrh.h"
#include "ot/ot-utils.h"
#include "ot/ot.h"

// emp-ot's LPN step (LpnF2::task) with the same outputs: the AES of the next groups of 4 outputs is
// computed ahead and their rows of the table are prefetched. With one ferret instance per thread the
// tables of all instances do not fit the caches, and the 40 random reads of one group alone kept the core
// waiting on memory (LPN step of 32 instances at once on an EPYC 7543, b13: 0.34 -> 0.28 s).
namespace cheetah {
// time in ferret's LPN step, summed over all instances and threads (reported with the rcot time, Keys::disconnect)
inline std::atomic<int64_t>& lpn_ns() {
    static std::atomic<int64_t> ns{0};
    return ns;
}
} // namespace cheetah

// The LPN step itself (emp's compute: the seed, then task() over `threads` ranges on the pool), or on the GPU in
// TRIPLE_GPU builds (ot/lpn_gpu.cu, bit for bit the same outputs; LPN_GPU=0 turns it off). Defined once, in the HE
// library (hpmpc_interface.cpp), so that every translation unit uses the same one.
template <>
void LpnF2<IO::NetIO, 10>::compute(block* nn, const block* kk, block s);

template <>
inline void LpnF2<IO::NetIO, 10>::task(block* nn, const block* kk, int64_t start, int64_t end) {
    constexpr int d = 10, ahead = 4;
    PRP prp(seed);
    const int64_t groups = end - 4 > start ? (end - 4 - start + 3) / 4 : 0; // task's groups of 4: j < end - 4
    alignas(16) block ring[ahead + 1][d];
    auto prepare = [&](int64_t g) { // __compute4's AES for group g, and a prefetch of its 40 rows
        block* t = ring[g % (ahead + 1)];
        for (int m = 0; m < d; ++m) t[m] = makeBlock(start + 4 * g, m);
        AES_ecb_encrypt_blks(t, d, &prp.aes);
        const uint32_t* r = reinterpret_cast<const uint32_t*>(t);
        for (int i = 0; i < 4 * d; ++i) {
            uint32_t index = r[i] & uint32_t(mask);
            index -= index >= uint32_t(k) ? uint32_t(k) : 0;
            _mm_prefetch(reinterpret_cast<const char*>(kk + index), _MM_HINT_T0);
        }
    };
    for (int64_t g = 0; g < std::min<int64_t>(ahead, groups); ++g) prepare(g);
    for (int64_t g = 0; g < groups; ++g) {
        if (g + ahead < groups)
            prepare(g + ahead);
        const uint32_t* r = reinterpret_cast<const uint32_t*>(ring[g % (ahead + 1)]);
        for (int m = 0; m < 4; ++m) {
            block acc = nn[start + 4 * g + m];
            for (int i = 0; i < d; ++i) {
                uint32_t index = r[m * d + i] & uint32_t(mask);
                index -= index >= uint32_t(k) ? uint32_t(k) : 0;
                acc = acc ^ kk[index];
            }
            nn[start + 4 * g + m] = acc;
        }
    }
    for (int64_t j = start + 4 * groups; j < end; ++j) __compute1(nn, kk, j, &prp);
}

namespace cheetah {

// Ferret's LPN parameters, emp-ot's ferret_b11 / b12 / b13 (CMake TRIPLE_FERRET, default b12): a smaller LPN
// table against more GGM trees. Defined once in the HE library, so every user of this header agrees.
const PrimalLPNParameter& ferret_param();

template <typename IO>
class SilentOT : public sci::OT<SilentOT<IO>> {
    std::atomic<int64_t> count_rcot_;
    cheetah::MITCCRH<8> mitccrh;
    int threads;
    IO** ios;
    emp::block seed;

  public:
    FerretCOT<IO>* ferret;
    static constexpr int64_t rot_chunk = 4096; // OTs per chunk of send/recv_rot_bits (64 KiB of COTs)

    SilentOT(int party, int threads, IO** ios, bool malicious = false, bool run_setup = true,
             std::string pre_file = "", bool warm_up = true)
        : threads(threads), ios(ios) {
        ferret = new FerretCOT<IO>(party, threads, ios, malicious, run_setup, ferret_param(), pre_file);

#if PRG_SEED != -1
        seed = _mm_set1_epi64x(int64_t(gemini::party_seed64(ios[0]->port, 1)));
        ferret->prg.reseed(&seed);
#endif
        if (warm_up) {
            block tmp;
            ferret->rcot(&tmp, 1);
        }
        count_rcot_ = 0;
    }

    ~SilentOT() {
        ferret->skip_file();
        delete ferret;
    }

    void flush() {
        for (int i = 0; i < threads; ++i) {
            ios[i]->flush();
        }
    }

    void send_impl(const block* data0, const block* data1, int64_t length) {
        send_ot_cm_cc(data0, data1, length);
    }

    void recv_impl(block* data, const bool* b, int64_t length) { recv_ot_cm_cc(data, b, length); }

    template <typename T>
    void send_impl(T** data, int length, int l) {
        send_ot_cm_cc(data, length, l);
    }

    template <typename T>
    void recv_impl(T* data, const uint8_t* b, int length, int l) {
        recv_ot_cm_cc(data, b, length, l);
    }

    template <typename T>
    void send_impl(T** data, int length, int N, int l) {
        send_ot_cm_cc(data, length, N, l);
    }

    template <typename T>
    void recv_impl(T* data, const uint8_t* b, int length, int N, int l) {
        recv_ot_cm_cc(data, b, length, N, l);
    }

    template <class T>
    void send_cot(T* data0, const T* corr, int length, int l) {
        send_ot_cam_cc(data0, corr, length, l);
        flush();
    }

    template <class T>
    void recv_cot(T* data, const bool* b, int length, int l) {
        recv_ot_cam_cc(data, b, length, l);
    }

    // chosen additive message, chosen choice
    // Sender chooses one message 'corr'. A correlation is defined by the addition
    // function: f(x) = x + corr Sender receives a random message 'x' as output
    // ('data0').
    template <class T>
    void send_ot_cam_cc(T* data0, const T* corr, int64_t length, int l) {
        T modulo_mask = (1ULL << l) - 1;
        if (l == sizeof(T) * 8)
            modulo_mask = (T)(-1ULL);
        block* rcm_data = new block[length];
        send_ot_rcm_cc(rcm_data, length);

        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();

        block pad[2 * ot_bsize];
        uint32_t y_size = (uint32_t)ceil((ot_bsize * l) / (float)(sizeof(T) * 8));
        uint32_t corrected_y_size, corrected_bsize;
        T y[y_size];
        T corr_data[ot_bsize];

        for (int64_t i = 0; i < length; i += ot_bsize) {
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = rcm_data[j];
                pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
            }

            ferret->mitccrh.template hash<ot_bsize, 2>(pad);

            for (int j = i; j < i + ot_bsize and j < length; ++j) {
                data0[j]         = _mm_extract_epi64(pad[2 * (j - i)], 0) & modulo_mask;
                corr_data[j - i] = (corr[j] + data0[j] + _mm_extract_epi64(pad[2 * (j - i) + 1], 0))
                                   & modulo_mask;
            }
            corrected_y_size
                = (uint32_t)ceil((std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            sci::pack_cot_messages(y, corr_data, corrected_y_size, corrected_bsize, l);
            ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
        }

        delete[] rcm_data;
    }

    // chosen additive message, chosen choice
    // Receiver chooses a choice bit 'b', and
    // receives 'x' if b = 0, and 'x + corr' if b = 1
    template <class T>
    void recv_ot_cam_cc(T* data, const bool* b, int64_t length, int l) {
        T modulo_mask = (1ULL << l) - 1;
        if (l == sizeof(T) * 8)
            modulo_mask = (T)(-1ULL);

        block* rcm_data = new block[length];
        recv_ot_rcm_cc(rcm_data, b, length);
        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        // ferret->io->flush();

        block pad[ot_bsize];

        uint32_t recvd_size = (uint32_t)ceil((ot_bsize * l) / (float)(sizeof(T) * 8));
        uint32_t corrected_recvd_size, corrected_bsize;
        T corr_data[ot_bsize];
        T recvd[recvd_size];

        for (int64_t i = 0; i < length; i += ot_bsize) {
            corrected_recvd_size
                = (uint32_t)ceil((std::min(ot_bsize, length - i) * l) / (float)(sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
            ferret->mitccrh.template hash<ot_bsize, 1>(pad);

            ferret->io->recv_data(recvd, sizeof(T) * corrected_recvd_size);

            sci::unpack_cot_messages(corr_data, recvd, corrected_bsize, l);

            for (int j = i; j < i + ot_bsize and j < length; ++j) {
                if (b[j])
                    data[j] = (corr_data[j - i] - _mm_extract_epi64(pad[j - i], 0)) & modulo_mask;
                else
                    data[j] = _mm_extract_epi64(pad[j - i], 0) & modulo_mask;
            }
        }

        delete[] rcm_data;
    }

    // chosen message, chosen choice
    void send_ot_cm_cc(const block* data0, const block* data1, int64_t length) {
        block* data = new block[length];
        send_ot_rcm_cc(data, length);

        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();

        block pad[2 * ot_bsize];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = data[j];
                pad[2 * (j - i) + 1] = data[j] ^ ferret->Delta;
            }
            // here, ferret depends on the template parameter "IO", making mitccrh
            // also dependent, hence we have to explicitly tell the compiler that
            // "hash" is a template function. See:
            // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
            ferret->mitccrh.template hash<ot_bsize, 2>(pad);
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = pad[2 * (j - i)] ^ data0[j];
                pad[2 * (j - i) + 1] = pad[2 * (j - i) + 1] ^ data1[j];
            }
            ferret->io->send_data(pad, 2 * sizeof(block) * std::min(ot_bsize, length - i));
        }
        delete[] data;
    }

    // chosen message, chosen choice
    void recv_ot_cm_cc(block* data, const bool* r, int64_t length) {
        recv_ot_rcm_cc(data, r, length);

        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        // ferret->io->flush();

        block res[2 * ot_bsize];
        block pad[ot_bsize];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
            ferret->mitccrh.template hash<ot_bsize, 1>(pad);
            ferret->io->recv_data(res, 2 * sizeof(block) * std::min(ot_bsize, length - i));
            for (int64_t j = 0; j < ot_bsize and j < length - i; ++j) {
                data[i + j] = res[2 * j + r[i + j]] ^ pad[j];
            }
        }
    }

    // chosen message, chosen choice.
    // Here, the 2nd dim of data is always 2. We use T** instead of T*[2] or two
    // arguments of T*, in order to be general and compatible with the API of
    // 1-out-of-N OT.
    template <typename T>
    void send_ot_cm_cc(T** data, int64_t length, int l) {
        block* rcm_data = new block[length];
        send_ot_rcm_cc(rcm_data, length);

        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();

        block pad[2 * ot_bsize];
        uint32_t y_size = (uint32_t)ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
        uint32_t corrected_y_size, corrected_bsize;
        T y[y_size];

        for (int64_t i = 0; i < length; i += ot_bsize) {
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = rcm_data[j];
                pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
            }
            // here, ferret depends on the template parameter "IO", making mitccrh
            // also dependent, hence we have to explicitly tell the compiler that
            // "hash" is a template function. See:
            // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
            ferret->mitccrh.template hash<ot_bsize, 2>(pad);

            corrected_y_size
                = (uint32_t)ceil((2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            sci::pack_ot_messages<T>((T*)y, data + i, pad, corrected_y_size, corrected_bsize, l, 2);

            ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
            ferret->io->flush();
        }
        delete[] rcm_data;
    }

    // chosen message, chosen choice
    // Here, r[i]'s value is always 0 or 1. We use uint8_t instead of bool, in
    // order to be general and compatible with the API of 1-out-of-N OT.
    template <typename T>
    void recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int l) {
        block* rcm_data = new block[length];
        recv_ot_rcm_cc(rcm_data, (const bool*)r, length);

        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        // ferret->io->flush();

        block pad[ot_bsize];

        uint32_t recvd_size = (uint32_t)ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
        uint32_t corrected_recvd_size, corrected_bsize;
        T recvd[recvd_size];

        for (int64_t i = 0; i < length; i += ot_bsize) {
            corrected_recvd_size
                = (uint32_t)ceil((2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

            memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
            ferret->mitccrh.template hash<ot_bsize, 1>(pad);

            sci::unpack_ot_messages<T>(data + i, r + i, (T*)recvd, pad, corrected_bsize, l, 2);
        }
        delete[] rcm_data;
    }

    // random correlated message, chosen choice
    void send_ot_rcm_cc(block* data0, int64_t length) {
        ferret->send_cot(data0, length);
        count_rcot_.fetch_add(length);
    }

    // random correlated message, chosen choice
    void recv_ot_rcm_cc(block* data, const bool* b, int64_t length) {
        ferret->recv_cot(data, b, length);
    }

    // random message, chosen choice
    void send_ot_rm_cc(block* data0, block* data1, int64_t length) {
        send_ot_rcm_cc(data0, length);
        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();

        block pad[ot_bsize * 2];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = data0[j];
                pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
            }
            ferret->mitccrh.template hash<ot_bsize, 2>(pad);
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                data0[j] = pad[2 * (j - i)];
                data1[j] = pad[2 * (j - i) + 1];
            }
        }
    }

    // random message, chosen choice
    void recv_ot_rm_cc(block* data, const bool* r, int64_t length) {
        recv_ot_rcm_cc(data, r, length);
        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        // ferret->io->flush();
        block pad[ot_bsize];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            std::memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
            ferret->mitccrh.template hash<ot_bsize, 1>(pad);
            std::memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
        }
    }

    // random message, random choice
    void send_ot_rm_rc(block* data0, block* data1, int64_t length) {
        timed_rcot(data0, length);

        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();

        block pad[ot_bsize * 2];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                pad[2 * (j - i)]     = data0[j];
                pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
            }
            ferret->mitccrh.template hash<ot_bsize, 2>(pad);
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                data0[j] = pad[2 * (j - i)];
                data1[j] = pad[2 * (j - i) + 1];
            }
        }
    }

    // random message, random choice
    void recv_ot_rm_rc(block* data, bool* r, int64_t length) {
        timed_rcot(data, length);
        for (int64_t i = 0; i < length; i++) {
            r[i] = getLSB(data[i]);
        }

        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        // ferret->io->flush();
        block pad[ot_bsize];
        for (int64_t i = 0; i < length; i += ot_bsize) {
            std::memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
            ferret->mitccrh.template hash<ot_bsize, 1>(pad);
            std::memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
        }
    }

    // Random OTs with 1-bit messages, packed 8 per byte (bit j of byte i belongs to OT 8i + j; length a
    // multiple of 8). The COTs are taken from ferret, hashed and reduced to bits in cache-sized chunks,
    // where send_ot_rm_rc<T> stores two 16-byte blocks and a T per OT and makes several passes over them.
    void send_rot_bits(uint8_t* m0, uint8_t* m1, int64_t length) {
        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();
        std::vector<block> buf(std::min<int64_t>(rot_chunk, length));
        block pad[2 * ot_bsize];
        for (int64_t i0 = 0; i0 < length; i0 += rot_chunk) {
            int64_t n = std::min<int64_t>(rot_chunk, length - i0);
            timed_rcot(buf.data(), n);
            for (int64_t i = 0; i < n; i += ot_bsize) {
                for (int j = 0; j < ot_bsize; j++) {
                    pad[2 * j]     = buf[i + j];
                    pad[2 * j + 1] = buf[i + j] ^ ferret->Delta;
                }
                ferret->mitccrh.template hash<ot_bsize, 2>(pad);
                uint8_t b0 = 0, b1 = 0;
                for (int j = 0; j < ot_bsize; j++) {
                    b0 |= uint8_t(_mm_cvtsi128_si64(pad[2 * j]) & 1) << j;
                    b1 |= uint8_t(_mm_cvtsi128_si64(pad[2 * j + 1]) & 1) << j;
                }
                m0[(i0 + i) / 8] = b0;
                m1[(i0 + i) / 8] = b1;
            }
        }
    }

    // receiver side of send_rot_bits: random choice bits c and the chosen message bits mc
    void recv_rot_bits(uint8_t* mc, uint8_t* c, int64_t length) {
        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        std::vector<block> buf(std::min<int64_t>(rot_chunk, length));
        block pad[ot_bsize];
        for (int64_t i0 = 0; i0 < length; i0 += rot_chunk) {
            int64_t n = std::min<int64_t>(rot_chunk, length - i0);
            timed_rcot(buf.data(), n);
            for (int64_t i = 0; i < n; i += ot_bsize) {
                uint8_t bc = 0, bm = 0;
                for (int j = 0; j < ot_bsize; j++) {
                    pad[j] = buf[i + j];
                    bc |= uint8_t(_mm_cvtsi128_si64(pad[j]) & 1) << j;
                }
                ferret->mitccrh.template hash<ot_bsize, 1>(pad);
                for (int j = 0; j < ot_bsize; j++) bm |= uint8_t(_mm_cvtsi128_si64(pad[j]) & 1) << j;
                c[(i0 + i) / 8]  = bc;
                mc[(i0 + i) / 8] = bm;
            }
        }
    }

    // send_rot_bits with k-bit messages (k <= 64), as k bit planes: bit p of the messages of OT 8i + j is
    // bit j of m0[p][i] / m1[p][i]. The k bits come from one hash per message (plane 0 = send_rot_bits'
    // bit), so a COT carries k independent message bits for the price of one.
    void send_rot_bitplanes(uint8_t* const* m0, uint8_t* const* m1, int k, int64_t length) {
        block s;
        ferret->prg.random_block(&s, 1);
        ferret->io->send_block(&s, 1);
        ferret->mitccrh.setS(s);
        ferret->io->flush();
        std::vector<block> buf(std::min<int64_t>(rot_chunk, length));
        block pad[2 * ot_bsize];
        for (int64_t i0 = 0; i0 < length; i0 += rot_chunk) {
            int64_t n = std::min<int64_t>(rot_chunk, length - i0);
            timed_rcot(buf.data(), n);
            for (int64_t i = 0; i < n; i += ot_bsize) {
                for (int j = 0; j < ot_bsize; j++) {
                    pad[2 * j]     = buf[i + j];
                    pad[2 * j + 1] = buf[i + j] ^ ferret->Delta;
                }
                ferret->mitccrh.template hash<ot_bsize, 2>(pad);
                uint64_t h0[ot_bsize], h1[ot_bsize];
                for (int j = 0; j < ot_bsize; j++) {
                    h0[j] = uint64_t(_mm_cvtsi128_si64(pad[2 * j]));
                    h1[j] = uint64_t(_mm_cvtsi128_si64(pad[2 * j + 1]));
                }
                for (int p = 0; p < k; p++) {
                    uint8_t b0 = 0, b1 = 0;
                    for (int j = 0; j < ot_bsize; j++) {
                        b0 |= uint8_t((h0[j] >> p) & 1) << j;
                        b1 |= uint8_t((h1[j] >> p) & 1) << j;
                    }
                    m0[p][(i0 + i) / 8] = b0;
                    m1[p][(i0 + i) / 8] = b1;
                }
            }
        }
    }

    // receiver side of send_rot_bitplanes: random choice bits c and the k chosen message bit planes mc
    void recv_rot_bitplanes(uint8_t* const* mc, uint8_t* c, int k, int64_t length) {
        block s;
        ferret->io->recv_block(&s, 1);
        ferret->mitccrh.setS(s);
        std::vector<block> buf(std::min<int64_t>(rot_chunk, length));
        block pad[ot_bsize];
        for (int64_t i0 = 0; i0 < length; i0 += rot_chunk) {
            int64_t n = std::min<int64_t>(rot_chunk, length - i0);
            timed_rcot(buf.data(), n);
            for (int64_t i = 0; i < n; i += ot_bsize) {
                uint8_t bc = 0;
                for (int j = 0; j < ot_bsize; j++) {
                    pad[j] = buf[i + j];
                    bc |= uint8_t(_mm_cvtsi128_si64(pad[j]) & 1) << j;
                }
                ferret->mitccrh.template hash<ot_bsize, 1>(pad);
                uint64_t h[ot_bsize];
                for (int j = 0; j < ot_bsize; j++) h[j] = uint64_t(_mm_cvtsi128_si64(pad[j]));
                for (int p = 0; p < k; p++) {
                    uint8_t bm = 0;
                    for (int j = 0; j < ot_bsize; j++) bm |= uint8_t((h[j] >> p) & 1) << j;
                    mc[p][(i0 + i) / 8] = bm;
                }
                c[(i0 + i) / 8] = bc;
            }
        }
    }

    // random message, random choice
    template <typename T>
    void send_ot_rm_rc(T* data0, T* data1, int64_t length, int l) {
        block* rm_data0 = new block[length];
        block* rm_data1 = new block[length];
        send_ot_rm_rc(rm_data0, rm_data1, length);

        T mask = (T)((1ULL << l) - 1ULL);

        for (int64_t i = 0; i < length; i++) {
            data0[i] = ((T)_mm_extract_epi64(rm_data0[i], 0)) & mask;
            data1[i] = ((T)_mm_extract_epi64(rm_data1[i], 0)) & mask;
        }

        delete[] rm_data0;
        delete[] rm_data1;
    }

    // random message, random choice
    template <typename T>
    void recv_ot_rm_rc(T* data, bool* r, int64_t length, int l) {
        block* rm_data = new block[length];
        recv_ot_rm_rc(rm_data, r, length);

        T mask = (T)((1ULL << l) - 1ULL);

        for (int64_t i = 0; i < length; i++) {
            data[i] = ((T)_mm_extract_epi64(rm_data[i], 0)) & mask;
        }

        delete[] rm_data;
    }

    // chosen message, chosen choice.
    // One-oo-N OT, where each message has l bits. Here, the 2nd dim of data is N.
    template <typename T>
    void send_ot_cm_cc(T** data, int64_t length, int N, int l) {
        int logN = (int)ceil(log2(N));

        block* rm_data0 = new block[length * logN];
        block* rm_data1 = new block[length * logN];
        send_ot_rm_cc(rm_data0, rm_data1, length * logN);

        block pad[ot_bsize * N];
        uint32_t y_size = (uint32_t)ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
        uint32_t corrected_y_size, corrected_bsize;
        T y[y_size];

        block* hash_in0 = new block[N - 1];
        block* hash_in1 = new block[N - 1];
        block* hash_out = new block[2 * N - 2];
        int idx         = 0;
        for (int x = 0; x < logN; x++) {
            for (int y = 0; y < (1 << x); y++) {
                hash_in0[idx] = makeBlock(y, 0);
                hash_in1[idx] = makeBlock((1 << x) + y, 0);
                idx++;
            }
        }

        for (int64_t i = 0; i < length; i += ot_bsize) {
            std::memset(pad, 0, sizeof(block) * N * ot_bsize);
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                mitccrh.renew_ks(rm_data0 + j * logN, logN);
                mitccrh.hash_exp(hash_out, hash_in0, logN);
                mitccrh.renew_ks(rm_data1 + j * logN, logN);
                mitccrh.hash_exp(hash_out + N - 1, hash_in1, logN);

                for (int64_t k = 0; k < N; k++) {
                    idx = 0;
                    for (int64_t s = 0; s < logN; s++) {
                        int mask = (1 << s) - 1;
                        int pref = k & mask;
                        if ((k & (1 << s)) == 0)
                            pad[(j - i) * N + k] ^= hash_out[idx + pref];
                        else
                            pad[(j - i) * N + k] ^= hash_out[idx + N - 1 + pref];
                        idx += 1 << s;
                    }
                }
            }

            corrected_y_size
                = (uint32_t)ceil((std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            sci::pack_ot_messages<T>((T*)y, data + i, pad, corrected_y_size, corrected_bsize, l, N);

            ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
        }

        delete[] hash_in0;
        delete[] hash_in1;
        delete[] hash_out;
        delete[] rm_data0;
        delete[] rm_data1;
    }

    // chosen message, chosen choice
    // One-oo-N OT, where each message has l bits. Here, r[i]'s value is in [0,
    // N).
    template <typename T>
    void recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int N, int l) {
        int logN = (int)ceil(log2(N));

        block* rm_data  = new block[length * logN];
        bool* b_choices = new bool[length * logN];
        for (int64_t i = 0; i < length; i++) {
            for (int64_t j = 0; j < logN; j++) {
                b_choices[i * logN + j] = (bool)((r[i] & (1 << j)) >> j);
            }
        }
        recv_ot_rm_cc(rm_data, b_choices, length * logN);

        block pad[ot_bsize];

        uint32_t recvd_size = (uint32_t)ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
        uint32_t corrected_recvd_size, corrected_bsize;
        T recvd[recvd_size];

        block* hash_out = new block[logN];
        block* hash_in  = new block[logN];

        for (int64_t i = 0; i < length; i += ot_bsize) {
            corrected_recvd_size
                = (uint32_t)ceil((std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8));
            corrected_bsize = std::min(ot_bsize, length - i);

            ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

            std::memset(pad, 0, sizeof(block) * ot_bsize);
            for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
                for (int64_t s = 0; s < logN; s++)
                    hash_in[s] = makeBlock(r[j] & ((1 << (s + 1)) - 1), 0);
                mitccrh.renew_ks(rm_data + j * logN, logN);
                mitccrh.hash_single(hash_out, hash_in, logN);

                for (int64_t s = 0; s < logN; s++) {
                    pad[j - i] ^= hash_out[s];
                }
            }

            sci::unpack_ot_messages<T>(data + i, r + i, (T*)recvd, pad, corrected_bsize, l, N);
        }
        delete[] hash_in;
        delete[] hash_out;
        delete[] rm_data;
        delete[] b_choices;
    }

    void send_batched_got(uint64_t* data, int num_ot, int l, int msgs_per_ot = 1) {
        throw std::logic_error("Not implemented");
    }

    void recv_batched_got(uint64_t* data, const uint8_t* r, int num_ot, int l,
                          int msgs_per_ot = 1) {
        throw std::logic_error("Not implemented");
    }

    void send_batched_cot(uint64_t* data0, uint64_t* corr, std::vector<int> msg_len, int num_ot,
                          int msgs_per_ot = 1) {
        throw std::logic_error("Not implemented");
    }

    void recv_batched_cot(uint64_t* data, bool* b, std::vector<int> msg_len, int num_ot,
                          int msgs_per_ot = 1) {
        throw std::logic_error("Not implemented");
    }

    int64_t get_rcot_count() const { return count_rcot_.load(); }

    // ferret extensions triggered by the COT requests of this instance, and the time spent in them
    int64_t extensions() const { return extensions_; }
    double rcot_seconds() const { return rcot_ns_ * 1e-9; }

  private:
    int64_t extensions_ = 0, rcot_ns_ = 0;
    void timed_rcot(block* data, int64_t n) {
        const int64_t left = ferret->ot_limit - ferret->ot_used;  // silent_ot_left() is private
        if (n > left)
            extensions_ += 1 + (n - left) / std::max<int64_t>(1, ferret->ot_limit);
        const auto t0 = std::chrono::steady_clock::now();
        ferret->rcot(data, n);
        rcot_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }

  public:
};

template <typename IO>
class SilentOTN : public sci::OT<SilentOTN<IO>> {
  public:
    SilentOT<IO>* silent_ot;
    int N;

    SilentOTN(SilentOT<IO>* silent_ot, int N) {
        this->silent_ot = silent_ot;
        this->N         = N;
    }

    void flush() { silent_ot->flush(); }

    template <typename T>
    void send_impl(T** data, int length, int l) {
        silent_ot->send_impl(data, length, N, l);
    }

    template <typename T>
    void recv_impl(T* data, const uint8_t* b, int length, int l) {
        silent_ot->recv_impl(data, b, length, N, l);
    }
};

} // namespace cheetah

#endif // CHEETAH_SILENT_OT_H
