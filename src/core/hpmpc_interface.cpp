#include "hpmpc_interface.hpp"
#include "core/conv_layout.hpp"

#include <algorithm>
#include <seal/ciphertext.h>
#include <seal/serializable.h>
#include <sstream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "emp-tool/utils/constants.h"
#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/multiplexer.hpp"

#ifndef TRIPLE_FERRET
#define TRIPLE_FERRET ferret_b12
#endif

namespace cheetah {
const PrimalLPNParameter& ferret_param() { return TRIPLE_FERRET; }
} // namespace cheetah

#if USE_LPN_GPU
#include "ot/lpn_gpu.h"
#endif

#if USE_LPN_GPU
namespace {
// TwoKeyPRP's two key schedules (the GGM trees' PRG), as the GPU takes them
void two_key_schedules(uint32_t keys[88]) {
    TwoKeyPRP prp(zero_block, makeBlock(0, 1));
    std::memcpy(keys, prp.aes_key[0].rd_key, 44 * sizeof(uint32_t));
    std::memcpy(keys + 44, prp.aes_key[1].rd_key, 44 * sizeof(uint32_t));
}

// MpcotReg::exec_parallel_sender/recver's split of the trees over the ios (group j < threads - 1: the trees j width ..
// (j + 1) width - 1 on ios[j], the last group the rest on ios[threads - 1]): f(first, end, io) per group, in parallel
// on the pool
template <typename F>
void per_io(MpcotReg<IO::NetIO>* mpcot, const F& f) {
    const int threads = mpcot->threads, trees = mpcot->tree_n, width = trees / threads;
    std::vector<std::future<void>> fut;
    for (int j = 0; j < threads - 1; ++j)
        fut.push_back(mpcot->pool->enqueue([=, &f] { f(j * width, (j + 1) * width, mpcot->ios[j]); }));
    f((threads - 1) * width, trees, mpcot->ios[threads - 1]);
    for (auto& x : fut) x.get();
}

int64_t mismatches(const block* a, const block* b, int64_t n) {
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) bad += !cmpBlock(a + i, b + i, 1);
    return bad;
}

// One semi-honest extension with MPCOT's trees and the LPN step on the GPU (lpn_gpu::Extension): the same messages on
// the same ios in the same order as MpcotReg::mpcot, then LpnF2::compute's seed, so the outputs are emp's bit for bit
// (and either party can run either path). False (nothing sent yet) when the device has no room left.
// FERRET_GPU_CHECK=1 compares the trees and the outputs with emp's CPU code.
// With `leaves` (a device buffer of n blocks), the outputs are built there and stay; only outputs keep_from .. n - 1
// come back to ot_output (the rest on demand, FerretCOT::rcot). Without, all of them.
bool extend_gpu(int party, block Delta, block* ot_output, MpcotReg<IO::NetIO>* mpcot, OTPre<IO::NetIO>* preot,
                LpnF2<IO::NetIO, 10>* lpn, const block* kk, block seed, std::chrono::steady_clock::time_point t0,
                void* leaves, int64_t keep_from) {
    const int depth = mpcot->tree_height, levels = depth - 1;
    const int64_t trees = mpcot->tree_n, leave_n = mpcot->leave_n, n = lpn->n;
    std::unique_ptr<cheetah::lpn_gpu::Extension> gpu;
    try {
        gpu = std::make_unique<cheetah::lpn_gpu::Extension>();
        if (leaves)
            gpu->use_leaves(leaves);
        gpu->reserve(trees, depth, lpn->k);
    } catch (const std::exception& e) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            std::cerr << "ferret on the GPU: " << e.what() << ", this extension (and the next ones that do not fit) on the CPU\n";
        return false;
    }
    gpu->upload_kk(kk, lpn->k);
    uint32_t keys[88];
    two_key_schedules(keys);
    static const bool check = getenv("FERRET_GPU_CHECK") && atoi(getenv("FERRET_GPU_CHECK")) != 0;
    std::vector<block> ref(check ? n : 0);  // emp's sparse vector, then its outputs (check)
    const block one = makeBlock(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFELL);
    int64_t bad_msgs = 0;
    if (party == ALICE) {
        std::vector<SPCOT_Sender<IO::NetIO>*> senders;
        mpcot->mpcot_init_sender(senders, preot);
        std::vector<block> seeds(trees), msg(size_t(trees) * levels * 2);
        for (int64_t i = 0; i < trees; ++i) seeds[i] = senders[i]->seed;
        gpu->sender_trees(seeds.data(), trees, depth, keys, msg.data());
        for (int64_t i = 0; i < trees; ++i) {
            SPCOT_Sender<IO::NetIO>* s = senders[i];
            const block* mi = msg.data() + 2 * i * levels;
            const block sum = ((mi[2 * levels - 2] ^ mi[2 * levels - 1]) & one) ^ mpcot->Delta_f2k;
            if (check) {
                s->compute(ref.data() + i * leave_n, mpcot->Delta_f2k);
                for (int h = 0; h < levels; ++h)
                    bad_msgs += !cmpBlock(&s->m[h], mi + 2 * h, 1) + !cmpBlock(&s->m[levels + h], mi + 2 * h + 1, 1);
                bad_msgs += !cmpBlock(&s->secret_sum_f2, &sum, 1);
            }
            for (int h = 0; h < levels; ++h) s->m[h] = mi[2 * h], s->m[levels + h] = mi[2 * h + 1];
            s->secret_sum_f2 = sum;
        }
        // SPCOT_Sender::send_f2k of every tree of a group (OTPre::send's pads, level by level, then the secret sum),
        // as one message: the same bytes, without emp's flush after every tree
        per_io(mpcot, [&](int first, int end, IO::NetIO* io) {
            std::vector<block> buf(size_t(end - first) * (2 * levels + 1));
            block* b = buf.data();
            for (int i = first; i < end; ++i) {
                const block* m = senders[i]->m;
                for (int h = 0; h < levels; ++h) {
                    const int64_t k = int64_t(i) * levels + h;
                    *b++ = m[h] ^ preot->pre_data[k];
                    *b++ = m[levels + h] ^ preot->pre_data[k + preot->n];
                }
                *b++ = senders[i]->secret_sum_f2;
            }
            io->send_data(buf.data(), buf.size() * sizeof(block));
            io->flush();
        });
        for (auto* s : senders) delete s;
    } else {
        std::vector<SPCOT_Recver<IO::NetIO>*> recvers;
        mpcot->mpcot_init_recver(recvers, preot);
        // SPCOT_Recver::recv_f2k of every tree of a group, from one message (OTPre::recv: the chosen pad ^ pre_data)
        std::vector<block> msg(size_t(trees) * levels), secret(trees);
        std::vector<uint8_t> bits(size_t(trees) * levels);
        per_io(mpcot, [&](int first, int end, IO::NetIO* io) {
            std::vector<block> buf(size_t(end - first) * (2 * levels + 1));
            io->recv_data(buf.data(), buf.size() * sizeof(block));
            const block* b = buf.data();
            for (int i = first; i < end; ++i) {
                SPCOT_Recver<IO::NetIO>* r = recvers[i];
                for (int h = 0; h < levels; ++h, b += 2) {
                    const int64_t k = int64_t(i) * levels + h;
                    r->m[h] = preot->pre_data[k] ^ b[r->b[h] ? 1 : 0];
                    msg[k] = r->m[h], bits[k] = r->b[h];
                }
                r->secret_sum_f2 = *b++;
                secret[i] = r->secret_sum_f2;
            }
        });
        gpu->recver_trees(msg.data(), bits.data(), secret.data(), trees, depth, keys);
        if (check)
            for (int64_t i = 0; i < trees; ++i) recvers[i]->compute(ref.data() + i * leave_n);
        for (auto* r : recvers) delete r;
    }
    int64_t bad_trees = 0;
    if (check) {
        gpu->download(ot_output, n);
        bad_trees = mismatches(ot_output, ref.data(), n);
    }
    const auto t1 = std::chrono::steady_clock::now();
    cheetah::mpcot_ns() += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    // LpnF2::compute: the seed, then the groups of 4 and the trailing outputs of every task range
    lpn->seed = cmpBlock(&seed, &zero_block, 1) ? lpn->seed_gen() : seed;
    PRP prp(lpn->seed);
    uint32_t rk[44];
    std::memcpy(rk, prp.aes.rd_key, sizeof(rk));
    const int64_t width = n / lpn->threads;
    auto range = [&](int t, int64_t& start, int64_t& end) {
        start = t * width, end = t == lpn->threads - 1 ? n : std::min((t + 1) * width, n);
        return end - 4 > start ? (end - 4 - start + 3) / 4 : int64_t(0);
    };
    int64_t start, end;
    std::vector<int64_t> singles;
    for (int t = 0; t < lpn->threads; ++t) {
        const int64_t groups = range(t, start, end);
        gpu->lpn(lpn->k, uint32_t(lpn->mask), rk, start, groups);
        for (int64_t j = start + 4 * groups; j < end; ++j) singles.push_back(j);
    }
    for (size_t i = 0; i < singles.size(); i += 64)
        gpu->lpn_single(lpn->k, rk, singles.data() + i, int(std::min<size_t>(64, singles.size() - i)));
    if (leaves && !check)
        gpu->download_range(ot_output + keep_from, keep_from, n - keep_from);
    else
        gpu->download(ot_output, n);
    cheetah::lpn_ns() += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t1).count();
    if (check) {
        for (int t = 0; t < lpn->threads; ++t) {
            range(t, start, end);
            lpn->task(ref.data(), kk, start, end);
        }
        const int64_t bad_out = mismatches(ot_output, ref.data(), n);
        std::cerr << "FERRET_GPU_CHECK P" << party << ": " << trees << " trees of " << leave_n << " leaves, messages "
                  << (bad_msgs ? "DIFFER" : "equal") << ", trees " << (bad_trees ? "DIFFER" : "equal") << " ("
                  << bad_trees << "), outputs " << (bad_out ? "DIFFER" : "equal") << " (" << bad_out << " of " << n
                  << ")\n";
    }
    return true;
}

// FerretCOT instances whose outputs (ot_data's extensions) stay on the device: their buffer, and the part of ot_data
// that is on the host too
struct DevState {
    void* leaves    = nullptr;
    bool on_device  = false;  // the last extension is in `leaves`
    int64_t host_lo = 0, host_hi = 0;
};
std::mutex g_dev_mutex;
std::unordered_map<const void*, DevState> g_dev;
// a FerretCOT's state, made at its first extension into ot_data (no device buffer if there was no room)
DevState* dev_state(const void* ferret, int64_t n, bool make) {
    std::lock_guard<std::mutex> lock(g_dev_mutex);
    auto it = g_dev.find(ferret);
    if (it != g_dev.end())
        return &it->second;
    if (!make)
        return nullptr;
    static const bool off = getenv("ROT_GPU") && atoi(getenv("ROT_GPU")) == 0;
    DevState& d = g_dev[ferret];
    if (!off)
        d.leaves = cheetah::lpn_gpu::device_alloc(size_t(n) * sizeof(block));
    return &d;
}
} // namespace
#endif

namespace cheetah {
bool ferret_on_gpu() {
#if USE_LPN_GPU
    return lpn_gpu::mpcot_available();
#else
    return false;
#endif
}

bool ferret_on_device(const void* ferret) {
#if USE_LPN_GPU
    std::lock_guard<std::mutex> lock(g_dev_mutex);
    auto it = g_dev.find(ferret);
    return it != g_dev.end() && it->second.leaves != nullptr;
#else
    (void) ferret;
    return false;
#endif
}

void rot_bits_device(const void* cots, bool on_device, int64_t count, const block& s, uint64_t gid0, const block* delta,
                     int k, uint8_t* out, int64_t stride) {
#if USE_LPN_GPU
    uint64_t sv[2] = {uint64_t(_mm_extract_epi64(s, 0)), uint64_t(_mm_extract_epi64(s, 1))}, dv[2] = {0, 0};
    if (delta)
        dv[0] = uint64_t(_mm_extract_epi64(*delta, 0)), dv[1] = uint64_t(_mm_extract_epi64(*delta, 1));
    lpn_gpu::rot_bits(cots, on_device, count, sv, gid0, delta ? dv : nullptr, k, out, stride);
    static const bool check = getenv("ROT_GPU_CHECK") && atoi(getenv("ROT_GPU_CHECK")) != 0;
    if (!check)
        return;
    // emp's MITCCRH<8> on the same COTs
    std::vector<block> x(count);
    if (on_device) {
        lpn_gpu::Extension e;
        e.use_leaves(const_cast<void*>(cots));
        e.download(x.data(), count);
    } else
        std::memcpy(x.data(), cots, size_t(count) * sizeof(block));
    emp::MITCCRH<8> ref;
    ref.setS(s);
    ref.gid = gid0;
    int64_t bad = 0, first_bad = -1, last_bad = -1;
    auto bit = [&](int q, int64_t o) { return (out[q * stride + o / 8] >> (o % 8)) & 1; };
    for (int64_t o0 = 0; o0 < count; o0 += 8) {
        block pad[16];
        for (int j = 0; j < 8; ++j) {
            const block v = o0 + j < count ? x[o0 + j] : zero_block;
            if (delta)
                pad[2 * j] = v, pad[2 * j + 1] = v ^ *delta;
            else
                pad[j] = v;
        }
        if (delta)
            ref.hash<8, 2>(pad);
        else
            ref.hash<8, 1>(pad);
        for (int j = 0; j < 8 && o0 + j < count; ++j) {
            const int64_t o = o0 + j;
            const int64_t bad0 = bad;
            for (int q = 0; q < k; ++q) {
                if (delta) {
                    bad += bit(q, o) != ((uint64_t(_mm_extract_epi64(pad[2 * j], 0)) >> q) & 1);
                    bad += bit(k + q, o) != ((uint64_t(_mm_extract_epi64(pad[2 * j + 1], 0)) >> q) & 1);
                } else {
                    bad += bit(1 + q, o) != ((uint64_t(_mm_extract_epi64(pad[j], 0)) >> q) & 1);
                }
            }
            if (!delta)
                bad += bit(0, o) != int(_mm_extract_epi64(x[o], 0) & 1);
            if (bad != bad0) {
                if (first_bad < 0)
                    first_bad = o;
                last_bad = o;
            }
        }
    }
    static std::atomic<int> shown{0};
    if (bad || shown++ < 4)
        std::cerr << "ROT_GPU_CHECK " << (delta ? "sender" : "receiver") << ": " << count << " OTs, k " << k << ", "
                  << (bad ? "DIFFER" : "equal") << " (" << bad << " bits, OTs " << first_bad << " .. " << last_bad
                  << ", gid0 " << gid0 << ", " << (on_device ? "device" : "host") << ")\n";
#else
    (void) cots, (void) on_device, (void) count, (void) s, (void) gid0, (void) delta, (void) k, (void) out, (void) stride;
    throw std::logic_error("rot_bits_device without a GPU build");
#endif
}
} // namespace cheetah

template <>
void FerretCOT<IO::NetIO>::extend(block* ot_output, MpcotReg<IO::NetIO>* mpcot, OTPre<IO::NetIO>* preot,
                                 LpnF2<IO::NetIO, 10>* lpn, block* ot_input, block seed) {
    const auto t0 = std::chrono::steady_clock::now();
    if (party == ALICE)
        mpcot->sender_init(Delta);
    else
        mpcot->recver_init();
#if USE_LPN_GPU
    // semi-honest, with the leaves exactly the LPN's outputs (b12: 2507 trees of 4096)
    if (!is_malicious && cheetah::lpn_gpu::mpcot_available() && int64_t(mpcot->tree_n) * mpcot->leave_n == lpn->n) {
        // ot_data, where rcot's extensions go (ConvTriple takes COTs in chunks), is pinned once: the outputs are copied
        // into it directly (until the destructor)
        DevState* dev = nullptr;
        if (ot_output == ot_data && lpn->n == param.n) {
            // the outputs stay on the device; the host gets the last M (the next extension's pre-OTs) now, the rest
            // when rcot hands it out. Where they cannot stay, ot_data is pinned for the copies (FERRET_PIN=0 / 1: never
            // / always; pinning when most COTs stay on the device cost more than it saved)
            dev = dev_state(this, param.n, true);
            static const int pin = getenv("FERRET_PIN") ? atoi(getenv("FERRET_PIN")) : -1;
            if (pin == 1 || (pin == -1 && !dev->leaves))
                cheetah::lpn_gpu::pin(ot_data, size_t(param.n) * sizeof(block));
        }
        void* leaves = dev ? dev->leaves : nullptr;
        if (extend_gpu(party, Delta, ot_output, mpcot, preot, lpn, ot_input + mpcot->consist_check_cot_num, seed, t0,
                       leaves, ot_limit)) {
            if (dev)
                dev->on_device = leaves != nullptr, dev->host_lo = leaves ? ot_limit : 0, dev->host_hi = param.n;
            return;
        }
    }
    if (ot_output == ot_data)
        if (DevState* dev = dev_state(this, param.n, false))
            dev->on_device = false, dev->host_lo = 0, dev->host_hi = param.n;
#endif
    mpcot->mpcot(ot_output, preot, ot_input);
    cheetah::mpcot_ns() += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    lpn->compute(ot_output, ot_input + mpcot->consist_check_cot_num, seed);
}

template <>
FerretCOT<IO::NetIO>::~FerretCOT() {
#if USE_LPN_GPU
    if (ot_data != nullptr)
        cheetah::lpn_gpu::unpin(ot_data);
    {
        std::lock_guard<std::mutex> lock(g_dev_mutex);
        auto it = g_dev.find(this);
        if (it != g_dev.end()) {
            cheetah::lpn_gpu::device_free(it->second.leaves);
            g_dev.erase(it);
        }
    }
#endif
    if (ot_pre_data != nullptr) {
        if (party == ALICE)
            write_pre_data128_to_file((void*) ot_pre_data, (__uint128_t) Delta, pre_ot_filename);
        else
            write_pre_data128_to_file((void*) ot_pre_data, (__uint128_t) 0, pre_ot_filename);
        delete[] ot_pre_data;
    }
    if (ot_data != nullptr)
        delete[] ot_data;
    if (pre_ot != nullptr)
        delete pre_ot;
    delete base_cot;
    delete pool;
    if (lpn_f2 != nullptr)
        delete lpn_f2;
    if (mpcot != nullptr)
        delete mpcot;
}

// emp's rcot, with ot_data possibly on the device (GPU builds): a consumer (cheetah::device_consumer, SilentOT's
// rot_bits) gets each range where it is (on the device, or on the host), instead of a copy in `data`; everything else
// is copied to the host first (in pieces of at least 1M COTs)
template <>
void FerretCOT<IO::NetIO>::rcot(block* data, int64_t num) {
    if (ot_data == nullptr) {
        ot_data = new block[param.n];
#if USE_LPN_GPU
        // with the outputs on the device, the host buffer only receives the ranges it needs: untouched pages stay
        // unallocated (every extension writes before anything reads)
        if (!cheetah::lpn_gpu::mpcot_available())
#endif
        memset(ot_data, 0, param.n * sizeof(block));
    }
    if (extend_initialized == false)
        error("Run setup before extending");
    cheetah::DeviceConsumer* dc = cheetah::device_consumer();
    // ot_data[ot_used .. ot_used + k - 1] to dst, or to the device consumer
    auto take = [&](block* dst, int64_t k) {
        if (k <= 0)
            return;
#if USE_LPN_GPU
        DevState* dev = dev_state(this, param.n, false);
        if (dev && dev->on_device) {
            if (dc) {
                dc->consume(static_cast<const block*>(dev->leaves) + ot_used, k, true);
                return;
            }
            if (ot_used < dev->host_lo || ot_used + k > dev->host_hi) {
                const int64_t hi = std::min<int64_t>(ot_limit, ot_used + std::max<int64_t>(k, 1 << 20));
                cheetah::lpn_gpu::Extension e;
                e.use_leaves(dev->leaves);
                e.download_range(ot_data + ot_used, ot_used, hi - ot_used);
                dev->host_lo = ot_used, dev->host_hi = hi;
            }
        }
#endif
        if (dc)
            dc->consume(ot_data + ot_used, k, false);
        else
            memcpy(dst, ot_data + ot_used, k * sizeof(block));
    };
    if (num <= silent_ot_left()) {
        take(data, num);
        ot_used += num;
        return;
    }
    block* pt      = data;
    int64_t gened  = silent_ot_left();
    if (gened > 0) {
        take(pt, gened);
        pt += gened;
    }
    int64_t round_inplace = (num - gened - M) / ot_limit;
    int64_t last_round_ot = num - gened - round_inplace * ot_limit;
    bool round_memcpy     = last_round_ot > ot_limit ? true : false;
    if (round_memcpy)
        last_round_ot -= ot_limit;
    if (dc && (round_inplace > 0 || round_memcpy))
        error("rcot: a consumer takes at most ot_limit COTs per call");
    for (int64_t i = 0; i < round_inplace; ++i) {
        extend_f2k(pt);
        ot_used = ot_limit;
        pt += ot_limit;
    }
    if (round_memcpy) {
        extend_f2k();
        take(pt, ot_limit);
        pt += ot_limit;
    }
    if (last_round_ot > 0) {
        extend_f2k();
        take(pt, last_round_ot);
        ot_used = last_round_ot;
    }
}

template <>
void LpnF2<IO::NetIO, 10>::compute(block* nn, const block* kk, block s) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!cmpBlock(&s, &zero_block, 1))
        seed = s;
    else
        seed = seed_gen();
    const int64_t width = n / threads;
#if USE_LPN_GPU
    if (cheetah::lpn_gpu::available()) {
        // the groups of 4 of every task range on the GPU, the at most 7 trailing outputs of a range here
        PRP prp(seed);
        uint32_t rk[44];
        std::memcpy(rk, prp.aes.rd_key, sizeof(rk));
        for (int i = 0; i < threads; ++i) {
            const int64_t start = i * width, end = i == threads - 1 ? n : std::min((i + 1) * width, n);
            const int64_t groups = end - 4 > start ? (end - 4 - start + 3) / 4 : 0;
            cheetah::lpn_gpu::compute(nn, kk, n, k, uint32_t(mask), rk, start, groups);
            for (int64_t j = start + 4 * groups; j < end; ++j) __compute1(nn, kk, j, &prp);
        }
        cheetah::lpn_ns() += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
#endif
    std::vector<std::future<void>> fut;
    for (int i = 0; i < threads - 1; ++i) {
        const int64_t start = i * width, end = std::min((i + 1) * width, n);
        fut.push_back(pool->enqueue([this, nn, kk, start, end]() { task(nn, kk, start, end); }));
    }
    task(nn, kk, (threads - 1) * width, n);
    for (auto& f : fut) f.get();
    cheetah::lpn_ns() += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
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

// Accumulator for aggregated TRIPLE_STATS
struct TripleStatEntry {
    double mb_sent = 0;
    double mb_recv = 0;
    double time_s  = 0;
};

static std::unordered_map<std::string, TripleStatEntry> g_triple_stats;
static std::mutex g_triple_stats_mutex;  // generators on two threads (conv triples on side channels)

static void accumulateTripleStat(const std::string& type, double sent, double recv, double time) {
    std::lock_guard<std::mutex> lock(g_triple_stats_mutex);
    auto& e = g_triple_stats[type];
    e.mb_sent += sent;
    e.mb_recv += recv;
    e.time_s  += time;
}

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
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_triples, " BOOL COT MULT triples (threads: ", threads, ")");
    require_tuple_count_multiple_of_8(num_triples);
    uint64_t num_bytes = (num_triples + 7) / 8;
    // std::atomic<int> setup = 0;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(num_triples);
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, int start, int end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        // auto start_setup = measure::now();

        // sci::OTPack<IO::NetIO> pack(ios + wid, 1, cur_party, true, false);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, keys.ot_io(wid), keys.get_otpack(wid), false);

        // setup += Utils::time_diff(start_setup);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / ot_threads / 8));
            switch (cur_party) {
            case emp::ALICE:
                cot_multiply_shares(emp::ALICE, triple_gen.otpack, a + total, b + total, c + total, current * 8);
                break;
            case emp::BOB:
                cot_multiply_shares(emp::BOB, triple_gen.otpack, a + total, b + total, c + total, current * 8);
                break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Bool COT Mult triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Bool COT Mult triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BOOL_COT_MULT", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

    // Utils::log(Utils::Level::INFO, "P", party - 1, ": Setup time [s]: ",
    //            Utils::to_sec(setup.load())
    //                / (num_triples > static_cast<size_t>(threads) ? threads : num_triples));

    // keys.disconnect();
}

struct BoolMultRounds {
    Keys<IO::NetIO>* keys = nullptr;
    int party = 0, rounds = 0, threads = 0;
    uint64_t num_bytes = 0;
    std::unique_ptr<gemini::ThreadPool> pool;
    struct Worker {
        size_t start = 0, end = 0;
        std::vector<uint8_t> r0, r1, s, rs;  // rounds blocks of end - start bytes each
    };
    std::vector<Worker> w;
    decltype(measure::now()) t0;
    double ot_seconds = 0;
    std::vector<uint8_t> mine, theirs;  // one round's corrections: choice corrections, then masked values
    std::thread gen;                    // the random OTs (boolCOTMultRoundsBegin)
    double wait_seconds = 0;            // until the first round had them
    double t_corr = 0, t_xchg = 0, t_out = 0, t_between = 0;  // round phases, and the caller's work between rounds
    decltype(measure::now()) last_end;
    bool any_round = false;
    // fn(wid) on every worker with a byte range, in parallel
    template <typename F>
    void each(F fn) {
        std::vector<std::future<void>> fut;
        for (int wid = 0; wid < int(w.size()); ++wid)
            if (w[wid].start < w[wid].end)
                fut.push_back(pool->enqueue([&fn, wid] { fn(wid); }));
        for (auto& f : fut) f.get();
    }
};

BoolMultRounds* boolCOTMultRoundsBegin(uint64_t num_triples, int rounds, const std::string& ip, int port, int party,
                                       int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_triples, " x ", rounds,
               " BOOL COT MULT triples in dependent rounds (threads: ", threads, ")");
    require_tuple_count_multiple_of_8(num_triples);
    auto* h      = new BoolMultRounds;
    h->party     = party;
    h->rounds    = rounds;
    h->threads   = threads;
    h->num_bytes = num_triples / 8;
    h->keys      = &Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);
    h->t0        = measure::now();
    h->keys->get_ios(threads);
    h->keys->ensure_ot(num_triples * uint64_t(rounds));
    const int ot_threads = h->keys->ot_workers();  // one worker per OT pack, LaunchWorks' split of the bytes
    const uint64_t ot_call = h->keys->next_ot_call();
    h->pool = std::make_unique<gemini::ThreadPool>(ot_threads);
    h->w.resize(ot_threads);
    const size_t load = (h->num_bytes + ot_threads - 1) / ot_threads;
    for (int wid = 0; wid < ot_threads; ++wid) {
        h->w[wid].start = std::min<size_t>(size_t(wid) * load, h->num_bytes);
        h->w[wid].end   = std::min<size_t>(h->w[wid].start + load, h->num_bytes);
    }
    // all rounds' random OTs: as cot_multiply_shares, the reversed instance first, then the straight one; on a thread
    // of its own (the caller may do other work meanwhile; the first round waits)
    h->gen = std::thread([h, ot_call, rounds] {
    h->each([h, ot_call, rounds](int wid) {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        auto& W           = h->w[wid];
        const size_t nb   = W.end - W.start;
        const int64_t n   = int64_t(nb) * 8 * rounds;
        W.r0.resize(nb * rounds), W.r1.resize(nb * rounds), W.s.resize(nb * rounds), W.rs.resize(nb * rounds);
        const int cur_party = wid & 1 ? OTHER_PARTY(h->party) : h->party;
        const auto* otpack  = h->keys->get_otpack(wid);
        if (cur_party == emp::ALICE) {
            otpack->silent_ot_reversed->recv_rot_bits(W.rs.data(), W.s.data(), n);
            otpack->io->flush();
            otpack->silent_ot->send_rot_bits(W.r0.data(), W.r1.data(), n);
        } else {
            otpack->silent_ot_reversed->send_rot_bits(W.r0.data(), W.r1.data(), n);
            otpack->io->flush();
            otpack->silent_ot->recv_rot_bits(W.rs.data(), W.s.data(), n);
        }
        otpack->io->flush();
    });
    h->ot_seconds = Utils::to_sec(Utils::time_diff(h->t0));
    });
    return h;
}

void boolCOTMultRound(BoolMultRounds* h, int round, const uint8_t* a, const uint8_t* b, uint8_t* c) {
    // The corrections of all workers' ranges into one buffer (on the pool), exchanged over A2B_ROUND_CHANNELS channels
    // (default 4: the bytes per round are few, so many small messages mostly wait for their slowest one), then the
    // outputs (on the pool). Which party sends first follows the channel's pack role, as cot_multiply_shares.
    static const int nch_env = getenv("A2B_ROUND_CHANNELS") ? std::max(1, atoi(getenv("A2B_ROUND_CHANNELS"))) : 4;
    if (h->gen.joinable()) {
        h->gen.join();
        h->wait_seconds = Utils::to_sec(Utils::time_diff(h->t0));
    }
    auto tr0 = measure::now();
    if (h->any_round)
        h->t_between += Utils::to_sec(Utils::time_diff(h->last_end));
    const size_t nb  = h->num_bytes;
    auto& mine       = h->mine;
    auto& theirs     = h->theirs;
    mine.resize(2 * nb), theirs.resize(2 * nb);
    h->each([h, round, a, b](int wid) {
        auto& W          = h->w[wid];
        const size_t n   = W.end - W.start, off = size_t(round) * n;
        const uint8_t *r0 = W.r0.data() + off, *r1 = W.r1.data() + off, *s = W.s.data() + off;
        uint8_t* e = h->mine.data() + W.start;           // choice corrections
        uint8_t* m = h->mine.data() + h->num_bytes + W.start;  // masked values
        for (size_t i = 0; i < n; ++i) e[i] = s[i] ^ a[W.start + i];
        for (size_t i = 0; i < n; ++i) m[i] = b[W.start + i] ^ r0[i] ^ r1[i];
    });
    auto tr1 = measure::now();
    h->t_corr += Utils::to_sec(Utils::time_diff(tr0));
    const int nch = std::min<int>(nch_env, int(h->w.size()));
    std::vector<std::future<void>> fut;
    for (int ch = 0; ch < nch; ++ch)
        fut.push_back(h->pool->enqueue([h, ch, nch, nb] {
            const size_t lo = 2 * nb * ch / nch, hi = 2 * nb * (ch + 1) / nch;
            auto* io = h->keys->get_otpack(ch)->io;
            if ((ch & 1 ? OTHER_PARTY(h->party) : h->party) == emp::ALICE) {
                io->send_data(h->mine.data() + lo, hi - lo);
                io->flush();
                io->recv_data(h->theirs.data() + lo, hi - lo);
            } else {
                io->recv_data(h->theirs.data() + lo, hi - lo);
                io->send_data(h->mine.data() + lo, hi - lo);
                io->flush();
            }
        }));
    for (auto& f : fut) f.get();
    auto tr2 = measure::now();
    h->t_xchg += Utils::to_sec(Utils::time_diff(tr1));
    h->each([h, round, a, b, c](int wid) {
        auto& W          = h->w[wid];
        const size_t n   = W.end - W.start, off = size_t(round) * n;
        const uint8_t *r0 = W.r0.data() + off, *r1 = W.r1.data() + off, *rs = W.rs.data() + off;
        const uint8_t* tc = h->theirs.data() + W.start;
        const uint8_t* tm = h->theirs.data() + h->num_bytes + W.start;
        for (size_t i = 0; i < n; ++i) {
            const uint8_t ai = a[W.start + i], bi = b[W.start + i];
            const uint8_t rcv_mul = rs[i] ^ (ai & tm[i]);  // cot_multiply_shares
            const uint8_t snd_mul = (~tc[i] & r0[i]) ^ (tc[i] & r1[i]);
            c[W.start + i]        = (ai & bi) ^ rcv_mul ^ snd_mul;
        }
    });
    h->t_out += Utils::to_sec(Utils::time_diff(tr2));
    h->last_end  = measure::now();
    h->any_round = true;
}

void boolCOTMultRoundsEnd(BoolMultRounds* h) {
    if (h->gen.joinable())
        h->gen.join();
    const double sec = Utils::to_sec(Utils::time_diff(h->t0));
    auto** ios       = h->keys->get_ios(h->threads);
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < h->threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter      = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", h->party - 1, ", PID", h->keys->get_io_offset(), ": Bool COT Mult rounds (",
               h->rounds, ")   s PRE: ", sec, " (random OTs ", h->ot_seconds, " s, first round waited until ", h->wait_seconds,
               " s; rounds: corrections ", h->t_corr, " s, exchange ", h->t_xchg, " s, outputs ", h->t_out,
               " s, between rounds ", h->t_between, " s)   MB SENT PRE: ", data_sent,
               "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BOOL_COT_MULT", data_sent, data_recv, sec);
    delete h;
}

void generateRandomMultiplicationsCheetah(uint8_t a[], uint8_t b[], uint64_t num_muls,
                                          const std::string& ip, int port, int party,
                                          int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_muls, " random multiplications (threads: ", threads, ")");
    require_tuple_count_multiple_of_8(num_muls);
    uint64_t num_bytes = (num_muls + 7) / 8;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();
    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(num_muls);
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();


    auto func = [&](int wid, int start, int end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
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
            Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", io_offset, ": Unknown party");
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Random mul   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Random mul   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("RANDOM_MUL", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[],
                                int bitlength [[maybe_unused]], uint64_t num_triples,
                                const std::string& ip, int port, int party, int threads,
                                TripleGenMethod method, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_triples, " BOOL triples (threads: ", threads, ")");
    require_tuple_count_multiple_of_8(num_triples);
    uint64_t num_bytes = (num_triples + 7) / 8;
    // std::atomic<int> setup = 0;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(num_triples);
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, int start, int end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        // auto start_setup = measure::now();

        // sci::OTPack<IO::NetIO> pack(ios + wid, 1, cur_party, true, false);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, keys.ot_io(wid), keys.get_otpack(wid), false);

        // setup += Utils::time_diff(start_setup);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / ot_threads));
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

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Bool triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Bool triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BOOL", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

    // Utils::log(Utils::Level::INFO, "P", party - 1, ": Setup time [s]: ",
    //            Utils::to_sec(setup.load())
    //                / (num_triples > static_cast<size_t>(threads) ? threads : num_triples));

    // keys.disconnect();
}

void generateBool3TupleCheetah(Beaver3Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party, int threads, unsigned io_offset,
                               bool party_local_bc) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_tuples, " BOOL3 tuples (threads: ", threads, ", party_local_bc: ", party_local_bc, ")");
    require_tuple_count_multiple_of_8(num_tuples);
    uint64_t num_bytes = (num_tuples + 7) / 8;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(2 * uint64_t(num_tuples));
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, int start, int end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        if (start >= end)
            return Code::OK;

        // The role alternation must match the pre-built per-wid OT packs; the field LOCALITY
        // is pinned to the REAL party instead (mode 1: P0 holds full b; mode 2: P1 holds full c).
        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        const int local_mode = !party_local_bc ? 0 : (party == emp::ALICE ? 1 : 2);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, keys.ot_io(wid), keys.get_otpack(wid), false);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / ot_threads / 8));
            Beaver3Tuples sub{
                tuples.a + total, tuples.b + total, tuples.c + total,
                tuples.ab + total, tuples.ac + total, tuples.bc + total,
                tuples.abc + total
            };
            switch (cur_party) {
                case emp::ALICE:
                    Server::tuple3_gen(triple_gen, sub, current * 8, local_mode);
                    break;
                case emp::BOB:
                    Client::tuple3_gen(triple_gen, sub, current * 8, local_mode);
                    break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Bool3 tuple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Bool3 tuple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BOOL3", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void generateBool4TupleCheetah(Beaver4Tuples tuples, uint64_t num_tuples, const std::string& ip,
                               int port, int party, int threads, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_tuples, " BOOL4 tuples (threads: ", threads, ")");
    require_tuple_count_multiple_of_8(num_tuples);
    uint64_t num_bytes = (num_tuples + 7) / 8;
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(3 * uint64_t(num_tuples));
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, int start, int end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        TripleGenerator<IO::NetIO> triple_gen(cur_party, keys.ot_io(wid), keys.get_otpack(wid), false);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / ot_threads / 8));
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

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_bytes, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Bool4 tuple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Bool4 tuple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BOOL4", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void generateArithTriplesCheetah(const UINT_TYPE a[], const UINT_TYPE b[], UINT_TYPE c[],
                                 int bitlength, uint64_t num_triples, const std::string& ip,
                                 int port, int party, int threads, Utils::PROTO proto,
                                 unsigned io_offset) {
    assert(bitlength == 32 && "[arith. triples] Unsupported bitlength");
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Generating ", num_triples,
               " ARITH triples ", Utils::proto_str(proto), " (threads: ", threads, ")");
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
                Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", io_offset, ": Unknown party");
            }
            }

            for (uint64_t i = 0; i < current; ++i) c[i + total] = static_cast<UINT_TYPE>(tmp_C(i));
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Arith triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Arith triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("ARITH", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

    keys.disconnect();
}

void generateFCTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, uint64_t com_dim, uint64_t dim2, int party,
                              int threads, Utils::PROTO proto, int factor,
                              const UINT_TYPE* prescribed) {
    auto meta = Utils::init_meta_fc(com_dim, dim2);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": Generating FC triples ", meta.input_shape, " x ",
               meta.weight_shape, " ", Utils::proto_str(proto), " (batch: ", batch, ", threads: ", threads, ")");

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

    if (proto == Utils::PROTO::AB2P) {
        const size_t fresh_period = tmp > 0 ? tmp : 1;  // weights repeat within each factor group
        switch (party) {
        case emp::ALICE: {
            Client::Protocol2Prescribed(ios, fc, meta, B, C, threads, batch, fresh_period);
            break;
        }
        case emp::BOB: {
            gemini::TensorShape out_shape({(int64_t)dim2});
            std::vector<Tensor<uint64_t>> P(batch);
            for (int i = 0; i < batch; ++i) {
                P[i].Reshape(out_shape);
                for (size_t j = 0; j < dim2; ++j)
                    P[i](j) = prescribed ? (uint64_t)prescribed[(size_t)i * dim2 + j] : 0;
            }
            Server::Protocol2Prescribed(meta, ios, fc, A, P, threads, batch, fresh_period);
            for (int i = 0; i < batch; ++i) C[i] = P[i];  // output share IS the prescribed value
            break;
        }
        }
    } else {
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
    }

    for (size_t i = 0; i < C.size(); ++i)
        for (size_t j = 0; j < dim2; ++j) {
            c[i * dim2 + j] = C[i](j);
        }

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": FC triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": FC triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("FC", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

    delete[] ai;
    delete[] bi;
}

void generateConvTriplesCheetahWrapper(Keys<IO::NetIO>& keys, const UINT_TYPE* a,
                                       const UINT_TYPE* b, UINT_TYPE* c, Utils::ConvParm parm,
                                       int party, int threads, Utils::PROTO proto, int factor,
                                       bool is_shared_input, const UINT_TYPE* prescribed) {
#if USE_CONV_CUDA
    if (proto == Utils::PROTO::AB2 || proto == Utils::PROTO::AB) {
        auto start = measure::now();
        auto** ios = keys.get_ios(threads);
        TROY::conv2d(ios, OTHER_PARTY(party), a, b, c, parm.batchsize, parm.ic,
                     parm.ih, parm.iw, parm.fh, parm.fw, parm.n_filters, parm.stride, parm.padding,
                     true, factor, proto == Utils::PROTO::AB);
        // Same accounting as the CPU path (the counters would otherwise be charged to the next layer type)
        std::string unit;
        double data_sent = 0, data_recv = 0;
        for (int i = 0; i < threads; ++i) {
            data_sent += Utils::to_MB(ios[i]->counter, unit);
            data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
            ios[i]->counter      = 0;
            ios[i]->recv_counter = 0;
        }
        Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
                   ": CONV triple (GPU)   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
        accumulateTripleStat("CONV", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
        return;
    }
#endif
    auto meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                      parm.n_filters, parm.stride, parm.padding, is_shared_input);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": Generating CONV triples ", meta.ishape, " x ", meta.fshape,
               " x ", parm.n_filters, ", stride: ", parm.stride, ", pad: ", parm.padding, ", ",
               Utils::proto_str(proto), " (batch: ", parm.batchsize, ", threads: ", threads, ")");

    if (Utils::getOutDim(parm) == gemini::GetConv2DOutShape(meta)) {
        generateConvTriplesCheetah(keys, a, b, c, meta, parm.batchsize, party, threads, proto,
                                   factor, prescribed);
    } else {
        Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": Adding padding manually");

        std::vector<UINT_TYPE> ai;
        std::tuple<int, int> dim;

        dim = Utils::pad_zero(a, ai, parm.ic, parm.ih, parm.iw, parm.padding, parm.batchsize);

        parm.ih      = std::get<0>(dim);
        parm.iw      = std::get<1>(dim);
        parm.padding = 0;

        meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                     parm.n_filters, parm.stride, parm.padding);
        generateConvTriplesCheetah(keys, ai.data(), b, c, meta, parm.batchsize, party, threads,
                                   proto, factor, prescribed);
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

    auto time_ntt = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": CONV_NTT   s PRE: ", time_ntt);
    accumulateTripleStat("CONV_NTT", 0, 0, time_ntt);

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

    Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", keys.get_io_offset(),
               ": send/recv[s]: ", Utils::to_sec(Utils::time_diff(tmp)));

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

    Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", keys.get_io_offset(),
               ": computation[s]: ", Utils::to_sec(Utils::time_diff(tmp)));

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

    Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", keys.get_io_offset(),
               ": recv/send[s]: ", Utils::to_sec(Utils::time_diff(tmp)));

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

    Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", keys.get_io_offset(),
               ": decryption[s]: ", Utils::to_sec(Utils::time_diff(tmp)));

    auto time = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triple   s PRE: ", time - time_ntt);
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("CONV", data_sent, data_recv, time - time_ntt);
}

void generateConvTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                                UINT_TYPE* c, const gemini::HomConv2DSS::Meta& meta, int batch,
                                int party, int threads, Utils::PROTO proto, int factor,
                                const UINT_TYPE* prescribed) {
    auto start = measure::now();
    auto& conv = keys.get_conv();
    auto** ios = keys.get_ios(threads);

    // AB2P: BOB's encrypted-filter cache, valid while the weights stay the same (ac_batch)
    std::vector<std::vector<seal::Ciphertext>> enc_B_ct;

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
        if (proto == Utils::PROTO::AB2P) {
            const bool fresh_filters
                = ac_batch_size > 0 ? (cur_batch % ac_batch_size == 0) : (cur_batch == 0);
            switch (party) {
            case emp::ALICE: {
                time_ntt += Utils::to_sec(Utils::time_diff(start_ntt));
                result = Client::Protocol2Prescribed(ios, conv, meta, B, C, threads, fresh_filters);
                time_ntt += Utils::to_sec(result.encryption);
                break;
            }
            case emp::BOB: {
                auto out_shape = gemini::GetConv2DOutShape(meta);
                const size_t n_out = out_shape.num_elements();
                Tensor<uint64_t> presc(out_shape);
                for (size_t i = 0; i < n_out; ++i)
                    presc.data()[i]
                        = prescribed ? (uint64_t)prescribed[(size_t)cur_batch * n_out + i] : 0;
                time_ntt += Utils::to_sec(Utils::time_diff(start_ntt));
                result = Server::Protocol2Prescribed(meta, ios, conv, A, presc, enc_B_ct, threads,
                                                     fresh_filters);
                C = presc;  // this party's output share IS the prescribed value
                break;
            }
            }

            send_recv += Utils::to_sec(result.send_recv);
            compute += Utils::to_sec(result.cipher_op);
            recv_send += Utils::to_sec(result.serial);
            decode += Utils::to_sec(result.decryption);
            decode += Utils::to_sec(result.plain_op);

            if (result.ret != Code::OK) {
                Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(),
                           ": CONV (AB2P) failed: ", CodeMessage(result.ret));
            }

            for (long i = 0; i < C.NumElements(); ++i) c[i + C.NumElements() * cur_batch] = C.data()[i];
            continue;
        }
        switch (party) {
        case emp::ALICE: {
            Code c;
            if (cur_batch % ac_batch_size == 0) {
                enc_B.clear();
                if ((c = conv.encodeFilters(B, meta, enc_B, threads)) != Code::OK) {
                    Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(), ": Filters encoding failed: ", CodeMessage(c));
                }
                if ((c = conv.filtersToNtt(enc_B, threads)) != Code::OK) {
                    Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(), ": Filters to NTT failed: ", CodeMessage(c));
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
                        Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(),
                                   ": Filters encoding failed: ", CodeMessage(c));
                    }
                    if ((c = conv.filtersToNtt(enc_B, threads)) != Code::OK) {
                        Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(), ": Filters to NTT failed: ", CodeMessage(c));
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
            Utils::log(Utils::Level::ERROR, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV failed: ", CodeMessage(result.ret));
        }

        // Utils::print_results(result, 0, batch, threads);
        for (long i = 0; i < C.NumElements(); ++i) c[i + C.NumElements() * cur_batch] = C.data()[i];
    }

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV_NTT   s PRE: ", time_ntt);
    accumulateTripleStat("CONV_NTT", 0, 0, time_ntt);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": send/recv[s]: ", send_recv);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": compute[s]: ", compute);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": recv/send[s]: ", recv_send);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": decode[s]: ", decode);

    auto time_total = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": CONV triple   s PRE: ", time_total - time_ntt);
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("CONV", data_sent, data_recv, time_total - time_ntt);

    delete[] ai;
    delete[] bi;
}

void generateBNTriplesCheetah(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b,
                              UINT_TYPE* c, int batch, size_t num_ele, size_t h, size_t w,
                              int party, int threads, Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_bn(num_ele, h, w);
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": Generating BN triples ", meta.ishape, " x ", meta.vec_shape,
               " ", Utils::proto_str(proto), " (batch: ", batch, ", threads: ", threads, ")");

    auto start = measure::now();

    meta.is_shared_input = proto == Utils::PROTO::AB;
    auto& bn             = keys.get_bn();
    auto** ios           = keys.get_ios(threads);

    size_t ac_batch_size = batch / factor;
    // Slot-encoded BN (the layers where HomBNSS chooses it over one ciphertext per channel): all images at
    // once, in full ciphertexts; the scales are the same for every image of a lane
    const size_t hw = h * w, per_image = num_ele * hw;
    if (((hw + POLY_MOD - 1) / POLY_MOD) * num_ele >= ((per_image + POLY_MOD - 1) / POLY_MOD) * 3) {
        const size_t len = per_image * batch;
        Tensor<uint64_t> X({static_cast<long>(len)}), S({static_cast<long>(len)}), C;
        for (size_t bb = 0; bb < size_t(batch); ++bb)
            for (size_t ch = 0; ch < num_ele; ++ch)
                for (size_t p = 0; p < hw; ++p) {
                    const size_t i = bb * per_image + ch * hw + p;
                    X(i) = a != nullptr ? a[i] : 0;
                    S(i) = b != nullptr ? b[ch + num_ele * (bb / ac_batch_size)] : 0;
                }
        auto code = BN::vector_product(ios, bn, party == emp::ALICE, X, S, C, meta.is_shared_input,
                                       meta.target_base_mod, threads, proto);
        if (code != Code::OK)
            Utils::log(Utils::Level::ERROR, "BN triples failed: ", CodeMessage(code));
        for (size_t i = 0; i < len; ++i) c[i] = C(i);
        batch = 0; // done
    }
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

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": BN triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": BN triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BN", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void generateBNTriplesBatched(Keys<IO::NetIO>& keys, const std::vector<BNTripleLayer>& layers, int party,
                              int threads, Utils::PROTO proto, int factor) {
    // the slot-encoded layers (generateBNTriplesCheetah's choice) are concatenated, the others go alone
    std::vector<size_t> slot, offset;
    size_t total = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& L = layers[l];
        const size_t hw = L.h * L.w, per_image = L.num_ele * hw;
        if (((hw + POLY_MOD - 1) / POLY_MOD) * L.num_ele >= ((per_image + POLY_MOD - 1) / POLY_MOD) * 3) {
            slot.push_back(l);
            offset.push_back(total);
            total += per_image * L.batch;
        } else {
            generateBNTriplesCheetah(keys, L.a, L.b, L.c, L.batch, L.num_ele, L.h, L.w, party, threads, proto,
                                     factor);
        }
    }
    if (slot.empty())
        return;
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": Generating BN triples of ",
               slot.size(), " layers in one product (", total, " elements, ", Utils::proto_str(proto),
               ", threads: ", threads, ")");
    auto start = measure::now();
    auto& bn   = keys.get_bn();
    auto** ios = keys.get_ios(threads);

    Tensor<uint64_t> X({static_cast<long>(total)}), S({static_cast<long>(total)}), C;
    for (size_t k = 0; k < slot.size(); ++k) {
        const auto& L = layers[slot[k]];
        const size_t hw = L.h * L.w, per_image = L.num_ele * hw, ac_batch_size = L.batch / factor;
        for (size_t bb = 0; bb < size_t(L.batch); ++bb)
            for (size_t ch = 0; ch < L.num_ele; ++ch)
                for (size_t p = 0; p < hw; ++p) {
                    const size_t i = bb * per_image + ch * hw + p;
                    X(offset[k] + i) = L.a != nullptr ? L.a[i] : 0;
                    S(offset[k] + i) = L.b != nullptr ? L.b[ch + L.num_ele * (bb / ac_batch_size)] : 0;
                }
    }
    auto code = BN::vector_product(ios, bn, party == emp::ALICE, X, S, C, proto == Utils::PROTO::AB, PLAIN_MOD,
                                   threads, proto);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "BN triples failed: ", CodeMessage(code));
    for (size_t k = 0; k < slot.size(); ++k) {
        const auto& L = layers[slot[k]];
        const size_t n = L.num_ele * L.h * L.w * L.batch;
        for (size_t i = 0; i < n; ++i) L.c[i] = C(offset[k] + i);
    }

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": BN triple   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": BN triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("BN", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
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
    Utils::log(Utils::Level::INFO, "P", party - 1, ": time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    Utils::log(Utils::Level::INFO, "P", party - 1, ": data[", st, "]: ", Utils::to_MB(data, st));

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

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(uint64_t(num_input));
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    uint8_t* sel = new uint8_t[num_input];
    uint64_t* x  = new uint64_t[num_input];
    uint64_t* y  = new uint64_t[num_input];

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
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

    gemini::ThreadPool tpool(ot_threads); // one OT pack and channel per worker, as for the bool triples
    gemini::LaunchWorks(tpool, num_input, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": Multiplex   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": Multiplex   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("MULTIPLEX", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

#ifdef VERIFY
    if (party == emp::BOB) {
        ios[0]->send_data(sel, sizeof(*sel) * num_input);
        ios[0]->send_data(x32, sizeof(*x32) * num_input);
        ios[0]->send_data(y32, sizeof(*y32) * num_input);
        ios[0]->flush();
    } else {
        Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", io_offset, ": Verifying MULTIPLEX: ", num_input);
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
            Utils::log(Utils::Level::PASSED, "P", party - 1, ", PID", io_offset, ": MULTIPLEX: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "P", party - 1, ", PID", io_offset, ": MULTIPLEX: FAILED");
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
    keys.ensure_ot(uint64_t(num_triples));
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
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

    gemini::ThreadPool tpool(ot_threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    delete[] a;
    delete[] b;

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": OT   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": OT   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("OT", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

    keys.disconnect();
}

void generateCOT(int party, const UINT_TYPE* a, const uint8_t* b, UINT_TYPE* c,
                 const unsigned& num_triples, const std::string& ip, int port, int threads,
                 int io_offset) {
    Utils::log(Utils::Level::DEBUG, "P", party - 1, ", PID", io_offset, ": Generating ", num_triples, " COT triples");
    auto& keys = Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);

    auto start = measure::now();

    auto** ios = keys.get_ios(threads);
    keys.ensure_ot(uint64_t(num_triples));
    const int ot_threads = keys.ot_workers(); // one worker per OT pack
    const uint64_t ot_call = keys.next_ot_call();

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        emp::DetSeedScope det_scope(Keys<IO::NetIO>::ot_seed_tag(ot_call, wid), gemini::kSeeded);  // reproducible OT
        if (start >= end)
            return Code::OK;

        size_t n = end - start;
        auto* ot = keys.get_otpack(wid);
        // the OT packs of odd workers are set up with the parties swapped (see Keys): there ALICE's
        // sending instance, paired with BOB's receiving one, is silent_ot_reversed
        auto* silent = wid & 1 ? ot->silent_ot_reversed : ot->silent_ot;

        switch (party) {
        case emp::ALICE: {
            silent->send_cot(c + start, a + start, n, 32);
            for (size_t i = 0; i < n; ++i) {
                c[i + start] = -c[i + start] & moduloMask;
            }
            break;
        }
        case emp::BOB: {
            uint8_t* sel = new uint8_t[n];
            for (size_t i = 0; i < n; ++i) sel[i] = get_nth(b, start + i);

            silent->recv_cot(c + start, (bool*)sel, n, 32);
            delete[] sel;
            break;
        }
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(ot_threads); // one OT pack and channel per worker
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": COT   s PRE: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset, ": COT   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("COT", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));

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
            Utils::log(Utils::Level::PASSED, "P", party - 1, ", PID", io_offset, ": COT: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "P", party - 1, ", PID", io_offset, ": COT: FAILED");
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

    auto time_ntt = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": CONV_NTT   s PRE: ", time_ntt);
    accumulateTripleStat("CONV_NTT", 0, 0, time_ntt);

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
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triple   s PRE: ", time - time_ntt);
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < 2; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triple   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("CONV", data_sent, data_recv, time - time_ntt);
}

void generateConvTriplesPacked(Keys<IO::NetIO>& keys, const UINT_TYPE* a, const UINT_TYPE* b, UINT_TYPE* c,
                               Utils::ConvParm parm, int party, int threads, Utils::PROTO proto, int factor) {
    auto start = measure::now();
    auto** ios = keys.get_ios(threads);
    ConvLayout::strided_conv(a, b, c, parm.batchsize, parm.ic, parm.ih, parm.iw, parm.fh, parm.fw,
                             parm.n_filters, parm.stride, parm.padding, factor,
                             [&](const UINT_TYPE* x, const UINT_TYPE* w, UINT_TYPE* out, size_t bs,
                                 size_t ic, size_t ih, size_t iw, size_t kh, size_t kw) {
        keys.get_packed_conv().conv(ios, party, x, w, out, bs, ic, ih, iw, kh, kw, parm.n_filters,
                                    proto == Utils::PROTO::AB, threads);
    });
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < threads; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter      = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(),
               ": CONV triple (packed)   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("CONV", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void generateConvTriplesPackedBatch(Keys<IO::NetIO>& keys, const std::vector<Utils::ConvParm>& parms,
                                    UINT_TYPE** a, UINT_TYPE** b, UINT_TYPE* c, int party, int threads,
                                    Utils::PROTO proto, const std::function<void(size_t)>& ready,
                                    IO::NetIO** own_ios) {
    std::vector<size_t> batch(parms.size()), offset(parms.size() + 1, 0);
    for (size_t i = 0; i < parms.size(); i++) {
        const auto& p = parms[i];
        size_t nh = (p.ih + 2 * p.padding - p.fh) / p.stride + 1, nw = (p.iw + 2 * p.padding - p.fw) / p.stride + 1;
        batch[i]      = p.batchsize;
        offset[i + 1] = offset[i] + p.batchsize * p.n_filters * nh * nw;
    }
    if (threads < 4 && !own_ios) { // the pipeline takes 4 channels
        for (size_t i = 0; i < parms.size(); i++) {
            if (ready)
                ready(i);
            generateConvTriplesPacked(keys, a ? a[i] : nullptr, b ? b[i] : nullptr, c + offset[i], parms[i], party, threads,
                                      proto);
        }
        return;
    }
    auto start = measure::now();
    // own_ios: 4 channels of the caller's (alongside the OT packs, which keep the regular ones)
    auto** ios = own_ios ? own_ios : keys.get_ios(threads);
    const int nios = own_ios ? 4 : threads;
    keys.get_packed_conv().conv_pipelined(
        ios, party, batch,
        [&](size_t i) {
            if (ready)
                ready(i);
            const auto& p = parms[i];
            auto r = std::make_shared<ConvLayout::Reduced<UINT_TYPE>>(a ? a[i] : nullptr, b ? b[i] : nullptr, c + offset[i],
                                                                      p.batchsize, p.ic, p.ih, p.iw, p.fh, p.fw,
                                                                      p.n_filters, p.stride, p.padding);
            return PackedConv2D::Job{r->x, r->w, r->c, r->bs, r->ic, r->ih, r->iw, r->kh, r->kw, r->oc,
                                     [r] { r->finish(); }};
        },
        proto == Utils::PROTO::AB, threads);
    std::string unit;
    double data_sent = 0, data_recv = 0;
    for (int i = 0; i < nios; ++i) {
        data_sent += Utils::to_MB(ios[i]->counter, unit);
        data_recv += Utils::to_MB(ios[i]->recv_counter, unit);
        ios[i]->counter      = 0;
        ios[i]->recv_counter = 0;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", keys.get_io_offset(), ": CONV triples (packed, ",
               parms.size(), " layers pipelined)   MB SENT PRE: ", data_sent, "   MB RECEIVED PRE: ", data_recv);
    accumulateTripleStat("CONV", data_sent, data_recv, Utils::to_sec(Utils::time_diff(start)));
}

void printTripleStats(int party, unsigned io_offset) {
    if (g_triple_stats.empty())
        return;

    double total_sent = 0, total_recv = 0, total_time = 0;
    for (const auto& [type, e] : g_triple_stats) {
        Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
                   ": --TRIPLE_STATS (Aggregated)-- ", type,
                   "   MB SENT PRE: ", e.mb_sent,
                   "   MB RECEIVED PRE: ", e.mb_recv,
                   "   s PRE: ", e.time_s);
        total_sent += e.mb_sent;
        total_recv += e.mb_recv;
        total_time += e.time_s;
    }
    Utils::log(Utils::Level::INFO, "P", party - 1, ", PID", io_offset,
               ": --TRIPLE_STATS (Total)--",
               "   MB SENT PRE: ", total_sent,
               "   MB RECEIVED PRE: ", total_recv,
               "   s PRE: ", total_time);
}

void resetTripleStats() {
    g_triple_stats.clear();
}

void getTripleStat(const std::string& type, double& mb_sent, double& mb_recv, double& time_s) {
    auto it = g_triple_stats.find(type);
    TripleStatEntry e = it == g_triple_stats.end() ? TripleStatEntry{} : it->second;
    mb_sent = e.mb_sent, mb_recv = e.mb_recv, time_s = e.time_s;
}

} // namespace Iface
