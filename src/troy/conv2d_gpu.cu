#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "constants.hpp"
#include "core/conv_layout.hpp"
#include "conv2d_gpu.cuh"

#include <troy/troy.h>

namespace TROY {

static unsigned char ALICE = 1;
static unsigned char BOB   = 2;

#ifdef CONV_MAX_BATCH_SIZE
constexpr size_t MAX_BATCHSIZE = CONV_MAX_BATCH_SIZE;
#else
constexpr size_t MAX_BATCHSIZE = 16;
#endif

troy::HeContextPointer setup() {
    using namespace troy;
    size_t poly_mod   = POLY_MOD;
    size_t plain_mod  = PLAIN_MOD;
    SchemeType scheme = SchemeType::BFV;

    EncryptionParameters parms(scheme);
    // Both primes carry data (no key-switching prime), like the CPU path's SEAL context: 109 bits leave
    // room to flood the noise with 64 bits before the switch down to the 60-bit prime.
    parms.set_coeff_modulus(CoeffModulus::create(poly_mod, {60, 49}));
    parms.set_use_special_prime_for_encryption(true);
    parms.set_plain_modulus(plain_mod);
    parms.set_poly_modulus_degree(poly_mod);
#if PRG_SEED != -1
    return HeContext::create(parms, true, SecurityLevel::Classical128, PRG_SEED);
#else
    return HeContext::create(parms, true, SecurityLevel::Classical128);
#endif
}

namespace {

// One context, encoder and key pair per process, on the device when there is one. Creating them per
// convolution re-derived and re-uploaded the NTT tables and generated a new key for every layer.
struct HeState {
    troy::HeContextPointer he;
    std::unique_ptr<troy::linear::PolynomialEncoderRing2k<INT_TYPE>> encoder;
    troy::ParmsID first_parms_id;
    std::unique_ptr<troy::KeyGenerator> keygen;
    std::unique_ptr<troy::Encryptor> encryptor;
    std::unique_ptr<troy::Evaluator> evaluator;
    std::unique_ptr<troy::Decryptor> decryptor;
    // Encrypts zero under the other party's public key, from its own OS-seeded generator: the context's
    // generator is seeded with PRG_SEED, so both parties would draw the same re-randomization.
    std::unique_ptr<troy::Encryptor> rerandomizer;
    std::unique_ptr<troy::utils::RandomGenerator> prng;
    // For NTTs of 1x1 weight polynomials through smaller NTTs (encode_weights_1x1): per prime, the
    // exponent of the evaluation point psi^e of each NTT slot, and for each a = 1 .. log N - 1 the
    // size-(N >> a) NTT tables (on the device) and the slot of each of their evaluation points
    size_t n = 0;
    std::vector<troy::Modulus> moduli;
    std::vector<std::vector<uint64_t>> power;    // [prime][m] = psi^m, m < 2N
    std::vector<std::vector<uint32_t>> exponent; // [prime][slot]
    std::vector<troy::utils::Array<troy::utils::NTTTables>> small_tables;             // [a]
    std::vector<std::vector<std::unordered_map<uint64_t, uint32_t>>> small_slot_of; // [a][prime]
};

// evaluation points of the NTT of X, reduced: slot j of an NTT evaluates at probe[j]
std::vector<uint64_t> ntt_of_x(const std::vector<troy::Modulus>& moduli, size_t n,
                               troy::utils::ConstSlice<troy::utils::NTTTables> tables) {
    std::vector<uint64_t> probe(moduli.size() * n, 0);
    for (size_t l = 0; l < moduli.size(); l++) probe[l * n + 1] = 1;
    troy::utils::ntt_inplace_p(troy::utils::Slice<uint64_t>(probe.data(), probe.size(), false, nullptr), n, tables);
    for (size_t l = 0; l < moduli.size(); l++)
        for (size_t j = 0; j < n; j++) probe[l * n + j] %= moduli[l].value();
    return probe;
}

void setup_small_ntts(HeState& st) {
    auto cm = st.he->first_context_data().value()->parms().coeff_modulus();
    size_t L = cm.size(), n = st.he->first_context_data().value()->parms().poly_modulus_degree();
    int logn = __builtin_ctzll(n);
    st.n     = n;
    st.moduli.assign(cm.raw_pointer(), cm.raw_pointer() + L);
    troy::utils::ConstSlice<troy::Modulus> moduli(st.moduli.data(), L, false, nullptr);
    auto full  = troy::utils::NTTTables::create_ntt_tables(logn, moduli);
    auto probe = ntt_of_x(st.moduli, n, full.const_reference());
    st.power.assign(L, std::vector<uint64_t>(2 * n));
    st.exponent.assign(L, std::vector<uint32_t>(n));
    for (size_t l = 0; l < L; l++) {
        uint64_t q = st.moduli[l].value(), psi = full[l].root();
        std::unordered_map<uint64_t, uint32_t> exponent_of;
        st.power[l][0] = 1;
        for (size_t m = 1; m < 2 * n; m++) st.power[l][m] = uint64_t((unsigned __int128)st.power[l][m - 1] * psi % q);
        for (size_t m = 0; m < 2 * n; m++) exponent_of[st.power[l][m]] = m;
        for (size_t j = 0; j < n; j++) st.exponent[l][j] = exponent_of.at(probe[l * n + j]);
    }
    st.small_tables.resize(logn);
    st.small_slot_of.assign(logn, std::vector<std::unordered_map<uint64_t, uint32_t>>(L));
    for (int a = 1; a < logn; a++) {
        auto tables = troy::utils::NTTTables::create_ntt_tables(logn - a, moduli);
        auto points = ntt_of_x(st.moduli, n >> a, tables.const_reference());
        for (size_t l = 0; l < L; l++)
            for (size_t j = 0; j < (n >> a); j++) st.small_slot_of[a][l][points[l * (n >> a) + j]] = j;
        if (troy::utils::device_count() > 0) {
            for (size_t l = 0; l < L; l++) tables[l].to_device_inplace();
            tables.to_device_inplace();
        }
        st.small_tables[a] = std::move(tables);
    }
}

HeState& he_state() {
    // Never destroyed: device memory must not be released after the CUDA runtime has shut down.
    static HeState* state = [] {
        auto* st    = new HeState;
        st->he      = setup();
        st->encoder = std::make_unique<troy::linear::PolynomialEncoderRing2k<INT_TYPE>>(st->he, BIT_LEN);
        st->first_parms_id = st->encoder->context()->first_context_data_pointer()->parms_id();
        setup_small_ntts(*st);
        if (troy::utils::device_count() > 0) {
            st->he->to_device_inplace();
            st->encoder->to_device_inplace();
        } else {
            std::cerr << RED << "Couldn't find a GPU" << NC << "\n";
        }
        st->keygen    = std::make_unique<troy::KeyGenerator>(st->he);
        st->encryptor = std::make_unique<troy::Encryptor>(st->he);
        st->encryptor->set_secret_key(st->keygen->secret_key());
        st->evaluator = std::make_unique<troy::Evaluator>(st->he);
        st->decryptor = std::make_unique<troy::Decryptor>(st->he, st->keygen->secret_key());
        return st;
    }();
    return *state;
}

// The first convolution exchanges public keys, so that outputs can be re-randomized under the key of
// the party that decrypts them.
void exchange_public_keys(IO::NetIO** ios, int party) {
    HeState& st = he_state();
    if (st.rerandomizer)
        return;
    std::stringstream mine;
    st.keygen->create_public_key(false).save(mine, st.he);
    std::stringstream theirs;
    if (party == ALICE) {
        send(ios, mine);
        theirs = recv(ios);
    } else {
        theirs = recv(ios);
        send(ios, mine);
    }
    troy::PublicKey other = troy::PublicKey::load_new(theirs, st.he);
    if (troy::utils::device_count() > 0)
        other.to_device_inplace();
    st.rerandomizer = std::make_unique<troy::Encryptor>(st.he);
    st.rerandomizer->set_public_key(other);
    INT_TYPE seed[4];
    random_ring(seed, 4);
    st.prng = std::make_unique<troy::utils::RandomGenerator>(
        (__uint128_t(seed[0]) << 96) | (__uint128_t(seed[1]) << 64) | (uint64_t(seed[2]) << 32) | seed[3]);
}

// c0 += e mod each prime, for one uniform 64-bit e per coefficient
__global__ void flood_kernel(uint64_t* c0, const uint64_t* noise, const troy::Modulus* moduli, size_t n,
                             size_t moduli_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * moduli_count)
        return;
    const troy::Modulus& q = moduli[idx / n];
    uint64_t v             = c0[idx] + q.reduce(noise[idx % n]);
    c0[idx]                = v >= q.value() ? v - q.value() : v;
}

// Before a result goes back to the key owner, whose own ciphertext it was computed from: its c1 and its
// noise are functions of the weights. Add an encryption of zero under the key owner's public key (a fresh
// c1) and a uniform 64-bit noise term (hides the weight-dependent noise), like HomConv2DSS's
// flood_ciphertext. At the full modulus, so that the switch to the last prime scales both away.
void rerandomize(troy::linear::Cipher2d& y) {
    HeState& st = he_state();
    std::vector<troy::Ciphertext*> cts;
    for (auto& row : y.data())
        for (auto& ct : row) cts.push_back(&ct);
    if (cts.empty())
        return;
    auto context_data = st.he->get_context_data(cts[0]->parms_id()).value();
    auto moduli       = context_data->parms().coeff_modulus();
    size_t n = cts[0]->poly_modulus_degree(), L = cts[0]->coeff_modulus_size();
    bool device      = cts[0]->on_device();
    constexpr size_t chunk = 256; // ciphertexts per batch of zero encryptions
    std::vector<troy::Ciphertext> zeros;
    troy::utils::Array<uint64_t> noise(n * chunk, device);
    for (size_t i0 = 0; i0 < cts.size(); i0 += chunk) {
        size_t m = std::min(chunk, cts.size() - i0);
        zeros.assign(m, troy::Ciphertext());
        std::vector<troy::Ciphertext*> zp;
        std::vector<const troy::Ciphertext*> zc;
        for (auto& z : zeros) zp.push_back(&z), zc.push_back(&z);
        st.rerandomizer->encrypt_zero_asymmetric_batched(zp, cts[0]->parms_id(), st.prng.get());
        std::vector<troy::Ciphertext*> dst(cts.begin() + i0, cts.begin() + i0 + m);
        st.evaluator->add_inplace_batched(dst, zc);
        st.prng->fill_uint64s(noise.reference());
        for (size_t i = 0; i < m; i++) {
            uint64_t* c0          = dst[i]->poly(0).raw_pointer();
            const uint64_t* e     = noise.raw_pointer() + i * n;
            if (device) {
                size_t threads = n * L, block = troy::utils::KERNEL_THREAD_COUNT;
                flood_kernel<<<(threads + block - 1) / block, block>>>(c0, e, moduli.raw_pointer(), n, L);
            } else
                for (size_t j = 0; j < n * L; j++) {
                    const troy::Modulus& q = moduli[j / n];
                    uint64_t v             = c0[j] + q.reduce(e[j % n]);
                    c0[j]                  = v >= q.value() ? v - q.value() : v;
                }
        }
    }
    if (device)
        troy::utils::stream_sync_concrete();
}

// spread[p][i] of weight plaintext p = (output group, input group) of a chunk that starts at output
// channel o0, in the layout of Conv2dHelper::encode_weights: coefficient i lies in slot i / (hb*wb) of
// ci*co slots (output channel slot / ci, reversed input channel slot % ci) and holds the flipped kernel
// tap at position i % (hb*wb), or 0
__global__ void spread_weights_kernel(INT_TYPE* spread, const INT_TYPE* w, size_t o0, size_t oc, size_t ic,
                                      size_t kh, size_t kw, size_t hb, size_t wb, size_t ci, size_t co,
                                      size_t in_groups, size_t count, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= count * n)
        return;
    size_t p = idx / n, i = idx % n, block = hb * wb;
    size_t slot = i / block, pos = i % block, a = pos / wb, d = pos % wb;
    INT_TYPE v = 0;
    if (slot < ci * co && a < kh && d < kw) {
        size_t o = o0 + p / in_groups * co + slot / ci, c = p % in_groups * ci + ci - 1 - slot % ci;
        if (o < oc && c < ic)
            v = w[((o * ic + c) * kh + kh - 1 - a) * kw + kw - 1 - d];
    }
    spread[idx] = v;
}

// small[p][l][s] = weight of slot s of 1x1 weight plaintext p, lifted to prime l (centered)
__global__ void scatter_slots_kernel(uint64_t* small, const INT_TYPE* w, const troy::Modulus* moduli, size_t L,
                                     size_t n_small, size_t o0, size_t oc, size_t ic, size_t ci, size_t co,
                                     size_t in_groups, size_t count) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= count * L * n_small)
        return;
    size_t p = idx / (L * n_small), l = idx / n_small % L, slot = idx % n_small;
    uint64_t v = 0;
    if (slot < ci * co) {
        size_t o = o0 + p / in_groups * co + slot / ci, c = p % in_groups * ci + ci - 1 - slot % ci;
        if (o < oc && c < ic) {
            INT_TYPE u = w[o * ic + c];
            v          = u >> (sizeof(INT_TYPE) * 8 - 1) ? moduli[l].value() - uint64_t(INT_TYPE(-u)) : u;
        }
    }
    small[idx] = v;
}

// dst[p][l][j] = small[p][l][gather[l][j]], reduced
__global__ void gather_slots_kernel(uint64_t* const* dst, const uint64_t* small, const uint32_t* gather,
                                    const troy::Modulus* moduli, size_t L, size_t n, size_t n_small, size_t count) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= count * L * n)
        return;
    size_t p = idx / (L * n), l = idx / n % L, j = idx % n;
    dst[p][l * n + j] = moduli[l].reduce(small[(p * L + l) * n_small + gather[l * n + j]]);
}

// A 1x1 weight polynomial is V(X^B), one weight per slot of B = hb*wb coefficients. For B = 2^a m, m odd,
// X^B maps the N evaluation points of the NTT onto the N >> a points of a size-(N >> a) negacyclic NTT:
// NTT(W)[j] = NTT_small(V)[gather[j]], an NTT of size 64 (56x56 layers), 256 (28x28) or 1024 (14x14)
// instead of 4096 per weight plaintext. Empty when that does not apply.
std::optional<troy::linear::Plain2d> encode_weights_1x1(const troy::linear::Conv2dHelper& h, const INT_TYPE* w_dev,
                                                        size_t o0, size_t oc_total) {
    HeState& st = he_state();
    size_t B = h.image_height_block * h.image_width_block, n = st.n, L = st.moduli.size();
    int a = B ? __builtin_ctzll(B) : 0;
    if (h.kernel_height != 1 || h.kernel_width != 1 || a == 0 || (n >> a) < 2 || a >= int(st.small_tables.size()))
        return std::nullopt;
    size_t n_small = n >> a, ci = h.input_channel_block, co = h.output_channel_block;
    size_t in_groups = (h.input_channels + ci - 1) / ci, out_groups = (h.output_channels + co - 1) / co;
    size_t count = in_groups * out_groups;
    std::vector<uint32_t> gather(L * n);
    for (size_t l = 0; l < L; l++)
        for (size_t j = 0; j < n; j++)
            gather[l * n + j] = st.small_slot_of[a][l].at(st.power[l][(st.exponent[l][j] * B) & (2 * n - 1)]);
    auto gather_dev = troy::utils::Array<uint32_t>::create_uninitialized(gather.size(), true);
    gather_dev.copy_from_slice(troy::utils::ConstSlice<uint32_t>(gather.data(), gather.size(), false, nullptr));
    auto moduli = st.he->first_context_data().value()->parms().coeff_modulus();

    auto small = troy::utils::Array<uint64_t>::create_uninitialized(count * L * n_small, true);
    size_t block = troy::utils::KERNEL_THREAD_COUNT, threads = count * L * n_small;
    scatter_slots_kernel<<<(threads + block - 1) / block, block>>>(
        small.raw_pointer(), w_dev, moduli.raw_pointer(), L, n_small, o0, std::min(oc_total, o0 + h.output_channels),
        h.input_channels, ci, co, in_groups, count);
    troy::utils::ntt_inplace_ps(small.reference(), count, n_small, st.small_tables[a].const_reference());

    std::vector<troy::Plaintext> plain(count);
    std::vector<uint64_t*> dst(count);
    for (size_t p = 0; p < count; p++) {
        plain[p].to_device_inplace();
        plain[p].resize_rns(*st.he, st.first_parms_id, false);
        plain[p].is_ntt_form() = true;
        dst[p]                 = plain[p].data().raw_pointer();
    }
    auto dst_dev = troy::utils::Array<uint64_t*>::create_uninitialized(count, true);
    dst_dev.copy_from_slice(troy::utils::ConstSlice<uint64_t*>(dst.data(), count, false, nullptr));
    threads = count * L * n;
    gather_slots_kernel<<<(threads + block - 1) / block, block>>>(dst_dev.raw_pointer(), small.raw_pointer(),
                                                                   gather_dev.raw_pointer(), moduli.raw_pointer(), L, n,
                                                                   n_small, count);
    troy::linear::Plain2d out;
    for (size_t og = 0; og < out_groups; og++) {
        out.data().emplace_back();
        for (size_t g = 0; g < in_groups; g++) out.data().back().push_back(std::move(plain[og * in_groups + g]));
    }
    return out;
}

// The weight plaintexts of output channels [o0, o0 + h.output_channels), built on the device from the
// layer's weights (w_dev): the host-side spreads were N words per plaintext, mostly zeros, all uploaded
troy::linear::Plain2d encode_weights_on_device(const troy::linear::Conv2dHelper& h,
                                                const troy::linear::PolynomialEncoderRing2k<INT_TYPE>& encoder,
                                                const troy::Evaluator& evaluator, const INT_TYPE* w_dev,
                                                size_t o0, size_t oc_total) {
    if (auto small = encode_weights_1x1(h, w_dev, o0, oc_total))
        return std::move(*small);
    size_t n = h.slot_count, ci = h.input_channel_block, co = h.output_channel_block;
    size_t in_groups = (h.input_channels + ci - 1) / ci, out_groups = (h.output_channels + co - 1) / co;
    size_t count = in_groups * out_groups;
    auto spread  = troy::utils::Array<INT_TYPE>::create_uninitialized(count * n, true);
    size_t threads = count * n, block = troy::utils::KERNEL_THREAD_COUNT;
    spread_weights_kernel<<<(threads + block - 1) / block, block>>>(
        spread.raw_pointer(), w_dev, o0, std::min(oc_total, o0 + h.output_channels), h.input_channels,
        h.kernel_height, h.kernel_width, h.image_height_block, h.image_width_block, ci, co, in_groups, count, n);
    std::vector<troy::Plaintext> plain(count);
    std::vector<troy::Plaintext*> ptrs;
    troy::utils::ConstSliceVec<INT_TYPE> source;
    for (size_t p = 0; p < count; p++) {
        ptrs.push_back(&plain[p]);
        source.push_back(spread.const_slice(p * n, (p + 1) * n));
    }
    encoder.centralize_slice_batched(source, std::nullopt, ptrs);
    evaluator.transform_plain_to_ntt_inplace_batched(ptrs, ptrs[0]->parms_id());
    troy::linear::Plain2d out;
    for (size_t og = 0; og < out_groups; og++) {
        out.data().emplace_back();
        for (size_t g = 0; g < in_groups; g++) out.data().back().push_back(std::move(plain[og * in_groups + g]));
    }
    return out;
}

// A layer has ceil(ic/ci) * ceil(oc/co) weight plaintexts of a full polynomial each. The
// communication-optimal tiling uses whole-image tiles, where ci = co = 1 on large images: ic * oc
// plaintexts, more than a GPU holds for the wide ResNet layers. Encode and multiply them one
// output-channel chunk at a time. The chunks keep the layer's tiling, so the encrypted input and the
// output layout stay the same, and each chunk reuses the previous chunk's device memory.
troy::linear::Cipher2d conv_by_output_chunks(const troy::linear::Conv2dHelper& helper,
                                             const troy::Evaluator& evaluator,
                                             const troy::linear::PolynomialEncoderRing2k<INT_TYPE>& encoder,
                                             const troy::linear::Cipher2d& x, const INT_TYPE* w,
                                             size_t ic, size_t oc, size_t kh, size_t kw) {
    constexpr size_t max_plaintexts = 4096; // about 256 MiB of NTT-form weights per chunk
    size_t co        = helper.output_channel_block;
    size_t in_groups = (ic + helper.input_channel_block - 1) / helper.input_channel_block;
    size_t chunk_oc  = std::max<size_t>(1, max_plaintexts / in_groups) * co;
    troy::linear::Cipher2d y;
    bool device = encoder.on_device();
    troy::utils::Array<INT_TYPE> w_dev;
    if (device) {
        w_dev = troy::utils::Array<INT_TYPE>::create_uninitialized(oc * ic * kh * kw, true);
        w_dev.copy_from_slice(troy::utils::ConstSlice<INT_TYPE>(w, oc * ic * kh * kw, false, nullptr));
    }
    for (size_t o0 = 0; o0 < oc; o0 += chunk_oc) {
        troy::linear::Conv2dHelper h = helper; // same blocks
        h.output_channels = std::min(oc - o0, chunk_oc);
        auto yc = h.conv2d(evaluator, x,
                           device ? encode_weights_on_device(h, encoder, evaluator, w_dev.raw_pointer(), o0, oc)
                                  : h.encode_weights_ring2k(encoder, w + o0 * ic * kh * kw, std::nullopt));
        if (o0 == 0) {
            y = std::move(yc);
            for (auto& row : y.data()) row.reserve((oc + co - 1) / co);
        } else
            for (size_t b = 0; b < yc.data().size(); b++)
                for (auto& cipher : yc.data()[b]) y.data()[b].push_back(std::move(cipher));
    }
    return y;
}

} // namespace

void conv2d(IO::NetIO** ios, int party, const INT_TYPE* a, const INT_TYPE* b, INT_TYPE* c,
            size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
            size_t stride, size_t padding, bool mod_switch, int factor, bool is_ab) {
    auto start = measure::now();
    exchange_public_keys(ios, party);

    // The HE routines below compute stride-1 convolutions; ConvLayout pads and splits strided ones.
    ConvLayout::strided_conv(a, b, c, bs, ic, ih, iw, kh, kw, oc, stride, padding, factor,
                             [&](const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* out, size_t bs,
                                 size_t ic, size_t ih, size_t iw, size_t kh, size_t kw) {
#if REVERSE_GPU == 0
        if (is_ab)
            conv2d_ab(ios, party, x, w, out, bs, ic, ih, iw, kh, kw, oc, 1, mod_switch);
        else
            conv2d_ab2(ios, party, x, w, out, bs, ic, ih, iw, kh, kw, oc, 1, mod_switch);
#else
        if (is_ab)
            conv2d_ab_reverse(ios, party, x, w, out, bs, ic, ih, iw, kh, kw, oc, 1, mod_switch);
        else
            conv2d_ab2_reverse(ios, party, x, w, out, bs, ic, ih, iw, kh, kw, oc, 1, mod_switch);
#endif
    });

    double time = std::chrono::duration<double, std::milli>(measure::now() - start).count();

    std::cerr << "P" << party - 1 << ": CONV triple time + NTT[s]: " << time / 1000.0 << "\n";
    std::cerr << "P" << party - 1
              << ": CONV triple data[MiB]: " << (1.0 * ios[0]->counter) / (1 << 20) << "\n";
}

void conv2d_dummy(IO::NetIO** ios, int party, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                  size_t kw, size_t oc, size_t stride, size_t padding, bool mod_switch) {
    vector<INT_TYPE> x = random_polynomial(bs * ic * ih * iw, PLAIN_MOD);
    vector<INT_TYPE> w = random_polynomial(oc * ic * kh * kw, PLAIN_MOD);

    size_t oh = dim(ih, kh, stride, padding);
    size_t ow = dim(iw, kw, stride, padding);
    vector<INT_TYPE> c(bs * oc * oh * ow);

    conv2d(ios, party, x.data(), w.data(), c.data(), bs, ic, ih, iw, kh, kw, oc, stride, padding,
           mod_switch);
}

void conv2d_ab2(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                size_t stride, bool mod_switch) {
    using namespace troy;
    HeState& st  = he_state();
    auto& he     = st.he;
    auto& encoder = *st.encoder;

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    const Encryptor& encryptor = *st.encryptor;
    const Evaluator& evaluator = *st.evaluator;
    const Decryptor& decryptor = *st.decryptor;

    vector<INT_TYPE> R = random_polynomial(bs * oc * oh * ow);

    [[maybe_unused]] size_t size = 0;
    for (size_t cur = 0; cur < bs;) {
        auto batch_size = std::min(bs - cur, MAX_BATCHSIZE);
        linear::Conv2dHelper helper(batch_size, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptLeft);
        auto x_offset = ih * iw * ic * cur;
        auto r_offset = oh * ow * oc * cur;

        if (party == ALICE) {
            linear::Cipher2d x_encrypted
                = helper.encrypt_inputs_ring2k(encryptor, encoder, x + x_offset, std::nullopt);
            std::stringstream x_serialized;
            x_encrypted.save(x_serialized, he);
            send(ios, x_serialized);

            auto y_serialized = recv(ios);
            auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
            vector<INT_TYPE> y_decrypted
                = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
            size = bs
                   * apply_stride(c, y_decrypted.data(), stride, batch_size, ic, ih, iw, kh, kw, oc,
                                  cur);
        } else {
            linear::Plain2d x_encoded;
            if (x)
                x_encoded = helper.encode_inputs_ring2k(encoder, x + x_offset, std::nullopt, true);

            linear::Plain2d R_encoded
                = helper.encode_outputs_ring2k(encoder, R.data() + r_offset, std::nullopt);

            auto stream      = recv(ios);
            auto x_encrypted = linear::Cipher2d::load_new(stream, he);

            if (x)
                x_encrypted.add_plain_inplace(evaluator, x_encoded);

            linear::Cipher2d y_encrypted
                = conv_by_output_chunks(helper, evaluator, encoder, x_encrypted, w, ic, oc, kh, kw);
            y_encrypted.sub_plain_inplace(evaluator, R_encoded);
            rerandomize(y_encrypted);
            if (mod_switch)
                y_encrypted.mod_switch_to_next_inplace(evaluator);

            std::stringstream y_serialized;
            helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
            send(ios, y_serialized);
            size = bs
                   * apply_stride(c, R.data() + r_offset, stride, batch_size, ic, ih, iw, kh, kw,
                                  oc, cur);
        }
        cur += batch_size;
    }

#ifdef VERIFY
    if (party == ALICE) {
        std::cout << PURPLE << "Verifying CONV" << NC << "\n";
        size_t nh = dim(ih, kh, stride, 0);
        size_t nw = dim(iw, kw, stride, 0);
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> w2(oc * ic * kh * kw);
        std::vector<INT_TYPE> R(bs * oc * nh * nw);

        ios[0]->recv_data(x2.data(), bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(R.data(), R.size() * sizeof(INT_TYPE));

        add_inplace(R, c, PLAIN_MOD);
        add_inplace(x2, x, PLAIN_MOD);
        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w2.data(), PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
        }
    } else {
        if (x)
            ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        else {
            std::vector<INT_TYPE> zeros(bs * ic * ih * iw, 0);
            ios[0]->send_data(zeros.data(), bs * ic * ih * iw * sizeof(INT_TYPE));
        }
        ios[0]->send_data(w, oc * ic * kw * kh * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
}

void conv2d_ab(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
               size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
               size_t stride, bool mod_switch) {
    using namespace troy;
    HeState& st  = he_state();
    auto& he     = st.he;
    auto& encoder = *st.encoder;

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    const Encryptor& encryptor = *st.encryptor;
    const Evaluator& evaluator = *st.evaluator;
    const Decryptor& decryptor = *st.decryptor;

    [[maybe_unused]] size_t size = 0;
    for (size_t cur = 0; cur < bs;) {
        auto batch_size = std::min(bs - cur, MAX_BATCHSIZE);
        linear::Conv2dHelper helper(batch_size, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptLeft);
        auto x_offset = ih * iw * ic * cur;

        vector<INT_TYPE> R = random_polynomial(batch_size * oc * oh * ow);
        linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

        linear::Cipher2d x_encrypted
            = helper.encrypt_inputs_ring2k(encryptor, encoder, x + x_offset, std::nullopt);

        std::stringstream x_serialized;
        x_encrypted.save(x_serialized, he);

        std::stringstream received_x_serialized;
        if (party == ALICE) {
            send(ios, x_serialized);
            received_x_serialized = recv(ios);
        } else {
            received_x_serialized = recv(ios);
            send(ios, x_serialized);
        }

        auto other_x_encrypted = linear::Cipher2d::load_new(received_x_serialized, he);
        // conv(x_other + x_own, w_own): the own-share term x_own * w_own belongs to the triple too
        other_x_encrypted.add_plain_inplace(
            evaluator, helper.encode_inputs_ring2k(encoder, x + x_offset, std::nullopt, true));

        linear::Cipher2d y_encrypted
            = conv_by_output_chunks(helper, evaluator, encoder, other_x_encrypted, w, ic, oc, kh, kw);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);
        rerandomize(y_encrypted);
        if (mod_switch)
            y_encrypted.mod_switch_to_next_inplace(evaluator);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);

        std::stringstream received_y_serialized;
        if (party == ALICE) {
            send(ios, y_serialized);
            received_y_serialized = recv(ios);
        } else {
            received_y_serialized = recv(ios);
            send(ios, y_serialized);
        }

        auto other_y_encrypted = helper.deserialize_outputs(evaluator, received_y_serialized);
        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, other_y_encrypted);

        add_inplace(y_decrypted, R.data(), PLAIN_MOD);

        size = bs * apply_stride(c, y_decrypted.data(), stride, batch_size, ic, ih, iw, kh, kw, oc, cur);
        cur += batch_size;
    }

#ifdef VERIFY
    if (party == ALICE) {
        std::cout << PURPLE << "Verifying CONV" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> w2(oc * ic * kh * kw);
        std::vector<INT_TYPE> c2(size);

        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(c2.data(), c2.size() * sizeof(INT_TYPE));

        add_inplace(x2, x, PLAIN_MOD); // A0 + A1
        add_inplace(c2, c, PLAIN_MOD); // C0 + C1
        add_inplace(w2, w, PLAIN_MOD); // B0 + B1

        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w2.data(), PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(c2, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
        }
    } else {
        ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->send_data(w, oc * ic * kh * kw * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
}

// The output masks R must be uniform over the ring and unpredictable: rand() was never seeded and never
// exceeds 2^31, so every run used the same masks, with their top bit always 0.
std::vector<INT_TYPE> random_polynomial(size_t size, uint64_t max_value) {
    std::vector<INT_TYPE> result(size);
    random_ring(result.data(), size);
    if (max_value < (uint64_t(1) << (8 * sizeof(INT_TYPE))))
        for (auto& v : result) v %= max_value;
    return result;
}

vector<INT_TYPE> ideal_conv(const INT_TYPE* x, const INT_TYPE* w, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride) {
    size_t oh = (ih - kh) / stride + 1;
    size_t ow = (iw - kw) / stride + 1;

    vector<INT_TYPE> y_truth(bs * oc * oh * ow, 0);

    for (size_t b = 0; b < bs; b++) {
        for (size_t o = 0; o < oc; o++) {
            for (size_t i = 0; i < oh; i++) {
                for (size_t j = 0; j < ow; j++) {
                    for (size_t c = 0; c < ic; c++) {
                        for (size_t p = 0; p < kh; p++) {
                            for (size_t q = 0; q < kw; q++) {
                                add_mod_inplace(
                                    y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                    multiply_mod(x[b * ic * ih * iw + c * ih * iw
                                                   + (i * stride + p) * iw + (j * stride + q)],
                                                 w[o * ic * kh * kw + c * kh * kw + p * kw + q], t),
                                    t);
                            }
                        }
                    }
                }
            }
        }
    }
    return y_truth;
}

size_t apply_stride(INT_TYPE* dest, const INT_TYPE* x, const size_t& stride, const size_t& bs,
                    const size_t& ic, const size_t& ih, const size_t& iw, const size_t& kh,
                    const size_t& kw, const size_t& oc, const size_t batch_offset) {
    size_t oh  = (ih - kh) + 1;
    size_t ow  = (iw - kw) + 1;
    size_t nh  = (ih - kh) / stride + 1;
    size_t nw  = (iw - kw) / stride + 1;
    auto nsize = oc * nh * nw;

    for (size_t b = 0; b < bs; ++b) {
        for (size_t c = 0; c < oc; ++c) {
            for (size_t h = 0; h < oh; h += stride) {
                for (size_t w = 0; w < ow; w += stride) {
                    size_t out_h = h / stride;
                    size_t out_w = w / stride;
                    dest[(b + batch_offset) * nsize + c * nh * nw + out_h * nw + out_w]
                        = x[b * oc * oh * ow + c * oh * ow + h * ow + w];
                }
            }
        }
    }
    return nsize;
}

void add_inplace(std::vector<INT_TYPE>& a, const INT_TYPE* b, size_t t) {
    for (size_t i = 0; i < a.size(); ++i) add_mod_inplace(a[i], b[i], t);
}

void conv2d_ab2_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w,
                        INT_TYPE* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                        size_t kw, size_t oc, size_t stride, bool mod_switch) {
    using namespace troy;
    HeState& st  = he_state();
    auto& he     = st.he;
    auto& encoder = *st.encoder;
    const ParmsID parmsid = st.first_parms_id;

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper_enc(bs, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptRight);

    const Encryptor& encryptor = *st.encryptor;
    const Evaluator& evaluator = *st.evaluator;
    const Decryptor& decryptor = *st.decryptor;

    linear::Cipher2d w_encrypted;
    if (party == BOB) {
        w_encrypted = helper_enc.encrypt_weights_ring2k(encryptor, encoder, w, std::nullopt);
        std::stringstream w_serialized;
        w_encrypted.save(w_serialized, he);
        send(ios, w_serialized);
    } else {
        auto stream = recv(ios);
        w_encrypted = linear::Cipher2d::load_new(stream, he);
    }

    [[maybe_unused]] size_t size = 0;
    for (size_t cur = 0; cur < bs;) {
        auto batch_size = std::min(bs - cur, MAX_BATCHSIZE);
        linear::Conv2dHelper helper(batch_size, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptLeft);
        auto x_offset = ih * iw * ic * cur;

        if (party == BOB) {
            auto y_serialized = recv(ios);
            auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
            vector<INT_TYPE> y_decrypted
                = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
            size = bs
                   * apply_stride(c, y_decrypted.data(), stride, batch_size, ic, ih, iw, kh, kw, oc,
                                  cur);

        } else {
            vector<INT_TYPE> R = random_polynomial(batch_size * oc * oh * ow);

            linear::Plain2d x_encoded
                = helper.encode_inputs_ring2k(encoder, x + x_offset, parmsid);
            linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), parmsid);

            linear::Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
            y_encrypted.sub_plain_inplace(evaluator, R_encoded);
            rerandomize(y_encrypted);
            if (mod_switch)
                y_encrypted.mod_switch_to_next_inplace(evaluator);

            std::stringstream y_serialized;
            helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
            send(ios, y_serialized);

            size = bs * apply_stride(c, R.data(), stride, batch_size, ic, ih, iw, kh, kw, oc, cur);
        }
        cur += batch_size;
    }

#ifdef VERIFY
    if (party == BOB) {
        std::cout << PURPLE << "Verifying CONV REVERSED" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> R(size);

        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(R.data(), R.size() * sizeof(INT_TYPE));

        add_inplace(R, c, PLAIN_MOD);
        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w, PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
        }
    } else {
        ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
}

void conv2d_ab_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w,
                       INT_TYPE* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                       size_t kw, size_t oc, size_t stride, bool mod_switch) {
    using namespace troy;
    HeState& st  = he_state();
    auto& he     = st.he;
    auto& encoder = *st.encoder;
    const ParmsID parmsid = st.first_parms_id;

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper_enc(bs, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptRight);

    const Encryptor& encryptor = *st.encryptor;
    const Evaluator& evaluator = *st.evaluator;
    const Decryptor& decryptor = *st.decryptor;

    linear::Cipher2d w_encrypted;
    w_encrypted = helper_enc.encrypt_weights_ring2k(encryptor, encoder, w, std::nullopt);
    std::stringstream w_serialized;
    w_encrypted.save(w_serialized, he);

    if (party == ALICE) {
        send(ios, w_serialized);
        auto stream = recv(ios);
        w_encrypted = linear::Cipher2d::load_new(stream, he);
    } else {
        auto stream = recv(ios);
        send(ios, w_serialized);
        w_encrypted = linear::Cipher2d::load_new(stream, he);
    }

    [[maybe_unused]] size_t size = 0;
    for (size_t cur = 0; cur < bs;) {
        auto batch_size = std::min(bs - cur, MAX_BATCHSIZE);
        linear::Conv2dHelper helper(batch_size, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                    linear::MatmulObjective::EncryptLeft);
        auto x_offset = ih * iw * ic * cur;

        if (party == BOB) {
            vector<INT_TYPE> R = random_polynomial(batch_size * oc * oh * ow);

            linear::Plain2d x_encoded
                = helper.encode_inputs_ring2k(encoder, x + x_offset, parmsid);
            linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), parmsid);

            linear::Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
            y_encrypted.sub_plain_inplace(evaluator, R_encoded);
            rerandomize(y_encrypted);
            if (mod_switch)
                y_encrypted.mod_switch_to_next_inplace(evaluator);

            std::stringstream y_serialized;
            helper.serialize_outputs(evaluator, y_encrypted, y_serialized);

            auto recveived_y = recv(ios);
            send(ios, y_serialized);

            y_encrypted = helper.deserialize_outputs(evaluator, recveived_y);
            vector<INT_TYPE> y_decrypted
                = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);

            add_inplace(y_decrypted, R.data(), PLAIN_MOD);

            size = bs
                   * apply_stride(c, y_decrypted.data(), stride, batch_size, ic, ih, iw, kh, kw, oc,
                                  cur);

        } else {
            vector<INT_TYPE> R = random_polynomial(batch_size * oc * oh * ow);

            linear::Plain2d x_encoded
                = helper.encode_inputs_ring2k(encoder, x + x_offset, parmsid);
            linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), parmsid);

            linear::Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
            y_encrypted.sub_plain_inplace(evaluator, R_encoded);
            rerandomize(y_encrypted);
            if (mod_switch)
                y_encrypted.mod_switch_to_next_inplace(evaluator);

            std::stringstream y_serialized;
            helper.serialize_outputs(evaluator, y_encrypted, y_serialized);

            send(ios, y_serialized);
            y_serialized = recv(ios);
            y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
            vector<INT_TYPE> y_decrypted
                = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);

            add_inplace(R, y_decrypted.data(), PLAIN_MOD);

            size = bs * apply_stride(c, R.data(), stride, batch_size, ic, ih, iw, kh, kw, oc, cur);
        }
        cur += batch_size;
    }

#ifdef VERIFY
    if (party == BOB) {
        std::cout << PURPLE << "Verifying CONV REVERSED" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> w2(oc * ic * kh * kw);
        std::vector<INT_TYPE> c2(size);

        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(c2.data(), c2.size() * sizeof(INT_TYPE));

        add_inplace(x2, x, PLAIN_MOD); // A0 + A1
        add_inplace(c2, c, PLAIN_MOD); // C0 + C1
        add_inplace(w2, w, PLAIN_MOD); // Bß + B1

        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w2.data(), PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(c2, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
        }
    } else {
        ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->send_data(w, oc * ic * kh * kw * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
}

} // namespace TROY
