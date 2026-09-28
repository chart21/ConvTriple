#ifndef ELEM_HPP_
#define ELEM_HPP_

#include "gemini/core/prg_party.h"
#include <algorithm>
#include <vector>

#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include <core/utils.hpp>

#include <io/send.hpp>

#include <ot/prg.h>

using std::vector;

constexpr int64_t slot_count = POLY_MOD;

int64_t neg_mod(int64_t val, int64_t mod) { return ((val % mod) + mod) % mod; }

vector<uint64_t> ideal_functionality(vector<uint64_t>& inArr, vector<uint64_t>& multArr) {
    vector<uint64_t> result(inArr.size(), 0ULL);

    for (size_t i = 0; i < inArr.size(); i++) {
        result[i] = multArr[i] * inArr[i];
    }
    return result;
}

static void asymmetric_encrypt_zero(const seal::SEALContext& context,
                                    const seal::PublicKey& public_key,
                                    const seal::parms_id_type parms_id, bool is_ntt_form,
                                    std::shared_ptr<seal::UniformRandomGenerator> prng,
                                    seal::Ciphertext& destination) {
    using namespace seal;
    using namespace seal::util;
    // We use a fresh memory pool with `clear_on_destruction' enabled
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

    auto& context_data        = *context.get_context_data(parms_id);
    auto& parms               = context_data.parms();
    auto& coeff_modulus       = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count        = parms.poly_modulus_degree();
    auto ntt_tables           = context_data.small_ntt_tables();
    size_t encrypted_size     = public_key.data().size();

    // Make destination have right size and parms_id
    // Ciphertext (c_0,c_1, ...)
    destination.resize(context, parms_id, encrypted_size);
    destination.is_ntt_form() = is_ntt_form;
    destination.scale()       = 1.0;

    // Generate u <-- R_3
    auto u(allocate_poly(coeff_count, coeff_modulus_size, pool));
    sample_poly_ternary(prng, parms, u.get());

    // c[j] = u * public_key[j]
    for (size_t i = 0; i < coeff_modulus_size; i++) {
        ntt_negacyclic_harvey_lazy(u.get() + i * coeff_count, ntt_tables[i]);
        for (size_t j = 0; j < encrypted_size; j++) {
            dyadic_product_coeffmod(u.get() + i * coeff_count,
                                    public_key.data().data(j) + i * coeff_count, coeff_count,
                                    coeff_modulus[i], destination.data(j) + i * coeff_count);

            // Addition with e_0, e_1 is in non-NTT form
            if (!is_ntt_form) {
                inverse_ntt_negacyclic_harvey(destination.data(j) + i * coeff_count, ntt_tables[i]);
            }
        }
    }

    // Generate e_j <-- chi
    // c[j] = public_key[j] * u + e[j]
#if USE_APPROX_RESHARE
    // NOTE(wen-jie) we skip e[0] here since e[0] is replaced by the secret sharing random.
    for (size_t j = 1; j < encrypted_size; j++)
#else
    for (size_t j = 0; j < encrypted_size; j++)
#endif
    {
        SEAL_NOISE_SAMPLER(prng, parms, u.get());
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            // Addition with e_0, e_1 is in NTT form
            if (is_ntt_form) {
                ntt_negacyclic_harvey(u.get() + i * coeff_count, ntt_tables[i]);
            }
            add_poly_coeffmod(u.get() + i * coeff_count, destination.data(j) + i * coeff_count,
                              coeff_count, coeff_modulus[i], destination.data(j) + i * coeff_count);
        }
    }
}

void set_poly_coeffs_uniform(uint64_t* poly, uint32_t bitlen,
                             std::shared_ptr<seal::UniformRandomGenerator> random,
                             std::shared_ptr<const seal::SEALContext::ContextData>& context_data) {
    assert(bitlen < 128 && bitlen > 0);
    auto& parms            = context_data->parms();
    auto& coeff_modulus    = parms.coeff_modulus();
    size_t coeff_count     = parms.poly_modulus_degree();
    size_t coeff_mod_count = coeff_modulus.size();
    uint64_t bitlen_mask   = (1ULL << (bitlen % 64)) - 1;

    seal::RandomToStandardAdapter engine(random);
    for (size_t i = 0; i < coeff_count; i++) {
        if (bitlen < 64) {
            uint64_t noise = (uint64_t(engine()) << 32) | engine();
            noise &= bitlen_mask;
            for (size_t j = 0; j < coeff_mod_count; j++) {
                poly[i + (j * coeff_count)]
                    = seal::util::barrett_reduce_64(noise, coeff_modulus[j]);
            }
        } else {
            uint64_t noise[2]; // LSB || MSB
            for (int j = 0; j < 2; j++) {
                noise[0] = (uint64_t(engine()) << 32) | engine();
                noise[1] = (uint64_t(engine()) << 32) | engine();
            }
            noise[1] &= bitlen_mask;
            for (size_t j = 0; j < coeff_mod_count; j++) {
                poly[i + (j * coeff_count)]
                    = seal::util::barrett_reduce_128(noise, coeff_modulus[j]);
            }
        }
    }
}

static void set_poly_coeffs_uniform(uint64_t* poly, int bitlen,
                                    std::shared_ptr<seal::UniformRandomGenerator> prng,
                                    const seal::EncryptionParameters& parms) {
    using namespace seal::util;
    if (bitlen < 0 || bitlen > 64) {
        LOG(WARNING) << "set_poly_coeffs_uniform invalid bitlen";
    }

    auto& coeff_modulus    = parms.coeff_modulus();
    size_t coeff_count     = parms.poly_modulus_degree();
    size_t coeff_mod_count = coeff_modulus.size();
    uint64_t bitlen_mask   = (1ULL << (bitlen % 64)) - 1;

    // sample random in [0, 2^bitlen) then convert it to the RNS form
    const size_t nbytes = mul_safe(coeff_count, sizeof(uint64_t));
    if (prng) {
        prng->generate(nbytes, reinterpret_cast<seal::seal_byte*>(poly));
    } else {
        auto _prng = parms.random_generator()->create();
        _prng->generate(nbytes, reinterpret_cast<seal::seal_byte*>(poly));
    }

    uint64_t* dst_ptr = poly + coeff_count;
    for (size_t j = 1; j < coeff_mod_count; ++j) {
        const uint64_t* src_ptr = poly;
        for (size_t i = 0; i < coeff_count; ++i, ++src_ptr) {
            *dst_ptr++ = barrett_reduce_64(*src_ptr & bitlen_mask, coeff_modulus[j]);
        }
    }

    dst_ptr = poly;
    for (size_t i = 0; i < coeff_count; ++i, ++dst_ptr) {
        *dst_ptr = barrett_reduce_64(*dst_ptr & bitlen_mask, coeff_modulus[0]);
    }
}

void flood_ciphertext(seal::Ciphertext& ct,
                      std::shared_ptr<const seal::SEALContext::ContextData>& context_data,
                      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    auto prng   = context_data->parms().random_generator()->create();
    auto& parms = context_data->parms();
    const int noise_len
        = context_data->total_coeff_modulus_bit_count() - parms.plain_modulus().bit_count() - 1;

    auto mempool = seal::MemoryManager::GetPool();
    auto random
        = seal::util::allocate_uint(ct.coeff_modulus_size() * ct.poly_modulus_degree(), mempool);

    set_poly_coeffs_uniform(random.get(), std::min(64, noise_len), prng, parms);

    size_t n            = ct.poly_modulus_degree();
    auto dst_ptr        = ct.data();
    auto rns_ptr        = random.get();
    auto& coeff_modulus = parms.coeff_modulus();
    for (size_t i = 0; i < ct.coeff_modulus_size(); i++) {
        seal::util::add_poly_coeffmod(rns_ptr, dst_ptr, n, coeff_modulus[i], dst_ptr);
        rns_ptr += n;
        dst_ptr += n;
    }

    // auto& parms            = context_data->parms();
    // auto& coeff_modulus    = parms.coeff_modulus();
    // size_t coeff_count     = parms.poly_modulus_degree();
    // size_t coeff_mod_count = coeff_modulus.size();

    // auto noise(seal::util::allocate_poly(coeff_count, coeff_mod_count, pool));
    // std::shared_ptr<seal::UniformRandomGenerator> random(parms.random_generator()->create());

    // set_poly_coeffs_uniform(noise.get(), noise_len - 1, random, context_data);
    // for (size_t i = 0; i < coeff_mod_count; i++) {
    //     seal::util::add_poly_coeffmod(noise.get() + (i * coeff_count),
    //                                   ct.data() + (i * coeff_count), coeff_count,
    //                                   coeff_modulus[i], ct.data() + (i * coeff_count));
    // }

    // set_poly_coeffs_uniform(noise.get(), noise_len - 1, random, context_data);
    // for (size_t i = 0; i < coeff_mod_count; i++) {
    //     seal::util::add_poly_coeffmod(noise.get() + (i * coeff_count),
    //                                   ct.data(1) + (i * coeff_count), coeff_count,
    //                                   coeff_modulus[i], ct.data(1) + (i * coeff_count));
    // }
}

void decode(seal::Plaintext& pt, std::vector<uint64_t>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) vec[i] = i >= pt.coeff_count() ? 0 : pt[i];
}

void vec2PolyBFV(const uint64_t* vec, size_t len, seal::Plaintext& pt, seal::Modulus plain_mod) {
    pt.parms_id() = seal::parms_id_zero; // foo SEAL when using BFV
    pt.resize(len);

    for (size_t i = 0; i < len; ++i) {
        pt[i] = vec[i];
    }
}

void encode(std::vector<uint64_t>& vec, seal::Plaintext& pt, seal::Modulus mod) {
    // pt.parms_id() = seal::parms_id_zero;
    vec2PolyBFV(vec.data(), vec.size(), pt, mod);
}

template <class PKEY>
void elemwise_product_ab2(seal::SEALContext* context, IO::NetIO* io, seal::Encryptor* encryptor,
                          seal::Decryptor* decryptor, int32_t size, uint64_t* inArr,
                          uint64_t* multArr, uint64_t* outputArr, uint64_t prime_mod, int party,
                          PKEY pkey) {
    using namespace seal;
    auto encoder   = new BatchEncoder(*context);
    auto evaluator = new Evaluator(*context);

    int num_ct = ceil(float(size) / slot_count);

    if (party == emp::BOB) {
        vector<Ciphertext> ct(num_ct);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j] = inArr[j + offset] % prime_mod;
            }
            encoder->encode(tmp_vec, tmp_pt);
            encryptor->encrypt_symmetric(tmp_pt, ct[i]);
        }
        IO::send_encrypted_vector(*io, ct);
        io->flush();

        vector<Ciphertext> enc_result;
        IO::recv_encrypted_vector(*io, *context, enc_result);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            decryptor->decrypt(enc_result[i], tmp_pt);
            encoder->decode(tmp_pt, tmp_vec);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                outputArr[j + offset] = tmp_vec[j];
            }
        }

#ifdef VERIFY
        Utils::log(Utils::Level::DEBUG, "VERIFIYING ELEM. MULT");
        std::vector<uint64_t> a(size);
        std::vector<uint64_t> b(size);
        std::vector<uint64_t> R(size);
        io->recv_data(a.data(), size * sizeof(uint64_t));
        io->recv_data(b.data(), size * sizeof(uint64_t));
        io->recv_data(R.data(), size * sizeof(uint64_t));

        for (int i = 0; i < size; ++i) {
            a[i] += inArr[i];
        }
        auto result = ideal_functionality(a, b);
        bool passed = true;
        for (int i = 0; i < size; i++) {
            if (((outputArr[i] + R[i]) % prime_mod) != result[i] % prime_mod) {
                passed = false;
            }
        }
        if (passed)
            Utils::log(Utils::Level::PASSED, "ELEM. MULT.: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "ELEM. MULT.: FAILED");
#endif
    } else // party == ALICE
    {
        vector<Plaintext> multArr_pt(num_ct);
        vector<Plaintext> inArr_pt(num_ct);

        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            vector<uint64_t> tmp_vec2(slot_count, 0);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j] = multArr[j + offset] % prime_mod;
                if (inArr)
                    tmp_vec2[j] = inArr[j + offset] % prime_mod;
            }
            encoder->encode(tmp_vec, multArr_pt[i]);
            if (inArr)
                encoder->encode(tmp_vec2, inArr_pt[i]);
        }

        sci::PRG128 prg;
#if PRG_SEED != -1
        {
            uint64_t seed[2];
            gemini::party_seed(io->port, (uint64_t(3) << 32) | gemini::next_call(io->port), seed);
            prg.reseed(seed);
        }
#endif
        vector<Plaintext> enc_noise(num_ct);
        // vector<vector<uint64_t>> secret_share(num_ct, vector<uint64_t>(slot_count, 0));
        vector<uint64_t> secret_share(num_ct * slot_count, 0);
        for (int i = 0; i < num_ct; i++) {
            prg.random_mod_p<uint64_t>(secret_share.data() + i * slot_count, slot_count, prime_mod);
            std::vector<uint64_t> tmp(secret_share.data() + i * slot_count,
                                      secret_share.data() + (i + 1) * slot_count);
            encoder->encode(tmp, enc_noise[i]);
        }
        std::memcpy(outputArr, secret_share.data(), sizeof(uint64_t) * size);

        vector<Ciphertext> ct;
        IO::recv_encrypted_vector(*io, *context, ct);

        vector<Ciphertext> enc_result(num_ct);
        for (int i = 0; i < num_ct; i++) {
            if (inArr) {
                evaluator->add_plain(ct[i], inArr_pt[i], enc_result[i]);
                evaluator->multiply_plain_inplace(enc_result[i], multArr_pt[i]);
            } else {
                evaluator->multiply_plain(ct[i], multArr_pt[i], enc_result[i]);
            }
            evaluator->sub_plain_inplace(enc_result[i], enc_noise[i]);

            // evaluator->mod_switch_to_next_inplace(enc_result[i]);

            parms_id_type parms_id = enc_result[i].parms_id();
            std::shared_ptr<const SEALContext::ContextData> context_data
                = context->get_context_data(parms_id);

            flood_ciphertext(enc_result[i], context_data);

            evaluator->mod_switch_to_inplace(enc_result[i], context->last_parms_id());
            seal::Ciphertext zero;
            auto prng = context_data->parms().random_generator()->create();
            asymmetric_encrypt_zero(*context, pkey, enc_result[i].parms_id(),
                                    enc_result[i].is_ntt_form(), prng, zero);
            evaluator->add_inplace(enc_result[i], zero);
        }
        IO::send_encrypted_vector(*io, enc_result);
        io->flush();

#ifdef VERIFY
        io->send_data(inArr, size * sizeof(uint64_t));
        io->send_data(multArr, size * sizeof(uint64_t));
        io->send_data(outputArr, size * sizeof(uint64_t));
        io->flush();
#endif
    }
    delete encoder;
    delete evaluator;
}

template <class PKEY>
void elemwise_product_ab(seal::SEALContext* context, IO::NetIO* io, seal::Encryptor* encryptor,
                         seal::Decryptor* decryptor, int32_t size, uint64_t* inArr,
                         uint64_t* multArr, uint64_t* outputArr, uint64_t prime_mod, int party,
                         PKEY pkey) {
    using namespace seal;
    // auto encoder   = new BatchEncoder(*context);
    auto evaluator = new Evaluator(*context);
    auto mod       = context->first_context_data()->parms().plain_modulus();

    int num_ct = ceil(float(size) / slot_count);

    if (party == emp::BOB) {
        vector<Plaintext> inArr_pt(num_ct);
        vector<Plaintext> multArr_pt(num_ct);
        vector<Serializable<Ciphertext>> A1_ct(num_ct, encryptor->encrypt_zero());

        sci::PRG128 prg;
#if PRG_SEED != -1
        {
            uint64_t seed[2];
            gemini::party_seed(io->port, (uint64_t(3) << 32) | gemini::next_call(io->port), seed);
            prg.reseed(seed);
        }
#endif
        vector<uint64_t> secret_share(num_ct * slot_count, 0);
        vector<Plaintext> enc_noise(num_ct);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            // prg.random_mod_p<uint64_t>(secret_share.data() + i * slot_count, slot_count, 11);
            std::vector<uint64_t> tmp(secret_share.data() + i * slot_count,
                                      secret_share.data() + (i + 1) * slot_count);
            vector<uint64_t> tmp_vec(slot_count, 0);
            vector<uint64_t> tmp_vec2(slot_count, 0);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j]  = multArr[j + offset] % prime_mod;
                tmp_vec2[j] = inArr[j + offset] % prime_mod;
            }
            encode(tmp, enc_noise[i], mod);
            // encoder->encode(tmp, enc_noise[i]);
            encode(tmp_vec, multArr_pt[i], mod);
            // encoder->encode(tmp_vec, multArr_pt[i]);
            encode(tmp_vec2, inArr_pt[i], mod);
            // encoder->encode(tmp_vec2, inArr_pt[i]);
            A1_ct[i] = encryptor->encrypt_symmetric(inArr_pt[i]);
        }

        IO::send_encrypted_vector(*io, A1_ct);
        vector<Ciphertext> ct;
        IO::recv_encrypted_vector(*io, *context, ct);

        vector<Ciphertext> enc_result(num_ct);
        for (int i = 0; i < num_ct; i++) {
            evaluator->add_plain(ct[i], inArr_pt[i], enc_result[i]);
            // evaluator->multiply_plain_inplace(enc_result[i], multArr_pt[i]);
            // evaluator->sub_plain_inplace(enc_result[i], enc_noise[i]);

            parms_id_type parms_id = enc_result[i].parms_id();
            std::shared_ptr<const SEALContext::ContextData> context_data
                = context->get_context_data(parms_id);

            flood_ciphertext(enc_result[i], context_data);

            evaluator->mod_switch_to_inplace(enc_result[i], context->last_parms_id());
            seal::Ciphertext zero;
            auto prng = context_data->parms().random_generator()->create();
            asymmetric_encrypt_zero(*context, pkey, enc_result[i].parms_id(),
                                    enc_result[i].is_ntt_form(), prng, zero);
            evaluator->add_inplace(enc_result[i], zero);
        }

        IO::send_encrypted_vector(*io, enc_result);
        IO::recv_encrypted_vector(*io, *context, ct);

        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            decryptor->decrypt(ct[i], tmp_pt);
            decode(tmp_pt, tmp_vec);
            // encoder->decode(tmp_pt, tmp_vec);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                outputArr[j + offset] = (tmp_vec[j] + secret_share[j + offset]) % prime_mod;
            }
        }
#ifdef VERIFY
        Utils::log(Utils::Level::DEBUG, "VERIFIYING ELEM. MULT");
        std::vector<uint64_t> a(size);
        std::vector<uint64_t> b(size);
        std::vector<uint64_t> c(size);
        io->recv_data(a.data(), size * sizeof(uint64_t));
        io->recv_data(b.data(), size * sizeof(uint64_t));
        io->recv_data(c.data(), size * sizeof(uint64_t));

        for (int i = 0; i < size; ++i) {
            a[i] += inArr[i];
            b[i] += multArr[i];
        }
        auto result = ideal_functionality(a, b);
        bool passed = true;
        for (int i = 0; i < size; i++) {
            if (((outputArr[i] + c[i]) % prime_mod) != result[i] % prime_mod) {
                passed = false;
                std::cout << (outputArr[i] + c[i]) % prime_mod << ", " << result[i] % prime_mod
                          << "\n";
                break;
            }
        }
        if (passed)
            Utils::log(Utils::Level::PASSED, "ELEM. MULT.: PASSED");
        else
            Utils::log(Utils::Level::FAILED, "ELEM. MULT.: FAILED");
#endif
    } else // party == ALICE
    {
        vector<Plaintext> multArr_pt(num_ct);
        vector<Plaintext> inArr_pt(num_ct);
        vector<Serializable<Ciphertext>> A1_ct(num_ct, encryptor->encrypt_zero());

        sci::PRG128 prg;
#if PRG_SEED != -1
        {
            uint64_t seed[2];
            gemini::party_seed(io->port, (uint64_t(3) << 32) | gemini::next_call(io->port), seed);
            prg.reseed(seed);
        }
#endif
        vector<uint64_t> secret_share(num_ct * slot_count, 0);
        vector<Plaintext> enc_noise(num_ct);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            // prg.random_mod_p<uint64_t>(secret_share.data() + i * slot_count, slot_count, 11);
            std::vector<uint64_t> tmp(secret_share.data() + i * slot_count,
                                      secret_share.data() + (i + 1) * slot_count);
            vector<uint64_t> tmp_vec(slot_count, 0);
            vector<uint64_t> tmp_vec2(slot_count, 0);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j]  = multArr[j + offset] % prime_mod;
                tmp_vec2[j] = inArr[j + offset] % prime_mod;
            }
            encode(tmp, enc_noise[i], mod);
            // encoder->encode(tmp, enc_noise[i]);
            encode(tmp_vec, multArr_pt[i], mod);
            // encoder->encode(tmp_vec, multArr_pt[i]);
            encode(tmp_vec2, inArr_pt[i], mod);
            // encoder->encode(tmp_vec2, inArr_pt[i]);
            A1_ct[i] = encryptor->encrypt_symmetric(inArr_pt[i]);
        }

        vector<Ciphertext> ct;
        IO::recv_encrypted_vector(*io, *context, ct);
        IO::send_encrypted_vector(*io, A1_ct);

        vector<Ciphertext> enc_result(num_ct);
        for (int i = 0; i < num_ct; i++) {
            evaluator->add_plain(ct[i], inArr_pt[i], enc_result[i]);
            // evaluator->multiply_plain_inplace(enc_result[i], multArr_pt[i]);
            // evaluator->sub_plain_inplace(enc_result[i], enc_noise[i]);

            parms_id_type parms_id = enc_result[i].parms_id();
            std::shared_ptr<const SEALContext::ContextData> context_data
                = context->get_context_data(parms_id);

            flood_ciphertext(enc_result[i], context_data);

            evaluator->mod_switch_to_inplace(enc_result[i], context->last_parms_id());
            seal::Ciphertext zero;
            auto prng = context_data->parms().random_generator()->create();
            asymmetric_encrypt_zero(*context, pkey, enc_result[i].parms_id(),
                                    enc_result[i].is_ntt_form(), prng, zero);
            evaluator->add_inplace(enc_result[i], zero);
        }

        IO::recv_encrypted_vector(*io, *context, ct);
        IO::send_encrypted_vector(*io, enc_result);
        io->flush();

        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            decryptor->decrypt(ct[i], tmp_pt);
            decode(tmp_pt, tmp_vec);
            // encoder->decode(tmp_pt, tmp_vec);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                outputArr[j + offset] = (tmp_vec[j] + secret_share[j + offset]) % prime_mod;
            }
        }
#ifdef VERIFY
        io->send_data(inArr, size * sizeof(uint64_t), false);
        io->send_data(multArr, size * sizeof(uint64_t), false);
        io->send_data(outputArr, size * sizeof(uint64_t), false);
        io->flush();
#endif
    }
    // delete encoder;
    delete evaluator;
}

#endif
