#ifndef OT_PROTO_HPP_
#define OT_PROTO_HPP_

#include "gemini/core/prg_party.h"
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <utility>

#include "constants.hpp"

#include "io/net_io_channel.hpp"

#include "ot/bit-triple-generator.h"
#include "ot/silent_ot.h"

#include "core/utils.hpp"
#include "ot/cheetah-ot_pack.h"

template <typename Datatype>
struct Beaver3TuplesD {
    Datatype* a;
    Datatype* b;
    Datatype* c;

    Datatype* ab;
    Datatype* ac;
    Datatype* bc;
    
    Datatype* abc;
};

template <typename Datatype>
struct Beaver4TuplesD {
    Datatype* a;
    Datatype* b;
    Datatype* c;
    Datatype* d;

    Datatype* ab;
    Datatype* ac;
    Datatype* ad;
    Datatype* bc;
    Datatype* bd;
    Datatype* cd;

    Datatype* abc;
    Datatype* abd;
    Datatype* acd;
    Datatype* bcd;

    Datatype* abcd;
};

using Beaver3Tuples = Beaver3TuplesD<uint8_t>;
using Beaver4Tuples = Beaver4TuplesD<uint8_t>; 

/// Packs bits in-place without resizing the buffer.
inline void pack_bool(uint8_t* bytes, size_t num_bits) {
    for (size_t i = 1; i < num_bits; ++i) {
        uint8_t tmp = 0;
        std::swap(tmp, bytes[i]);
        bytes[i / 8] |= tmp << (i % 8);
    }
}

/// Compute shares for c = a * b using correlated oblivious transfer
/// (a, b, c bit-packed; num_shares a multiple of 8). The random OTs come bit-packed from
/// send_rot_bits / recv_rot_bits - the same bits as send_ot_rm_rc<T> / recv_ot_rm_rc<T> with l = 1,
/// without two 16-byte blocks per OT and the byte-to-bit repacking.
template <typename IO>
void cot_multiply_shares(int party, const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* b, uint8_t* c, size_t num_shares) {
    const size_t num_bytes = (num_shares + 7) / 8;
    std::vector<uint8_t> r0(num_bytes), r1(num_bytes), s(num_bytes), rs(num_bytes),
        my_choice_corrections(num_bytes), my_masked_value(num_bytes), their_choice_corrections(num_bytes),
        their_masked_value(num_bytes);

    switch (party) {
        case emp::ALICE: {
            otpack->silent_ot_reversed->recv_rot_bits(rs.data(), s.data(), int64_t(num_shares));
            otpack->io->flush();
            otpack->silent_ot->send_rot_bits(r0.data(), r1.data(), int64_t(num_shares));
            break;
        }
        case emp::BOB: {
            otpack->silent_ot_reversed->send_rot_bits(r0.data(), r1.data(), int64_t(num_shares));
            otpack->io->flush();
            otpack->silent_ot->recv_rot_bits(rs.data(), s.data(), int64_t(num_shares));
            break;
        }
    }
    otpack->io->flush();

    for (size_t i = 0; i < num_bytes; ++i) my_choice_corrections[i] = s[i] ^ a[i];
    for (size_t i = 0; i < num_bytes; ++i) my_masked_value[i] = b[i] ^ r0[i] ^ r1[i];

    switch (party) {
        case emp::ALICE: {
            otpack->io->send_data(my_choice_corrections.data(), num_bytes);
            otpack->io->send_data(my_masked_value.data(), num_bytes);
            otpack->io->recv_data(their_choice_corrections.data(), num_bytes);
            otpack->io->recv_data(their_masked_value.data(), num_bytes);
            break;
        }
        case emp::BOB: {
            otpack->io->recv_data(their_choice_corrections.data(), num_bytes);
            otpack->io->recv_data(their_masked_value.data(), num_bytes);
            otpack->io->send_data(my_choice_corrections.data(), num_bytes);
            otpack->io->send_data(my_masked_value.data(), num_bytes);
            break;
        }
    }
    otpack->io->flush();

    for (size_t i = 0; i < num_bytes; ++i) {
        // For one OT direction, let x be the receiver's input bit and y be the sender's input bit. The
        // receiver has random choice s and selected OT mask rs; the sender has random masks r0 and r1.
        //
        // Receiver -> Sender: e = s ^ x
        // Sender -> Receiver: m = y ^ r0 ^ r1
        // Receiver output:    rs ^ (x & m)
        // Sender output:      e ? r1 : r0
        //
        // The matching receiver/sender outputs, held by opposite parties, share the cross term x & y:
        // when x is 0 their masks match and cancel; when x is 1 the receiver also XORs in m and the
        // sender uses the other mask, leaving y. This party's local rcv_mul and snd_mul come from
        // opposite OT directions, so c[i] also XORs in the local product a[i] & b[i].
        const uint8_t rcv_mul = rs[i] ^ (a[i] & their_masked_value[i]);
        const uint8_t snd_mul = (~their_choice_corrections[i] & r0[i]) ^ (their_choice_corrections[i] & r1[i]);
        c[i] = (a[i] & b[i]) ^ rcv_mul ^ snd_mul;
    }
}

/// Shares of x[p] * y (p < k) for k bit-packed shared values x[p] with one common factor y. One random OT
/// per direction carries all k products in its k message bits (send/recv_rot_bitplanes), where
/// cot_multiply_shares takes one OT per product: per element and party, 1 choice-correction bit plus k
/// masked bits. A party whose y share is a fresh random value can take it from its receiving OTs'
/// random choices instead (y is written) and sends no correction: {alice,bob}_y_from_choice, by role.
template <typename IO>
void cot_outer_multiply(int party, const sci::OTPack<IO>* otpack, uint8_t* const* x, int k, uint8_t* y,
                        uint8_t* const* out, size_t num_shares, bool alice_y_from_choice, bool bob_y_from_choice) {
    const size_t nb = (num_shares + 7) / 8;
    const bool my_choice = party == emp::ALICE ? alice_y_from_choice : bob_y_from_choice;
    const bool their_choice = party == emp::ALICE ? bob_y_from_choice : alice_y_from_choice;
    std::vector<uint8_t> rs(k * nb), r0(k * nb), r1(k * nb), s(nb);
    std::vector<uint8_t*> prs(k), pr0(k), pr1(k);
    for (int p = 0; p < k; p++) prs[p] = rs.data() + p * nb, pr0[p] = r0.data() + p * nb, pr1[p] = r1.data() + p * nb;

    switch (party) {  // same directions as cot_multiply_shares
        case emp::ALICE:
            otpack->silent_ot_reversed->recv_rot_bitplanes(prs.data(), s.data(), k, int64_t(num_shares));
            otpack->io->flush();
            otpack->silent_ot->send_rot_bitplanes(pr0.data(), pr1.data(), k, int64_t(num_shares));
            break;
        case emp::BOB:
            otpack->silent_ot_reversed->send_rot_bitplanes(pr0.data(), pr1.data(), k, int64_t(num_shares));
            otpack->io->flush();
            otpack->silent_ot->recv_rot_bitplanes(prs.data(), s.data(), k, int64_t(num_shares));
            break;
    }
    otpack->io->flush();

    // [choice correction] + k masked planes, one message each way
    const size_t my_len = (my_choice ? 0 : nb) + k * nb, their_len = (their_choice ? 0 : nb) + k * nb;
    std::vector<uint8_t> mine(my_len), theirs(their_len);
    uint8_t* w = mine.data();
    if (my_choice)
        std::memcpy(y, s.data(), nb);
    else
        for (size_t i = 0; i < nb; ++i) *w++ = s[i] ^ y[i];
    for (int p = 0; p < k; p++)
        for (size_t i = 0; i < nb; ++i) *w++ = x[p][i] ^ r0[p * nb + i] ^ r1[p * nb + i];
    if (party == emp::ALICE) {
        otpack->io->send_data(mine.data(), my_len);
        otpack->io->recv_data(theirs.data(), their_len);
    } else {
        otpack->io->recv_data(theirs.data(), their_len);
        otpack->io->send_data(mine.data(), my_len);
    }
    otpack->io->flush();

    // as in cot_multiply_shares: receiver output rs ^ (y & m), sender output e ? r1 : r0, plus the local product
    const uint8_t* e = their_choice ? nullptr : theirs.data();
    const uint8_t* m = theirs.data() + (their_choice ? 0 : nb);
    for (int p = 0; p < k; p++)
        for (size_t i = 0; i < nb; ++i) {
            const uint8_t ei = e ? e[i] : 0;
            const uint8_t rcv = rs[p * nb + i] ^ (y[i] & m[p * nb + i]);
            const uint8_t snd = (~ei & r0[p * nb + i]) ^ (ei & r1[p * nb + i]);
            out[p][i] = (x[p][i] & y[i]) ^ rcv ^ snd;
        }
}

static constexpr size_t LEN(const size_t& numTriple, const bool& packed) {
    return numTriple / (packed ? 8 : 1);
}

template <class T>
T bitmask(int l) {
    int bits = sizeof(T) * 8;

    if (l >= bits)
        return ~0ULL;
    else
        return ~(~0ULL << l);
}

namespace Server {

template <class Channel>
void triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                size_t numTriple, const bool& packed, TripleGenMethod method);

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed);

// party_local_mode: 0 = normal (all fields XOR-shared); 1 = this side is P0 (holds full b, zero
// c share); 2 = this side is P1 (zero b share, full c). See RESHARE_OPT_SIM.
template <class Channel>
void tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples,
                int party_local_mode = 0);

template <class Channel>
void tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples);

/// Generate random (a, u), (b, v) with ab = u ^ v. See also `Client::mul_gen`.
/// Implements Algorithm 1 from https://ia.cr/2013/552.
template <class IO>
void mul_gen(const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* u, size_t num_muls);

} // namespace Server

namespace Client {

template <class Channel>
void triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                size_t numTriple, const bool& packed, TripleGenMethod method);

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed);

// party_local_mode: 0 = normal (all fields XOR-shared); 1 = this side is P0 (holds full b, zero
// c share); 2 = this side is P1 (zero b share, full c). See RESHARE_OPT_SIM.
template <class Channel>
void tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples,
                int party_local_mode = 0);

template <class Channel>
void tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples);

/// Generate random (a, u), (b, v) with ab = u ^ v. See also `Server::mul_gen`.
/// Implements Algorithm 1 from https://ia.cr/2013/552.
template <class IO>
void mul_gen(const sci::OTPack<IO>* otpack, uint8_t* b, uint8_t* v, size_t num_muls);

} // namespace Client

template <class Channel>
void Server::triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                        size_t numTriple, const bool& packed, TripleGenMethod method) {

    if (packed) {
        numTriple *= 8;
    }

    Triple trips(a, b, c, numTriple, packed);
    triple.get(emp::ALICE, &trips, method);

#ifdef VERIFY
    size_t len = numTriple / 8;
    Utils::log(Utils::Level::DEBUG, "VERIFYING OT");
    Utils::log(Utils::Level::DEBUG, numTriple);

    uint8_t* a2 = new uint8_t[len];
    uint8_t* b2 = new uint8_t[len];
    uint8_t* c2 = new uint8_t[len];

    triple.io->recv_data(a2, sizeof(uint8_t) * len);
    triple.io->recv_data(b2, sizeof(uint8_t) * len);
    triple.io->recv_data(c2, sizeof(uint8_t) * len);

    bool same = true;
    for (size_t i = 0; i < len; ++i) {
        if (((b2[i] ^ b[i]) & (a[i] ^ a2[i])) != (c2[i] ^ c[i])) {
            same = false;
            std::cout << i << "\n";
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "OT: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "OT: FAILED");
    delete[] a2;
    delete[] b2;
    delete[] c2;
#endif
}

inline void pack_buffers(std::initializer_list<uint8_t*> sources, uint8_t * destination, size_t num_bytes_in_source) {
    size_t i = 0;
    for(uint8_t* source : sources) {
        std::copy_n(source, num_bytes_in_source, destination + i * num_bytes_in_source);
        ++i;
    }
}

inline void unpack_buffers(uint8_t * source, std::initializer_list<uint8_t*> destinations, size_t num_bytes_in_source){
    size_t i = 0;
    for(uint8_t * destination : destinations) {
        std::copy_n(source + i * num_bytes_in_source, num_bytes_in_source, destination);
        ++i;
    }
}

inline void require_tuple_count_multiple_of_8(size_t num_tuples) {
    if (num_tuples % 8 == 0)
        return;

    std::cerr << "tuple generation requires num_tuples to be divisible by 8, got "
              << num_tuples << "\n";
    std::exit(EXIT_FAILURE);
}

template <class Channel>
void Server::tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples,
                        int party_local_mode) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;

    sci::PRG128 prg;
#if PRG_SEED != -1
    uint64_t seed[2];
    gemini::party_seed(generator.io->port, (uint64_t(2) << 32) | gemini::next_call(generator.io->port), seed);
#else
    std::random_device r;
    const uint64_t seed[2] = {r(), r()};
#endif
    prg.reseed(seed);

    if (party_local_mode != 0) {
        // Party-local mask fields (RESHARE_OPT_SIM support): b is known ENTIRELY to P0 (peer
        // share = 0), c entirely to P1. The mode is pinned to the REAL party (the protocol role
        // may be alternated per thread chunk to match the pre-built OT packs). Only valid when
        // b/c exclusively mask wires whose value the respective party already knows (the
        // zero-add MSB adders' A2B input wires).
        prg.random_data(data.a, num_bytes);
        if (party_local_mode == 1) {  // we are P0: full b, zero c
            prg.random_data(data.b, num_bytes);
            std::memset(data.c, 0, num_bytes);
        } else {  // we are P1: zero b, full c
            std::memset(data.b, 0, num_bytes);
            prg.random_data(data.c, num_bytes);
        }
        cot_multiply_shares(emp::ALICE, generator.otpack, data.a, data.b, data.ab, num_tuples);
    } else {
        Server::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed,
                           triple_gen_method);
    }

    // ac, bc, abc: one OT per direction for all three. A fresh random c share is the OTs' choice (no
    // correction); only a party-local zero c (the real P0 in mode 1, ALICE role here) is corrected.
    uint8_t* const xs[3] = {data.a, data.b, data.ab};
    uint8_t* const outs[3] = {data.ac, data.bc, data.abc};
    cot_outer_multiply(emp::ALICE, generator.otpack, xs, 3, data.c, outs, num_tuples,
                       party_local_mode != 1, party_local_mode != 2);
}

template <class Channel>
void Server::tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Server::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    Server::triple_gen(generator, data.c, data.d, data.cd, num_bytes, packed, triple_gen_method);

    // the 9 products {a, b, ab} x {c, d, cd}: one OT per direction and factor y in {c, d, cd} (all
    // three in one call), carrying the three products with a, b and ab
    auto lhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(9 * num_bytes);
    pack_buffers({data.a, data.a, data.a, data.b, data.b, data.b, data.ab, data.ab, data.ab}, lhs.get(), num_bytes);
    pack_buffers({data.c, data.d, data.cd}, rhs.get(), num_bytes);
    uint8_t* const xs[3] = {lhs.get(), lhs.get() + 3 * num_bytes, lhs.get() + 6 * num_bytes};
    uint8_t* const outs[3] = {out.get(), out.get() + 3 * num_bytes, out.get() + 6 * num_bytes};
    cot_outer_multiply(emp::ALICE, generator.otpack, xs, 3, rhs.get(), outs, 3 * num_tuples, false, false);
    unpack_buffers(out.get(),
                   {data.ac, data.ad, data.acd, data.bc, data.bd, data.bcd, data.abc, data.abd,
                    data.abcd},
                   num_bytes);
}

template <class Channel>
void Server::RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed) {
    size_t len = LEN(numTriple, packed);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    triple_gen(triple, a, b, c, numTriple, packed);

    delete[] a;
    delete[] b;
    delete[] c;
}

template <class Channel>
void Client::triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                        size_t numTriple, const bool& packed, TripleGenMethod method) {

    if (packed) {
        numTriple *= 8;
    }

    Triple trips(a, b, c, numTriple, packed);
    triple.get(emp::BOB, &trips, method);

#ifdef VERIFY
    size_t len = numTriple / 8;
    triple.io->send_data(a, sizeof(uint8_t) * len, false);
    triple.io->send_data(b, sizeof(uint8_t) * len, false);
    triple.io->send_data(c, sizeof(uint8_t) * len, false);
    triple.io->flush();
#endif
}

template <class Channel>
void Client::RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed) {
    size_t len = LEN(numTriple, packed);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    triple_gen(triple, a, b, c, numTriple, packed);

    delete[] a;
    delete[] b;
    delete[] c;
}

template <class Channel>
void Client::tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples,
                        int party_local_mode) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;

    sci::PRG128 prg;
#if PRG_SEED != -1
    uint64_t seed[2];
    gemini::party_seed(generator.io->port, (uint64_t(2) << 32) | gemini::next_call(generator.io->port), seed);
#else
    std::random_device r;
    const uint64_t seed[2] = {r(), r()};
#endif
    prg.reseed(seed);

    if (party_local_mode != 0) {
        // see Server::tuple3_gen; the mode follows the REAL party, not the (alternated) role
        prg.random_data(data.a, num_bytes);
        if (party_local_mode == 1) {  // we are P0: full b, zero c
            prg.random_data(data.b, num_bytes);
            std::memset(data.c, 0, num_bytes);
        } else {  // we are P1: zero b, full c
            std::memset(data.b, 0, num_bytes);
            prg.random_data(data.c, num_bytes);
        }
        cot_multiply_shares(emp::BOB, generator.otpack, data.a, data.b, data.ab, num_tuples);
    } else {
        Client::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    }

    // see Server::tuple3_gen (BOB role here: the real P0's zero c in mode 1 is ours)
    uint8_t* const xs[3] = {data.a, data.b, data.ab};
    uint8_t* const outs[3] = {data.ac, data.bc, data.abc};
    cot_outer_multiply(emp::BOB, generator.otpack, xs, 3, data.c, outs, num_tuples,
                       party_local_mode != 2, party_local_mode != 1);
}

template <class Channel>
void Client::tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Client::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    Client::triple_gen(generator, data.c, data.d, data.cd, num_bytes, packed, triple_gen_method);

    // the 9 products {a, b, ab} x {c, d, cd}: one OT per direction and factor y in {c, d, cd} (all
    // three in one call), carrying the three products with a, b and ab
    auto lhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(9 * num_bytes);
    pack_buffers({data.a, data.a, data.a, data.b, data.b, data.b, data.ab, data.ab, data.ab}, lhs.get(), num_bytes);
    pack_buffers({data.c, data.d, data.cd}, rhs.get(), num_bytes);
    uint8_t* const xs[3] = {lhs.get(), lhs.get() + 3 * num_bytes, lhs.get() + 6 * num_bytes};
    uint8_t* const outs[3] = {out.get(), out.get() + 3 * num_bytes, out.get() + 6 * num_bytes};
    cot_outer_multiply(emp::BOB, generator.otpack, xs, 3, rhs.get(), outs, 3 * num_tuples, false, false);
    unpack_buffers(out.get(),
                   {data.ac, data.ad, data.acd, data.bc, data.bd, data.bcd, data.abc, data.abd,
                    data.abcd},
                   num_bytes);
}

template <class IO>
void Server::mul_gen(const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* u, size_t num_muls){
    auto a_buf = std::make_unique<bool[]>(num_muls);
    auto x_a = std::make_unique<uint8_t[]>(num_muls);
    otpack->silent_ot_reversed->template recv_ot_rm_rc<uint8_t>(x_a.get(), a_buf.get(), num_muls, 1);
    otpack->io->flush();

    // pack `bool`s
    for (size_t i = 0; i < num_muls; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        u[byte_idx] |= x_a[i] << bit_idx;
        a[byte_idx] |= static_cast<uint8_t>(a_buf[i]) << bit_idx;
    }
}

template <class IO>
void Client::mul_gen(const sci::OTPack<IO>* otpack, uint8_t* b, uint8_t* v, size_t num_muls){
    auto x0 = std::make_unique<uint8_t[]>(num_muls);
    auto x1 = std::make_unique<uint8_t[]>(num_muls);
    otpack->silent_ot_reversed->template send_ot_rm_rc<uint8_t>(x0.get(), x1.get(), num_muls, 1);
    otpack->io->flush();

    for (size_t i = 0; i < num_muls; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        uint8_t x0_bit = x0[i] << bit_idx;
        b[byte_idx] |= x0_bit ^ (x1[i] << bit_idx); 
        v[byte_idx] |= x0_bit;
    }
}

#endif
