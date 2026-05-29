#ifndef OT_PROTO_HPP_
#define OT_PROTO_HPP_

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
template <typename IO>
void cot_multiply_shares(int party, const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* b, uint8_t* c, size_t num_shares) {
    size_t num_bytes = (num_shares + 7) / 8;
    auto r0 = std::make_unique<uint8_t[]>(num_shares),
        r1 = std::make_unique<uint8_t[]>(num_shares),
        s = std::make_unique<uint8_t[]>(num_shares),
        rs = std::make_unique<uint8_t[]>(num_shares),
        my_choice_corrections = std::make_unique<uint8_t[]>(num_bytes),
        my_masked_value = std::make_unique<uint8_t[]>(num_bytes),
        their_choice_corrections = std::make_unique<uint8_t[]>(num_bytes),
        their_masked_value = std::make_unique<uint8_t[]>(num_bytes);

    switch (party) {
        case emp::ALICE: {
            otpack->silent_ot_reversed->template recv_ot_rm_rc<uint8_t>(rs.get(), reinterpret_cast<bool*>(s.get()), num_shares, 1);
            otpack->io->flush();
            otpack->silent_ot->template send_ot_rm_rc<uint8_t>(r0.get(), r1.get(), num_shares, 1);
            break;
        }
        case emp::BOB: {
            otpack->silent_ot_reversed->template send_ot_rm_rc<uint8_t>(r0.get(), r1.get(), num_shares, 1);
            otpack->io->flush();
            otpack->silent_ot->recv_ot_rm_rc(rs.get(), reinterpret_cast<bool*>(s.get()), num_shares, 1);
            break;
        }
    }
    otpack->io->flush();

    pack_bool(s.get(), num_shares);
    pack_bool(rs.get(), num_shares);
    pack_bool(r0.get(), num_shares);
    pack_bool(r1.get(), num_shares);

    for (size_t i = 0; i < num_bytes; ++i) my_choice_corrections[i] = s[i] ^ a[i];
    for (size_t i = 0; i < num_bytes; ++i) my_masked_value[i] = b[i] ^ r0[i] ^ r1[i];

    switch (party) {
        case emp::ALICE: {
            otpack->io->send_data(my_choice_corrections.get(), num_bytes);
            otpack->io->send_data(my_masked_value.get(), num_bytes);
            otpack->io->recv_data(their_choice_corrections.get(), num_bytes);
            otpack->io->recv_data(their_masked_value.get(), num_bytes);
            break;
        }
        case emp::BOB: {
            otpack->io->recv_data(their_choice_corrections.get(), num_bytes);
            otpack->io->recv_data(their_masked_value.get(), num_bytes);
            otpack->io->send_data(my_choice_corrections.get(), num_bytes);
            otpack->io->send_data(my_masked_value.get(), num_bytes);
            break;
        }
    }
    otpack->io->flush();

    for (size_t i = 0; i < num_bytes; ++i) {
        // choice ? `rs` ^ (`r0` ^ `r1` ^ message) : `rs`
        uint8_t rcv_mul = rs[i] ^ (a[i] & their_masked_value[i]);
        // their `s` == real choice ? `r0` : `r1`
        // correction is `true` means that their random `s` != real choice
        uint8_t snd_mul = (~their_choice_corrections[i] & r0[i]) ^ (their_choice_corrections[i] & r1[i]);

        // For one OT direction, let x be the receiver's input bit and y be the
        // sender's input bit. The receiver has random choice s and selected OT
        // mask rs; the sender has random masks r0 and r1.
        //
        // Receiver -> Sender: e = s ^ x
        // Sender -> Receiver: m = y ^ r0 ^ r1
        // Receiver output:    rs ^ (x & m)
        // Sender output:      e ? r1 : r0
        //
        // The matching receiver/sender outputs, held by opposite parties, share
        // the cross term x & y: when x is 0 their masks match and cancel; when
        // x is 1 the receiver also XORs in m and the sender uses the other mask,
        // leaving y. This party's local rcv_mul and snd_mul come from opposite
        // OT directions, so c[i] also XORs in the local product a[i] & b[i].

        c[i] = (a[i] & b[i]) ^ rcv_mul ^ snd_mul;
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

template <class Channel>
void tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples);

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

template <class Channel>
void tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples);

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
void Server::tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Server::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    sci::PRG128 prg;
    std::random_device r;
    const uint64_t seed[2] = {r(), r()};
    prg.reseed(seed);
    prg.random_data(data.c, num_bytes);

    auto lhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(3 * num_bytes);

    pack_buffers({data.a, data.b, data.ab}, lhs.get(), num_bytes);
    pack_buffers({data.c, data.c, data.c}, rhs.get(), num_bytes);
    cot_multiply_shares(emp::ALICE, generator.otpack, lhs.get(), rhs.get(), out.get(),
                        3 * num_tuples);
    unpack_buffers(out.get(), {data.ac, data.bc, data.abc}, num_bytes);
}

template <class Channel>
void Server::tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Server::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    Server::triple_gen(generator, data.c, data.d, data.cd, num_bytes, packed, triple_gen_method);

    auto lhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(9 * num_bytes);

    pack_buffers({data.a, data.a, data.b, data.b, data.ab, data.ab, data.a, data.b, data.ab},
                 lhs.get(), num_bytes);
    pack_buffers({data.c, data.d, data.c, data.d, data.c, data.d, data.cd, data.cd, data.cd},
                 rhs.get(), num_bytes);
    cot_multiply_shares(emp::ALICE, generator.otpack, lhs.get(), rhs.get(), out.get(),
                        9 * num_tuples);
    unpack_buffers(out.get(),
                   {data.ac, data.ad, data.bc, data.bd, data.abc, data.abd, data.acd,
                    data.bcd, data.abcd},
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
void Client::tuple3_gen(TripleGenerator<Channel>& generator, Beaver3Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Client::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    sci::PRG128 prg;
    std::random_device r;
    const uint64_t seed[2] = {r(), r()};
    prg.reseed(seed);
    prg.random_data(data.c, num_bytes);

    auto lhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(3 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(3 * num_bytes);

    pack_buffers({data.a, data.b, data.ab}, lhs.get(), num_bytes);
    pack_buffers({data.c, data.c, data.c}, rhs.get(), num_bytes);
    cot_multiply_shares(emp::BOB, generator.otpack, lhs.get(), rhs.get(), out.get(),
                        3 * num_tuples);
    unpack_buffers(out.get(), {data.ac, data.bc, data.abc}, num_bytes);
}

template <class Channel>
void Client::tuple4_gen(TripleGenerator<Channel>& generator, Beaver4Tuples data, size_t num_tuples) {
    require_tuple_count_multiple_of_8(num_tuples);

    const bool packed = true;
    const TripleGenMethod triple_gen_method = TripleGenMethod::_2ROT;
    const size_t num_bytes = (num_tuples + 7) / 8;
    
    Client::triple_gen(generator, data.a, data.b, data.ab, num_bytes, packed, triple_gen_method);
    Client::triple_gen(generator, data.c, data.d, data.cd, num_bytes, packed, triple_gen_method);

    auto lhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto rhs = std::make_unique<uint8_t[]>(9 * num_bytes);
    auto out = std::make_unique<uint8_t[]>(9 * num_bytes);

    pack_buffers({data.a, data.a, data.b, data.b, data.ab, data.ab, data.a, data.b, data.ab},
                 lhs.get(), num_bytes);
    pack_buffers({data.c, data.d, data.c, data.d, data.c, data.d, data.cd, data.cd, data.cd},
                 rhs.get(), num_bytes);
    cot_multiply_shares(emp::BOB, generator.otpack, lhs.get(), rhs.get(), out.get(),
                        9 * num_tuples);
    unpack_buffers(out.get(),
                   {data.ac, data.ad, data.bc, data.bd, data.abc, data.abd, data.acd,
                    data.bcd, data.abcd},
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
