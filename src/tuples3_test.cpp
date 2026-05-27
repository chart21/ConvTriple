#include "core/hpmpc_interface.hpp"
#include "core/keys.hpp"
#include "core/protocols/ot_proto.hpp"
#include "core/utils.hpp"
#include "emp-tool/utils/constants.h"
#include "ot/bit-triple-generator.h"

#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <string>

namespace {

struct Buffers {
    std::unique_ptr<uint8_t[]> a;
    std::unique_ptr<uint8_t[]> b;
    std::unique_ptr<uint8_t[]> c;
    std::unique_ptr<uint8_t[]> ab;
    std::unique_ptr<uint8_t[]> ac;
    std::unique_ptr<uint8_t[]> bc;
    std::unique_ptr<uint8_t[]> abc;
    size_t num_bytes = 0;

    explicit Buffers(size_t num_bytes)
        : a(new uint8_t[num_bytes]()),
          b(new uint8_t[num_bytes]()),
          c(new uint8_t[num_bytes]()),
          ab(new uint8_t[num_bytes]()),
          ac(new uint8_t[num_bytes]()),
          bc(new uint8_t[num_bytes]()),
          abc(new uint8_t[num_bytes]()),
          num_bytes(num_bytes) {}

    Beaver3Tuples as_tuples() const {
        return {
            .a   = a.get(),
            .b   = b.get(),
            .c   = c.get(),
            .ab  = ab.get(),
            .ac  = ac.get(),
            .bc  = bc.get(),
            .abc = abc.get(),
        };
    }
};

std::string party_name(int party) {
    return party == emp::ALICE ? "ALICE" : "BOB";
}

uint8_t get_bit(const uint8_t* bytes, size_t i) {
    return (bytes[i / 8] >> (i % 8)) & 1;
}

bool should_eq_2(const char* name, const uint8_t* result, const uint8_t* term_0,
                 const uint8_t* term_1, size_t i) {
    const auto r  = get_bit(result, i);
    const auto t0 = get_bit(term_0, i);
    const auto t1 = get_bit(term_1, i);
    if (r == (t0 & t1))
        return true;

    std::cout << std::format("TUPLE GENERATION TEST {} FAILED at bit {}: {} != {} & {}\n",
                             name, i, r, t0, t1);
    return false;
}

bool should_eq_3(const char* name, const uint8_t* result, const uint8_t* term_0,
                 const uint8_t* term_1, const uint8_t* term_2, size_t i) {
    const auto r  = get_bit(result, i);
    const auto t0 = get_bit(term_0, i);
    const auto t1 = get_bit(term_1, i);
    const auto t2 = get_bit(term_2, i);
    if (r == (t0 & t1 & t2))
        return true;

    std::cout << std::format("TUPLE GENERATION TEST {} FAILED at bit {}: {} != {} & {} & {}\n",
                             name, i, r, t0, t1, t2);
    return false;
}

void dump_bit(const char* name, const uint8_t* alice_share, const uint8_t* bob_share,
              uint8_t expected, size_t i) {
    const auto alice = get_bit(alice_share, i);
    const auto bob   = get_bit(bob_share, i);
    const auto value = alice ^ bob;

    std::cout << std::format("{}: A={} B={} revealed={} expected={}\n", name, alice, bob,
                             value, expected);
}

void dump_tuple(const Buffers& alice, const Buffers& bob, size_t i) {
    const auto a = get_bit(alice.a.get(), i) ^ get_bit(bob.a.get(), i);
    const auto b = get_bit(alice.b.get(), i) ^ get_bit(bob.b.get(), i);
    const auto c = get_bit(alice.c.get(), i) ^ get_bit(bob.c.get(), i);

    std::cout << std::format("tuple bit {}:\n", i);
    dump_bit("  a  ", alice.a.get(), bob.a.get(), a, i);
    dump_bit("  b  ", alice.b.get(), bob.b.get(), b, i);
    dump_bit("  c  ", alice.c.get(), bob.c.get(), c, i);
    dump_bit("  ab ", alice.ab.get(), bob.ab.get(), a & b, i);
    dump_bit("  ac ", alice.ac.get(), bob.ac.get(), a & c, i);
    dump_bit("  bc ", alice.bc.get(), bob.bc.get(), b & c, i);
    dump_bit("  abc", alice.abc.get(), bob.abc.get(), a & b & c, i);
}

bool check_results(const Buffers& alice, const Buffers& bob, size_t num_tuples) {
    Buffers revealed(alice.num_bytes);

    for (size_t i = 0; i < alice.num_bytes; ++i) {
        revealed.a[i]   = alice.a[i] ^ bob.a[i];
        revealed.b[i]   = alice.b[i] ^ bob.b[i];
        revealed.c[i]   = alice.c[i] ^ bob.c[i];
        revealed.ab[i]  = alice.ab[i] ^ bob.ab[i];
        revealed.ac[i]  = alice.ac[i] ^ bob.ac[i];
        revealed.bc[i]  = alice.bc[i] ^ bob.bc[i];
        revealed.abc[i] = alice.abc[i] ^ bob.abc[i];
    }

    for (size_t i = 0; i < num_tuples; ++i) {
        if (!should_eq_2("ab", revealed.ab.get(), revealed.a.get(), revealed.b.get(), i)
            || !should_eq_2("ac", revealed.ac.get(), revealed.a.get(), revealed.c.get(), i)
            || !should_eq_2("bc", revealed.bc.get(), revealed.b.get(), revealed.c.get(), i)
            || !should_eq_3("abc", revealed.abc.get(), revealed.a.get(), revealed.b.get(),
                            revealed.c.get(), i)) {
            const size_t dump_count = num_tuples < 32 ? num_tuples : 32;
            std::cout << std::format("Produced tuple3 values by party share (first {} bits):\n",
                                     dump_count);
            for (size_t j = 0; j < dump_count; ++j) dump_tuple(alice, bob, j);
            return false;
        }
    }

    return true;
}

void send_buffers(IO::NetIO& io, const Buffers& buffers) {
    io.send_data(buffers.a.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.b.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.c.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.ab.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.ac.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.bc.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.abc.get(), static_cast<int>(buffers.num_bytes));
    io.flush();
}

void recv_buffers(IO::NetIO& io, Buffers& buffers) {
    io.recv_data(buffers.a.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.b.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.c.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.ab.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.ac.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.bc.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.abc.get(), static_cast<int>(buffers.num_bytes));
}

bool run(int party, const std::string& ip, int port, size_t num_tuples) {
    const auto name      = party_name(party);
    const size_t num_bytes = (num_tuples + 7) / 8;
    Buffers buffers(num_bytes);
    int threads = 1;
    int io_offset = 1;

    std::cout << std::format("{}: generating {} bool3 tuples on port {}\n", name, num_tuples,
                             port);

    auto& keys = Iface::Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);
    auto** ios = keys.get_ios(threads);

    TripleGenerator<IO::NetIO> triple_gen(party, ios[0], keys.get_otpack(0), false);

    auto tuple_gen_start = measure::now();
    Iface::generateBool3TupleCheetah(buffers.as_tuples(), num_tuples, ip, port, party);
    const auto secs_passed = Utils::to_sec(Utils::time_diff(tuple_gen_start));
    std::cout << std::format("{}: tuples generated in: {:.2f} seconds / {:.2f} triples per second\n",
        party_name(party), secs_passed, static_cast<double>(num_tuples) / secs_passed);

    bool passed = true;
    if (party == emp::BOB) {
        send_buffers(*ios[0], buffers);
        std::cout << "BOB: sent shares to ALICE for verification\n";
    } else {
        Buffers bob_buffers(num_bytes);
        recv_buffers(*ios[0], bob_buffers);
        passed = check_results(buffers, bob_buffers, num_tuples);
        std::cout << (passed ? "TUPLE GENERATION TEST PASSED\n" : "TUPLE GENERATION TEST FAILED\n");
    }

    keys.disconnect();
    return passed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cout << argv[0] << " <party: 0=ALICE, 1=BOB> [port] [host] [num_tuples]\n";
        return EXEC_FAILED;
    }

    const int party_arg = std::strtol(argv[1], nullptr, 10);
    if (party_arg != 0 && party_arg != 1) {
        std::cout << "party must be 0 for ALICE or 1 for BOB\n";
        return EXEC_FAILED;
    }

    const int party          = party_arg == 0 ? emp::ALICE : emp::BOB;
    const int port           = argc >= 3 ? std::strtol(argv[2], nullptr, 10) : 7777;
    const std::string host   = argc >= 4 ? argv[3] : "127.0.0.1";
    const size_t num_tuples  = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 8;
    const std::string ip     = party == emp::BOB ? host : "";
    const auto start         = measure::now();

    std::cout << "TUPLE3: starting the test\n";
    const bool passed = run(party, ip, port, num_tuples);

    const auto secs_passed = Utils::to_sec(Utils::time_diff(start));
    std::cout << std::format("{}: seconds passed: {}\n", party_name(party), secs_passed);

    return passed ? 0 : EXEC_FAILED;
}
