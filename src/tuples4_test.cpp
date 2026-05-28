#include "core/keys.hpp"
#include "core/hpmpc_interface.hpp"
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
    std::unique_ptr<uint8_t[]> d;
    std::unique_ptr<uint8_t[]> ab;
    std::unique_ptr<uint8_t[]> ac;
    std::unique_ptr<uint8_t[]> ad;
    std::unique_ptr<uint8_t[]> bc;
    std::unique_ptr<uint8_t[]> bd;
    std::unique_ptr<uint8_t[]> cd;
    std::unique_ptr<uint8_t[]> abc;
    std::unique_ptr<uint8_t[]> abd;
    std::unique_ptr<uint8_t[]> acd;
    std::unique_ptr<uint8_t[]> bcd;
    std::unique_ptr<uint8_t[]> abcd;
    size_t num_bytes = 0;

    explicit Buffers(size_t num_bytes)
        : a(new uint8_t[num_bytes]()),
          b(new uint8_t[num_bytes]()),
          c(new uint8_t[num_bytes]()),
          d(new uint8_t[num_bytes]()),
          ab(new uint8_t[num_bytes]()),
          ac(new uint8_t[num_bytes]()),
          ad(new uint8_t[num_bytes]()),
          bc(new uint8_t[num_bytes]()),
          bd(new uint8_t[num_bytes]()),
          cd(new uint8_t[num_bytes]()),
          abc(new uint8_t[num_bytes]()),
          abd(new uint8_t[num_bytes]()),
          acd(new uint8_t[num_bytes]()),
          bcd(new uint8_t[num_bytes]()),
          abcd(new uint8_t[num_bytes]()),
          num_bytes(num_bytes) {}

    Beaver4Tuples as_tuples() const {
        return {
            .a    = a.get(),
            .b    = b.get(),
            .c    = c.get(),
            .d    = d.get(),
            .ab   = ab.get(),
            .ac   = ac.get(),
            .ad   = ad.get(),
            .bc   = bc.get(),
            .bd   = bd.get(),
            .cd   = cd.get(),
            .abc  = abc.get(),
            .abd  = abd.get(),
            .acd  = acd.get(),
            .bcd  = bcd.get(),
            .abcd = abcd.get(),
        };
    }
};

std::string party_name(int party) {
    return party == emp::ALICE ? "ALICE" : "BOB";
}

uint8_t get_bit(const uint8_t* bytes, size_t i) {
    return (bytes[i / 8] >> (i % 8)) & 1;
}

void reveal(const Buffers& alice, const Buffers& bob, Buffers& out) {
    for (size_t i = 0; i < alice.num_bytes; ++i) {
        out.a[i]    = alice.a[i] ^ bob.a[i];
        out.b[i]    = alice.b[i] ^ bob.b[i];
        out.c[i]    = alice.c[i] ^ bob.c[i];
        out.d[i]    = alice.d[i] ^ bob.d[i];
        out.ab[i]   = alice.ab[i] ^ bob.ab[i];
        out.ac[i]   = alice.ac[i] ^ bob.ac[i];
        out.ad[i]   = alice.ad[i] ^ bob.ad[i];
        out.bc[i]   = alice.bc[i] ^ bob.bc[i];
        out.bd[i]   = alice.bd[i] ^ bob.bd[i];
        out.cd[i]   = alice.cd[i] ^ bob.cd[i];
        out.abc[i]  = alice.abc[i] ^ bob.abc[i];
        out.abd[i]  = alice.abd[i] ^ bob.abd[i];
        out.acd[i]  = alice.acd[i] ^ bob.acd[i];
        out.bcd[i]  = alice.bcd[i] ^ bob.bcd[i];
        out.abcd[i] = alice.abcd[i] ^ bob.abcd[i];
    }
}

bool check_bit(const char* name, uint8_t actual, uint8_t expected, size_t bit) {
    if (actual == expected)
        return true;

    std::cout << std::format("TUPLE4 TEST {} FAILED at bit {}: {} != {}\n", name, bit,
                             actual, expected);
    return false;
}

bool check_results(const Buffers& alice, const Buffers& bob, size_t num_tuples) {
    Buffers r(alice.num_bytes);
    reveal(alice, bob, r);

    for (size_t i = 0; i < num_tuples; ++i) {
        const auto a = get_bit(r.a.get(), i);
        const auto b = get_bit(r.b.get(), i);
        const auto c = get_bit(r.c.get(), i);
        const auto d = get_bit(r.d.get(), i);

        if (!check_bit("ab", get_bit(r.ab.get(), i), a & b, i)
            || !check_bit("ac", get_bit(r.ac.get(), i), a & c, i)
            || !check_bit("ad", get_bit(r.ad.get(), i), a & d, i)
            || !check_bit("bc", get_bit(r.bc.get(), i), b & c, i)
            || !check_bit("bd", get_bit(r.bd.get(), i), b & d, i)
            || !check_bit("cd", get_bit(r.cd.get(), i), c & d, i)
            || !check_bit("abc", get_bit(r.abc.get(), i), a & b & c, i)
            || !check_bit("abd", get_bit(r.abd.get(), i), a & b & d, i)
            || !check_bit("acd", get_bit(r.acd.get(), i), a & c & d, i)
            || !check_bit("bcd", get_bit(r.bcd.get(), i), b & c & d, i)
            || !check_bit("abcd", get_bit(r.abcd.get(), i), a & b & c & d, i)) {
            return false;
        }
    }

    return true;
}

void send_buffers(IO::NetIO& io, const Buffers& buffers) {
    io.send_data(buffers.a.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.b.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.c.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.d.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.ab.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.ac.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.ad.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.bc.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.bd.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.cd.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.abc.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.abd.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.acd.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.bcd.get(), static_cast<int>(buffers.num_bytes));
    io.send_data(buffers.abcd.get(), static_cast<int>(buffers.num_bytes));
    io.flush();
}

void recv_buffers(IO::NetIO& io, Buffers& buffers) {
    io.recv_data(buffers.a.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.b.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.c.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.d.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.ab.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.ac.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.ad.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.bc.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.bd.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.cd.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.abc.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.abd.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.acd.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.bcd.get(), static_cast<int>(buffers.num_bytes));
    io.recv_data(buffers.abcd.get(), static_cast<int>(buffers.num_bytes));
}

bool run(int party, const std::string& ip, int port, size_t num_tuples) {
    const auto name        = party_name(party);
    const size_t num_bytes = (num_tuples + 7) / 8;
    Buffers buffers(num_bytes);
    int threads = 1;
    int io_offset = 1;

    std::cout << std::format("{}: generating {} bool4 tuples on port {}\n", name, num_tuples,
                             port);

    auto& keys = Iface::Keys<IO::NetIO>::instance(party, ip, port, threads, io_offset);
    auto** ios = keys.get_ios(threads);

    auto tuple_gen_start = measure::now();

    Iface::generateBool4TupleCheetah(buffers.as_tuples(), num_tuples, ip, port, party, 1, 1);

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
        std::cout << (passed ? "TUPLE4 GENERATION TEST PASSED\n"
                             : "TUPLE4 GENERATION TEST FAILED\n");
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

    const int party         = party_arg == 0 ? emp::ALICE : emp::BOB;
    const int port          = argc >= 3 ? std::strtol(argv[2], nullptr, 10) : 7777;
    const std::string host  = argc >= 4 ? argv[3] : "127.0.0.1";
    const size_t num_tuples = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 8;
    const std::string ip    = party == emp::BOB ? host : "";
    const auto start        = measure::now();

    std::cout << "TUPLE4: starting the test\n";
    const bool passed = run(party, ip, port, num_tuples);

    const auto secs_passed = Utils::to_sec(Utils::time_diff(start));
    std::cout << std::format("{}: seconds passed: {}\n", party_name(party), secs_passed);

    return passed ? 0 : EXEC_FAILED;
}
