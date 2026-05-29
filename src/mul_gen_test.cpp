#include "core/hpmpc_interface.hpp"
#include "core/utils.hpp"
#include "emp-tool/utils/constants.h"
#include "io/net_io_channel.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Buffers {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;

    explicit Buffers(size_t num_bytes) : a(num_bytes), b(num_bytes) {}
};

std::string party_name(int party) { return party == emp::ALICE ? "ALICE" : "BOB"; }

uint8_t get_bit(const std::vector<uint8_t>& bytes, size_t i) {
    return (bytes[i / 8] >> (i % 8)) & 1U;
}

void send_buffers(IO::NetIO& io, const Buffers& buffers) {
    io.send_data(buffers.a.data(), static_cast<int>(buffers.a.size()));
    io.send_data(buffers.b.data(), static_cast<int>(buffers.b.size()));
    io.flush();
}

void recv_buffers(IO::NetIO& io, Buffers& buffers) {
    io.recv_data(buffers.a.data(), static_cast<int>(buffers.a.size()));
    io.recv_data(buffers.b.data(), static_cast<int>(buffers.b.size()));
}

void dump_mul(const Buffers& alice, const Buffers& bob, size_t i) {
    const auto a0 = get_bit(alice.a, i);
    const auto a1 = get_bit(bob.a, i);
    const auto b0 = get_bit(alice.b, i);
    const auto b1 = get_bit(bob.b, i);

    std::cout << std::format("  bit {}: a0={} a1={} b0={} b1={} lhs={} rhs={}\n", i, a0, a1, b0, b1,
                             a0 & a1, b0 ^ b1);
}

bool check_results(const Buffers& alice, const Buffers& bob, size_t num_muls) {
    for (size_t i = 0; i < num_muls; ++i) {
        const auto lhs = get_bit(alice.a, i) & get_bit(bob.a, i);
        const auto rhs = get_bit(alice.b, i) ^ get_bit(bob.b, i);
        if (lhs == rhs)
            continue;

        std::cout << std::format("MUL_GEN TEST FAILED at bit {}: {} != {}\n", i, lhs, rhs);
        const size_t dump_count = std::min(num_muls, static_cast<size_t>(32));
        std::cout << std::format("Produced random multiplication shares (first {} bits):\n",
                                 dump_count);
        for (size_t j = 0; j < dump_count; ++j) dump_mul(alice, bob, j);
        return false;
    }

    return true;
}

bool verify(IO::NetIO& io, int party, const Buffers& buffers, size_t num_muls) {
    bool passed = true;
    if (party == emp::BOB) {
        send_buffers(io, buffers);

        uint8_t verdict = 0;
        io.recv_data(&verdict, sizeof(verdict));
        passed = verdict == 1;
        std::cout << (passed ? "BOB: MUL_GEN TEST PASSED\n" : "BOB: MUL_GEN TEST FAILED\n");
    } else {
        Buffers bob_buffers(buffers.a.size());
        recv_buffers(io, bob_buffers);
        passed = check_results(buffers, bob_buffers, num_muls);

        uint8_t verdict = passed ? 1 : 0;
        io.send_data(&verdict, sizeof(verdict));
        io.flush();
        std::cout << (passed ? "ALICE: MUL_GEN TEST PASSED\n" : "ALICE: MUL_GEN TEST FAILED\n");
    }

    return passed;
}

bool run(int party, const std::string& host, int port, size_t num_muls, size_t samples) {
    const size_t num_bytes = (num_muls + 7) / 8;
    Buffers buffers(num_bytes);

    const auto gen_start = measure::now();
    for (size_t i = 0; i < samples; ++i) {
        Iface::generateRandomMultiplicationsCheetah(buffers.a.data(), buffers.b.data(), num_muls,
                                                    host, port, party, 1, 1);
    }

    const auto gen_secs   = Utils::to_sec(Utils::time_diff(gen_start));
    const auto total_muls = static_cast<double>(num_muls) * static_cast<double>(samples);
    const auto throughput = gen_secs > 0 ? total_muls / gen_secs : 0;
    std::cout << std::format("{}: generated {} random multiplications x {} sample(s) in {:.4f}s\n",
                             party_name(party), num_muls, samples, gen_secs);
    std::cout << std::format("{}: {:.2f} mul/s\n", party_name(party), throughput);

    const char* address = party == emp::ALICE ? nullptr : host.c_str();
    IO::NetIO verify_io(address, port);
    return verify(verify_io, party, buffers, num_muls);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cout << argv[0] << " <party: 0=ALICE, 1=BOB> [port] [host] [num_muls] [samples]\n";
        return EXEC_FAILED;
    }

    const int party_arg = std::strtol(argv[1], nullptr, 10);
    if (party_arg != 0 && party_arg != 1) {
        std::cout << "party must be 0 for ALICE or 1 for BOB\n";
        return EXEC_FAILED;
    }

    const int party        = party_arg == 0 ? emp::ALICE : emp::BOB;
    const int port         = argc >= 3 ? std::strtol(argv[2], nullptr, 10) : 7777;
    const std::string host = argc >= 4 ? argv[3] : "127.0.0.1";
    const size_t num_muls  = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 8;
    const size_t samples   = argc >= 6 ? std::strtoull(argv[5], nullptr, 10) : 1;

    if (num_muls == 0 || samples == 0) {
        std::cout << "num_muls and samples must be greater than zero\n";
        return EXEC_FAILED;
    }

    std::cout << "MUL_GEN: starting the test\n";
    const bool passed = run(party, host, port, num_muls, samples);

    return passed ? 0 : EXEC_FAILED;
}
