// cheetah_arith_test PARTY N [PROTO] [IP] [PORT] [THREADS]: n elementwise (arithmetic) triples modulo 2^BIT_LEN, checked;
// PARTY 1 = ALICE (holds a), 2 = BOB (holds b); PROTO ab2 (default) or ab (both hold shares of a and b).
// ARITH_OT=1 takes Gilboa multiplication over silent COTs instead of HE (64-bit builds).
#include "core/hpmpc_interface.hpp"
#include "io/net_io_channel.hpp"

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " PARTY N [ab2|ab] [IP] [PORT] [THREADS]\n";
        return 1;
    }
    const int party = std::atoi(argv[1]);
    const uint64_t n = std::strtoull(argv[2], nullptr, 10);
    const bool ab = argc > 3 && std::string(argv[3]) == "ab";
    const std::string ip = argc > 4 ? argv[4] : "127.0.0.1";
    const int port = argc > 5 ? std::atoi(argv[5]) : 34000;
    const int threads = argc > 6 ? std::atoi(argv[6]) : 8;
    using U = Iface::UINT_TYPE;
    std::mt19937_64 rng(1234 + party);
    std::vector<U> a(n), b(n), c(n);
    for (auto& x : a) x = U(rng());
    for (auto& x : b) x = U(rng());
    const bool alice = party == emp::ALICE;
    Iface::generateArithTriplesCheetah(ab || alice ? a.data() : nullptr, ab || !alice ? b.data() : nullptr, c.data(),
                                       int(BIT_LEN), n, ip, port, party, threads,
                                       ab ? Utils::PROTO::AB : Utils::PROTO::AB2, 0);
    // check: the parties exchange their values on a channel of their own
    IO::NetIO io(alice ? nullptr : ip.c_str(), port + 100, true);
    std::vector<U> oa(n), ob(n), oc(n);
    if (alice) {
        io.recv_data(oa.data(), n * sizeof(U)), io.recv_data(ob.data(), n * sizeof(U)), io.recv_data(oc.data(), n * sizeof(U));
        uint64_t bad = 0;
        for (uint64_t i = 0; i < n; i++) {
            const U x = ab ? U(a[i] + oa[i]) : a[i], y = ab ? U(b[i] + ob[i]) : ob[i];
            bad += U(c[i] + oc[i]) != U(x * y);
        }
        std::cout << "ARITH " << (ab ? "AB" : "AB2") << " " << BIT_LEN << "-bit, " << n << " triples: " << bad
                  << " wrong" << std::endl;
        return bad != 0;
    }
    io.send_data(a.data(), n * sizeof(U)), io.send_data(b.data(), n * sizeof(U)), io.send_data(c.data(), n * sizeof(U));
    io.flush();
    return 0;
}
