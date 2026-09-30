// Two-party check of the conv triples produced by generateConvTriplesCheetahWrapper (CPU/SEAL or, when
// built with TRIPLE_GPU, GPU/troy): c1 + c2 must equal conv(x1 + x2, w1 + w2) exactly, and the traffic per
// layer is reported from the weight holder's view.
//
// Usage: cheetah_conv_triple_test <party 1|2> <port> <ab 0|1> [suite] [rounds] [--packed|--pipelined]
//   ab 0: AB2 (A_KNOWN=1): party 1 holds w, party 2 holds x;  ab 1: AB, both hold shares of x and w
//   suite: small    - strided/padded shapes, weight chunks, batches beyond one GPU batch (default)
//          cifar    - every conv of the CIFAR-10 ResNet50, batch 10
//          stress   - the cifar shapes, `rounds` times in a shuffled order
//          imagenet - the 53 convs of ResNet50-Cheetah on 224x224 inputs, batch 1
//   --packed: generateConvTriplesPacked (the GPU path's packing on the CPU) instead of the wrapper
//   --pipelined: all convs of the suite in one generateConvTriplesPackedBatch (per-layer times not available)
//   env: CONV_TEST_THREADS (default 1), CONV_TEST_IP (party 1's address, default 127.0.0.1), CONV_REPACK=1 (repacking)
#include "core/hpmpc_interface.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using U = uint32_t;
struct Shape {
    int ic, h, k, s, p, oc, bs = 1;
};

// Plain strided, zero-padded convolution over a batch (NCHW, OIHW), mod 2^32
std::vector<U> ideal(const std::vector<U>& x, const std::vector<U>& w, const Shape& sh) {
    int n = (sh.h + 2 * sh.p - sh.k) / sh.s + 1;
    std::vector<U> y((size_t)sh.bs * sh.oc * n * n, 0);
    for (int b = 0; b < sh.bs; b++)
        for (int o = 0; o < sh.oc; o++)
            for (int c = 0; c < sh.ic; c++)
                for (int a = 0; a < sh.k; a++)
                    for (int d = 0; d < sh.k; d++) {
                        U wv = w[(((size_t)o * sh.ic + c) * sh.k + a) * sh.k + d];
                        const U* xc = x.data() + ((size_t)b * sh.ic + c) * sh.h * sh.h;
                        U* yo       = y.data() + ((size_t)b * sh.oc + o) * n * n;
                        for (int i = 0; i < n; i++) {
                            int yy = i * sh.s + a - sh.p;
                            if (yy < 0 || yy >= sh.h) continue;
                            for (int j = 0; j < n; j++) {
                                int xx = j * sh.s + d - sh.p;
                                if (xx >= 0 && xx < sh.h) yo[i * n + j] += xc[yy * sh.h + xx] * wv;
                            }
                        }
                    }
    return y;
}

const std::vector<Shape> kCifar = {
    {1024, 2, 1, 2, 0, 2048, 10}, {1024, 2, 1, 1, 0, 256, 10}, {1024, 2, 1, 1, 0, 512, 10}, {128, 4, 1, 1, 0, 512, 10},
    {128, 4, 3, 1, 1, 128, 10},   {128, 8, 3, 2, 1, 128, 10},  {2048, 1, 1, 1, 0, 512, 10}, {256, 2, 1, 1, 0, 1024, 10},
    {256, 2, 3, 1, 1, 256, 10},   {256, 4, 3, 2, 1, 256, 10},  {256, 8, 1, 1, 0, 128, 10},  {256, 8, 1, 2, 0, 512, 10},
    {256, 8, 1, 1, 0, 64, 10},    {3, 32, 7, 2, 3, 64, 10},    {512, 1, 1, 1, 0, 2048, 10}, {512, 1, 3, 1, 1, 512, 10},
    {512, 2, 3, 2, 1, 512, 10},   {512, 4, 1, 2, 0, 1024, 10}, {512, 4, 1, 1, 0, 128, 10},  {512, 4, 1, 1, 0, 256, 10},
    {64, 8, 1, 1, 0, 256, 10},    {64, 8, 1, 1, 0, 64, 10},    {64, 8, 3, 1, 1, 64, 10}};

// ResNet50-Cheetah on ImageNet as hpmpc issues it (inputs of strided 3x3/7x7 convs arrive pre-padded), with counts
const std::vector<std::pair<Shape, int>> kImagenet = {
    {{3, 230, 7, 2, 0, 64}, 1},     {{64, 56, 1, 1, 0, 256}, 4},    {{64, 56, 1, 1, 0, 64}, 1},
    {{64, 56, 3, 1, 1, 64}, 3},     {{256, 56, 1, 1, 0, 64}, 2},    {{256, 56, 1, 2, 0, 512}, 1},
    {{256, 56, 1, 1, 0, 128}, 1},   {{128, 58, 3, 2, 0, 128}, 1},   {{128, 28, 1, 1, 0, 512}, 4},
    {{512, 28, 1, 1, 0, 128}, 3},   {{128, 28, 3, 1, 1, 128}, 3},   {{512, 28, 1, 2, 0, 1024}, 1},
    {{512, 28, 1, 1, 0, 256}, 1},   {{256, 30, 3, 2, 0, 256}, 1},   {{256, 14, 1, 1, 0, 1024}, 6},
    {{1024, 14, 1, 1, 0, 256}, 5},  {{256, 14, 3, 1, 1, 256}, 5},   {{1024, 14, 1, 2, 0, 2048}, 1},
    {{1024, 14, 1, 1, 0, 512}, 1},  {{512, 16, 3, 2, 0, 512}, 1},   {{512, 7, 1, 1, 0, 2048}, 3},
    {{2048, 7, 1, 1, 0, 512}, 2},   {{512, 7, 3, 1, 1, 512}, 2}};

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    bool packed    = std::erase(args, "--packed") > 0;
    bool pipelined = std::erase(args, "--pipelined") > 0;
    if (args.size() < 3) {
        fprintf(stderr, "usage: %s <party 1|2> <port> <ab 0|1> [small|cifar|stress|imagenet] [rounds] [--packed|--pipelined]\n", argv[0]);
        return 2;
    }
    int party = std::stoi(args[0]), port = std::stoi(args[1]);
    bool ab           = std::stoi(args[2]);
    std::string suite = args.size() > 3 ? args[3] : "small";

    std::vector<Shape> shapes;
    if (suite == "small") // strided/padded; 2 and 3 weight chunks; batches of 3, 20 and 17 images
        shapes = {{3, 16, 7, 2, 3, 8},  {8, 14, 3, 2, 1, 16},    {16, 14, 1, 2, 0, 32},     {4, 8, 3, 1, 1, 8},
                  {5, 9, 3, 2, 1, 6},   {128, 40, 3, 1, 1, 128}, {256, 30, 3, 2, 1, 160},   {8, 14, 3, 2, 1, 16, 3},
                  {16, 8, 3, 1, 1, 32, 20}, {32, 8, 1, 2, 0, 64, 17}};
    else if (suite == "cifar")
        shapes = kCifar;
    else if (suite == "stress") {
        std::vector<Shape> order = kCifar;
        std::mt19937 shared(99); // the same order in both processes
        for (int r = 0, rounds = args.size() > 4 ? std::stoi(args[4]) : 20; r < rounds; r++) {
            std::shuffle(order.begin(), order.end(), shared);
            shapes.insert(shapes.end(), order.begin(), order.end());
        }
    } else if (suite == "imagenet")
        for (auto& [sh, count] : kImagenet)
            for (int i = 0; i < count; i++) shapes.push_back(sh);
    else {
        fprintf(stderr, "unknown suite %s\n", suite.c_str());
        return 2;
    }

    // threads for the HE work (both backends); the communication runs over one channel either way.
    // Party 2 connects to CONV_TEST_IP (party 1 listens), for runs on two hosts.
    const int threads   = getenv("CONV_TEST_THREADS") ? atoi(getenv("CONV_TEST_THREADS")) : 1;
    const char* peer_ip = getenv("CONV_TEST_IP") ? getenv("CONV_TEST_IP") : "127.0.0.1";
    Iface::conv_repack_ab() = ab; // with CONV_REPACK=1: both parties evaluate, so both send Galois keys
    auto& keys = Iface::Keys<IO::NetIO>::instance(party, peer_ip, port, threads, 1);
    auto* io   = keys.get_ios(threads)[0];
    std::mt19937 rng(1234 + party);
    int failed = 0;
    double total_time = 0, total_sent = 0, total_recv = 0;
    std::map<std::string, std::array<double, 4>> groups; // shape -> count, sent, recv, time
    auto proto = ab ? Utils::PROTO::AB : Utils::PROTO::AB2;
    bool hold_x = ab || party == 2, hold_w = ab || party == 1;

    // party 2 hands its shares to party 1, which reconstructs and compares; returns the wrong outputs
    auto check = [&](const Shape& sh, const std::vector<U>& x, const std::vector<U>& w, const U* c, size_t c_size) {
        size_t bad = 0;
        if (party == 2) {
            io->send_data(x.data(), x.size() * 4);
            io->send_data(w.data(), w.size() * 4);
            io->send_data(c, c_size * 4);
            io->flush();
        } else {
            std::vector<U> x2(x.size()), w2(w.size()), c2(c_size);
            io->recv_data(x2.data(), x2.size() * 4);
            io->recv_data(w2.data(), w2.size() * 4);
            io->recv_data(c2.data(), c2.size() * 4);
            for (size_t i = 0; i < x.size(); i++) x2[i] += x[i];
            for (size_t i = 0; i < w.size(); i++) w2[i] += w[i];
            auto y = ideal(x2, w2, sh);
            for (size_t i = 0; i < c_size; i++) bad += (U)(c[i] + c2[i]) != y[i];
        }
        io->counter = io->recv_counter = 0; // keep the check out of the traffic
        return bad;
    };

    if (pipelined) {
        std::vector<std::vector<U>> xs, ws;
        std::vector<U*> xp, wp;
        std::vector<Utils::ConvParm> parms;
        std::vector<size_t> offset{0};
        for (auto& sh : shapes) { // the same inputs as layer by layer
            int n = (sh.h + 2 * sh.p - sh.k) / sh.s + 1;
            xs.emplace_back((size_t)sh.bs * sh.ic * sh.h * sh.h);
            ws.emplace_back((size_t)sh.oc * sh.ic * sh.k * sh.k);
            for (auto& v : xs.back()) v = hold_x ? rng() : 0;
            for (auto& v : ws.back()) v = hold_w ? rng() : 0;
            xp.push_back(xs.back().data()), wp.push_back(ws.back().data());
            parms.push_back({sh.bs, sh.ic, sh.h, sh.h, sh.ic, sh.k, sh.k, sh.oc, sh.s, sh.p});
            offset.push_back(offset.back() + (size_t)sh.bs * sh.oc * n * n);
        }
        std::vector<U> c(offset.back());
        double sent0, recv0, t0, sent1, recv1, t1;
        Iface::getTripleStat("CONV", sent0, recv0, t0);
        auto start = std::chrono::steady_clock::now();
        Iface::generateConvTriplesPackedBatch(keys, parms, hold_x ? xp.data() : nullptr, hold_w ? wp.data() : nullptr,
                                              c.data(), party, threads, proto);
        total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Iface::getTripleStat("CONV", sent1, recv1, t1);
        total_sent = sent1 - sent0, total_recv = recv1 - recv0;
        for (size_t i = 0; i < shapes.size(); i++)
            if (check(shapes[i], xs[i], ws[i], c.data() + offset[i], offset[i + 1] - offset[i])) {
                failed++;
                if (party == 1)
                    printf("conv %zu (ic=%d h=%d k=%d s=%d) FAILED\n", i, shapes[i].ic, shapes[i].h, shapes[i].k, shapes[i].s);
            }
        if (party == 1)
            printf("%zu convs, %d failed: total %.3f s, sent %.2f MiB, recv %.2f MiB (%.2f MiB both ways)\n",
                   shapes.size(), failed, total_time, total_sent, total_recv, total_sent + total_recv);
        return failed;
    }

    for (auto& sh : shapes) {
        int n = (sh.h + 2 * sh.p - sh.k) / sh.s + 1;
        std::vector<U> x((size_t)sh.bs * sh.ic * sh.h * sh.h), w((size_t)sh.oc * sh.ic * sh.k * sh.k),
            c((size_t)sh.bs * sh.oc * n * n);
        for (auto& v : x) v = hold_x ? rng() : 0;
        for (auto& v : w) v = hold_w ? rng() : 0;

        Utils::ConvParm parm{sh.bs, sh.ic, sh.h, sh.h, sh.ic, sh.k, sh.k, sh.oc, sh.s, sh.p};
        double sent0, recv0, t0, sent1, recv1, t1;
        Iface::getTripleStat("CONV", sent0, recv0, t0);
        auto start = std::chrono::steady_clock::now();
        if (packed)
            Iface::generateConvTriplesPacked(keys, hold_x ? x.data() : nullptr, hold_w ? w.data() : nullptr, c.data(),
                                             parm, party, threads, proto);
        else
            Iface::generateConvTriplesCheetahWrapper(keys, hold_x ? x.data() : nullptr, hold_w ? w.data() : nullptr,
                                                     c.data(), parm, party, threads, proto, 1, ab);
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Iface::getTripleStat("CONV", sent1, recv1, t1);
        double sent = sent1 - sent0, recv = recv1 - recv0;
        total_time += t, total_sent += sent, total_recv += recv;

        size_t bad = check(sh, x, w, c.data(), c.size());
        failed += bad > 0;

        char name[96];
        snprintf(name, sizeof name, "ic=%d h=%d k=%d s=%d p=%d oc=%d bs=%d", sh.ic, sh.h, sh.k, sh.s, sh.p, sh.oc, sh.bs);
        auto& g = groups[name];
        g[0] += 1, g[1] += sent, g[2] += recv, g[3] += t;
        if (party == 1 && (bad || suite != "imagenet"))
            printf("conv %-44s %7.3f s  sent %8.2f MiB  recv %8.2f MiB  %s\n", name, t, sent, recv,
                   bad ? "FAILED" : "PASSED");
    }

    if (party == 1) {
        if (suite == "imagenet") {
            printf("%-44s %3s %12s %12s %9s\n", "layer (party 1 = weight holder)", "n", "sent MiB", "recv MiB", "time s");
            for (auto& [name, g] : groups)
                printf("%-44s %3.0f %12.2f %12.2f %9.3f\n", name.c_str(), g[0], g[1], g[2], g[3]);
        }
        printf("%zu convs, %d failed: total %.3f s, sent %.2f MiB, recv %.2f MiB (%.2f MiB both ways)\n", shapes.size(),
               failed, total_time, total_sent, total_recv, total_sent + total_recv);
    }
    return failed;
}
