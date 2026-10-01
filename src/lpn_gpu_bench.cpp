// lpn_gpu_bench [instances]: ferret's LPN step (b12: n = 10268672, k = 238000) on the CPU (emp's AES, ConvTriple's grouping)
// against lpn_gpu::compute: bit-for-bit check, then the time of `instances` concurrent steps on each
#include <emp-tool/emp-tool.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "ot/lpn_gpu.h"

using namespace emp;

static void lpn_cpu(block* nn, const block* kk, int64_t start, int64_t groups, int64_t k, uint32_t mask, const block& seed) {
    PRP prp(seed);
    for (int64_t g = 0; g < groups; g++) {
        const int64_t i = start + 4 * g;
        block t[10];
        for (int m = 0; m < 10; m++) t[m] = makeBlock(i, m);
        AES_ecb_encrypt_blks(t, 10, &prp.aes);
        const uint32_t* r = reinterpret_cast<const uint32_t*>(t);
        for (int m = 0; m < 4; m++)
            for (int j = 0; j < 10; j++) {
                uint32_t index = r[m * 10 + j] & mask;
                index -= index >= uint32_t(k) ? uint32_t(k) : 0;
                nn[i + m] = nn[i + m] ^ kk[index];
            }
    }
}

int main(int argc, char** argv) {
    const int inst = argc > 1 ? atoi(argv[1]) : 16;
    const int64_t n = 10268672, k = 238000;
    uint32_t mask = 1;
    while (mask < k) mask = (mask << 1) | 1;
    const int64_t groups = (n - 4 + 3) / 4;  // one task range [0, n): j < n - 4
    PRG prg;
    block seed;
    prg.random_block(&seed, 1);
    PRP prp(seed);
    uint32_t rk[44];
    std::memcpy(rk, prp.aes.rd_key, sizeof(rk));
    // AES check
    std::vector<block> tb(1024);
    for (int t = 0; t < 1024; t++) tb[t] = makeBlock(7, t);
    AES_ecb_encrypt_blks(tb.data(), 1024, &prp.aes);
    cheetah::lpn_gpu::aes_reference_check(rk, reinterpret_cast<const uint32_t*>(tb.data()), 1024);
    printf("AES: GPU = AES-NI on 1024 blocks\n");
    // per instance: kk, nn
    std::vector<std::vector<block>> kk(inst, std::vector<block>(k)), nn(inst, std::vector<block>(n)), ref(inst);
    for (int s = 0; s < inst; s++) {
        prg.random_block(kk[s].data(), k);
        prg.random_block(nn[s].data(), n);
        ref[s] = nn[s];
    }
    auto t0 = std::chrono::steady_clock::now();
    lpn_cpu(ref[0].data(), kk[0].data(), 0, groups, k, mask, seed);
    double cpu1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<block> g0 = nn[0];
    cheetah::lpn_gpu::compute(g0.data(), kk[0].data(), n, k, mask, rk, 0, groups);  // warm-up + check
    if (std::memcmp(g0.data(), ref[0].data(), size_t(n) * 16) != 0) {
        printf("MISMATCH between CPU and GPU LPN\n");
        return 1;
    }
    printf("LPN: GPU = CPU on %ld outputs\n", (long) n);
    t0 = std::chrono::steady_clock::now();
    cheetah::lpn_gpu::compute(g0.data(), kk[0].data(), n, k, mask, rk, 0, groups);
    double gpu1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // concurrent instances, one thread each
    auto run = [&](bool gpu) {
        std::vector<std::thread> th;
        auto t = std::chrono::steady_clock::now();
        for (int s = 0; s < inst; s++)
            th.emplace_back([&, s] {
                if (gpu) cheetah::lpn_gpu::compute(nn[s].data(), kk[s].data(), n, k, mask, rk, 0, groups);
                else lpn_cpu(nn[s].data(), kk[s].data(), 0, groups, k, mask, seed);
            });
        for (auto& x : th) x.join();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    };
    double cpuN = run(false);
    run(true);  // the contexts' buffers
    double gpuN = run(true);
    printf("one instance: CPU %.3f s, GPU %.3f s (incl. copies); %d concurrent: CPU %.3f s, GPU %.3f s\n", cpu1, gpu1, inst,
           cpuN, gpuN);
    return 0;
}
