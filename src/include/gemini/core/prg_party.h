#pragma once
#include <cstdint>
#include <map>
#include <mutex>

#ifndef PRG_SEED
#define PRG_SEED 42
#endif

// With PRG_SEED != -1 all randomness of the triple generation is derived from PRG_SEED for one party
// and PRG_SEED + 1 for the other (so the parties hold different keys and masks), which makes a run
// reproducible: every stream is addressed by a tag that does not depend on thread scheduling.
namespace gemini {

constexpr bool kSeeded = PRG_SEED != -1;  // reproducible runs (emp::DetSeedScope)

inline int& prg_party() { // emp::ALICE = 1, emp::BOB = 2
    static int party = 1;
    return party;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// the seed of the stream (a, b) of this party
inline void party_seed(uint64_t a, uint64_t b, uint64_t out[2]) {
    const uint64_t s = splitmix64(uint64_t(PRG_SEED + prg_party() - 1));
    out[0] = splitmix64(s ^ splitmix64(a));
    out[1] = splitmix64(out[0] ^ splitmix64(b + 0x5bd1e995ULL));
}

inline uint64_t party_seed64(uint64_t a = 0, uint64_t b = 0) {
    uint64_t s[2];
    party_seed(a, b, s);
    return s[0];
}

// n-th call on a channel (the calls on one channel are sequential)
inline uint64_t next_call(uint64_t channel) {
    static std::mutex m;
    static std::map<uint64_t, uint64_t> calls;
    std::lock_guard<std::mutex> lock(m);
    return calls[channel]++;
}

} // namespace gemini
