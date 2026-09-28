/*
Authors: Deevashwer Rathee
Modified by Zhicong Huang.
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef TRIPLE_GENERATOR_H__
#define TRIPLE_GENERATOR_H__
// #include "OT/emp-ot.h"
#include "ot/cheetah-ot_pack.h"
#include "ot/ot.h"

#define TGEN_PRINT_TIME 1
#define TGEN_PRINT_COMP 0
#define TGEN_PRINT_COMM 1
#if TGEN_PRINT_COMP || TGEN_PRINT_COMM || TGEN_PRINT_TIME
#include <iomanip>
#endif

enum TripleGenMethod {
    Ideal,          // (Insecure) Ideal Functionality
    _2ROT,          // 1 Bit Triple from 2 ROT
    _16KKOT_to_4OT, // 2 Bit Triples from 1oo16 KKOT to 1oo4 OT
    _8KKOT,         // 2 Correlated Bit Triples from 1oo8 KKOT
    _2COT,          // 1 Bit Triple from 2 COT
};

class Triple {
  public:
    bool packed;
    bool wraped = false;
    uint8_t* ai;
    uint8_t* bi;
    uint8_t* ci;
    uint64_t num_triples, num_bytes, offset;

    Triple(int num_triples, bool packed = false, int offset = 0) {
        assert((offset < num_triples) || (num_triples == 0));
        this->num_triples = num_triples;
        this->packed      = packed;
        if (packed) {
            assert(num_triples % 8 == 0);
            assert(offset % 8 == 0);
            this->num_bytes = num_triples / 8;
        } else
            this->num_bytes = num_triples;
        if (offset == 0)
            this->offset = 1;
        else
            this->offset = offset;
        assert((num_triples % this->offset) == 0);
        this->ai = new uint8_t[num_bytes];
        this->bi = new uint8_t[num_bytes];
        this->ci = new uint8_t[num_bytes];
    }

    Triple(uint8_t* a, uint8_t* b, uint8_t* c, int num_triples, bool packed = false,
           int offset = 0) {
        assert((offset < num_triples) || (num_triples == 0));
        this->num_triples = num_triples;
        this->packed      = packed;
        if (packed) {
            assert(num_triples % 8 == 0);
            assert(offset % 8 == 0);
            this->num_bytes = num_triples / 8;
        } else
            this->num_bytes = num_triples;
        if (offset == 0)
            this->offset = 1;
        else
            this->offset = offset;
        assert((num_triples % this->offset) == 0);
        this->ai = a;
        this->bi = b;
        this->ci = c;
        wraped   = true;
    }

    ~Triple() {
        if (!wraped) {
            delete[] ai;
            delete[] bi;
            delete[] ci;
        }
    }
};

#define BSIZE 67108864 // Default buffer size
#define CSIZE 8192     // Default chunk size
template <typename IO>
class TripleGenerator {
    __m256i seed;

  public:
    IO* io                        = nullptr;
    const sci::OTPack<IO>* otpack = nullptr;
    sci::PRG128* prg;
    int party;

    TripleGenerator(int Party, IO* io, const sci::OTPack<IO>* otpack, bool enable_buffer = false,
                    uint64_t buffr_size = BSIZE, uint64_t chunk_size = CSIZE) {
        this->io     = io;
        this->otpack = otpack;
        this->prg    = new sci::PRG128;
#if PRG_SEED != -1
        seed = _mm256_set1_epi32(PRG_SEED + io->port);
        prg->reseed(&seed);
#endif

        if (enable_buffer) {
            this->party      = Party;
            this->_buffSize  = (buffr_size >> 3) > 1 ? ((buffr_size >> 3) << 3) : 8;
            this->_buffBytes = _buffSize >> 3;
            this->_chunkSize = (chunk_size >> 3) > 1 ? ((chunk_size >> 3) << 3) : 8;

            refillBuffer();
        }
    }

    ~TripleGenerator() {
        delete prg;
        // delete buffer
        if (_buffEnable) {
            if (_Bai)
                delete[] _Bai;
            if (_Bbi)
                delete[] _Bbi;
            if (_Bci)
                delete[] _Bci;
        }
    }

    inline bool isBufferEnabled() { return this->_buffEnable; }

    inline int getBufferPointer() { return this->_buffPtr; }

    inline double getMillTime() { return this->mill_time; }

    inline double getMillComm() { return this->mill_comm; }

    void addMillTime(double time) { this->mill_time += time; }

    void addMillComm(double comm) { this->mill_comm += comm; }

    /* Refill the buffer with new triples
      Resets the buffer size to num_triples
      Resets the chunk size to min(_buffSize, CSIZE)
      Generates triples in chunks of size _chunkSize with the specified method
      Resets the buffer pointer to 0
      Enables the buffer
    */
    void refillBuffer(TripleGenMethod method = _2ROT) {
        _Bai = new uint8_t[_buffBytes];
        _Bbi = new uint8_t[_buffBytes];
        _Bci = new uint8_t[_buffBytes];

        _chunkSize  = std::min(_buffSize, _chunkSize);
        int nChunks = ceil((double)_buffSize / _chunkSize);
        for (int i = 0; i < nChunks; i++) {
            int start      = i * _chunkSize;
            int startBytes = start / 8;
            int end        = std::min(_buffSize, (i + 1) * _chunkSize);
            generate(party, _Bai + startBytes, _Bbi + startBytes, _Bci + startBytes, end - start,
                     method, true);
        }
        _buffPtr    = 0;
        _buffEnable = true;
        _nRefill++;
    }

    /* get triples from the buffer
      - If the buffer is disabled or triples are not packed, generates triples using the specified
      method
      - Fills the input triples object with triples from the buffer
      - If the pointer reaches the end of the buffer, resets the buffer size to max(_buffSize,
      num_triples) and refills
      - Copies triples from the buffer to the input triples object and increments the buffer pointer
    */
    void get(int party, Triple* triples, TripleGenMethod method = _2ROT) {
        if ((!_buffEnable) || (!triples->packed)) {
            generate(party, triples->ai, triples->bi, triples->ci, triples->num_triples, method,
                     triples->packed, triples->offset);
            return;
        }
        if (_buffPtr + triples->num_triples < _buffSize) {
            int startBytes = _buffPtr >> 3;
            memcpy(triples->ai, _Bai + startBytes, triples->num_bytes);
            memcpy(triples->bi, _Bbi + startBytes, triples->num_bytes);
            memcpy(triples->ci, _Bci + startBytes, triples->num_bytes);
            _buffPtr += (triples->num_bytes) << 3;
        } else {
            int n_trips = triples->num_triples;
            int r_bytes = 0;
            if (_buffPtr < _buffSize) { // copy the remaining triples from the buffer
                r_bytes        = (_buffSize - _buffPtr) >> 3;
                int startBytes = _buffPtr >> 3;
                memcpy(triples->ai, _Bai + startBytes, r_bytes);
                memcpy(triples->bi, _Bbi + startBytes, r_bytes);
                memcpy(triples->ci, _Bci + startBytes, r_bytes);
                _buffPtr += (r_bytes << 3);
            }
            n_trips -= (r_bytes << 3);

            // generate triples in chunks of size _chunkSize
            int nChunks = ceil((double)n_trips / _chunkSize);
            for (int i = 0; i < nChunks; i++) {
                uint64_t start      = i * _chunkSize;
                uint64_t startBytes = start >> 3;
                uint64_t end        = static_cast<int64_t>((i + 1) * _chunkSize) < n_trips
                                          ? ((i + 1) * _chunkSize)
                                          : n_trips;

#if TGEN_PRINT_COMM || TGEN_PRINT_TIME
                std::string f_tag = "NEW | 3get";
#endif
#if TGEN_PRINT_COMM
                int _wcomm = 10;
                std::stringstream log_comm;
                uint64_t start_comm = io->counter;
                uint64_t total_comm = 0;
                log_comm << "P" << party << " COMM | ";
                log_comm << f_tag;
                log_comm << ": n_trips = " << n_trips;
                log_comm << ", r_bytes = " << r_bytes;
                log_comm << ", _buffPtr = " << _buffPtr;
                log_comm << ", _buffSize = " << _buffSize;
                log_comm << ", _chunkSize = " << _chunkSize;
                log_comm << std::endl;
#endif
#if TGEN_PRINT_TIME
                int _wtime = 10;
                std::stringstream log_time;
                auto start_time                          = std::chrono::system_clock::now();
                auto end_time                            = std::chrono::system_clock::now();
                std::chrono::duration<double> total_time = end_time - start_time;
                log_time << "P" << party << " TIME | ";
                log_time << f_tag;
                log_time << ": n_trips = " << n_trips;
                log_time << ", r_bytes = " << r_bytes;
                log_time << ", _buffPtr = " << _buffPtr;
                log_time << ", _buffSize = " << _buffSize;
                log_time << ", _chunkSize = " << _chunkSize;
                log_time << std::endl;
#endif
                generate(party, triples->ai + r_bytes + startBytes,
                         triples->bi + r_bytes + startBytes, triples->ci + r_bytes + startBytes,
                         end - start, method, true);
#if TGEN_PRINT_TIME
                end_time                                    = std::chrono::system_clock::now();
                std::chrono::duration<double> generate_time = end_time - (start_time + total_time);
                total_time += generate_time;
                log_time << "P" << party << " TIME | ";
                log_time << f_tag;
                log_time << ": generate = " << std::setw(_wtime) << generate_time.count() * 1000;
                log_time << " ms";
                log_time << std::endl;
#endif
#if TGEN_PRINT_COMM
                uint64_t generate_comm = io->counter - (start_comm + total_comm);
                total_comm += generate_comm;
                log_comm << "P" << party << " COMM | ";
                log_comm << f_tag;
                log_comm << ": generate = " << std::setw(_wcomm) << generate_comm;
                log_comm << " bytes";
                log_comm << std::endl;
#endif
#if TGEN_PRINT_COMM
                std::cout << log_comm.str();
#endif
#if TGEN_PRINT_TIME
                std::cout << log_time.str();
#endif
            }
            return;
        }
    }

    void generate(int party, Triple* triples, TripleGenMethod method) {
        // generate(party, triples->ai, triples->bi, triples->ci, triples->num_triples,
        //         method, triples->packed, triples->offset);
        get(party, triples, method);
    }

  public:
    // Buffer implementation to use pre-generated triples from the offline phase
    uint8_t* _Bai       = nullptr;    // Buffer for Ai
    uint8_t* _Bbi       = nullptr;    // Buffer for Bi
    uint8_t* _Bci       = nullptr;    // Buffer for Ci
    uint64_t _buffSize  = BSIZE;      // Number of triples in the buffer (always multiple of 8)
    uint64_t _buffBytes = BSIZE >> 3; // Number of bytes in the buffer (always = _buffSize/8)
    uint64_t _chunkSize
        = CSIZE;               // Number of triples to be generated in one go (always multiple of 8)
    bool _buffEnable  = false; // Flag to enable/disable buffer
    uint64_t _buffPtr = 0;     // Pointer to the current triple in the buffer
    int _nRefill      = 0;     // Number of times the buffer has been refilled
    double mill_time  = 0;     // Time taken for millioinaires' protocol
    double mill_comm  = 0;     // Communication in millionaires' protocol

    void generate(int party, uint8_t* ai, uint8_t* bi, uint8_t* ci, uint64_t num_triples,
                  TripleGenMethod method, bool packed = false, int offset = 1) {
        if (!num_triples)
            return;
        switch (method) {
        case Ideal: {
            int num_bytes = ceil((double)num_triples / 8);
            if (party == emp::ALICE) {
                uint8_t* a = new uint8_t[num_bytes];
                uint8_t* b = new uint8_t[num_bytes];
                uint8_t* c = new uint8_t[num_bytes];
                if (packed) {
                    prg->random_data(ai, num_bytes);
                    prg->random_data(bi, num_bytes);
                    prg->random_data(ci, num_bytes);
                } else {
                    prg->random_bool((bool*)ai, num_triples);
                    prg->random_bool((bool*)bi, num_triples);
                    prg->random_bool((bool*)ci, num_triples);
                }
                prg->random_data(a, num_bytes);
                prg->random_data(b, num_bytes);
                for (size_t i = 0; i < num_triples; i += 8) {
                    c[i / 8] = a[i / 8] & b[i / 8];
                    if (packed) {
                        a[i / 8] ^= ai[i / 8];
                        b[i / 8] ^= bi[i / 8];
                        c[i / 8] ^= ci[i / 8];
                    } else {
                        uint8_t temp_a, temp_b, temp_c;
                        if (num_triples - i >= 8) {
                            temp_a = bool_to_uint8(ai + i, 8);
                            temp_b = bool_to_uint8(bi + i, 8);
                            temp_c = bool_to_uint8(ci + i, 8);
                        } else {
                            temp_a = bool_to_uint8(ai + i, num_triples - i);
                            temp_b = bool_to_uint8(bi + i, num_triples - i);
                            temp_c = bool_to_uint8(ci + i, num_triples - i);
                        }
                        a[i / 8] ^= temp_a;
                        b[i / 8] ^= temp_b;
                        c[i / 8] ^= temp_c;
                    }
                }
                io->send_data(a, num_bytes);
                io->send_data(b, num_bytes);
                io->send_data(c, num_bytes);
                delete[] a;
                delete[] b;
                delete[] c;
            } else {
                if (packed) {
                    io->recv_data(ai, num_bytes);
                    io->recv_data(bi, num_bytes);
                    io->recv_data(ci, num_bytes);
                } else {
                    uint8_t* a = new uint8_t[num_bytes];
                    uint8_t* b = new uint8_t[num_bytes];
                    uint8_t* c = new uint8_t[num_bytes];
                    io->recv_data(a, num_bytes);
                    io->recv_data(b, num_bytes);
                    io->recv_data(c, num_bytes);

                    for (size_t i = 0; i < num_triples; i += 8) {
                        if (num_triples - i >= 8) {
                            uint8_to_bool(ai + i, a[i / 8], 8);
                            uint8_to_bool(bi + i, b[i / 8], 8);
                            uint8_to_bool(ci + i, c[i / 8], 8);
                        } else {
                            uint8_to_bool(ai + i, a[i / 8], num_triples - i);
                            uint8_to_bool(bi + i, b[i / 8], num_triples - i);
                            uint8_to_bool(ci + i, c[i / 8], num_triples - i);
                        }
                    }
                    delete[] a;
                    delete[] b;
                    delete[] c;
                }
            }
            break;
        }
        case _2ROT: {
            if (packed) { // bit-packed: 64 triples per word
                size_t nb = num_triples / 8;
                std::vector<uint8_t> u(nb), v(nb);
                switch (party) {
                case emp::ALICE:
                    otpack->silent_ot_reversed->recv_rot_bits(u.data(), ai, num_triples);
                    otpack->silent_ot->send_rot_bits(v.data(), bi, num_triples);
                    io->flush();
                    break;
                case emp::BOB:
                    otpack->silent_ot_reversed->send_rot_bits(v.data(), bi, num_triples);
                    io->flush();
                    otpack->silent_ot->recv_rot_bits(u.data(), ai, num_triples);
                    break;
                }
                // b = m0 ^ m1, c = (a & b) ^ u ^ v, as below
                size_t nw = nb / 8;
                auto* a64 = reinterpret_cast<uint64_t*>(ai);
                auto* b64 = reinterpret_cast<uint64_t*>(bi);
                auto* c64 = reinterpret_cast<uint64_t*>(ci);
                auto* u64 = reinterpret_cast<const uint64_t*>(u.data());
                auto* v64 = reinterpret_cast<const uint64_t*>(v.data());
                for (size_t i = 0; i < nw; i++) {
                    uint64_t bw = b64[i] ^ v64[i];
                    b64[i]      = bw;
                    c64[i]      = (a64[i] & bw) ^ u64[i] ^ v64[i];
                }
                for (size_t i = nw * 8; i < nb; i++) {
                    bi[i] ^= v[i];
                    ci[i] = (ai[i] & bi[i]) ^ u[i] ^ v[i];
                }
                break;
            }
            // #if USE_CHEETAH
            uint8_t *a, *b, *c;
            if (packed) {
                a = new uint8_t[num_triples];
                b = new uint8_t[num_triples];
                c = new uint8_t[num_triples];
            } else {
                a = ai;
                b = bi;
                c = ci;
            }
            uint8_t *u, *v;
            u = new uint8_t[num_triples];
            v = new uint8_t[num_triples];
            switch (party) {
            case emp::ALICE: {
                otpack->silent_ot_reversed->template recv_ot_rm_rc<uint8_t>(u, (bool*)a,
                                                                            num_triples, 1);
                otpack->silent_ot->send_ot_rm_rc(v, b, num_triples, 1);
                io->flush();
                break;
            }
            case emp::BOB: {
                otpack->silent_ot_reversed->template send_ot_rm_rc<uint8_t>(v, b, num_triples, 1);
                io->flush();
                otpack->silent_ot->recv_ot_rm_rc(u, (bool*)a, num_triples, 1);
                break;
            }
            }

            for (size_t i = 0; i < num_triples; i++) b[i] = b[i] ^ v[i];
            for (size_t i = 0; i < num_triples; i++) c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];

            delete[] u;
            delete[] v;
            if (packed) {
                for (size_t i = 0; i < num_triples; i += 8) {
                    ai[i / 8] = bool_to_uint8(a + i, 8);
                    bi[i / 8] = bool_to_uint8(b + i, 8);
                    ci[i / 8] = bool_to_uint8(c + i, 8);
                }
                delete[] a;
                delete[] b;
                delete[] c;
            }
            // #else
            //           throw std::invalid_argument("To be implemented");
            // #endif
            break;
        }
        case _2COT: {
            uint8_t *a, *b, *c;
            if (packed) {
                a = new uint8_t[num_triples];
                b = new uint8_t[num_triples];
                c = new uint8_t[num_triples];
            } else {
                a = ai;
                b = bi;
                c = ci;
            }
            prg->random_bool((bool*)a, num_triples);
            prg->random_bool((bool*)b, num_triples);
            uint8_t *u, *v;
            u = new uint8_t[num_triples];
            v = new uint8_t[num_triples];
            switch (party) {
            case emp::ALICE: {
                otpack->silent_ot_reversed->template recv_ot_rm_rc<uint8_t>(u, (bool*)a,
                                                                            num_triples, 1);
                otpack->silent_ot->send_ot_rm_rc(v, b, num_triples, 1);
                io->flush();
                break;
            }
            case emp::BOB: {
                otpack->silent_ot_reversed->template send_ot_rm_rc<uint8_t>(v, b, num_triples, 1);
                io->flush();
                otpack->silent_ot->recv_ot_rm_rc(u, (bool*)a, num_triples, 1);
                break;
            }
            }

            for (size_t i = 0; i < num_triples; i++) b[i] = b[i] ^ v[i];
            for (size_t i = 0; i < num_triples; i++) c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];

            delete[] u;
            delete[] v;
            if (packed) {
                for (size_t i = 0; i < num_triples; i += 8) {
                    ai[i / 8] = bool_to_uint8(a + i, 8);
                    bi[i / 8] = bool_to_uint8(b + i, 8);
                    ci[i / 8] = bool_to_uint8(c + i, 8);
                }
                delete[] a;
                delete[] b;
                delete[] c;
            }
            break;
        }
        case _16KKOT_to_4OT: {
            assert((num_triples & 1) == 0); // num_triples is even
            uint8_t *a, *b, *c;
            if (packed) {
                int num_bytes = ceil((double)num_triples / 8);
                a             = new uint8_t[num_triples];
                b             = new uint8_t[num_triples];
                c             = new uint8_t[num_triples];

                uint8_to_bool_arr(a, ai, num_bytes);
                uint8_to_bool_arr(b, bi, num_bytes);
            } else {
                a = ai;
                b = bi;
                c = ci;
            }
            // prg->random_bool((bool*)a, num_triples);
            // prg->random_bool((bool*)b, num_triples);
            switch (party) {
            case emp::ALICE: {
                prg->random_bool((bool*)c, num_triples);
                uint8_t** ot_messages; // (num_triples/2) X 16
                ot_messages = new uint8_t*[num_triples / 2];
                for (size_t i = 0; i < num_triples; i += 2) ot_messages[i / 2] = new uint8_t[16];
                for (int j = 0; j < 16; j++) {
                    uint8_t bits_j[4]; // a01 || b01 || a11 || b11 (LSB->MSB)
                    uint8_to_bool(bits_j, j, 4);
                    for (size_t i = 0; i < num_triples; i += 2) {
                        ot_messages[i / 2][j]
                            = ((((a[i + 1] ^ bits_j[2]) & (b[i + 1] ^ bits_j[3])) ^ c[i + 1]) << 1)
                              | (((a[i] ^ bits_j[0]) & (b[i] ^ bits_j[1])) ^ c[i]);
                    }
                }
                // otpack->kkot_16->send(ot_messages, num_triples/2, 2);
                otpack->kkot[3]->send(ot_messages, num_triples / 2, 2);
                io->flush();
                for (size_t i = 0; i < num_triples; i += 2) delete[] ot_messages[i / 2];
                delete[] ot_messages;
                break;
            }
            case emp::BOB: {
                uint8_t* ot_selection = new uint8_t[(size_t)num_triples / 2];
                uint8_t* ot_result    = new uint8_t[(size_t)num_triples / 2];
                for (size_t i = 0; i < num_triples; i += 2) {
                    ot_selection[i / 2] = (b[i + 1] << 3) | (a[i + 1] << 2) | (b[i] << 1) | a[i];
                }
                // otpack->kkot_16->recv(ot_result, ot_selection, num_triples/2, 2);
                otpack->kkot[3]->recv(ot_result, ot_selection, num_triples / 2, 2);
                for (size_t i = 0; i < num_triples; i += 2) {
                    c[i]     = ot_result[i / 2] & 1;
                    c[i + 1] = ot_result[i / 2] >> 1;
                }
                delete[] ot_selection;
                delete[] ot_result;
                break;
            }
            }
            if (packed) {
                for (size_t i = 0; i < num_triples; i += 8) {
                    // ai[i / 8] = bool_to_uint8(a + i, 8);
                    // bi[i / 8] = bool_to_uint8(b + i, 8);
                    ci[i / 8] = bool_to_uint8(c + i, 8);
                }
                delete[] a;
                delete[] b;
                delete[] c;
            }
            break;
        }
        case _8KKOT: {
            assert((num_triples & 1) == 0); // num_triples is even
            uint8_t *a, *b, *c;
            if (packed) {
                a = new uint8_t[num_triples];
                b = new uint8_t[num_triples];
                c = new uint8_t[num_triples];
            } else {
                a = ai;
                b = bi;
                c = ci;
            }
            for (size_t i = 0; i < num_triples; i += 2 * offset) {
                prg->random_bool((bool*)a + i, offset);
                memcpy(a + i + offset, a + i, offset);
            }
            prg->random_bool((bool*)b, num_triples);
            switch (party) {
            case emp::ALICE: {
                prg->random_bool((bool*)c, num_triples);
                uint8_t** ot_messages; // (num_triples/2) X 8
                ot_messages = new uint8_t*[num_triples / 2];
                for (size_t i = 0; i < num_triples; i += 2) ot_messages[i / 2] = new uint8_t[8];
                for (int j = 0; j < 8; j++) {
                    uint8_t bits_j[3]; // a01 || b01 || b11 (LSB->MSB)
                    uint8_to_bool(bits_j, j, 3);
                    for (size_t i = 0; i < num_triples; i += 2 * offset) {
                        for (int k = 0; k < offset; k++) {
                            ot_messages[i / 2 + k][j]
                                = ((((a[i + k] ^ bits_j[0]) & (b[i + offset + k] ^ bits_j[2]))
                                    ^ c[i + offset + k])
                                   << 1)
                                  | (((a[i + k] ^ bits_j[0]) & (b[i + k] ^ bits_j[1])) ^ c[i + k]);
                        }
                    }
                }
                // otpack->kkot_8->send(ot_messages, num_triples/2, 2);
                otpack->kkot[2]->send(ot_messages, num_triples / 2, 2);
                for (size_t i = 0; i < num_triples; i += 2) delete[] ot_messages[i / 2];
                delete[] ot_messages;
                break;
            }
            case emp::BOB: {
                uint8_t* ot_selection = new uint8_t[(size_t)num_triples / 2];
                uint8_t* ot_result    = new uint8_t[(size_t)num_triples / 2];
                for (size_t i = 0; i < num_triples; i += 2 * offset) {
                    for (int k = 0; k < offset; k++)
                        ot_selection[i / 2 + k]
                            = (b[i + offset + k] << 2) | (b[i + k] << 1) | a[i + k];
                }
                // otpack->kkot_8->recv(ot_result, ot_selection, num_triples/2, 2);
                otpack->kkot[2]->recv(ot_result, ot_selection, num_triples / 2, 2);
                for (size_t i = 0; i < num_triples; i += 2 * offset) {
                    for (int k = 0; k < offset; k++) {
                        c[i + k]          = ot_result[i / 2 + k] & 1;
                        c[i + offset + k] = ot_result[i / 2 + k] >> 1;
                    }
                }
                delete[] ot_selection;
                delete[] ot_result;
                break;
            }
            }
            if (packed) {
                for (size_t i = 0; i < num_triples; i += 8) {
                    ai[i / 8] = bool_to_uint8(a + i, 8);
                    bi[i / 8] = bool_to_uint8(b + i, 8);
                    ci[i / 8] = bool_to_uint8(c + i, 8);
                }
                delete[] a;
                delete[] b;
                delete[] c;
            }
            break;
        }
        }
    }

    inline uint8_t bool_to_uint8(const uint8_t* data, size_t len) {
        if (len != 0)
            len = (len > 8 ? 8 : len);
        else
            len = 8;
        uint8_t res = 0;
        for (size_t i = 0; i < len; ++i) {
            if (data[i])
                res |= (1ULL << i);
        }
        return res;
    }

    inline void uint8_to_bool(uint8_t* data, uint8_t input, int length) {
        for (int i = 0; i < length; ++i) {
            data[i] = (input & 1);
            input >>= 1;
        }
    }

    inline void uint8_to_bool_arr(uint8_t* data, uint8_t* input, int length) {
        for (int i = 0; i < length; i++) {
            uint8_to_bool(data + i * 8, input[i], 8);
        }
    }
};

#endif // TRIPLE_GENERATOR_H__
