// Author: Zhicong Huang
#ifndef CHEETAH_OT_PACK_H__
#define CHEETAH_OT_PACK_H__

#include "ot/silent_ot.h"

#define KKOT_TYPES 8

#define PRE_OT_DATA_REG_SEND_FILE_ALICE "./data/pre_ot_data_reg_send_alice_"
#define PRE_OT_DATA_REG_SEND_FILE_BOB "./data/pre_ot_data_reg_send_bob_"
#define PRE_OT_DATA_REG_RECV_FILE_ALICE "./data/pre_ot_data_reg_recv_alice_"
#define PRE_OT_DATA_REG_RECV_FILE_BOB "./data/pre_ot_data_reg_recv_bob_"

namespace sci {

template <typename T>
class OTPack {
  public:
    cheetah::SilentOT<T>* silent_ot          = nullptr;
    cheetah::SilentOT<T>* silent_ot_reversed = nullptr;

    cheetah::SilentOTN<T>* kkot[KKOT_TYPES] = {nullptr};

    // iknp_straight and iknp_reversed: party
    // acts as sender in straight and receiver in reversed.
    // Needed for MUX calls.
    cheetah::SilentOT<T>* iknp_straight = nullptr;
    cheetah::SilentOT<T>* iknp_reversed = nullptr;
    T* io                               = nullptr;
    int party;
    // bool do_setup = false;

    OTPack(T** ios, int threads, int party, bool do_setup = true, bool malicious = true) {
        // std::cout << "using silent ot pack" << std::endl;

        this->party = party;
        // this->do_setup = do_setup;
        this->io = ios[0];

        auto post_fix = std::to_string(ios[0]->port);

        silent_ot = new cheetah::SilentOT<T>(party, threads, ios, false, true,
                                             party == emp::ALICE
                                                 ? PRE_OT_DATA_REG_SEND_FILE_ALICE + post_fix
                                                 : PRE_OT_DATA_REG_RECV_FILE_BOB + post_fix);

        silent_ot_reversed = new cheetah::SilentOT<T>(
            3 - party, threads, ios, false, true,
            party == emp::ALICE ? PRE_OT_DATA_REG_RECV_FILE_ALICE + post_fix
                                : PRE_OT_DATA_REG_SEND_FILE_BOB + post_fix);

        for (int i = 0; i < KKOT_TYPES; i++) {
            kkot[i] = new cheetah::SilentOTN<T>(silent_ot, 1 << (i + 1));
        }

        iknp_straight = silent_ot;
        iknp_reversed = silent_ot_reversed;
    }

    ~OTPack() {
        if (silent_ot)
            delete silent_ot;
        for (int i = 0; i < KKOT_TYPES; i++) {
            if (kkot[i])
                delete kkot[i];
        }
        if (iknp_reversed)
            delete iknp_reversed;
    }

    void SetupBaseOTs() {}

    /*
     * DISCLAIMER:
     * OTPack copy method avoids computing setup keys for each OT instance by
     * reusing the keys generated (through base OTs) for another OT instance.
     * Ideally, the PRGs within OT instances, using the same keys, should use
     * mutually exclusive counters for security. However, the current
     * implementation does not support this.
     */

    // void copy(OTPack<T> *copy_from) {
    // assert(this->do_setup == false && copy_from->do_setup == true);
    // SplitKKOT<T> *kkot_base = copy_from->kkot[0];
    // SplitIKNP<T> *iknp_s_base = copy_from->iknp_straight;
    // SplitIKNP<T> *iknp_r_base = copy_from->iknp_reversed;

    // switch (this->party) {
    // case 1:
    // for (int i = 0; i < KKOT_TYPES; i++) {
    // this->kkot[i]->setup_send(kkot_base->k0, kkot_base->s);
    //}
    // this->iknp_straight->setup_send(iknp_s_base->k0, iknp_s_base->s);
    // this->iknp_reversed->setup_recv(iknp_r_base->k0, iknp_r_base->k1);
    // break;
    // case 2:
    // for (int i = 0; i < KKOT_TYPES; i++) {
    // this->kkot[i]->setup_recv(kkot_base->k0, kkot_base->k1);
    //}
    // this->iknp_straight->setup_recv(iknp_s_base->k0, iknp_s_base->k1);
    // this->iknp_reversed->setup_send(iknp_r_base->k0, iknp_r_base->s);
    // break;
    //}
    // this->do_setup = true;
    // return;
    //}
};

} // namespace sci
#endif // CHEETAH_OT_PACK_H__
