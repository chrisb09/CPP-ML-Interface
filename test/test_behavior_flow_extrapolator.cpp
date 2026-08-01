/**
 * test_behavior_flow_extrapolator.cpp
 *
 * Unit test for MLCouplingBehaviorFlowExtrapolator.
 * Tests coupling/inference timing, stride, HDF-avoidance, and end-of-sim.
 * Calls should_send_data() before should_perform_inference() each iteration,
 * matching the application's ml_step() order.
 */

#include <cmath>
#include <iostream>

#include "behavior/ml_coupling_behavior_flow_extrapolator.hpp"

static bool g_any_failure = false;

static void report(bool ok, const char* label) {
    if (ok) {
        std::cout << "[PASS] " << label << "\n";
    } else {
        std::cerr << "[FAIL] " << label << "\n";
        g_any_failure = true;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Basic timing — stride=1, no HDF interference
// ---------------------------------------------------------------------------
// inference_interval=5, coupled=5, increment=12, total=300,
// hdf=300 (never fires), scaling=1, fw=2, stride=1, start=5
//
// Expected: send on calls 1-5, inference at 5 and 10, ... with delta=24.

static void test_basic_timing() {
    MLCouplingBehaviorFlowExtrapolator beh(5, 5, 12, 300, 300, 1.0, 2, 1, 5, 0);

    // Calls 1-4: send, no inference
    for (int call = 1; call <= 4; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || inf) {
            report(false, ("basic: call " + std::to_string(call)
                           + " send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "basic: calls 1-4 send, no inf");

    // Call 5: send + inference
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        int delta = beh.time_step_delta();
        if (!send || !inf || delta != 24) {
            report(false, ("basic: call 5 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)
                           + " delta=" + std::to_string(delta)).c_str());
            return;
        }
    }
    report(true, "basic: call 5 inference delta=24");

    // Calls 6-9: send, no inference (next_inference_step=10)
    for (int call = 6; call <= 9; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || inf) {
            report(false, ("basic: call " + std::to_string(call)
                           + " send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "basic: calls 6-9 send, no inf");

    // Call 10: send + inference
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("basic: call 10 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "basic: call 10 inference");
}

// ---------------------------------------------------------------------------
// Test 2: Stride — coupled_steps_before_inference * stride spacing
// ---------------------------------------------------------------------------
// inference_interval=10, coupled=3, increment=1, scaling=2.5, fw=1,
// input_step_distance=3 → stride=8, start=20.
// Expected: sends at calls 4 (dist=16), 12 (dist=8), 20 (dist=0 + inf).

static void test_stride() {
    MLCouplingBehaviorFlowExtrapolator beh(10, 3, 1, 100, 100, 2.5, 1, 3, 20, 0);

    // Calls 1-3: no send (dist=19,18,17 — none %8==0)
    for (int call = 1; call <= 3; ++call) {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (send || inf) {
            report(false, ("stride: call " + std::to_string(call)
                           + " unexpected send=" + std::to_string(send)).c_str());
            return;
        }
    }
    report(true, "stride: calls 1-3 no send");

    // Call 4: send only (dist=16, 16%8==0)
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || inf) {
            report(false, "stride: call 4 should send");
            return;
        }
    }
    report(true, "stride: call 4 send");

    // Calls 5-11: no send (dist=15..9, none %8==0)
    for (int call = 5; call <= 11; ++call) {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (send || inf) {
            report(false, ("stride: call " + std::to_string(call)
                           + " unexpected send=" + std::to_string(send)).c_str());
            return;
        }
    }
    report(true, "stride: calls 5-11 no send");

    // Call 12: send only (dist=8, 8%8==0)
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || inf) {
            report(false, "stride: call 12 should send");
            return;
        }
    }
    report(true, "stride: call 12 send");

    // Calls 13-19: no send (dist=7..1, none %8==0)
    for (int call = 13; call <= 19; ++call) {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (send || inf) {
            report(false, ("stride: call " + std::to_string(call)
                           + " unexpected send=" + std::to_string(send)).c_str());
            return;
        }
    }
    report(true, "stride: calls 13-19 no send");

    // Call 20: send + inference (dist=0)
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, "stride: call 20 should send+inf");
            return;
        }
    }
    report(true, "stride: call 20 send+inf");
}

// ---------------------------------------------------------------------------
// Test 3: HDF-avoidance shift
// ---------------------------------------------------------------------------
// params: interval=5, coupled=3, increment=12, hdf=50, total=300,
//         scaling=1, fw=2 → delta=24, start=25, offset=0
//
// With effective_global_step_ tracking:
//   Call 25: eff=25, next_global = 25+24+5 = 54, 54%50=4 < 26 → SAFE → next_inf=30
//   Call 30: eff=54, next_global = 54+24+5 = 83, 83%50=33 ≥ 26 → UNSAFE
//            next_hdf=100, safe=101, delay=18 → next_inf=30+5+18=53
//   Call 53: eff=101 → fires

static void test_hdf_avoidance() {
    MLCouplingBehaviorFlowExtrapolator beh(5, 3, 12, 50, 300, 1.0, 2, 1, 25, 0);

    for (int call = 1; call <= 24; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    // Call 25: inference fires (eff_global=25, next_global=54 is safe)
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        int delta = beh.time_step_delta();
        if (!send || !inf || delta != 24) {
            report(false, ("hdf: call 25 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)
                           + " delta=" + std::to_string(delta)).c_str());
            return;
        }
    }
    report(true, "hdf: call 25 inference fires (next_global=54 safe)");

    // Calls 26-29: no inference (next_inf=30)
    for (int call = 26; call <= 29; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (inf) {
            report(false, ("hdf: call " + std::to_string(call) + " unexpected inf").c_str());
            return;
        }
    }
    report(true, "hdf: calls 26-29 no inference");

    // Call 30: inference fires (eff_global=54, next_global=83 UNSAFE → next_inf=53)
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("hdf: call 30 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "hdf: call 30 inference fires (next_global=83 unsafe, next_inf shifts to 53)");

    // Calls 31-52: no inference (next_inf=53)
    // Sends when dist = 53 - projected < 3: projected=call (before increment), dist=53-call
    for (int call = 31; call <= 52; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        long long int dist = 53 - static_cast<long long int>(call);
        bool send_expected = (dist >= 0 && dist < 3 && dist % 1 == 0);
        if (send != send_expected || inf) {
            report(false, ("hdf: call " + std::to_string(call)
                           + " send=" + std::to_string(send)
                           + " (expected " + std::to_string(send_expected) + ")"
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "hdf: calls 31-52 no inference, correct sends");

    // Call 53: shifted inference fires
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("hdf: call 53 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "hdf: call 53 shifted inference fires");
}

// ---------------------------------------------------------------------------
// Test 4: global_step_offset shifts HDF check
// ---------------------------------------------------------------------------
// params: interval=5, coupled=3, increment=1, hdf=50, total=300,
//         scaling=1, fw=1 → delta=1, start=25, offset=100
//
// With effective_global_step_ tracking (eff starts at offset=100):
//   Call 25: eff=125, next_global=125+1+5=131, 131%50=31 < 49 → safe → next_inf=30
//   Call 30: eff=131, next_global=137, 137%50=37 < 49 → safe → next_inf=35
//   Call 35: eff=137, next_global=143, 143%50=43 < 49 → safe → next_inf=40
//   Call 40: eff=143, next_global=149, 149%50=49, NOT < 49 → UNSAFE
//            next_hdf=150, safe=151, delay=2 → next_inf=47
//   Call 47: fires

static void test_global_step_offset() {
    MLCouplingBehaviorFlowExtrapolator beh(5, 3, 1, 50, 300, 1.0, 1, 1, 25, 100);

    for (int call = 1; call <= 24; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    // Calls 25, 30, 35: all safe inferences
    for (int expected_call : {25, 30, 35}) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("offset: call " + std::to_string(expected_call)
                           + " send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
        // Drain calls between inferences (4 non-inference calls each)
        if (expected_call < 35) {
            for (int i = 0; i < 4; ++i) {
                beh.should_send_data();
                beh.should_perform_inference();
            }
        }
    }
    report(true, "offset: calls 25, 30, 35 inference (all safe with offset=100)");

    // Calls 36-39: not inference
    for (int call = 36; call <= 39; ++call) {
        beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (inf) {
            report(false, ("offset: call " + std::to_string(call) + " unexpected inf").c_str());
            return;
        }
    }

    // Call 40: inference fires but next_global=149 UNSAFE → next_inf shifts to 47
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("offset: call 40 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "offset: call 40 inference fires (next_global=149 unsafe, shifts to 47)");

    // Calls 41-46: no inference (sends when dist<3: calls 45,46)
    for (int call = 41; call <= 46; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        long long int dist = 47 - static_cast<long long int>(call);
        bool send_expected = (dist >= 0 && dist < 3);
        if (send != send_expected || inf) {
            report(false, ("offset: call " + std::to_string(call)
                           + " send=" + std::to_string(send)
                           + " (expected " + std::to_string(send_expected) + ")"
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }

    // Call 47: shifted inference fires
    {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("offset: call 47 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "offset: call 47 shifted inference fires");
}

// ---------------------------------------------------------------------------
// Test 5: End-of-simulation
// ---------------------------------------------------------------------------

static void test_end_of_simulation() {
    // total=20, interval=10, start=10
    MLCouplingBehaviorFlowExtrapolator beh(10, 2, 1, 50, 20, 1.0, 1, 1, 10, 0);

    for (int call = 1; call <= 9; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    // Call 10: last inference, next=20, next_logical+1=21 >= 20 → end
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        int delta = beh.time_step_delta();
        if (!send || !inf || delta != 1) {
            report(false, ("eos: call 10 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)
                           + " delta=" + std::to_string(delta)).c_str());
            return;
        }
    }
    report(true, "eos: call 10 last inference delta=1");

    // Call 11: no send, no inf (past end)
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (send || inf) {
            report(false, ("eos: call 11 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "eos: call 11 no send, no inf (past end)");
}

// ---------------------------------------------------------------------------
// Test 6: Cumulative-delta regression — effective_global_step_ tracking
// ---------------------------------------------------------------------------
// Reproduces the MMCP artifact bug: without effective_global_step_ the 3rd
// inference cycle computes next_global ≈ 30 instead of ~101, causing a
// spurious HDF-unsafe shift.
//
// Parameters: interval=5, coupled=5, increment=12, hdf=50, total=300,
//             scaling=1, fw=2 → delta=24, start=5, offset=10
// Derived timing (correct):
//   Inference 1: logical=5,  eff_global=15  → next eff_global=44 → unsafe (44%50=44 ≥ 26)
//                → delay=7 → next_inference_logical=17
//   Inference 2: logical=17, eff_global=51  → next eff_global=80 → unsafe (80%50=30 ≥ 26)
//                → delay=21 → next_inference_logical=43
//   Inference 3: logical=43, eff_global=101 → next eff_global=130 → unsafe (130%50=30 ≥ 26)
//                → delay=21 → next_inference_logical=69
//
// The old (buggy) code used next_logical + global_step_offset without
// tracking cumulative jumps, so it computed next_global≈30 at cycle 3
// and applied the wrong delay.

static void test_cumulative_delta_regression() {
    // artifact params: interval=5, coupled=5, increment=12, hdf=50,
    //                  total=300, scaling=1, fw=2, stride=1, start=5, offset=10
    MLCouplingBehaviorFlowExtrapolator beh(5, 5, 12, 50, 300, 1.0, 2, 1, 5, 10);

    // Expected inference logical steps derived from Python simulation above.
    // These are the only calls on which should_perform_inference() must return true.
    const long long int expected_inf[] = {5, 17, 43, 69, 95};
    const int n_expected = 5;
    int inf_idx = 0;

    bool ok = true;
    for (int call = 1; call <= 110 && ok; ++call) {
        bool send = beh.should_send_data();
        bool inf  = beh.should_perform_inference();

        if (inf) {
            if (inf_idx >= n_expected) {
                report(false, ("cumulative: unexpected extra inference at call "
                               + std::to_string(call)).c_str());
                ok = false;
                break;
            }
            if (call != expected_inf[inf_idx]) {
                report(false, ("cumulative: inference " + std::to_string(inf_idx + 1)
                               + " expected at call " + std::to_string(expected_inf[inf_idx])
                               + " but fired at call " + std::to_string(call)).c_str());
                ok = false;
                break;
            }
            ++inf_idx;
        }
    }

    if (ok) {
        if (inf_idx != n_expected) {
            report(false, ("cumulative: only " + std::to_string(inf_idx)
                           + " of " + std::to_string(n_expected)
                           + " expected inferences fired").c_str());
        } else {
            report(true, "cumulative: all 5 inferences at correct logical steps (5,17,43,69,95 via eff_global)");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    test_basic_timing();
    test_stride();
    test_hdf_avoidance();
    test_global_step_offset();
    test_end_of_simulation();
    test_cumulative_delta_regression();

    if (g_any_failure) {
        std::cerr << "\nSome tests FAILED.\n";
        return 1;
    }
    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
