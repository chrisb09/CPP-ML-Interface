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
// hdf=50, increment=24. Unsafe remainder: 0 or >= 26.
// inference_start_step=25: after inference at call 25, next_logical=30,
// global=30, 30%50=30 >= 26 → unsafe → shift to 30+(50-29)=51.

static void test_hdf_avoidance() {
    MLCouplingBehaviorFlowExtrapolator beh(5, 3, 12, 50, 300, 1.0, 2, 1, 25, 0);

    int coupled = 3, stride = 1;

    for (int call = 1; call <= 24; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    // Call 25: inference fires + records HDF shift
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        int delta = beh.time_step_delta();
        if (!send || !inf || delta != 24) {
            report(false, ("hdf: call 25 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)
                           + " delta=" + std::to_string(delta)).c_str());
            return;
        }
    }
    report(true, "hdf: call 25 inference fires");

    // Calls 26-50: should not fire inference (shifted to 51).
    // Send when dist = 51 - (logical+1) < 3*1 && >=0 && %1==0.
    // At logical=25 (after inference), calls 26..50 have dist = 51-(logical+1):
    //   logical 25 → dist 25, not < 3 → no send
    //   logical 26 → dist 24, not < 3 → no send
    //   ...
    //   logical 48 → dist 2, < 3 → send
    //   logical 49 → dist 1, < 3 → send
    //   logical 50 → dist 0 → send
    // So sends only at calls 49, 50 (and 51).
    for (int call = 26; call <= 50; ++call) {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        // projection: logical_before = call-1 (since each call increments once)
        long long int projected = static_cast<long long int>(call);
        long long int dist = 51 - projected;
        bool send_expected = (dist >= 0 && dist < coupled * stride && dist % stride == 0);
        if (send != send_expected || inf) {
            report(false, ("hdf: call " + std::to_string(call)
                           + " send=" + std::to_string(send)
                           + " (expected " + std::to_string(send_expected) + ")"
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "hdf: calls 26-50 no inference, correct sends");

    // Call 51: shifted inference fires
    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, ("hdf: call 51 send=" + std::to_string(send)
                           + " inf=" + std::to_string(inf)).c_str());
            return;
        }
    }
    report(true, "hdf: call 51 shifted inference fires");
}

// ---------------------------------------------------------------------------
// Test 4: global_step_offset shifts HDF check
// ---------------------------------------------------------------------------

static void test_global_step_offset() {
    MLCouplingBehaviorFlowExtrapolator beh(5, 3, 1, 50, 300, 1.0, 1, 1, 25, 100);

    for (int call = 1; call <= 24; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, "offset: call 25 inference with offset=100");
            return;
        }
    }
    report(true, "offset: call 25 inference with offset=100");

    for (int call = 26; call <= 50; ++call) {
        beh.should_send_data();
        beh.should_perform_inference();
    }

    {
        bool send = beh.should_send_data();
        bool inf = beh.should_perform_inference();
        if (!send || !inf) {
            report(false, "offset: call 51 shifted inference with offset");
            return;
        }
    }
    report(true, "offset: call 51 shifted inference with offset");
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
// Main
// ---------------------------------------------------------------------------

int main() {
    test_basic_timing();
    test_stride();
    test_hdf_avoidance();
    test_global_step_offset();
    test_end_of_simulation();

    if (g_any_failure) {
        std::cerr << "\nSome tests FAILED.\n";
        return 1;
    }
    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
