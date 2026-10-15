// Standalone, dependency-free test. Compile with -DUSE_SCOREP and
// -Itest/scorep_stub to check region ancestry without the Score-P runtime.
#include <cassert>
#include <iostream>

#include "application/ml_coupling_application_flow_extrapolator.hpp"
#include "behavior/ml_coupling_behavior_flow_extrapolator.hpp"

using Data = MLCouplingData<float>;
using Flow = MLCouplingApplicationFlowExtrapolator<float, float>;

static Data fields()
{
    std::vector<MLCouplingTensor<float>> tensors;
    for (int field = 0; field < 3; ++field) {
        tensors.push_back(MLCouplingTensor<float>::from_flat_copy(std::vector<float>(8, 0), {2, 2, 2}));
    }
    return Data(std::move(tensors));
}

struct Provider : MLCouplingLibrary<float, float> {
    Provider() : MLCouplingLibrary<float, float>(0) {}
    std::vector<std::vector<float>> inputs;
    void static_inference(Data* input, Data* output) override
    {
        // Model an unconditional external AIX region: only its ancestor selects it.
#ifdef USE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("aix_internal", SCOREP_USER_REGION_TYPE_COMMON)
#endif
        const float* src = static_cast<const float*>((*input)[0].root());
        inputs.emplace_back(src, src + (*input)[0].numel());
        float* dst = static_cast<float*>((*output)[0].root());
        std::fill_n(dst, (*output)[0].numel(), static_cast<float>(inputs.size()));
#ifdef USE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("aix_internal")
#endif
    }
};

struct ObservedFlow : Flow {
    using Flow::Flow;
    int prepared = 0;
    Data preprocess_coupling_input(Data input) override
    {
        ++prepared;
        return Flow::preprocess_coupling_input(std::move(input));
    }
};

struct ObservedBehavior : MLCouplingBehavior {
    MLCouplingBehaviorFlowExtrapolator behavior{5, 5, 12, 50, 300, 1.0, 2, 1, 5, 10};
    ObservedFlow& app;
    int send_queries = 0;
    int inference_queries = 0;
    int prepared_before = 0;
    explicit ObservedBehavior(ObservedFlow& app) : app(app) {}
    bool should_send_data() override
    {
        assert(send_queries == inference_queries);
        prepared_before = app.prepared;
        ++send_queries;
        return behavior.should_send_data();
    }
    bool should_perform_inference() override
    {
        assert(send_queries == inference_queries + 1);
        assert(app.prepared == prepared_before);
        ++inference_queries;
        return behavior.should_perform_inference();
    }
    int time_step_delta() override { return behavior.time_step_delta(); }
};

static std::vector<std::vector<float>> run_flow(bool configured, bool manual_gate)
{
    ml_coupling_scorep::set_detailed_regions_enabled(manual_gate);
    ObservedFlow app(fields(), fields(), nullptr, 2, 0, 5, 2);
    MLCouplingApplication<float, float>& base = app;
    ObservedBehavior behavior(app);
    Provider provider;
    if (configured) app.set_scorep_warmup_inference_count(2);
#ifdef USE_SCOREP
    scorep_test::reset(); // Initialization regions are deliberately outside selection.
#endif
    const std::vector<int> expected{5, 17, 43, 69, 95};
    std::vector<int> inference_calls;
    for (int call = 1; call <= 110; ++call) {
        for (auto& tensor : app.coupling_input) {
            std::fill_n(static_cast<float*>(tensor.root()), tensor.numel(), static_cast<float>(call));
        }
        const int delta = base.ml_step(provider, behavior);
        if (delta) {
            assert(delta == 24);
            inference_calls.push_back(call);
            assert(static_cast<float*>(app.coupling_output[0].root())[0] == inference_calls.size());
        }
        assert(ml_coupling_scorep::detailed_regions_are_enabled() == manual_gate);
#ifdef USE_SCOREP
        assert(scorep_test::count("cmi_inference_steady")
            == (configured ? std::max(0, static_cast<int>(inference_calls.size()) - 2) : 0));
#endif
    }
    assert(inference_calls == expected);
    assert(behavior.send_queries == 110 && behavior.inference_queries == 110);
    assert(app.prepared == 25);
#ifdef USE_SCOREP
    assert(scorep_test::stack.empty());
    assert(scorep_test::count("cmi_inference_steady") == (configured ? 3 : 0));
    assert(scorep_test::count("aix_internal") == 5);
    assert(scorep_test::count("aix_internal", true) == (configured ? 3 : 0));
    for (const char* name : {"app_provider_inference", "app_finalize_output"}) {
        assert(scorep_test::count(name) == (configured ? 3 : (manual_gate ? 5 : 0)));
        assert(scorep_test::count(name, true) == (configured ? 3 : 0));
    }
    assert(scorep_test::count("flowex_reconstruct_output") == (configured ? 3 : 5));
    assert(scorep_test::count("flowex_reconstruct_output", true) == (configured ? 3 : 0));
    assert(scorep_test::count("app_prepare_input") == (configured ? 3 : (manual_gate ? 25 : 0)));
    assert(scorep_test::count("flowex_extract_cubes") == (configured ? 3 : 25));
    assert(scorep_test::count("app_prepare_input", true) == (configured ? 3 : 0));
    assert(scorep_test::count("flowex_extract_cubes", true) == (configured ? 3 : 0));
#endif
    return provider.inputs;
}

struct AlwaysInfer : MLCouplingBehavior {
    bool should_send_data() override { return false; }
    bool should_perform_inference() override { return true; }
    int time_step_delta() override { return 0; }
};

struct HistoryThenInfer : MLCouplingBehavior {
    int call = 0;
    bool should_send_data() override { return call == 0; }
    bool should_perform_inference() override { return ++call > 1; }
    int time_step_delta() override { return 1; }
};

int main()
{
    // Profiling cannot change assembled history, with either old manual gate.
    assert(run_flow(false, true) == run_flow(true, true));
    assert(run_flow(false, false) == run_flow(true, false));

    ml_coupling_scorep::set_detailed_regions_enabled(false);
    MLCouplingApplication<float, float> app(fields(), fields());
    AlwaysInfer behavior;
    Provider provider;
    app.set_scorep_warmup_inference_count(2);
#ifdef USE_SCOREP
    scorep_test::reset();
#endif
    for (int call = 0; call < 4; ++call) {
        assert(app.ml_step(provider, behavior) == 0); // Count inference, not returned delta.
        assert(!ml_coupling_scorep::detailed_regions_are_enabled());
    }
    assert(provider.inputs.size() == 4);
#ifdef USE_SCOREP
    assert(scorep_test::count("cmi_inference_steady") == 2);
    assert(scorep_test::count("app_prepare_input", true) == 2);
#endif
    app.set_scorep_warmup_inference_count(0);
    app.ml_step(provider, behavior);
#ifdef USE_SCOREP
    assert(scorep_test::count("cmi_inference_steady") == 3);
    assert(scorep_test::stack.empty());
#endif
    Flow flow(fields(), fields(), nullptr, 2);
    flow.set_scorep_warmup_inference_count(2);
    HistoryThenInfer no_send;
    Provider reused_input;
    MLCouplingApplication<float, float>& flow_base = flow;
#ifdef USE_SCOREP
    scorep_test::reset();
#endif
    for (int call = 0; call < 4; ++call) {
        assert(flow_base.ml_step(reused_input, no_send) == (call > 0 ? 1 : 0));
    }
    assert(reused_input.inputs.size() == 3);
#ifdef USE_SCOREP
    assert(scorep_test::count("cmi_inference_steady") == 1);
    assert(scorep_test::count("app_provider_inference", true) == 1);
    assert(scorep_test::count("app_prepare_input") == 0); // Selected inference reused history.
    assert(scorep_test::count("flowex_extract_cubes") == 0);
    assert(scorep_test::stack.empty());
#endif
    std::cout << "Warmup, scheduling, output/history, scope ancestry and manual-gate tests passed\n";
}
