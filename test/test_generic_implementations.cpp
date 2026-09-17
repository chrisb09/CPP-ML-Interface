#include <cassert>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "behavior/ml_coupling_behavior_generic.hpp"
#include "normalization/ml_coupling_normalization_generic.hpp"
#include "library/ml_coupling_library_generic.hpp"
#include "application/ml_coupling_application_generic.hpp"
#include "ml_coupling.hpp"

void test_behavior_generic()
{
    int call_count = 0;
    int step = 0;

    MLCouplingBehaviorGeneric behavior(
        [&]() {
            call_count++;
            return step % 2 == 0;
        },
        [&]() {
            return 5;
        },
        [&]() {
            return true;
        }
    );

    step = 0;
    assert(behavior.should_perform_inference() == true);
    assert(behavior.time_step_delta() == 5);
    assert(behavior.should_send_data() == true);
    assert(call_count == 1);

    step = 1;
    assert(behavior.should_perform_inference() == false);
    assert(call_count == 2);

    // Test null callbacks throw
    bool threw = false;
    try {
        MLCouplingBehaviorGeneric invalid(nullptr, []() { return 0; }, []() { return false; });
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "[PASS] test_behavior_generic\n";
}

void test_normalization_generic()
{
    MLCouplingNormalizationGeneric<float, float> norm(
        [](MLCouplingData<float> data) {
            for (size_t t = 0; t < data.size(); ++t) {
                float* ptr = static_cast<float*>(data[t].root());
                for (size_t i = 0; i < data[t].numel(); ++i) {
                    ptr[i] *= 2.0f;
                }
            }
        },
        [](MLCouplingData<float> data) {
            for (size_t t = 0; t < data.size(); ++t) {
                float* ptr = static_cast<float*>(data[t].root());
                for (size_t i = 0; i < data[t].numel(); ++i) {
                    ptr[i] *= 0.5f;
                }
            }
        },
        [](std::ostream& os) {
            os << "CustomNormalization";
        }
    );

    float raw_in[] = {1.0f, 2.0f, 3.0f};
    MLCouplingData<float> in_data;
    in_data.add_tensor(MLCouplingTensor<float>::wrap_flat(raw_in, {3}));

    norm.normalize_input(in_data);
    assert(raw_in[0] == 2.0f);
    assert(raw_in[1] == 4.0f);
    assert(raw_in[2] == 6.0f);

    norm.denormalize_output(in_data);
    assert(raw_in[0] == 1.0f);
    assert(raw_in[1] == 2.0f);
    assert(raw_in[2] == 3.0f);

    std::ostringstream ss;
    ss << norm;
    assert(ss.str() == "CustomNormalization");

    std::cout << "[PASS] test_normalization_generic\n";
}

void test_library_generic_static()
{
    MLCouplingLibraryGeneric<float, float> lib(
        [](MLCouplingData<float>* in, MLCouplingData<float>* out) {
            assert(in->size() >= 1);
            assert(out->size() >= 1);
            const float* in_ptr = static_cast<const float*>((*in)[0].root());
            float* out_ptr = static_cast<float*>((*out)[0].root());
            for (size_t i = 0; i < (*in)[0].numel(); ++i) {
                out_ptr[i] = in_ptr[i] * 10.0f;
            }
        },
        [](MLCouplingData<float>*, MLCouplingData<float>*) -> std::map<std::string, double> {
            return {{"loss", 0.042}};
        },
        [](std::size_t local_iter) -> std::size_t {
            return local_iter * 2;
        }
    );

    float in_buf[] = {1.0f, 2.0f, 3.0f};
    float out_buf[] = {0.0f, 0.0f, 0.0f};

    MLCouplingData<float> in_data;
    in_data.add_tensor(MLCouplingTensor<float>::wrap_flat(in_buf, {3}));
    MLCouplingData<float> out_data;
    out_data.add_tensor(MLCouplingTensor<float>::wrap_flat(out_buf, {3}));

    lib.static_inference(&in_data, &out_data);
    assert(out_buf[0] == 10.0f);
    assert(out_buf[1] == 20.0f);
    assert(out_buf[2] == 30.0f);

    auto train_res = lib.static_train(&in_data, &out_data);
    assert(train_res.at("loss") == 0.042);

    assert(lib.get_synchronized_iterations(5) == 10);

    std::cout << "[PASS] test_library_generic_static\n";
}

void test_library_generic_flexible()
{
    MLCouplingLibraryGeneric<float, float> lib(
        [](MLCouplingData<float>* in, MLCouplingData<float>* out) {
            const float* in_ptr = static_cast<const float*>((*in)[0].root());
            float* out_ptr = static_cast<float*>((*out)[0].root());
            for (size_t i = 0; i < (*out)[0].numel(); ++i) {
                out_ptr[i] = in_ptr[i] + 1.0f;
            }
        }
    );

    float in_buf1[] = {1.0f, 2.0f};
    float in_buf2[] = {3.0f, 4.0f};
    float out_buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    MLCouplingData<float> d1, d2, d_out;
    d1.add_tensor(MLCouplingTensor<float>::wrap_flat(in_buf1, {1, 2}));
    d2.add_tensor(MLCouplingTensor<float>::wrap_flat(in_buf2, {1, 2}));
    d_out.add_tensor(MLCouplingTensor<float>::wrap_flat(out_buf, {1, 4}));

    lib.flex_ordered_set(d1);
    lib.flex_ordered_set(d2);
    lib.flex_ordered_inference(&d_out);

    assert(out_buf[0] == 2.0f);
    assert(out_buf[1] == 3.0f);
    assert(out_buf[2] == 4.0f);
    assert(out_buf[3] == 5.0f);

    std::cout << "[PASS] test_library_generic_flexible\n";
}

void test_application_generic()
{
    float in_buf[] = {1.0f, 2.0f, 3.0f};
    float out_buf[] = {0.0f, 0.0f, 0.0f};

    MLCouplingData<float> cin, cout;
    cin.add_tensor(MLCouplingTensor<float>::wrap_flat(in_buf, {3}));
    cout.add_tensor(MLCouplingTensor<float>::wrap_flat(out_buf, {3}));

    bool pre_called = false;
    bool post_called = false;

    MLCouplingApplicationGeneric<float, float> app(
        cin, cout,
        [&](MLCouplingData<float> in) {
            pre_called = true;
            return in;
        },
        [&](MLCouplingData<float> out) {
            post_called = true;
            return out;
        }
    );

    MLCouplingLibraryGeneric<float, float> lib(
        [](MLCouplingData<float>* in, MLCouplingData<float>* out) {
            const float* ip = static_cast<const float*>((*in)[0].root());
            float* op = static_cast<float*>((*out)[0].root());
            for (size_t i = 0; i < (*in)[0].numel(); ++i) op[i] = ip[i] * 3.0f;
        }
    );

    MLCouplingBehaviorGeneric behavior(
        []() { return true; },
        []() { return 1; },
        []() { return true; }
    );

    int delta = app.ml_step(lib, behavior);
    assert(delta == 1);
    assert(pre_called);
    assert(post_called);
    assert(out_buf[0] == 3.0f);
    assert(out_buf[1] == 6.0f);
    assert(out_buf[2] == 9.0f);

    std::cout << "[PASS] test_application_generic\n";
}

void test_full_pipeline_with_mlcoupling()
{
    float in_buf[] = {2.0f, 4.0f};
    float out_buf[] = {0.0f, 0.0f};

    MLCouplingData<float> cin, cout;
    cin.add_tensor(MLCouplingTensor<float>::wrap_flat(in_buf, {2}));
    cout.add_tensor(MLCouplingTensor<float>::wrap_flat(out_buf, {2}));

    auto lib = new MLCouplingLibraryGeneric<float, float>(
        [](MLCouplingData<float>* in, MLCouplingData<float>* out) {
            const float* ip = static_cast<const float*>((*in)[0].root());
            float* op = static_cast<float*>((*out)[0].root());
            op[0] = ip[0] + 100.0f;
            op[1] = ip[1] + 200.0f;
        }
    );

    auto app = new MLCouplingApplicationGeneric<float, float>(cin, cout);

    auto behavior = new MLCouplingBehaviorGeneric(
        []() { return true; },
        []() { return 7; },
        []() { return true; }
    );

    MLCoupling<float, float> coupling(lib, app, behavior);

    int delta = coupling.step();
    assert(delta == 7);
    assert(out_buf[0] == 102.0f);
    assert(out_buf[1] == 204.0f);

    std::cout << "[PASS] test_full_pipeline_with_mlcoupling\n";
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    test_behavior_generic();
    test_normalization_generic();
    test_library_generic_static();
    test_library_generic_flexible();
    test_application_generic();
    test_full_pipeline_with_mlcoupling();

    std::cout << "\nAll generic implementation tests PASSED.\n";
    return 0;
}
