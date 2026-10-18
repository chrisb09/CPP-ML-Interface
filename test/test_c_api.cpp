#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "c_api.h"

static void test_error_handling() {
    cmi_clear_last_error();
    assert(strlen(cmi_get_last_error()) == 0);

    // Trigger null pointer error
    int status = cmi_tensor_get_data(nullptr, nullptr);
    assert(status == CMI_ERROR_NULL_POINTER);
    assert(strlen(cmi_get_last_error()) > 0);
    assert(strcmp(cmi_status_string(CMI_ERROR_NULL_POINTER), "Null pointer") == 0);

    cmi_clear_last_error();
    assert(strlen(cmi_get_last_error()) == 0);

    std::cout << "[PASS] test_error_handling\n";
}

static void test_tensor_lifecycle() {
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int dims[2] = {2, 3};

    // 1. Create flat wrapped tensor
    cmi_tensor_t t = nullptr;
    int st = cmi_tensor_create_flat(&t, data, dims, 2, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);
    assert(st == CMI_SUCCESS);
    assert(t != nullptr);

    // 2. Query attributes
    void* raw_data = nullptr;
    assert(cmi_tensor_get_data(t, &raw_data) == CMI_SUCCESS);
    assert(raw_data == data);

    const int* out_dims = nullptr;
    int ndims = 0;
    assert(cmi_tensor_get_dims(t, &out_dims, &ndims) == CMI_SUCCESS);
    assert(ndims == 2);
    assert(out_dims[0] == 2 && out_dims[1] == 3);

    int layout = 0;
    assert(cmi_tensor_get_layout(t, &layout) == CMI_SUCCESS);
    assert(layout == CMI_LAYOUT_CONTIGUOUS);

    int dtype = 0;
    assert(cmi_tensor_get_data_type(t, &dtype) == CMI_SUCCESS);
    assert(dtype == CMI_DTYPE_FLOAT);

    size_t numel = 0;
    assert(cmi_tensor_get_numel(t, &numel) == CMI_SUCCESS);
    assert(numel == 6);

    int ownership = -1;
    assert(cmi_tensor_get_ownership(t, &ownership) == CMI_SUCCESS);
    assert(ownership == CMI_OWNERSHIP_EXTERNAL);

    // 3. Deep copy
    cmi_tensor_t t_copy = nullptr;
    assert(cmi_tensor_deep_copy(t, &t_copy) == CMI_SUCCESS);
    void* copy_data = nullptr;
    assert(cmi_tensor_get_data(t_copy, &copy_data) == CMI_SUCCESS);
    assert(copy_data != data);
    assert(static_cast<float*>(copy_data)[0] == 1.0f);

    // 4. Flatten to FortranContiguous
    cmi_tensor_t t_fortran = nullptr;
    assert(cmi_tensor_flatten(t, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &t_fortran) == CMI_SUCCESS);
    int f_layout = 0;
    assert(cmi_tensor_get_layout(t_fortran, &f_layout) == CMI_SUCCESS);
    assert(f_layout == CMI_LAYOUT_FORTRAN_CONTIGUOUS);

    // 5. Create from copy
    cmi_tensor_t t_from_copy = nullptr;
    assert(cmi_tensor_create_from_copy(&t_from_copy, data, dims, 2, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS) == CMI_SUCCESS);
    void* from_copy_ptr = nullptr;
    assert(cmi_tensor_get_data(t_from_copy, &from_copy_ptr) == CMI_SUCCESS);
    assert(from_copy_ptr != data);
    assert(static_cast<float*>(from_copy_ptr)[5] == 6.0f);

    // 6. Destroy all
    assert(cmi_tensor_destroy(t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_copy) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_fortran) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_from_copy) == CMI_SUCCESS);

    std::cout << "[PASS] test_tensor_lifecycle\n";
}

static void test_data_lifecycle() {
    cmi_data_t d = nullptr;
    assert(cmi_data_create(&d, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(d != nullptr);

    int size = -1;
    assert(cmi_data_size(d, &size) == CMI_SUCCESS);
    assert(size == 0);

    float val1[2] = {10.0f, 20.0f};
    int dims1[1] = {2};
    cmi_tensor_t t1 = nullptr;
    assert(cmi_tensor_create_flat(&t1, val1, dims1, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    float val2[3] = {30.0f, 40.0f, 50.0f};
    int dims2[1] = {3};
    cmi_tensor_t t2 = nullptr;
    assert(cmi_tensor_create_flat(&t2, val2, dims2, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    assert(cmi_data_add_tensor(d, t1) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d, t2) == CMI_SUCCESS);
    assert(cmi_data_size(d, &size) == CMI_SUCCESS);
    assert(size == 2);

    cmi_tensor_t retrieved = nullptr;
    assert(cmi_data_get_tensor(d, 1, &retrieved) == CMI_SUCCESS);
    void* r_ptr = nullptr;
    assert(cmi_tensor_get_data(retrieved, &r_ptr) == CMI_SUCCESS);
    assert(static_cast<float*>(r_ptr)[0] == 30.0f);
    assert(static_cast<float*>(r_ptr)[2] == 50.0f);

    assert(cmi_tensor_destroy(retrieved) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t1) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t2) == CMI_SUCCESS);
    assert(cmi_data_destroy(d) == CMI_SUCCESS);

    std::cout << "[PASS] test_data_lifecycle\n";
}

static void test_introspection() {
    int beh_count = cmi_get_class_count("behavior");
    assert(beh_count > 0);

    bool found_default = false;
    for (int i = 0; i < beh_count; ++i) {
        const char* name = cmi_get_class_name("behavior", i);
        if (strcmp(name, "MLCouplingBehaviorDefault") == 0) {
            found_default = true;
        }
    }
    assert(found_default);

    assert(cmi_is_class_registered("behavior", "Default") == true);
    assert(cmi_is_class_registered("behavior", "MLCouplingBehaviorDefault") == true);
    assert(cmi_is_class_registered("behavior", "NonExistentBehavior") == false);

    assert(cmi_is_class_registered("library", "Dummy") == true);
    assert(cmi_is_class_registered("normalization", "MinMax") == true);

    std::cout << "[PASS] test_introspection\n";
}

static bool my_should_infer(void* user_data) {
    int* counter = static_cast<int*>(user_data);
    (*counter)++;
    return true;
}

static int my_time_delta(void* user_data) {
    (void)user_data;
    return 3;
}

static bool my_should_send(void* user_data) {
    (void)user_data;
    return true;
}

static void test_behavior_c_api() {
    // 1. Default
    cmi_behavior_t def_b = nullptr;
    assert(cmi_behavior_create_default(&def_b) == CMI_SUCCESS);
    bool infer = false;
    assert(cmi_behavior_should_perform_inference(def_b, &infer) == CMI_SUCCESS);
    assert(infer == true);
    int delta = -1;
    assert(cmi_behavior_time_step_delta(def_b, &delta) == CMI_SUCCESS);
    assert(delta == 0);
    assert(cmi_behavior_destroy(def_b) == CMI_SUCCESS);

    // 2. Periodic
    cmi_behavior_t per_b = nullptr;
    assert(cmi_behavior_create_periodic(&per_b, 10, 0, 1, 5) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(per_b) == CMI_SUCCESS);

    // 3. Generic with C function pointers + user_data
    int counter = 0;
    cmi_behavior_t gen_b = nullptr;
    assert(cmi_behavior_create_generic(&gen_b, my_should_infer, my_time_delta, my_should_send, &counter) == CMI_SUCCESS);
    assert(cmi_behavior_should_perform_inference(gen_b, &infer) == CMI_SUCCESS);
    assert(infer == true);
    assert(counter == 1);
    assert(cmi_behavior_time_step_delta(gen_b, &delta) == CMI_SUCCESS);
    assert(delta == 3);
    assert(cmi_behavior_destroy(gen_b) == CMI_SUCCESS);

    // 4. By name
    cmi_behavior_t name_b = nullptr;
    assert(cmi_behavior_create_by_name(&name_b, "Default", nullptr, nullptr, nullptr, 0) == CMI_SUCCESS);
    assert(cmi_behavior_should_perform_inference(name_b, &infer) == CMI_SUCCESS);
    assert(infer == true);
    assert(cmi_behavior_destroy(name_b) == CMI_SUCCESS);

    std::cout << "[PASS] test_behavior_c_api\n";
}

static void my_c_inference(cmi_data_t in, cmi_data_t out, void* user_data) {
    int* scale = static_cast<int*>(user_data);
    cmi_tensor_t in_t = nullptr;
    cmi_tensor_t out_t = nullptr;
    cmi_data_get_tensor(in, 0, &in_t);
    cmi_data_get_tensor(out, 0, &out_t);

    void* in_ptr = nullptr;
    void* out_ptr = nullptr;
    cmi_tensor_get_data(in_t, &in_ptr);
    cmi_tensor_get_data(out_t, &out_ptr);

    size_t numel = 0;
    cmi_tensor_get_numel(in_t, &numel);

    const float* ip = static_cast<const float*>(in_ptr);
    float* op = static_cast<float*>(out_ptr);
    for (size_t i = 0; i < numel; ++i) {
        op[i] = ip[i] * (*scale);
    }

    cmi_tensor_destroy(in_t);
    cmi_tensor_destroy(out_t);
}

static void test_library_c_api() {
    int scale_factor = 7;
    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, my_c_inference, &scale_factor) == CMI_SUCCESS);

    float in_val[3] = {1.0f, 2.0f, 3.0f};
    float out_val[3] = {0.0f, 0.0f, 0.0f};
    int dims[1] = {3};

    cmi_tensor_t tin = nullptr, tout = nullptr;
    cmi_tensor_create_flat(&tin, in_val, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);
    cmi_tensor_create_flat(&tout, out_val, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);

    cmi_data_t din = nullptr, dout = nullptr;
    cmi_data_create(&din, CMI_DTYPE_FLOAT);
    cmi_data_create(&dout, CMI_DTYPE_FLOAT);
    cmi_data_add_tensor(din, tin);
    cmi_data_add_tensor(dout, tout);

    // Static inference
    assert(cmi_library_inference(lib, din, dout) == CMI_SUCCESS);
    assert(out_val[0] == 7.0f);
    assert(out_val[1] == 14.0f);
    assert(out_val[2] == 21.0f);

    // Flexible ordered inference fallback
    float flex_out[3] = {0.0f, 0.0f, 0.0f};
    cmi_tensor_t t_flex = nullptr;
    cmi_tensor_create_flat(&t_flex, flex_out, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);
    cmi_data_t d_flex = nullptr;
    cmi_data_create(&d_flex, CMI_DTYPE_FLOAT);
    cmi_data_add_tensor(d_flex, t_flex);

    assert(cmi_library_flex_ordered_set(lib, din) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_inference(lib, d_flex) == CMI_SUCCESS);
    assert(flex_out[0] == 7.0f);
    assert(flex_out[1] == 14.0f);
    assert(flex_out[2] == 21.0f);

    cmi_tensor_destroy(tin);
    cmi_tensor_destroy(tout);
    cmi_tensor_destroy(t_flex);
    cmi_data_destroy(din);
    cmi_data_destroy(dout);
    cmi_data_destroy(d_flex);
    cmi_library_destroy(lib);

    std::cout << "[PASS] test_library_c_api\n";
}

static void test_normalization_c_api() {
    cmi_normalization_t norm = nullptr;
    assert(cmi_normalization_create_minmax(&norm, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                           0.0, 10.0, 0.0, 10.0) == CMI_SUCCESS);

    float val[2] = {5.0f, 10.0f};
    int dims[1] = {2};
    cmi_tensor_t t = nullptr;
    cmi_tensor_create_flat(&t, val, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);
    cmi_data_t d = nullptr;
    cmi_data_create(&d, CMI_DTYPE_FLOAT);
    cmi_data_add_tensor(d, t);

    assert(cmi_normalization_normalize_input(norm, d) == CMI_SUCCESS);
    assert(std::fabs(val[0] - 0.5f) < 1e-5f);
    assert(std::fabs(val[1] - 1.0f) < 1e-5f);

    assert(cmi_normalization_denormalize_output(norm, d) == CMI_SUCCESS);
    assert(std::fabs(val[0] - 5.0f) < 1e-5f);
    assert(std::fabs(val[1] - 10.0f) < 1e-5f);

    cmi_tensor_destroy(t);
    cmi_data_destroy(d);
    cmi_normalization_destroy(norm);

    std::cout << "[PASS] test_normalization_c_api\n";
}

static void test_full_coupling_c_api() {
    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    int dims[1] = {2};

    cmi_tensor_t tin = nullptr, tout = nullptr;
    cmi_tensor_create_flat(&tin, in_val, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);
    cmi_tensor_create_flat(&tout, out_val, dims, 1, CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL);

    cmi_data_t din = nullptr, dout = nullptr;
    cmi_data_create(&din, CMI_DTYPE_FLOAT);
    cmi_data_create(&dout, CMI_DTYPE_FLOAT);
    cmi_data_add_tensor(din, tin);
    cmi_data_add_tensor(dout, tout);

    int factor = 50;
    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, my_c_inference, &factor) == CMI_SUCCESS);

    cmi_application_t app = nullptr;
    assert(cmi_application_create_generic(&app, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                         CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                         din, dout, nullptr) == CMI_SUCCESS);

    int count = 0;
    cmi_behavior_t beh = nullptr;
    assert(cmi_behavior_create_generic(&beh, my_should_infer, my_time_delta, my_should_send, &count) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create(&coupling,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               lib, app, beh) == CMI_SUCCESS);

    int delta = 0;
    assert(cmi_coupling_step(coupling, &delta) == CMI_SUCCESS);
    assert(delta == 3);
    assert(out_val[0] == 50.0f);
    assert(out_val[1] == 100.0f);

    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_application_destroy(app) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(beh) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(tin) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(tout) == CMI_SUCCESS);
    assert(cmi_data_destroy(din) == CMI_SUCCESS);
    assert(cmi_data_destroy(dout) == CMI_SUCCESS);

    std::cout << "[PASS] test_full_coupling_c_api\n";
}

static void test_legacy_c_api() {
    // Test that create_library and destroy_library work for Dummy
    void* handle = create_library("Dummy", 2, 2, nullptr, nullptr, 0); // 2 = float
    assert(handle != nullptr);
    destroy_library(handle);

    std::cout << "[PASS] test_legacy_c_api\n";
}

struct FlexCtx {
    int scale = 0;
    int train_calls = 0;
};

static void my_flex_inference(cmi_data_t in, cmi_data_t out, void* user_data) {
    FlexCtx* ctx = static_cast<FlexCtx*>(user_data);
    cmi_tensor_t in_t = nullptr;
    cmi_tensor_t out_t = nullptr;
    assert(cmi_data_get_tensor(in, 0, &in_t) == CMI_SUCCESS);
    assert(cmi_data_get_tensor(out, 0, &out_t) == CMI_SUCCESS);

    void* in_ptr = nullptr;
    void* out_ptr = nullptr;
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    assert(cmi_tensor_get_data(out_t, &out_ptr) == CMI_SUCCESS);

    size_t numel = 0;
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    const float* ip = static_cast<const float*>(in_ptr);
    float* op = static_cast<float*>(out_ptr);
    for (size_t i = 0; i < numel; ++i) {
        op[i] = ip[i] * static_cast<float>(ctx->scale);
    }

    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
}

static void my_flex_train(cmi_data_t in, cmi_data_t target, void* user_data) {
    (void)in;
    (void)target;
    FlexCtx* ctx = static_cast<FlexCtx*>(user_data);
    ctx->train_calls++;
}

static void test_flex_get_and_train_c_api() {
    FlexCtx ctx;
    ctx.scale = 5;
    ctx.train_calls = 0;

    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic_with_training(
        &lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        my_flex_inference, my_flex_train, &ctx) == CMI_SUCCESS);

    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    float target_val[2] = {0.0f, 0.0f};
    int dims[1] = {2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr, t_target = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_target, target_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr, d_target = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_target, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_target, t_target) == CMI_SUCCESS);

    // Ordered training invokes the user training callback.
    assert(cmi_library_flex_ordered_set(lib, d_in) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_set_target(lib, d_target) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_train(lib, 1) == CMI_SUCCESS);
    assert(ctx.train_calls == 1);

    // Ordered inference followed by result retrieval.
    assert(cmi_library_flex_ordered_set(lib, d_in) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_inference(lib, d_out) == CMI_SUCCESS);
    assert(out_val[0] == 5.0f);
    assert(out_val[1] == 10.0f);
    assert(cmi_library_flex_ordered_get(lib, d_out) == CMI_SUCCESS);
    assert(out_val[0] == 5.0f);
    assert(out_val[1] == 10.0f);

    // Keyed training invokes the user training callback.
    const char* in_keys[1] = {"input"};
    const char* target_keys[1] = {"target"};
    assert(cmi_library_flex_keyed_set(lib, "input", d_in) == CMI_SUCCESS);
    assert(cmi_library_flex_keyed_set_target(lib, "target", d_target) == CMI_SUCCESS);
    assert(cmi_library_flex_keyed_train(lib, 2, in_keys, 1, target_keys, 1) == CMI_SUCCESS);
    assert(ctx.train_calls == 2);

    // Keyed inference followed by result retrieval.
    out_val[0] = 0.0f;
    out_val[1] = 0.0f;
    assert(cmi_library_flex_keyed_set(lib, "input", d_in) == CMI_SUCCESS);
    const char* out_keys[1] = {"output"};
    assert(cmi_library_flex_keyed_inference(lib, in_keys, 1, out_keys, 1, d_out) == CMI_SUCCESS);
    assert(out_val[0] == 5.0f);
    assert(out_val[1] == 10.0f);
    assert(cmi_library_flex_keyed_get(lib, "output", d_out) == CMI_SUCCESS);
    assert(out_val[0] == 5.0f);
    assert(out_val[1] == 10.0f);

    // Mismatched output types are rejected.
    cmi_data_t d_wrong = nullptr;
    assert(cmi_data_create(&d_wrong, CMI_DTYPE_DOUBLE) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_get(lib, d_wrong) == CMI_ERROR_TYPE_MISMATCH);
    assert(cmi_library_flex_keyed_get(lib, "output", d_wrong) == CMI_ERROR_TYPE_MISMATCH);

    assert(cmi_data_destroy(d_wrong) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_target) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_target) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);

    std::cout << "[PASS] test_flex_get_and_train_c_api\n";
}

struct AppCtx {
    float* final_output = nullptr;
    int pre_calls = 0;
    int post_calls = 0;
};

static void my_preprocess(cmi_data_t input, cmi_data_t out_library_input, void* user_data) {
    AppCtx* ctx = static_cast<AppCtx*>(user_data);
    ctx->pre_calls++;

    cmi_tensor_t in_t = nullptr;
    assert(cmi_data_get_tensor(input, 0, &in_t) == CMI_SUCCESS);
    void* in_ptr = nullptr;
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    const int* dims = nullptr;
    int ndims = 0;
    assert(cmi_tensor_get_dims(in_t, &dims, &ndims) == CMI_SUCCESS);
    size_t numel = 0;
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    std::vector<float> transformed(numel);
    const float* src = static_cast<const float*>(in_ptr);
    for (size_t i = 0; i < numel; ++i) {
        transformed[i] = 2.0f * src[i];
    }

    cmi_tensor_t out_t = nullptr;
    assert(cmi_tensor_create_from_copy(&out_t, transformed.data(), dims, ndims,
                                       CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(out_library_input, out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
}

static void my_postprocess(cmi_data_t library_output, cmi_data_t out_coupling_output, void* user_data) {
    AppCtx* ctx = static_cast<AppCtx*>(user_data);
    ctx->post_calls++;

    cmi_tensor_t in_t = nullptr;
    assert(cmi_data_get_tensor(library_output, 0, &in_t) == CMI_SUCCESS);
    void* in_ptr = nullptr;
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    const int* dims = nullptr;
    int ndims = 0;
    assert(cmi_tensor_get_dims(in_t, &dims, &ndims) == CMI_SUCCESS);
    size_t numel = 0;
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    const float* src = static_cast<const float*>(in_ptr);
    for (size_t i = 0; i < numel; ++i) {
        ctx->final_output[i] = 3.0f * src[i];
    }

    cmi_tensor_t out_t = nullptr;
    assert(cmi_tensor_create_flat(&out_t, ctx->final_output, dims, ndims,
                                  CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS,
                                  CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(out_coupling_output, out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
}

static void test_application_callbacks_c_api() {
    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    int dims[1] = {2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    int scale = 10;
    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                      my_c_inference, &scale) == CMI_SUCCESS);

    AppCtx ctx;
    ctx.final_output = out_val;

    cmi_application_t app = nullptr;
    assert(cmi_application_create_generic_with_callbacks(
        &app, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        d_in, d_out, my_preprocess, my_postprocess, nullptr, &ctx) == CMI_SUCCESS);

    cmi_behavior_t beh = nullptr;
    assert(cmi_behavior_create_default(&beh) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create(&coupling,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               lib, app, beh) == CMI_SUCCESS);

    int delta = -1;
    assert(cmi_coupling_step(coupling, &delta) == CMI_SUCCESS);
    assert(ctx.pre_calls == 1);
    assert(ctx.post_calls == 1);
    // [1, 2] -> preprocess *2 -> [2, 4] -> inference *10 -> [20, 40] -> postprocess *3 -> [60, 120]
    assert(out_val[0] == 60.0f);
    assert(out_val[1] == 120.0f);

    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_application_destroy(app) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(beh) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::cout << "[PASS] test_application_callbacks_c_api\n";
}

static void test_config_with_data_c_api() {
    const char* config_path = "cmi_test_config_with_data.toml";
    {
        std::ofstream config(config_path);
        config << "coupling_type = \"STATIC\"\n\n";
        config << "[library]\nclass = \"Dummy\"\n\n";
        config << "[application]\nclass = \"MLCouplingApplicationGeneric\"\n";
    }

    float in_val[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int dims[2] = {2, 2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 2, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 2, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create_from_config_with_data(
        &coupling, config_path,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        d_in, d_out) == CMI_SUCCESS);
    assert(coupling != nullptr);
    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);

    // Null buffers and type mismatches are rejected before construction.
    coupling = nullptr;
    assert(cmi_coupling_create_from_config_with_data(
        &coupling, config_path,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        nullptr, d_out) == CMI_ERROR_NULL_POINTER);
    assert(coupling == nullptr);

    coupling = nullptr;
    assert(cmi_coupling_create_from_config_with_data(
        &coupling, config_path,
        CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE,
        d_in, d_out) == CMI_ERROR_TYPE_MISMATCH);
    assert(coupling == nullptr);

    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::remove(config_path);

    std::cout << "[PASS] test_config_with_data_c_api\n";
}

struct MixedCtx {
    double* lib_out;
    float* final_out;
};

static void mixed_c_inference(cmi_data_t in, cmi_data_t out, void* user_data) {
    MixedCtx* ctx = static_cast<MixedCtx*>(user_data);
    cmi_tensor_t in_t = nullptr;
    assert(cmi_data_get_tensor(in, 0, &in_t) == CMI_SUCCESS);
    void* in_ptr = nullptr;
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    const int* dims = nullptr;
    int ndims = 0;
    assert(cmi_tensor_get_dims(in_t, &dims, &ndims) == CMI_SUCCESS);
    size_t numel = 0;
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    const float* src = static_cast<const float*>(in_ptr);
    for (size_t i = 0; i < numel; ++i) {
        ctx->lib_out[i] = static_cast<double>(src[i]) * 10.0;
    }

    cmi_tensor_t out_t = nullptr;
    assert(cmi_tensor_create_flat(&out_t, ctx->lib_out, dims, ndims, CMI_DTYPE_DOUBLE,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(out, out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
}

static void mixed_c_postprocess(cmi_data_t library_output, cmi_data_t out_coupling_output, void* user_data) {
    MixedCtx* ctx = static_cast<MixedCtx*>(user_data);
    cmi_tensor_t in_t = nullptr;
    assert(cmi_data_get_tensor(library_output, 0, &in_t) == CMI_SUCCESS);
    void* in_ptr = nullptr;
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    const int* dims = nullptr;
    int ndims = 0;
    assert(cmi_tensor_get_dims(in_t, &dims, &ndims) == CMI_SUCCESS);
    size_t numel = 0;
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    const double* src = static_cast<const double*>(in_ptr);
    for (size_t i = 0; i < numel; ++i) {
        ctx->final_out[i] = static_cast<float>(src[i] * 3.0);
    }

    cmi_tensor_t out_t = nullptr;
    assert(cmi_tensor_create_flat(&out_t, ctx->final_out, dims, ndims, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(out_coupling_output, out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
}

static void test_config_string_and_overrides_c_api() {
    const char* valid_config =
        "coupling_type = \"STATIC\"\n"
        "[library]\nclass = \"Dummy\"\n"
        "[application]\nclass = \"MLCouplingApplicationGeneric\"\n";
    const char* invalid_library_config =
        "coupling_type = \"STATIC\"\n"
        "[library]\nclass = \"MissingLibrary\"\n"
        "[application]\nclass = \"MLCouplingApplicationGeneric\"\n";

    float in_val[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int dims[2] = {2, 2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 2, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 2, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create_from_config_string_with_data(
        &coupling, valid_config, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, d_in, d_out) == CMI_SUCCESS);
    assert(coupling != nullptr);
    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);

    cmi_config_overrides_t overrides = nullptr;
    assert(cmi_config_overrides_create(&overrides) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "class",
                                    CMI_PARAM_STRING, "Dummy") == CMI_SUCCESS);

    coupling = nullptr;
    assert(cmi_coupling_create_from_config_string_with_data_and_overrides(
        &coupling, invalid_library_config, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, d_in, d_out, overrides) == CMI_SUCCESS);
    assert(coupling != nullptr);
    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);

    cmi_config_overrides_t dotted = nullptr;
    assert(cmi_config_overrides_create(&dotted) == CMI_SUCCESS);
    assert(cmi_config_overrides_set_dotted(dotted, "library.class",
                                           CMI_PARAM_STRING, "Dummy") == CMI_SUCCESS);

    coupling = nullptr;
    assert(cmi_coupling_create_from_config_string_with_data_and_overrides(
        &coupling, invalid_library_config, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, d_in, d_out, dotted) == CMI_SUCCESS);
    assert(coupling != nullptr);
    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);

    const char* config_path = "cmi_test_config_overrides.toml";
    FILE* config_file = fopen(config_path, "w");
    assert(config_file != nullptr);
    fputs(invalid_library_config, config_file);
    fclose(config_file);

    coupling = nullptr;
    assert(cmi_coupling_create_from_config_with_data_and_overrides(
        &coupling, config_path, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, d_in, d_out, overrides) == CMI_SUCCESS);
    assert(coupling != nullptr);
    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    std::remove(config_path);

    int64_t int_value = 7;
    double double_value = 1.5;
    bool bool_value = true;
    const char* string_values[2] = {"alpha", "beta"};
    assert(cmi_config_overrides_set(overrides, "behavior", "inference_interval",
                                    CMI_PARAM_INT64, &int_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "batch_size",
                                    CMI_PARAM_DOUBLE, &double_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "behavior", "enabled",
                                    CMI_PARAM_BOOL, &bool_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set_string_array(overrides, "library", "labels",
                                                 string_values, 2) == CMI_SUCCESS);
    assert(cmi_config_overrides_set_dotted_string_array(dotted, "library.labels",
                                                        string_values, 2) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "raw",
                                    CMI_PARAM_RAW, out_val) == CMI_SUCCESS);

    assert(cmi_config_overrides_destroy(overrides) == CMI_SUCCESS);
    assert(cmi_config_overrides_destroy(dotted) == CMI_SUCCESS);

    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::cout << "[PASS] test_config_string_and_overrides_c_api\n";
}

static void test_coupling_merge_strategy_c_api() {
    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    int dims[1] = {2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    int scale = 1;
    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                      my_c_inference, &scale) == CMI_SUCCESS);
    cmi_application_t app = nullptr;
    assert(cmi_application_create_generic(&app, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                          CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                          d_in, d_out, nullptr) == CMI_SUCCESS);
    cmi_behavior_t beh = nullptr;
    assert(cmi_behavior_create_default(&beh) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create(&coupling,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               lib, app, beh) == CMI_SUCCESS);
    assert(cmi_coupling_set_merge_strategy(coupling, CMI_MERGE_LIST) == CMI_SUCCESS);
    assert(cmi_coupling_set_merge_strategy(coupling, CMI_MERGE_STACK) == CMI_SUCCESS);
    assert(cmi_coupling_set_merge_strategy(nullptr, CMI_MERGE_LIST) == CMI_ERROR_NULL_POINTER);

    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_application_destroy(app) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(beh) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::cout << "[PASS] test_coupling_merge_strategy_c_api\n";
}

static void test_mixed_type_dispatch_c_api() {
    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    double lib_out[2] = {0.0, 0.0};
    int dims[1] = {2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    MixedCtx ctx;
    ctx.lib_out = lib_out;
    ctx.final_out = out_val;

    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_DOUBLE,
                                      mixed_c_inference, &ctx) == CMI_SUCCESS);

    cmi_application_t app = nullptr;
    assert(cmi_application_create_generic_with_callbacks(
        &app, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_DOUBLE,
        d_in, d_out, nullptr, mixed_c_postprocess, nullptr, &ctx) == CMI_SUCCESS);

    cmi_behavior_t beh = nullptr;
    assert(cmi_behavior_create_default(&beh) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create(&coupling,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_DOUBLE,
                               lib, app, beh) == CMI_SUCCESS);

    int delta = -1;
    assert(cmi_coupling_step(coupling, &delta) == CMI_SUCCESS);
    // [1, 2] -> library *10 -> [10, 20] as double -> postprocess *3 -> [30, 60] as float
    assert(out_val[0] == 30.0f);
    assert(out_val[1] == 60.0f);

    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_application_destroy(app) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(beh) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::cout << "[PASS] test_mixed_type_dispatch_c_api\n";
}

static void test_training_tracker_c_api() {
    float in_val[2] = {1.0f, 2.0f};
    float out_val[2] = {0.0f, 0.0f};
    int dims[1] = {2};

    cmi_tensor_t t_in = nullptr, t_out = nullptr;
    assert(cmi_tensor_create_flat(&t_in, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&t_out, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);

    cmi_data_t d_in = nullptr, d_out = nullptr;
    assert(cmi_data_create(&d_in, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&d_out, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_in, t_in) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(d_out, t_out) == CMI_SUCCESS);

    int scale = 1;
    cmi_library_t lib = nullptr;
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                      my_c_inference, &scale) == CMI_SUCCESS);
    cmi_application_t app = nullptr;
    assert(cmi_application_create_generic(&app, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                          CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                          d_in, d_out, nullptr) == CMI_SUCCESS);
    cmi_behavior_t beh = nullptr;
    assert(cmi_behavior_create_default(&beh) == CMI_SUCCESS);

    cmi_coupling_t coupling = nullptr;
    assert(cmi_coupling_create(&coupling,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                               lib, app, beh) == CMI_SUCCESS);

    const char* fields[1] = {"loss"};
    double values[1] = {0.5};

    assert(cmi_coupling_track(coupling, "loss") == CMI_SUCCESS);
    assert(cmi_coupling_log_training_metrics(coupling, 1, fields, values, 1) == CMI_SUCCESS);

    double current = -1.0;
    assert(cmi_coupling_get_current(coupling, "loss", &current) == CMI_SUCCESS);
    assert(current == 0.5);

    values[0] = 0.25;
    assert(cmi_coupling_log_training_metrics(coupling, 2, fields, values, 1) == CMI_SUCCESS);

    double history[4] = {0.0, 0.0, 0.0, 0.0};
    int count = -1;
    assert(cmi_coupling_get_history(coupling, "loss", history, 4, &count) == CMI_SUCCESS);
    assert(count == 2);
    assert(history[0] == 0.5);
    assert(history[1] == 0.25);

    count = -1;
    assert(cmi_coupling_get_history(coupling, "loss", nullptr, 0, &count) == CMI_SUCCESS);
    assert(count == 2);

    assert(cmi_coupling_get_current(coupling, "accuracy", &current) == CMI_ERROR_RUNTIME);

    assert(cmi_coupling_destroy(coupling) == CMI_SUCCESS);
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_application_destroy(app) == CMI_SUCCESS);
    assert(cmi_behavior_destroy(beh) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_in) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(t_out) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_in) == CMI_SUCCESS);
    assert(cmi_data_destroy(d_out) == CMI_SUCCESS);

    std::cout << "[PASS] test_training_tracker_c_api\n";
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    test_error_handling();
    test_tensor_lifecycle();
    test_data_lifecycle();
    test_introspection();
    test_behavior_c_api();
    test_library_c_api();
    test_normalization_c_api();
    test_full_coupling_c_api();
    test_legacy_c_api();
    test_flex_get_and_train_c_api();
    test_application_callbacks_c_api();
    test_config_with_data_c_api();
    test_config_string_and_overrides_c_api();
    test_coupling_merge_strategy_c_api();
    test_mixed_type_dispatch_c_api();
    test_training_tracker_c_api();

    std::cout << "\nAll C API tests PASSED.\n";
    return 0;
}
