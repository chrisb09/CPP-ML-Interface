#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

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

    std::cout << "\nAll C API tests PASSED.\n";
    return 0;
}
