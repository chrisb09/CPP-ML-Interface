/*
 * Pure C99 test for the CPP-ML-Interface C API.
 *
 * This file is compiled with the C compiler (not C++) to verify that c_api.h
 * is valid C99 and that the exported symbols have C linkage.
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "c_api.h"

static void c99_inference(cmi_data_t in, cmi_data_t out, void *user_data)
{
    int *scale = (int *)user_data;
    cmi_tensor_t in_t = NULL;
    cmi_tensor_t out_t = NULL;
    void *in_ptr = NULL;
    void *out_ptr = NULL;
    size_t numel = 0;
    const float *src;
    float *dst;
    size_t i;

    assert(cmi_data_get_tensor(in, 0, &in_t) == CMI_SUCCESS);
    assert(cmi_data_get_tensor(out, 0, &out_t) == CMI_SUCCESS);
    assert(cmi_tensor_get_data(in_t, &in_ptr) == CMI_SUCCESS);
    assert(cmi_tensor_get_data(out_t, &out_ptr) == CMI_SUCCESS);
    assert(cmi_tensor_get_numel(in_t, &numel) == CMI_SUCCESS);

    src = (const float *)in_ptr;
    dst = (float *)out_ptr;
    for (i = 0; i < numel; ++i) {
        dst[i] = src[i] * (float)(*scale);
    }

    assert(cmi_tensor_destroy(in_t) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(out_t) == CMI_SUCCESS);
}

int main(void)
{
    float in_val[3] = {1.0f, 2.0f, 3.0f};
    float out_val[3] = {0.0f, 0.0f, 0.0f};
    int dims[1] = {3};
    cmi_tensor_t tin = NULL;
    cmi_tensor_t tout = NULL;
    cmi_data_t din = NULL;
    cmi_data_t dout = NULL;
    cmi_library_t lib = NULL;
    cmi_config_overrides_t overrides = NULL;
    int scale = 4;
    int64_t int_value = 1;
    double double_value = 2.0;
    bool bool_value = true;

    /* Error handling */
    cmi_clear_last_error();
    assert(strlen(cmi_get_last_error()) == 0);
    assert(cmi_tensor_get_data(NULL, NULL) == CMI_ERROR_NULL_POINTER);
    assert(strlen(cmi_get_last_error()) > 0);
    assert(strcmp(cmi_status_string(CMI_ERROR_NULL_POINTER), "Null pointer") == 0);

    /* Tensor and data containers */
    assert(cmi_tensor_create_flat(&tin, in_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS,
                                  CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_tensor_create_flat(&tout, out_val, dims, 1, CMI_DTYPE_FLOAT,
                                  CMI_LAYOUT_CONTIGUOUS,
                                  CMI_OWNERSHIP_EXTERNAL) == CMI_SUCCESS);
    assert(cmi_data_create(&din, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_create(&dout, CMI_DTYPE_FLOAT) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(din, tin) == CMI_SUCCESS);
    assert(cmi_data_add_tensor(dout, tout) == CMI_SUCCESS);

    /* Generic library and flexible ordered inference */
    assert(cmi_library_create_generic(&lib, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
                                      c99_inference, &scale) == CMI_SUCCESS);
    assert(cmi_library_set_merge_strategy(lib, CMI_MERGE_LIST) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_set(lib, din) == CMI_SUCCESS);
    assert(cmi_library_flex_ordered_inference(lib, dout) == CMI_SUCCESS);
    assert(out_val[0] == 4.0f);
    assert(out_val[1] == 8.0f);
    assert(out_val[2] == 12.0f);
    assert(cmi_library_flex_ordered_get(lib, dout) == CMI_SUCCESS);

    /* Config override handle */
    assert(cmi_config_overrides_create(&overrides) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "batch_size",
                                    CMI_PARAM_INT64,
                                    &int_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "scale",
                                    CMI_PARAM_DOUBLE,
                                    &double_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "flag",
                                    CMI_PARAM_BOOL,
                                    &bool_value) == CMI_SUCCESS);
    assert(cmi_config_overrides_set(overrides, "library", "class",
                                    CMI_PARAM_STRING,
                                    "Dummy") == CMI_SUCCESS);
    assert(cmi_config_overrides_destroy(overrides) == CMI_SUCCESS);

    /* Cleanup */
    assert(cmi_library_destroy(lib) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(tin) == CMI_SUCCESS);
    assert(cmi_tensor_destroy(tout) == CMI_SUCCESS);
    assert(cmi_data_destroy(din) == CMI_SUCCESS);
    assert(cmi_data_destroy(dout) == CMI_SUCCESS);

    printf("All C99 API tests PASSED.\n");
    return 0;
}
