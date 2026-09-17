#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <mutex>

#include "c_api.h"
#include "generated_registry.hpp"
#include "data/ml_coupling_data.hpp"
#include "data/ml_coupling_data_type.hpp"
#include "data/ml_coupling_memory_layout.hpp"
#include "behavior/ml_coupling_behavior.hpp"
#include "behavior/ml_coupling_behavior_default.hpp"
#include "behavior/ml_coupling_behavior_periodic.hpp"
#include "behavior/ml_coupling_behavior_generic.hpp"
#include "normalization/ml_coupling_normalization.hpp"
#include "normalization/ml_coupling_minmax_normalization.hpp"
#include "normalization/ml_coupling_normalization_generic.hpp"
#include "library/ml_coupling_library.hpp"
#include "library/ml_coupling_library_generic.hpp"
#include "application/ml_coupling_application.hpp"
#include "application/ml_coupling_application_generic.hpp"
#include "ml_coupling.hpp"
#include "config.hpp"

// ============================================================================
// Error Handling Infrastructure
// ============================================================================

thread_local std::string g_last_error_message = "";
thread_local int g_last_error_code = CMI_SUCCESS;

static void set_error(int code, const std::string& msg) {
    g_last_error_code = code;
    g_last_error_message = msg;
}

static void clear_error() {
    g_last_error_code = CMI_SUCCESS;
    g_last_error_message.clear();
}

extern "C" const char* cmi_get_last_error(void) {
    return g_last_error_message.c_str();
}

extern "C" void cmi_clear_last_error(void) {
    clear_error();
}

extern "C" const char* cmi_status_string(int status) {
    switch (status) {
        case CMI_SUCCESS:                return "Success";
        case CMI_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case CMI_ERROR_NULL_POINTER:     return "Null pointer";
        case CMI_ERROR_OUT_OF_RANGE:     return "Out of range";
        case CMI_ERROR_TYPE_MISMATCH:    return "Type mismatch";
        case CMI_ERROR_RUNTIME:          return "Runtime error";
        case CMI_ERROR_NOT_IMPLEMENTED:  return "Not implemented";
        default:                         return "Unknown error";
    }
}

// ============================================================================
// Internal Opaque Handle Structures
// ============================================================================

struct cmi_tensor_s {
    int data_type = CMI_DTYPE_INVALID;
    std::variant<
        std::monostate,
        MLCouplingTensor<double>,
        MLCouplingTensor<float>,
        MLCouplingTensor<int8_t>,
        MLCouplingTensor<int16_t>,
        MLCouplingTensor<int32_t>,
        MLCouplingTensor<int64_t>,
        MLCouplingTensor<uint8_t>,
        MLCouplingTensor<uint16_t>
    > tensor;
    std::vector<int> cached_dims;
};

struct cmi_data_s {
    int data_type = CMI_DTYPE_INVALID;
    std::variant<
        std::monostate,
        MLCouplingData<double>,
        MLCouplingData<float>,
        MLCouplingData<int8_t>,
        MLCouplingData<int16_t>,
        MLCouplingData<int32_t>,
        MLCouplingData<int64_t>,
        MLCouplingData<uint8_t>,
        MLCouplingData<uint16_t>
    > data;
};

struct cmi_behavior_s {
    std::unique_ptr<MLCouplingBehavior> obj;
    MLCouplingBehavior* release_raw() { return obj.release(); }
};

struct cmi_normalization_s {
    int in_type = CMI_DTYPE_INVALID;
    int out_type = CMI_DTYPE_INVALID;
    virtual ~cmi_normalization_s() = default;
    virtual void normalize(cmi_data_s* input) = 0;
    virtual void denormalize(cmi_data_s* output) = 0;
    virtual void* get_raw() = 0;
    virtual void* release_raw() = 0;
};

template <typename In, typename Out>
struct cmi_normalization_impl : public cmi_normalization_s {
    std::unique_ptr<MLCouplingNormalization<In, Out>> obj;

    cmi_normalization_impl(std::unique_ptr<MLCouplingNormalization<In, Out>> o)
        : obj(std::move(o)) {
        in_type = static_cast<int>(to_ml_coupling_data_type<In>());
        out_type = static_cast<int>(to_ml_coupling_data_type<Out>());
    }

    void normalize(cmi_data_s* input) override {
        if (!obj) throw std::runtime_error("Normalization object already released or destroyed.");
        auto& d = std::get<MLCouplingData<In>>(input->data);
        obj->normalize_input(d);
    }

    void denormalize(cmi_data_s* output) override {
        if (!obj) throw std::runtime_error("Normalization object already released or destroyed.");
        auto& d = std::get<MLCouplingData<Out>>(output->data);
        obj->denormalize_output(d);
    }

    void* get_raw() override { return obj.get(); }
    void* release_raw() override { return obj.release(); }
};

struct cmi_library_s {
    int in_type = CMI_DTYPE_INVALID;
    int out_type = CMI_DTYPE_INVALID;
    virtual ~cmi_library_s() = default;
    virtual void static_inference(cmi_data_s* in, cmi_data_s* out) = 0;
    virtual void static_train(cmi_data_s* in, cmi_data_s* target) = 0;
    virtual void flex_ordered_set(cmi_data_s* in) = 0;
    virtual void flex_ordered_set_target(cmi_data_s* target) = 0;
    virtual void flex_ordered_inference(cmi_data_s* fallback_output) = 0;
    virtual void flex_keyed_set(const std::string& key, cmi_data_s* in) = 0;
    virtual void flex_keyed_set_target(const std::string& key, cmi_data_s* target) = 0;
    virtual void flex_keyed_inference(const std::vector<std::string>& in_keys,
                                      const std::vector<std::string>& out_keys,
                                      cmi_data_s* fallback_output) = 0;
    virtual void set_rank(int r) = 0;
    virtual void set_merge_strategy(int s) = 0;
    virtual void* get_raw() = 0;
    virtual void* release_raw() = 0;
};

template <typename In, typename Out>
struct cmi_library_impl : public cmi_library_s {
    std::unique_ptr<MLCouplingLibrary<In, Out>> obj;

    cmi_library_impl(std::unique_ptr<MLCouplingLibrary<In, Out>> o)
        : obj(std::move(o)) {
        in_type = static_cast<int>(to_ml_coupling_data_type<In>());
        out_type = static_cast<int>(to_ml_coupling_data_type<Out>());
    }

    void static_inference(cmi_data_s* in, cmi_data_s* out) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        auto& in_data = std::get<MLCouplingData<In>>(in->data);
        auto& out_data = std::get<MLCouplingData<Out>>(out->data);
        obj->static_inference(&in_data, &out_data);
    }

    void static_train(cmi_data_s* in, cmi_data_s* target) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        auto& in_data = std::get<MLCouplingData<In>>(in->data);
        auto& target_data = std::get<MLCouplingData<Out>>(target->data);
        obj->static_train(&in_data, &target_data);
    }

    void flex_ordered_set(cmi_data_s* in) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->flex_ordered_set(std::get<MLCouplingData<In>>(in->data));
    }

    void flex_ordered_set_target(cmi_data_s* target) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->flex_ordered_set_target(std::get<MLCouplingData<Out>>(target->data));
    }

    void flex_ordered_inference(cmi_data_s* fallback_output) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        if (fallback_output) {
            auto& out = std::get<MLCouplingData<Out>>(fallback_output->data);
            obj->flex_ordered_inference(&out);
        } else {
            obj->flex_ordered_inference(nullptr);
        }
    }

    void flex_keyed_set(const std::string& key, cmi_data_s* in) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->flex_keyed_set(key, std::get<MLCouplingData<In>>(in->data));
    }

    void flex_keyed_set_target(const std::string& key, cmi_data_s* target) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->flex_keyed_set_target(key, std::get<MLCouplingData<Out>>(target->data));
    }

    void flex_keyed_inference(const std::vector<std::string>& in_keys,
                              const std::vector<std::string>& out_keys,
                              cmi_data_s* fallback_output) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        if (fallback_output) {
            auto& out = std::get<MLCouplingData<Out>>(fallback_output->data);
            obj->flex_keyed_inference(in_keys, out_keys, &out);
        } else {
            obj->flex_keyed_inference(in_keys, out_keys, nullptr);
        }
    }

    void set_rank(int r) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->set_rank(r);
    }

    void set_merge_strategy(int s) override {
        if (!obj) throw std::runtime_error("Library object already released or destroyed.");
        obj->set_merge_strategy(static_cast<MLCouplingMergeStrategy>(s));
    }

    void* get_raw() override { return obj.get(); }
    void* release_raw() override { return obj.release(); }
};

struct cmi_application_s {
    int ci_type = CMI_DTYPE_INVALID;
    int co_type = CMI_DTYPE_INVALID;
    int li_type = CMI_DTYPE_INVALID;
    int lo_type = CMI_DTYPE_INVALID;
    virtual ~cmi_application_s() = default;
    virtual int ml_step(cmi_library_s* lib, cmi_behavior_s* behavior) = 0;
    virtual void* get_raw() = 0;
    virtual void* release_raw() = 0;
};

template <typename CI, typename CO, typename LI, typename LO>
struct cmi_application_impl : public cmi_application_s {
    std::unique_ptr<MLCouplingApplication<CI, CO, LI, LO>> obj;

    cmi_application_impl(std::unique_ptr<MLCouplingApplication<CI, CO, LI, LO>> o)
        : obj(std::move(o)) {
        ci_type = static_cast<int>(to_ml_coupling_data_type<CI>());
        co_type = static_cast<int>(to_ml_coupling_data_type<CO>());
        li_type = static_cast<int>(to_ml_coupling_data_type<LI>());
        lo_type = static_cast<int>(to_ml_coupling_data_type<LO>());
    }

    int ml_step(cmi_library_s* lib_wrapper, cmi_behavior_s* behavior_wrapper) override {
        if (!obj) throw std::runtime_error("Application object already released or destroyed.");
        auto* raw_lib = static_cast<MLCouplingLibrary<LI, LO>*>(lib_wrapper->get_raw());
        if (!raw_lib) throw std::runtime_error("Library pointer inside wrapper is null.");
        auto* raw_beh = behavior_wrapper ? behavior_wrapper->obj.get() : nullptr;
        if (!raw_beh) throw std::runtime_error("Behavior pointer inside wrapper is null.");
        return obj->ml_step(*raw_lib, *raw_beh);
    }

    void* get_raw() override { return obj.get(); }
    void* release_raw() override { return obj.release(); }
};

struct cmi_coupling_s {
    int ci_type = CMI_DTYPE_INVALID;
    int co_type = CMI_DTYPE_INVALID;
    int li_type = CMI_DTYPE_INVALID;
    int lo_type = CMI_DTYPE_INVALID;
    virtual ~cmi_coupling_s() = default;
    virtual int step() = 0;
    virtual void train_step(int64_t step_id) = 0;
};

template <typename CI, typename CO, typename LI, typename LO>
struct cmi_coupling_impl : public cmi_coupling_s {
    std::unique_ptr<MLCoupling<CI, CO, LI, LO>> obj;

    cmi_coupling_impl(std::unique_ptr<MLCoupling<CI, CO, LI, LO>> o)
        : obj(std::move(o)) {
        ci_type = static_cast<int>(to_ml_coupling_data_type<CI>());
        co_type = static_cast<int>(to_ml_coupling_data_type<CO>());
        li_type = static_cast<int>(to_ml_coupling_data_type<LI>());
        lo_type = static_cast<int>(to_ml_coupling_data_type<LO>());
    }

    int step() override {
        if (!obj) throw std::runtime_error("Coupling object is null.");
        return obj->step();
    }

    void train_step(int64_t step_id) override {
        if (!obj) throw std::runtime_error("Coupling object is null.");
        obj->train_step(static_cast<long long>(step_id));
    }
};

// Map of legacy pointers to deleters
static std::unordered_map<void*, std::function<void()>> g_legacy_deleters;
static std::mutex g_legacy_deleters_mutex;

// Helper to convert C array parameters to params_map
static std::unordered_map<std::string, std::pair<int, void*>> pack_c_params(
    const char** param_names,
    const void** param_values,
    const int* param_types,
    int param_count)
{
    std::unordered_map<std::string, std::pair<int, void*>> params_map;
    for (int i = 0; i < param_count; ++i) {
        int type_tag = param_types ? param_types[i] : 0;
        params_map[std::string(param_names[i])] = std::make_pair(type_tag, const_cast<void*>(param_values[i]));
    }
    return params_map;
}

// ============================================================================
// Tensor Implementation
// ============================================================================

extern "C" int cmi_tensor_create_flat(cmi_tensor_t* out,
                                      void* data,
                                      const int* dims,
                                      int ndims,
                                      int data_type,
                                      int layout,
                                      int ownership)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out pointer is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data pointer is null"); return CMI_ERROR_NULL_POINTER; }
    if (!dims || ndims <= 0) { set_error(CMI_ERROR_INVALID_ARGUMENT, "dims is null or ndims <= 0"); return CMI_ERROR_INVALID_ARGUMENT; }

    std::vector<int> dims_vec(dims, dims + ndims);
    auto ml_layout = static_cast<MLCouplingMemoryLayout>(layout);
    auto ml_owner = static_cast<MLCouplingOwnership>(ownership);

    try {
        auto t = std::make_unique<cmi_tensor_s>();
        t->data_type = data_type;

        switch (data_type) {
            case CMI_DTYPE_DOUBLE:
                t->tensor = MLCouplingTensor<double>::wrap_flat(static_cast<double*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_FLOAT:
                t->tensor = MLCouplingTensor<float>::wrap_flat(static_cast<float*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_INT8:
                t->tensor = MLCouplingTensor<int8_t>::wrap_flat(static_cast<int8_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_INT16:
                t->tensor = MLCouplingTensor<int16_t>::wrap_flat(static_cast<int16_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_INT32:
                t->tensor = MLCouplingTensor<int32_t>::wrap_flat(static_cast<int32_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_INT64:
                t->tensor = MLCouplingTensor<int64_t>::wrap_flat(static_cast<int64_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_UINT8:
                t->tensor = MLCouplingTensor<uint8_t>::wrap_flat(static_cast<uint8_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            case CMI_DTYPE_UINT16:
                t->tensor = MLCouplingTensor<uint16_t>::wrap_flat(static_cast<uint16_t*>(data), dims_vec, ml_layout, ml_owner);
                break;
            default:
                set_error(CMI_ERROR_INVALID_ARGUMENT, "Unsupported data type: " + std::to_string(data_type));
                return CMI_ERROR_INVALID_ARGUMENT;
        }

        clear_error();
        *out = t.release();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_tensor_create_from_copy(cmi_tensor_t* out,
                                           const void* data,
                                           const int* dims,
                                           int ndims,
                                           int data_type,
                                           int layout)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out pointer is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data pointer is null"); return CMI_ERROR_NULL_POINTER; }
    if (!dims || ndims <= 0) { set_error(CMI_ERROR_INVALID_ARGUMENT, "dims is null or ndims <= 0"); return CMI_ERROR_INVALID_ARGUMENT; }

    std::vector<int> dims_vec(dims, dims + ndims);
    size_t total = 1;
    for (int d : dims_vec) total *= static_cast<size_t>(d);
    auto ml_layout = static_cast<MLCouplingMemoryLayout>(layout);

    try {
        auto t = std::make_unique<cmi_tensor_s>();
        t->data_type = data_type;

        #define COPY_TENSOR_CASE(DTYPE, CPP_TYPE) \
            case DTYPE: { \
                const CPP_TYPE* p = static_cast<const CPP_TYPE*>(data); \
                std::vector<CPP_TYPE> vec(p, p + total); \
                t->tensor = MLCouplingTensor<CPP_TYPE>::from_flat_copy(std::move(vec), dims_vec, ml_layout); \
                break; \
            }

        switch (data_type) {
            COPY_TENSOR_CASE(CMI_DTYPE_DOUBLE, double)
            COPY_TENSOR_CASE(CMI_DTYPE_FLOAT,  float)
            COPY_TENSOR_CASE(CMI_DTYPE_INT8,   int8_t)
            COPY_TENSOR_CASE(CMI_DTYPE_INT16,  int16_t)
            COPY_TENSOR_CASE(CMI_DTYPE_INT32,  int32_t)
            COPY_TENSOR_CASE(CMI_DTYPE_INT64,  int64_t)
            COPY_TENSOR_CASE(CMI_DTYPE_UINT8,  uint8_t)
            COPY_TENSOR_CASE(CMI_DTYPE_UINT16, uint16_t)
            default:
                set_error(CMI_ERROR_INVALID_ARGUMENT, "Unsupported data type: " + std::to_string(data_type));
                return CMI_ERROR_INVALID_ARGUMENT;
        }
        #undef COPY_TENSOR_CASE

        clear_error();
        *out = t.release();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_tensor_destroy(cmi_tensor_t tensor) {
    if (!tensor) return CMI_SUCCESS;
    delete tensor;
    return CMI_SUCCESS;
}

extern "C" int cmi_tensor_get_data(cmi_tensor_t tensor, void** out_data) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_data) { set_error(CMI_ERROR_NULL_POINTER, "out_data is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([out_data](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            *out_data = arg.root();
            return CMI_SUCCESS;
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_get_dims(cmi_tensor_t tensor, const int** out_dims, int* out_ndims) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_dims || !out_ndims) { set_error(CMI_ERROR_NULL_POINTER, "out_dims or out_ndims is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([tensor, out_dims, out_ndims](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            tensor->cached_dims = arg.dimensions();
            *out_dims = tensor->cached_dims.data();
            *out_ndims = static_cast<int>(tensor->cached_dims.size());
            return CMI_SUCCESS;
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_get_layout(cmi_tensor_t tensor, int* out_layout) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_layout) { set_error(CMI_ERROR_NULL_POINTER, "out_layout is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([out_layout](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            *out_layout = static_cast<int>(arg.layout());
            return CMI_SUCCESS;
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_get_data_type(cmi_tensor_t tensor, int* out_data_type) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_data_type) { set_error(CMI_ERROR_NULL_POINTER, "out_data_type is null"); return CMI_ERROR_NULL_POINTER; }
    *out_data_type = tensor->data_type;
    return CMI_SUCCESS;
}

extern "C" int cmi_tensor_get_numel(cmi_tensor_t tensor, size_t* out_numel) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_numel) { set_error(CMI_ERROR_NULL_POINTER, "out_numel is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([out_numel](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            *out_numel = arg.numel();
            return CMI_SUCCESS;
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_get_ownership(cmi_tensor_t tensor, int* out_ownership) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_ownership) { set_error(CMI_ERROR_NULL_POINTER, "out_ownership is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([out_ownership](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            *out_ownership = static_cast<int>(arg.ownership());
            return CMI_SUCCESS;
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_flatten(cmi_tensor_t tensor, int target_layout, cmi_tensor_t* out) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }

    auto ml_target = static_cast<MLCouplingMemoryLayout>(target_layout);

    return std::visit([tensor, ml_target, out](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            try {
                auto new_t = std::make_unique<cmi_tensor_s>();
                new_t->data_type = tensor->data_type;
                new_t->tensor = arg.flatten(ml_target);
                *out = new_t.release();
                clear_error();
                return CMI_SUCCESS;
            } catch (const std::exception& e) {
                set_error(CMI_ERROR_RUNTIME, e.what());
                return CMI_ERROR_RUNTIME;
            }
        }
    }, tensor->tensor);
}

extern "C" int cmi_tensor_deep_copy(cmi_tensor_t tensor, cmi_tensor_t* out) {
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([tensor, out](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "tensor is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            try {
                auto new_t = std::make_unique<cmi_tensor_s>();
                new_t->data_type = tensor->data_type;
                new_t->tensor = arg.deep_copy();
                *out = new_t.release();
                clear_error();
                return CMI_SUCCESS;
            } catch (const std::exception& e) {
                set_error(CMI_ERROR_RUNTIME, e.what());
                return CMI_ERROR_RUNTIME;
            }
        }
    }, tensor->tensor);
}

// ============================================================================
// Data (Collection of Tensors) Implementation
// ============================================================================

extern "C" int cmi_data_create(cmi_data_t* out, int data_type) {
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }

    auto d = std::make_unique<cmi_data_s>();
    d->data_type = data_type;

    switch (data_type) {
        case CMI_DTYPE_DOUBLE: d->data = MLCouplingData<double>(); break;
        case CMI_DTYPE_FLOAT:  d->data = MLCouplingData<float>();  break;
        case CMI_DTYPE_INT8:   d->data = MLCouplingData<int8_t>(); break;
        case CMI_DTYPE_INT16:  d->data = MLCouplingData<int16_t>();break;
        case CMI_DTYPE_INT32:  d->data = MLCouplingData<int32_t>();break;
        case CMI_DTYPE_INT64:  d->data = MLCouplingData<int64_t>();break;
        case CMI_DTYPE_UINT8:  d->data = MLCouplingData<uint8_t>();break;
        case CMI_DTYPE_UINT16: d->data = MLCouplingData<uint16_t>();break;
        default:
            set_error(CMI_ERROR_INVALID_ARGUMENT, "Unsupported data type: " + std::to_string(data_type));
            return CMI_ERROR_INVALID_ARGUMENT;
    }

    clear_error();
    *out = d.release();
    return CMI_SUCCESS;
}

extern "C" int cmi_data_destroy(cmi_data_t data) {
    if (!data) return CMI_SUCCESS;
    delete data;
    return CMI_SUCCESS;
}

extern "C" int cmi_data_add_tensor(cmi_data_t data, cmi_tensor_t tensor) {
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    if (!tensor) { set_error(CMI_ERROR_NULL_POINTER, "tensor is null"); return CMI_ERROR_NULL_POINTER; }

    if (data->data_type != tensor->data_type) {
        set_error(CMI_ERROR_TYPE_MISMATCH, "data and tensor types do not match");
        return CMI_ERROR_TYPE_MISMATCH;
    }

    #define ADD_TENSOR_CASE(DTYPE, CPP_TYPE) \
        case DTYPE: { \
            auto& d = std::get<MLCouplingData<CPP_TYPE>>(data->data); \
            auto& t = std::get<MLCouplingTensor<CPP_TYPE>>(tensor->tensor); \
            d.add_tensor(t); \
            break; \
        }

    try {
        switch (data->data_type) {
            ADD_TENSOR_CASE(CMI_DTYPE_DOUBLE, double)
            ADD_TENSOR_CASE(CMI_DTYPE_FLOAT,  float)
            ADD_TENSOR_CASE(CMI_DTYPE_INT8,   int8_t)
            ADD_TENSOR_CASE(CMI_DTYPE_INT16,  int16_t)
            ADD_TENSOR_CASE(CMI_DTYPE_INT32,  int32_t)
            ADD_TENSOR_CASE(CMI_DTYPE_INT64,  int64_t)
            ADD_TENSOR_CASE(CMI_DTYPE_UINT8,  uint8_t)
            ADD_TENSOR_CASE(CMI_DTYPE_UINT16, uint16_t)
            default:
                set_error(CMI_ERROR_INVALID_ARGUMENT, "Unsupported data type");
                return CMI_ERROR_INVALID_ARGUMENT;
        }
        #undef ADD_TENSOR_CASE
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_data_get_tensor(cmi_data_t data, int index, cmi_tensor_t* out_tensor) {
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_tensor) { set_error(CMI_ERROR_NULL_POINTER, "out_tensor is null"); return CMI_ERROR_NULL_POINTER; }

    #define GET_TENSOR_CASE(DTYPE, CPP_TYPE) \
        case DTYPE: { \
            auto& d = std::get<MLCouplingData<CPP_TYPE>>(data->data); \
            if (index < 0 || static_cast<size_t>(index) >= d.size()) { \
                set_error(CMI_ERROR_OUT_OF_RANGE, "Index out of range: " + std::to_string(index)); \
                return CMI_ERROR_OUT_OF_RANGE; \
            } \
            auto t = std::make_unique<cmi_tensor_s>(); \
            t->data_type = data->data_type; \
            t->tensor = d[index]; \
            *out_tensor = t.release(); \
            break; \
        }

    try {
        switch (data->data_type) {
            GET_TENSOR_CASE(CMI_DTYPE_DOUBLE, double)
            GET_TENSOR_CASE(CMI_DTYPE_FLOAT,  float)
            GET_TENSOR_CASE(CMI_DTYPE_INT8,   int8_t)
            GET_TENSOR_CASE(CMI_DTYPE_INT16,  int16_t)
            GET_TENSOR_CASE(CMI_DTYPE_INT32,  int32_t)
            GET_TENSOR_CASE(CMI_DTYPE_INT64,  int64_t)
            GET_TENSOR_CASE(CMI_DTYPE_UINT8,  uint8_t)
            GET_TENSOR_CASE(CMI_DTYPE_UINT16, uint16_t)
            default:
                set_error(CMI_ERROR_INVALID_ARGUMENT, "Unsupported data type");
                return CMI_ERROR_INVALID_ARGUMENT;
        }
        #undef GET_TENSOR_CASE
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_data_size(cmi_data_t data, int* out_size) {
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_size) { set_error(CMI_ERROR_NULL_POINTER, "out_size is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([out_size](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "data is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            *out_size = static_cast<int>(arg.size());
            return CMI_SUCCESS;
        }
    }, data->data);
}

extern "C" int cmi_data_get_data_type(cmi_data_t data, int* out_data_type) {
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out_data_type) { set_error(CMI_ERROR_NULL_POINTER, "out_data_type is null"); return CMI_ERROR_NULL_POINTER; }
    *out_data_type = data->data_type;
    return CMI_SUCCESS;
}

extern "C" int cmi_data_deep_copy(cmi_data_t data, cmi_data_t* out) {
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }

    return std::visit([data, out](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            set_error(CMI_ERROR_INVALID_ARGUMENT, "data is uninitialized");
            return CMI_ERROR_INVALID_ARGUMENT;
        } else {
            try {
                auto new_d = std::make_unique<cmi_data_s>();
                new_d->data_type = data->data_type;
                new_d->data = arg.deep_copy();
                *out = new_d.release();
                clear_error();
                return CMI_SUCCESS;
            } catch (const std::exception& e) {
                set_error(CMI_ERROR_RUNTIME, e.what());
                return CMI_ERROR_RUNTIME;
            }
        }
    }, data->data);
}

// ============================================================================
// Behavior Implementation
// ============================================================================

extern "C" int cmi_behavior_create_generic(cmi_behavior_t* out,
                                           cmi_should_infer_fn should_infer,
                                           cmi_time_step_delta_fn time_step_delta,
                                           cmi_should_send_data_fn should_send_data,
                                           void* user_data)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!should_infer || !time_step_delta || !should_send_data) {
        set_error(CMI_ERROR_NULL_POINTER, "Callback function pointers must not be null");
        return CMI_ERROR_NULL_POINTER;
    }

    try {
        auto beh = std::make_unique<MLCouplingBehaviorGeneric>(
            [should_infer, user_data]() { return should_infer(user_data); },
            [time_step_delta, user_data]() { return time_step_delta(user_data); },
            [should_send_data, user_data]() { return should_send_data(user_data); }
        );
        auto w = std::make_unique<cmi_behavior_s>();
        w->obj = std::move(beh);
        *out = w.release();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_create_default(cmi_behavior_t* out) {
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        auto w = std::make_unique<cmi_behavior_s>();
        w->obj = std::make_unique<MLCouplingBehaviorDefault>();
        *out = w.release();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_create_periodic(cmi_behavior_t* out,
                                            int inference_interval,
                                            int coupled_steps_before_inference,
                                            int coupled_steps_stride,
                                            int step_increment_after_inference)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        auto w = std::make_unique<cmi_behavior_s>();
        w->obj = std::make_unique<MLCouplingBehaviorPeriodic>(
            inference_interval,
            coupled_steps_before_inference,
            coupled_steps_stride,
            step_increment_after_inference
        );
        *out = w.release();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_create_by_name(cmi_behavior_t* out,
                                           const char* name,
                                           const char** param_names,
                                           const void** param_values,
                                           const int* param_types,
                                           int param_count)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!name) { set_error(CMI_ERROR_NULL_POINTER, "name is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        auto params = pack_c_params(param_names, param_values, param_types, param_count);
        auto* beh = create_instance_mlcouplingbehavior(name, params);
        if (!beh) {
            set_error(CMI_ERROR_RUNTIME, "Factory failed to create behavior: " + std::string(name));
            return CMI_ERROR_RUNTIME;
        }
        auto w = std::make_unique<cmi_behavior_s>();
        w->obj = std::unique_ptr<MLCouplingBehavior>(beh);
        *out = w.release();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_should_perform_inference(cmi_behavior_t behavior, bool* out) {
    if (!behavior || !behavior->obj) { set_error(CMI_ERROR_NULL_POINTER, "behavior is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        *out = behavior->obj->should_perform_inference();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_time_step_delta(cmi_behavior_t behavior, int* out) {
    if (!behavior || !behavior->obj) { set_error(CMI_ERROR_NULL_POINTER, "behavior is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        *out = behavior->obj->time_step_delta();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_should_send_data(cmi_behavior_t behavior, bool* out) {
    if (!behavior || !behavior->obj) { set_error(CMI_ERROR_NULL_POINTER, "behavior is null"); return CMI_ERROR_NULL_POINTER; }
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        *out = behavior->obj->should_send_data();
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_behavior_destroy(cmi_behavior_t behavior) {
    if (!behavior) return CMI_SUCCESS;
    delete behavior;
    return CMI_SUCCESS;
}

// ============================================================================
// Normalization Implementation
// ============================================================================

#define DISPATCH_IN_OUT_TYPES(in_t, out_t, ...) \
    if (in_t == CMI_DTYPE_FLOAT && out_t == CMI_DTYPE_FLOAT) { \
        using InType = float; using OutType = float; __VA_ARGS__ \
    } else if (in_t == CMI_DTYPE_DOUBLE && out_t == CMI_DTYPE_DOUBLE) { \
        using InType = double; using OutType = double; __VA_ARGS__ \
    } else if (in_t == CMI_DTYPE_FLOAT && out_t == CMI_DTYPE_DOUBLE) { \
        using InType = float; using OutType = double; __VA_ARGS__ \
    } else if (in_t == CMI_DTYPE_DOUBLE && out_t == CMI_DTYPE_FLOAT) { \
        using InType = double; using OutType = float; __VA_ARGS__ \
    } else { \
        set_error(CMI_ERROR_TYPE_MISMATCH, "Only float and double in/out combinations are supported."); \
        return CMI_ERROR_TYPE_MISMATCH; \
    }

extern "C" int cmi_normalization_create_generic(cmi_normalization_t* out,
                                                int in_type,
                                                int out_type,
                                                cmi_normalize_fn normalize_fn,
                                                cmi_denormalize_fn denormalize_fn,
                                                void* user_data)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!normalize_fn || !denormalize_fn) {
        set_error(CMI_ERROR_NULL_POINTER, "Callback functions must not be null");
        return CMI_ERROR_NULL_POINTER;
    }

    try {
        DISPATCH_IN_OUT_TYPES(in_type, out_type, {
            auto norm = std::make_unique<MLCouplingNormalizationGeneric<InType, OutType>>(
                [normalize_fn, user_data, in_type](MLCouplingData<InType> in_data) {
                    cmi_data_s w;
                    w.data_type = in_type;
                    w.data = in_data;
                    normalize_fn(&w, user_data);
                },
                [denormalize_fn, user_data, out_type](MLCouplingData<OutType> out_data) {
                    cmi_data_s w;
                    w.data_type = out_type;
                    w.data = out_data;
                    denormalize_fn(&w, user_data);
                }
            );
            *out = new cmi_normalization_impl<InType, OutType>(std::move(norm));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_normalization_create_minmax(cmi_normalization_t* out,
                                               int in_type,
                                               int out_type,
                                               double in_min,
                                               double in_max,
                                               double out_min,
                                               double out_max)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        DISPATCH_IN_OUT_TYPES(in_type, out_type, {
            auto norm = std::make_unique<MLCouplingMinMaxNormalization<InType, OutType>>(
                static_cast<InType>(in_min), static_cast<InType>(in_max),
                static_cast<OutType>(out_min), static_cast<OutType>(out_max)
            );
            *out = new cmi_normalization_impl<InType, OutType>(std::move(norm));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_normalization_create_by_name(cmi_normalization_t* out,
                                                const char* name,
                                                int in_type,
                                                int out_type,
                                                const char** param_names,
                                                const void** param_values,
                                                const int* param_types,
                                                int param_count)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!name) { set_error(CMI_ERROR_NULL_POINTER, "name is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        auto params = pack_c_params(param_names, param_values, param_types, param_count);
        DISPATCH_IN_OUT_TYPES(in_type, out_type, {
            auto* norm = create_instance_mlcouplingnormalization<InType, OutType>(name, params);
            if (!norm) {
                set_error(CMI_ERROR_RUNTIME, "Factory failed to create normalization: " + std::string(name));
                return CMI_ERROR_RUNTIME;
            }
            *out = new cmi_normalization_impl<InType, OutType>(std::unique_ptr<MLCouplingNormalization<InType, OutType>>(norm));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_normalization_normalize_input(cmi_normalization_t norm, cmi_data_t input_data) {
    if (!norm) { set_error(CMI_ERROR_NULL_POINTER, "norm is null"); return CMI_ERROR_NULL_POINTER; }
    if (!input_data) { set_error(CMI_ERROR_NULL_POINTER, "input_data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        norm->normalize(input_data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_normalization_denormalize_output(cmi_normalization_t norm, cmi_data_t output_data) {
    if (!norm) { set_error(CMI_ERROR_NULL_POINTER, "norm is null"); return CMI_ERROR_NULL_POINTER; }
    if (!output_data) { set_error(CMI_ERROR_NULL_POINTER, "output_data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        norm->denormalize(output_data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_normalization_destroy(cmi_normalization_t norm) {
    if (!norm) return CMI_SUCCESS;
    delete norm;
    return CMI_SUCCESS;
}

// ============================================================================
// Library Implementation
// ============================================================================

extern "C" int cmi_library_create_generic(cmi_library_t* out,
                                         int in_type,
                                         int out_type,
                                         cmi_inference_fn inference_fn,
                                         void* user_data)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!inference_fn) { set_error(CMI_ERROR_NULL_POINTER, "inference_fn is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        DISPATCH_IN_OUT_TYPES(in_type, out_type, {
            auto lib = std::make_unique<MLCouplingLibraryGeneric<InType, OutType>>(
                [inference_fn, user_data, in_type, out_type](MLCouplingData<InType>* in, MLCouplingData<OutType>* out) {
                    cmi_data_s in_w;
                    in_w.data_type = in_type;
                    in_w.data = *in;

                    cmi_data_s out_w;
                    out_w.data_type = out_type;
                    out_w.data = *out;

                    inference_fn(&in_w, &out_w, user_data);
                }
            );
            *out = new cmi_library_impl<InType, OutType>(std::move(lib));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_create_by_name(cmi_library_t* out,
                                         const char* name,
                                         int in_type,
                                         int out_type,
                                         const char** param_names,
                                         const void** param_values,
                                         const int* param_types,
                                         int param_count)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!name) { set_error(CMI_ERROR_NULL_POINTER, "name is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        auto params = pack_c_params(param_names, param_values, param_types, param_count);
        DISPATCH_IN_OUT_TYPES(in_type, out_type, {
            auto* lib = create_instance_mlcouplinglibrary<InType, OutType>(name, params);
            if (!lib) {
                set_error(CMI_ERROR_RUNTIME, "Factory failed to create library: " + std::string(name));
                return CMI_ERROR_RUNTIME;
            }
            *out = new cmi_library_impl<InType, OutType>(std::unique_ptr<MLCouplingLibrary<InType, OutType>>(lib));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_set_rank(cmi_library_t lib, int rank) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->set_rank(rank);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_set_merge_strategy(cmi_library_t lib, int strategy) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->set_merge_strategy(strategy);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_inference(cmi_library_t lib, cmi_data_t input, cmi_data_t output) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!input) { set_error(CMI_ERROR_NULL_POINTER, "input is null"); return CMI_ERROR_NULL_POINTER; }
    if (!output) { set_error(CMI_ERROR_NULL_POINTER, "output is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        lib->static_inference(input, output);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_train(cmi_library_t lib, cmi_data_t input, cmi_data_t target) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!input) { set_error(CMI_ERROR_NULL_POINTER, "input is null"); return CMI_ERROR_NULL_POINTER; }
    if (!target) { set_error(CMI_ERROR_NULL_POINTER, "target is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        lib->static_train(input, target);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_ordered_set(cmi_library_t lib, cmi_data_t data) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->flex_ordered_set(data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_ordered_set_target(cmi_library_t lib, cmi_data_t data) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->flex_ordered_set_target(data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_ordered_inference(cmi_library_t lib, cmi_data_t fallback_output) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->flex_ordered_inference(fallback_output);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_keyed_set(cmi_library_t lib, const char* key, cmi_data_t data) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!key) { set_error(CMI_ERROR_NULL_POINTER, "key is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->flex_keyed_set(key, data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_keyed_set_target(cmi_library_t lib, const char* key, cmi_data_t data) {
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!key) { set_error(CMI_ERROR_NULL_POINTER, "key is null"); return CMI_ERROR_NULL_POINTER; }
    if (!data) { set_error(CMI_ERROR_NULL_POINTER, "data is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        lib->flex_keyed_set_target(key, data);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_flex_keyed_inference(cmi_library_t lib,
                                              const char** in_keys,
                                              int in_key_count,
                                              const char** out_keys,
                                              int out_key_count,
                                              cmi_data_t fallback_output)
{
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        std::vector<std::string> ik, ok;
        if (in_keys && in_key_count > 0) {
            for (int i = 0; i < in_key_count; ++i) ik.push_back(in_keys[i]);
        }
        if (out_keys && out_key_count > 0) {
            for (int i = 0; i < out_key_count; ++i) ok.push_back(out_keys[i]);
        }
        lib->flex_keyed_inference(ik, ok, fallback_output);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_library_destroy(cmi_library_t lib) {
    if (!lib) return CMI_SUCCESS;
    delete lib;
    return CMI_SUCCESS;
}

// ============================================================================
// Application Implementation
// ============================================================================

#define DISPATCH_4_TYPES(ci, co, li, lo, ...) \
    if (ci == CMI_DTYPE_FLOAT && co == CMI_DTYPE_FLOAT && li == CMI_DTYPE_FLOAT && lo == CMI_DTYPE_FLOAT) { \
        using CIType = float; using COType = float; using LIType = float; using LOType = float; __VA_ARGS__ \
    } else if (ci == CMI_DTYPE_DOUBLE && co == CMI_DTYPE_DOUBLE && li == CMI_DTYPE_DOUBLE && lo == CMI_DTYPE_DOUBLE) { \
        using CIType = double; using COType = double; using LIType = double; using LOType = double; __VA_ARGS__ \
    } else if (ci == CMI_DTYPE_FLOAT && co == CMI_DTYPE_FLOAT && li == CMI_DTYPE_DOUBLE && lo == CMI_DTYPE_DOUBLE) { \
        using CIType = float; using COType = float; using LIType = double; using LOType = double; __VA_ARGS__ \
    } else if (ci == CMI_DTYPE_DOUBLE && co == CMI_DTYPE_DOUBLE && li == CMI_DTYPE_FLOAT && lo == CMI_DTYPE_FLOAT) { \
        using CIType = double; using COType = double; using LIType = float; using LOType = float; __VA_ARGS__ \
    } else { \
        set_error(CMI_ERROR_TYPE_MISMATCH, "Unsupported 4-type combination (must be float/double combinations)."); \
        return CMI_ERROR_TYPE_MISMATCH; \
    }

extern "C" int cmi_application_create_generic(cmi_application_t* out,
                                            int coupling_in_type,
                                            int coupling_out_type,
                                            int library_in_type,
                                            int library_out_type,
                                            cmi_data_t coupling_input,
                                            cmi_data_t coupling_output,
                                            cmi_normalization_t normalization)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!coupling_input || !coupling_output) {
        set_error(CMI_ERROR_NULL_POINTER, "coupling_input and coupling_output must not be null");
        return CMI_ERROR_NULL_POINTER;
    }

    try {
        DISPATCH_4_TYPES(coupling_in_type, coupling_out_type, library_in_type, library_out_type, {
            auto cin = std::get<MLCouplingData<CIType>>(coupling_input->data);
            auto cout = std::get<MLCouplingData<COType>>(coupling_output->data);

            MLCouplingNormalization<LIType, COType>* raw_norm = nullptr;
            if (normalization) {
                raw_norm = static_cast<MLCouplingNormalization<LIType, COType>*>(normalization->release_raw());
            }

            auto app = std::make_unique<MLCouplingApplicationGeneric<CIType, COType, LIType, LOType>>(
                cin, cout, nullptr, nullptr, nullptr, raw_norm
            );

            *out = new cmi_application_impl<CIType, COType, LIType, LOType>(std::move(app));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_application_create_by_name(cmi_application_t* out,
                                            const char* name,
                                            int coupling_in_type,
                                            int coupling_out_type,
                                            int library_in_type,
                                            int library_out_type,
                                            const char** param_names,
                                            const void** param_values,
                                            const int* param_types,
                                            int param_count)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!name) { set_error(CMI_ERROR_NULL_POINTER, "name is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        auto params = pack_c_params(param_names, param_values, param_types, param_count);
        DISPATCH_4_TYPES(coupling_in_type, coupling_out_type, library_in_type, library_out_type, {
            auto* app = create_instance_mlcouplingapplication<CIType, COType, LIType, LOType>(name, params);
            if (!app) {
                set_error(CMI_ERROR_RUNTIME, "Factory failed to create application: " + std::string(name));
                return CMI_ERROR_RUNTIME;
            }
            *out = new cmi_application_impl<CIType, COType, LIType, LOType>(
                std::unique_ptr<MLCouplingApplication<CIType, COType, LIType, LOType>>(app)
            );
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_application_ml_step(cmi_application_t app,
                                     cmi_library_t lib,
                                     cmi_behavior_t behavior,
                                     int* out_delta)
{
    if (!app) { set_error(CMI_ERROR_NULL_POINTER, "app is null"); return CMI_ERROR_NULL_POINTER; }
    if (!lib) { set_error(CMI_ERROR_NULL_POINTER, "lib is null"); return CMI_ERROR_NULL_POINTER; }
    if (!behavior) { set_error(CMI_ERROR_NULL_POINTER, "behavior is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        int delta = app->ml_step(lib, behavior);
        if (out_delta) *out_delta = delta;
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_application_destroy(cmi_application_t app) {
    if (!app) return CMI_SUCCESS;
    delete app;
    return CMI_SUCCESS;
}

// ============================================================================
// MLCoupling Orchestrator Implementation
// ============================================================================

extern "C" int cmi_coupling_create(cmi_coupling_t* out,
                                 int coupling_in_type,
                                 int coupling_out_type,
                                 int library_in_type,
                                 int library_out_type,
                                 cmi_library_t library,
                                 cmi_application_t application,
                                 cmi_behavior_t behavior)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!library) { set_error(CMI_ERROR_NULL_POINTER, "library is null"); return CMI_ERROR_NULL_POINTER; }
    if (!application) { set_error(CMI_ERROR_NULL_POINTER, "application is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        DISPATCH_4_TYPES(coupling_in_type, coupling_out_type, library_in_type, library_out_type, {
            auto* raw_lib = static_cast<MLCouplingLibrary<LIType, LOType>*>(library->release_raw());
            auto* raw_app = static_cast<MLCouplingApplication<CIType, COType, LIType, LOType>*>(application->release_raw());
            MLCouplingBehavior* raw_beh = behavior ? behavior->release_raw() : nullptr;

            auto coupling = std::make_unique<MLCoupling<CIType, COType, LIType, LOType>>(
                raw_lib, raw_app, raw_beh
            );

            *out = new cmi_coupling_impl<CIType, COType, LIType, LOType>(std::move(coupling));
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_coupling_create_from_config(cmi_coupling_t* out,
                                             const char* config_path,
                                             int coupling_in_type,
                                             int coupling_out_type,
                                             int library_in_type,
                                             int library_out_type)
{
    if (!out) { set_error(CMI_ERROR_NULL_POINTER, "out is null"); return CMI_ERROR_NULL_POINTER; }
    if (!config_path) { set_error(CMI_ERROR_NULL_POINTER, "config_path is null"); return CMI_ERROR_NULL_POINTER; }

    try {
        DISPATCH_4_TYPES(coupling_in_type, coupling_out_type, library_in_type, library_out_type, {
            std::ifstream file(config_path);
            if (!file.is_open()) {
                set_error(CMI_ERROR_RUNTIME, "Could not open configuration file: " + std::string(config_path));
                return CMI_ERROR_RUNTIME;
            }
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            MLCouplingData<CIType> cin;
            MLCouplingData<COType> cout;

            auto* coupling_ptr = create_mlcoupling_from_config_with_library_types<CIType, COType, LIType, LOType>(
                content, cin, cout, ConfigOverrides()
            );

            if (!coupling_ptr) {
                set_error(CMI_ERROR_RUNTIME, "Failed to create coupling from config file");
                return CMI_ERROR_RUNTIME;
            }

            *out = new cmi_coupling_impl<CIType, COType, LIType, LOType>(
                std::unique_ptr<MLCoupling<CIType, COType, LIType, LOType>>(coupling_ptr)
            );
            clear_error();
            return CMI_SUCCESS;
        })
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_coupling_step(cmi_coupling_t coupling, int* out_delta) {
    if (!coupling) { set_error(CMI_ERROR_NULL_POINTER, "coupling is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        int delta = coupling->step();
        if (out_delta) *out_delta = delta;
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_coupling_train_step(cmi_coupling_t coupling, int64_t step_id) {
    if (!coupling) { set_error(CMI_ERROR_NULL_POINTER, "coupling is null"); return CMI_ERROR_NULL_POINTER; }
    try {
        coupling->train_step(step_id);
        clear_error();
        return CMI_SUCCESS;
    } catch (const std::exception& e) {
        set_error(CMI_ERROR_RUNTIME, e.what());
        return CMI_ERROR_RUNTIME;
    }
}

extern "C" int cmi_coupling_destroy(cmi_coupling_t coupling) {
    if (!coupling) return CMI_SUCCESS;
    delete coupling;
    return CMI_SUCCESS;
}

// ============================================================================
// Introspection Implementation
// ============================================================================

thread_local std::string g_last_class_name_query = "";

extern "C" int cmi_get_class_count(const char* category) {
    if (!category) return -1;
    try {
        return cmi_registry_get_class_count(category);
    } catch (...) {
        return -1;
    }
}

extern "C" const char* cmi_get_class_name(const char* category, int index) {
    if (!category || index < 0) return "";
    try {
        g_last_class_name_query = cmi_registry_get_class_name(category, index);
        return g_last_class_name_query.c_str();
    } catch (...) {
        return "";
    }
}

extern "C" bool cmi_is_class_registered(const char* category, const char* name) {
    if (!category || !name) return false;
    try {
        return cmi_registry_is_class_registered(category, name);
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Backward Compatibility Implementation
// ============================================================================

extern "C" void* create_library(const char* name,
                                int in_selection,
                                int out_selection,
                                char** param_names,
                                void** params,
                                int param_count)
{
    std::unordered_map<std::string, std::pair<int, void*>> params_map;
    for (int i = 0; i < param_count; ++i) {
        params_map[std::string(param_names[i])] = std::pair(0, params[i]);
    }

    MLCouplingCAPISupportedTypes in_tag = get_capi_type_tag(in_selection);
    MLCouplingCAPISupportedTypes out_tag = get_capi_type_tag(out_selection);

    void* result = std::visit([&](auto in_t, auto out_t) -> void* {
        using InType = typename decltype(in_t)::type;
        using OutType = typename decltype(out_t)::type;

        constexpr bool aix_supported_scalar =
            std::is_same_v<InType, float> || std::is_same_v<InType, double>;

        if (std::string(name) == "Aixelerator" &&
            (!std::is_same_v<InType, OutType> || !aix_supported_scalar)) {
            std::cerr << "Warning: AIxelerator library requires same-type float/float or double/double data. Falling back to double.\n";
            auto* lib = create_instance_mlcouplinglibrary<double, double>(name, params_map);
            if (lib) {
                std::lock_guard<std::mutex> lock(g_legacy_deleters_mutex);
                g_legacy_deleters[lib] = [lib]() { delete lib; };
            }
            return static_cast<void*>(lib);
        }

        auto* lib = create_instance_mlcouplinglibrary<InType, OutType>(name, params_map);
        if (lib) {
            std::lock_guard<std::mutex> lock(g_legacy_deleters_mutex);
            g_legacy_deleters[lib] = [lib]() { delete lib; };
        }
        return static_cast<void*>(lib);

    }, in_tag, out_tag);

    return result;
}

extern "C" void destroy_library(void* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(g_legacy_deleters_mutex);
    auto it = g_legacy_deleters.find(handle);
    if (it != g_legacy_deleters.end()) {
        it->second();
        g_legacy_deleters.erase(it);
    }
}
