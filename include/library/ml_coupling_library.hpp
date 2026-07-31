#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include "../data/ml_coupling_data.hpp"

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

#if defined(MPI_FOUND)
#include <mpi.h>
#define MLCOUPLING_PROVIDER_HAS_MPI 1
#elif defined(__has_include)
#if __has_include(<mpi.h>)
#include <mpi.h>
#define MLCOUPLING_PROVIDER_HAS_MPI 1
#endif
#endif


enum class MLCouplingMergeStrategy {
    List,
    Stack,
    Auto,
    None
};

// @category: library
template <typename LibraryInput, typename LibraryOutput>
class MLCouplingLibrary
{
public:
    MLCouplingMergeStrategy merge_strategy = MLCouplingMergeStrategy::Auto;

    void set_merge_strategy(MLCouplingMergeStrategy strategy) {
        merge_strategy = strategy;
    }

    int rank = 0;

    MLCouplingLibrary()
    {
#ifdef MLCOUPLING_PROVIDER_HAS_MPI
        // check if MPI is initialized and get the rank if it is, otherwise set rank to -1
        int flag = 0;
        MPI_Initialized(&flag);
        if (flag)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
#endif
    }

    // Manually set the rank if needed, for example if MPI is initialized after the provider is created, or if we use something other than MPI for parallelism

    MLCouplingLibrary(int rank)
    {
        this->rank = rank;
    }

    virtual ~MLCouplingLibrary() = default;

    void set_rank(int rank)
    {
        this->rank = rank;
    }

    // --- Tier 0: Static (Mandatory) ---
    // Perform inference with the ML model and get the output data
    virtual void static_inference(MLCouplingData<LibraryInput> *input,
                                  MLCouplingData<LibraryOutput> *output) = 0;

    virtual std::map<std::string, double> static_train(MLCouplingData<LibraryInput> *input,
                                                        MLCouplingData<LibraryOutput> *target)
    {
        (void)input;
        (void)target;
        throw std::runtime_error("static_train not implemented for this provider");
    }

    virtual std::size_t get_synchronized_iterations(std::size_t local_iterations) const
    {
        return local_iterations;
    }

    // --- Tier 1: Ordered Flexible (Optional, with Fallback) ---
    virtual void flex_ordered_set(MLCouplingData<LibraryInput> data)
    {
        staged_inputs.push_back(std::move(data));
    }

    virtual void flex_ordered_set_target(MLCouplingData<LibraryOutput> data)
    {
        staged_targets.push_back(std::move(data));
    }

    virtual void flex_ordered_get(MLCouplingData<LibraryOutput> *data)
    {
        // Fallback is a no-op because flex_ordered_inference populates the fallback_output directly.
        (void)data;
    }

    // Simple FNV-1a 64-bit hash for data comparison
    static unsigned long long fnv1a(const void* data, size_t bytes) {
        unsigned long long h = 14695981039346656037ULL;
        const unsigned char* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    static void debug_log_merged(const std::string& label, MLCouplingData<LibraryInput>& data, int rank) {
        if (!std::getenv("DEBUG_PROVIDER_INPUT")) return;
        for (size_t ti = 0; ti < data.size(); ++ti) {
            auto &t = data[ti];
            int n = static_cast<int>(t.numel());
            unsigned long long h = fnv1a(t.root(), n * sizeof(LibraryInput));
            double sum = 0;
            float first = 0, last = 0;
            if constexpr (std::is_same_v<LibraryInput, float>) {
                const float* raw = static_cast<const float*>(t.root());
                first = raw[0]; last = raw[n-1];
                for (int i = 0; i < n; ++i) sum += raw[i];
            }
            std::cerr << label << " rank=" << rank << " tensor=" << ti << " shape=";
            for (int d : t.dimensions()) std::cerr << d << " ";
            std::cerr << " numel=" << n << " sum=" << sum << " fnv1a=" << h << " first=" << first << " last=" << last << std::endl;
        }
    }

    virtual void flex_ordered_inference(MLCouplingData<LibraryOutput> *fallback_output = nullptr)
    {
        if (staged_inputs.size() > 0 && fallback_output != nullptr)
        {
            merge_data(staged_inputs, last_merged_input);
            debug_log_merged("DEBUG MERGED INPUT", last_merged_input, this->rank);
            static_inference(&last_merged_input, fallback_output);
            staged_inputs.clear();
            staged_targets.clear();
        }
        else
        {
            throw std::runtime_error("flex_ordered_inference fallback requires at least 1 staged input and a valid fallback_output.");
        }
    }

    virtual std::map<std::string, double> flex_ordered_train(long long step_id)
    {
        (void)step_id;
        if (staged_inputs.size() > 0 && staged_targets.size() > 0)
        {
            merge_data(staged_inputs, last_merged_input);
            merge_data(staged_targets, last_merged_target);
            auto res = static_train(&last_merged_input, &last_merged_target);
            staged_inputs.clear();
            staged_targets.clear();
            return res;
        }
        throw std::runtime_error("flex_ordered_train fallback expects at least 1 staged input and 1 staged target.");
    }

    // --- Tier 2: Keyed Flexible (Optional, with Fallback) ---
    virtual void flex_keyed_set(const std::string &key, MLCouplingData<LibraryInput> data)
    {
        keyed_inputs[key] = std::move(data);
    }

    virtual void flex_keyed_set_target(const std::string &key, MLCouplingData<LibraryOutput> data)
    {
        keyed_targets[key] = std::move(data);
    }

    virtual void flex_keyed_get(const std::string &key, MLCouplingData<LibraryOutput> *data)
    {
        // Fallback is a no-op because flex_keyed_inference populates the fallback_output directly.
        (void)key;
        (void)data;
    }

    virtual void flex_keyed_inference(const std::vector<std::string> &in_keys,
                                      const std::vector<std::string> &out_keys,
                                       MLCouplingData<LibraryOutput> *fallback_output = nullptr)
    {
        (void)out_keys; // The fallback maps all expected outputs to the static buffer
        if (in_keys.size() > 0 && fallback_output != nullptr)
        {
            merge_data(in_keys, keyed_inputs, last_merged_input);
            debug_log_merged("DEBUG MERGED INPUT", last_merged_input, this->rank);
            static_inference(&last_merged_input, fallback_output);
        }
        else
        {
            throw std::runtime_error("flex_keyed_inference fallback requires at least 1 input key and a valid fallback_output.");
        }
    }

    virtual std::map<std::string, double> flex_keyed_train(long long step_id,
                                                           const std::vector<std::string> &in_keys,
                                                           const std::vector<std::string> &target_keys)
    {
        (void)step_id;
        if (in_keys.size() > 0 && target_keys.size() > 0)
        {
            merge_data(in_keys, keyed_inputs, last_merged_input);
            merge_data(target_keys, keyed_targets, last_merged_target);
            return static_train(&last_merged_input, &last_merged_target);
        }
        throw std::runtime_error("flex_keyed_train fallback expects at least 1 input key and 1 target key.");
    }


protected:
    // Defensive guard for the flex-fallback merge path. When the provider is
    // configured with coupling_type=FLEXIBLE, last_merged_input starts empty and
    // the merge allocates its own (owned) buffer, which the service is then
    // lazily bound to. This guard exists to catch the misuse case where a user
    // configures coupling_type=STATIC (passing a user-managed external input
    // buffer to the constructor) but then *also* calls the flex API
    // (ordered()/keyed()). In that scenario the merge would write into the
    // user's external buffer; if that buffer is not sized for the post-merge
    // layout, silently reallocating would disconnect the service from the data
    // (the anomaly we fixed). We throw loudly instead. Owned buffers (from a
    // previous merge) may always be replaced.
    template <typename T>
    static void guarantee_fallback_buffer_fit(const MLCouplingData<T> &existing_data,
                                              size_t t_idx,
                                              const std::vector<int> &new_dims)
    {
        if (existing_data.size() <= t_idx)
            return; // no buffer present at this tensor index -> merge allocates freely
        const auto &existing_tensor = existing_data[t_idx];
        if (existing_tensor.ownership() != MLCouplingOwnershipExternal)
            return; // merge-allocated buffer from a previous step: may be replaced
        if (existing_tensor.dimensions() == new_dims && existing_tensor.is_contiguous())
            return; // fits -> will be reused by the merge

        std::ostringstream existing_shape, expected_shape;
        for (size_t i = 0; i < existing_tensor.dimensions().size(); ++i)
            existing_shape << (i ? "x" : "") << existing_tensor.dimensions()[i];
        if (existing_tensor.dimensions().empty()) existing_shape << "scalar";
        for (size_t i = 0; i < new_dims.size(); ++i)
            expected_shape << (i ? "x" : "") << new_dims[i];
        if (new_dims.empty()) expected_shape << "scalar";

        throw std::runtime_error(
            "MLCoupling flex-fallback merge: the user-provided (static-coupling) "
            "input buffer has shape [" + existing_shape.str() + "] but merging the "
            "staged inputs requires shape [" + expected_shape.str() + "]. "
            "When using the flexible API with a static-fallback buffer, you must size "
            "that buffer for the *post-merge* layout (e.g. [batch, num_inputs, ...] "
            "for STACK merging), not for a single staged input. Either provide a "
            "correctly-sized buffer, change the merge strategy, or use the flexible "
            "API with coupling_type=FLEXIBLE (recommended).");
    }

    template <typename T>
    MLCouplingData<T> stack_data(const std::vector<MLCouplingData<T>> &data_list, MLCouplingData<T> &existing_data)
    {
        if (data_list.empty()) return MLCouplingData<T>();
        
        size_t num_tensors = data_list[0].size();
        size_t m = data_list.size();
        
        MLCouplingData<T> stacked_data;
        
        for (size_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_merge_stack)
            SCOREP_USER_REGION_BEGIN(handle_merge_stack, "merge_stack", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            auto first_dim = data_list[0][t_idx].dimensions();
            if (first_dim.empty()) {
                throw std::runtime_error("Cannot stack scalar tensors.");
            }
            int B = first_dim[0];
            
            for (size_t i = 1; i < m; ++i) {
                if (data_list[i].size() != num_tensors) {
                    throw std::runtime_error("Stack fallback: Tensor count mismatch.");
                }
                if (data_list[i][t_idx].dimensions() != first_dim) {
                    throw std::runtime_error("Stack fallback: Tensor shape mismatch.");
                }
            }
            
            std::vector<int> new_dims;
            new_dims.push_back(B);
            new_dims.push_back(static_cast<int>(m));
            for (size_t d = 1; d < first_dim.size(); ++d) {
                new_dims.push_back(first_dim[d]);
            }
            
            size_t slice_numel = 1;
            for (size_t d = 1; d < first_dim.size(); ++d) {
                slice_numel *= first_dim[d];
            }
            
            size_t total_numel = static_cast<size_t>(B) * m * slice_numel;
            T* buffer = nullptr;
            bool reuse_buffer = false;

            guarantee_fallback_buffer_fit(existing_data, t_idx, new_dims);

#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_flex_merge_allocate_stack)
            SCOREP_USER_REGION_BEGIN(handle_flex_merge_allocate_stack, "flex_merge_allocate_or_reuse", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            if (existing_data.size() > t_idx && existing_data[t_idx].dimensions() == new_dims && existing_data[t_idx].is_contiguous()) {
                buffer = static_cast<T*>(existing_data[t_idx].root());
                reuse_buffer = true;
            } else {
                buffer = new T[total_numel];
            }
#ifdef USE_SCOREP
            SCOREP_USER_REGION_END(handle_flex_merge_allocate_stack)
#endif
            
            for (size_t i = 0; i < m; ++i) {
                const auto& tensor = data_list[i][t_idx];
                if (tensor.is_contiguous()) {
#ifdef USE_SCOREP
                    SCOREP_USER_REGION_DEFINE(handle_flex_merge_copy_stack)
                    SCOREP_USER_REGION_BEGIN(handle_flex_merge_copy_stack, "flex_merge_copy", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                    const T* src = static_cast<const T*>(tensor.root());
                    for (int b = 0; b < B; ++b) {
                        std::copy(
                            src + b * slice_numel,
                            src + (b + 1) * slice_numel,
                            buffer + (b * m + i) * slice_numel
                        );
                    }
#ifdef USE_SCOREP
                    SCOREP_USER_REGION_END(handle_flex_merge_copy_stack)
#endif
                } else {
#ifdef USE_SCOREP
                    SCOREP_USER_REGION_DEFINE(handle_flex_merge_flatten_stack)
                    SCOREP_USER_REGION_BEGIN(handle_flex_merge_flatten_stack, "flex_merge_flatten_noncontiguous", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                    std::vector<T> flat = tensor.as_flat_vector();
#ifdef USE_SCOREP
                    SCOREP_USER_REGION_END(handle_flex_merge_flatten_stack)
                    SCOREP_USER_REGION_DEFINE(handle_flex_merge_copy_stack2)
                    SCOREP_USER_REGION_BEGIN(handle_flex_merge_copy_stack2, "flex_merge_copy", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                    for (int b = 0; b < B; ++b) {
                        std::copy(
                            flat.begin() + b * slice_numel,
                            flat.begin() + (b + 1) * slice_numel,
                            buffer + (b * m + i) * slice_numel
                        );
                    }
#ifdef USE_SCOREP
                    SCOREP_USER_REGION_END(handle_flex_merge_copy_stack2)
#endif
                }
            }
            
            if (reuse_buffer) {
                stacked_data.add_tensor(existing_data[t_idx]);
            } else {
                auto new_tensor = MLCouplingTensor<T>::wrap_flat(
                    buffer, new_dims, MLCouplingMemLayoutContiguous, MLCouplingOwnershipOwned
                );
                stacked_data.add_tensor(std::move(new_tensor));
            }
#ifdef USE_SCOREP
            SCOREP_USER_REGION_END(handle_merge_stack)
#endif
        }
        return stacked_data;
    }

    template <typename T>
    MLCouplingData<T> list_data(const std::vector<MLCouplingData<T>> &data_list, MLCouplingData<T> &existing_data)
    {
        if (data_list.empty()) return MLCouplingData<T>();

        // If there's only one item in the list, copy it into existing_data to preserve its shape and reuse the buffer.
        if (data_list.size() == 1) {
            MLCouplingData<T> merged_data;
            size_t num_tensors = data_list[0].size();
            for (size_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
                const auto& tensor = data_list[0][t_idx];
                const auto& dims = tensor.dimensions();
                
                T* buffer = nullptr;
                bool reuse_buffer = false;
                guarantee_fallback_buffer_fit(existing_data, t_idx, dims);
                if (existing_data.size() > t_idx && existing_data[t_idx].dimensions() == dims && existing_data[t_idx].is_contiguous()) {
                    buffer = static_cast<T*>(existing_data[t_idx].root());
                    reuse_buffer = true;
                } else {
                    buffer = new T[tensor.numel()];
                }
                
                if (tensor.is_contiguous()) {
                    std::copy(static_cast<const T*>(tensor.root()), static_cast<const T*>(tensor.root()) + tensor.numel(), buffer);
                } else {
                    std::vector<T> flat = tensor.as_flat_vector();
                    std::copy(flat.begin(), flat.end(), buffer);
                }
                
                if (reuse_buffer) {
                    merged_data.add_tensor(existing_data[t_idx]);
                } else {
                    merged_data.add_tensor(MLCouplingTensor<T>::wrap_flat(
                        buffer, dims, MLCouplingMemLayoutContiguous, MLCouplingOwnershipOwned
                    ));
                }
            }
            return merged_data;
        }

        size_t num_tensors = data_list[0].size();
        size_t m = data_list.size();  // number of staged inputs

        MLCouplingData<T> merged_data;

        // Each element of data_list is one staged input. Each staged input has
        // num_tensors tensors (usually 1). For each tensor index t_idx we:
        //   1. Check all staged inputs share the same batch size B (dim 0).
        //   2. Compute each input's per-batch-item slice size (product of dims[1..]).
        //   3. For every batch item b, concatenate the slices from all m inputs
        //      into a flat [total_slice_numel] row.
        // Output tensor shape: [B, TotalFeaturesPerBatch].

        for (size_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_merge_list)
            SCOREP_USER_REGION_BEGIN(handle_merge_list, "merge_list", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            auto first_dim = data_list[0][t_idx].dimensions();
            if (first_dim.empty()) {
                throw std::runtime_error("List fallback: Cannot merge scalar tensors. Expected at least a batch dimension.");
            }

            int B = first_dim[0];
            size_t total_slice_numel = 0;
            std::vector<size_t> slice_numels(m);

            for (size_t i = 0; i < m; ++i) {
                if (data_list[i].size() != num_tensors) {
                    throw std::runtime_error("List fallback: Tensor count mismatch across staged inputs.");
                }
                auto cur_dim = data_list[i][t_idx].dimensions();
                if (cur_dim.empty()) {
                    throw std::runtime_error("List fallback: Encountered a scalar tensor, expected a batch dimension.");
                }
                if (cur_dim[0] != B) {
                    throw std::runtime_error(
                        "List fallback: Batch size mismatch. All staged inputs must share the same batch size (dim 0).");
                }

                // Slice numel = product of all dims except dim 0 (the batch dim).
                size_t slice_numel = 1;
                for (size_t d = 1; d < cur_dim.size(); ++d) {
                    slice_numel *= static_cast<size_t>(cur_dim[d]);
                }
                slice_numels[i] = slice_numel;
                total_slice_numel += slice_numel;
            }

            // Output shape: [B, total_slice_numel]
            std::vector<int> new_dims = {B, static_cast<int>(total_slice_numel)};
            size_t total_numel = static_cast<size_t>(B) * total_slice_numel;

            T* buffer = nullptr;
            bool reuse_buffer = false;
            guarantee_fallback_buffer_fit(existing_data, t_idx, new_dims);
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_flex_merge_allocate_list)
            SCOREP_USER_REGION_BEGIN(handle_flex_merge_allocate_list, "flex_merge_allocate_or_reuse", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            if (existing_data.size() > t_idx &&
                existing_data[t_idx].dimensions() == new_dims &&
                existing_data[t_idx].is_contiguous()) {
                buffer = static_cast<T*>(existing_data[t_idx].root());
                reuse_buffer = true;
            } else {
                buffer = new T[total_numel];
            }
#ifdef USE_SCOREP
            SCOREP_USER_REGION_END(handle_flex_merge_allocate_list)
#endif

            // For each batch item b, copy the per-slice data from each input i.
            for (int b = 0; b < B; ++b) {
                size_t row_offset = static_cast<size_t>(b) * total_slice_numel;
                size_t col_offset = 0;
                for (size_t i = 0; i < m; ++i) {
                    const auto& tensor = data_list[i][t_idx];
                    size_t sn = slice_numels[i];
                    if (tensor.is_contiguous()) {
#ifdef USE_SCOREP
                        SCOREP_USER_REGION_DEFINE(handle_flex_merge_copy_list)
                        SCOREP_USER_REGION_BEGIN(handle_flex_merge_copy_list, "flex_merge_copy", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                        const T* src = static_cast<const T*>(tensor.root());
                        std::copy(src + b * sn, src + (b + 1) * sn, buffer + row_offset + col_offset);
#ifdef USE_SCOREP
                        SCOREP_USER_REGION_END(handle_flex_merge_copy_list)
#endif
                    } else {
#ifdef USE_SCOREP
                        SCOREP_USER_REGION_DEFINE(handle_flex_merge_flatten_list)
                        SCOREP_USER_REGION_BEGIN(handle_flex_merge_flatten_list, "flex_merge_flatten_noncontiguous", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                        std::vector<T> flat = tensor.as_flat_vector();
#ifdef USE_SCOREP
                        SCOREP_USER_REGION_END(handle_flex_merge_flatten_list)
                        SCOREP_USER_REGION_DEFINE(handle_flex_merge_copy_list2)
                        SCOREP_USER_REGION_BEGIN(handle_flex_merge_copy_list2, "flex_merge_copy", SCOREP_USER_REGION_TYPE_COMMON)
#endif
                        std::copy(flat.begin() + b * sn, flat.begin() + (b + 1) * sn,
                                  buffer + row_offset + col_offset);
#ifdef USE_SCOREP
                        SCOREP_USER_REGION_END(handle_flex_merge_copy_list2)
#endif
                    }
                    col_offset += sn;
                }
            }

            if (reuse_buffer) {
                merged_data.add_tensor(existing_data[t_idx]);
            } else {
                auto new_tensor = MLCouplingTensor<T>::wrap_flat(
                    buffer, new_dims, MLCouplingMemLayoutContiguous, MLCouplingOwnershipOwned);
                merged_data.add_tensor(std::move(new_tensor));
            }
#ifdef USE_SCOREP
            SCOREP_USER_REGION_END(handle_merge_list)
#endif
        }
        return merged_data;
    }


    template <typename T>
    bool can_stack(const std::vector<MLCouplingData<T>> &data_list) const
    {
        if (data_list.empty()) return false;
        size_t num_tensors = data_list[0].size();
        for (const auto& data : data_list) {
            if (data.size() != num_tensors) return false;
        }
        for (size_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
            auto first_dim = data_list[0][t_idx].dimensions();
            if (first_dim.empty()) return false; // Cannot stack scalars
            for (size_t i = 1; i < data_list.size(); ++i) {
                if (data_list[i][t_idx].dimensions() != first_dim) return false;
            }
        }
        return true;
    }

    template <typename T>
    void merge_data(const std::vector<MLCouplingData<T>> &data_list, MLCouplingData<T> &existing_data)
    {
        if (data_list.empty()) return;

        // Optimization: If there's exactly 1 input, and it's already contiguous, just pass it through by reference.
        if (data_list.size() == 1 && merge_strategy != MLCouplingMergeStrategy::Stack) {
            bool all_contiguous = true;
            for (size_t i = 0; i < data_list[0].size(); ++i) {
                if (!data_list[0][i].is_contiguous()) {
                    all_contiguous = false;
                    break;
                }
            }
            if (all_contiguous) {
                existing_data = data_list[0];
                return;
            }
        }

        if (merge_strategy == MLCouplingMergeStrategy::None) {
            existing_data = MLCouplingData<T>();
            for (const auto& data : data_list) {
                for (size_t i = 0; i < data.size(); ++i) {
                    existing_data.add_tensor(data[i]);
                }
            }
            return;
        }
        if (merge_strategy == MLCouplingMergeStrategy::Stack || 
            (merge_strategy == MLCouplingMergeStrategy::Auto && data_list.size() > 1 && can_stack(data_list))) {
            existing_data = stack_data(data_list, existing_data);
        } else {
            existing_data = list_data(data_list, existing_data);
        }
    }

    template <typename T>
    void merge_data(const std::vector<std::string> &keys, const std::map<std::string, MLCouplingData<T>> &data_map, MLCouplingData<T> &existing_data)
    {
        std::vector<MLCouplingData<T>> ordered_list;
        for (const auto &key : keys) {
            auto it = data_map.find(key);
            if (it != data_map.end()) {
                ordered_list.push_back(it->second);
            } else {
                throw std::runtime_error("flex_keyed fallback: key '" + key + "' not found.");
            }
        }
        if (merge_strategy == MLCouplingMergeStrategy::None) {
            existing_data = MLCouplingData<T>();
            for (const auto& data : ordered_list) {
                for (size_t i = 0; i < data.size(); ++i) {
                    existing_data.add_tensor(data[i]);
                }
            }
            return;
        }
        if (merge_strategy == MLCouplingMergeStrategy::Stack || 
            (merge_strategy == MLCouplingMergeStrategy::Auto && can_stack(ordered_list))) {
            existing_data = stack_data(ordered_list, existing_data);
        } else {
            existing_data = list_data(ordered_list, existing_data);
        }
    }

    std::vector<MLCouplingData<LibraryInput>> staged_inputs;
    std::vector<MLCouplingData<LibraryOutput>> staged_targets;
    std::vector<MLCouplingData<LibraryOutput> *> staged_outputs;
    std::map<std::string, MLCouplingData<LibraryInput>> keyed_inputs;
    std::map<std::string, MLCouplingData<LibraryOutput>> keyed_targets;
    std::map<std::string, MLCouplingData<LibraryOutput> *> keyed_outputs;

    MLCouplingData<LibraryInput> last_merged_input;
    MLCouplingData<LibraryOutput> last_merged_target;
};
