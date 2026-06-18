#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include "../data/ml_coupling_data.hpp"

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
    Auto
};

// @category: provider
template <typename In, typename Out>
class MLCouplingProvider
{
public:
    MLCouplingMergeStrategy merge_strategy = MLCouplingMergeStrategy::Auto;

    void set_merge_strategy(MLCouplingMergeStrategy strategy) {
        merge_strategy = strategy;
    }

    int rank = 0;

    MLCouplingProvider()
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

    MLCouplingProvider(int rank)
    {
        this->rank = rank;
    }

    virtual ~MLCouplingProvider() = default;

    void set_rank(int rank)
    {
        this->rank = rank;
    }

    // --- Tier 0: Static (Mandatory) ---
    // Perform inference with the ML model and get the output data
    virtual void static_inference(MLCouplingData<In> *input,
                                  MLCouplingData<Out> *output) = 0;

    virtual std::map<std::string, double> static_train(MLCouplingData<In> *input,
                                                       MLCouplingData<Out> *target)
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
    virtual void flex_ordered_set(MLCouplingData<In> data)
    {
        staged_inputs.push_back(std::move(data));
    }

    virtual void flex_ordered_set_target(MLCouplingData<Out> data)
    {
        staged_targets.push_back(std::move(data));
    }

    virtual void flex_ordered_get(MLCouplingData<Out> *data)
    {
        // Fallback is a no-op because flex_ordered_inference populates the fallback_output directly.
        (void)data;
    }

    virtual void flex_ordered_inference(MLCouplingData<Out> *fallback_output = nullptr)
    {
        if (staged_inputs.size() > 0 && fallback_output != nullptr)
        {
            merge_data(staged_inputs, last_merged_input);
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
    virtual void flex_keyed_set(const std::string &key, MLCouplingData<In> data)
    {
        keyed_inputs[key] = std::move(data);
    }

    virtual void flex_keyed_set_target(const std::string &key, MLCouplingData<Out> data)
    {
        keyed_targets[key] = std::move(data);
    }

    virtual void flex_keyed_get(const std::string &key, MLCouplingData<Out> *data)
    {
        // Fallback is a no-op because flex_keyed_inference populates the fallback_output directly.
        (void)key;
        (void)data;
    }

    virtual void flex_keyed_inference(const std::vector<std::string> &in_keys,
                                      const std::vector<std::string> &out_keys,
                                      MLCouplingData<Out> *fallback_output = nullptr)
    {
        (void)out_keys; // The fallback maps all expected outputs to the static buffer
        if (in_keys.size() > 0 && fallback_output != nullptr)
        {
            merge_data(in_keys, keyed_inputs, last_merged_input);
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
    template <typename T>
    MLCouplingData<T> stack_data(const std::vector<MLCouplingData<T>> &data_list, MLCouplingData<T> &existing_data)
    {
        if (data_list.empty()) return MLCouplingData<T>();
        
        size_t num_tensors = data_list[0].size();
        size_t m = data_list.size();
        
        MLCouplingData<T> stacked_data;
        
        for (size_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
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

            if (existing_data.size() > t_idx && existing_data[t_idx].dimensions() == new_dims && existing_data[t_idx].is_contiguous()) {
                buffer = static_cast<T*>(existing_data[t_idx].root());
                reuse_buffer = true;
            } else {
                buffer = new T[total_numel];
            }
            
            for (size_t i = 0; i < m; ++i) {
                const auto& tensor = data_list[i][t_idx];
                if (tensor.is_contiguous()) {
                    const T* src = static_cast<const T*>(tensor.root());
                    for (int b = 0; b < B; ++b) {
                        std::copy(
                            src + b * slice_numel,
                            src + (b + 1) * slice_numel,
                            buffer + (b * m + i) * slice_numel
                        );
                    }
                } else {
                    std::vector<T> flat = tensor.as_flat_vector();
                    for (int b = 0; b < B; ++b) {
                        std::copy(
                            flat.begin() + b * slice_numel,
                            flat.begin() + (b + 1) * slice_numel,
                            buffer + (b * m + i) * slice_numel
                        );
                    }
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
        }
        return stacked_data;
    }

    template <typename T>
    MLCouplingData<T> list_data(const std::vector<MLCouplingData<T>> &data_list, MLCouplingData<T> &existing_data)
    {
        if (data_list.empty()) return MLCouplingData<T>();

        // If there's only one item in the list, just deep-copy it to preserve its shape.
        if (data_list.size() == 1) {
            return data_list[0].deep_copy();
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
            if (existing_data.size() > t_idx &&
                existing_data[t_idx].dimensions() == new_dims &&
                existing_data[t_idx].is_contiguous()) {
                buffer = static_cast<T*>(existing_data[t_idx].root());
                reuse_buffer = true;
            } else {
                buffer = new T[total_numel];
            }

            // For each batch item b, copy the per-slice data from each input i.
            for (int b = 0; b < B; ++b) {
                size_t row_offset = static_cast<size_t>(b) * total_slice_numel;
                size_t col_offset = 0;
                for (size_t i = 0; i < m; ++i) {
                    const auto& tensor = data_list[i][t_idx];
                    size_t sn = slice_numels[i];
                    if (tensor.is_contiguous()) {
                        const T* src = static_cast<const T*>(tensor.root());
                        std::copy(src + b * sn, src + (b + 1) * sn, buffer + row_offset + col_offset);
                    } else {
                        std::vector<T> flat = tensor.as_flat_vector();
                        std::copy(flat.begin() + b * sn, flat.begin() + (b + 1) * sn,
                                  buffer + row_offset + col_offset);
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
        if (merge_strategy == MLCouplingMergeStrategy::Stack || 
            (merge_strategy == MLCouplingMergeStrategy::Auto && can_stack(data_list))) {
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
        if (merge_strategy == MLCouplingMergeStrategy::Stack || 
            (merge_strategy == MLCouplingMergeStrategy::Auto && can_stack(ordered_list))) {
            existing_data = stack_data(ordered_list, existing_data);
        } else {
            existing_data = list_data(ordered_list, existing_data);
        }
    }

    std::vector<MLCouplingData<In>> staged_inputs;
    std::vector<MLCouplingData<Out>> staged_targets;
    std::vector<MLCouplingData<Out> *> staged_outputs;
    std::map<std::string, MLCouplingData<In>> keyed_inputs;
    std::map<std::string, MLCouplingData<Out>> keyed_targets;
    std::map<std::string, MLCouplingData<Out> *> keyed_outputs;

    MLCouplingData<In> last_merged_input;
    MLCouplingData<Out> last_merged_target;
};
