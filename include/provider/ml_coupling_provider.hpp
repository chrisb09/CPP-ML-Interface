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
    Stack
};

// @category: provider
template <typename In, typename Out>
class MLCouplingProvider
{
public:
    MLCouplingMergeStrategy merge_strategy = MLCouplingMergeStrategy::List;

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
            MLCouplingData<In> merged_in = merge_data(staged_inputs);
            static_inference(&merged_in, fallback_output);
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
            MLCouplingData<In> merged_in = merge_data(staged_inputs);
            MLCouplingData<Out> merged_targets = merge_data(staged_targets);
            auto res = static_train(&merged_in, &merged_targets);
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
            MLCouplingData<In> merged_in = merge_data(in_keys, keyed_inputs);
            static_inference(&merged_in, fallback_output);
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
            MLCouplingData<In> merged_in = merge_data(in_keys, keyed_inputs);
            MLCouplingData<Out> merged_targets = merge_data(target_keys, keyed_targets);
            return static_train(&merged_in, &merged_targets);
        }
        throw std::runtime_error("flex_keyed_train fallback expects at least 1 input key and 1 target key.");
    }


protected:
    template <typename T>
    MLCouplingData<T> stack_data(const std::vector<MLCouplingData<T>> &data_list)
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
            T* buffer = new T[total_numel];
            
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
            
            auto new_tensor = MLCouplingTensor<T>::wrap_flat(
                buffer, new_dims, MLCouplingMemLayoutContiguous, MLCouplingOwnershipOwned
            );
            
            stacked_data.add_tensor(std::move(new_tensor));
        }
        return stacked_data;
    }

    template <typename T>
    MLCouplingData<T> merge_data(const std::vector<MLCouplingData<T>> &data_list)
    {
        if (merge_strategy == MLCouplingMergeStrategy::Stack) {
            return stack_data(data_list);
        }

        MLCouplingData<T> merged;
        for (const auto &data : data_list)
        {
            for (const auto &tensor : data)
            {
                merged.add_tensor(tensor);
            }
        }
        return merged;
    }

    template <typename T>
    MLCouplingData<T> merge_data(const std::vector<std::string> &keys, const std::map<std::string, MLCouplingData<T>> &data_map)
    {
        if (merge_strategy == MLCouplingMergeStrategy::Stack) {
            std::vector<MLCouplingData<T>> ordered_list;
            for (const auto &key : keys) {
                auto it = data_map.find(key);
                if (it != data_map.end()) {
                    ordered_list.push_back(it->second);
                } else {
                    throw std::runtime_error("flex_keyed fallback: key '" + key + "' not found.");
                }
            }
            return stack_data(ordered_list);
        }

        MLCouplingData<T> merged;
        for (const auto &key : keys)
        {
            auto it = data_map.find(key);
            if (it != data_map.end())
            {
                for (const auto &tensor : it->second)
                {
                    merged.add_tensor(tensor);
                }
            }
            else
            {
                throw std::runtime_error("flex_keyed fallback: key '" + key + "' not found.");
            }
        }
        return merged;
    }

    std::vector<MLCouplingData<In>> staged_inputs;
    std::vector<MLCouplingData<Out>> staged_targets;
    std::vector<MLCouplingData<Out> *> staged_outputs;
    std::map<std::string, MLCouplingData<In>> keyed_inputs;
    std::map<std::string, MLCouplingData<Out>> keyed_targets;
    std::map<std::string, MLCouplingData<Out> *> keyed_outputs;
};
