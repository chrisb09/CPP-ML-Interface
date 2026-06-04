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

// @category: provider
template <typename In, typename Out>
class MLCouplingProvider
{
public:
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
    MLCouplingData<T> merge_data(const std::vector<MLCouplingData<T>> &data_list)
    {
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
