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
        staged_outputs.push_back(data);
    }

    virtual void flex_ordered_inference()
    {
        if (staged_inputs.size() == 1 && staged_outputs.size() == 1)
        {
            static_inference(&(staged_inputs[0]), staged_outputs[0]);
            staged_inputs.clear();
            staged_outputs.clear();
        }
        else
        {
            throw std::runtime_error("flex_ordered_inference fallback only supports exactly 1 staged input and 1 staged output.");
        }
    }

    virtual std::map<std::string, double> flex_ordered_train(long long step_id)
    {
        (void)step_id;
        if (staged_inputs.size() == 1 && staged_targets.size() == 1)
        {
            auto res = static_train(&(staged_inputs[0]), &(staged_targets[0]));
            staged_inputs.clear();
            staged_targets.clear();
            return res;
        }
        throw std::runtime_error("flex_ordered_train fallback expects 1 staged input and 1 staged target.");
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
        keyed_outputs[key] = data;
    }

    virtual void flex_keyed_inference(const std::vector<std::string> &in_keys,
                                      const std::vector<std::string> &out_keys)
    {
        if (in_keys.size() == 1 && out_keys.size() == 1)
        {
            static_inference(&(keyed_inputs.at(in_keys[0])), keyed_outputs.at(out_keys[0]));
        }
        else
        {
            throw std::runtime_error("flex_keyed_inference fallback only supports exactly 1 input key and 1 output key.");
        }
    }

    virtual std::map<std::string, double> flex_keyed_train(long long step_id,
                                                           const std::vector<std::string> &in_keys,
                                                           const std::vector<std::string> &target_keys)
    {
        (void)step_id;
        if (in_keys.size() == 1 && target_keys.size() == 1)
        {
            return static_train(&(keyed_inputs.at(in_keys[0])), &(keyed_targets.at(target_keys[0])));
        }
        throw std::runtime_error("flex_keyed_train fallback expects 1 input key and 1 target key.");
    }

protected:
    std::vector<MLCouplingData<In>> staged_inputs;
    std::vector<MLCouplingData<Out>> staged_targets;
    std::vector<MLCouplingData<Out> *> staged_outputs;
    std::map<std::string, MLCouplingData<In>> keyed_inputs;
    std::map<std::string, MLCouplingData<Out>> keyed_targets;
    std::map<std::string, MLCouplingData<Out> *> keyed_outputs;
};
