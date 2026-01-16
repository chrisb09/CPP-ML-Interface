#pragma once

#include "ml_coupling_provider.hpp"
#include <string>

// @registry_name: Smartsim
// @registry_aliases: smartsim, SmartSim
template <typename In, typename Out>
class MLCouplingProviderSmartsim : public MLCouplingProvider<In, Out>
{

public:
    MLCouplingProviderSmartsim(std::string host = "localhost",
                               int port = 6379,
                               int nodes = 1,
                               int tasks_per_node = 1,
                               int cpus_per_task = 1,
                               int gpus_per_task = 0)
    {
        // TODO: So far the parameters are just dummy parameters.
    }

    void init() override
    {
        // TODO:
    }

    void send_data(MLCouplingData<In> input_data_after_preprocessing) override
    {
        // TODO
    }

    MLCouplingData<Out> inference(MLCouplingData<In> input_data_after_preprocessing) override
    {
        // TODO
        return MLCouplingData<Out>(nullptr, std::vector<std::vector<int>>{});
    }

    void finalize() override
    {
        // TODO:
    }

private:
};