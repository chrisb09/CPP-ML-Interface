#pragma once

#include "ml_coupling_provider.hpp"

template <typename In, typename Out>
class MLCouplingProviderSmartsim : MLCouplingProvider
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

    void inference() override
    {
        // TODO
    }

    void finalize() override
    {
        // TODO:
    }

private:
};