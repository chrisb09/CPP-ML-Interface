#pragma once

#include "ml_coupling_provider.hpp"

// @registry_name: Aixelerate
// @registry_aliases: aixelerate, AIxelerate
template <typename In, typename Out>
class MLCouplingProviderAixelerate : public MLCouplingProvider<In, Out> {

    public:

        void send_data(MLCouplingData<In> input_data_after_preprocessing) override
        {
            // TODO
        }

        MLCouplingData<Out> inference(MLCouplingData<In> input_data_after_preprocessing) override
        {
            // TODO
            return MLCouplingData<Out>(nullptr, std::vector<std::vector<int>>{});
        }

};