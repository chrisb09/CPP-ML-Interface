#pragma once

#include "ml_coupling_provider.hpp"

// @registry_name: Phydll
// @registry_aliases: phydll, PhyDLL
template <typename In, typename Out>
class MLCouplingProviderPhydll : public MLCouplingProvider<In, Out> {

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