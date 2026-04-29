#pragma once

#include "ml_coupling_provider_flexible.hpp"

// @registry_name: Phydll
// @registry_aliases: phydll, PhyDLL
template <typename In, typename Out>
class MLCouplingProviderPhydll : public MLCouplingProviderFlexible<In, Out> {

    public:

        void send_data(MLCouplingData<In> input_data_after_preprocessing) override
        {
            // TODO
        }

        void inference(MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out>& output_data_before_postprocessing) override
        {
            // TODO
        }

};
