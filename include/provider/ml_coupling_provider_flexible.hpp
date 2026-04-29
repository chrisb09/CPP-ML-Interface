#pragma once

#include "ml_coupling_provider.hpp"

/*
* This class contains the virtual methods that allow us to control the coupling step more flexibly, for example by allowing us to send data to the ML model without necessarily performing inference, which could be useful for certain cases.
*/

template <typename In, typename Out>
class MLCouplingProviderFlexible : public MLCouplingProvider<In, Out> {

    protected:
        long long next_message_id = 0;
        long long last_message_inferred_id = -1;

    public:

        MLCouplingProviderFlexible() = default;

        // Essentially for the coupling step: send data to the ML model
        // We could also have a "fake" implementation of this for providers that don't need really support this by pooling the data locally and then sending it during inference...
        virtual void send_data(MLCouplingData<In> input_data_after_preprocessing) = 0;

};