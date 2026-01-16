#pragma once

#include "../data_processor/ml_coupling_data.hpp"

// @category: normalization
template <typename In, typename Out>
class MLCouplingNormalization {
    public:
    
        // Normalize the input data
        virtual void normalize_input(MLCouplingData<In> input_data) = 0;

        // Denormalize the output data
        virtual void denormalize_output(MLCouplingData<Out> output_data) = 0;
};