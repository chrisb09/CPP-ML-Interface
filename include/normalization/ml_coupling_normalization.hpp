#pragma once

// @category: normalization
template <typename In, typename Out>
class MLCouplingNormalization {
    public:
    
        // Normalize the input data
        virtual void normalize_input(In* input_data, int input_data_size) = 0;

        // Denormalize the output data
        virtual void denormalize_output(Out* output_data, int output_data_size) = 0;
};