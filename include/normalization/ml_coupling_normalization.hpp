#pragma once

template <typename In, typename Out>
class MLCouplingNormalization {
    public:
        virtual ~MLCouplingNormalization() = default;

        // Normalize the input data
        virtual void normalize_input(In& input_data) = 0;

        // Denormalize the output data
        virtual void denormalize_output(Out& output_data) = 0;
};