#pragma once

#include "../ml_coupling_data.hpp"
#include <ostream>

// @category: normalization
template <typename In, typename Out>
class MLCouplingNormalization {
    public:
    
        // Normalize the input data
        virtual void normalize_input(MLCouplingData<In> input_data) = 0;

        // Denormalize the output data
        virtual void denormalize_output(MLCouplingData<Out> output_data) = 0;

        // Virtual destructor for polymorphic classes
        virtual ~MLCouplingNormalization() = default;

        // Print representation
        virtual void print(std::ostream& os) const = 0;
};