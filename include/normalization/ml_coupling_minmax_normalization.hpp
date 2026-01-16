#pragma once

#include <type_traits>

#include "ml_coupling_normalization.hpp"

#include <limits>

// @registry_name: MinMax
// @registry_aliases: minmax, min-max, MinMaxNormalization
//template <typename In, typename Out>
template <typename In, typename Out,
          typename = std::enable_if_t<std::is_arithmetic_v<In> && std::is_arithmetic_v<Out>>>
class MLCouplingMinMaxNormalization : public MLCouplingNormalization<In, Out> {
    public:
        MLCouplingMinMaxNormalization(In input_min, In input_max, Out output_min, Out output_max)
            : input_min(input_min), input_max(input_max),
              output_min(output_min), output_max(output_max) {}



        void normalize_input(In* input_data, int input_data_size) override {
            // Min-Max normalization: y = (x - min) / (max - min)
            for (int i = 0; i < input_data_size; ++i) {
                input_data[i] = (input_data[i] - input_min) / (input_max - input_min);
            }
        }

        void denormalize_output(Out* output_data, int output_data_size) override {
            // Min-Max denormalization: x = y * (max - min) + min
            for (int i = 0; i < output_data_size; ++i) {
                output_data[i] = output_data[i] * (output_max - output_min) + output_min;
            }
        }

    private:
        In input_min;
        In input_max;
        Out output_min;
        Out output_max;
};