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

        // The downside of this approach is that if later on bigger or smaller min/max values appear,
        // the normalization will be off. However, this is a dummy implementation for now.
        MLCouplingMinMaxNormalization(In* input_data, int input_data_size,
                                     Out* output_data, int output_data_size) {
            // Compute min and max for input data
            input_min = std::numeric_limits<In>::max();
            input_max = std::numeric_limits<In>::lowest();
            for (int i = 0; i < input_data_size; ++i) {
                if (input_data[i] < input_min) {
                    input_min = input_data[i];
                }
                if (input_data[i] > input_max) {
                    input_max = input_data[i];
                }
            }

            // Compute min and max for output data
            output_min = std::numeric_limits<Out>::max();
            output_max = std::numeric_limits<Out>::lowest();
            for (int i = 0; i < output_data_size; ++i) {
                if (output_data[i] < output_min) {
                    output_min = output_data[i];
                }
                if (output_data[i] > output_max) {
                    output_max = output_data[i];
                }
            }
        }

        MLCouplingMinMaxNormalization(MLCouplingData<In> input_data,
                                     MLCouplingData<Out> output_data) {
            // use the previous constructor
            *this = MLCouplingMinMaxNormalization(input_data.data.data(),
                                                  input_data.data.size(),
                                                  output_data.data.data(),
                                                  output_data.data.size());
        }

        void normalize_input(In* input_data, int input_data_size) {
            // Min-Max normalization: y = (x - min) / (max - min)
            for (int i = 0; i < input_data_size; ++i) {
                input_data[i] = (input_data[i] - input_min) / (input_max - input_min);
            }
        }

        void normalize_input(MLCouplingData<In> input_data) override {
            normalize_input(input_data.data.data(), input_data.data.size());
        }

        void denormalize_output(Out* output_data, int output_data_size) {
            // Min-Max denormalization: x = y * (max - min) + min
            for (int i = 0; i < output_data_size; ++i) {
                output_data[i] = output_data[i] * (output_max - output_min) + output_min;
            }
        }

        void denormalize_output(MLCouplingData<Out> output_data) override {
            denormalize_output(output_data.data.data(), output_data.data.size());
        }

    private:
        In input_min;
        In input_max;
        Out output_min;
        Out output_max;
};