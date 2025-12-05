#pragma once

#include "ml_coupling_normalization.hpp"

template <typename In, typename Out>
class MLCouplingMinMaxNormalization : public MLCouplingNormalization<In, Out> {
    public:
        MLCouplingMinMaxNormalization(const In& input_min, const In& input_max,
                                      const Out& output_min, const Out& output_max)
            : input_min_(input_min), input_max_(input_max),
              output_min_(output_min), output_max_(output_max) {}

        void normalize_input(In& input_data) override {
            // Min-Max normalization: y = (x - min) / (max - min)
            input_data = (input_data - input_min_) / (input_max_ - input_min_);
        }

        void denormalize_output(Out& output_data) override {
            // Min-Max denormalization: x = y * (max - min) + min
            output_data = output_data * (output_max_ - output_min_) + output_min_;
        }

    private:
        In input_min_;
        In input_max_;
        Out output_min_;
        Out output_max_;
};