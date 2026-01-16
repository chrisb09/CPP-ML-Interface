#pragma once

#include <memory>
#include <vector>

#include "ml_coupling_data.hpp"

#include "../normalization/ml_coupling_normalization.hpp"

// data pre- and post-processing to get

// @category: data_processor
template <typename In, typename Out>
class MLCouplingDataProcessor {

    public:
        MLCouplingData<In> input_data;
        MLCouplingData<In> input_data_after_preprocessing;
        MLCouplingData<Out> output_data_before_postprocessing;
        MLCouplingData<Out> output_data;
        // Right now we copy the vectors
        // Maybe there's a better alternative
        // But I'm not sure shennanigans are worth it
        // since the cost of copying this
        // should be minimal
        MLCouplingDataProcessor(
            std::vector<In*> input_data,
            std::vector<std::vector<int>> input_data_dimensions,
            std::vector<Out*> output_data,
            std::vector<std::vector<int>> output_data_dimensions,
            MLCouplingNormalization<In, Out>* normalization) {
            this->input_data = input_data;
            this->input_data_dimensions = input_data_dimensions;
            this->output_data = output_data;
            this->output_data_dimensions = output_data_dimensions;
            this->normalization.reset(normalization);
        }

        MLCouplingDataProcessor(
            MLCouplingData<In> input_data,
            MLCouplingData<Out> output_data,
            MLCouplingNormalization<In, Out>* normalization) {
            this->input_data = input_data;
            this->output_data = output_data;
            this->normalization.reset(normalization);
        }

        protected:

        // We assume normalization and denormalization to be in-situ
        void normalize_input(In& input_data) {
            // Check if normalization is set
            if (normalization) {
                normalization->normalize_input(input_data);
            }
        }

        void denormalize_output(Out& output_data) {
            // Check if normalization is set
            if (normalization) {
                normalization->denormalize_output(output_data);
            }
        }

    private:
        std::unique_ptr<MLCouplingNormalization<In, Out>> normalization;

};