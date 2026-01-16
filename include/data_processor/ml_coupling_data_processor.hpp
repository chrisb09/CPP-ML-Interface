#pragma once

#include <memory>
#include <vector>

#include "../normalization/ml_coupling_normalization.hpp"

// data pre- and post-processing to get

// @category: data_processor
template <typename In, typename Out>
class MLCouplingDataProcessor {

    public:
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

    protected:
        std::vector<In*> input_data;
        std::vector<std::vector<int>> input_data_dimensions;
        /*
        * This should allow us to have an for example
        * a single memory space in the input data 
        * which has the dimension of 16x16x256
        * This is specified by a single entry in the 
        * dimensions variable, which itself is a 
        * vector with 3 entries: 16,16,256
        * This implicitly defined the memory segment 
        * as well
        */
       std::vector<Out*> output_data;
       std::vector<std::vector<int>> output_data_dimensions;
       /*
       * It works analogous for the output data
       */



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