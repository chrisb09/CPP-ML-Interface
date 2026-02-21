#pragma once

#include <memory>

#include "../ml_coupling_data.hpp"
#include "../provider/ml_coupling_provider.hpp"
#include "../normalization/ml_coupling_normalization.hpp"

// @category: application
template <typename In, typename Out>
class MLCouplingApplication {
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

        MLCouplingApplication(
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

        MLCouplingApplication(
            MLCouplingData<In> input_data,
            MLCouplingData<Out> output_data,
            MLCouplingNormalization<In, Out>* normalization) {
            this->input_data = input_data;
            this->output_data = output_data;
            this->normalization.reset(normalization);
        }

        void step(bool perform_coupling, bool perform_inference) {
            if (!perform_coupling && !perform_inference) {
                return;
            }
            input_data_after_preprocessing = preprocess(input_data);
            if (perform_coupling) {
                coupling_step();
            }
            if (perform_inference) {
                output_data = ml_step(output_data_before_postprocessing);
            }
        }

    protected:
        virtual MLCouplingData<In> preprocess(MLCouplingData<In> input_data) { return input_data; }

        // Run the coupling step logic (sending data)
        // If you only need data from one step, you can implement this as a no-op
        // and just implement the ml_step()
        virtual void coupling_step(MLCouplingData<In> input_data_after_preprocessing) = 0;

        // Run the ML application logic (inference, maybe later training as well)
        virtual MLCouplingData<Out> ml_step(MLCouplingData<In> input_data_after_preprocessing) = 0;

        virtual MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) { return output_data_before_postprocessing; };


        std::unique_ptr<MLCouplingNormalization<In, Out>> normalization;
};