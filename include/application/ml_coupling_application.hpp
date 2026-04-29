#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "../data/ml_coupling_data.hpp"
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

        // The full constructor
        MLCouplingApplication(
            std::vector<In*> input_data,
            std::vector<std::vector<int>> input_data_dimensions,
            std::vector<Out*> output_data,
            std::vector<std::vector<int>> output_data_dimensions,
            MLCouplingNormalization<In, Out>* normalization) {
            if (input_data.size() != input_data_dimensions.size()) {
                throw std::invalid_argument("MLCouplingApplication(): input_data and input_data_dimensions sizes must match");
            }
            if (output_data.size() != output_data_dimensions.size()) {
                throw std::invalid_argument("MLCouplingApplication(): output_data and output_data_dimensions sizes must match");
            }

            std::vector<MLCouplingTensor<In>> input_tensors;
            input_tensors.reserve(input_data.size());
            for (size_t i = 0; i < input_data.size(); ++i) {
                input_tensors.push_back(MLCouplingTensor<In>::wrap_flat(input_data[i], input_data_dimensions[i]));
            }

            std::vector<MLCouplingTensor<Out>> output_tensors;
            output_tensors.reserve(output_data.size());
            for (size_t i = 0; i < output_data.size(); ++i) {
                output_tensors.push_back(MLCouplingTensor<Out>::wrap_flat(output_data[i], output_data_dimensions[i]));
            }

            this->input_data = MLCouplingData<In>(std::move(input_tensors));
            this->output_data = MLCouplingData<Out>(std::move(output_tensors));
            this->input_data_after_preprocessing = this->input_data;
            this->output_data_before_postprocessing = this->output_data;
            this->normalization.reset(normalization);
        }

        /*
        * This is a shorthand constructor, it assumes that the preprocssing and postprocessing are either in-situ or not needed at all, so it uses the input and output data directly as the preprocessed and postprocessed data.
        */
        MLCouplingApplication(
            MLCouplingData<In> input_data,
            MLCouplingData<Out> output_data,
            MLCouplingNormalization<In, Out>* normalization)
            : MLCouplingApplication(std::move(input_data),
                                    MLCouplingData<In>(),
                                    MLCouplingData<Out>(),
                                    std::move(output_data),
                                    normalization) {}

        MLCouplingApplication(
            MLCouplingData<In> input_data,
            MLCouplingData<In> input_data_after_preprocessing,
            MLCouplingData<Out> output_data_before_postprocessing,
            MLCouplingData<Out> output_data,
            MLCouplingNormalization<In, Out>* normalization) {
            this->input_data = std::move(input_data);
            this->output_data = std::move(output_data);
            if (input_data_after_preprocessing.empty()) {
                this->input_data_after_preprocessing = this->input_data;
            } else {
                this->input_data_after_preprocessing = std::move(input_data_after_preprocessing);
            }
            if (output_data_before_postprocessing.empty()) {
                this->output_data_before_postprocessing = this->output_data;
            } else {
                this->output_data_before_postprocessing = std::move(output_data_before_postprocessing);
            }
            this->normalization.reset(normalization);
        }

        void step(bool perform_coupling, bool perform_inference) {
            if (!perform_coupling && !perform_inference) {
                return;
            }
            input_data_after_preprocessing = preprocess(input_data);
            if (perform_coupling) {
                coupling_step(input_data_after_preprocessing);
            }
            if (perform_inference) {
                output_data_before_postprocessing = ml_step(input_data_after_preprocessing);
                output_data = postprocess(output_data_before_postprocessing);
            }
        }

        virtual ~MLCouplingApplication() = default;

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
