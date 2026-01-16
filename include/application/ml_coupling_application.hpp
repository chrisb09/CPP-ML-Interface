#pragma once

#include <memory>

#include "../data_processor/ml_coupling_data.hpp"
#include "../provider/ml_coupling_provider.hpp"
#include "../data_processor/ml_coupling_data_processor.hpp"

// @category: application
template <typename In, typename Out>
class MLCouplingApplication {
    public:
        MLCouplingApplication(std::unique_ptr<MLCouplingDataProcessor<In, Out>> data_processor)
            : data_processor(std::move(data_processor)) {}

        void step(bool perform_coupling, bool perform_inference) {
            if (!perform_coupling && !perform_inference) {
                return;
            }
            if (data_processor){
                data_processor->input_data_after_preprocessing = preprocess(data_processor->input_data);
            }
            if (perform_coupling) {
                coupling_step();
            }
            if (perform_inference) {
                data_processor->output_data = ml_step(data_processor->input_data_before_postprocessing);
            }
        }

    protected:
        virtual MLCouplingData<In> preprocess(MLCouplingData<In> input_data) {}

        // Run the coupling step logic (sending data)
        // If you only need data from one step, you can implement this as a no-op
        // and just implement the ml_step()
        virtual void coupling_step(MLCouplingData<In> input_data_after_preprocessing) = 0;

        // Run the ML application logic (inference, maybe later training as well)
        virtual MLCouplingData<Out> ml_step(MLCouplingData<In> input_data_after_preprocessing) = 0;

        virtual MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) {}

    protected:
        std::unique_ptr<MLCouplingDataProcessor<In, Out>> data_processor;
};