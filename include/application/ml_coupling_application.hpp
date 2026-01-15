#pragma once

#include "ml_coupling_provider.hpp"
#include "../data_processor/ml_coupling_data_processor.hpp"

template <typename In, typename Out>
class MLCouplingApplication {
    public:
        MLCouplingApplication(std::unique_ptr<MLCouplingDataProcessor<In, Out>> data_processor)
            : data_processor(std::move(data_processor)) {}
        virtual ~MLCouplingApplication() = default;

        // Initialize the ML coupling application
        virtual void init() = 0;

        // Finalize and clean up resources
        virtual void finalize() = 0;

        void ml_step() {
            normalize_input();
            infer_ml_model(); // Should call provider->inference() internally   
            denormalize_output();
        }

    protected:
        virtual void preprocess() {} = 0;

        // Run the ML coupling application logic
        virtual void infer_ml_model() = 0;

        virtual void postprocess() {} = 0;

    private:
        std::unique_ptr<MLCouplingDataProcessor<In, Out>> data_processor;
};