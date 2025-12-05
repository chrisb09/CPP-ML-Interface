#pragma once

#include "ml_coupling_provider.hpp"
#include "ml_coupling_normalization.hpp"
#include "ml_coupling_application_properties.hpp"

template <typename In, typename Out>
class MLCouplingApplication : public MLCouplingApplicationProperties {
    public:
        MLCouplingApplication(std::shared_ptr<MLCouplingNormalization<In, Out>> normalization)
            : normalization(normalization) {}
        virtual ~MLCouplingApplication() = default;

        // Initialize the ML coupling application
        virtual void init() = 0;

        // Run the ML coupling application logic
        virtual void infer_ml_model() = 0;

        // Finalize and clean up resources
        virtual void finalize() = 0;

        void ml_step() {
            normalize_input();
            infer_ml_model(); // Should call provider->inference() internally   
            denormalize_output();
        }

    protected:
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
        std::shared_ptr<MLCouplingNormalization<In, Out>> normalization;
};