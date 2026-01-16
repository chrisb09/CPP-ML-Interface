#pragma once

#include "../data_processor/ml_coupling_data.hpp"

#include "ml_coupling_application.hpp"


// @registry_name: TurbulenceClosure
// @registry_aliases: turbulence-closure, turbulence_closure, turbulence
template <typename In, typename Out>
class MLCouplingApplicationTurbulenceClosure : public MLCouplingApplication<In, Out> {
    public:
        virtual ~MLCouplingApplicationTurbulenceClosure() = default;

        MLCouplingData<In> preprocess(MLCouplingData<In> input_data) override {
            // TODO: Implement turbulence closure specific preprocessing here
            data_processor->normalize_input();
            uniform_filtering();
            downsampling();
            return input_data;
        }

        MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) override {
            // TODO: Implement turbulence closure specific postprocessing here
            data_processor->denormalize_output();
            compute_tau_ij();
            return output_data_before_postprocessing;
        }

        // Pre- and post-processing are already called in MLCoupling's ml_step()
        // Just an example so far
        MLCouplingData<Out> ml_step(MLCouplingData<In> input_data_after_preprocessing) override {
            // Implement turbulence closure specific inference logic here

        }
        
    private:
        void uniform_filtering() {
            // Placeholder for uniform filtering implementation
        }

        void downsampling() {
            // Placeholder for downsampling implementation
        }

        void compute_tau_ij() {
            // Placeholder for computing subgrid-scale stress tensor tau_ij
        }

};