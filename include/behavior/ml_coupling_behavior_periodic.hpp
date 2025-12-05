#pragma once

#include "ml_coupling_behavior.hpp"
#include "ml_coupling_application_properties.hpp"

class MLCouplingBehaviorPeriodic : public MLCouplingBehavior {
    public:
        MLCouplingBehaviorPeriodic(MLCouplingApplicationProperties* properties, int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference) : MLCouplingBehavior(properties)
        {
            this->inference_interval = inference_interval;
            this->coupled_steps_before_inference = coupled_steps_before_inference;
            this->coupled_steps_stride = coupled_steps_stride;
            this->step_increment_after_inference = step_increment_after_inference;
        }

        bool should_perform_inference() override {
            // Determine if we are at an inference step
            step_count++;

            // Check if we have enough prior coupled steps to perform inference
            if (step_count % inference_interval == 0) {
                // Check if we have enough prior coupled steps
                if (step_count >= coupled_steps_before_inference * coupled_steps_stride) {
                    return true;
                }
            }
            return false;
        }

        int time_step_delta() override {
            return step_increment_after_inference;
        }

        //TODO: check logic
        bool should_send_data() override {
            // Determine if we should send data in the current step
            // We send data for the required number of coupled steps before inference
            if (properties->supports_coupling_without_inference()) {
                long long int steps_since_last_inference = step_count % inference_interval;
                long long int steps_until_next_inference = inference_interval - steps_since_last_inference;

                if (steps_until_next_inference < coupled_steps_before_inference * coupled_steps_stride &&
                    steps_until_next_inference % coupled_steps_stride == 0) {
                    return true;
                }
            }
            return false;
        }

    private:
        // Some simulations may run for a very long time so using
        // some kind of larger integer type is safer
        // Also, by keeping track of step count based on calls to 
        // should_perform_inference, the user does not need to pass in
        // the current step
        // This should reflect the unmodified simulation step count
        // so exclude any time step increases due to the ML coupling
        long long int step_count = 0;
        // Every N (real) steps, perform inference
        int inference_interval;
        // How many steps should be coupled (their data is sent) before the inference step
        int coupled_steps_before_inference;
        // Stride between coupled steps
        int coupled_steps_stride;
        // How many steps to increment after inference
        int step_increment_after_inference;

        // If we require 5 steps before inference, and have a stride of 24,
        // then when we call the inference at step i*N (where N is inference_interval),
        // we would send data from steps:
        // i*N - (5-0)*24, i*N - (5-1)*24, i*N - (5-2)*24, i*N - (5-3)*24, i*N - (5-4)*24
        // so we can only run inference once i*N >= 5*24, or more generally
        // i*inference_interval >= coupled_steps_before_inference * coupled_steps_stride
        // for some non-negative integer i
        
};