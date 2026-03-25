#pragma once

#include <functional>
#include <stdexcept>

#include "ml_coupling_behavior.hpp"

inline bool allow_inference_at_all_steps(int step) {
    (void)step;
    return false;
}

// @registry_name: Periodic
// @registry_aliases: periodic
class MLCouplingBehaviorPeriodic : public MLCouplingBehavior {
    public:

        /**
         * @brief Constructs an MLCouplingBehaviorPeriodic object with specified inference and coupling parameters.
         *
         * @param inference_interval Interval (in steps) between ML inference operations.
         * @param coupled_steps_before_inference Number of coupled steps to perform before each inference.
         * @param coupled_steps_stride Stride (step size) for coupled steps between inferences.
         * @param step_increment_after_inference Step increment to apply after each inference operation.
         */
        MLCouplingBehaviorPeriodic(
            int inference_interval,
            int coupled_steps_before_inference,
            int coupled_steps_stride,
            int step_increment_after_inference,
            std::function<bool(int)> prohibit_inference = allow_inference_at_all_steps)
        : inference_interval(inference_interval), coupled_steps_before_inference(coupled_steps_before_inference), coupled_steps_stride(coupled_steps_stride), step_increment_after_inference(step_increment_after_inference), prohibit_inference(prohibit_inference) {
//: inference_interval(inference_interval), coupled_steps_before_inference(coupled_steps_before_inference), coupled_steps_stride(coupled_steps_stride), step_increment_after_inference(step_increment_after_inference) {
            // Validate input parameters
            if (inference_interval <= 0) {
                throw std::invalid_argument("Inference interval must be greater than 0.");
            }
            if (coupled_steps_before_inference < 0) {
                throw std::invalid_argument("Number of coupled steps before inference cannot be negative.");
            }
            if (coupled_steps_stride <= 0) {
                throw std::invalid_argument("Coupled steps stride must be greater than 0.");
            }
            if (step_increment_after_inference < 0) {
                throw std::invalid_argument("Step increment after inference cannot be negative.");
            }
        }

        bool should_perform_inference() override {
            // Determine if we are at an inference step
            step_count++;

            if (prohibit_inference(step_count)) {
                return false;
            }

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
            long long int steps_since_last_inference = step_count % inference_interval;
            long long int steps_until_next_inference = inference_interval - steps_since_last_inference;

            if (steps_until_next_inference < coupled_steps_before_inference * coupled_steps_stride &&
                steps_until_next_inference % coupled_steps_stride == 0) {
                return true;
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
        // Optional function to determine if inference should be prohibited at a given step
        // It's purpose is to allow us to replicate the desired behavior in which steps that would save the current state can't be simulated with the ML model, so inference should be prohibited at those steps, although this in practice only makes sense if step_increment_after_inference is >= 1, otherwise we'd still have a normal simulation step every step anyway
        std::function<bool(int)> prohibit_inference;

        // If we require 5 steps before inference, and have a stride of 24,
        // then when we call the inference at step i*N (where N is inference_interval),
        // we would send data from steps:
        // i*N - (5-0)*24, i*N - (5-1)*24, i*N - (5-2)*24, i*N - (5-3)*24, i*N - (5-4)*24
        // so we can only run inference once i*N >= 5*24, or more generally
        // i*inference_interval >= coupled_steps_before_inference * coupled_steps_stride
        // for some non-negative integer i
        
};