#pragma once

#include "ml_coupling_behavior.hpp"

// @registry_name: Default
// @registry_aliases: default
class MLCouplingBehaviorDefault : public MLCouplingBehavior {
    public:
        MLCouplingBehaviorDefault() {
            
        }

        bool should_perform_inference() override {
            // Default behavior: always perform inference
            return true;
        }

        int time_step_delta() override {
            // Default behavior: do not increase time step
            return 0;
        }

        bool should_send_data() override {
            // Default behavior: always send data 
            return true;
        }
};