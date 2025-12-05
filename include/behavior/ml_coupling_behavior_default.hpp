#pragma once

#include "ml_coupling_behavior.hpp"
#include "ml_coupling_application_properties.hpp"

class MLCouplingBehaviorDefault : public MLCouplingBehavior {
    public:
        MLCouplingBehaviorDefault(MLCouplingApplicationProperties* properties) 
            : MLCouplingBehavior(properties) {
            
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
            // Default behavior: always send data if supported
            return properties->supports_coupling_without_inference();
        }
};