#pragma once

class MLCouplingApplicationProperties {
    public:
        virtual ~MLCouplingApplicationProperties() = default;

        // The following methods inform the behavior manager about capabilities

        // Indicate whether the application supports coupling without inference
        virtual bool supports_coupling_without_inference() const = 0;

        // Indicate whether the application supports time step increase
        virtual bool supports_time_step_increase() const = 0;
};