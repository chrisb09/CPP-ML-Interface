#pragma once

#include <memory>
#include <span>
#include <vector>

#include "provider/ml_coupling_provider.hpp"
#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior.hpp"


template <typename In, typename Out>
class MLCoupling
{
    // I'd see this class as essentially the API that the user
    // would interact with.
    public:
        void init(
                  MLCouplingDataProcessor<In, Out>* data_processor,
                  MLCouplingProvider<In, Out>* provider,
                  MLCouplingApplication<In, Out>* application,
                  MLCouplingBehavior<In, Out>* behavior = nullptr)
        {
            this->data_processor.reset(data_processor);
            this->provider.reset(provider);
            this->provider->init();
            this->application.reset(application);
            if (behavior == nullptr) {
                // Use default behavior if none provided
                this->behavior.reset(new MLCouplingBehaviorDefault());
            } else {
                this->behavior.reset(behavior);
            }
        }

        // Run the ML coupling application logic
        virtual void infer_ml_model() = 0;

        // Finalize and clean up resources
        virtual void finalize() = 0;

private:
    std::unique_ptr<MLCouplingProvider<In, Out>> provider;
    std::unique_ptr<MLCouplingApplication<In, Out>> application;
    std::unique_ptr<MLCouplingBehavior<In, Out>> behavior;
};