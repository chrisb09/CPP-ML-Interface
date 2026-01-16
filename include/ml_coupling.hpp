#pragma once

#include <memory>
#include <span>
#include <vector>

#include "provider/ml_coupling_provider.hpp"
#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior.hpp"
#include "behavior/ml_coupling_behavior_default.hpp"


template <typename In, typename Out>
class MLCoupling
{
    // I'd see this class as essentially the API that the user
    // would interact with.
    public:
        MLCoupling<In, Out>(
                  MLCouplingProvider<In, Out>* provider,
                  MLCouplingApplication<In, Out>* application,
                  MLCouplingBehavior* behavior = nullptr)
        {
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

        void ml_step() {
            if (application && behavior && behavior->should_perform_inference()) {
                application->data_processor->input_data_after_preprocessing = application->preprocess(application->data_processor->input_data);
                application->ml_step();
                application->data_processor->output_data_before_postprocessing = application->postprocess(application->data_processor->output_data);
            }
        }

        ~MLCoupling() {
            if (provider) {
                provider->finalize();
            }
            if (application) {
                application->finalize();
            }
        }

private:
    std::unique_ptr<MLCouplingProvider<In, Out>> provider;
    std::unique_ptr<MLCouplingApplication<In, Out>> application;
    std::unique_ptr<MLCouplingBehavior> behavior;
};