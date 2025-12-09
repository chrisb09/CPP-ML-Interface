#pragma once

#include <memory>
#include <span>
#include <vector>

#include "ml_coupling_provider.hpp"
#include "ml_coupling_application.hpp"
#include "ml_coupling_behavior.hpp"

template <typename In, typename Out>
class MLCoupling
{
    // I'd see this class as essentially the API that the user
    // would interact with.
    public:
        void init(In** input_data,
                  int input_data_count, // for the mai-a u/v/w this would be 3
                  std::vector<int> input_data_sizes, // sizes of each input data array
                  Out** output_data,
                  size_t output_data_count,
                  std::vector<int> output_data_sizes,
                
                  MLCouplingProvider<In, Out>* provider,
                  MLCouplingApplication<In, Out>* application,
                  MLCouplingBehavior<In, Out>* behavior = nullptr)
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

        // Run the ML coupling application logic
        virtual void infer_ml_model() = 0;

        // Finalize and clean up resources
        virtual void finalize() = 0;

protected:
    // span allows us to view external data without copying
    // and not assuming ownership like vector would
    std::vector<std::span<In>> input_data;
    std::vector<std::span<Out>> output_data;

private:
    std::unique_ptr<MLCouplingProvider<In, Out>> provider;
    std::unique_ptr<MLCouplingApplication<In, Out>> application;
    std::unique_ptr<MLCouplingBehavior<In, Out>> behavior;
};