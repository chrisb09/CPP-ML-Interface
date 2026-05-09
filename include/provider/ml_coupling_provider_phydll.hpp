#pragma once

#include "ml_coupling_provider_flexible.hpp"

# ifdef WITH_PHYDLL
extern "C" {
#include "phydll.h"
}
# endif

// @registry_name: Phydll
// @registry_aliases: phydll, PhyDLL
template <typename In, typename Out>
class MLCouplingProviderPhydll : public MLCouplingProviderFlexible<In, Out> {

    public:

                MLCouplingProviderPhydll(std::string model_file,
                                                                 std::string backend = "TORCH",
                                                                 std::string device = "GPU",
                                                                 MLCouplingData<In>* input_after_preprocessing = nullptr,
                                                                 MLCouplingData<Out>* output_before_postprocessing = nullptr)
                        : input_after_preprocessing(input_after_preprocessing),
                            output_before_postprocessing(output_before_postprocessing) {
            # ifndef WITH_PHYDLL
                guarantee(false, "PhyDLL provider is not enabled. Please make sure WITH_PHYDLL is defined and the necessary dependencies are installed.");
            # else
                char mode[] = "physical";
                phydll_init(mode);

            // handshake with the dl side
            // we first send the the model_file_path, the backend and the device as environment variables, before starting the main data exchange happens with inference

            # endif
        }

        void set_io_buffers(MLCouplingData<In>* input_after_preprocessing,
                            MLCouplingData<Out>* output_before_postprocessing) override {
            this->input_after_preprocessing = input_after_preprocessing;
            this->output_before_postprocessing = output_before_postprocessing;
        }


        virtual void inference(MLCouplingData<In>* input_after_preprocessing,
                               MLCouplingData<Out>* output_before_postprocessing) override
        {
            guarantee(input_after_preprocessing != nullptr, "PhyDLL inference requires input_after_preprocessing.");
            guarantee(output_before_postprocessing != nullptr, "PhyDLL inference requires output_before_postprocessing.");

            # ifdef WITH_PHYDLL

            # endif

        }

        void send_data(const std::vector<std::string>& keys, MLCouplingData<In>& input_data_after_preprocessing) override {
            return; // TODO
        }

        void inference(std::vector<std::string> input_keys, std::vector<std::string> output_keys) override {
            return; // TODO
        }

        void receive_data(std::vector<std::string> keys, MLCouplingData<Out>& output_data) override {
            return; // TODO
        }

        bool is_flexible() override {
            return false; // for now
        }

    private:
        MLCouplingData<In>* input_after_preprocessing = nullptr;
        MLCouplingData<Out>* output_before_postprocessing = nullptr;

};
