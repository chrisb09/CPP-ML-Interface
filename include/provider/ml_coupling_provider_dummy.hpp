#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ml_coupling_provider_flexible.hpp"
#include "../tool.h"



// @registry_name: Dummy
// @registry_aliases: dummy, Dummy
template <typename In, typename Out>
class MLCouplingProviderDummy : public MLCouplingProvider<In, Out>
{

public:


    MLCouplingProviderDummy(MLCouplingData<In> *input_after_preprocessing = nullptr,
                            MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : input_after_preprocessing(input_after_preprocessing),
          output_before_postprocessing(output_before_postprocessing)
    {
    }

    void set_io_buffers(MLCouplingData<In> *input_after_preprocessing,
                        MLCouplingData<Out> *output_before_postprocessing) override
    {
        this->input_after_preprocessing = input_after_preprocessing;
        this->output_before_postprocessing = output_before_postprocessing;
    }

    void inference(MLCouplingData<In> *input_after_preprocessing,
                   MLCouplingData<Out> *output_before_postprocessing) override
    {
        guarantee(false, "Dummy provider does not implement anything.");
    }

private:
    void initialize_service(MLCouplingData<In> *input_after_preprocessing,
                            MLCouplingData<Out> *output_before_postprocessing)
    {
            guarantee(false, "Dummy provider does not implement anything.");
    }

    MLCouplingData<In> *input_after_preprocessing = nullptr;
    MLCouplingData<Out> *output_before_postprocessing = nullptr;
};
