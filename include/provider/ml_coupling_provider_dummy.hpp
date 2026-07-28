#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../library/ml_coupling_library.hpp"
#include "../tool.h"
#include "../logging.hpp"


// @registry_name: Dummy
// @registry_aliases: dummy, Dummy
template <typename In, typename Out>
class MLCouplingLibraryDummy : public MLCouplingLibrary<In, Out>
{

public:


    MLCouplingLibraryDummy(MLCouplingData<In> *input_after_preprocessing = nullptr,
                            MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : input_after_preprocessing(input_after_preprocessing),
          output_before_postprocessing(output_before_postprocessing)
    {
        logging::debug("Initialized dummy provider. This does not implement any actual functionality and is just a placeholder.");
    }

    void static_inference(MLCouplingData<In> *input_after_preprocessing,
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
