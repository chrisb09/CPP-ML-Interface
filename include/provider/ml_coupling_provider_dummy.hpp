#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../library/ml_coupling_library.hpp"
#include "../tool.h"
#include "../logging.hpp"

/**
 * @file ml_coupling_provider_dummy.hpp
 * @brief Placeholder / mock inference provider for testing and validation.
 */

/**
 * @brief Mock / dummy inference provider.
 *
 * Useful for validating pipeline configuration and interface linkages without
 * requiring GPU runtimes or external ML dependencies.
 *
 * @tparam In Primitive scalar type of input features.
 * @tparam Out Primitive scalar type of output features.
 * @ingroup cpp_library
 */
// @registry_name: Dummy
// @registry_aliases: dummy, Dummy
template <typename In, typename Out>
class MLCouplingLibraryDummy : public MLCouplingLibrary<In, Out>
{

public:
    /**
     * @brief Constructs a dummy provider.
     * @param input_after_preprocessing Optional preprocessed input data buffer.
     * @param output_before_postprocessing Optional output prediction buffer.
     */
    MLCouplingLibraryDummy(MLCouplingData<In> *input_after_preprocessing = nullptr,
                            MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : input_after_preprocessing(input_after_preprocessing),
          output_before_postprocessing(output_before_postprocessing)
    {
        logging::debug("Initialized dummy provider. This does not implement any actual functionality and is just a placeholder.");
    }

    /**
     * @brief Fails intentionally to signify dummy execution.
     * @param input_after_preprocessing Input data.
     * @param output_before_postprocessing Output data.
     */
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
