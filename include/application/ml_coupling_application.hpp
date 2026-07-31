#pragma once

#include "scorep_profiling_state.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../data/ml_coupling_data.hpp"
#include "../library/ml_coupling_library.hpp"
#include "../behavior/ml_coupling_behavior.hpp"
#include "../normalization/ml_coupling_normalization.hpp"

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// @category: application
// Applications own conversion between the public coupling boundary and the
// tensor boundary consumed by an interchangeable coupling library.
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class MLCouplingApplication
{
public:
    MLCouplingData<CouplingInput> coupling_input;
    MLCouplingData<LibraryInput> library_input;
    MLCouplingData<LibraryOutput> library_output;
    MLCouplingData<CouplingOutput> coupling_output;

    MLCouplingApplication(MLCouplingData<CouplingInput> coupling_input,
                          MLCouplingData<CouplingOutput> coupling_output,
                          MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr)
        : MLCouplingApplication(std::move(coupling_input),
                                MLCouplingData<LibraryInput>(),
                                MLCouplingData<LibraryOutput>(),
                                std::move(coupling_output),
                                normalization)
    {
    }

    MLCouplingApplication(MLCouplingData<CouplingInput> coupling_input,
                          MLCouplingData<LibraryInput> library_input,
                          MLCouplingData<LibraryOutput> library_output,
                          MLCouplingData<CouplingOutput> coupling_output,
                          MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr)
        : coupling_input(std::move(coupling_input)),
          library_input(std::move(library_input)),
          library_output(std::move(library_output)),
          coupling_output(std::move(coupling_output)),
          normalization(normalization)
    {
        // Same-type applications need no conversion and can use their public
        // buffers directly. Mixed-type applications allocate their own library
        // buffers in the derived constructor.
        if constexpr (std::is_same_v<CouplingInput, LibraryInput>)
        {
            if (this->library_input.empty()) this->library_input = this->coupling_input;
        }
        if constexpr (std::is_same_v<CouplingOutput, LibraryOutput>)
        {
            if (this->library_output.empty()) this->library_output = this->coupling_output;
        }
    }

    virtual int ml_step(MLCouplingLibrary<LibraryInput, LibraryOutput>& library,
                        MLCouplingBehavior& behavior)
    {
        if (behavior.should_perform_inference())
        {
            prepare_library_input();
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_app_provider_inference)
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
            SCOREP_USER_REGION_BEGIN(handle_app_provider_inference, "app_provider_inference", SCOREP_USER_REGION_TYPE_COMMON)
            }
#endif
            library.static_inference(&library_input, &library_output);
#ifdef USE_SCOREP
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
            SCOREP_USER_REGION_END(handle_app_provider_inference)
            }
#endif
            finalize_coupling_output();
            return behavior.time_step_delta();
        }
        return 0;
    }

    void prepare_library_input()
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_app_prepare_input)
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_BEGIN(handle_app_prepare_input, "app_prepare_input", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        library_input = preprocess_coupling_input(coupling_input);
#ifdef USE_SCOREP
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_END(handle_app_prepare_input)
        }
#endif
    }

    void finalize_coupling_output()
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_app_finalize_output)
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_BEGIN(handle_app_finalize_output, "app_finalize_output", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        coupling_output = postprocess_library_output(library_output);
#ifdef USE_SCOREP
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_END(handle_app_finalize_output)
        }
#endif
    }

    std::pair<MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*> get_library_buffers()
    {
        return std::make_pair(&library_input, &library_output);
    }

    virtual ~MLCouplingApplication() = default;

protected:
    virtual MLCouplingData<LibraryInput>
    preprocess_coupling_input(MLCouplingData<CouplingInput> input)
    {
        if constexpr (std::is_same_v<CouplingInput, LibraryInput>)
        {
            return input;
        }
        else
        {
            throw std::logic_error("MLCouplingApplication requires preprocess_coupling_input() for mixed input types");
        }
    }

    virtual MLCouplingData<CouplingOutput>
    postprocess_library_output(MLCouplingData<LibraryOutput> output)
    {
        if constexpr (std::is_same_v<CouplingOutput, LibraryOutput>)
        {
            return output;
        }
        else
        {
            throw std::logic_error("MLCouplingApplication requires postprocess_library_output() for mixed output types");
        }
    }

    std::unique_ptr<MLCouplingNormalization<LibraryInput, CouplingOutput>> normalization;
};
