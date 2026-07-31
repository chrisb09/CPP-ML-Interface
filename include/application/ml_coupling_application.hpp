#pragma once

#include "scorep_profiling_state.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "../data/ml_coupling_data.hpp"
#include "../provider/ml_coupling_provider.hpp"
#include "../behavior/ml_coupling_behavior.hpp"
#include "../normalization/ml_coupling_normalization.hpp"

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// @category: application
template <typename In, typename Out>
class MLCouplingApplication
{
public:
    MLCouplingData<In> input_data;
    MLCouplingData<In> input_data_after_preprocessing;
    MLCouplingData<Out> output_data_before_postprocessing;
    MLCouplingData<Out> output_data;

    // The full constructor
    MLCouplingApplication(
        std::vector<In *> input_data,
        std::vector<std::vector<int>> input_data_dimensions,
        std::vector<Out *> output_data,
        std::vector<std::vector<int>> output_data_dimensions,
        MLCouplingNormalization<In, Out> *normalization)
    {
        if (input_data.size() != input_data_dimensions.size())
        {
            throw std::invalid_argument("MLCouplingApplication(): input_data and input_data_dimensions sizes must match");
        }
        if (output_data.size() != output_data_dimensions.size())
        {
            throw std::invalid_argument("MLCouplingApplication(): output_data and output_data_dimensions sizes must match");
        }

        std::vector<MLCouplingTensor<In>> input_tensors;
        input_tensors.reserve(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            input_tensors.push_back(MLCouplingTensor<In>::wrap_flat(input_data[i], input_data_dimensions[i]));
        }

        std::vector<MLCouplingTensor<Out>> output_tensors;
        output_tensors.reserve(output_data.size());
        for (size_t i = 0; i < output_data.size(); ++i)
        {
            output_tensors.push_back(MLCouplingTensor<Out>::wrap_flat(output_data[i], output_data_dimensions[i]));
        }

        this->input_data = MLCouplingData<In>(std::move(input_tensors));
        this->output_data = MLCouplingData<Out>(std::move(output_tensors));
        this->input_data_after_preprocessing = this->input_data;
        this->output_data_before_postprocessing = this->output_data;
        this->normalization.reset(normalization);
    }

    /*
     * This is a shorthand constructor, it assumes that the preprocssing and postprocessing are either in-situ or not needed at all, so it uses the input and output data directly as the preprocessed and postprocessed data.
     */
    MLCouplingApplication(
        MLCouplingData<In> input_data,
        MLCouplingData<Out> output_data,
        MLCouplingNormalization<In, Out> *normalization)
        : MLCouplingApplication(std::move(input_data),
                                MLCouplingData<In>(),
                                MLCouplingData<Out>(),
                                std::move(output_data),
                                normalization) {}

    MLCouplingApplication(
        MLCouplingData<In> input_data,
        MLCouplingData<In> input_data_after_preprocessing,
        MLCouplingData<Out> output_data_before_postprocessing,
        MLCouplingData<Out> output_data,
        MLCouplingNormalization<In, Out> *normalization)
    {
        this->input_data = std::move(input_data);
        this->output_data = std::move(output_data);
        if (input_data_after_preprocessing.empty())
        {
            this->input_data_after_preprocessing = this->input_data;
        }
        else
        {
            this->input_data_after_preprocessing = std::move(input_data_after_preprocessing);
        }
        if (output_data_before_postprocessing.empty())
        {
            this->output_data_before_postprocessing = this->output_data;
        }
        else
        {
            this->output_data_before_postprocessing = std::move(output_data_before_postprocessing);
        }
        this->normalization.reset(normalization);
    }

    virtual int ml_step(MLCouplingProvider<In,Out>& provider, MLCouplingBehavior& behavior)
    {
        if (behavior.should_perform_inference())
        {
            prepare_input();
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_app_provider_inference)
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
            SCOREP_USER_REGION_BEGIN(handle_app_provider_inference, "app_provider_inference", SCOREP_USER_REGION_TYPE_COMMON)
            }
#endif
            provider.static_inference(&input_data_after_preprocessing,
                                       &output_data_before_postprocessing);
#ifdef USE_SCOREP
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
            SCOREP_USER_REGION_END(handle_app_provider_inference)
            }
#endif
            finalize_output();
            return behavior.time_step_delta();
        }
        return 0;
    }

    void prepare_input()
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_app_prepare_input)
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_BEGIN(handle_app_prepare_input, "app_prepare_input", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        input_data_after_preprocessing = preprocess(input_data);
#ifdef USE_SCOREP
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_END(handle_app_prepare_input)
        }
#endif
    }

    void finalize_output()
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_app_finalize_output)
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_BEGIN(handle_app_finalize_output, "app_finalize_output", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        output_data = postprocess(output_data_before_postprocessing);
#ifdef USE_SCOREP
        if (ml_coupling_scorep::detailed_regions_are_enabled()) {
        SCOREP_USER_REGION_END(handle_app_finalize_output)
        }
#endif
    }

    std::pair<MLCouplingData<In> *, MLCouplingData<Out> *> get_pre_post_buffers()
    {
        return std::make_pair(&input_data_after_preprocessing, &output_data_before_postprocessing);
    }

    virtual ~MLCouplingApplication() = default;

protected:
    virtual MLCouplingData<In> preprocess(MLCouplingData<In> input_data) { return input_data; }

    virtual void coupling_step(MLCouplingData<In> input_data_after_preprocessing) { (void)input_data_after_preprocessing; }

    virtual MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) { return output_data_before_postprocessing; };

    std::unique_ptr<MLCouplingNormalization<In, Out>> normalization;
};
