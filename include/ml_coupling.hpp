#pragma once

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "provider/ml_coupling_provider.hpp"
#include "provider/ml_coupling_provider_flexible.hpp"
#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior.hpp"
#include "behavior/ml_coupling_behavior_default.hpp"
#include "coupling_type.hpp"
#include "config_overrides.hpp"
#include "tool.h"

// To avoid circular dependency issues with the config, we forward declare the MLCoupling class here and include the config at the end of this file
enum class ConfigCastMode : int;

template <typename In, typename Out>
class MLCoupling;

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    ConfigCastMode cast_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    const ConfigOverrides &overrides);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    ConfigCastMode cast_mode,
    const ConfigOverrides &overrides);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    ConfigCastMode cast_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    const ConfigOverrides &overrides);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    ConfigCastMode cast_mode,
    const ConfigOverrides &overrides);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode);

template <typename In, typename Out>
class MLCoupling
{
    // I'd see this class as essentially the API that the user
    // would interact with.
    friend int main(int, char **);

public:
    MLCoupling<In, Out>(
        MLCouplingProvider<In, Out> *provider,
        MLCouplingApplication<In, Out> *application,
        MLCouplingBehavior *behavior = nullptr)
    {
        this->provider.reset(provider);
        this->application.reset(application);
        this->coupling_type = CouplingType::STATIC;
        this->input_after_preprocessing = &(this->application->input_data_after_preprocessing);
        this->output_before_postprocessing = &(this->application->output_data_before_postprocessing);
        if (coupling_type == CouplingType::STATIC && (this->input_after_preprocessing == nullptr || this->output_before_postprocessing == nullptr))
        {
            guarantee(false, "CouplingType STATIC requires non-null pre/post buffers.");
        }
        this->provider->set_io_buffers(this->input_after_preprocessing, this->output_before_postprocessing);
        if (behavior == nullptr)
        {
            // Use default behavior if none provided
            this->behavior.reset(new MLCouplingBehaviorDefault());
        }
        else
        {
            this->behavior.reset(behavior);
        }
    }

    MLCoupling<In, Out>(
        MLCouplingProvider<In, Out> *provider,
        MLCouplingApplication<In, Out> *application,
        MLCouplingBehavior *behavior,
        CouplingType coupling_type,
        MLCouplingData<In> *input_after_preprocessing,
        MLCouplingData<Out> *output_before_postprocessing)
    {
        this->provider.reset(provider);
        this->application.reset(application);
        this->coupling_type = coupling_type;
        this->input_after_preprocessing = input_after_preprocessing;
        this->output_before_postprocessing = output_before_postprocessing;
        if (behavior == nullptr)
        {
            this->behavior.reset(new MLCouplingBehaviorDefault());
        }
        else
        {
            this->behavior.reset(behavior);
        }

        if (this->coupling_type == CouplingType::FLEXIBLE && !is_flexible())
        {
            guarantee(false, "CouplingType FLEXIBLE requested, but provider does not support flexible coupling.");
        }
        if (this->coupling_type == CouplingType::STATIC &&
            (this->input_after_preprocessing == nullptr || this->output_before_postprocessing == nullptr))
        {
            guarantee(false, "CouplingType STATIC requires non-null pre/post buffers.");
        }

        if (this->coupling_type == CouplingType::STATIC)
        {
            this->provider->set_io_buffers(this->input_after_preprocessing, this->output_before_postprocessing);
        }
    }

    ~MLCoupling()
    {
        // Parameter lifetime management is handled by config construction.
        // We intentionally avoid deleting untyped void* entries here.
    }

    void ml_step()
    {
        if (provider && application && behavior && behavior->should_perform_inference())
        {
            if (application->input_data_after_preprocessing.empty())
            {
                application->input_data_after_preprocessing = application->input_data;
            }
            if (application->output_data_before_postprocessing.empty())
            {
                application->output_data_before_postprocessing = application->output_data;
            }
            if (coupling_type == CouplingType::STATIC)
            {
                provider->inference(input_after_preprocessing,
                                    output_before_postprocessing);
                application->output_data = application->output_data_before_postprocessing;
            }
            else
            {
                provider->inference(&(application->input_data_after_preprocessing),
                                    &(application->output_data_before_postprocessing));
                application->output_data = application->output_data_before_postprocessing;
            }
        }
    }

    bool is_flexible() const
    {
        return dynamic_cast<MLCouplingProviderFlexible<In, Out> *>(provider.get()) != nullptr;
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path, MLCouplingData<In> input_data, MLCouplingData<Out> output_data)
    {
        return create_mlcoupling_from_config_file(config_file_path, std::move(input_data), std::move(output_data));
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   ConfigCastMode cast_mode)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  cast_mode);
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  overrides);
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   ConfigCastMode cast_mode,
                                                   const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  cast_mode,
                                                  overrides);
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  std::move(input_data_after_preprocessing),
                                                  std::move(output_data_before_postprocessing));
    }

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing,
                                                   ConfigCastMode cast_mode)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  std::move(input_data_after_preprocessing),
                                                  std::move(output_data_before_postprocessing),
                                                  cast_mode);
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str, MLCouplingData<In> input_data, MLCouplingData<Out> output_data)
    {
        return create_mlcoupling_from_config(config_str, std::move(input_data), std::move(output_data));
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          ConfigCastMode cast_mode)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             cast_mode);
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             overrides);
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          ConfigCastMode cast_mode,
                                                          const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             cast_mode,
                                             overrides);
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          MLCouplingData<In> input_data_after_preprocessing,
                                                          MLCouplingData<Out> output_data_before_postprocessing)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             std::move(input_data_after_preprocessing),
                                             std::move(output_data_before_postprocessing));
    }

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          MLCouplingData<In> input_data_after_preprocessing,
                                                          MLCouplingData<Out> output_data_before_postprocessing,
                                                          ConfigCastMode cast_mode)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             std::move(input_data_after_preprocessing),
                                             std::move(output_data_before_postprocessing),
                                             cast_mode);
    }

protected:
    void set_parameters(std::vector<void *> params)
    {
        parameters = std::move(params);
    }

private:
    std::vector<void *> parameters; // Store parameters for provider, application, and behavior, so we can free them in the destructor

    std::unique_ptr<MLCouplingProvider<In, Out>> provider;
    std::unique_ptr<MLCouplingApplication<In, Out>> application;
    std::unique_ptr<MLCouplingBehavior> behavior;
    CouplingType coupling_type = CouplingType::STATIC;
    MLCouplingData<In> *input_after_preprocessing = nullptr;
    MLCouplingData<Out> *output_before_postprocessing = nullptr;
};

// As mentioned at the top of this file, including the config.hpp here should avoid circular dependency issues since config.hpp only needs to know about the MLCoupling class declaration

#include "config.hpp"
