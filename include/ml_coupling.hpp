#pragma once

#include <memory>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "library/ml_coupling_library.hpp"
#include "application/ml_coupling_application.hpp"
#include "normalization/ml_coupling_minmax_normalization.hpp"
#include <string>

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif
#include "behavior/ml_coupling_behavior.hpp"
#include "behavior/ml_coupling_behavior_default.hpp"
#include "coupling_type.hpp"
#include "config_overrides.hpp"
#include "tool.h"
#include "training_tracker.hpp"
#include "scorep_profiling_state.hpp"
#include <optional>
#include "logging.hpp"

// To avoid circular dependency issues with the config, we forward declare the MLCoupling class here and include the config at the end of this file
enum class ConfigCastMode : int;
enum class ConfigParameterMatchMode : int;

template <typename CouplingInput, typename CouplingOutput,
          typename LibraryInput = CouplingInput, typename LibraryOutput = CouplingOutput>
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
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    const ConfigOverrides &overrides);

template <typename CouplingInput, typename CouplingOutput,
          typename LibraryInput, typename LibraryOutput>
MLCoupling<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput> *
create_mlcoupling_from_config_with_library_types(
    const std::string &config_str,
    MLCouplingData<CouplingInput> coupling_input,
    MLCouplingData<CouplingOutput> coupling_output,
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
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode,
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
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config(
    const std::string &config_str,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode,
    const ConfigOverrides &overrides);

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
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode);

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
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode,
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
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode,
    const ConfigOverrides &overrides);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode);

template <typename In, typename Out>
MLCoupling<In, Out> *create_mlcoupling_from_config_file(
    const std::string &config_file_path,
    MLCouplingData<In> input_data,
    MLCouplingData<Out> output_data,
    MLCouplingData<In> input_data_after_preprocessing,
    MLCouplingData<Out> output_data_before_postprocessing,
    ConfigCastMode cast_mode,
    ConfigParameterMatchMode parameter_match_mode,
    const ConfigOverrides &overrides);

template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput,
          typename LibraryOutput>
class MLCoupling
{
    // I'd see this class as essentially the API that the user
    // would interact with.
    friend int main(int, char **);

 public:
    using In = CouplingInput;
    using Out = CouplingOutput;

    MLCoupling(
        MLCouplingLibrary<LibraryInput, LibraryOutput> *library,
        MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput> *application,
        MLCouplingBehavior *behavior = nullptr,
        std::string log_level = "",
        std::optional<bool> error_separate = std::nullopt)
    {
        if (!log_level.empty()) {
            logging::set_level(logging::get_level(log_level));
        }
        if (error_separate.has_value()) {
            logging::set_error_seperate(*error_separate);
        }

        this->library.reset(library);
        this->application.reset(application);
        this->coupling_type = CouplingType::STATIC;
        this->library_input = &(this->application->library_input);
        this->library_output = &(this->application->library_output);
        if (coupling_type == CouplingType::STATIC && (this->library_input == nullptr || this->library_output == nullptr))
        {
            guarantee(false, "CouplingType STATIC requires non-null pre/post buffers.");
        }

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

    MLCoupling(
        MLCouplingLibrary<LibraryInput, LibraryOutput> *library,
        MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput> *application,
        MLCouplingBehavior *behavior,
        CouplingType coupling_type,
        MLCouplingData<LibraryInput> *library_input,
        MLCouplingData<LibraryOutput> *library_output,
        std::string log_level = "",
        std::optional<bool> error_separate = std::nullopt)
    {
        if (!log_level.empty()) {
            logging::set_level(logging::get_level(log_level));
        }
        if (error_separate.has_value()) {
            logging::set_error_seperate(*error_separate);
        }

        this->library.reset(library);
        this->application.reset(application);
        this->coupling_type = coupling_type;
        this->library_input = library_input;
        this->library_output = library_output;
        if (behavior == nullptr)
        {
            this->behavior.reset(new MLCouplingBehaviorDefault());
        }
        else
        {
            this->behavior.reset(behavior);
        }

        if (this->coupling_type == CouplingType::STATIC &&
            (this->library_input == nullptr || this->library_output == nullptr))
        {
            guarantee(false, "CouplingType STATIC requires non-null pre/post buffers.");
        }
    }

    ~MLCoupling()
    {
        // Parameter lifetime management is handled by config construction.
        // We intentionally avoid deleting untyped void* entries here.
    }

    // --- High-level Static API ---

    /**
     * @brief Performs the ML coupling step.
     * Delegates to the application's ml_step() which drives the coupling-library/behavior
     * orchestration. Returns the time-step delta (0 for no inference, N for inference).
     */
    int step()
    {
        if (library && application && behavior)
        {
            return application->ml_step(*library, *behavior);
        }
        return 0;
    }

    void set_scorep_detailed_regions_enabled(bool enabled)
    {
        ml_coupling_scorep::set_detailed_regions_enabled(enabled);
    }

    /**
     * @brief Configures the fallback merge strategy for flexible execution.
     * @param strategy The strategy to use (List or Stack).
     */
    void set_merge_strategy(MLCouplingMergeStrategy strategy)
    {
        if (library)
        {
            library->set_merge_strategy(strategy);
        }
    }

    /**
     * @brief Performs a standard static training step.
     * Applies preprocessing to inputs, runs static training, and logs any tracked metadata.
     * @param step_id An identifier for the current training step/iteration.
     */
    void train_step(long long step_id)
    {
        if (library && application)
        {
            application->prepare_library_input();
            auto metadata = library->static_train(&(application->library_input),
                                                  &(application->library_output));
            training_tracker.log(step_id, metadata);
        }
    }

    // --- Proxy Views ---

    /**
     * @brief Proxy class for Ordered Flexible Coupling.
     * Allows staging inputs and targets sequentially without keys.
     */
    class OrderedProxy
    {
        MLCoupling *parent;

    public:
        OrderedProxy(MLCoupling *p) : parent(p) {}

        /**
         * @brief Stages an input tensor/dataset for the next inference or training step.
         * Applies preprocessing to the data.
         * @param data The input data.
         * @return A reference to the proxy for chaining.
         */
        OrderedProxy &set(MLCouplingData<In> data)
        {
            parent->application->coupling_input = std::move(data);
            parent->application->prepare_library_input();
            parent->library->flex_ordered_set(parent->application->library_input);
            return *this;
        }

        /**
         * @brief Stages a target tensor/dataset (labels) for the next training step.
         * Note: Uses type `Out` as targets correspond to model output representations.
         * @param data The target data.
         * @return A reference to the proxy for chaining.
         */
        OrderedProxy &set_target(MLCouplingData<LibraryOutput> data)
        {
            parent->library->flex_ordered_set_target(std::move(data));
            return *this;
        }

        /**
         * @brief Executes ordered inference using the staged inputs.
         * @return A reference to the proxy for chaining.
         */
        OrderedProxy &inference()
        {
            parent->library->flex_ordered_inference(&(parent->application->library_output));
            return *this;
        }

        /**
         * @brief Retrieves an output from the ordered inference results.
         * Applies postprocessing to the retrieved data.
         * @param data The destination buffer for the output data.
         * @return A reference to the proxy for chaining.
         */
        OrderedProxy &get(MLCouplingData<Out> &data)
        {
            parent->library->flex_ordered_get(&(parent->application->library_output));
            parent->application->finalize_coupling_output();
            data = parent->application->coupling_output;
            return *this;
        }

        /**
         * @brief Executes ordered training using the staged inputs and targets.
         * @param step_id The identifier for the current training step.
         * @return A map containing training metadata (e.g., loss) returned by the provider.
         */
        std::map<std::string, double> train(long long step_id)
        {
            auto metadata = parent->library->flex_ordered_train(step_id);
            parent->training_tracker.log(step_id, metadata);
            return metadata;
        }
    };

    /**
     * @brief Proxy class for Keyed Flexible Coupling.
     * Allows staging and retrieving inputs and targets using string keys.
     */
    class KeyedProxy
    {
        MLCoupling *parent;

    public:
        KeyedProxy(MLCoupling *p) : parent(p) {}

        /**
         * @brief Stages a keyed input tensor/dataset for the next inference or training step.
         * Applies preprocessing to the data.
         * @param key The string key identifying the input.
         * @param data The input data.
         * @return A reference to the proxy for chaining.
         */
        KeyedProxy &set(const std::string &key, MLCouplingData<In> data)
        {
            parent->application->coupling_input = std::move(data);
            parent->application->prepare_library_input();
            parent->library->flex_keyed_set(key, parent->application->library_input);
            return *this;
        }

        /**
         * @brief Stages a keyed target tensor/dataset (labels) for the next training step.
         * Note: Uses type `Out` as targets correspond to model output representations.
         * @param key The string key identifying the target.
         * @param data The target data.
         * @return A reference to the proxy for chaining.
         */
        KeyedProxy &set_target(const std::string &key, MLCouplingData<LibraryOutput> data)
        {
            parent->library->flex_keyed_set_target(key, std::move(data));
            return *this;
        }

        /**
         * @brief Executes keyed inference specifying which staged inputs to use and which outputs to expect.
         * @param in_keys A list of keys for the inputs to be used.
         * @param out_keys A list of keys for the expected outputs.
         * @return A reference to the proxy for chaining.
         */
        KeyedProxy &inference(const std::vector<std::string> &in_keys, const std::vector<std::string> &out_keys)
        {
            parent->library->flex_keyed_inference(in_keys, out_keys, &(parent->application->library_output));
            return *this;
        }

        /**
         * @brief Retrieves a keyed output from the inference results.
         * Applies postprocessing to the retrieved data.
         * @param key The string key identifying the output.
         * @param data The destination buffer for the output data.
         * @return A reference to the proxy for chaining.
         */
        KeyedProxy &get(const std::string &key, MLCouplingData<Out> &data)
        {
            parent->library->flex_keyed_get(key, &(parent->application->library_output));
            parent->application->finalize_coupling_output();
            data = parent->application->coupling_output;
            return *this;
        }

        /**
         * @brief Executes keyed training using specified staged inputs and targets.
         * @param step_id The identifier for the current training step.
         * @param in_keys A list of keys for the inputs to be used.
         * @param target_keys A list of keys for the targets to be used.
         * @return A map containing training metadata (e.g., loss) returned by the provider.
         */
        std::map<std::string, double> train(long long step_id, const std::vector<std::string> &in_keys, const std::vector<std::string> &target_keys)
        {
            auto metadata = parent->library->flex_keyed_train(step_id, in_keys, target_keys);
            parent->training_tracker.log(step_id, metadata);
            return metadata;
        }
    };

    /**
     * @brief Provides access to the Ordered Flexible API.
     * @return An OrderedProxy for chaining flexible operations.
     */
    OrderedProxy ordered() { return OrderedProxy(this); }

    /**
     * @brief Provides access to the Keyed Flexible API.
     * @return A KeyedProxy for chaining flexible operations.
     */
    KeyedProxy keyed() { return KeyedProxy(this); }

    // --- Tracking API ---

    /**
     * @brief Enables tracking for a specific metadata field returned during training.
     * @param field The name of the field to track (e.g., "loss", "accuracy").
     */
    void track(const std::string &field) { training_tracker.track(field); }

    /**
     * @brief Retrieves the history of a specific tracked field.
     * @param field The name of the field.
     * @return A vector of historical values for the given field.
     */
    std::vector<double> get_history(const std::string &field) const { return training_tracker.get_history(field); }

    /**
     * @brief Retrieves the history for all tracked fields.
     * @return A map of field names to their historical value vectors.
     */
    std::map<std::string, std::vector<double>> get_history() const { return training_tracker.get_history(); }

    /**
     * @brief Retrieves the most recent value for a specific tracked field.
     * @param field The name of the field.
     * @return The most recent value.
     */
    double get_current(const std::string &field) const { return training_tracker.get_current(field); }

    /**
     * @brief Retrieves the most recent values for all tracked fields.
     * @return A map of field names to their most recent values.
     */
    std::map<std::string, double> get_current() const { return training_tracker.get_current(); }

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
                                                   ConfigCastMode cast_mode,
                                                   ConfigParameterMatchMode parameter_match_mode)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  cast_mode,
                                                  parameter_match_mode);
    }

    static MLCoupling<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput> *create_from_config(
        const std::string &config_file_path,
        MLCouplingData<In> input_data,
        MLCouplingData<Out> output_data,
        const ConfigOverrides &overrides)
    {
        std::ifstream config_file(config_file_path);
        if (!config_file)
        {
            throw std::runtime_error("Failed to open configuration file: " + config_file_path);
        }
        return create_mlcoupling_from_config_with_library_types<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
            std::string(std::istreambuf_iterator<char>(config_file), std::istreambuf_iterator<char>()),
            std::move(input_data), std::move(output_data), overrides);
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
                                                   ConfigCastMode cast_mode,
                                                   ConfigParameterMatchMode parameter_match_mode,
                                                   const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  cast_mode,
                                                  parameter_match_mode,
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

    static MLCoupling<In, Out> *create_from_config(const std::string &config_file_path,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing,
                                                   ConfigCastMode cast_mode,
                                                   ConfigParameterMatchMode parameter_match_mode)
    {
        return create_mlcoupling_from_config_file(config_file_path,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  std::move(input_data_after_preprocessing),
                                                  std::move(output_data_before_postprocessing),
                                                  cast_mode,
                                                  parameter_match_mode);
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
                                                          ConfigCastMode cast_mode,
                                                          ConfigParameterMatchMode parameter_match_mode)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             cast_mode,
                                             parameter_match_mode);
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
                                                          ConfigCastMode cast_mode,
                                                          ConfigParameterMatchMode parameter_match_mode,
                                                          const ConfigOverrides &overrides)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             cast_mode,
                                             parameter_match_mode,
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

    static MLCoupling<In, Out> *create_from_config_string(const std::string &config_str,
                                                          MLCouplingData<In> input_data,
                                                          MLCouplingData<Out> output_data,
                                                          MLCouplingData<In> input_data_after_preprocessing,
                                                          MLCouplingData<Out> output_data_before_postprocessing,
                                                          ConfigCastMode cast_mode,
                                                          ConfigParameterMatchMode parameter_match_mode)
    {
        return create_mlcoupling_from_config(config_str,
                                             std::move(input_data),
                                             std::move(output_data),
                                             std::move(input_data_after_preprocessing),
                                             std::move(output_data_before_postprocessing),
                                             cast_mode,
                                             parameter_match_mode);
    }

protected:
    void set_parameters(std::vector<void *> params)
    {
        parameters = std::move(params);
    }

private:
    std::vector<void *> parameters; // Store parameters for library, application, and behavior, so we can free them in the destructor

    std::unique_ptr<MLCouplingLibrary<LibraryInput, LibraryOutput>> library;
    std::unique_ptr<MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>> application;
    std::unique_ptr<MLCouplingBehavior> behavior;
    CouplingType coupling_type = CouplingType::STATIC;
    MLCouplingData<LibraryInput> *library_input = nullptr;
    MLCouplingData<LibraryOutput> *library_output = nullptr;

    TrainingTracker training_tracker;
};

// As mentioned at the top of this file, including the config.hpp here should avoid circular dependency issues since config.hpp only needs to know about the MLCoupling class declaration

#include "config.hpp"
