#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cctype>

#include "logging.hpp"

#include "library/ml_coupling_library.hpp" // MLCouplingLibrary 
#include "normalization/ml_coupling_normalization.hpp" // MLCouplingNormalization 
#include "behavior/ml_coupling_behavior.hpp" // MLCouplingBehavior 
#include "application/ml_coupling_application.hpp" // MLCouplingApplication 

// Includes for subclasses of MLCouplingLibrary
#include "provider/ml_coupling_provider_phydll.hpp" // MLCouplingLibraryPhydll 
#include "provider/ml_coupling_provider_smartsim.hpp" // MLCouplingLibrarySmartsim 
#include "provider/ml_coupling_provider_dummy.hpp" // MLCouplingLibraryDummy 
#include "provider/ml_coupling_provider_aixelerator.hpp" // MLCouplingLibraryAixelerator 

// Includes for subclasses of MLCouplingNormalization
#include "normalization/ml_coupling_minmax_normalization.hpp" // MLCouplingMinMaxNormalization 

// Includes for subclasses of MLCouplingBehavior
#include "behavior/ml_coupling_behavior_default.hpp" // MLCouplingBehaviorDefault 
#include "behavior/ml_coupling_behavior_periodic.hpp" // MLCouplingBehaviorPeriodic 
#include "behavior/ml_coupling_behavior_flow_extrapolator.hpp" // MLCouplingBehaviorFlowExtrapolator 

// Includes for subclasses of MLCouplingApplication
#include "application/ml_coupling_application_flow_extrapolator.hpp" // MLCouplingApplicationFlowExtrapolator 
#include "application/ml_coupling_application_turbulence_closure.hpp" // MLCouplingApplicationTurbulenceClosure 



// Lookup function for MLCouplingLibrary (library)
// Maps registry names and aliases to actual class names
inline std::string resolve_library_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"Phydll", "MLCouplingLibraryPhydll"},
        {"phydll", "MLCouplingLibraryPhydll"},
        {"PhyDLL", "MLCouplingLibraryPhydll"},
        {"Smartsim", "MLCouplingLibrarySmartsim"},
        {"smartsim", "MLCouplingLibrarySmartsim"},
        {"SmartSim", "MLCouplingLibrarySmartsim"},
        {"Dummy", "MLCouplingLibraryDummy"},
        {"dummy", "MLCouplingLibraryDummy"},
        {"Dummy", "MLCouplingLibraryDummy"},
        {"Aixelerator", "MLCouplingLibraryAixelerator"},
        {"aixelerator", "MLCouplingLibraryAixelerator"},
        {"AIxelerator", "MLCouplingLibraryAixelerator"},
        {"aix", "MLCouplingLibraryAixelerator"},
        {"AIx", "MLCouplingLibraryAixelerator"},
        {"AIX", "MLCouplingLibraryAixelerator"},
    };

    auto it = lookup.find(name_or_alias);
    if (it != lookup.end()) {
        return it->second;
    }
    return name_or_alias; // Return as-is if no mapping found
}

// Lookup function for MLCouplingNormalization (normalization)
// Maps registry names and aliases to actual class names
inline std::string resolve_normalization_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"MinMax", "MLCouplingMinMaxNormalization"},
        {"minmax", "MLCouplingMinMaxNormalization"},
        {"min-max", "MLCouplingMinMaxNormalization"},
        {"MinMaxNormalization", "MLCouplingMinMaxNormalization"},
    };

    auto it = lookup.find(name_or_alias);
    if (it != lookup.end()) {
        return it->second;
    }
    return name_or_alias; // Return as-is if no mapping found
}

// Lookup function for MLCouplingBehavior (behavior)
// Maps registry names and aliases to actual class names
inline std::string resolve_behavior_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"Default", "MLCouplingBehaviorDefault"},
        {"default", "MLCouplingBehaviorDefault"},
        {"Periodic", "MLCouplingBehaviorPeriodic"},
        {"periodic", "MLCouplingBehaviorPeriodic"},
        {"FlowExtrapolatorBehavior", "MLCouplingBehaviorFlowExtrapolator"},
        {"flow-extrapolator-behavior", "MLCouplingBehaviorFlowExtrapolator"},
        {"maia-flow-extrapolator-behavior", "MLCouplingBehaviorFlowExtrapolator"},
    };

    auto it = lookup.find(name_or_alias);
    if (it != lookup.end()) {
        return it->second;
    }
    return name_or_alias; // Return as-is if no mapping found
}

// Lookup function for MLCouplingApplication (application)
// Maps registry names and aliases to actual class names
inline std::string resolve_application_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"MLCouplingApplicationFlowExtrapolator", "MLCouplingApplicationFlowExtrapolator"},
        {"flow-extrapolator", "MLCouplingApplicationFlowExtrapolator"},
        {"flow_extrapolator", "MLCouplingApplicationFlowExtrapolator"},
        {"maia-flow-extrapolator", "MLCouplingApplicationFlowExtrapolator"},
        {"TurbulenceClosure", "MLCouplingApplicationTurbulenceClosure"},
        {"turbulence-closure", "MLCouplingApplicationTurbulenceClosure"},
        {"turbulence_closure", "MLCouplingApplicationTurbulenceClosure"},
        {"turbulence", "MLCouplingApplicationTurbulenceClosure"},
    };

    auto it = lookup.find(name_or_alias);
    if (it != lookup.end()) {
        return it->second;
    }
    return name_or_alias; // Return as-is if no mapping found
}

inline std::string resolve_class_name(const std::string& name_or_alias) {
    // This function checks all categories for a matching name or alias and returns the resolved class name.
    std::string resolved;
    resolved = resolve_application_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_behavior_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_normalization_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_library_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    return name_or_alias; // Return as-is if no mapping found in any category
}

// Lookup function to resolve category names to base class names
inline std::string resolve_category_to_base_class(const std::string& category) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"library", "MLCouplingLibrary"},
        {"normalization", "MLCouplingNormalization"},
        {"behavior", "MLCouplingBehavior"},
        {"application", "MLCouplingApplication"},
    };

    auto it = lookup.find(category);
    if (it != lookup.end()) {
        return it->second;
    }
    return category; // Return as-is if no mapping found
}

// Get constructor parameter dependencies for a given class
// Returns pairs of (base_class_type, parameter_name) for parameters that are base classes
inline std::vector<std::pair<std::string, std::string>> get_constructor_dependencies(const std::string& class_name) {
    std::vector<std::pair<std::string, std::string>> dependencies;

    if (class_name == "MLCouplingLibraryPhydll") {
    } else if (class_name == "MLCouplingLibrarySmartsim") {
    } else if (class_name == "MLCouplingLibraryDummy") {
    } else if (class_name == "MLCouplingLibraryAixelerator") {
    } else if (class_name == "MLCouplingMinMaxNormalization") {
    } else if (class_name == "MLCouplingBehaviorDefault") {
    } else if (class_name == "MLCouplingBehaviorPeriodic") {
    } else if (class_name == "MLCouplingBehaviorFlowExtrapolator") {
    } else if (class_name == "MLCouplingApplicationFlowExtrapolator") {
    } else if (class_name == "MLCouplingApplicationTurbulenceClosure") {
        dependencies.push_back({"MLCouplingNormalization", "normalization"});
    }

    return dependencies;
}

// Get constructor signatures for a given class (for help messages)
inline std::vector<std::string> get_constructor_signatures(const std::string& class_name) {
    std::vector<std::string> signatures;

    if (class_name == "MLCouplingLibraryPhydll") {
        signatures.push_back("MLCouplingLibraryPhydll(std::string model_file, std::string backend = \"TORCH\", std::string device = \"GPU\", int batch_size = 0, std::string transport_layout = \"auto\", MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr)");
        return signatures;
    }

    if (class_name == "MLCouplingLibrarySmartsim") {
        signatures.push_back("MLCouplingLibrarySmartsim(std::string device, std::string model_backend, std::string model_path, std::string model_name = \"model\", std::string host = \"\", int port = - 1, int nodes = - 1, int num_gpus = - 1, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, int command_timeout = - 1, int socket_timeout = - 1, int model_timeout = - 1, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }, const std::vector<std::string>& tf_input_keys = { }, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr)");
        signatures.push_back("MLCouplingLibrarySmartsim(std::string device, std::string model_backend, std::string_view model, std::string model_name = \"model\", std::string host = \"\", int port = - 1, int nodes = - 1, int num_gpus = - 1, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, int command_timeout = - 1, int socket_timeout = - 1, int model_timeout = - 1, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }, const std::vector<std::string>& tf_input_keys = { }, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr)");
        return signatures;
    }

    if (class_name == "MLCouplingLibraryDummy") {
        signatures.push_back("MLCouplingLibraryDummy(MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr)");
        return signatures;
    }

    if (class_name == "MLCouplingLibraryAixelerator") {
        signatures.push_back("MLCouplingLibraryAixelerator(std::string model_file, int batchsize = 1, MPI_Comm app_comm = MPI_COMM_WORLD, bool enable_hybrid = false, std::optional<float> host_fraction = std::nullopt, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr, std::string communication_mode = \"collective\")");
        return signatures;
    }

    if (class_name == "MLCouplingMinMaxNormalization") {
        signatures.push_back("MLCouplingMinMaxNormalization(In input_min, In input_max, Out output_min, Out output_max)");
        signatures.push_back("MLCouplingMinMaxNormalization(In* input_data, int input_data_size, Out* output_data, int output_data_size)");
        signatures.push_back("MLCouplingMinMaxNormalization(MLCouplingData<In> input_data, MLCouplingData<Out> output_data)");
        return signatures;
    }

    if (class_name == "MLCouplingBehaviorDefault") {
        signatures.push_back("MLCouplingBehaviorDefault()");
        return signatures;
    }

    if (class_name == "MLCouplingBehaviorPeriodic") {
        signatures.push_back("MLCouplingBehaviorPeriodic(int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference, std::function<bool ( int )> prohibit_inference = allow_inference_at_all_steps)");
        return signatures;
    }

    if (class_name == "MLCouplingBehaviorFlowExtrapolator") {
        signatures.push_back("MLCouplingBehaviorFlowExtrapolator(int inference_interval, int coupled_steps_before_inference, int step_increment_after_inference, int hdf_output_interval, int total_timesteps, double scaling_factor = 1.0, int forecast_window = 1, int input_step_distance = 1, int inference_start_step = 0, int global_step_offset = 0)");
        return signatures;
    }

    if (class_name == "MLCouplingApplicationFlowExtrapolator") {
        signatures.push_back("MLCouplingApplicationFlowExtrapolator(MLCouplingData<CouplingInput> coupling_input, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr, int cube_dimension = 8, int cube_overlap = 0, int input_sequence_length = 1, int forecast_window = 1, int n_ghost_layers = 0)");
        signatures.push_back("MLCouplingApplicationFlowExtrapolator(MLCouplingData<CouplingInput> coupling_input, MLCouplingData<LibraryInput> library_input, MLCouplingData<LibraryOutput> library_output, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr, int cube_dimension = 8, int cube_overlap = 0, int input_sequence_length = 1, int forecast_window = 1, int n_ghost_layers = 0)");
        return signatures;
    }

    if (class_name == "MLCouplingApplicationTurbulenceClosure") {
        signatures.push_back("MLCouplingApplicationTurbulenceClosure(MLCouplingData<CouplingInput> coupling_input, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization)");
        signatures.push_back("MLCouplingApplicationTurbulenceClosure(MLCouplingData<CouplingInput> coupling_input, MLCouplingData<LibraryInput> library_input, MLCouplingData<LibraryOutput> library_output, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization)");
        return signatures;
    }

    return signatures;
}

// Print constructor help to console/log
inline void print_constructor_help(const std::string& class_name) {
    auto sigs = get_constructor_signatures(class_name);
    if (sigs.empty()) { logging::info("No constructors found for " + class_name); return; }
    logging::info("Available constructors for " + class_name + ":");
    for (const auto &s : sigs) logging::info("  " + s);
}

// Get all subclasses of a given base class name
inline std::vector<std::string> get_subclasses(const std::string& base_class_name) {
    std::vector<std::string> subclasses;

    if (base_class_name == "MLCouplingLibrary") {
        subclasses.push_back("MLCouplingLibraryPhydll");
        subclasses.push_back("MLCouplingLibrarySmartsim");
        subclasses.push_back("MLCouplingLibraryDummy");
        subclasses.push_back("MLCouplingLibraryAixelerator");
    }

    if (base_class_name == "MLCouplingNormalization") {
        subclasses.push_back("MLCouplingMinMaxNormalization");
    }

    if (base_class_name == "MLCouplingBehavior") {
        subclasses.push_back("MLCouplingBehaviorDefault");
        subclasses.push_back("MLCouplingBehaviorPeriodic");
        subclasses.push_back("MLCouplingBehaviorFlowExtrapolator");
    }

    if (base_class_name == "MLCouplingApplication") {
        subclasses.push_back("MLCouplingApplicationFlowExtrapolator");
        subclasses.push_back("MLCouplingApplicationTurbulenceClosure");
    }

    return subclasses;
}

// Get all superclasses of a given class name (from subclass up to base class)
inline std::vector<std::string> get_superclasses(const std::string& class_name) {
    std::vector<std::string> superclasses;
    static const std::unordered_map<std::string, std::string> hierarchy = {
        {"MLCouplingLibraryPhydll", "MLCouplingLibrary"},
        {"MLCouplingLibrarySmartsim", "MLCouplingLibrary"},
        {"MLCouplingLibraryDummy", "MLCouplingLibrary"},
        {"MLCouplingLibraryAixelerator", "MLCouplingLibrary"},
        {"MLCouplingMinMaxNormalization", "MLCouplingNormalization"},
        {"MLCouplingBehaviorDefault", "MLCouplingBehavior"},
        {"MLCouplingBehaviorPeriodic", "MLCouplingBehavior"},
        {"MLCouplingBehaviorFlowExtrapolator", "MLCouplingBehavior"},
        {"MLCouplingApplicationFlowExtrapolator", "MLCouplingApplication"},
        {"MLCouplingApplicationTurbulenceClosure", "MLCouplingApplication"},
    };

    auto it = hierarchy.find(class_name);
    if (it != hierarchy.end()) {
        std::string current = it->second;
        superclasses.push_back(current);
        // Note: Currently only supports single inheritance (one level up).
        // If multi-level hierarchies are needed, extend this recursively.
    }
    return superclasses;
}

// ---------------------------------------------------------------------------
// Runtime type identification via typeid comparison
// Returns the human-readable class name for a given (possibly polymorphic) object.
// ---------------------------------------------------------------------------

template<typename LibraryInput, typename LibraryOutput>
inline std::string get_type_name(const MLCouplingLibrary<LibraryInput, LibraryOutput>* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingLibraryPhydll<LibraryInput, LibraryOutput>)) return "MLCouplingLibraryPhydll";
    if (typeid(*obj) == typeid(MLCouplingLibrarySmartsim<LibraryInput, LibraryOutput>)) return "MLCouplingLibrarySmartsim";
    if (typeid(*obj) == typeid(MLCouplingLibraryDummy<LibraryInput, LibraryOutput>)) return "MLCouplingLibraryDummy";
    if (typeid(*obj) == typeid(MLCouplingLibraryAixelerator<LibraryInput, LibraryOutput>)) return "MLCouplingLibraryAixelerator";
    if (typeid(*obj) == typeid(MLCouplingLibrary<LibraryInput, LibraryOutput>)) return "MLCouplingLibrary";
    return "unknown";
}

template<typename LibraryInput, typename LibraryOutput>
inline std::string get_type_name(const MLCouplingLibrary<LibraryInput, LibraryOutput>& obj) {
    return get_type_name(&obj);
}

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingNormalization<In, Out>* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingMinMaxNormalization<In, Out>)) return "MLCouplingMinMaxNormalization";
    if (typeid(*obj) == typeid(MLCouplingNormalization<In, Out>)) return "MLCouplingNormalization";
    return "unknown";
}

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingNormalization<In, Out>& obj) {
    return get_type_name(&obj);
}

inline std::string get_type_name(const MLCouplingBehavior* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingBehaviorDefault)) return "MLCouplingBehaviorDefault";
    if (typeid(*obj) == typeid(MLCouplingBehaviorPeriodic)) return "MLCouplingBehaviorPeriodic";
    if (typeid(*obj) == typeid(MLCouplingBehaviorFlowExtrapolator)) return "MLCouplingBehaviorFlowExtrapolator";
    if (typeid(*obj) == typeid(MLCouplingBehavior)) return "MLCouplingBehavior";
    return "unknown";
}

inline std::string get_type_name(const MLCouplingBehavior& obj) {
    return get_type_name(&obj);
}

template<typename CouplingInput, typename CouplingOutput, typename LibraryInput, typename LibraryOutput>
inline std::string get_type_name(const MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingApplicationFlowExtrapolator<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>)) return "MLCouplingApplicationFlowExtrapolator";
    if (typeid(*obj) == typeid(MLCouplingApplicationTurbulenceClosure<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>)) return "MLCouplingApplicationTurbulenceClosure";
    if (typeid(*obj) == typeid(MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>)) return "MLCouplingApplication";
    return "unknown";
}

template<typename CouplingInput, typename CouplingOutput, typename LibraryInput, typename LibraryOutput>
inline std::string get_type_name(const MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>& obj) {
    return get_type_name(&obj);
}

enum class ConfigCastMode : int { Strict, Relaxed };

enum class ConfigParameterMatchMode : int { Strict, Lenient };

inline ConfigCastMode& config_cast_mode_storage() {
    static ConfigCastMode mode = ConfigCastMode::Relaxed;
    return mode;
}

inline void set_config_cast_mode(ConfigCastMode mode) {
    config_cast_mode_storage() = mode;
}

inline ConfigCastMode get_config_cast_mode() {
    return config_cast_mode_storage();
}

inline ConfigParameterMatchMode& config_parameter_match_mode_storage() {
    static ConfigParameterMatchMode mode = ConfigParameterMatchMode::Strict;
    return mode;
}

inline void set_config_parameter_match_mode(ConfigParameterMatchMode mode) {
    config_parameter_match_mode_storage() = mode;
}

inline ConfigParameterMatchMode get_config_parameter_match_mode() {
    return config_parameter_match_mode_storage();
}

inline bool config_parameter_names_match(const std::unordered_map<std::string, std::pair<int, void*>>& parameter, const std::vector<std::string>& allowed_names) {
    for (const auto& entry : parameter) {
        if (std::find(allowed_names.begin(), allowed_names.end(), entry.first) == allowed_names.end()) {
            return false;
        }
    }
    return true;
}

inline std::string config_cast_to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool config_try_parse_bool(const std::string& value, bool& out) {
    auto lower = config_cast_to_lower(value);
    if (lower == "true" || lower == "1") { out = true; return true; }
    if (lower == "false" || lower == "0") { out = false; return true; }
    return false;
}

template<typename T>
inline T config_cast_from_int64(int64_t value, ConfigCastMode mode) {
    if constexpr (std::is_same_v<T, bool>) {
        if (mode == ConfigCastMode::Strict && value != 0 && value != 1) {
            throw std::runtime_error("Strict cast failed: int64_t to bool requires 0 or 1.");
        }
        return value != 0;
    } else if constexpr (std::is_integral_v<T>) {
        if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) || value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
            throw std::runtime_error("Cast failed: int64_t value out of range for target integral type.");
        }
        return static_cast<T>(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (mode == ConfigCastMode::Strict) {
            throw std::runtime_error("Strict cast failed: int64_t to string is not allowed.");
        }
        return std::to_string(value);
    }
    throw std::runtime_error("Unsupported target type for int64_t config cast.");
}

template<typename T>
inline T config_cast_from_double(double value, ConfigCastMode mode) {
    if constexpr (std::is_same_v<T, bool>) {
        if (mode == ConfigCastMode::Strict && value != 0.0 && value != 1.0) {
            throw std::runtime_error("Strict cast failed: double to bool requires 0.0 or 1.0.");
        }
        return value != 0.0;
    } else if constexpr (std::is_integral_v<T>) {
        if (mode == ConfigCastMode::Strict && std::floor(value) != value) {
            throw std::runtime_error("Strict cast failed: double to integral requires an integer-valued source.");
        }
        if (value < static_cast<double>(std::numeric_limits<T>::min()) || value > static_cast<double>(std::numeric_limits<T>::max())) {
            throw std::runtime_error("Cast failed: double value out of range for target integral type.");
        }
        return static_cast<T>(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) {
            if (mode == ConfigCastMode::Strict && (value < -std::numeric_limits<float>::max() || value > std::numeric_limits<float>::max())) {
                throw std::runtime_error("Strict cast failed: double out of range for float.");
            }
        }
        return static_cast<T>(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (mode == ConfigCastMode::Strict) {
            throw std::runtime_error("Strict cast failed: double to string is not allowed.");
        }
        return std::to_string(value);
    }
    throw std::runtime_error("Unsupported target type for double config cast.");
}

template<typename T>
inline T config_cast_from_bool(bool value, ConfigCastMode mode) {
    if constexpr (std::is_same_v<T, bool>) {
        return value;
    } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
        if (mode == ConfigCastMode::Strict) {
            throw std::runtime_error("Strict cast failed: bool to numeric is not allowed.");
        }
        return static_cast<T>(value ? 1 : 0);
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (mode == ConfigCastMode::Strict) {
            throw std::runtime_error("Strict cast failed: bool to string is not allowed.");
        }
        return value ? std::string("true") : std::string("false");
    }
    throw std::runtime_error("Unsupported target type for bool config cast.");
}

template<typename T>
inline T config_cast_from_string(const std::string& value, ConfigCastMode mode) {
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    }

    if (mode == ConfigCastMode::Strict) {
        throw std::runtime_error("Strict cast failed: string source only supports std::string target.");
    }

    if constexpr (std::is_same_v<T, bool>) {
        bool parsed = false;
        if (!config_try_parse_bool(value, parsed)) {
            throw std::runtime_error("Relaxed cast failed: could not parse string as bool: " + value);
        }
        return parsed;
    }

    if constexpr (std::is_integral_v<T>) {
        size_t pos = 0;
        long long parsed = std::stoll(value, &pos);
        if (pos != value.size()) {
            throw std::runtime_error("Relaxed cast failed: trailing characters in integral string: " + value);
        }
        if (parsed < static_cast<long long>(std::numeric_limits<T>::min()) || parsed > static_cast<long long>(std::numeric_limits<T>::max())) {
            throw std::runtime_error("Relaxed cast failed: parsed integer out of target range.");
        }
        return static_cast<T>(parsed);
    }

    if constexpr (std::is_floating_point_v<T>) {
        size_t pos = 0;
        double parsed = std::stod(value, &pos);
        if (pos != value.size()) {
            throw std::runtime_error("Relaxed cast failed: trailing characters in floating string: " + value);
        }
        return static_cast<T>(parsed);
    }

    throw std::runtime_error("Unsupported target type for string config cast.");
}

// Helper to extract and cast a config parameter based on its runtime type tag.
// Type tags: 0 = no static cast, 1 = int64_t, 2 = double, 3 = std::string (char*), 4 = bool
template<typename T>
T config_param_cast(const std::pair<int, void*>& param) {
    const auto mode = get_config_cast_mode();
    switch (param.first) {
        case 0:
            return *reinterpret_cast<T*>(param.second); // No static cast, just reinterpret
        case 1:
            return config_cast_from_int64<T>(*reinterpret_cast<int64_t*>(param.second), mode);
        case 2:
            return config_cast_from_double<T>(*reinterpret_cast<double*>(param.second), mode);
        case 3:
            return config_cast_from_string<T>(std::string(reinterpret_cast<char*>(param.second)), mode);
        case 4:
            return config_cast_from_bool<T>(*reinterpret_cast<bool*>(param.second), mode);
        default:
            throw std::runtime_error("Unsupported type tag for config cast: " + std::to_string(param.first));
    }
}

template<typename LibraryInput, typename LibraryOutput>
MLCouplingLibrary<LibraryInput, LibraryOutput>* create_instance_mlcouplinglibrary(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_library_class_name(class_name);

    if (resolved_class_name == "MLCouplingLibraryPhydll") {
        // Constructor with 7 parameter(s)
        // Parameters: std::string model_file, std::string backend = "TORCH", std::string device = "GPU", int batch_size = 0, std::string transport_layout = "auto", MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 1 && parameter.size() <= 7) && config_parameter_names_match(parameter, {"model_file", "backend", "device", "batch_size", "transport_layout", "input_after_preprocessing", "output_before_postprocessing"}))) && parameter.find("model_file") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingLibraryPhydll with parameters: " << "model_file=" << config_param_cast<std::string>(parameter.at("model_file")) << ", ""backend=" << (parameter.find("backend") != parameter.end() ? config_param_cast<std::string>(parameter.at("backend")) : (std::string)"TORCH") << ", ""device=" << (parameter.find("device") != parameter.end() ? config_param_cast<std::string>(parameter.at("device")) : (std::string)"GPU") << ", ""batch_size=" << (parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0) << ", ""transport_layout=" << (parameter.find("transport_layout") != parameter.end() ? config_param_cast<std::string>(parameter.at("transport_layout")) : (std::string)"auto") << ", ""input_after_preprocessing=<" << (parameter.find("input_after_preprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""output_before_postprocessing=<" << (parameter.find("output_before_postprocessing") != parameter.end() ? "provided" : "default") << ">";
                logging::debug(create_log_stream.str());
                return new MLCouplingLibraryPhydll<LibraryInput, LibraryOutput>(config_param_cast<std::string>(parameter.at("model_file")), parameter.find("backend") != parameter.end() ? config_param_cast<std::string>(parameter.at("backend")) : (std::string)"TORCH", parameter.find("device") != parameter.end() ? config_param_cast<std::string>(parameter.at("device")) : (std::string)"GPU", parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0, parameter.find("transport_layout") != parameter.end() ? config_param_cast<std::string>(parameter.at("transport_layout")) : (std::string)"auto", parameter.find("input_after_preprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("input_after_preprocessing").second) : (MLCouplingData<LibraryInput>*)nullptr, parameter.find("output_before_postprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("output_before_postprocessing").second) : (MLCouplingData<LibraryOutput>*)nullptr);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingLibraryPhydll: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingLibrarySmartsim") {
        // Constructor with 20 parameter(s)
        // Parameters: std::string device, std::string model_backend, std::string model_path, std::string model_name = "model", std::string host = "", int port = - 1, int nodes = - 1, int num_gpus = - 1, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, int command_timeout = - 1, int socket_timeout = - 1, int model_timeout = - 1, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }, const std::vector<std::string>& tf_input_keys = { }, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 3 && parameter.size() <= 20) && config_parameter_names_match(parameter, {"device", "model_backend", "model_path", "model_name", "host", "port", "nodes", "num_gpus", "first_gpu", "batch_size", "min_batch_size", "min_batch_timeout", "command_timeout", "socket_timeout", "model_timeout", "tf_input_labels", "tf_output_labels", "tf_input_keys", "input_after_preprocessing", "output_before_postprocessing"}))) && parameter.find("device") != parameter.end() && parameter.find("model_backend") != parameter.end() && parameter.find("model_path") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingLibrarySmartsim with parameters: " << "device=" << config_param_cast<std::string>(parameter.at("device")) << ", ""model_backend=" << config_param_cast<std::string>(parameter.at("model_backend")) << ", ""model_path=" << config_param_cast<std::string>(parameter.at("model_path")) << ", ""model_name=" << (parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model") << ", ""host=" << (parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"") << ", ""port=" << (parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)- 1) << ", ""nodes=" << (parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)- 1) << ", ""num_gpus=" << (parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)- 1) << ", ""first_gpu=" << (parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0) << ", ""batch_size=" << (parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0) << ", ""min_batch_size=" << (parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0) << ", ""min_batch_timeout=" << (parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0) << ", ""command_timeout=" << (parameter.find("command_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("command_timeout")) : (int)- 1) << ", ""socket_timeout=" << (parameter.find("socket_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("socket_timeout")) : (int)- 1) << ", ""model_timeout=" << (parameter.find("model_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("model_timeout")) : (int)- 1) << ", ""tf_input_labels=<" << (parameter.find("tf_input_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_output_labels=<" << (parameter.find("tf_output_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_input_keys=<" << (parameter.find("tf_input_keys") != parameter.end() ? "provided" : "default") << ">" << ", ""input_after_preprocessing=<" << (parameter.find("input_after_preprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""output_before_postprocessing=<" << (parameter.find("output_before_postprocessing") != parameter.end() ? "provided" : "default") << ">";
                logging::debug(create_log_stream.str());
                return new MLCouplingLibrarySmartsim<LibraryInput, LibraryOutput>(config_param_cast<std::string>(parameter.at("device")), config_param_cast<std::string>(parameter.at("model_backend")), config_param_cast<std::string>(parameter.at("model_path")), parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model", parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"", parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)- 1, parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)- 1, parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)- 1, parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0, parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0, parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0, parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0, parameter.find("command_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("command_timeout")) : (int)- 1, parameter.find("socket_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("socket_timeout")) : (int)- 1, parameter.find("model_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("model_timeout")) : (int)- 1, parameter.find("tf_input_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_output_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_output_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_input_keys") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_keys").second) : (std::vector<std::string>){ }, parameter.find("input_after_preprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("input_after_preprocessing").second) : (MLCouplingData<LibraryInput>*)nullptr, parameter.find("output_before_postprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("output_before_postprocessing").second) : (MLCouplingData<LibraryOutput>*)nullptr);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingLibrarySmartsim: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 20 parameter(s)
        // Parameters: std::string device, std::string model_backend, std::string_view model, std::string model_name = "model", std::string host = "", int port = - 1, int nodes = - 1, int num_gpus = - 1, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, int command_timeout = - 1, int socket_timeout = - 1, int model_timeout = - 1, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }, const std::vector<std::string>& tf_input_keys = { }, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 3 && parameter.size() <= 20) && config_parameter_names_match(parameter, {"device", "model_backend", "model", "model_name", "host", "port", "nodes", "num_gpus", "first_gpu", "batch_size", "min_batch_size", "min_batch_timeout", "command_timeout", "socket_timeout", "model_timeout", "tf_input_labels", "tf_output_labels", "tf_input_keys", "input_after_preprocessing", "output_before_postprocessing"}))) && parameter.find("device") != parameter.end() && parameter.find("model_backend") != parameter.end() && parameter.find("model") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingLibrarySmartsim with parameters: " << "device=" << config_param_cast<std::string>(parameter.at("device")) << ", ""model_backend=" << config_param_cast<std::string>(parameter.at("model_backend")) << ", ""model=<provided>" << ", ""model_name=" << (parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model") << ", ""host=" << (parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"") << ", ""port=" << (parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)- 1) << ", ""nodes=" << (parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)- 1) << ", ""num_gpus=" << (parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)- 1) << ", ""first_gpu=" << (parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0) << ", ""batch_size=" << (parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0) << ", ""min_batch_size=" << (parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0) << ", ""min_batch_timeout=" << (parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0) << ", ""command_timeout=" << (parameter.find("command_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("command_timeout")) : (int)- 1) << ", ""socket_timeout=" << (parameter.find("socket_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("socket_timeout")) : (int)- 1) << ", ""model_timeout=" << (parameter.find("model_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("model_timeout")) : (int)- 1) << ", ""tf_input_labels=<" << (parameter.find("tf_input_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_output_labels=<" << (parameter.find("tf_output_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_input_keys=<" << (parameter.find("tf_input_keys") != parameter.end() ? "provided" : "default") << ">" << ", ""input_after_preprocessing=<" << (parameter.find("input_after_preprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""output_before_postprocessing=<" << (parameter.find("output_before_postprocessing") != parameter.end() ? "provided" : "default") << ">";
                logging::debug(create_log_stream.str());
                return new MLCouplingLibrarySmartsim<LibraryInput, LibraryOutput>(config_param_cast<std::string>(parameter.at("device")), config_param_cast<std::string>(parameter.at("model_backend")), *reinterpret_cast<std::string_view*>(parameter.at("model").second), parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model", parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"", parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)- 1, parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)- 1, parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)- 1, parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0, parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0, parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0, parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0, parameter.find("command_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("command_timeout")) : (int)- 1, parameter.find("socket_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("socket_timeout")) : (int)- 1, parameter.find("model_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("model_timeout")) : (int)- 1, parameter.find("tf_input_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_output_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_output_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_input_keys") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_keys").second) : (std::vector<std::string>){ }, parameter.find("input_after_preprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("input_after_preprocessing").second) : (MLCouplingData<LibraryInput>*)nullptr, parameter.find("output_before_postprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("output_before_postprocessing").second) : (MLCouplingData<LibraryOutput>*)nullptr);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingLibrarySmartsim: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingLibraryDummy") {
        // Constructor with 2 parameter(s)
        // Parameters: MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 0 && parameter.size() <= 2) && config_parameter_names_match(parameter, {"input_after_preprocessing", "output_before_postprocessing"})))) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingLibraryDummy with parameters: " << "input_after_preprocessing=<" << (parameter.find("input_after_preprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""output_before_postprocessing=<" << (parameter.find("output_before_postprocessing") != parameter.end() ? "provided" : "default") << ">";
                logging::debug(create_log_stream.str());
                return new MLCouplingLibraryDummy<LibraryInput, LibraryOutput>(parameter.find("input_after_preprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("input_after_preprocessing").second) : (MLCouplingData<LibraryInput>*)nullptr, parameter.find("output_before_postprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("output_before_postprocessing").second) : (MLCouplingData<LibraryOutput>*)nullptr);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingLibraryDummy: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingLibraryAixelerator") {
        // Constructor with 8 parameter(s)
        // Parameters: std::string model_file, int batchsize = 1, MPI_Comm app_comm = MPI_COMM_WORLD, bool enable_hybrid = false, std::optional<float> host_fraction = std::nullopt, MLCouplingData<In>* input_after_preprocessing = nullptr, MLCouplingData<Out>* output_before_postprocessing = nullptr, std::string communication_mode = "collective"
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 1 && parameter.size() <= 8) && config_parameter_names_match(parameter, {"model_file", "batchsize", "app_comm", "enable_hybrid", "host_fraction", "input_after_preprocessing", "output_before_postprocessing", "communication_mode"}))) && parameter.find("model_file") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingLibraryAixelerator with parameters: " << "model_file=" << config_param_cast<std::string>(parameter.at("model_file")) << ", ""batchsize=" << (parameter.find("batchsize") != parameter.end() ? config_param_cast<int>(parameter.at("batchsize")) : (int)1) << ", ""app_comm=<" << (parameter.find("app_comm") != parameter.end() ? "provided" : "default") << ">" << ", ""enable_hybrid=" << (parameter.find("enable_hybrid") != parameter.end() ? config_param_cast<bool>(parameter.at("enable_hybrid")) : (bool)false) << ", ""host_fraction=<" << (parameter.find("host_fraction") != parameter.end() ? "provided" : "default") << ">" << ", ""input_after_preprocessing=<" << (parameter.find("input_after_preprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""output_before_postprocessing=<" << (parameter.find("output_before_postprocessing") != parameter.end() ? "provided" : "default") << ">" << ", ""communication_mode=" << (parameter.find("communication_mode") != parameter.end() ? config_param_cast<std::string>(parameter.at("communication_mode")) : (std::string)"collective");
                logging::debug(create_log_stream.str());
                return new MLCouplingLibraryAixelerator<LibraryInput, LibraryOutput>(config_param_cast<std::string>(parameter.at("model_file")), parameter.find("batchsize") != parameter.end() ? config_param_cast<int>(parameter.at("batchsize")) : (int)1, parameter.find("app_comm") != parameter.end() ? reinterpret_cast<MPI_Comm>(parameter.at("app_comm").second) : (MPI_Comm)MPI_COMM_WORLD, parameter.find("enable_hybrid") != parameter.end() ? config_param_cast<bool>(parameter.at("enable_hybrid")) : (bool)false, parameter.find("host_fraction") != parameter.end() ? *reinterpret_cast<std::optional<float>*>(parameter.at("host_fraction").second) : (std::optional<float>)std::nullopt, parameter.find("input_after_preprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("input_after_preprocessing").second) : (MLCouplingData<LibraryInput>*)nullptr, parameter.find("output_before_postprocessing") != parameter.end() ? reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("output_before_postprocessing").second) : (MLCouplingData<LibraryOutput>*)nullptr, parameter.find("communication_mode") != parameter.end() ? config_param_cast<std::string>(parameter.at("communication_mode")) : (std::string)"collective");
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingLibraryAixelerator: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

template<typename In, typename Out>
MLCouplingNormalization<In, Out>* create_instance_mlcouplingnormalization(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_normalization_class_name(class_name);

    if (resolved_class_name == "MLCouplingMinMaxNormalization") {
        // Constructor with 4 parameter(s)
        // Parameters: In input_min, In input_max, Out output_min, Out output_max
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 4) && config_parameter_names_match(parameter, {"input_min", "input_max", "output_min", "output_max"}))) && parameter.find("input_min") != parameter.end() && parameter.find("input_max") != parameter.end() && parameter.find("output_min") != parameter.end() && parameter.find("output_max") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_min=" << config_param_cast<In>(parameter.at("input_min")) << ", ""input_max=" << config_param_cast<In>(parameter.at("input_max")) << ", ""output_min=" << config_param_cast<Out>(parameter.at("output_min")) << ", ""output_max=" << config_param_cast<Out>(parameter.at("output_max"));
                logging::debug(create_log_stream.str());
                return new MLCouplingMinMaxNormalization<In, Out>(config_param_cast<In>(parameter.at("input_min")), config_param_cast<In>(parameter.at("input_max")), config_param_cast<Out>(parameter.at("output_min")), config_param_cast<Out>(parameter.at("output_max")));
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingMinMaxNormalization: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 4 parameter(s)
        // Parameters: In* input_data, int input_data_size, Out* output_data, int output_data_size
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 4) && config_parameter_names_match(parameter, {"input_data", "input_data_size", "output_data", "output_data_size"}))) && parameter.find("input_data") != parameter.end() && parameter.find("input_data_size") != parameter.end() && parameter.find("output_data") != parameter.end() && parameter.find("output_data_size") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_data=" << reinterpret_cast<In*>(parameter.at("input_data").second) << ", ""input_data_size=" << config_param_cast<int>(parameter.at("input_data_size")) << ", ""output_data=" << reinterpret_cast<Out*>(parameter.at("output_data").second) << ", ""output_data_size=" << config_param_cast<int>(parameter.at("output_data_size"));
                logging::debug(create_log_stream.str());
                return new MLCouplingMinMaxNormalization<In, Out>(reinterpret_cast<In*>(parameter.at("input_data").second), config_param_cast<int>(parameter.at("input_data_size")), reinterpret_cast<Out*>(parameter.at("output_data").second), config_param_cast<int>(parameter.at("output_data_size")));
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingMinMaxNormalization: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 2 parameter(s)
        // Parameters: MLCouplingData<In> input_data, MLCouplingData<Out> output_data
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 2) && config_parameter_names_match(parameter, {"input_data", "output_data"}))) && parameter.find("input_data") != parameter.end() && parameter.find("output_data") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_data=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second)) << ", ""output_data=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second));
                logging::debug(create_log_stream.str());
                return new MLCouplingMinMaxNormalization<In, Out>(*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second));
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingMinMaxNormalization: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

inline MLCouplingBehavior* create_instance_mlcouplingbehavior(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_behavior_class_name(class_name);

    if (resolved_class_name == "MLCouplingBehaviorDefault") {
        // Constructor with 0 parameter(s)
        // Parameters: 
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 0) && config_parameter_names_match(parameter, {})))) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingBehaviorDefault with parameters: ";
                logging::debug(create_log_stream.str());
                return new MLCouplingBehaviorDefault();
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingBehaviorDefault: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingBehaviorPeriodic") {
        // Constructor with 5 parameter(s)
        // Parameters: int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference, std::function<bool ( int )> prohibit_inference = allow_inference_at_all_steps
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 4 && parameter.size() <= 5) && config_parameter_names_match(parameter, {"inference_interval", "coupled_steps_before_inference", "coupled_steps_stride", "step_increment_after_inference", "prohibit_inference"}))) && parameter.find("inference_interval") != parameter.end() && parameter.find("coupled_steps_before_inference") != parameter.end() && parameter.find("coupled_steps_stride") != parameter.end() && parameter.find("step_increment_after_inference") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingBehaviorPeriodic with parameters: " << "inference_interval=" << config_param_cast<int>(parameter.at("inference_interval")) << ", ""coupled_steps_before_inference=" << config_param_cast<int>(parameter.at("coupled_steps_before_inference")) << ", ""coupled_steps_stride=" << config_param_cast<int>(parameter.at("coupled_steps_stride")) << ", ""step_increment_after_inference=" << config_param_cast<int>(parameter.at("step_increment_after_inference")) << ", ""prohibit_inference=<" << (parameter.find("prohibit_inference") != parameter.end() ? "provided" : "default") << ">";
                logging::debug(create_log_stream.str());
                return new MLCouplingBehaviorPeriodic(config_param_cast<int>(parameter.at("inference_interval")), config_param_cast<int>(parameter.at("coupled_steps_before_inference")), config_param_cast<int>(parameter.at("coupled_steps_stride")), config_param_cast<int>(parameter.at("step_increment_after_inference")), parameter.find("prohibit_inference") != parameter.end() ? *reinterpret_cast<std::function<bool ( int )>*>(parameter.at("prohibit_inference").second) : (std::function<bool ( int )>)allow_inference_at_all_steps);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingBehaviorPeriodic: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingBehaviorFlowExtrapolator") {
        // Constructor with 10 parameter(s)
        // Parameters: int inference_interval, int coupled_steps_before_inference, int step_increment_after_inference, int hdf_output_interval, int total_timesteps, double scaling_factor = 1.0, int forecast_window = 1, int input_step_distance = 1, int inference_start_step = 0, int global_step_offset = 0
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 5 && parameter.size() <= 10) && config_parameter_names_match(parameter, {"inference_interval", "coupled_steps_before_inference", "step_increment_after_inference", "hdf_output_interval", "total_timesteps", "scaling_factor", "forecast_window", "input_step_distance", "inference_start_step", "global_step_offset"}))) && parameter.find("inference_interval") != parameter.end() && parameter.find("coupled_steps_before_inference") != parameter.end() && parameter.find("step_increment_after_inference") != parameter.end() && parameter.find("hdf_output_interval") != parameter.end() && parameter.find("total_timesteps") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingBehaviorFlowExtrapolator with parameters: " << "inference_interval=" << config_param_cast<int>(parameter.at("inference_interval")) << ", ""coupled_steps_before_inference=" << config_param_cast<int>(parameter.at("coupled_steps_before_inference")) << ", ""step_increment_after_inference=" << config_param_cast<int>(parameter.at("step_increment_after_inference")) << ", ""hdf_output_interval=" << config_param_cast<int>(parameter.at("hdf_output_interval")) << ", ""total_timesteps=" << config_param_cast<int>(parameter.at("total_timesteps")) << ", ""scaling_factor=" << (parameter.find("scaling_factor") != parameter.end() ? config_param_cast<double>(parameter.at("scaling_factor")) : (double)1.0) << ", ""forecast_window=" << (parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1) << ", ""input_step_distance=" << (parameter.find("input_step_distance") != parameter.end() ? config_param_cast<int>(parameter.at("input_step_distance")) : (int)1) << ", ""inference_start_step=" << (parameter.find("inference_start_step") != parameter.end() ? config_param_cast<int>(parameter.at("inference_start_step")) : (int)0) << ", ""global_step_offset=" << (parameter.find("global_step_offset") != parameter.end() ? config_param_cast<int>(parameter.at("global_step_offset")) : (int)0);
                logging::debug(create_log_stream.str());
                return new MLCouplingBehaviorFlowExtrapolator(config_param_cast<int>(parameter.at("inference_interval")), config_param_cast<int>(parameter.at("coupled_steps_before_inference")), config_param_cast<int>(parameter.at("step_increment_after_inference")), config_param_cast<int>(parameter.at("hdf_output_interval")), config_param_cast<int>(parameter.at("total_timesteps")), parameter.find("scaling_factor") != parameter.end() ? config_param_cast<double>(parameter.at("scaling_factor")) : (double)1.0, parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1, parameter.find("input_step_distance") != parameter.end() ? config_param_cast<int>(parameter.at("input_step_distance")) : (int)1, parameter.find("inference_start_step") != parameter.end() ? config_param_cast<int>(parameter.at("inference_start_step")) : (int)0, parameter.find("global_step_offset") != parameter.end() ? config_param_cast<int>(parameter.at("global_step_offset")) : (int)0);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingBehaviorFlowExtrapolator: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

template<typename CouplingInput, typename CouplingOutput, typename LibraryInput = CouplingInput, typename LibraryOutput = CouplingOutput>
MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>* create_instance_mlcouplingapplication(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_application_class_name(class_name);

    if (resolved_class_name == "MLCouplingApplicationFlowExtrapolator") {
        // Constructor with 8 parameter(s)
        // Parameters: MLCouplingData<CouplingInput> coupling_input, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr, int cube_dimension = 8, int cube_overlap = 0, int input_sequence_length = 1, int forecast_window = 1, int n_ghost_layers = 0
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 2 && parameter.size() <= 8) && config_parameter_names_match(parameter, {"coupling_input", "coupling_output", "normalization", "cube_dimension", "cube_overlap", "input_sequence_length", "forecast_window", "n_ghost_layers"}))) && parameter.find("coupling_input") != parameter.end() && parameter.find("coupling_output") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingApplicationFlowExtrapolator with parameters: " << "coupling_input=" << (*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second)) << ", ""coupling_output=" << (*reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second)) << ", ""normalization=<" << (parameter.find("normalization") != parameter.end() ? "provided" : "default") << ">" << ", ""cube_dimension=" << (parameter.find("cube_dimension") != parameter.end() ? config_param_cast<int>(parameter.at("cube_dimension")) : (int)8) << ", ""cube_overlap=" << (parameter.find("cube_overlap") != parameter.end() ? config_param_cast<int>(parameter.at("cube_overlap")) : (int)0) << ", ""input_sequence_length=" << (parameter.find("input_sequence_length") != parameter.end() ? config_param_cast<int>(parameter.at("input_sequence_length")) : (int)1) << ", ""forecast_window=" << (parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1) << ", ""n_ghost_layers=" << (parameter.find("n_ghost_layers") != parameter.end() ? config_param_cast<int>(parameter.at("n_ghost_layers")) : (int)0);
                logging::debug(create_log_stream.str());
                return new MLCouplingApplicationFlowExtrapolator<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second), *reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second), parameter.find("normalization") != parameter.end() ? reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second) : (MLCouplingNormalization<LibraryInput, CouplingOutput>*)nullptr, parameter.find("cube_dimension") != parameter.end() ? config_param_cast<int>(parameter.at("cube_dimension")) : (int)8, parameter.find("cube_overlap") != parameter.end() ? config_param_cast<int>(parameter.at("cube_overlap")) : (int)0, parameter.find("input_sequence_length") != parameter.end() ? config_param_cast<int>(parameter.at("input_sequence_length")) : (int)1, parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1, parameter.find("n_ghost_layers") != parameter.end() ? config_param_cast<int>(parameter.at("n_ghost_layers")) : (int)0);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingApplicationFlowExtrapolator: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 10 parameter(s)
        // Parameters: MLCouplingData<CouplingInput> coupling_input, MLCouplingData<LibraryInput> library_input, MLCouplingData<LibraryOutput> library_output, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr, int cube_dimension = 8, int cube_overlap = 0, int input_sequence_length = 1, int forecast_window = 1, int n_ghost_layers = 0
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() >= 4 && parameter.size() <= 10) && config_parameter_names_match(parameter, {"coupling_input", "library_input", "library_output", "coupling_output", "normalization", "cube_dimension", "cube_overlap", "input_sequence_length", "forecast_window", "n_ghost_layers"}))) && parameter.find("coupling_input") != parameter.end() && parameter.find("library_input") != parameter.end() && parameter.find("library_output") != parameter.end() && parameter.find("coupling_output") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingApplicationFlowExtrapolator with parameters: " << "coupling_input=" << (*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second)) << ", ""library_input=" << (*reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("library_input").second)) << ", ""library_output=" << (*reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("library_output").second)) << ", ""coupling_output=" << (*reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second)) << ", ""normalization=<" << (parameter.find("normalization") != parameter.end() ? "provided" : "default") << ">" << ", ""cube_dimension=" << (parameter.find("cube_dimension") != parameter.end() ? config_param_cast<int>(parameter.at("cube_dimension")) : (int)8) << ", ""cube_overlap=" << (parameter.find("cube_overlap") != parameter.end() ? config_param_cast<int>(parameter.at("cube_overlap")) : (int)0) << ", ""input_sequence_length=" << (parameter.find("input_sequence_length") != parameter.end() ? config_param_cast<int>(parameter.at("input_sequence_length")) : (int)1) << ", ""forecast_window=" << (parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1) << ", ""n_ghost_layers=" << (parameter.find("n_ghost_layers") != parameter.end() ? config_param_cast<int>(parameter.at("n_ghost_layers")) : (int)0);
                logging::debug(create_log_stream.str());
                return new MLCouplingApplicationFlowExtrapolator<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second), *reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("library_input").second), *reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("library_output").second), *reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second), parameter.find("normalization") != parameter.end() ? reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second) : (MLCouplingNormalization<LibraryInput, CouplingOutput>*)nullptr, parameter.find("cube_dimension") != parameter.end() ? config_param_cast<int>(parameter.at("cube_dimension")) : (int)8, parameter.find("cube_overlap") != parameter.end() ? config_param_cast<int>(parameter.at("cube_overlap")) : (int)0, parameter.find("input_sequence_length") != parameter.end() ? config_param_cast<int>(parameter.at("input_sequence_length")) : (int)1, parameter.find("forecast_window") != parameter.end() ? config_param_cast<int>(parameter.at("forecast_window")) : (int)1, parameter.find("n_ghost_layers") != parameter.end() ? config_param_cast<int>(parameter.at("n_ghost_layers")) : (int)0);
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingApplicationFlowExtrapolator: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingApplicationTurbulenceClosure") {
        // Constructor with 3 parameter(s)
        // Parameters: MLCouplingData<CouplingInput> coupling_input, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 3) && config_parameter_names_match(parameter, {"coupling_input", "coupling_output", "normalization"}))) && parameter.find("coupling_input") != parameter.end() && parameter.find("coupling_output") != parameter.end() && parameter.find("normalization") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingApplicationTurbulenceClosure with parameters: " << "coupling_input=" << (*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second)) << ", ""coupling_output=" << (*reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second)) << ", ""normalization=" << reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second);
                logging::debug(create_log_stream.str());
                return new MLCouplingApplicationTurbulenceClosure<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second), *reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second), reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second));
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingApplicationTurbulenceClosure: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 5 parameter(s)
        // Parameters: MLCouplingData<CouplingInput> coupling_input, MLCouplingData<LibraryInput> library_input, MLCouplingData<LibraryOutput> library_output, MLCouplingData<CouplingOutput> coupling_output, MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization
        if ((get_config_parameter_match_mode() == ConfigParameterMatchMode::Lenient || ((parameter.size() == 5) && config_parameter_names_match(parameter, {"coupling_input", "library_input", "library_output", "coupling_output", "normalization"}))) && parameter.find("coupling_input") != parameter.end() && parameter.find("library_input") != parameter.end() && parameter.find("library_output") != parameter.end() && parameter.find("coupling_output") != parameter.end() && parameter.find("normalization") != parameter.end()) {
            try {
                std::ostringstream create_log_stream;
                create_log_stream << "Creating instance of MLCouplingApplicationTurbulenceClosure with parameters: " << "coupling_input=" << (*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second)) << ", ""library_input=" << (*reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("library_input").second)) << ", ""library_output=" << (*reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("library_output").second)) << ", ""coupling_output=" << (*reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second)) << ", ""normalization=" << reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second);
                logging::debug(create_log_stream.str());
                return new MLCouplingApplicationTurbulenceClosure<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(*reinterpret_cast<MLCouplingData<CouplingInput>*>(parameter.at("coupling_input").second), *reinterpret_cast<MLCouplingData<LibraryInput>*>(parameter.at("library_input").second), *reinterpret_cast<MLCouplingData<LibraryOutput>*>(parameter.at("library_output").second), *reinterpret_cast<MLCouplingData<CouplingOutput>*>(parameter.at("coupling_output").second), reinterpret_cast<MLCouplingNormalization<LibraryInput, CouplingOutput>*>(parameter.at("normalization").second));
            } catch (const std::exception& e) {
                logging::error(std::string("Exception in factory for class MLCouplingApplicationTurbulenceClosure: ") + e.what());
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

