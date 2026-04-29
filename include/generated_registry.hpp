#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cctype>

#include "provider/ml_coupling_provider.hpp" // MLCouplingProvider 
#include "normalization/ml_coupling_normalization.hpp" // MLCouplingNormalization 
#include "behavior/ml_coupling_behavior.hpp" // MLCouplingBehavior 
#include "application/ml_coupling_application.hpp" // MLCouplingApplication 

// Includes for subclasses of MLCouplingProvider
#include "provider/ml_coupling_provider_aixelerator.hpp" // MLCouplingProviderAixelerator 
#include "provider/ml_coupling_provider_phydll.hpp" // MLCouplingProviderPhydll 
#include "provider/ml_coupling_provider_smartsim.hpp" // MLCouplingProviderSmartsim 

// Includes for subclasses of MLCouplingNormalization
#include "normalization/ml_coupling_minmax_normalization.hpp" // MLCouplingMinMaxNormalization 

// Includes for subclasses of MLCouplingBehavior
#include "behavior/ml_coupling_behavior_default.hpp" // MLCouplingBehaviorDefault 
#include "behavior/ml_coupling_behavior_periodic.hpp" // MLCouplingBehaviorPeriodic 

// Includes for subclasses of MLCouplingApplication
#include "application/ml_coupling_application_turbulence_closure.hpp" // MLCouplingApplicationTurbulenceClosure 



// Lookup function for MLCouplingProvider (provider)
// Maps registry names and aliases to actual class names
inline std::string resolve_provider_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"Aixelerator", "MLCouplingProviderAixelerator"},
        {"aixelerator", "MLCouplingProviderAixelerator"},
        {"AIxelerator", "MLCouplingProviderAixelerator"},
        {"aix", "MLCouplingProviderAixelerator"},
        {"AIx", "MLCouplingProviderAixelerator"},
        {"AIX", "MLCouplingProviderAixelerator"},
        {"Phydll", "MLCouplingProviderPhydll"},
        {"phydll", "MLCouplingProviderPhydll"},
        {"PhyDLL", "MLCouplingProviderPhydll"},
        {"Smartsim", "MLCouplingProviderSmartsim"},
        {"smartsim", "MLCouplingProviderSmartsim"},
        {"SmartSim", "MLCouplingProviderSmartsim"},
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
    resolved = resolve_normalization_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_behavior_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_provider_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    return name_or_alias; // Return as-is if no mapping found in any category
}

// Lookup function to resolve category names to base class names
inline std::string resolve_category_to_base_class(const std::string& category) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"provider", "MLCouplingProvider"},
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

    if (class_name == "MLCouplingProviderAixelerator") {
    } else if (class_name == "MLCouplingProviderPhydll") {
    } else if (class_name == "MLCouplingProviderSmartsim") {
    } else if (class_name == "MLCouplingMinMaxNormalization") {
    } else if (class_name == "MLCouplingBehaviorDefault") {
    } else if (class_name == "MLCouplingBehaviorPeriodic") {
    } else if (class_name == "MLCouplingApplicationTurbulenceClosure") {
        dependencies.push_back({"MLCouplingNormalization", "normalization"});
    }

    return dependencies;
}

// Get constructor signatures for a given class (for help messages)
inline std::vector<std::string> get_constructor_signatures(const std::string& class_name) {
    std::vector<std::string> signatures;

    if (class_name == "MLCouplingProviderAixelerator") {
        signatures.push_back("MLCouplingProviderAixelerator(std::string model_file, int batchsize = 1, MPI_Comm app_comm = MPI_COMM_WORLD, bool enable_hybrid = false, std::optional<float> host_fraction = std :: nullopt)");
        return signatures;
    }

    if (class_name == "MLCouplingProviderPhydll") {
        signatures.push_back("MLCouplingProviderPhydll()");
        return signatures;
    }

    if (class_name == "MLCouplingProviderSmartsim") {
        signatures.push_back("MLCouplingProviderSmartsim(std::string device, std::string model_backend, std::string model_path, std::string model_name = \"model\", std::string host = \"localhost\", int port = 6379, int nodes = 1, int num_gpus = 0, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { })");
        signatures.push_back("MLCouplingProviderSmartsim(std::string device, std::string model_backend, std::string_view model, std::string model_name = \"model\", std::string host = \"localhost\", int port = 6379, int nodes = 1, int num_gpus = 0, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { })");
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

    if (class_name == "MLCouplingApplicationTurbulenceClosure") {
        signatures.push_back("MLCouplingApplicationTurbulenceClosure(MLCouplingData<In> input_data, MLCouplingData<Out> output_data, MLCouplingNormalization<In, Out>* normalization)");
        signatures.push_back("MLCouplingApplicationTurbulenceClosure(MLCouplingData<In> input_data, MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out> output_data_before_postprocessing, MLCouplingData<Out> output_data, MLCouplingNormalization<In, Out>* normalization)");
        return signatures;
    }

    return signatures;
}

// Print constructor help to console/log
inline void print_constructor_help(const std::string& class_name) {
    auto sigs = get_constructor_signatures(class_name);
    if (sigs.empty()) { std::cout << "No constructors found for " << class_name << std::endl; return; }
    std::cout << "Available constructors for " << class_name << ":" << std::endl;
    for (const auto &s : sigs) std::cout << "  " << s << std::endl;
}

// Get all subclasses of a given base class name
inline std::vector<std::string> get_subclasses(const std::string& base_class_name) {
    std::vector<std::string> subclasses;

    if (base_class_name == "MLCouplingProvider") {
        subclasses.push_back("MLCouplingProviderAixelerator");
        subclasses.push_back("MLCouplingProviderPhydll");
        subclasses.push_back("MLCouplingProviderSmartsim");
    }

    if (base_class_name == "MLCouplingNormalization") {
        subclasses.push_back("MLCouplingMinMaxNormalization");
    }

    if (base_class_name == "MLCouplingBehavior") {
        subclasses.push_back("MLCouplingBehaviorDefault");
        subclasses.push_back("MLCouplingBehaviorPeriodic");
    }

    if (base_class_name == "MLCouplingApplication") {
        subclasses.push_back("MLCouplingApplicationTurbulenceClosure");
    }

    return subclasses;
}

// Get all superclasses of a given class name (from subclass up to base class)
inline std::vector<std::string> get_superclasses(const std::string& class_name) {
    std::vector<std::string> superclasses;
    static const std::unordered_map<std::string, std::string> hierarchy = {
        {"MLCouplingProviderAixelerator", "MLCouplingProvider"},
        {"MLCouplingProviderPhydll", "MLCouplingProvider"},
        {"MLCouplingProviderSmartsim", "MLCouplingProvider"},
        {"MLCouplingMinMaxNormalization", "MLCouplingNormalization"},
        {"MLCouplingBehaviorDefault", "MLCouplingBehavior"},
        {"MLCouplingBehaviorPeriodic", "MLCouplingBehavior"},
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

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingProvider<In, Out>* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingProviderAixelerator<In, Out>)) return "MLCouplingProviderAixelerator";
    if (typeid(*obj) == typeid(MLCouplingProviderPhydll<In, Out>)) return "MLCouplingProviderPhydll";
    if (typeid(*obj) == typeid(MLCouplingProviderSmartsim<In, Out>)) return "MLCouplingProviderSmartsim";
    if (typeid(*obj) == typeid(MLCouplingProvider<In, Out>)) return "MLCouplingProvider";
    return "unknown";
}

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingProvider<In, Out>& obj) {
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
    if (typeid(*obj) == typeid(MLCouplingBehavior)) return "MLCouplingBehavior";
    return "unknown";
}

inline std::string get_type_name(const MLCouplingBehavior& obj) {
    return get_type_name(&obj);
}

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingApplication<In, Out>* obj) {
    if (!obj) return "nullptr";
    if (typeid(*obj) == typeid(MLCouplingApplicationTurbulenceClosure<In, Out>)) return "MLCouplingApplicationTurbulenceClosure";
    if (typeid(*obj) == typeid(MLCouplingApplication<In, Out>)) return "MLCouplingApplication";
    return "unknown";
}

template<typename In, typename Out>
inline std::string get_type_name(const MLCouplingApplication<In, Out>& obj) {
    return get_type_name(&obj);
}

enum class ConfigCastMode : int { Strict, Relaxed };

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

template<typename In, typename Out>
MLCouplingProvider<In, Out>* create_instance_mlcouplingprovider(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_provider_class_name(class_name);

    if (resolved_class_name == "MLCouplingProviderAixelerator") {
        // Constructor with 5 parameter(s)
        // Parameters: std::string model_file, int batchsize = 1, MPI_Comm app_comm = MPI_COMM_WORLD, bool enable_hybrid = false, std::optional<float> host_fraction = std :: nullopt
        if (parameter.size() >= 1 && parameter.size() <= 5) {
            try {
                std::cout << "Creating instance of MLCouplingProviderAixelerator with parameters: " << "model_file=" << config_param_cast<std::string>(parameter.at("model_file")) << ", ""batchsize=" << (parameter.find("batchsize") != parameter.end() ? config_param_cast<int>(parameter.at("batchsize")) : (int)1) << ", ""app_comm=<" << (parameter.find("app_comm") != parameter.end() ? "provided" : "default") << ">" << ", ""enable_hybrid=" << (parameter.find("enable_hybrid") != parameter.end() ? config_param_cast<bool>(parameter.at("enable_hybrid")) : (bool)false) << ", ""host_fraction=<" << (parameter.find("host_fraction") != parameter.end() ? "provided" : "default") << ">" << std::endl;
                return new MLCouplingProviderAixelerator<In, Out>(config_param_cast<std::string>(parameter.at("model_file")), parameter.find("batchsize") != parameter.end() ? config_param_cast<int>(parameter.at("batchsize")) : (int)1, parameter.find("app_comm") != parameter.end() ? *reinterpret_cast<MPI_Comm*>(parameter.at("app_comm").second) : (MPI_Comm)MPI_COMM_WORLD, parameter.find("enable_hybrid") != parameter.end() ? config_param_cast<bool>(parameter.at("enable_hybrid")) : (bool)false, parameter.find("host_fraction") != parameter.end() ? *reinterpret_cast<std::optional<float>*>(parameter.at("host_fraction").second) : (std::optional<float>)std :: nullopt);
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingProviderPhydll") {
        // Constructor with 0 parameter(s)
        // Parameters: 
        if (parameter.size() == 0) {
            try {
                std::cout << "Creating instance of MLCouplingProviderPhydll with parameters: " << std::endl;
                return new MLCouplingProviderPhydll<In, Out>();
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingProviderSmartsim") {
        // Constructor with 14 parameter(s)
        // Parameters: std::string device, std::string model_backend, std::string model_path, std::string model_name = "model", std::string host = "localhost", int port = 6379, int nodes = 1, int num_gpus = 0, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }
        if (parameter.size() >= 3 && parameter.size() <= 14) {
            try {
                std::cout << "Creating instance of MLCouplingProviderSmartsim with parameters: " << "device=" << config_param_cast<std::string>(parameter.at("device")) << ", ""model_backend=" << config_param_cast<std::string>(parameter.at("model_backend")) << ", ""model_path=" << config_param_cast<std::string>(parameter.at("model_path")) << ", ""model_name=" << (parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model") << ", ""host=" << (parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost") << ", ""port=" << (parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379) << ", ""nodes=" << (parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1) << ", ""num_gpus=" << (parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)0) << ", ""first_gpu=" << (parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0) << ", ""batch_size=" << (parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0) << ", ""min_batch_size=" << (parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0) << ", ""min_batch_timeout=" << (parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0) << ", ""tf_input_labels=<" << (parameter.find("tf_input_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_output_labels=<" << (parameter.find("tf_output_labels") != parameter.end() ? "provided" : "default") << ">" << std::endl;
                return new MLCouplingProviderSmartsim<In, Out>(config_param_cast<std::string>(parameter.at("device")), config_param_cast<std::string>(parameter.at("model_backend")), config_param_cast<std::string>(parameter.at("model_path")), parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model", parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost", parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379, parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1, parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)0, parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0, parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0, parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0, parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0, parameter.find("tf_input_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_output_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_output_labels").second) : (std::vector<std::string>){ });
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 14 parameter(s)
        // Parameters: std::string device, std::string model_backend, std::string_view model, std::string model_name = "model", std::string host = "localhost", int port = 6379, int nodes = 1, int num_gpus = 0, int first_gpu = 0, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::vector<std::string>& tf_input_labels = { }, const std::vector<std::string>& tf_output_labels = { }
        if (parameter.size() >= 3 && parameter.size() <= 14) {
            try {
                std::cout << "Creating instance of MLCouplingProviderSmartsim with parameters: " << "device=" << config_param_cast<std::string>(parameter.at("device")) << ", ""model_backend=" << config_param_cast<std::string>(parameter.at("model_backend")) << ", ""model=<provided>" << ", ""model_name=" << (parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model") << ", ""host=" << (parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost") << ", ""port=" << (parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379) << ", ""nodes=" << (parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1) << ", ""num_gpus=" << (parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)0) << ", ""first_gpu=" << (parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0) << ", ""batch_size=" << (parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0) << ", ""min_batch_size=" << (parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0) << ", ""min_batch_timeout=" << (parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0) << ", ""tf_input_labels=<" << (parameter.find("tf_input_labels") != parameter.end() ? "provided" : "default") << ">" << ", ""tf_output_labels=<" << (parameter.find("tf_output_labels") != parameter.end() ? "provided" : "default") << ">" << std::endl;
                return new MLCouplingProviderSmartsim<In, Out>(config_param_cast<std::string>(parameter.at("device")), config_param_cast<std::string>(parameter.at("model_backend")), *reinterpret_cast<std::string_view*>(parameter.at("model").second), parameter.find("model_name") != parameter.end() ? config_param_cast<std::string>(parameter.at("model_name")) : (std::string)"model", parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost", parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379, parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1, parameter.find("num_gpus") != parameter.end() ? config_param_cast<int>(parameter.at("num_gpus")) : (int)0, parameter.find("first_gpu") != parameter.end() ? config_param_cast<int>(parameter.at("first_gpu")) : (int)0, parameter.find("batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("batch_size")) : (int)0, parameter.find("min_batch_size") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_size")) : (int)0, parameter.find("min_batch_timeout") != parameter.end() ? config_param_cast<int>(parameter.at("min_batch_timeout")) : (int)0, parameter.find("tf_input_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_input_labels").second) : (std::vector<std::string>){ }, parameter.find("tf_output_labels") != parameter.end() ? *reinterpret_cast<std::vector<std::string>*>(parameter.at("tf_output_labels").second) : (std::vector<std::string>){ });
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
        if (parameter.size() == 4) {
            try {
                std::cout << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_min=<provided>" << ", ""input_max=<provided>" << ", ""output_min=<provided>" << ", ""output_max=<provided>" << std::endl;
                return new MLCouplingMinMaxNormalization<In, Out>(*reinterpret_cast<In*>(parameter.at("input_min").second), *reinterpret_cast<In*>(parameter.at("input_max").second), *reinterpret_cast<Out*>(parameter.at("output_min").second), *reinterpret_cast<Out*>(parameter.at("output_max").second));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 4 parameter(s)
        // Parameters: In* input_data, int input_data_size, Out* output_data, int output_data_size
        if (parameter.size() == 4) {
            try {
                std::cout << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_data=" << reinterpret_cast<In*>(parameter.at("input_data").second) << ", ""input_data_size=" << config_param_cast<int>(parameter.at("input_data_size")) << ", ""output_data=" << reinterpret_cast<Out*>(parameter.at("output_data").second) << ", ""output_data_size=" << config_param_cast<int>(parameter.at("output_data_size")) << std::endl;
                return new MLCouplingMinMaxNormalization<In, Out>(reinterpret_cast<In*>(parameter.at("input_data").second), config_param_cast<int>(parameter.at("input_data_size")), reinterpret_cast<Out*>(parameter.at("output_data").second), config_param_cast<int>(parameter.at("output_data_size")));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 2 parameter(s)
        // Parameters: MLCouplingData<In> input_data, MLCouplingData<Out> output_data
        if (parameter.size() == 2) {
            try {
                std::cout << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_data=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second)) << ", ""output_data=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second)) << std::endl;
                return new MLCouplingMinMaxNormalization<In, Out>(*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

MLCouplingBehavior* create_instance_mlcouplingbehavior(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_behavior_class_name(class_name);

    if (resolved_class_name == "MLCouplingBehaviorDefault") {
        // Constructor with 0 parameter(s)
        // Parameters: 
        if (parameter.size() == 0) {
            try {
                std::cout << "Creating instance of MLCouplingBehaviorDefault with parameters: " << std::endl;
                return new MLCouplingBehaviorDefault();
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    } else if (resolved_class_name == "MLCouplingBehaviorPeriodic") {
        // Constructor with 5 parameter(s)
        // Parameters: int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference, std::function<bool ( int )> prohibit_inference = allow_inference_at_all_steps
        if (parameter.size() >= 4 && parameter.size() <= 5) {
            try {
                std::cout << "Creating instance of MLCouplingBehaviorPeriodic with parameters: " << "inference_interval=" << config_param_cast<int>(parameter.at("inference_interval")) << ", ""coupled_steps_before_inference=" << config_param_cast<int>(parameter.at("coupled_steps_before_inference")) << ", ""coupled_steps_stride=" << config_param_cast<int>(parameter.at("coupled_steps_stride")) << ", ""step_increment_after_inference=" << config_param_cast<int>(parameter.at("step_increment_after_inference")) << ", ""prohibit_inference=<" << (parameter.find("prohibit_inference") != parameter.end() ? "provided" : "default") << ">" << std::endl;
                return new MLCouplingBehaviorPeriodic(config_param_cast<int>(parameter.at("inference_interval")), config_param_cast<int>(parameter.at("coupled_steps_before_inference")), config_param_cast<int>(parameter.at("coupled_steps_stride")), config_param_cast<int>(parameter.at("step_increment_after_inference")), parameter.find("prohibit_inference") != parameter.end() ? *reinterpret_cast<std::function<bool ( int )>*>(parameter.at("prohibit_inference").second) : (std::function<bool ( int )>)allow_inference_at_all_steps);
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

template<typename In, typename Out>
MLCouplingApplication<In, Out>* create_instance_mlcouplingapplication(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_application_class_name(class_name);

    if (resolved_class_name == "MLCouplingApplicationTurbulenceClosure") {
        // Constructor with 3 parameter(s)
        // Parameters: MLCouplingData<In> input_data, MLCouplingData<Out> output_data, MLCouplingNormalization<In, Out>* normalization
        if (parameter.size() == 3) {
            try {
                std::cout << "Creating instance of MLCouplingApplicationTurbulenceClosure with parameters: " << "input_data=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second)) << ", ""output_data=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second)) << ", ""normalization=" << reinterpret_cast<MLCouplingNormalization<In, Out>*>(parameter.at("normalization").second) << std::endl;
                return new MLCouplingApplicationTurbulenceClosure<In, Out>(*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second), reinterpret_cast<MLCouplingNormalization<In, Out>*>(parameter.at("normalization").second));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 5 parameter(s)
        // Parameters: MLCouplingData<In> input_data, MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out> output_data_before_postprocessing, MLCouplingData<Out> output_data, MLCouplingNormalization<In, Out>* normalization
        if (parameter.size() == 5) {
            try {
                std::cout << "Creating instance of MLCouplingApplicationTurbulenceClosure with parameters: " << "input_data=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second)) << ", ""input_data_after_preprocessing=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data_after_preprocessing").second)) << ", ""output_data_before_postprocessing=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data_before_postprocessing").second)) << ", ""output_data=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second)) << ", ""normalization=" << reinterpret_cast<MLCouplingNormalization<In, Out>*>(parameter.at("normalization").second) << std::endl;
                return new MLCouplingApplicationTurbulenceClosure<In, Out>(*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second), *reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data_after_preprocessing").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data_before_postprocessing").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second), reinterpret_cast<MLCouplingNormalization<In, Out>*>(parameter.at("normalization").second));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

