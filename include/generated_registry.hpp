#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <typeinfo>

#include "provider/ml_coupling_provider.hpp" // MLCouplingProvider 
#include "normalization/ml_coupling_normalization.hpp" // MLCouplingNormalization 
#include "behavior/ml_coupling_behavior.hpp" // MLCouplingBehavior 
#include "application/ml_coupling_application.hpp" // MLCouplingApplication 

// Includes for subclasses of MLCouplingProvider
#include "provider/ml_coupling_provider_aixelerate.hpp" // MLCouplingProviderAixelerate 
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
        {"Aixelerate", "MLCouplingProviderAixelerate"},
        {"aixelerate", "MLCouplingProviderAixelerate"},
        {"AIxelerate", "MLCouplingProviderAixelerate"},
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
    resolved = resolve_behavior_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_provider_class_name(name_or_alias);
    if (resolved != name_or_alias) {
        return resolved;
    }
    resolved = resolve_normalization_class_name(name_or_alias);
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

    if (class_name == "MLCouplingProviderAixelerate") {
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

    if (class_name == "MLCouplingProviderAixelerate") {
        signatures.push_back("MLCouplingProviderAixelerate()");
        return signatures;
    }

    if (class_name == "MLCouplingProviderPhydll") {
        signatures.push_back("MLCouplingProviderPhydll()");
        return signatures;
    }

    if (class_name == "MLCouplingProviderSmartsim") {
        signatures.push_back("MLCouplingProviderSmartsim(std::string host = \"localhost\", int port = 6379, int nodes = 1, int tasks_per_node = 1, int cpus_per_task = 1, int gpus_per_task = 0)");
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
        signatures.push_back("MLCouplingBehaviorPeriodic(int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference)");
        return signatures;
    }

    if (class_name == "MLCouplingApplicationTurbulenceClosure") {
        signatures.push_back("MLCouplingApplicationTurbulenceClosure(MLCouplingData<In> input_data, MLCouplingData<Out> output_data, MLCouplingNormalization<In, Out>* normalization)");
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
        subclasses.push_back("MLCouplingProviderAixelerate");
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
        {"MLCouplingProviderAixelerate", "MLCouplingProvider"},
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
    if (typeid(*obj) == typeid(MLCouplingProviderAixelerate<In, Out>)) return "MLCouplingProviderAixelerate";
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

// Helper to extract and cast a config parameter based on its runtime type tag.
// Type tags: 0 = no static cast, 1 = int64_t, 2 = double, 3 = std::string (char*), 4 = bool
template<typename T>
T config_param_cast(const std::pair<int, void*>& param) {
    switch (param.first) {
        case 0: return *reinterpret_cast<T*>(param.second); // No static cast, just reinterpret
        case 1: return static_cast<T>(*reinterpret_cast<int64_t*>(param.second));
        case 2: return static_cast<T>(*reinterpret_cast<double*>(param.second));
        case 4: return static_cast<T>(*reinterpret_cast<bool*>(param.second));
        default: throw std::runtime_error("Unsupported type tag for numeric cast: " + std::to_string(param.first));
    }
}

// Specialization for std::string
template<>
inline std::string config_param_cast<std::string>(const std::pair<int, void*>& param) {
    if (param.first == 3) return std::string(reinterpret_cast<char*>(param.second));
    throw std::runtime_error("Expected string (type tag 3), got: " + std::to_string(param.first));
}

template<typename In, typename Out>
MLCouplingProvider<In, Out>* create_instance_mlcouplingprovider(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_provider_class_name(class_name);

    if (resolved_class_name == "MLCouplingProviderAixelerate") {
        // Constructor with 0 parameter(s)
        // Parameters: 
        if (parameter.size() == 0) {
            try {
                std::cout << "Creating instance of MLCouplingProviderAixelerate with parameters: " << std::endl;
                return new MLCouplingProviderAixelerate<In, Out>();
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
        // Constructor with 6 parameter(s)
        // Parameters: std::string host = "localhost", int port = 6379, int nodes = 1, int tasks_per_node = 1, int cpus_per_task = 1, int gpus_per_task = 0
        if (parameter.size() >= 0 && parameter.size() <= 6) {
            try {
                std::cout << "Creating instance of MLCouplingProviderSmartsim with parameters: " << "host=" << (parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost") << ", ""port=" << (parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379) << ", ""nodes=" << (parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1) << ", ""tasks_per_node=" << (parameter.find("tasks_per_node") != parameter.end() ? config_param_cast<int>(parameter.at("tasks_per_node")) : (int)1) << ", ""cpus_per_task=" << (parameter.find("cpus_per_task") != parameter.end() ? config_param_cast<int>(parameter.at("cpus_per_task")) : (int)1) << ", ""gpus_per_task=" << (parameter.find("gpus_per_task") != parameter.end() ? config_param_cast<int>(parameter.at("gpus_per_task")) : (int)0) << std::endl;
                return new MLCouplingProviderSmartsim<In, Out>(parameter.find("host") != parameter.end() ? config_param_cast<std::string>(parameter.at("host")) : (std::string)"localhost", parameter.find("port") != parameter.end() ? config_param_cast<int>(parameter.at("port")) : (int)6379, parameter.find("nodes") != parameter.end() ? config_param_cast<int>(parameter.at("nodes")) : (int)1, parameter.find("tasks_per_node") != parameter.end() ? config_param_cast<int>(parameter.at("tasks_per_node")) : (int)1, parameter.find("cpus_per_task") != parameter.end() ? config_param_cast<int>(parameter.at("cpus_per_task")) : (int)1, parameter.find("gpus_per_task") != parameter.end() ? config_param_cast<int>(parameter.at("gpus_per_task")) : (int)0);
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
                std::cout << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_min=" << config_param_cast<In>(parameter.at("input_min")) << ", ""input_max=" << config_param_cast<In>(parameter.at("input_max")) << ", ""output_min=" << config_param_cast<Out>(parameter.at("output_min")) << ", ""output_max=" << config_param_cast<Out>(parameter.at("output_max")) << std::endl;
                return new MLCouplingMinMaxNormalization<In, Out>(config_param_cast<In>(parameter.at("input_min")), config_param_cast<In>(parameter.at("input_max")), config_param_cast<Out>(parameter.at("output_min")), config_param_cast<Out>(parameter.at("output_max")));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        // Constructor with 4 parameter(s)
        // Parameters: In* input_data, int input_data_size, Out* output_data, int output_data_size
        if (parameter.size() == 4) {
            try {
                std::cout << "Creating instance of MLCouplingMinMaxNormalization with parameters: " << "input_data=" << (*reinterpret_cast<In **>(parameter.at("input_data").second)) << ", ""input_data_size=" << config_param_cast<int>(parameter.at("input_data_size")) << ", ""output_data=" << (*reinterpret_cast<Out **>(parameter.at("output_data").second)) << ", ""output_data_size=" << config_param_cast<int>(parameter.at("output_data_size")) << std::endl;
                return new MLCouplingMinMaxNormalization<In, Out>(*reinterpret_cast<In **>(parameter.at("input_data").second), config_param_cast<int>(parameter.at("input_data_size")), *reinterpret_cast<Out **>(parameter.at("output_data").second), config_param_cast<int>(parameter.at("output_data_size")));
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
        // Constructor with 4 parameter(s)
        // Parameters: int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference
        if (parameter.size() == 4) {
            try {
                std::cout << "Creating instance of MLCouplingBehaviorPeriodic with parameters: " << "inference_interval=" << config_param_cast<int>(parameter.at("inference_interval")) << ", ""coupled_steps_before_inference=" << config_param_cast<int>(parameter.at("coupled_steps_before_inference")) << ", ""coupled_steps_stride=" << config_param_cast<int>(parameter.at("coupled_steps_stride")) << ", ""step_increment_after_inference=" << config_param_cast<int>(parameter.at("step_increment_after_inference")) << std::endl;
                return new MLCouplingBehaviorPeriodic(config_param_cast<int>(parameter.at("inference_interval")), config_param_cast<int>(parameter.at("coupled_steps_before_inference")), config_param_cast<int>(parameter.at("coupled_steps_stride")), config_param_cast<int>(parameter.at("step_increment_after_inference")));
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
                std::cout << "Creating instance of MLCouplingApplicationTurbulenceClosure with parameters: " << "input_data=" << (*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second)) << ", ""output_data=" << (*reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second)) << ", ""normalization=" << (*reinterpret_cast<MLCouplingNormalization<In, Out> **>(parameter.at("normalization").second)) << std::endl;
                return new MLCouplingApplicationTurbulenceClosure<In, Out>(*reinterpret_cast<MLCouplingData<In>*>(parameter.at("input_data").second), *reinterpret_cast<MLCouplingData<Out>*>(parameter.at("output_data").second), *reinterpret_cast<MLCouplingNormalization<In, Out> **>(parameter.at("normalization").second));
            } catch (...) {
                // Handle exceptions if necessary
            }
        }
        return nullptr;
    }
    return nullptr;
}

