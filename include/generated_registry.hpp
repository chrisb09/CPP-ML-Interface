#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "provider/ml_coupling_provider.hpp" // MLCouplingProvider 
#include "normalization/ml_coupling_normalization.hpp" // MLCouplingNormalization 
#include "data_processor/ml_coupling_data_processor.hpp" // MLCouplingDataProcessor 
#include "behavior/ml_coupling_behavior.hpp" // MLCouplingBehavior 
#include "application/ml_coupling_application.hpp" // MLCouplingApplication 

// Includes for subclasses of MLCouplingProvider
#include "provider/ml_coupling_provider_aixelerate.hpp" // MLCouplingProviderAixelerate 
#include "provider/ml_coupling_provider_phydll.hpp" // MLCouplingProviderPhydll 
#include "provider/ml_coupling_provider_smartsim.hpp" // MLCouplingProviderSmartsim 

// Includes for subclasses of MLCouplingNormalization
#include "normalization/ml_coupling_minmax_normalization.hpp" // MLCouplingMinMaxNormalization 

// Includes for subclasses of MLCouplingDataProcessor
#include "data_processor/ml_coupling_data_processor_simple.hpp" // MLCouplingDataProcessorSimple 

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

// Lookup function for MLCouplingDataProcessor (data_processor)
// Maps registry names and aliases to actual class names
inline std::string resolve_data_processor_class_name(const std::string& name_or_alias) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"Simple", "MLCouplingDataProcessorSimple"},
        {"simple", "MLCouplingDataProcessorSimple"},
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

// Lookup function to resolve category names to base class names
// Maps category names (e.g., 'provider', 'behavior') to base class names
inline std::string resolve_category_to_base_class(const std::string& category) {
    static const std::unordered_map<std::string, std::string> lookup = {
        {"provider", "MLCouplingProvider"},
        {"normalization", "MLCouplingNormalization"},
        {"data_processor", "MLCouplingDataProcessor"},
        {"behavior", "MLCouplingBehavior"},
        {"application", "MLCouplingApplication"},
    };
    
    auto it = lookup.find(category);
    if (it != lookup.end()) {
        return it->second;
    }
    return category; // Return as-is if no mapping found
}

template<typename In, typename Out>
MLCouplingProvider<In, Out>* create_instance_mlcouplingprovider(const std::string &class_name, std::vector<void*> parameter) {

        if (class_name == "MLCouplingProviderAixelerate") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                try {
                    return new MLCouplingProviderAixelerate<In, Out>();
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        } else if (class_name == "MLCouplingProviderPhydll") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                try {
                    return new MLCouplingProviderPhydll<In, Out>();
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        } else if (class_name == "MLCouplingProviderSmartsim") {
            // Constructor with 6 parameter(s)
            // Parameters: std::string host = "localhost", int port = 6379, int nodes = 1, int tasks_per_node = 1, int cpus_per_task = 1, int gpus_per_task = 0
            if (parameter.size() == 6) {
                try {
                    return new MLCouplingProviderSmartsim<In, Out>(*reinterpret_cast<std::string*>(parameter[0]), *reinterpret_cast<int*>(parameter[1]), *reinterpret_cast<int*>(parameter[2]), *reinterpret_cast<int*>(parameter[3]), *reinterpret_cast<int*>(parameter[4]), *reinterpret_cast<int*>(parameter[5]));
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        }


    return nullptr;
}

template<typename In, typename Out>
MLCouplingProvider<In, Out>* create_instance_mlcouplingprovider(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_provider_class_name(class_name);

        if (resolved_class_name == "MLCouplingProviderAixelerate") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    return create_instance_mlcouplingprovider<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        } else if (resolved_class_name == "MLCouplingProviderPhydll") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    return create_instance_mlcouplingprovider<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        } else if (resolved_class_name == "MLCouplingProviderSmartsim") {
            // Constructor with 6 parameter(s)
            // Parameters: std::string host = "localhost", int port = 6379, int nodes = 1, int tasks_per_node = 1, int cpus_per_task = 1, int gpus_per_task = 0
            if (parameter.size() >= 0 && parameter.size() <= 6) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    if (parameter.find("host") != parameter.end()) {
                        params_vector.push_back(parameter.at("host"));
                    } else {
                        static std::string default_host = "localhost";
                        params_vector.push_back(&default_host);
                    }
                    if (parameter.find("port") != parameter.end()) {
                        params_vector.push_back(parameter.at("port"));
                    } else {
                        static int default_port = 6379;
                        params_vector.push_back(&default_port);
                    }
                    if (parameter.find("nodes") != parameter.end()) {
                        params_vector.push_back(parameter.at("nodes"));
                    } else {
                        static int default_nodes = 1;
                        params_vector.push_back(&default_nodes);
                    }
                    if (parameter.find("tasks_per_node") != parameter.end()) {
                        params_vector.push_back(parameter.at("tasks_per_node"));
                    } else {
                        static int default_tasks_per_node = 1;
                        params_vector.push_back(&default_tasks_per_node);
                    }
                    if (parameter.find("cpus_per_task") != parameter.end()) {
                        params_vector.push_back(parameter.at("cpus_per_task"));
                    } else {
                        static int default_cpus_per_task = 1;
                        params_vector.push_back(&default_cpus_per_task);
                    }
                    if (parameter.find("gpus_per_task") != parameter.end()) {
                        params_vector.push_back(parameter.at("gpus_per_task"));
                    } else {
                        static int default_gpus_per_task = 0;
                        params_vector.push_back(&default_gpus_per_task);
                    }
                    return create_instance_mlcouplingprovider<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        }
    return nullptr;
    }

template<typename In, typename Out>
MLCouplingNormalization<In, Out>* create_instance_mlcouplingnormalization(const std::string &class_name, std::vector<void*> parameter) {

        if (class_name == "MLCouplingMinMaxNormalization") {
            // Constructor with 4 parameter(s)
            // Parameters: In input_min, In input_max, Out output_min, Out output_max
            if (parameter.size() == 4) {
                try {
                    return new MLCouplingMinMaxNormalization<In, Out>(*reinterpret_cast<In*>(parameter[0]), *reinterpret_cast<In*>(parameter[1]), *reinterpret_cast<Out*>(parameter[2]), *reinterpret_cast<Out*>(parameter[3]));
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        }


    return nullptr;
}

template<typename In, typename Out>
MLCouplingNormalization<In, Out>* create_instance_mlcouplingnormalization(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_normalization_class_name(class_name);

        if (resolved_class_name == "MLCouplingMinMaxNormalization") {
            // Constructor with 4 parameter(s)
            // Parameters: In input_min, In input_max, Out output_min, Out output_max
            if (parameter.size() == 4) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    params_vector.push_back(parameter.at("input_min"));
                    params_vector.push_back(parameter.at("input_max"));
                    params_vector.push_back(parameter.at("output_min"));
                    params_vector.push_back(parameter.at("output_max"));
                    return create_instance_mlcouplingnormalization<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        }
    return nullptr;
    }

template<typename In, typename Out>
MLCouplingDataProcessor<In, Out>* create_instance_mlcouplingdataprocessor(const std::string &class_name, std::vector<void*> parameter) {

        if (class_name == "MLCouplingDataProcessorSimple") {
            // Constructor with 4 parameter(s)
            // Parameters: std::vector<In *> input_data, std::vector<std::vector<int>> input_data_dimensions, std::vector<Out *> output_data, std::vector<std::vector<int>> output_data_dimensions
            if (parameter.size() == 4) {
                try {
                    return new MLCouplingDataProcessorSimple<In, Out>(*reinterpret_cast<std::vector<In *>*>(parameter[0]), *reinterpret_cast<std::vector<std::vector<int>>*>(parameter[1]), *reinterpret_cast<std::vector<Out *>*>(parameter[2]), *reinterpret_cast<std::vector<std::vector<int>>*>(parameter[3]));
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        }


    return nullptr;
}

template<typename In, typename Out>
MLCouplingDataProcessor<In, Out>* create_instance_mlcouplingdataprocessor(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_data_processor_class_name(class_name);

        if (resolved_class_name == "MLCouplingDataProcessorSimple") {
            // Constructor with 4 parameter(s)
            // Parameters: std::vector<In *> input_data, std::vector<std::vector<int>> input_data_dimensions, std::vector<Out *> output_data, std::vector<std::vector<int>> output_data_dimensions
            if (parameter.size() == 4) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    params_vector.push_back(parameter.at("input_data"));
                    params_vector.push_back(parameter.at("input_data_dimensions"));
                    params_vector.push_back(parameter.at("output_data"));
                    params_vector.push_back(parameter.at("output_data_dimensions"));
                    return create_instance_mlcouplingdataprocessor<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        }
    return nullptr;
    }

MLCouplingBehavior* create_instance_mlcouplingbehavior(const std::string &class_name, std::vector<void*> parameter) {

        if (class_name == "MLCouplingBehaviorDefault") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                try {
                    return new MLCouplingBehaviorDefault();
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        } else if (class_name == "MLCouplingBehaviorPeriodic") {
            // Constructor with 4 parameter(s)
            // Parameters: int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference
            if (parameter.size() == 4) {
                try {
                    return new MLCouplingBehaviorPeriodic(*reinterpret_cast<int*>(parameter[0]), *reinterpret_cast<int*>(parameter[1]), *reinterpret_cast<int*>(parameter[2]), *reinterpret_cast<int*>(parameter[3]));
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        }


    return nullptr;
}

MLCouplingBehavior* create_instance_mlcouplingbehavior(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_behavior_class_name(class_name);

        if (resolved_class_name == "MLCouplingBehaviorDefault") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    return create_instance_mlcouplingbehavior(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        } else if (resolved_class_name == "MLCouplingBehaviorPeriodic") {
            // Constructor with 4 parameter(s)
            // Parameters: int inference_interval, int coupled_steps_before_inference, int coupled_steps_stride, int step_increment_after_inference
            if (parameter.size() == 4) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    params_vector.push_back(parameter.at("inference_interval"));
                    params_vector.push_back(parameter.at("coupled_steps_before_inference"));
                    params_vector.push_back(parameter.at("coupled_steps_stride"));
                    params_vector.push_back(parameter.at("step_increment_after_inference"));
                    return create_instance_mlcouplingbehavior(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        }
    return nullptr;
    }

template<typename In, typename Out>
MLCouplingApplication<In, Out>* create_instance_mlcouplingapplication(const std::string &class_name, std::vector<void*> parameter) {

        if (class_name == "MLCouplingApplicationTurbulenceClosure") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                try {
                    return new MLCouplingApplicationTurbulenceClosure<In, Out>();
                } catch (...) {
                    // Handle constructor exceptions if necessary
                }
            }
            return nullptr;
        }


    return nullptr;
}

template<typename In, typename Out>
MLCouplingApplication<In, Out>* create_instance_mlcouplingapplication(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {
    // Resolve name or alias to actual class name
    std::string resolved_class_name = resolve_application_class_name(class_name);

        if (resolved_class_name == "MLCouplingApplicationTurbulenceClosure") {
            // Constructor with 0 parameter(s)
            // Parameters: 
            if (parameter.size() == 0) {
                std::vector<void*> params_vector;
                try {
                    // Extract parameters from the map
                    return create_instance_mlcouplingapplication<In, Out>(class_name, params_vector);
                } catch (...) {
                    // Handle exceptions if necessary
                }
            }
            return nullptr;
        }
    return nullptr;
    }

