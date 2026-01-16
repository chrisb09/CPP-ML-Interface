#pragma once

#include <string>
#include <vector>

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
            // Parameters: std::string host, int port, int nodes, int tasks_per_node, int cpus_per_task, int gpus_per_task
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

