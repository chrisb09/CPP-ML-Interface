#pragma once

// This file is supposed to provide the definitions for the functions
// that allow us to read a configuration string (potentially from a file, but
// not necessarily) and set up the MLCoupling instance accordingly.

// This entails parsing the configuration, checking in which order the various
// components (normalization, data processor, behavior, application, provider)
// need to be created and initialized, and then creating and initializing them
// with the appropriate parameters obtained from the configuration.

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <any>
#include <cstring>
#include <algorithm>
#include <functional>
#include <tuple>

// Include toml++ for parsing TOML configurations
#include <toml++/toml.h>

#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior.hpp"
#include "normalization/ml_coupling_normalization.hpp"
#include "generated_registry.hpp"
#include "ml_coupling.hpp"

inline void print_failed_constructor(std::string class_name, std::unordered_map<std::string, std::pair<int, void*>> params, std::unordered_map<std::string, std::pair<int, void*>>& module_instances) {
    std::string resolved_class_name = resolve_class_name(class_name);
    std::cerr << "Failed to create instance of class: " << class_name << std::endl;
    std::cerr << "Resolved class name: " << resolved_class_name << std::endl;
    std::cerr << "Provided parameters:" << std::endl;

    // First, collect printable representations for each parameter so we can
    // compute the maximum left-column width and then align the type column.
    struct Entry { std::string left; std::string type_str; };
    std::vector<Entry> entries;
    entries.reserve(params.size());

    for (const auto& [param_name, param_value] : params) {
        std::string value_str;
        switch (param_value.first) {
            case 1: // int64_t
                value_str = std::to_string(*reinterpret_cast<int64_t*>(param_value.second));
                break;
            case 2: // double
                value_str = std::to_string(*reinterpret_cast<double*>(param_value.second));
                break;
            case 3: // std::string
                value_str = *reinterpret_cast<std::string*>(param_value.second);
                break;
            case 4: // bool
                value_str = (*reinterpret_cast<bool*>(param_value.second) ? "true" : "false");
                break;
            default:
                value_str = "Unknown type";
        }

        std::string type_str = (param_value.first == 1 ? "int64_t" : (param_value.first == 2 ? "double" : (param_value.first == 3 ? "std::string" : (param_value.first == 4 ? "bool" : "unknown"))));
        std::string left = "  " + param_name + " = " + value_str;
        entries.push_back({left, type_str});
    }

    // compute max left width
    size_t max_left = 0;
    for (const auto &e : entries) {
        if (e.left.size() > max_left) max_left = e.left.size();
    }

    // print with aligned type column
    for (const auto &e : entries) {
        std::cerr << e.left;
        size_t pad = 2; // spaces between value and type column
        if (max_left > e.left.size()) pad += (max_left - e.left.size());
        std::cerr << std::string(pad, ' ');
        std::cerr << "(config parsed type: " << e.type_str << ")" << std::endl;
    }
    std::cerr << "Available module instances that could be used as parameters:" << std::endl;
    for (const auto& [instance_name, instance_value] : module_instances) {
        std::cerr << "  " << instance_name << "" << std::endl;
    }
    print_constructor_help(resolved_class_name);
    std::cerr << std::endl << "Note: The config parsing uses types that may be automatically cast to fit the constructor parameters. For example, all integer values in the config are parsed as int64_t, but if the constructor expects an int, it will be cast accordingly. If there is a type mismatch that cannot be resolved, the constructor will fail." << std::endl;
}

template <typename In, typename Out>
inline void* create_mlcoupling_object(
    std::string module_classname, 
    std::unordered_map<std::string, std::pair<int, void*>> module_params, 
    std::unordered_map<std::string, std::pair<int, void*>>& module_instances, 
    std::function<void*(const std::string&, const std::unordered_map<std::string, std::pair<int, void*>>&)> create_instance_function) {

    std::vector<std::pair<std::string, std::string>> constructor_dependencies = get_constructor_dependencies(module_classname);

    for (const auto& [dependency_type, dependency_name] : constructor_dependencies) {
        std::vector<std::string> possible_class_names = std::vector(get_subclasses(dependency_type));
        possible_class_names.push_back(dependency_type);
        bool found_dependency = false;
        for (const auto& class_name : possible_class_names) {
            if (module_instances.count(class_name)) {
                module_params[dependency_name] = module_instances[class_name];
                found_dependency = true;
                break;
            }
        }
        if (!found_dependency) {
            std::cerr << "Error: Dependency " << dependency_name << " of type " << dependency_type << " required by module " << module_classname << " is not a recognized class name." << std::endl;
            std::cerr << "Searched for class names: ";
            for (const auto& class_name : possible_class_names) {
                std::cerr << class_name << " ";            }
            std::cerr << std::endl;
            return nullptr;
        }
    }

    std::cout << "Creating module instance of class: " << module_classname << " with " << module_params.size() << " parameters" << std::endl;
    return create_instance_function(module_classname, module_params);
}// Function to create and configure an MLCoupling instance based on a configuration string
template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str, MLCouplingData<In> input_data, MLCouplingData<Out> output_data) {
    try {
        toml::v3::table config = toml::parse(config_str);

        // Let's iterate through the configs "sections" or "categories" or whatever you want to call them (like [provider], [application], etc. but not hardcoded but all that is in the config)

        std::unordered_map<std::string, std::unordered_map<std::string, std::any>> sections;

        std::vector<int64_t> int64_t_params;
        std::vector<int> int_params;
        std::vector<double> double_params;
        std::vector<std::string> string_params;
        std::vector<bool> bool_params;

        std::vector<std::vector<std::any>> type_params;
        type_params.reserve(4);
        type_params.push_back(std::vector<std::any>());
        //type_params.push_back(int_params);
        type_params.push_back(std::vector<std::any>());
        type_params.push_back(std::vector<std::any>());
        type_params.push_back(std::vector<std::any>());

        std::unordered_map<std::string, std::unordered_map<std::string, std::pair<int, int>>> sections_temp_ids;

        /*
        * 0: nothing
        * 1: int64_t
        * 2: double
        * 3: std::string
        * 4: bool
        */

        config.for_each([&sections, &type_params, &sections_temp_ids](const toml::key& key, const toml::v3::node& node) {
            std::cout << "Section: " << key << std::endl;
            if (node.is_table()) {
                std::string key_str = std::string(key.str());
                sections[key_str] = {};
                const auto& table = *node.as_table();
                table.for_each([&sections, &key_str, &type_params, &sections_temp_ids](const toml::key& k, auto&& v) {
                    std::cout << "  " << k << " = ";
                    v.visit([](auto&& val) { std::cout << val; });
                    std::cout << std::endl;
                    // Extract raw C++ types from TOML values
                    // as<T>() returns pointer to wrapper, dereference to get the actual value via get()
                    std::string key = std::string(k.str());
                    if (v.is_integer()) {
                        sections[key_str][key] = std::any(v.template as<int64_t>()->get());
                        type_params[0].push_back(sections[key_str][key]);
                        sections_temp_ids[key_str][key] = std::make_pair(1, type_params[0].size() - 1);
                    } else if (v.is_floating_point()) {
                        sections[key_str][key] = std::any(v.template as<double>()->get());
                        type_params[1].push_back(sections[key_str][key]);
                        sections_temp_ids[key_str][key] = std::make_pair(2, type_params[1].size() - 1);
                    } else if (v.is_string()) {
                        sections[key_str][key] = std::any(std::string(v.template as<std::string>()->get()));
                        type_params[2].push_back(sections[key_str][key]);
                        sections_temp_ids[key_str][key] = std::make_pair(3, type_params[2].size() - 1);
                    } else if (v.is_boolean()) {
                        sections[key_str][key] = std::any(v.template as<bool>()->get());
                        type_params[3].push_back(sections[key_str][key]);
                        sections_temp_ids[key_str][key] = std::make_pair(4, type_params[3].size() - 1);
                    }
                });
            }
        });
        
        int64_t* int64_t_array = new int64_t[type_params[0].size()];
        double* double_array = new double[type_params[1].size()];
        
        // Calculate total string buffer size
        int string_array_size = 0;
        std::vector<int> string_array_offsets;
        for (size_t i = 0; i < type_params[2].size(); ++i) {
            string_array_offsets.push_back(string_array_size);
            string_array_size += std::any_cast<std::string>(type_params[2][i]).size() + 1;
        }
        
        // Allocate as char array for string storage
        char* string_array = new char[string_array_size];
        bool* bool_array = new bool[type_params[3].size()];

        for (size_t i = 0; i < type_params[0].size(); ++i) {
            int64_t_array[i] = std::any_cast<int64_t>(type_params[0][i]);
        }
        
        for (size_t i = 0; i < type_params[1].size(); ++i) {
            double_array[i] = std::any_cast<double>(type_params[1][i]);
        }
        
        // Copy strings into char buffer
        for (size_t i = 0; i < type_params[2].size(); ++i) {
            const std::string& str = std::any_cast<std::string>(type_params[2][i]);
            std::strcpy(string_array + string_array_offsets[i], str.c_str());
        }
        
        for (size_t i = 0; i < type_params[3].size(); ++i) {
            bool_array[i] = std::any_cast<bool>(type_params[3][i]);
        }

        std::unordered_map<std::string, std::pair<int, void*>> normalization_params;
        std::unordered_map<std::string, std::pair<int, void*>> provider_params;
        std::unordered_map<std::string, std::pair<int, void*>> behavior_params;
        std::unordered_map<std::string, std::pair<int, void*>> application_params;

        std::string normalization_class_name;
        std::string provider_class_name;
        std::string behavior_class_name;
        std::string application_class_name;

        for (const auto &[section, values] : sections_temp_ids) {
            std::cout << "Section: " << section << std::endl;
            std::unordered_map<std::string, std::pair<int, void*>> *params_map = nullptr;
            if (section == "normalization") {
                params_map = &normalization_params;
            } else if (section == "provider") {
                params_map = &provider_params;
            } else if (section == "behavior") {
                params_map = &behavior_params;
            } else if (section == "application") {
                params_map = &application_params;
            }
            for (const auto &[key, id_pair] : values) {
                int type_index = id_pair.first;
                int index_in_type = id_pair.second;
                std::cout << "  Key: " << key << " -> Type index: " << type_index << ", Index in type: " << index_in_type;
                bool cont = false;
                if (key == "class") {
                    if (type_index == 3) { // string
                        cont = true;
                        if (section == "normalization") {
                            normalization_class_name = std::any_cast<std::string>(type_params[2][index_in_type]);
                        } else if (section == "provider") {
                            provider_class_name = std::any_cast<std::string>(type_params[2][index_in_type]);
                        } else if (section == "behavior") {
                            behavior_class_name = std::any_cast<std::string>(type_params[2][index_in_type]);
                        } else if (section == "application") {
                            application_class_name = std::any_cast<std::string>(type_params[2][index_in_type]);
                        } else {
                            cont = false;
                        }
                    }
                }
                if (cont) {
                    std::cout << std::endl;
                    continue;
                }
                // Print the value based on type
                if (type_index == 0) {
                    std::cout << " -> Value: " << "different type (0)";
                } else
                if (type_index == 1) {
                    std::cout << " -> Value: " << int64_t_array[index_in_type];
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(1, &int64_t_array[index_in_type]);
                    }
                } else if (type_index == 2) {
                    std::cout << " -> Value: " << double_array[index_in_type];
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(2, &double_array[index_in_type]);
                    }
                } else if (type_index == 3) {
                    std::cout << " -> Value: " << (string_array + string_array_offsets[index_in_type]);
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(3, string_array + string_array_offsets[index_in_type]);
                    }
                } else if (type_index == 4) {
                    std::cout << " -> Value: " << bool_array[index_in_type];
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(4, &bool_array[index_in_type]);
                    }
                }
                std::cout << std::endl;
            }
        }

        MLCouplingNormalization<In, Out>* normalization = nullptr;
        MLCouplingProvider<In, Out>* provider = nullptr;
        MLCouplingBehavior* behavior = nullptr;
        MLCouplingApplication<In, Out>* application = nullptr;
        MLCoupling<In, Out>* coupling = nullptr;

        std::unordered_map<std::string, std::pair<int, void*>> module_instances;

        if (!normalization_class_name.empty()) {
            normalization = static_cast<MLCouplingNormalization<In, Out>*>(create_mlcoupling_object<In, Out>(normalization_class_name, normalization_params, module_instances, create_instance_mlcouplingnormalization<In, Out>));
            if (normalization) {
                module_instances[resolve_normalization_class_name(normalization_class_name)] = std::make_pair(-1, static_cast<void*>(normalization));
            } else {
                std::cerr << "Error: Failed to create normalization instance of class: " << normalization_class_name << std::endl;
                print_failed_constructor(normalization_class_name, normalization_params, module_instances);
                exit(1);
            }
            In dummy_double_0 = -0.5; // Just a dummy variable to demonstrate passing parameters
            In dummy_double_1 = 0.5;
            std::vector<int> dummy_dimensions = std::vector<int>{2};
            MLCouplingData<In> dummy_input_data = MLCouplingData<In>(std::vector<In*>{&dummy_double_0, &dummy_double_1}, std::vector<std::vector<int>>{dummy_dimensions});
            std::cout << "Normalization: " << *normalization << std::endl;
            std::cout << "Dummy value before normalization: " << dummy_double_0 << ", " << dummy_double_1 << std::endl;
            normalization->normalize_input(dummy_input_data);
            std::cout << "Dummy value after normalization: " << dummy_double_0 << ", " << dummy_double_1 << std::endl;
        }

        if (provider_class_name.empty()) {
            std::cerr << "Error: No provider class specified in configuration." << std::endl;
            return nullptr;
        } else {
            provider = static_cast<MLCouplingProvider<In, Out>*>(create_mlcoupling_object<In, Out>(provider_class_name, provider_params, module_instances, create_instance_mlcouplingprovider<In, Out>));
            if (provider) {
                module_instances[resolve_provider_class_name(provider_class_name)] = std::make_pair(-1, static_cast<void*>(provider));
            } else {
                std::cerr << "Error: Failed to create provider instance of class: " << provider_class_name << std::endl;
                print_failed_constructor(provider_class_name, provider_params, module_instances);
                exit(1);
            }
        }

        if (behavior_class_name.empty()) {
            std::cerr << "Error: No behavior class specified in configuration." << std::endl;
            std::cout << "Proceeding without behavior (defaulting to always perform inference every step)." << std::endl;
            behavior = new MLCouplingBehaviorDefault();
            module_instances[resolve_behavior_class_name("MLCouplingBehaviorDefault")] = std::make_pair(-1, static_cast<void*>(behavior));
        } else {
            behavior = static_cast<MLCouplingBehavior*>(create_mlcoupling_object<In, Out>(behavior_class_name, behavior_params, module_instances, create_instance_mlcouplingbehavior));
            if (behavior) {
                module_instances[resolve_behavior_class_name(behavior_class_name)] = std::make_pair(-1, static_cast<void*>(behavior));
            } else {
                std::cerr << "Error: Failed to create behavior instance of class: " << behavior_class_name << std::endl;
                print_failed_constructor(behavior_class_name, behavior_params, module_instances);
                exit(1);
            }
        }

        if (application_class_name.empty()) {
            std::cerr << "Error: No application class specified in configuration." << std::endl;
            exit(1);
        } else {
            application_params["input_data"] = std::make_pair(0, static_cast<void*>(&input_data));
            application_params["output_data"] = std::make_pair(0, static_cast<void*>(&output_data));
            application = static_cast<MLCouplingApplication<In, Out>*>(create_mlcoupling_object<In, Out>(application_class_name, application_params, module_instances, create_instance_mlcouplingapplication<In, Out>));
            if (application) {
                module_instances[resolve_application_class_name(application_class_name)] = std::make_pair(0, static_cast<void*>(application));
            } else {
                std::cerr << "Error: Failed to create application instance of class: " << application_class_name << std::endl;
                print_failed_constructor(application_class_name, application_params, module_instances);
                return nullptr;
            }
        }

        if (normalization) {
            std::cout << "Created normalization instance: " << *normalization << " of type " << get_type_name(*normalization) << std::endl;
        }
        if (provider) {
            std::cout << "Created provider instance at " << provider << " of type " << get_type_name(*provider) << std::endl;
        }
        if (behavior) {
            std::cout << "Created behavior instance at " << behavior << " of type " << get_type_name(*behavior) << std::endl;
        }
        if (application) {
            std::cout << "Created application instance at " << application << " of type " << get_type_name(*application) << std::endl;
        }

        return new MLCoupling<In, Out>(provider, application, behavior);

    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed: " << err << std::endl;
        std::cerr << "Please check the configuration format and try again." << std::endl;
        exit(1);
    }

    return nullptr;
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path, MLCouplingData<In> input_data, MLCouplingData<Out> output_data) {
    // Open and read the file content
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file_path);
    }
    std::string config_str((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    return create_mlcoupling_from_config<In, Out>(config_str, input_data, output_data);
}