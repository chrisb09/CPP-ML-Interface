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
#include <sstream>
#include <tuple>
#include <optional>

// Include toml++ for parsing TOML configurations
#include <toml++/toml.h>

#include "config_overrides.hpp"

#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior.hpp"
#include "normalization/ml_coupling_normalization.hpp"
#include "generated_registry.hpp"
#include "logging.hpp"


inline void print_failed_constructor(std::string class_name, std::unordered_map<std::string, std::pair<int, void*>> params, std::unordered_map<std::string, std::pair<int, void*>>& module_instances) {
    std::string resolved_class_name = resolve_class_name(class_name);
    logging::error("Failed to create instance of class: " + class_name);
    logging::error("Resolved class name: " + resolved_class_name);
    logging::error("Provided parameters:");
    // First, collect printable representations for each parameter so we can
    // compute the maximum left-column width and then align the type column.
    struct Entry { std::string left; std::string type_str; };
    std::vector<Entry> entries;
    entries.reserve(params.size());

    for (const auto& [param_name, param_value] : params) {
        std::string value_str;
        try {
            switch (param_value.first) {
                case 1: // int64_t
                    value_str = std::to_string(*reinterpret_cast<int64_t*>(param_value.second));
                    break;
                case 2: // double
                    value_str = std::to_string(*reinterpret_cast<double*>(param_value.second));
                    break;
                case 3: // std::string
                    value_str = std::string(reinterpret_cast<char*>(param_value.second));
                    break;
                case 4: // bool
                    value_str = (*reinterpret_cast<bool*>(param_value.second) ? "true" : "false");
                    break;
                case 5: { // std::vector<std::string>
                    const auto& values = *reinterpret_cast<std::vector<std::string>*>(param_value.second);
                    value_str = "[";
                    for (size_t i = 0; i < values.size(); ++i) {
                        if (i > 0) value_str += ", ";
                        value_str += values[i];
                    }
                    value_str += "]";
                    break;
                }
                default:
                    value_str = "Unknown type";
            }
        } catch (const std::exception& e) {
            std::string val_type = (param_value.first == 1 ? "int64_t" : (param_value.first == 2 ? "double" : (param_value.first == 3 ? "std::string" : (param_value.first == 4 ? "bool" : (param_value.first == 5 ? "std::vector<std::string>" : "unknown")))));
            logging::error("Error retrieving value for parameter " + param_name + " which is of type " + val_type + ". This might different than the constructor expected. Error: " + e.what());
            return;
        } catch (...) {
            value_str = "Unknown error retrieving value";
            return;
        }

        std::string type_str = (param_value.first == 1 ? "int64_t" : (param_value.first == 2 ? "double" : (param_value.first == 3 ? "std::string" : (param_value.first == 4 ? "bool" : (param_value.first == 5 ? "std::vector<std::string>" : "unknown")))));
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
        size_t pad = 2; // spaces between value and type column
        if (max_left > e.left.size()) pad += (max_left - e.left.size());
        logging::error(e.left + std::string(pad, ' ') + "(config parsed type: " + e.type_str + ")");
    }
    logging::error("Available module instances that could be used as parameters:");
    for (const auto& [instance_name, instance_value] : module_instances) {
        logging::error("  " + instance_name);
    }
    print_constructor_help(resolved_class_name);
    logging::error("Note: The config parsing uses types that may be automatically cast to fit the constructor parameters. For example, all integer values in the config are parsed as int64_t, but if the constructor expects an int, it will be cast accordingly. If there is a type mismatch that cannot be resolved, the constructor will fail.");
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
            logging::error("Dependency " + dependency_name + " of type " + dependency_type + " required by module " + module_classname + " is not a recognized class name.");
            std::string searched = "Searched for class names:";
            for (const auto& class_name : possible_class_names) {
                searched += " " + class_name;
            }
            logging::error(searched);
            return nullptr;
        }
    }

    logging::info("Creating module instance of class: " + module_classname + " with " + std::to_string(module_params.size()) + " parameters");
    return create_instance_function(module_classname, module_params);
}

struct ConfigCastModeGuard {
    ConfigCastMode previous_mode;

    explicit ConfigCastModeGuard(ConfigCastMode new_mode)
        : previous_mode(get_config_cast_mode()) {
        set_config_cast_mode(new_mode);
    }

    ~ConfigCastModeGuard() {
        set_config_cast_mode(previous_mode);
    }
};

inline std::optional<std::pair<std::string, std::string>> split_dotted_override_key(const std::string& dotted_key) {
    const auto dot_pos = dotted_key.find('.');
    if (dot_pos == std::string::npos || dot_pos == 0 || dot_pos == dotted_key.size() - 1) {
        return std::nullopt;
    }
    return std::make_pair(dotted_key.substr(0, dot_pos), dotted_key.substr(dot_pos + 1));
}

inline ConfigSectionOverrides normalize_config_overrides(const ConfigOverrides& overrides) {
    ConfigSectionOverrides normalized = overrides.sections;

    for (const auto& [dotted_key, value] : overrides.dotted) {
        auto split_key = split_dotted_override_key(dotted_key);
        if (!split_key.has_value()) {
            logging::warning("Ignoring invalid dotted override key: '" + dotted_key + "'. Expected format 'section.key'.");
            continue;
        }
        const auto& [section, key] = *split_key;
        normalized[section][key] = value;
    }

    return normalized;
}

inline void upsert_override_value(
    const std::string& section,
    const std::string& key,
    const ConfigOverrideValue& value,
    std::vector<std::vector<std::any>>& type_params,
    std::unordered_map<std::string, std::unordered_map<std::string, std::pair<int, int>>>& sections_temp_ids) {

    std::visit([&](const auto& typed_value) {
        using ValueType = std::decay_t<decltype(typed_value)>;
        if constexpr (std::is_same_v<ValueType, int64_t>) {
            type_params[0].push_back(std::any(typed_value));
            sections_temp_ids[section][key] = std::make_pair(1, static_cast<int>(type_params[0].size() - 1));
        } else if constexpr (std::is_same_v<ValueType, double>) {
            type_params[1].push_back(std::any(typed_value));
            sections_temp_ids[section][key] = std::make_pair(2, static_cast<int>(type_params[1].size() - 1));
        } else if constexpr (std::is_same_v<ValueType, std::string>) {
            type_params[2].push_back(std::any(typed_value));
            sections_temp_ids[section][key] = std::make_pair(3, static_cast<int>(type_params[2].size() - 1));
        } else if constexpr (std::is_same_v<ValueType, bool>) {
            type_params[3].push_back(std::any(typed_value));
            sections_temp_ids[section][key] = std::make_pair(4, static_cast<int>(type_params[3].size() - 1));
        } else if constexpr (std::is_same_v<ValueType, std::vector<std::string>>) {
            type_params[4].push_back(std::any(typed_value));
            sections_temp_ids[section][key] = std::make_pair(5, static_cast<int>(type_params[4].size() - 1));
        }
    }, value);
}


// Function to create and configure an MLCoupling instance based on a configuration string
template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_impl(const std::string &config_str,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        MLCouplingData<In> input_data_after_preprocessing,
                                                        MLCouplingData<Out> output_data_before_postprocessing,
                                                        bool allow_two_buffer_application_fallback,
                                                        ConfigCastMode cast_mode,
                                                        const ConfigOverrides& overrides) {
    ConfigCastModeGuard cast_mode_guard(cast_mode);
    try {
        toml::v3::table config = toml::parse(config_str);

        // Let's iterate through the configs "sections" or "categories" or whatever you want to call them (like [provider], [application], etc. but not hardcoded but all that is in the config)

        std::unordered_map<std::string, std::unordered_map<std::string, std::any>> sections;

        std::vector<int64_t> int64_t_params;
        std::vector<int> int_params;
        std::vector<double> double_params;
        std::vector<std::string> string_params;
        std::vector<bool> bool_params;
        std::vector<std::vector<std::string>> vector_string_params;

        std::vector<std::vector<std::any>> type_params;
        type_params.reserve(5);
        type_params.push_back(std::vector<std::any>());
        //type_params.push_back(int_params);
        type_params.push_back(std::vector<std::any>());
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
        * 5: std::vector<std::string>
        */

        config.for_each([&sections, &type_params, &sections_temp_ids](const toml::key& key, const toml::v3::node& node) {
            logging::debug("Section: " + std::string(key.str()));
            if (node.is_table()) {
                std::string key_str = std::string(key.str());
                sections[key_str] = {};
                const auto& table = *node.as_table();
                table.for_each([&sections, &key_str, &type_params, &sections_temp_ids](const toml::key& k, auto&& v) {
                    std::ostringstream val_stream;
                    v.visit([&val_stream](auto&& val) { val_stream << val; });
                    logging::debug("  " + std::string(k.str()) + " = " + val_stream.str() + " " + typeid(v).name());
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
                    } else if (v.is_array()) {
                        const auto* arr = v.as_array();
                        if (arr) {
                            bool all_strings = true;
                            std::vector<std::string> values;
                            values.reserve(arr->size());
                            for (const auto& elem : *arr) {
                                if (!elem.is_string()) {
                                    all_strings = false;
                                    break;
                                }
                                values.push_back(std::string(elem.as_string()->get()));
                            }
                            if (all_strings) {
                                sections[key_str][key] = std::any(values);
                                type_params[4].push_back(sections[key_str][key]);
                                sections_temp_ids[key_str][key] = std::make_pair(5, type_params[4].size() - 1);
                            } else {
                                logging::warning("Array parameter '" + key + "' in section '" + key_str + "' is not a homogeneous string array and will be ignored.");
                            }
                        }
                    }
                });
            }
        });

        const ConfigSectionOverrides normalized_overrides = normalize_config_overrides(overrides);
        for (const auto& [section, values] : normalized_overrides) {
            for (const auto& [key, value] : values) {
                upsert_override_value(section, key, value, type_params, sections_temp_ids);
                logging::debug("Applied config override: " + section + "." + key);
            }
        }
        
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
        std::vector<std::vector<std::string>> vector_string_array(type_params[4].size());

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

        for (size_t i = 0; i < type_params[4].size(); ++i) {
            vector_string_array[i] = std::any_cast<std::vector<std::string>>(type_params[4][i]);
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
            logging::debug("Section: " + section);
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
                std::string message = "  Key: " + key + " -> Type index: " + std::to_string(type_index) + ", Index in type: " + std::to_string(index_in_type);
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
                    logging::debug(message);
                    continue;
                }
                // Print the value based on type
                if (type_index == 0) {
                    message += " -> Value: different type (0)";
                } else
                if (type_index == 1) {
                    message += " -> Value: " + std::to_string(int64_t_array[index_in_type]);
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(1, &int64_t_array[index_in_type]);
                    }
                } else if (type_index == 2) {
                    message += " -> Value: " + std::to_string(double_array[index_in_type]);
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(2, &double_array[index_in_type]);
                    }
                } else if (type_index == 3) {
                    message += " -> Value: ";
                    message += (string_array + string_array_offsets[index_in_type]);
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(3, string_array + string_array_offsets[index_in_type]);
                    }
                } else if (type_index == 4) {
                    message += " -> Value: ";
                    message += (bool_array[index_in_type] ? "true" : "false");
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(4, &bool_array[index_in_type]);
                    }
                } else if (type_index == 5) {
                    message += " -> Value: [";
                    for (size_t idx = 0; idx < vector_string_array[index_in_type].size(); ++idx) {
                        if (idx > 0) message += ", ";
                        message += vector_string_array[index_in_type][idx];
                    }
                    message += "]";
                    if (params_map) {
                        (*params_map)[key] = std::pair<int, void*>(5, &vector_string_array[index_in_type]);
                    }
                }
                logging::debug(message);
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
                logging::error("Failed to create normalization instance of class: " + normalization_class_name);
                print_failed_constructor(normalization_class_name, normalization_params, module_instances);
                exit(1);
            }
            In dummy_values[2] = {static_cast<In>(-0.5), static_cast<In>(0.5)};
            std::vector<int> dummy_dimensions = std::vector<int>{2};
            MLCouplingData<In> dummy_input_data =
                MLCouplingData<In>(std::vector<MLCouplingTensor<In>>{
                    MLCouplingTensor<In>::wrap_flat(dummy_values, dummy_dimensions)
                });
            std::ostringstream norm_stream;
            norm_stream << *normalization;
            logging::debug("Normalization: " + norm_stream.str());
            std::ostringstream before_stream;
            before_stream << dummy_values[0] << ", " << dummy_values[1];
            logging::debug("Dummy value before normalization: " + before_stream.str());
            normalization->normalize_input(dummy_input_data);
            std::ostringstream after_stream;
            after_stream << dummy_values[0] << ", " << dummy_values[1];
            logging::debug("Dummy value after normalization: " + after_stream.str());
        }

        if (provider_class_name.empty()) {
            logging::error("No provider class specified in configuration.");
            return nullptr;
        } else {
            provider = static_cast<MLCouplingProvider<In, Out>*>(create_mlcoupling_object<In, Out>(provider_class_name, provider_params, module_instances, create_instance_mlcouplingprovider<In, Out>));
            if (provider) {
                module_instances[resolve_provider_class_name(provider_class_name)] = std::make_pair(-1, static_cast<void*>(provider));
            } else {
                logging::error("Failed to create provider instance of class: " + provider_class_name);
                print_failed_constructor(provider_class_name, provider_params, module_instances);
                exit(1);
            }
        }

        if (behavior_class_name.empty()) {
            logging::warning("No behavior class specified in configuration.");
            logging::info("Proceeding without behavior (defaulting to always perform inference every step).");
            behavior = new MLCouplingBehaviorDefault();
            module_instances[resolve_behavior_class_name("MLCouplingBehaviorDefault")] = std::make_pair(-1, static_cast<void*>(behavior));
        } else {
            behavior = static_cast<MLCouplingBehavior*>(create_mlcoupling_object<In, Out>(behavior_class_name, behavior_params, module_instances, create_instance_mlcouplingbehavior));
            if (behavior) {
                module_instances[resolve_behavior_class_name(behavior_class_name)] = std::make_pair(-1, static_cast<void*>(behavior));
            } else {
                logging::error("Failed to create behavior instance of class: " + behavior_class_name);
                print_failed_constructor(behavior_class_name, behavior_params, module_instances);
                exit(1);
            }
        }

        if (application_class_name.empty()) {
            logging::error("No application class specified in configuration.");
            exit(1);
        } else {
            application_params["input_data"] = std::make_pair(0, static_cast<void*>(&input_data));
            application_params["output_data"] = std::make_pair(0, static_cast<void*>(&output_data));
            application_params["input_data_after_preprocessing"] = std::make_pair(0, static_cast<void*>(&input_data_after_preprocessing));
            application_params["output_data_before_postprocessing"] = std::make_pair(0, static_cast<void*>(&output_data_before_postprocessing));
            application = static_cast<MLCouplingApplication<In, Out>*>(create_mlcoupling_object<In, Out>(application_class_name, application_params, module_instances, create_instance_mlcouplingapplication<In, Out>));
            if (application == nullptr && allow_two_buffer_application_fallback) {
                application_params.erase("input_data_after_preprocessing");
                application_params.erase("output_data_before_postprocessing");
                application = static_cast<MLCouplingApplication<In, Out>*>(create_mlcoupling_object<In, Out>(application_class_name, application_params, module_instances, create_instance_mlcouplingapplication<In, Out>));
            }
            if (application) {
                module_instances[resolve_application_class_name(application_class_name)] = std::make_pair(0, static_cast<void*>(application));
            } else {
                logging::error("Failed to create application instance of class: " + application_class_name);
                print_failed_constructor(application_class_name, application_params, module_instances);
                return nullptr;
            }
        }

        if (normalization) {
            std::ostringstream normalization_stream;
            normalization_stream << *normalization;
            logging::info("Created normalization instance: " + normalization_stream.str() + " of type " + get_type_name(*normalization));
        }
        if (provider) {
            std::ostringstream provider_ptr_stream;
            provider_ptr_stream << provider;
            logging::info("Created provider instance at " + provider_ptr_stream.str() + " of type " + get_type_name(*provider));
        }
        if (behavior) {
            std::ostringstream behavior_ptr_stream;
            behavior_ptr_stream << behavior;
            logging::info("Created behavior instance at " + behavior_ptr_stream.str() + " of type " + get_type_name(*behavior));
        }
        if (application) {
            std::ostringstream application_ptr_stream;
            application_ptr_stream << application;
            logging::info("Created application instance at " + application_ptr_stream.str() + " of type " + get_type_name(*application));
        }

        return new MLCoupling<In, Out>(provider, application, behavior);

    } catch (const toml::parse_error& err) {
        std::ostringstream err_stream;
        err_stream << err;
        logging::error("Parsing failed: " + err_stream.str());
        logging::error("Please check the configuration format and try again.");
        exit(1);
    }

    return nullptr;
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str, MLCouplingData<In> input_data, MLCouplingData<Out> output_data) {
    return create_mlcoupling_from_config(config_str, std::move(input_data), std::move(output_data), ConfigCastMode::Relaxed);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   ConfigCastMode cast_mode) {
    MLCouplingData<In> input_data_after_preprocessing = input_data;
    MLCouplingData<Out> output_data_before_postprocessing = output_data;
    return create_mlcoupling_from_config_impl<In, Out>(config_str,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       true,
                                                       cast_mode,
                                                       ConfigOverrides{});
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   const ConfigOverrides& overrides) {
    return create_mlcoupling_from_config(config_str,
                                         std::move(input_data),
                                         std::move(output_data),
                                         ConfigCastMode::Relaxed,
                                         overrides);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   ConfigCastMode cast_mode,
                                                   const ConfigOverrides& overrides) {
    MLCouplingData<In> input_data_after_preprocessing = input_data;
    MLCouplingData<Out> output_data_before_postprocessing = output_data;
    return create_mlcoupling_from_config_impl<In, Out>(config_str,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       true,
                                                       cast_mode,
                                                       overrides);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing) {
    return create_mlcoupling_from_config(config_str,
                                         std::move(input_data),
                                         std::move(output_data),
                                         std::move(input_data_after_preprocessing),
                                         std::move(output_data_before_postprocessing),
                                         ConfigCastMode::Relaxed);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing,
                                                   ConfigCastMode cast_mode) {
    return create_mlcoupling_from_config(config_str,
                                         std::move(input_data),
                                         std::move(output_data),
                                         std::move(input_data_after_preprocessing),
                                         std::move(output_data_before_postprocessing),
                                         cast_mode,
                                         ConfigOverrides{});
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str,
                                                   MLCouplingData<In> input_data,
                                                   MLCouplingData<Out> output_data,
                                                   MLCouplingData<In> input_data_after_preprocessing,
                                                   MLCouplingData<Out> output_data_before_postprocessing,
                                                   ConfigCastMode cast_mode,
                                                   const ConfigOverrides& overrides) {
    return create_mlcoupling_from_config_impl<In, Out>(config_str,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       false,
                                                       cast_mode,
                                                       overrides);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path, MLCouplingData<In> input_data, MLCouplingData<Out> output_data) {
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       ConfigCastMode::Relaxed);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        ConfigCastMode cast_mode) {
    MLCouplingData<In> input_data_after_preprocessing = input_data;
    MLCouplingData<Out> output_data_before_postprocessing = output_data;
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       cast_mode);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        const ConfigOverrides& overrides) {
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       ConfigCastMode::Relaxed,
                                                       overrides);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        ConfigCastMode cast_mode,
                                                        const ConfigOverrides& overrides) {
    MLCouplingData<In> input_data_after_preprocessing = input_data;
    MLCouplingData<Out> output_data_before_postprocessing = output_data;
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       cast_mode,
                                                       overrides);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        MLCouplingData<In> input_data_after_preprocessing,
                                                        MLCouplingData<Out> output_data_before_postprocessing) {
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       ConfigCastMode::Relaxed);
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        MLCouplingData<In> input_data_after_preprocessing,
                                                        MLCouplingData<Out> output_data_before_postprocessing,
                                                        ConfigCastMode cast_mode) {
    return create_mlcoupling_from_config_file<In, Out>(config_file_path,
                                                       std::move(input_data),
                                                       std::move(output_data),
                                                       std::move(input_data_after_preprocessing),
                                                       std::move(output_data_before_postprocessing),
                                                       cast_mode,
                                                       ConfigOverrides{});
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path,
                                                        MLCouplingData<In> input_data,
                                                        MLCouplingData<Out> output_data,
                                                        MLCouplingData<In> input_data_after_preprocessing,
                                                        MLCouplingData<Out> output_data_before_postprocessing,
                                                        ConfigCastMode cast_mode,
                                                        const ConfigOverrides& overrides) {
    // Open and read the file content
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file_path);
    }
    std::string config_str((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    return create_mlcoupling_from_config<In, Out>(config_str,
                                                  std::move(input_data),
                                                  std::move(output_data),
                                                  std::move(input_data_after_preprocessing),
                                                  std::move(output_data_before_postprocessing),
                                                  cast_mode,
                                                  overrides);
}
