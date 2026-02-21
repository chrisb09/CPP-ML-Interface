#include <cstddef>                  // Not sure if we need this
#include <vector>                   // std::vector
#include <unordered_map>            // std::unordered_map
#include <string>                   // std::string
#include <iostream>                  // std::cout, std::endl
#include <tuple>                    // std::pair

#include "c_api.h"                  // C API header
#include "generated_registry.hpp" 

void* create_provider(const char* name, int in_selection, int out_selection, char** param_names, void** params, int param_count) {
    std::unordered_map<std::string, std::pair<int, void*>> params_map;
    for (int i = 0; i < param_count; ++i) {
        params_map[std::string(param_names[i])] = std::pair(0, params[i]);
    }
    switch (in_selection) {
        case 0: // float
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<float, float>(name, params_map);
                case 1: // double
                    return create_instance_mlcouplingprovider<float, double>(name, params_map);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<float, int32_t>(name, params_map);
                default:
                    return nullptr; // Invalid out_selection
            }
        case 1: // double
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<double, float>(name, params_map);
                case 1: // double
                    return create_instance_mlcouplingprovider<double, double>(name, params_map);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<double, int32_t>(name, params_map);
                default:
                    return nullptr; // Invalid out_selection
            }
        case 2: // int32_t
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<int32_t, float>(name, params_map);
                case 1: // double
                    return create_instance_mlcouplingprovider<int32_t, double>(name, params_map);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<int32_t, int32_t>(name, params_map);
                default:
                    return nullptr; // Invalid out_selection
            }
        default:
            return nullptr; // Invalid in_selection
    }
    // fallback, should never reach here due to the default cases, but just in case
    return create_instance_mlcouplingprovider<double, double>(name, params_map);
}
