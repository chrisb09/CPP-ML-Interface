#include <cstddef>                  // Not sure if we need this
#include <vector>                   // std::vector
#include <unordered_map>            // std::unordered_map
#include <string>                   // std::string
#include <iostream>                  // std::cout, std::endl
#include <tuple>                    // std::pair

#include "c_api.h"                  // C API header
#include "generated_registry.hpp" 
#include "../include/data/ml_coupling_data_type.hpp" // get_type_tag function


void* create_provider(const char* name, int in_selection, int out_selection, char** param_names, void** params, int param_count) {
    std::unordered_map<std::string, std::pair<int, void*>> params_map;
    for (int i = 0; i < param_count; ++i) {
        params_map[std::string(param_names[i])] = std::pair(0, params[i]);
    }

    // 1. Convert integers to compile-time tags
    MLCouplingSupportedTypes in_tag = get_type_tag(in_selection);
    MLCouplingSupportedTypes out_tag = get_type_tag(out_selection);

    // 2. Visit BOTH tags simultaneously
    return std::visit([&](auto in_t, auto out_t) -> void* {
        
        // Extract the actual C++ types from the tags
        using InType = typename decltype(in_t)::type;
        using OutType = typename decltype(out_t)::type;

        // AIxeleratorService is only instantiated for float and double, with the
        // same scalar type on both input and output.
        constexpr bool aix_supported_scalar =
            std::is_same_v<InType, float> || std::is_same_v<InType, double>;

        if (std::string(name) == "Aixelerator" &&
            (!std::is_same_v<InType, OutType> || !aix_supported_scalar)) {
            // fallback
            std::cerr << "Warning: AIxelerator provider requires same-type float/float or double/double data. Falling back to double.\n";
            return static_cast<void*>(create_instance_mlcouplingprovider<double, double>(name, params_map));
        }
        
        return static_cast<void*>(create_instance_mlcouplingprovider<InType, OutType>(name, params_map));
        
    }, in_tag, out_tag);

    // fallback, should never reach here due to the default cases, but just in case
    return create_instance_mlcouplingprovider<double, double>(name, params_map);
}
