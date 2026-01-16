#include <cstddef>                  // Not sure if we need this
#include <vector>                   // std::vector

#include "c_api.h"                  // C API header
#include "generated_registry.hpp" 

void* create_provider(const char* name, int in_selection, int out_selection, void** params, int param_count) {
    std::vector<void*> params_vec(params, params + param_count);
    switch (in_selection) {
        case 0: // float
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<float, float>(name, params_vec);
                case 1: // double
                    return create_instance_mlcouplingprovider<float, double>(name, params_vec);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<float, int32_t>(name, params_vec);
                default:
                    return nullptr; // Invalid out_selection
            }
        case 1: // double
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<double, float>(name, params_vec);
                case 1: // double
                    return create_instance_mlcouplingprovider<double, double>(name, params_vec);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<double, int32_t>(name, params_vec);
                default:
                    return nullptr; // Invalid out_selection
            }
        case 2: // int32_t
            switch (out_selection) {
                case 0: // float
                    return create_instance_mlcouplingprovider<int32_t, float>(name, params_vec);
                case 1: // double
                    return create_instance_mlcouplingprovider<int32_t, double>(name, params_vec);
                case 2: // int32_t
                    return create_instance_mlcouplingprovider<int32_t, int32_t>(name, params_vec);
                default:
                    return nullptr; // Invalid out_selection
            }
        default:
            return nullptr; // Invalid in_selection
    }
    return create_instance_mlcouplingprovider<double, double>(name, params_vec);
}
