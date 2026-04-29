#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <mpi.h>

#include "ml_coupling_provider_flexible.hpp"
#include "../tool.h"

// for debugging and to disabled greyed out code in some IDEs
// # define WITH_AIX

#ifdef WITH_AIX
#include "aixeleratorService/aixeleratorService.h"
#endif


// @registry_name: Aixelerator
// @registry_aliases: aixelerator, AIxelerator, aix, AIx, AIX
template<typename In, typename Out>
class MLCouplingProviderAixelerator : public MLCouplingProvider<In, Out> {

    public:

        #ifdef WITH_AIX
        struct AIxServiceDummy {
            void inference() {}
        };

        using AIxServicePtr = std::conditional_t<
            std::is_same_v<In, float> || std::is_same_v<In, double>,
            std::unique_ptr<AIxeleratorService<In>>,
            std::unique_ptr<AIxServiceDummy>
        >;
        AIxServicePtr service;
        #endif
        std::string model_file;
        int batchsize;
        MPI_Comm app_comm;
        bool enable_hybrid;
        std::optional<float> host_fraction;

        MLCouplingProviderAixelerator(std::string model_file, int batchsize=1, MPI_Comm app_comm=MPI_COMM_WORLD, bool enable_hybrid=false, std::optional<float> host_fraction=std::nullopt) : model_file(std::move(model_file)), batchsize(batchsize), app_comm(app_comm), enable_hybrid(enable_hybrid), host_fraction(host_fraction) {
            // Unfortunately, we can't really do much in the constructor since we need the input and output data pointers to create the AIxeleratorService

            // Passing them through the constructor is a bit inconvenient with the config-based approach that we have.
            #ifndef WITH_AIX
            guarantee(false, "AIxelerator provider is not enabled. Please make sure WITH_AIX is defined and the necessary dependencies are installed.");
            #endif
        }

        void inference(MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out>& output_data_before_postprocessing) override
        {
            #ifdef WITH_AIX
            constexpr bool aix_supported_scalar =
                std::is_same_v<In, float> || std::is_same_v<In, double>;

            if constexpr (std::is_same_v<In, Out> && aix_supported_scalar) {
                if (!service) {
                    /*
                     AIxeleratorService(
                        std::string model_file,
                        std::vector<int64_t> input_shape, T* input_data,
                        std::vector<int64_t> output_shape, T* output_data,
                        int batchsize, MPI_Comm app_comm,
                        bool enable_hybrid = false,
                        std::optional<float> host_fraction = std::nullopt
                    );
                    */

                    // TODO: this only works for 1 tensor, we'd need to change AIxeleratorService to support multiple tensors if needed
                    // It also assumes that the tensor is flattened and in contiguous layout
                    service = std::make_unique<AIxeleratorService<In>>(model_file,
                        input_data_after_preprocessing[0].dimensions_as_int64(),
                        static_cast<In*>(input_data_after_preprocessing[0].root()),
                        output_data_before_postprocessing[0].dimensions_as_int64(),
                        static_cast<In*>(output_data_before_postprocessing[0].root()),
                        batchsize, // batchsize, we can make this more flexible later if needed
                        app_comm,
                        enable_hybrid,
                        host_fraction
                    );
                }
                service->inference();
            } else {
                guarantee(false, "MLCouplingProviderAixelerator currently only supports same-type float/float or double/double data. The linked AIxeleratorService backend only provides float and double template instantiations.");
            }
            #else
            guarantee(false, "AIxelerator provider is not enabled. Please make sure WITH_AIX is defined and the necessary dependencies are installed.");
            #endif
        }

};
