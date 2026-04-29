#pragma once

#include "../data/ml_coupling_data.hpp"

# ifdef MPI_FOUND
#include <mpi.h>
# endif

// @category: provider
template <typename In, typename Out>
class MLCouplingProvider {
    public:

        int rank = -1;

        MLCouplingProvider() {
            # ifdef MPI_FOUND
            // check if MPI is initialized and get the rank if it is, otherwise set rank to -1
            int flag;
            MPI_Initialized(&flag);
            if (flag) {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            }
            # endif
        }

        // Manually set the rank if needed, for example if MPI is initialized after the provider is created, or if we use something other than MPI for parallelism

        MLCouplingProvider(int rank) {
            this->rank = rank;
        }

        void set_rank(int rank) {
            this->rank = rank;
        }

        // Perform inference with the ML model and get the output data
        virtual void inference(MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out>& output_data_before_postprocessing) = 0;

        virtual ~MLCouplingProvider() = default;

        // Later, we might add a train() method here as well

        //TODO: consider adding a method to get the MPI communicator if needed, I'm not sure yet

        /*
        
        Apparently the 1D tensor functions were used to deal with Fortrans's column-major order, which
        would have been an issue since the ml models are accessed via C++ which uses row-major order.
        So by transforming them into a linear 1D array, the order issue is avoided.

        However, this means that for the Fortran wrapper that we plan to implement later, we probably need to
        reintroduce these 1D tensor functions to handle the data correctly between Fortran and C++.
        
        */
};