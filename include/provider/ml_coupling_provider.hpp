#pragma once

#include "../data_processor/ml_coupling_data.hpp"

// @category: provider
template <typename In, typename Out>
class MLCouplingProvider {
    public:

        MLCouplingProvider() = default;

        // Essentially for the coupling step: send data to the ML model
        virtual void send_data(MLCouplingData<In> input_data_after_preprocessing) = 0;

        // Perform inference with the ML model and get the output data
        virtual MLCouplingData<Out> inference(MLCouplingData<In> input_data_after_preprocessing) = 0;

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