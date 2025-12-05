#pragma once

template <typename In, typename Out>
class MLCouplingProvider {
    public:
        virtual ~MLCouplingProvider() = default;

        // Initialize the ML coupling provider
        virtual void init() = 0;

        // Perform inference using the ML model
        virtual void inference() = 0; // strategy_inference in the old code

        // Finalize and clean up resources
        virtual void finalize() = 0;

        //TODO: consider adding a method to get the MPI communicator if needed, I'm not sure yet

        /*
        
        Apparently the 1D tensor functions were used to deal with Fortrans's column-major order, which
        would have been an issue since the ml models are accessed via C++ which uses row-major order.
        So by transforming them into a linear 1D array, the order issue is avoided.

        However, this means that for the Fortran wrapper that we plan to implement later, we probably need to
        reintroduce these 1D tensor functions to handle the data correctly between Fortran and C++.
        
        */
};