#pragma once

#include <vector>


template <typename T>
struct MLCouplingData {

        /*
        * This should allow us to have an for example
        * a single memory space in the input data 
        * which has the dimension of 16x16x256
        * This is specified by a single entry in the 
        * dimensions variable, which itself is a 
        * vector with 3 entries: 16,16,256
        * This implicitly defined the memory segment 
        * as well
        */

    std::vector<T*> data;
    std::vector<std::vector<int>> data_dimensions;

    public:
        MLCouplingData(const std::vector<T*>& data,
                       const std::vector<std::vector<int>>& data_dimensions)
            : data(data), data_dimensions(data_dimensions) {}

        MLCouplingData() = default;
};