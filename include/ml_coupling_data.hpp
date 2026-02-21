#pragma once

#include <vector>
#include <iostream>


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

// Stream operator overload for MLCouplingData
template <typename T>
std::ostream& operator<<(std::ostream& os, const MLCouplingData<T>& coupling_data) {
    os << "MLCouplingData{data_segments=" << coupling_data.data.size();
    if (!coupling_data.data_dimensions.empty()) {
        os << ", dimensions=[";
        for (size_t i = 0; i < coupling_data.data_dimensions.size(); ++i) {
            if (i > 0) os << ", ";
            os << "[";
            for (size_t j = 0; j < coupling_data.data_dimensions[i].size(); ++j) {
                if (j > 0) os << ", ";
                os << coupling_data.data_dimensions[i][j];
            }
            os << "]";
        }
        os << "]";
    }
    os << "}";
    return os;
}