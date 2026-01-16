#pragma once

#include "ml_coupling_data_processor.hpp"

#include "../normalization/ml_coupling_normalization.hpp"

template <typename In, typename Out>
class MLCouplingDataProcessorSimple : public MLCouplingDataProcessor<In, Out> {

    MLCouplingDataProcessorSimple(std::vector<In*> input_data,
            std::vector<std::vector<int>> input_data_dimensions,
            std::vector<Out*> output_data,
            std::vector<std::vector<int>> output_data_dimensions) {
        super(input_data,
              input_data_dimensions,
              output_data,
              output_data_dimensions,
              new MLCouplingNormalization<In, Out>()
            );
    }


};