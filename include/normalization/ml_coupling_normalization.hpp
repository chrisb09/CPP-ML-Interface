#pragma once

#include "../data/ml_coupling_data.hpp"
#include <ostream>

/**
 * @file ml_coupling_normalization.hpp
 * @brief Abstract base class for feature normalization and denormalization.
 */

/**
 * @brief Abstract base class for data scaling and normalization transforms.
 *
 * Encapsulates bidirectional scaling: mapping raw simulation inputs into
 * normalized feature scales expected by the ML model, and unscaling raw model
 * predictions back into physical units.
 *
 * @tparam In Primitive scalar type of input features.
 * @tparam Out Primitive scalar type of output features.
 * @ingroup cpp_normalization
 */
// @category: normalization
template <typename In, typename Out>
class MLCouplingNormalization
{
public:
    /**
     * @brief Normalizes input data in-place prior to model inference.
     * @param input_data Simulation input data container.
     */
    virtual void normalize_input(MLCouplingData<In> input_data) = 0;

    /**
     * @brief Denormalizes model output predictions in-place to restore physical units.
     * @param output_data Model output prediction container.
     */
    virtual void denormalize_output(MLCouplingData<Out> output_data) = 0;

    virtual ~MLCouplingNormalization() = default;

    /**
     * @brief Outputs human-readable representation of the normalization configuration.
     * @param os Output stream.
     */
    virtual void print(std::ostream &os) const = 0;
};

template <typename In, typename Out>
std::ostream &operator<<(std::ostream &os, const MLCouplingNormalization<In, Out> &norm)
{
    norm.print(os);
    return os;
}
