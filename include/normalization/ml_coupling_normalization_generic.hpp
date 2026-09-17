#pragma once

#include "ml_coupling_normalization.hpp"
#include <functional>
#include <stdexcept>
#include <typeinfo>

/**
 * @file ml_coupling_normalization_generic.hpp
 * @brief Generic callback-based implementation of MLCouplingNormalization.
 */

/**
 * @brief Generic normalization delegating feature transformations to user callbacks.
 *
 * Allows custom scaling, standardization (z-score), log transforms, or unit conversions
 * without needing to create a dedicated C++ subclass.
 *
 * @tparam In Primitive scalar type of input features.
 * @tparam Out Primitive scalar type of model output features.
 * @ingroup cpp_core
 */
template <typename In, typename Out>
class MLCouplingNormalizationGeneric : public MLCouplingNormalization<In, Out>
{
public:
    using NormalizeFn   = std::function<void(MLCouplingData<In>)>;   /**< Signature for input normalization callback. */
    using DenormalizeFn = std::function<void(MLCouplingData<Out>)>;  /**< Signature for output denormalization callback. */
    using PrintFn       = std::function<void(std::ostream&)>;        /**< Signature for debug stream printing callback. */

    /**
     * @brief Constructs an MLCouplingNormalizationGeneric from user callbacks.
     *
     * @param normalize_input_fn Callback transforming input data in-place before inference.
     * @param denormalize_output_fn Callback transforming output data in-place after inference.
     * @param print_fn Optional callback to print human-readable configuration details to a stream.
     * @throws std::invalid_argument If either normalize_input_fn or denormalize_output_fn is null.
     */
    MLCouplingNormalizationGeneric(
        std::function<void(MLCouplingData<In>)> normalize_input_fn,
        std::function<void(MLCouplingData<Out>)> denormalize_output_fn,
        std::function<void(std::ostream&)> print_fn = nullptr)
        : normalize_fn_(std::move(normalize_input_fn)),
          denormalize_fn_(std::move(denormalize_output_fn)),
          print_fn_(std::move(print_fn))
    {
        if (!normalize_fn_)
            throw std::invalid_argument("MLCouplingNormalizationGeneric: normalize_input callback must not be null");
        if (!denormalize_fn_)
            throw std::invalid_argument("MLCouplingNormalizationGeneric: denormalize_output callback must not be null");
    }

    /**
     * @brief Executes the registered input normalization callback on @p input_data.
     * @param input_data Data container to normalize in-place.
     */
    void normalize_input(MLCouplingData<In> input_data) override
    {
        normalize_fn_(input_data);
    }

    /**
     * @brief Executes the registered output denormalization callback on @p output_data.
     * @param output_data Data container to denormalize in-place.
     */
    void denormalize_output(MLCouplingData<Out> output_data) override
    {
        denormalize_fn_(output_data);
    }

    /**
     * @brief Prints component details to the output stream.
     * @param os Target output stream.
     */
    void print(std::ostream &os) const override
    {
        if (print_fn_)
        {
            print_fn_(os);
        }
        else
        {
            os << "MLCouplingNormalizationGeneric";
        }
    }

private:
    NormalizeFn   normalize_fn_;
    DenormalizeFn denormalize_fn_;
    PrintFn       print_fn_;
};
