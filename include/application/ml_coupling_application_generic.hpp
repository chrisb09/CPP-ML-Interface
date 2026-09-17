#pragma once

#include "ml_coupling_application.hpp"
#include <functional>
#include <utility>

/**
 * @file ml_coupling_application_generic.hpp
 * @brief Generic callback-based implementation of MLCouplingApplication.
 */

/**
 * @brief Generic application pipeline delegating preprocessing, postprocessing, and execution to callbacks.
 *
 * Facilitates custom data transformations, shape conversions, or entire step orchestrations
 * via user-supplied callbacks (or C/Fortran function pointers) without creating a derived class.
 *
 * @tparam CouplingInput Primitive scalar type of simulation data before preprocessing.
 * @tparam CouplingOutput Primitive scalar type of simulation data after postprocessing.
 * @tparam LibraryInput Primitive scalar type of ML model input features.
 * @tparam LibraryOutput Primitive scalar type of ML model output predictions.
 * @ingroup cpp_core
 */
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class MLCouplingApplicationGeneric
    : public MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>
{
public:
    using Base = MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>;
    using PreprocessFn  = std::function<MLCouplingData<LibraryInput>(MLCouplingData<CouplingInput>)>;                     /**< Signature for preprocessing callback. */
    using PostprocessFn = std::function<MLCouplingData<CouplingOutput>(MLCouplingData<LibraryOutput>)>;                   /**< Signature for postprocessing callback. */
    using MlStepFn      = std::function<int(MLCouplingLibrary<LibraryInput, LibraryOutput>&, MLCouplingBehavior&)>;       /**< Signature for custom ML step orchestration callback. */

    /**
     * @brief Constructs a generic application with 2-buffer setup (coupling input and output).
     *
     * @param coupling_input Simulation input data buffer container.
     * @param coupling_output Simulation output data buffer container.
     * @param preprocess_fn Optional callback transforming coupling input into library input.
     * @param postprocess_fn Optional callback transforming library output into coupling output.
     * @param ml_step_fn Optional callback overriding the default step execution pipeline.
     * @param normalization Optional pointer to normalization component.
     */
    MLCouplingApplicationGeneric(
        MLCouplingData<CouplingInput> coupling_input,
        MLCouplingData<CouplingOutput> coupling_output,
        std::function<MLCouplingData<LibraryInput>(MLCouplingData<CouplingInput>)> preprocess_fn = nullptr,
        std::function<MLCouplingData<CouplingOutput>(MLCouplingData<LibraryOutput>)> postprocess_fn = nullptr,
        std::function<int(MLCouplingLibrary<LibraryInput, LibraryOutput>&, MLCouplingBehavior&)> ml_step_fn = nullptr,
        MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr)
        : Base(std::move(coupling_input), std::move(coupling_output), normalization),
          preprocess_fn_(std::move(preprocess_fn)),
          postprocess_fn_(std::move(postprocess_fn)),
          ml_step_fn_(std::move(ml_step_fn))
    {
    }

    /**
     * @brief Constructs a generic application with 4-buffer setup (distinct coupling and library buffers).
     *
     * @param coupling_input Simulation input buffer.
     * @param library_input Intermediate ML model input buffer.
     * @param library_output Intermediate ML model output buffer.
     * @param coupling_output Simulation output buffer.
     * @param preprocess_fn Optional callback transforming coupling input into library input.
     * @param postprocess_fn Optional callback transforming library output into coupling output.
     * @param ml_step_fn Optional callback overriding default step execution.
     * @param normalization Optional pointer to normalization component.
     */
    MLCouplingApplicationGeneric(
        MLCouplingData<CouplingInput> coupling_input,
        MLCouplingData<LibraryInput> library_input,
        MLCouplingData<LibraryOutput> library_output,
        MLCouplingData<CouplingOutput> coupling_output,
        std::function<MLCouplingData<LibraryInput>(MLCouplingData<CouplingInput>)> preprocess_fn = nullptr,
        std::function<MLCouplingData<CouplingOutput>(MLCouplingData<LibraryOutput>)> postprocess_fn = nullptr,
        std::function<int(MLCouplingLibrary<LibraryInput, LibraryOutput>&, MLCouplingBehavior&)> ml_step_fn = nullptr,
        MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization = nullptr)
        : Base(std::move(coupling_input), std::move(library_input),
               std::move(library_output), std::move(coupling_output), normalization),
          preprocess_fn_(std::move(preprocess_fn)),
          postprocess_fn_(std::move(postprocess_fn)),
          ml_step_fn_(std::move(ml_step_fn))
    {
    }

    /**
     * @brief Executes a single ML coupling step.
     *
     * If a custom @p ml_step_fn was supplied, delegates directly to it.
     * Otherwise, executes the default sequence: preprocess -> normalize -> infer -> denormalize -> postprocess.
     *
     * @param library ML inference engine reference.
     * @param behavior Stepping and interval decision controller reference.
     * @return Simulation timestep delta to advance.
     */
    int ml_step(MLCouplingLibrary<LibraryInput, LibraryOutput>& library,
                MLCouplingBehavior& behavior) override
    {
        if (ml_step_fn_)
        {
            return ml_step_fn_(library, behavior);
        }
        return Base::ml_step(library, behavior);
    }

protected:
    /**
     * @brief Preprocesses coupling input data into library input format.
     * @param input Raw simulation input data container.
     * @return Preprocessed library input data container.
     */
    MLCouplingData<LibraryInput>
    preprocess_coupling_input(MLCouplingData<CouplingInput> input) override
    {
        if (preprocess_fn_)
        {
            return preprocess_fn_(std::move(input));
        }
        return Base::preprocess_coupling_input(std::move(input));
    }

    /**
     * @brief Postprocesses library output predictions into simulation output format.
     * @param output Raw ML inference prediction container.
     * @return Postprocessed coupling output data container.
     */
    MLCouplingData<CouplingOutput>
    postprocess_library_output(MLCouplingData<LibraryOutput> output) override
    {
        if (postprocess_fn_)
        {
            return postprocess_fn_(std::move(output));
        }
        return Base::postprocess_library_output(std::move(output));
    }

private:
    PreprocessFn  preprocess_fn_;
    PostprocessFn postprocess_fn_;
    MlStepFn      ml_step_fn_;
};
