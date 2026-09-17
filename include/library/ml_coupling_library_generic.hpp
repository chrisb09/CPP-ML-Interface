#pragma once

#include "ml_coupling_library.hpp"
#include <functional>
#include <stdexcept>
#include <map>
#include <string>

/**
 * @file ml_coupling_library_generic.hpp
 * @brief Generic callback-based implementation of MLCouplingLibrary.
 */

/**
 * @brief Generic ML library provider delegating inference and training to callbacks.
 *
 * Provides a lightweight mechanism to couple custom inference models, mock testing engines,
 * or arbitrary C/Fortran routines directly into the MLCoupling workflow without writing
 * a custom C++ backend class. Inherits all tiered staging capabilities (Tier 0 static,
 * Tier 1 ordered flexible, Tier 2 keyed flexible) from @ref MLCouplingLibrary.
 *
 * @tparam LibraryInput Primitive scalar type of input features.
 * @tparam LibraryOutput Primitive scalar type of output features.
 * @ingroup cpp_core
 */
template <typename LibraryInput, typename LibraryOutput>
class MLCouplingLibraryGeneric : public MLCouplingLibrary<LibraryInput, LibraryOutput>
{
public:
    using InferenceFn = std::function<void(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)>;               /**< Signature for inference callback. */
    using TrainFn     = std::function<std::map<std::string, double>(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)>; /**< Signature for training callback. */
    using SyncIterFn  = std::function<std::size_t(std::size_t)>;                                                            /**< Signature for synchronized iteration query. */

    /**
     * @brief Constructs a generic library provider without MPI rank specification.
     *
     * @param inference_fn Mandatory callback executing model inference.
     * @param train_fn Optional callback executing in-situ training and returning metric key-value pairs.
     * @param sync_iter_fn Optional callback to synchronize iteration count across MPI ranks.
     * @throws std::invalid_argument If @p inference_fn is null.
     */
    MLCouplingLibraryGeneric(
        std::function<void(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)> inference_fn,
        std::function<std::map<std::string, double>(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)> train_fn = nullptr,
        std::function<std::size_t(std::size_t)> sync_iter_fn = nullptr)
        : MLCouplingLibrary<LibraryInput, LibraryOutput>(),
          inference_fn_(std::move(inference_fn)),
          train_fn_(std::move(train_fn)),
          sync_iter_fn_(std::move(sync_iter_fn))
    {
        if (!inference_fn_)
            throw std::invalid_argument("MLCouplingLibraryGeneric: inference callback must not be null");
    }

    /**
     * @brief Constructs a generic library provider with an explicit MPI rank.
     *
     * @param rank MPI process rank.
     * @param inference_fn Mandatory callback executing model inference.
     * @param train_fn Optional callback executing in-situ training.
     * @param sync_iter_fn Optional callback synchronizing iteration counts.
     * @throws std::invalid_argument If @p inference_fn is null.
     */
    MLCouplingLibraryGeneric(
        int rank,
        std::function<void(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)> inference_fn,
        std::function<std::map<std::string, double>(MLCouplingData<LibraryInput>*, MLCouplingData<LibraryOutput>*)> train_fn = nullptr,
        std::function<std::size_t(std::size_t)> sync_iter_fn = nullptr)
        : MLCouplingLibrary<LibraryInput, LibraryOutput>(rank),
          inference_fn_(std::move(inference_fn)),
          train_fn_(std::move(train_fn)),
          sync_iter_fn_(std::move(sync_iter_fn))
    {
        if (!inference_fn_)
            throw std::invalid_argument("MLCouplingLibraryGeneric: inference callback must not be null");
    }

    /**
     * @brief Executes synchronous inference by invoking the user-provided callback.
     *
     * @param input Container holding input feature tensors.
     * @param output Container populated with output inference tensors.
     */
    void static_inference(MLCouplingData<LibraryInput> *input,
                          MLCouplingData<LibraryOutput> *output) override
    {
        inference_fn_(input, output);
    }

    /**
     * @brief Executes synchronous training by invoking the user-provided training callback.
     *
     * If no training callback was supplied at construction, falls back to the base class default.
     *
     * @param input Training feature inputs.
     * @param target Ground-truth targets.
     * @return Map of training metric names to values.
     */
    std::map<std::string, double> static_train(MLCouplingData<LibraryInput> *input,
                                               MLCouplingData<LibraryOutput> *target) override
    {
        if (train_fn_)
        {
            return train_fn_(input, target);
        }
        return MLCouplingLibrary<LibraryInput, LibraryOutput>::static_train(input, target);
    }

    /**
     * @brief Returns globally synchronized training iteration count across ranks.
     *
     * @param local_iterations Number of iterations completed locally.
     * @return Synchronized iteration count.
     */
    std::size_t get_synchronized_iterations(std::size_t local_iterations) const override
    {
        if (sync_iter_fn_)
        {
            return sync_iter_fn_(local_iterations);
        }
        return MLCouplingLibrary<LibraryInput, LibraryOutput>::get_synchronized_iterations(local_iterations);
    }

private:
    InferenceFn inference_fn_;
    TrainFn     train_fn_;
    SyncIterFn  sync_iter_fn_;
};
