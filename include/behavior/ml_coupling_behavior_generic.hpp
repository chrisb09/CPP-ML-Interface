#pragma once

#include "ml_coupling_behavior.hpp"
#include <functional>
#include <stdexcept>

/**
 * @file ml_coupling_behavior_generic.hpp
 * @brief Generic callback-based implementation of MLCouplingBehavior.
 */

/**
 * @brief A generic behavior implementation delegating to user-supplied functions or lambdas.
 *
 * Eliminates the need to subclass @ref MLCouplingBehavior for custom simulation stepping logic.
 * Enables both C++ lambdas and C/Fortran function pointers (via `void*` user data forwarding)
 * to govern inference scheduling, timestep advances, and database transmission.
 *
 * @ingroup cpp_core
 */
class MLCouplingBehaviorGeneric : public MLCouplingBehavior
{
public:
    using ShouldInferFn   = std::function<bool()>;   /**< Signature for inference decision predicate. */
    using TimeStepDeltaFn = std::function<int()>;    /**< Signature for simulation timestep delta advancement. */
    using ShouldSendDataFn = std::function<bool()>;  /**< Signature for database transmission predicate. */

    /**
     * @brief Constructs an MLCouplingBehaviorGeneric from callback functions.
     *
     * @param should_infer Callback returning true if inference should be performed at the current step.
     * @param time_step_delta Callback returning the simulation timestep delta to advance.
     * @param should_send_data Callback returning true if data should be sent to the backend database.
     * @throws std::invalid_argument If any of the callback functions are empty / null.
     */
    MLCouplingBehaviorGeneric(std::function<bool()> should_infer,
                              std::function<int()> time_step_delta,
                              std::function<bool()> should_send_data)
        : should_infer_fn_(std::move(should_infer)),
          time_step_delta_fn_(std::move(time_step_delta)),
          should_send_data_fn_(std::move(should_send_data))
    {
        if (!should_infer_fn_)
            throw std::invalid_argument("MLCouplingBehaviorGeneric: should_infer callback must not be null");
        if (!time_step_delta_fn_)
            throw std::invalid_argument("MLCouplingBehaviorGeneric: time_step_delta callback must not be null");
        if (!should_send_data_fn_)
            throw std::invalid_argument("MLCouplingBehaviorGeneric: should_send_data callback must not be null");
    }

    /**
     * @brief Evaluates whether inference should be performed at this step.
     * @return Result of the registered `should_infer` callback.
     */
    bool should_perform_inference() override { return should_infer_fn_(); }

    /**
     * @brief Evaluates the timestep increment to advance the simulation.
     * @return Result of the registered `time_step_delta` callback.
     */
    int  time_step_delta()          override { return time_step_delta_fn_(); }

    /**
     * @brief Evaluates whether simulation data should be pushed to the database.
     * @return Result of the registered `should_send_data` callback.
     */
    bool should_send_data()         override { return should_send_data_fn_(); }

private:
    ShouldInferFn    should_infer_fn_;
    TimeStepDeltaFn  time_step_delta_fn_;
    ShouldSendDataFn should_send_data_fn_;
};
