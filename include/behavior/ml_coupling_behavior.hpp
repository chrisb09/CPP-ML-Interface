#pragma once

/**
 * @file ml_coupling_behavior.hpp
 * @brief Abstract base class defining coupling stepping behaviors and intervals.
 */

/**
 * @brief Abstract base class controlling simulation-ML execution timing and interval scheduling.
 *
 * Concrete behaviors determine when inference runs, how many simulation timesteps
 * are skipped or advanced after inference, and whether intermediate simulation state
 * should be pushed to external stores (such as PhyDLL's staging memory or SmartSim DB).
 *
 * @ingroup cpp_behavior
 */
// @category: behavior
class MLCouplingBehavior
{

public:
    /**
     * @brief Determines whether ML inference should be executed at the current step.
     * @return True to run inference, false to perform a pure physics step.
     */
    virtual bool should_perform_inference() = 0;

    /**
     * @brief Returns the simulation timestep delta to advance.
     * @return Positive integer (e.g. 1 for standard step, > 1 when accelerating via extrapolation).
     */
    virtual int time_step_delta() = 0;

    /**
     * @brief Determines whether simulation data should be transmitted to the database/client at this step.
     *
     * Enables backends like PhyDLL to accumulate history buffers across multiple steps
     * before triggering model inference.
     *
     * @return True to push current simulation data, false otherwise.
     */
    virtual bool should_send_data() = 0;

    virtual ~MLCouplingBehavior() = default;
};
