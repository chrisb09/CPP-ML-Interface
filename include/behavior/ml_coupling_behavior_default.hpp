#pragma once

#include "ml_coupling_behavior.hpp"

/**
 * @file ml_coupling_behavior_default.hpp
 * @brief Default behavior implementation that triggers inference on every step.
 */

/**
 * @brief Default behavior implementation.
 *
 * Always requests inference on every step (`should_perform_inference() == true`),
 * advances 0 steps ahead (`time_step_delta() == 0`), and always transmits data.
 *
 * @ingroup cpp_behavior
 */
// @registry_name: Default
// @registry_aliases: default
class MLCouplingBehaviorDefault : public MLCouplingBehavior
{
public:
    MLCouplingBehaviorDefault()
    {
    }

    /** @brief Always returns true (infers every step). */
    bool should_perform_inference() override
    {
        return true;
    }

    /** @brief Returns 0 (no time step increment). */
    int time_step_delta() override
    {
        return 0;
    }

    /** @brief Always returns true (transmits data every step). */
    bool should_send_data() override
    {
        return true;
    }
};
