#pragma once

#include <cmath>
#include <stdexcept>

#include "ml_coupling_behavior.hpp"

// @registry_name: FlowExtrapolatorBehavior
// @registry_aliases: flow-extrapolator-behavior, maia-flow-extrapolator-behavior
class MLCouplingBehaviorFlowExtrapolator : public MLCouplingBehavior
{
public:
    MLCouplingBehaviorFlowExtrapolator(
        int inference_interval,
        int coupled_steps_before_inference,
        int step_increment_after_inference,
        int hdf_output_interval,
        int total_timesteps,
        double scaling_factor = 1.0,
        int forecast_window = 1,
        int input_step_distance = 1,
        int inference_start_step = 0,
        int global_step_offset = 0)
        : inference_interval_(inference_interval),
          coupled_steps_before_inference_(coupled_steps_before_inference),
          step_increment_after_inference_(step_increment_after_inference),
          hdf_output_interval_(hdf_output_interval),
          total_timesteps_(total_timesteps),
          scaling_factor_(scaling_factor),
          forecast_window_(forecast_window),
          input_step_distance_(input_step_distance),
          inference_start_step_(inference_start_step),
          global_step_offset_(global_step_offset)
    {
        if (inference_interval_ <= 0)
        {
            throw std::invalid_argument("MLCouplingBehaviorFlowExtrapolator: inference_interval must be > 0");
        }
        if (coupled_steps_before_inference_ < 1)
        {
            throw std::invalid_argument("MLCouplingBehaviorFlowExtrapolator: coupled_steps_before_inference must be >= 1");
        }
        if (total_timesteps_ <= 0)
        {
            throw std::invalid_argument("MLCouplingBehaviorFlowExtrapolator: total_timesteps must be > 0");
        }
        stride_ = static_cast<int>(std::round(static_cast<double>(input_step_distance_) * scaling_factor_));
        if (stride_ <= 0)
        {
            stride_ = 1;
        }
        next_inference_step_ = inference_start_step_;
    }

    bool should_perform_inference() override
    {
        logical_step_count_++;
        if (logical_step_count_ != next_inference_step_)
        {
            return false;
        }

        int increment = time_step_delta();
        long long int next_logical = logical_step_count_ + inference_interval_;
        long long int next_global = next_logical + global_step_offset_;

        if (next_logical + increment >= total_timesteps_)
        {
            next_inference_step_ = static_cast<long long int>(total_timesteps_) + 1;
        }
        else if (is_hdf_unsafe(next_global, increment))
        {
            next_inference_step_ = next_logical
                + (hdf_output_interval_ - static_cast<int>((next_global - 1) % hdf_output_interval_));
        }
        else
        {
            next_inference_step_ = next_logical;
        }
        return true;
    }

    int time_step_delta() override
    {
        return static_cast<int>(std::round(static_cast<double>(step_increment_after_inference_)
                                           * scaling_factor_
                                           * static_cast<double>(forecast_window_)));
    }

    bool should_send_data() override
    {
        // logical_step_count_ is incremented by should_perform_inference(), which
        // is called after should_send_data() in the application's ml_step().
        // Compensate by projecting the count forward by one.
        long long int projected = logical_step_count_ + 1;
        long long int dist = next_inference_step_ - projected;
        return dist >= 0
            && dist < static_cast<long long int>(coupled_steps_before_inference_) * static_cast<long long int>(stride_)
            && (dist % stride_ == 0);
    }

private:
    bool is_hdf_unsafe(long long int next_global, int increment) const
    {
        if (hdf_output_interval_ <= 0)
        {
            return false;
        }
        int remainder = static_cast<int>(next_global % hdf_output_interval_);
        return !(remainder > 0 && remainder < (hdf_output_interval_ - increment));
    }

    long long int logical_step_count_ = 0;
    int inference_interval_;
    int coupled_steps_before_inference_;
    int step_increment_after_inference_;
    int hdf_output_interval_;
    int total_timesteps_;
    double scaling_factor_;
    int forecast_window_;
    int input_step_distance_;
    int inference_start_step_;
    int global_step_offset_;
    int stride_ = 1;
    long long int next_inference_step_ = 0;
};
