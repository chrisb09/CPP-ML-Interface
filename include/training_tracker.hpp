#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

/**
 * @file training_tracker.hpp
 * @brief Training metric monitoring and historical loss logging for in-situ ML training.
 */

/**
 * @brief Snapshot of training metric values recorded at a specific simulation timestep.
 * @ingroup cpp_config
 */
struct TrainingMetrics
{
    long long step_id;                    /**< Simulation iteration or step identifier. */
    std::map<std::string, double> values; /**< Metric name-to-value map (e.g. "loss" -> 0.012). */
};

/**
 * @brief Tracks and queries loss and validation metrics during in-situ ML training.
 *
 * Allows simulation drivers to register specific fields to monitor (e.g. "loss", "accuracy")
 * and inspect their time series evolution across training epochs.
 *
 * @ingroup cpp_config
 */
class TrainingTracker
{
public:
    /**
     * @brief Registers a named metric field for historical tracking.
     * @param field Name of the metric (e.g. "loss").
     */
    void track(const std::string &field)
    {
        enabled_fields.insert(field);
    }

    /**
     * @brief Logs training metrics produced at step @p step_id.
     * @param step_id Simulation step identifier.
     * @param provider_output Map of all metrics returned by the training provider.
     */
    void log(long long step_id, const std::map<std::string, double> &provider_output)
    {
        TrainingMetrics metrics;
        metrics.step_id = step_id;
        bool found = false;
        for (const auto &field : enabled_fields)
        {
            auto it = provider_output.find(field);
            if (it != provider_output.end())
            {
                metrics.values[field] = it->second;
                found = true;
            }
        }
        if (found)
        {
            history.push_back(std::move(metrics));
        }
    }

    /**
     * @brief Retrieves the time series of a specific tracked metric.
     * @param field Metric name.
     * @return Vector of recorded values ordered by step.
     */
    std::vector<double> get_history(const std::string &field) const
    {
        std::vector<double> result;
        for (const auto &metrics : history)
        {
            auto it = metrics.values.find(field);
            if (it != metrics.values.end())
            {
                result.push_back(it->second);
            }
        }
        return result;
    }

    /**
     * @brief Retrieves complete history for all tracked metrics.
     * @return Map associating each metric name with its sequence of recorded values.
     */
    std::map<std::string, std::vector<double>> get_history() const
    {
        std::map<std::string, std::vector<double>> result;
        for (const auto &field : enabled_fields)
        {
            result[field] = get_history(field);
        }
        return result;
    }

    /**
     * @brief Retrieves the most recently recorded value of a given metric.
     * @param field Metric name.
     * @return Most recent scalar value, or 0.0 if not found.
     */
    double get_latest(const std::string &field) const
    {
        for (auto it = history.rbegin(); it != history.rend(); ++it)
        {
            auto fit = it->values.find(field);
            if (fit != it->values.end())
            {
                return fit->second;
            }
        }
        return 0.0;
    }

private:
    std::set<std::string> enabled_fields;
    std::vector<TrainingMetrics> history;
};
