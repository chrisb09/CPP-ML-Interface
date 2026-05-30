#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

struct TrainingMetrics
{
    long long step_id;
    std::map<std::string, double> values;
};

class TrainingTracker
{
public:
    void track(const std::string &field)
    {
        enabled_fields.insert(field);
    }

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

    std::map<std::string, std::vector<double>> get_history() const
    {
        std::map<std::string, std::vector<double>> result;
        for (const auto &field : enabled_fields)
        {
            result[field] = get_history(field);
        }
        return result;
    }

    double get_current(const std::string &field) const
    {
        if (history.empty()) throw std::runtime_error("No training history available.");
        auto it = history.back().values.find(field);
        if (it != history.back().values.end())
        {
            return it->second;
        }
        throw std::runtime_error("Field '" + field + "' not found in the latest training step.");
    }

    std::map<std::string, double> get_current() const
    {
        if (history.empty()) return {};
        return history.back().values;
    }

    const std::vector<TrainingMetrics> &get_full_history() const
    {
        return history;
    }

private:
    std::set<std::string> enabled_fields;
    std::vector<TrainingMetrics> history;
};
