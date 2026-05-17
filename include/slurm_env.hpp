#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

// kinda cool to use namespace imo
// i use them not often enough i guess
namespace mlcoupling::slurm_env
{

inline int parse_int(const std::string &value, int fallback)
{
    if (value.empty())
    {
        return fallback;
    }

    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX)
    {
        return fallback;
    }
    return static_cast<int>(parsed);
}

inline std::string get_env_string(const std::string &name)
{
    const char *raw = std::getenv(name.c_str());
    return raw == nullptr ? std::string() : std::string(raw);
}

inline int get_env_int(const std::string &name, int fallback)
{
    const std::string raw = get_env_string(name);
    return raw.empty() ? fallback : parse_int(raw, fallback);
}

inline std::string trim_slurm_count_token(std::string value)
{
    const std::size_t paren = value.find('(');
    if (paren != std::string::npos)
    {
        value.resize(paren);
    }

    const std::size_t comma = value.find(',');
    if (comma != std::string::npos)
    {
        value.resize(comma);
    }

    value.erase(std::remove_if(value.begin(),
                               value.end(),
                               [](unsigned char c)
                               { return std::isspace(c) != 0; }),
                value.end());
    return value;
}

inline int parse_slurm_count(const std::string &raw, int fallback)
{
    std::string value = trim_slurm_count_token(raw);
    if (value.empty())
    {
        return fallback;
    }

    const std::size_t colon = value.rfind(':');
    const std::size_t equals = value.rfind('=');
    const std::size_t after_colon = colon == std::string::npos ? 0 : colon + 1;
    const std::size_t after_equals = equals == std::string::npos ? 0 : equals + 1;
    const std::size_t begin_hint = std::max(after_colon, after_equals);
    if (begin_hint > 0 && begin_hint < value.size())
    {
        value = value.substr(begin_hint);
    }

    std::size_t begin = 0;
    while (begin < value.size() && !std::isdigit(static_cast<unsigned char>(value[begin])))
    {
        ++begin;
    }
    if (begin == value.size())
    {
        return fallback;
    }

    std::size_t end = begin;
    while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end])))
    {
        ++end;
    }
    return parse_int(value.substr(begin, end - begin), fallback);
}

inline int get_env_slurm_count(const std::string &name, int fallback)
{
    const std::string raw = get_env_string(name);
    return raw.empty() ? fallback : parse_slurm_count(raw, fallback);
}

inline int het_group_nodes(int group, int fallback)
{
    const std::string suffix = std::to_string(group);
    int value = get_env_slurm_count("SLURM_JOB_NUM_NODES_HET_GROUP_" + suffix, -1);
    if (value < 0)
    {
        value = get_env_slurm_count("SLURM_JOB_NODES_HET_GROUP_" + suffix, -1);
    }
    return value > 0 ? value : fallback;
}

inline int het_group_gpus_per_node(int group, int fallback)
{
    const std::string suffix = std::to_string(group);
    int value = get_env_slurm_count("SLURM_GPUS_PER_NODE_HET_GROUP_" + suffix, -1);
    if (value < 0)
    {
        value = get_env_slurm_count("SLURM_GPUS_ON_NODE_HET_GROUP_" + suffix, -1);
    }
    if (value < 0)
    {
        const int gpus_per_task = get_env_slurm_count("SLURM_GPUS_PER_TASK_HET_GROUP_" + suffix, -1);
        const int tasks_per_node = get_env_slurm_count("SLURM_NTASKS_PER_NODE_HET_GROUP_" + suffix, -1);
        if (gpus_per_task >= 0 && tasks_per_node > 0)
        {
            value = gpus_per_task * tasks_per_node;
        }
    }
    if (value < 0)
    {
        const std::string tres_per_node = get_env_string("SLURM_TRES_PER_NODE_HET_GROUP_" + suffix);
        const std::size_t gpu_pos = tres_per_node.find("gres/gpu");
        if (gpu_pos != std::string::npos)
        {
            value = parse_slurm_count(tres_per_node.substr(gpu_pos), -1);
        }
    }
    if (value < 0)
    {
        const std::string tres_per_task = get_env_string("SLURM_TRES_PER_TASK_HET_GROUP_" + suffix);
        const std::size_t gpu_pos = tres_per_task.find("gres/gpu");
        const int tasks_per_node = get_env_slurm_count("SLURM_NTASKS_PER_NODE_HET_GROUP_" + suffix, -1);
        if (gpu_pos != std::string::npos && tasks_per_node > 0)
        {
            const int gpus_per_task = parse_slurm_count(tres_per_task.substr(gpu_pos), -1);
            if (gpus_per_task >= 0)
            {
                value = gpus_per_task * tasks_per_node;
            }
        }
    }
    return value >= 0 ? value : fallback;
}

inline int single_gpu_het_group(int fallback)
{
    const int het_size = get_env_int("SLURM_HET_SIZE", -1);
    if (het_size <= 0)
    {
        return fallback;
    }

    int only_gpu_group = -1;
    for (int group = 0; group < het_size; ++group)
    {
        const int gpus = het_group_gpus_per_node(group, 0);
        if (gpus <= 0)
        {
            continue;
        }
        if (only_gpu_group >= 0)
        {
            return fallback;
        }
        only_gpu_group = group;
    }
    return only_gpu_group >= 0 ? only_gpu_group : fallback;
}

} // namespace mlcoupling::slurm_env
