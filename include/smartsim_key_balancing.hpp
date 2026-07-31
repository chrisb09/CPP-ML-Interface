#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlcoupling::smartsim_key_balancing {

constexpr std::uint16_t redis_slot_count = 16384;

inline std::uint16_t crc16_xmodem(std::string_view value)
{
    std::uint16_t crc = 0;
    for (const unsigned char byte : value) {
        crc ^= static_cast<std::uint16_t>(byte) << 8;
        for (int bit = 0; bit < 8; ++bit) {
            crc = static_cast<std::uint16_t>((crc & 0x8000U) != 0U
                                                 ? (crc << 1) ^ 0x1021U
                                                 : crc << 1);
        }
    }
    return crc;
}

inline std::uint16_t redis_hash_slot(std::string_view tag)
{
    return static_cast<std::uint16_t>(crc16_xmodem(tag) % redis_slot_count);
}

inline bool enabled_from_environment()
{
    const char* value = std::getenv("SMARTSIM_BALANCED_KEYS");
    return value != nullptr && std::string_view(value) == "1";
}

// Redis Cluster assigns an initially balanced cluster contiguous, equally sized
// slot ranges. Query CLUSTER SLOTS when a cluster has been resharded.
class RedisKeyBalancer
{
public:
    RedisKeyBalancer() = default;

    RedisKeyBalancer(int rank, int inference_streams_per_node, int ml_nodes, bool enabled)
        : enabled_(enabled)
    {
        if (!enabled_) {
            return;
        }
        if (rank < 0 || inference_streams_per_node < 0 || ml_nodes < 1 || ml_nodes > redis_slot_count) {
            throw std::invalid_argument(
                "SMARTSIM_BALANCED_KEYS requires non-negative rank, a non-negative inference stream count, "
                "and 1..16384 ML nodes");
        }

        target_shard_ = shard_for_rank(rank, inference_streams_per_node, ml_nodes);
        tags_.reserve(static_cast<std::size_t>(ml_nodes));
        for (int shard = 0; shard < ml_nodes; ++shard) {
            tags_.push_back(make_tag(shard, ml_nodes));
        }
    }

    static RedisKeyBalancer from_environment(int rank, int gpus_per_node, int ml_nodes)
    {
        return RedisKeyBalancer(rank, gpus_per_node, ml_nodes, enabled_from_environment());
    }

    bool enabled() const noexcept { return enabled_; }

    int target_shard() const noexcept { return target_shard_; }

    const std::vector<std::string>& tags() const noexcept { return tags_; }

    std::string prefix_key(std::string_view key) const
    {
        if (!enabled_) {
            return std::string(key);
        }
        return "{" + tags_.at(static_cast<std::size_t>(target_shard_)) + "}." + std::string(key);
    }

    static int shard_for_rank(int rank, int inference_streams_per_node, int ml_nodes)
    {
        if (rank < 0 || inference_streams_per_node < 0 || ml_nodes < 1) {
            throw std::invalid_argument("invalid SmartSim rank-to-shard configuration");
        }
        // CPU inference has no device index, so it is one stream per node.
        const int streams = inference_streams_per_node > 0 ? inference_streams_per_node : 1;
        return (rank / streams) % ml_nodes;
    }

    static std::uint16_t slot_first(int shard, int ml_nodes)
    {
        validate_shard(shard, ml_nodes);
        return static_cast<std::uint16_t>(
            (static_cast<std::uint32_t>(shard) * redis_slot_count) / ml_nodes);
    }

    static std::uint16_t slot_last(int shard, int ml_nodes)
    {
        validate_shard(shard, ml_nodes);
        return static_cast<std::uint16_t>(
            ((static_cast<std::uint32_t>(shard + 1) * redis_slot_count) / ml_nodes) - 1U);
    }

private:
    static void validate_shard(int shard, int ml_nodes)
    {
        if (shard < 0 || shard >= ml_nodes || ml_nodes < 1 || ml_nodes > redis_slot_count) {
            throw std::invalid_argument("invalid SmartSim shard configuration");
        }
    }

    static std::string make_tag(int shard, int ml_nodes)
    {
        const std::uint16_t first = slot_first(shard, ml_nodes);
        const std::uint16_t last = slot_last(shard, ml_nodes);
        for (std::uint32_t nonce = 0; nonce < 1000000U; ++nonce) {
            const std::string tag = "smartsim-balanced-n" + std::to_string(ml_nodes) +
                                    "-s" + std::to_string(shard) +
                                    "-k" + std::to_string(nonce);
            const std::uint16_t slot = redis_hash_slot(tag);
            if (slot >= first && slot <= last) {
                return tag;
            }
        }
        throw std::runtime_error("failed to generate a Redis hash tag for SmartSim shard");
    }

    bool enabled_ = false;
    int target_shard_ = -1;
    std::vector<std::string> tags_;
};

} // namespace mlcoupling::smartsim_key_balancing
