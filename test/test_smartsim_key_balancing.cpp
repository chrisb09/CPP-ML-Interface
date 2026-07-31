#include "smartsim_key_balancing.hpp"

#include <array>
#include <cassert>
#include <string>

using mlcoupling::smartsim_key_balancing::RedisKeyBalancer;
using mlcoupling::smartsim_key_balancing::crc16_xmodem;
using mlcoupling::smartsim_key_balancing::redis_hash_slot;

int main()
{
    assert(crc16_xmodem("123456789") == 0x31C3U);

    const RedisKeyBalancer disabled(0, 4, 4, false);
    assert(disabled.prefix_key("input_0") == "input_0");

    std::array<std::array<int, 4>, 4> four_node_counts{};
    for (int rank = 0; rank < 96; ++rank) {
        const RedisKeyBalancer balancer(rank, 4, 4, true);
        const int shard = balancer.target_shard();
        const int gpu = rank % 4;
        ++four_node_counts[static_cast<std::size_t>(shard)][static_cast<std::size_t>(gpu)];

        for (int tag_shard = 0; tag_shard < 4; ++tag_shard) {
            const auto slot = redis_hash_slot(balancer.tags().at(static_cast<std::size_t>(tag_shard)));
            assert(slot >= RedisKeyBalancer::slot_first(tag_shard, 4));
            assert(slot <= RedisKeyBalancer::slot_last(tag_shard, 4));
        }
        const std::string key = balancer.prefix_key("input_0");
        assert(key.rfind("{", 0) == 0);
    }
    for (const auto& per_shard : four_node_counts) {
        for (const int count : per_shard) {
            assert(count == 6);
        }
    }

    std::array<std::array<int, 2>, 3> three_node_counts{};
    for (int rank = 0; rank < 18; ++rank) {
        const RedisKeyBalancer balancer(rank, 2, 3, true);
        ++three_node_counts[static_cast<std::size_t>(balancer.target_shard())]
                           [static_cast<std::size_t>(rank % 2)];
    }
    for (const auto& per_shard : three_node_counts) {
        for (const int count : per_shard) {
            assert(count == 3);
        }
    }

    std::array<int, 4> cpu_node_counts{};
    for (int rank = 0; rank < 16; ++rank) {
        const RedisKeyBalancer balancer(rank, 0, 4, true);
        ++cpu_node_counts[static_cast<std::size_t>(balancer.target_shard())];
    }
    for (const int count : cpu_node_counts) {
        assert(count == 4);
    }
}
