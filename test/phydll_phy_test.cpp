/**
 * phydll_phy_test.cpp
 *
 * MPMD regression harness for the C++ PhyDLL DL client.
 *
 * Each physics (solver) rank wraps deterministic tensors and drives
 * MLCouplingProviderPhydll directly through two static_inference calls,
 * verifying the received outputs numerically after each call.
 *
 * Usage:
 *   phydll_phy_test <mode> <model.pt> <transport_layout> <batch_chunk>
 *
 *   mode            18to1 | 1to18
 *   transport_layout  packed | uniform_chunks
 *   batch_chunk      DL inference chunk size (0 = full batch)
 *
 * Local batch B: rank 0 -> B=3, all other ranks -> B=2.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "tool.h"
#include "provider/ml_coupling_provider_phydll.hpp"

namespace {

bool g_any_failure = false;

int g_rank = 0;
int g_world_size = 1;

std::string g_mode;
std::string g_model;
std::string g_layout;
int g_batch_chunk = 0;

int local_batch() { return g_rank == 0 ? 3 : 2; }

long long feature_offset(int rank, int b) {
    // Deterministic per-rank base so cross-rank values are distinguishable.
    return 1000LL * rank + b;
}

bool check_sum18(const std::vector<float>& out) {
    const int B = local_batch();
    bool ok = true;
    for (int b = 0; b < B; ++b) {
        // x[b,j] = 18*base + j  (base = 1000*rank + b)
        // sum_j = 18*(18*base) + sum_{j=0}^{17} j = 324*base + 153
        const float expected = static_cast<float>(324 * feature_offset(g_rank, b) + 153);
        if (std::abs(out[static_cast<size_t>(b)] - expected) > 1e-3f) {
            std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " SUM18 mismatch at b=" << b
                      << " got=" << out[static_cast<size_t>(b)] << " expected=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}

bool check_expand18(const std::vector<float>& out) {
    const int B = local_batch();
    bool ok = true;
    for (int b = 0; b < B; ++b) {
        const float expected = static_cast<float>(feature_offset(g_rank, b));
        for (int j = 0; j < 18; ++j) {
            const float got = out[static_cast<size_t>(b * 18 + j)];
            if (std::abs(got - expected) > 1e-3f) {
                std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " EXPAND18 mismatch at b=" << b
                          << " j=" << j << " got=" << got << " expected=" << expected << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);

    // Required MPMD contract (see mini_app/CPP-ML-Interface-MPMD-Fix.md):
    // the solver forms its own isolated communicator with color 0 while the
    // DL client opts out with MPI_UNDEFINED. Without this matching split the
    // DL client's MPI_Comm_split blocks forever.
    const int color = 0;
    MPI_Comm solver_app_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, g_rank, &solver_app_comm);
    if (solver_app_comm == MPI_COMM_NULL) {
        std::cerr << "solver rank got MPI_COMM_NULL from split; aborting\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <mode> <model.pt> <transport_layout> <batch_chunk>\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    g_mode = argv[1];
    g_model = argv[2];
    g_layout = argv[3];
    g_batch_chunk = std::atoi(argv[4]);

    const int B = local_batch();
    std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " world=" << g_world_size
              << " mode=" << g_mode << " B=" << B << " layout=" << g_layout << "\n";

    const bool is_18to1 = (g_mode == "18to1");
    const int n_in = is_18to1 ? B * 18 : B * 1;
    const int n_out = is_18to1 ? B * 1 : B * 18;

    std::vector<float> in_buf(static_cast<size_t>(n_in));
    std::vector<float> out_buf(static_cast<size_t>(n_out), 0.0f);

    if (is_18to1) {
        for (int b = 0; b < B; ++b) {
            const long long base = feature_offset(g_rank, b);
            for (int j = 0; j < 18; ++j) {
                in_buf[static_cast<size_t>(b * 18 + j)] = static_cast<float>(18 * base + j);
            }
        }
    } else {
        for (int b = 0; b < B; ++b) {
            in_buf[static_cast<size_t>(b)] = static_cast<float>(feature_offset(g_rank, b));
        }
    }

    MLCouplingData<float> in_data;
    in_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        in_buf.data(),
        is_18to1 ? std::vector<int>{B, 18} : std::vector<int>{B, 1},
        MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal));
    MLCouplingData<float> out_data;
    out_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        out_buf.data(),
        is_18to1 ? std::vector<int>{B} : std::vector<int>{B, 18},
        MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal));

    try {
        MLCouplingProviderPhydll<float, float> provider(g_model, "TORCH", "CPU", g_batch_chunk, g_layout);

        for (int step = 0; step < 2; ++step) {
            std::fill(out_buf.begin(), out_buf.end(), 0.0f);
            provider.static_inference(&in_data, &out_data);

            const bool ok = is_18to1 ? check_sum18(out_buf) : check_expand18(out_buf);
            if (!ok) {
                g_any_failure = true;
            }
            std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " step=" << step
                      << (ok ? " OK" : " MISMATCH") << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " provider error: " << e.what() << "\n";
        g_any_failure = true;
    }
    std::cerr << "[PHYDLL:PHY] rank=" << g_rank << " provider destroyed, before finalize\n";
    std::fflush(stderr);

    int local_fail = g_any_failure ? 1 : 0;
    int global_fail = 0;
    // Reduce only over solver ranks; MPI_COMM_WORLD also contains the DL rank,
    // which is already tearing down and would never join this collective.
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, solver_app_comm);
    MPI_Comm_free(&solver_app_comm);
    MPI_Finalize();
    return global_fail;
}
