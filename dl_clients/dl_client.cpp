#include <iostream>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>

#include "phydll_dl_runtime.hpp"

#ifdef PHYDLL_DL_USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#if defined(USE_SCOREP) || defined(WITH_CUDA)
#include <c10/cuda/CUDAStream.h>
#endif
#endif

#ifdef USE_SCOREP
#if __has_include(<scorep/SCOREP_User.h>)
#include <scorep/SCOREP_User.h>
#elif __has_include(<SCOREP_User.h>)
#include <SCOREP_User.h>
#endif
SCOREP_USER_REGION_DEFINE(handle_dl_input_unpack);
SCOREP_USER_REGION_DEFINE(handle_dl_output_allocate);
SCOREP_USER_REGION_DEFINE(handle_dl_input_allocate);
SCOREP_USER_REGION_DEFINE(handle_dl_h2d);
SCOREP_USER_REGION_DEFINE(handle_dl_torch_forward);
SCOREP_USER_REGION_DEFINE(handle_dl_d2h);
SCOREP_USER_REGION_DEFINE(handle_dl_output_reorder);
SCOREP_USER_REGION_DEFINE(handle_dl_send_output);
#endif

namespace {
int get_env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

struct BcastMetaHeader
{
    int32_t magic = 0;
    int32_t version = 0;
    int32_t model_len = 0;
    int32_t backend_len = 0;
    int32_t device_len = 0;
    int32_t batch_size = 0;
    int32_t num_inputs = 0;
    int32_t num_outputs = 0;
    int64_t total_input = 0;
    int64_t total_output = 0;
    int32_t dtype = 0;
    int32_t layout = 0;
    int32_t num_input_dims = 0;
    int32_t num_output_dims = 0;
    int32_t layout_kind = 0;   // 0 = packed, 1 = uniform_chunks
    int32_t phy_count = 0;     // fields sent by the source PHY rank
    int32_t dl_count = 0;      // fields the source PHY rank expects back
    int64_t field_size = 0;    // per-field per-source size in doubles
};

struct BcastMeta
{
    bool valid = false;
    std::string model_path;
    std::string backend;
    std::string device;
    int batch_size = 0;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    int64_t total_input = 0;
    int64_t total_output = 0;
    int64_t field_size = 0;
    int layout_kind = 0;
    int phy_count = 0;
    int dl_count = 0;
};

BcastMeta receive_p2p_metadata(int source_rank)
{
    constexpr int kBcastMetaMagic = 0x4D4C434D; // "MLCM"
    constexpr int kBcastMetaVersion = 3;

    BcastMetaHeader header;
    MPI_Status status;
    std::fprintf(stderr, "[PHYDLL:DL] MPI_Recv header from source_rank=%d\n", source_rank); std::fflush(stderr);
    MPI_Recv(&header, sizeof(header), MPI_BYTE, source_rank, source_rank, MPI_COMM_WORLD, &status);
    std::fprintf(stderr, "[PHYDLL:DL] MPI_Recv header from source_rank=%d DONE (magic=%x)\n", source_rank, header.magic); std::fflush(stderr);

    if (header.magic != kBcastMetaMagic || header.version != kBcastMetaVersion)
    {
        return {};
    }

    const size_t payload_size = static_cast<size_t>(header.model_len + header.backend_len + header.device_len) +
                                (static_cast<size_t>(header.num_input_dims + header.num_output_dims) * sizeof(int64_t));
    std::fprintf(stderr, "[PHYDLL:DL] Allocating payload size %zu\n", payload_size); std::fflush(stderr);
    std::vector<unsigned char> payload(payload_size);
    if (payload_size > 0)
    {
        std::fprintf(stderr, "[PHYDLL:DL] MPI_Recv payload from source_rank=%d\n", source_rank); std::fflush(stderr);
        MPI_Recv(payload.data(), static_cast<int>(payload.size()), MPI_BYTE, source_rank, source_rank, MPI_COMM_WORLD, &status);
        std::fprintf(stderr, "[PHYDLL:DL] MPI_Recv payload from source_rank=%d DONE\n", source_rank); std::fflush(stderr);
    }

    BcastMeta meta;
    meta.valid = true;
    meta.total_input = header.total_input;
    meta.total_output = header.total_output;
    meta.batch_size = header.batch_size;
    meta.field_size = header.field_size;
    meta.layout_kind = header.layout_kind;
    meta.phy_count = header.phy_count;
    meta.dl_count = header.dl_count;

    size_t offset = 0;
    if (header.model_len > 0)
    {
        meta.model_path.assign(reinterpret_cast<const char *>(payload.data() + offset), header.model_len);
        offset += static_cast<size_t>(header.model_len);
    }
    if (header.backend_len > 0)
    {
        meta.backend.assign(reinterpret_cast<const char *>(payload.data() + offset), header.backend_len);
        offset += static_cast<size_t>(header.backend_len);
    }
    if (header.device_len > 0)
    {
        meta.device.assign(reinterpret_cast<const char *>(payload.data() + offset), header.device_len);
        offset += static_cast<size_t>(header.device_len);
    }

    if (header.num_input_dims > 0)
    {
        std::vector<int64_t> flat_in_dims(header.num_input_dims);
        std::memcpy(flat_in_dims.data(), payload.data() + offset, header.num_input_dims * sizeof(int64_t));
        offset += static_cast<size_t>(header.num_input_dims) * sizeof(int64_t);

        size_t d_idx = 0;
        for (int i = 0; i < header.num_inputs && d_idx < flat_in_dims.size(); ++i) {
            int64_t ndim = flat_in_dims[d_idx++];
            std::vector<int64_t> shape;
            for (int64_t d = 0; d < ndim && d_idx < flat_in_dims.size(); ++d) {
                shape.push_back(flat_in_dims[d_idx++]);
            }
            meta.input_shapes.push_back(shape);
        }
    }
    if (header.num_output_dims > 0)
    {
        std::vector<int64_t> flat_out_dims(header.num_output_dims);
        std::memcpy(flat_out_dims.data(), payload.data() + offset, header.num_output_dims * sizeof(int64_t));
        
        size_t d_idx = 0;
        for (int i = 0; i < header.num_outputs && d_idx < flat_out_dims.size(); ++i) {
            int64_t ndim = flat_out_dims[d_idx++];
            std::vector<int64_t> shape;
            for (int64_t d = 0; d < ndim && d_idx < flat_out_dims.size(); ++d) {
                shape.push_back(flat_out_dims[d_idx++]);
            }
            meta.output_shapes.push_back(shape);
        }
    }

    return meta;
}
} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Exclude this DL rank from solver-only collectives. PhyDLL subsequently performs
    // its own split on MPI_COMM_WORLD for the physics/DL communicators.
    const int color = MPI_UNDEFINED;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &local_comm);

    int local_dl_rank = 0;

    const int dl_count = get_env_int("PHYDLL_DL_FIELD_COUNT", get_env_int("PHYDLL_DL_COUNT", 1));

#ifdef PHYDLL_DL_USE_TORCH
    const int intra_threads = get_env_int("MLCOUPLING_INTRA_OP_THREADS", get_env_int("SLURM_CPUS_PER_TASK", -1));
    const int inter_threads = get_env_int("MLCOUPLING_INTER_OP_THREADS", -1);

    if (intra_threads > 0) {
        torch::set_num_threads(intra_threads);
    }
    if (inter_threads > 0) {
        torch::set_num_interop_threads(inter_threads);
    }
#endif

    std::fprintf(stderr, "[PHYDLL:DL] client started argv0=%s dl_count=%d torch=%s\n",
                 (argv && argv[0]) ? argv[0] : "(null)",
                 dl_count,
#ifdef PHYDLL_DL_USE_TORCH
                 "on"
#else
                 "off"
#endif
    );
#ifdef PHYDLL_DL_USE_TORCH
    if (intra_threads > 0 || inter_threads > 0) {
        std::fprintf(stderr, "[PHYDLL:DL] torch threads: intra=%d, inter=%d\n", 
                     (int)torch::get_num_threads(), (int)torch::get_num_interop_threads());
    }
#endif
    std::fflush(stderr);

    bool meta_initialized = false;
    bool model_loaded = false;
    std::string model_path;
    int64_t total_input_size = 0;
    int64_t total_output_size = 0;
    std::string device_name;
    BcastMeta final_meta;

#ifdef PHYDLL_DL_USE_TORCH
    torch::jit::script::Module model;
    torch::Device torch_device(torch::kCPU);
#endif

    phydll_dl::DlRuntime runtime(dl_count);
    runtime.initialize();

    std::fprintf(stderr, "[PHYDLL:DL] getting ndest\n"); std::fflush(stderr);
    int ndest = phydll_get_ndest();
    int *dests = phydll_get_dest();
    std::fprintf(stderr, "[PHYDLL:DL] ndest=%d, dests=%p\n", ndest, (void*)dests); std::fflush(stderr);
    std::vector<long long> rank_batch_sizes(ndest, 1);
    std::vector<long long> rank_field_sizes(ndest, 0);
    std::vector<long long> rank_total_input(ndest, 0);
    std::vector<long long> rank_total_output(ndest, 0);
    std::vector<int> rank_layout_kind(ndest, 0);
    std::vector<int> rank_phy_count(ndest, 0);
    for (int i = 0; i < ndest; ++i)
    {
        int source_rank = dests[i];
        std::fprintf(stderr, "[PHYDLL:DL] receiving metadata from dests[%d] = %d\n", i, source_rank); std::fflush(stderr);
        const auto p2p_meta = receive_p2p_metadata(source_rank);
        if (p2p_meta.valid)
        {
            if (!meta_initialized)
            {
                model_path = p2p_meta.model_path;
                device_name = p2p_meta.device;
                final_meta = p2p_meta;
                meta_initialized = true;
            }
            total_input_size += p2p_meta.total_input;
            total_output_size += p2p_meta.total_output;
            if (!p2p_meta.input_shapes.empty() && !p2p_meta.input_shapes.front().empty()) {
                rank_batch_sizes[i] = p2p_meta.input_shapes.front().front();
            }
            rank_field_sizes[i] = p2p_meta.field_size;
            rank_total_input[i] = p2p_meta.total_input;
            rank_total_output[i] = p2p_meta.total_output;
            rank_layout_kind[i] = p2p_meta.layout_kind;
            rank_phy_count[i] = p2p_meta.phy_count;
        }
    }
    std::fprintf(stderr, "[PHYDLL:DL] Finished receiving metadata from all %d sources\n", ndest); std::fflush(stderr);

    const bool uniform_chunks = rank_layout_kind[0] == 1;
    for (int i = 0; i < ndest; ++i)
    {
        if (rank_layout_kind[i] != rank_layout_kind[0])
        {
            std::fprintf(stderr, "[PHYDLL:DL] ERROR: mixed transport layouts across coupled ranks (rank %d: %d, rank 0: %d).\n",
                         i, rank_layout_kind[i], rank_layout_kind[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (uniform_chunks && rank_phy_count[i] != rank_phy_count[0])
        {
            std::fprintf(stderr, "[PHYDLL:DL] ERROR: uniform_chunks requires identical PHY field counts across ranks (rank %d: %d, rank 0: %d).\n",
                         i, rank_phy_count[i], rank_phy_count[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (uniform_chunks)
    {
        if (runtime.dl_count() != final_meta.dl_count)
        {
            std::fprintf(stderr,
                         "[PHYDLL:DL] ERROR: uniform_chunks provider expects dl_count=%d but the client was launched with PHYDLL_DL_FIELD_COUNT=%d.\n",
                         final_meta.dl_count, runtime.dl_count());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (runtime.phy_count() != rank_phy_count[0])
        {
            std::fprintf(stderr,
                         "[PHYDLL:DL] ERROR: uniform_chunks PHY field count mismatch: PhyDLL reports %d, provider sent %d.\n",
                         runtime.phy_count(), rank_phy_count[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    std::uint64_t frame_id = 0;
    while (runtime.is_running()) {
        std::fprintf(stderr, "[PHYDLL:DL] Waiting for frame %llu\n", (unsigned long long)frame_id); std::fflush(stderr);
        const auto frame = runtime.receive_frame();
        std::fprintf(stderr, "[PHYDLL:DL] Received frame %llu (has_meta=%d, data_size=%zu)\n", 
                     (unsigned long long)frame_id, frame.has_meta, frame.data.size()); 
        std::fflush(stderr);

        if (frame.has_meta && !meta_initialized && frame.meta.phase == phydll_dl::MetaPhase::Init) {
            model_path = frame.meta.entries.empty() ? std::string() : frame.meta.entries.front().model_path;
            device_name = frame.meta.entries.empty() ? std::string() : frame.meta.entries.front().device;
            total_input_size = 0;
            total_output_size = 0;
            // Note: frame.meta uses old flattened sizes. 
            // In a complete implementation we'd update frame.meta.entries parsing too, 
            // but forSTATIC api mode, total_input_size is already populated correctly via p2p_meta.
            rank_batch_sizes.resize(frame.meta.entries.size(), 1);
            for (size_t i = 0; i < frame.meta.entries.size(); ++i) {
                const auto& entry = frame.meta.entries[i];
                for (const auto size : entry.input_sizes) {
                    total_input_size += size;
                }
                for (const auto size : entry.output_sizes) {
                    total_output_size += size;
                }
                rank_batch_sizes[i] = 1;
            }
            meta_initialized = true;
        }

        if (meta_initialized && !model_loaded) {
#ifdef PHYDLL_DL_USE_TORCH
            const bool wants_gpu = !device_name.empty() && device_name != "CPU" && device_name != "cpu";
            if (wants_gpu) {
                // Query the communicator containing only the DL ranks (after initialization)
                MPI_Comm dl_comm = phydll_get_local_mpi_comm();
                MPI_Comm local_dl_comm = MPI_COMM_NULL;
                MPI_Comm_split_type(dl_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_dl_comm);
                MPI_Comm_rank(local_dl_comm, &local_dl_rank);
                MPI_Comm_free(&local_dl_comm);

                const bool cuda_ok = torch::cuda::is_available() && torch::cuda::device_count() > 0;
                if (cuda_ok) {
                    const int local_gpu_count = torch::cuda::device_count();
                    int gpu_id = local_dl_rank % local_gpu_count;
                    gpu_id = get_env_int("PHYDLL_DL_GPU_ID", gpu_id);
                    std::cerr << "[PHYDLL:DL] Using local DL rank " << local_dl_rank 
                              << " mapped to GPU device index: " << gpu_id << std::endl;
                    torch_device = torch::Device(torch::kCUDA, gpu_id);
                } else {
                    std::cerr << "[PHYDLL:DL] requested GPU but no CUDA device available; using CPU" << std::endl;
                    torch_device = torch::Device(torch::kCPU);
                }
            } else {
                torch_device = torch::Device(torch::kCPU);
            }

            if (!model_path.empty()) {
                try {
                    model = torch::jit::load(model_path);
                    model.eval();
                    model.to(torch_device);
                    std::cerr << "[PHYDLL:DL] model loaded" << std::endl;
                } catch (const c10::Error &e) {
                    std::cerr << "Failed to load TorchScript model: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
#else
            if (!model_path.empty()) {
                std::cerr << "Model path provided, but libtorch is not enabled." << std::endl;
            }
#endif

            std::fprintf(stderr, "[PHYDLL:DL] meta init model_path='%s' total_input=%lld total_output=%lld\n",
                         model_path.c_str(),
                         static_cast<long long>(total_input_size),
                         static_cast<long long>(total_output_size));
            std::fflush(stderr);
            model_loaded = true;
        }

        if (!meta_initialized) {
            continue;
        }

        const bool profile_details = frame_id > 0;
#ifdef USE_SCOREP
        if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_output_allocate, "dl_output_allocate", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        std::vector<double> output(static_cast<size_t>(runtime.field_size()) * static_cast<size_t>(runtime.dl_count()), 0.0);
#ifdef USE_SCOREP
        if (profile_details) SCOREP_USER_REGION_END(handle_dl_output_allocate);
#endif
        bool used_model = false;

#ifdef PHYDLL_DL_USE_TORCH
        if (!model_path.empty()) {
            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_input_allocate, "dl_input_allocate", SCOREP_USER_REGION_TYPE_COMMON);
            #endif
            std::vector<float> input(static_cast<size_t>(total_input_size));
            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_END(handle_dl_input_allocate);
            #endif
            int ndest = phydll_get_ndest();
            long long batch_size = 0;
            for (int i = 0; i < ndest; ++i) {
                batch_size += rank_batch_sizes[i];
            }
            const long long field_size_per_rank = runtime.field_size() / std::max(1, ndest);
            const long long input_per_rank_used = static_cast<long long>(total_input_size) / batch_size;

            std::fprintf(stderr, "[PHYDLL:DL] Frame %llu extracting data: total_input_size=%lld, batch=%lld, field/rank=%lld, used/rank=%lld\n",
                         (unsigned long long)frame_id, (long long)total_input_size, batch_size, field_size_per_rank, input_per_rank_used);
            std::fflush(stderr);

            long long offset_so_far = 0;
            long long src_rank_start = 0;
            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_input_unpack, "dl_input_unpack", SCOREP_USER_REGION_TYPE_COMMON);
            #endif
            if (uniform_chunks) {
                // combined frame is field-major:
                //   field f = [rank0 chunk f][rank1 chunk f]...
                // with per-rank chunk sizes rank_field_sizes[i] and aggregated
                // field size runtime.field_size() = sum_i rank_field_sizes[i].
                const long long phy_count = rank_phy_count[0];
                const long long agg = runtime.field_size();
                std::vector<long long> agg_offsets(ndest + 1, 0);
                std::vector<long long> in_offsets(ndest + 1, 0);
                for (int i = 0; i < ndest; ++i) {
                    agg_offsets[i + 1] = agg_offsets[i] + rank_field_sizes[i];
                    in_offsets[i + 1] = in_offsets[i] + rank_total_input[i];
                }
                for (int i = 0; i < ndest; ++i) {
                    const long long g_r = rank_field_sizes[i];
                    for (long long f = 0; f < phy_count; ++f) {
                        const long long src_base = f * agg + agg_offsets[i];
                        const long long dst_base = in_offsets[i] + f * g_r;
                        for (long long b = 0; b < g_r; ++b) {
                            input[static_cast<size_t>(dst_base + b)] =
                                static_cast<float>(frame.data[static_cast<size_t>(src_base + b)]);
                        }
                    }
                }
            } else {
                for (int i = 0; i < ndest; ++i) {
                    long long rank_batch = rank_batch_sizes[i];
                    for (long long s = 0; s < rank_batch; ++s) {
                        long long src_start = src_rank_start + s * input_per_rank_used;
                        long long dest_start = (offset_so_far + s) * input_per_rank_used;
                        for (long long j = 0; j < input_per_rank_used; ++j) {
                            input[dest_start + j] = static_cast<float>(frame.data[src_start + j]);
                        }
                    }
                    offset_so_far += rank_batch;
                    src_rank_start += rank_field_sizes[i];
                }
            }
            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_END(handle_dl_input_unpack);
            #endif

            std::fprintf(stderr, "[PHYDLL:DL] Frame %llu running inference\n", (unsigned long long)frame_id); std::fflush(stderr);

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            std::vector<int64_t> actual_shape = {batch_size};
            if (!final_meta.input_shapes.empty()) {
                const auto& shape = final_meta.input_shapes.front();
                // It might include a batch dimension from the PHY side, which we replace with batch_size.
                // Assuming shape[0] is the batch dimension.
                if (shape.size() > 1) {
                    for (size_t d = 1; d < shape.size(); ++d) {
                        actual_shape.push_back(shape[d]);
                    }
                } else {
                    actual_shape.push_back(input_per_rank_used);
                }
            } else {
                actual_shape.push_back(input_per_rank_used);
            }

            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_h2d, "dl_h2d", SCOREP_USER_REGION_TYPE_COMMON);
            #endif
            auto input_tensor = torch::from_blob(input.data(), {batch_size, input_per_rank_used}, options).clone();
            input_tensor = input_tensor.view(actual_shape);
            input_tensor = input_tensor.to(torch_device);
            if (torch_device.is_cuda()) {
                c10::cuda::getCurrentCUDAStream(torch_device.index()).synchronize();
            }
            #ifdef USE_SCOREP
            if (profile_details) SCOREP_USER_REGION_END(handle_dl_h2d);
            #endif
            try {
                torch::NoGradGuard no_grad;
                long long max_chunk_size = final_meta.batch_size > 0 ? static_cast<long long>(final_meta.batch_size) : batch_size;
                std::vector<torch::Tensor> outputs;
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_torch_forward, "dl_torch_forward", SCOREP_USER_REGION_TYPE_COMMON);
                #endif
                for (long long chunk_idx = 0; chunk_idx < batch_size; chunk_idx += max_chunk_size) {
                    long long chunk_size = std::min(max_chunk_size, batch_size - chunk_idx);
                    auto chunk_tensor = input_tensor.slice(0, chunk_idx, chunk_idx + chunk_size);
                    outputs.push_back(model.forward({chunk_tensor}).toTensor());
                }
                auto output_tensor = torch::cat(outputs, 0);
                if (torch_device.is_cuda()) {
                    c10::cuda::getCurrentCUDAStream(torch_device.index()).synchronize();
                }
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_END(handle_dl_torch_forward);
                #endif
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_d2h, "dl_d2h", SCOREP_USER_REGION_TYPE_COMMON);
                #endif
                output_tensor = output_tensor.to(torch::kCPU).contiguous().view({-1});
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_END(handle_dl_d2h);
                #endif

                auto output_ptr = output_tensor.data_ptr<float>();
                const long long outputs_per_rank_used = static_cast<long long>(total_output_size) / batch_size;

                const int64_t produced = static_cast<int64_t>(output_tensor.numel());
                if (produced != total_output_size) {
                    std::fprintf(stderr,
                                 "[PHYDLL:DL] ERROR: Torch model produced %lld elements but metadata declared %lld. Refusing to send mismatched output.\n",
                                 (long long)produced, (long long)total_output_size);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                offset_so_far = 0;
                long long dest_rank_start = 0;
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_BEGIN(handle_dl_output_reorder, "dl_output_reorder", SCOREP_USER_REGION_TYPE_COMMON);
                #endif
                if (uniform_chunks) {
                    const long long dl_count = runtime.dl_count();
                    const long long agg = runtime.field_size();
                    std::vector<long long> agg_offsets(ndest + 1, 0);
                    std::vector<long long> out_offsets(ndest + 1, 0);
                    for (int i = 0; i < ndest; ++i) {
                        agg_offsets[i + 1] = agg_offsets[i] + rank_field_sizes[i];
                        out_offsets[i + 1] = out_offsets[i] + rank_total_output[i];
                    }
                    for (int i = 0; i < ndest; ++i) {
                        const long long g_r = rank_field_sizes[i];
                        for (long long f = 0; f < dl_count; ++f) {
                            const long long src_base = out_offsets[i] + f * g_r;
                            const long long dst_base = f * agg + agg_offsets[i];
                            for (long long b = 0; b < g_r; ++b) {
                                output[static_cast<size_t>(dst_base + b)] =
                                    static_cast<double>(output_ptr[static_cast<size_t>(src_base + b)]);
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < ndest; ++i) {
                        long long rank_batch = rank_batch_sizes[i];
                        for (long long s = 0; s < rank_batch; ++s) {
                            long long dest_start = dest_rank_start + s * outputs_per_rank_used;
                            long long src_start = (offset_so_far + s) * outputs_per_rank_used;
                            for (long long j = 0; j < outputs_per_rank_used; ++j) {
                                output[dest_start + j] = static_cast<double>(output_ptr[src_start + j]);
                            }
                        }
                        offset_so_far += rank_batch;
                        dest_rank_start += rank_field_sizes[i];
                    }
                }
                #ifdef USE_SCOREP
                if (profile_details) SCOREP_USER_REGION_END(handle_dl_output_reorder);
                #endif
                used_model = true;
            } catch (const c10::Error &e) {
                std::cerr << "[PHYDLL:DL] forward failed: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            } catch (const std::exception &e) {
                std::cerr << "[PHYDLL:DL] forward exception: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
#endif

        if (!used_model) {
            const size_t out_copy = std::min(frame.data.size(), output.size());
            for (size_t i = 0; i < out_copy; ++i) {
                output[i] = -frame.data[i];
            }
        }

        #ifdef USE_SCOREP
        if (frame_id > 0) SCOREP_USER_REGION_BEGIN(handle_dl_send_output, "dl_send_output", SCOREP_USER_REGION_TYPE_COMMON);
        #endif
        runtime.send_output(output);
        #ifdef USE_SCOREP
        if (frame_id > 0) SCOREP_USER_REGION_END(handle_dl_send_output);
        #endif
        std::fprintf(stderr, "[PHYDLL:DL] sent output for frame %llu\n", (unsigned long long)frame_id); std::fflush(stderr);

        ++frame_id;
    }
    std::fprintf(stderr, "[PHYDLL:DL] exited main loop after %llu frames\n", (unsigned long long)frame_id); std::fflush(stderr);

    phydll_finalize();
    if (const char* barrier_env = std::getenv("PHYDLL_MPMD_SHUTDOWN_BARRIER");
        barrier_env != nullptr && std::strcmp(barrier_env, "1") == 0) {
        std::fprintf(stderr, "[PHYDLL:DL] waiting for solver teardown\n");
        std::fflush(stderr);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    std::fprintf(stderr, "[PHYDLL:DL] client exiting cleanly\n");
    std::fflush(stderr);
    return 0;
}
