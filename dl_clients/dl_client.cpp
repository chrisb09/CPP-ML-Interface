#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "phydll_dl_runtime.hpp"

#ifdef PHYDLL_DL_USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
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
    int32_t num_inputs = 0;
    int32_t num_outputs = 0;
    int64_t total_input = 0;
    int64_t total_output = 0;
    int32_t dtype = 0;
    int32_t layout = 0;
};

struct BcastMeta
{
    bool valid = false;
    std::string model_path;
    std::string backend;
    std::string device;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;
    int64_t total_input = 0;
    int64_t total_output = 0;
};

BcastMeta receive_p2p_metadata(int source_rank)
{
    constexpr int kBcastMetaMagic = 0x4D4C434D; // "MLCM"
    constexpr int kBcastMetaVersion = 1;

    BcastMetaHeader header;
    MPI_Status status;
    MPI_Recv(&header, sizeof(header), MPI_BYTE, source_rank, source_rank, MPI_COMM_WORLD, &status);
    if (header.magic != kBcastMetaMagic || header.version != kBcastMetaVersion)
    {
        return {};
    }

    const size_t payload_size = static_cast<size_t>(header.model_len + header.backend_len + header.device_len) +
                                (static_cast<size_t>(header.num_inputs + header.num_outputs) * sizeof(int64_t));
    std::vector<unsigned char> payload(payload_size);
    if (payload_size > 0)
    {
        MPI_Recv(payload.data(), static_cast<int>(payload.size()), MPI_BYTE, source_rank, source_rank, MPI_COMM_WORLD, &status);
    }

    BcastMeta meta;
    meta.valid = true;
    meta.total_input = header.total_input;
    meta.total_output = header.total_output;

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

    meta.input_sizes.resize(static_cast<size_t>(header.num_inputs));
    meta.output_sizes.resize(static_cast<size_t>(header.num_outputs));

    if (header.num_inputs > 0)
    {
        std::memcpy(meta.input_sizes.data(), payload.data() + offset, header.num_inputs * sizeof(int64_t));
        offset += static_cast<size_t>(header.num_inputs) * sizeof(int64_t);
    }
    if (header.num_outputs > 0)
    {
        std::memcpy(meta.output_sizes.data(), payload.data() + offset, header.num_outputs * sizeof(int64_t));
    }

    return meta;
}
} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int* appnum_ptr = nullptr;
    int appnum_flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum_ptr, &appnum_flag);
    const int app_id = appnum_flag ? *appnum_ptr : 0;

    // Participate in the MPMD split to avoid collective mismatches.
    const int color = (app_id == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &local_comm);

    const int dl_count = get_env_int("PHYDLL_DL_COUNT", 1);

    std::fprintf(stderr, "[PHYDLL:DL] client started argv0=%s dl_count=%d torch=%s\n",
                 (argv && argv[0]) ? argv[0] : "(null)",
                 dl_count,
#ifdef PHYDLL_DL_USE_TORCH
                 "on"
#else
                 "off"
#endif
    );
    std::fflush(stderr);

    bool meta_initialized = false;
    bool model_loaded = false;
    std::string model_path;
    int64_t total_input_size = 0;
    int64_t total_output_size = 0;
    std::string device_name;

#ifdef PHYDLL_DL_USE_TORCH
    torch::jit::script::Module model;
    torch::Device torch_device(torch::kCPU);
#endif

    phydll_dl::DlRuntime runtime(dl_count);
    runtime.initialize();

    std::fprintf(stderr, "[PHYDLL:DL] barrier before metadata\n");
    std::fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    std::fprintf(stderr, "[PHYDLL:DL] barrier done, receiving metadata\n");
    std::fflush(stderr);

    int ndest = phydll_get_ndest();
    int *dests = phydll_get_dest();
    for (int i = 0; i < ndest; ++i)
    {
        int source_rank = dests[i];
        const auto p2p_meta = receive_p2p_metadata(source_rank);
        std::fprintf(stderr, "[PHYDLL:DL] metadata received from rank %d valid=%d\n", source_rank, p2p_meta.valid ? 1 : 0);
        std::fflush(stderr);
        if (p2p_meta.valid)
        {
            if (!meta_initialized)
            {
                model_path = p2p_meta.model_path;
                device_name = p2p_meta.device;
                meta_initialized = true;
            }
            total_input_size += p2p_meta.total_input;
            total_output_size += p2p_meta.total_output;
        }
    }
    std::uint64_t frame_id = 0;
    while (runtime.is_running()) {
        std::fprintf(stderr, "[PHYDLL:DL] frame=%llu before receive\n",
                     static_cast<unsigned long long>(frame_id));
        std::fflush(stderr);

        const auto frame = runtime.receive_frame();

        std::fprintf(stderr, "[PHYDLL:DL] recv done, frame=%llu\n",
                     static_cast<unsigned long long>(frame_id));
        std::fflush(stderr);

        if (frame.has_meta && !meta_initialized && frame.meta.phase == phydll_dl::MetaPhase::Init) {
            model_path = frame.meta.entries.empty() ? std::string() : frame.meta.entries.front().model_path;
            device_name = frame.meta.entries.empty() ? std::string() : frame.meta.entries.front().device;
            total_input_size = 0;
            total_output_size = 0;
            for (const auto& entry : frame.meta.entries) {
                for (const auto size : entry.input_sizes) {
                    total_input_size += size;
                }
                for (const auto size : entry.output_sizes) {
                    total_output_size += size;
                }
            }
            meta_initialized = true;
        }

        if (meta_initialized && !model_loaded) {
#ifdef PHYDLL_DL_USE_TORCH
            const bool wants_gpu = !device_name.empty() && device_name != "CPU" && device_name != "cpu";
            if (wants_gpu) {
                const bool cuda_ok = torch::cuda::is_available() && torch::cuda::device_count() > 0;
                if (cuda_ok) {
                    torch_device = torch::Device(torch::kCUDA, 0);
                } else {
                    std::fprintf(stderr, "[PHYDLL:DL] requested GPU but no CUDA device available; using CPU\n");
                    std::fflush(stderr);
                    torch_device = torch::Device(torch::kCPU);
                }
            } else {
                torch_device = torch::Device(torch::kCPU);
            }

            std::fprintf(stderr, "[PHYDLL:DL] using device=%s\n", torch_device.is_cuda() ? "cuda" : "cpu");
            std::fflush(stderr);

            if (!model_path.empty()) {
                std::fprintf(stderr, "[PHYDLL:DL] loading model '%s'\n", model_path.c_str());
                std::fflush(stderr);
                try {
                    model = torch::jit::load(model_path);
                    model.eval();
                    model.to(torch_device);
                    std::fprintf(stderr, "[PHYDLL:DL] model loaded\n");
                    std::fflush(stderr);
                } catch (const c10::Error &e) {
                    std::cerr << "Failed to load TorchScript model: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                std::fprintf(stderr, "[PHYDLL:DL] no model path; continuing without model\n");
                std::fflush(stderr);
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

        std::vector<double> output(static_cast<size_t>(runtime.field_size()), 0.0);
        bool used_model = false;

#ifdef PHYDLL_DL_USE_TORCH
        if (!model_path.empty()) {
            std::vector<float> input(static_cast<size_t>(total_input_size));
            int ndest = phydll_get_ndest();
            const long long batch_size = std::max(1LL, static_cast<long long>(ndest));
            const long long input_per_rank_total = runtime.field_size() / batch_size;
            const long long input_per_rank_used = static_cast<long long>(total_input_size) / batch_size;
            const long long output_per_rank = runtime.field_size() / batch_size;

            for (long long b = 0; b < batch_size; ++b) {
                for (long long j = 0; j < input_per_rank_used; ++j) {
                    input[b * input_per_rank_used + j] = static_cast<float>(frame.data[b * input_per_rank_total + j]);
                }
            }

            std::fprintf(stderr, "[PHYDLL:DL] input vector size=%zu values=", input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                std::fprintf(stderr, "%s%.6f", (i == 0 ? "" : ","), input[i]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto input_tensor = torch::from_blob(input.data(), {batch_size, input_per_rank_used}, options).clone();
            input_tensor = input_tensor.to(torch_device);
            try {
                std::fprintf(stderr, "[PHYDLL:DL] forward input shape=[%lld,%lld] device=%s\n",
                             static_cast<long long>(input_tensor.size(0)),
                             static_cast<long long>(input_tensor.size(1)),
                             input_tensor.is_cuda() ? "cuda" : "cpu");
                std::fflush(stderr);

                auto output_tensor = model.forward({input_tensor}).toTensor();
                output_tensor = output_tensor.to(torch::kCPU).contiguous().view({-1});

                auto output_ptr = output_tensor.data_ptr<float>();
                const long long outputs_per_rank_used = static_cast<long long>(total_output_size) / batch_size;
                const long long output_stride = static_cast<long long>(runtime.field_size()) / batch_size;
                for (long long b = 0; b < batch_size; ++b) {
                    for (long long j = 0; j < outputs_per_rank_used; ++j) {
                        output[b * output_stride + j] = static_cast<double>(output_ptr[b * outputs_per_rank_used + j]);
                    }
                }
                used_model = true;
            } catch (const c10::Error &e) {
                std::fprintf(stderr, "[PHYDLL:DL] forward failed: %s\n", e.what());
                std::fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "[PHYDLL:DL] forward exception: %s\n", e.what());
                std::fflush(stderr);
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

        std::fprintf(stderr, "[PHYDLL:DL] send_output size=%zu\n", output.size());
        std::fflush(stderr);
        runtime.send_output(output);

        ++frame_id;
    }

    phydll_finalize();
    MPI_Finalize();
    return 0;
}
