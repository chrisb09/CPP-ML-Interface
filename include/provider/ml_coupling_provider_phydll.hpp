#pragma once

#include "../library/ml_coupling_library.hpp"
#include "../scorep_profiling_state.hpp"
#include "../data/ml_coupling_data_type.hpp"
#include "../data/ml_coupling_memory_layout.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

#ifdef WITH_PHYDLL
#include <mpi.h>
#endif

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

#ifdef WITH_PHYDLL
extern "C"
{
#include "phydll.h"
    int* phydll_get_dest();
    int phydll_get_ndest();
}
#endif

// @registry_name: Phydll
// @registry_aliases: phydll, PhyDLL
template <typename In, typename Out>
class MLCouplingLibraryPhydll : public MLCouplingLibrary<In, Out>
{

public:
    MLCouplingLibraryPhydll(std::string model_file,
                             std::string backend = "TORCH",
                             std::string device = "GPU",
                             int batch_size = 0,
                             std::string transport_layout = "auto",
                             MLCouplingData<In> *input_after_preprocessing = nullptr,
                             MLCouplingData<Out> *output_before_postprocessing = nullptr)
                : model_file(std::move(model_file)),
                    backend(std::move(backend)),
                    device(std::move(device)),
                    batch_size(batch_size),
                    input_after_preprocessing(input_after_preprocessing),
                    output_before_postprocessing(output_before_postprocessing)
    {
        if (transport_layout == "auto" || transport_layout.empty())
        {
            layout_config_ = TransportLayoutConfig::Auto;
        }
        else if (transport_layout == "uniform_chunks")
        {
            layout_config_ = TransportLayoutConfig::UniformChunks;
        }
        else if (transport_layout == "packed")
        {
            layout_config_ = TransportLayoutConfig::Packed;
        }
        else
        {
            guarantee(false, ("PhyDLL provider: unknown transport_layout '" + transport_layout +
                                 "'. Supported values are 'auto', 'uniform_chunks', and 'packed'.").c_str());
        }
#ifndef WITH_PHYDLL
        guarantee(false, "PhyDLL provider is not enabled. Please make sure WITH_PHYDLL is defined and the necessary dependencies are installed.");
#else
        initialize_phydll_if_needed();
        if (this->input_after_preprocessing && this->output_before_postprocessing) {
            initialize_if_needed();
        }
#endif
    }

    ~MLCouplingLibraryPhydll() override
    {
#ifdef WITH_PHYDLL
        if (phydll_initialized_)
        {
            phydll_finalize();
        }
#endif
    }

    void static_inference(MLCouplingData<In> *input_after_preprocessing,
                          MLCouplingData<Out> *output_before_postprocessing) override
    {
        guarantee(input_after_preprocessing != nullptr, "PhyDLL inference requires input_after_preprocessing.");
        guarantee(output_before_postprocessing != nullptr, "PhyDLL inference requires output_before_postprocessing.");

#ifdef WITH_PHYDLL
        this->input_after_preprocessing = input_after_preprocessing;
        this->output_before_postprocessing = output_before_postprocessing;
        initialize_if_needed();

#ifdef USE_SCOREP
        const bool profile_details = ml_coupling_scorep::detailed_regions_are_enabled();
        SCOREP_USER_REGION_DEFINE(handle_phydll_prepack)
         SCOREP_USER_REGION_DEFINE(handle_phydll_send)
         SCOREP_USER_REGION_DEFINE(handle_phydll_recv)
         SCOREP_USER_REGION_DEFINE(handle_phydll_unpack)
         SCOREP_USER_REGION_DEFINE(handle_phydll_library_static_step)
        SCOREP_USER_METRIC_LOCAL(bytes_sent_logical);
        SCOREP_USER_METRIC_LOCAL(bytes_sent_actual);
        SCOREP_USER_METRIC_LOCAL(bytes_recv_logical);
        SCOREP_USER_METRIC_LOCAL(bytes_recv_actual);
#endif

#ifdef USE_SCOREP
         if (profile_details) {
         SCOREP_USER_REGION_BEGIN(handle_phydll_library_static_step, "phydll_library_static_step", SCOREP_USER_REGION_TYPE_COMMON)
         SCOREP_USER_REGION_BEGIN(handle_phydll_prepack, "phydll_prepack", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        prepare_data_buffer();
#ifdef USE_SCOREP
        if (profile_details) {
        SCOREP_USER_REGION_END(handle_phydll_prepack)
        }
#endif

        if (std::getenv("DEBUG_PROVIDER_INPUT")) {
            auto &in = *this->input_after_preprocessing;
            for (size_t ti = 0; ti < in.size(); ++ti) {
                auto &t = in[ti];
                int n = static_cast<int>(t.numel());
                double sum = 0, ssq = 0;
                float first = 0, last = 0;
                if constexpr (std::is_same_v<In, float>) {
                    const float* raw = static_cast<const float*>(t.root());
                    first = raw[0]; last = raw[n-1];
                    for (int i = 0; i < n; ++i) { sum += raw[i]; ssq += raw[i]*raw[i]; }
                } else if constexpr (std::is_same_v<In, double>) {
                    const double* raw = static_cast<const double*>(t.root());
                    first = raw[0]; last = raw[n-1];
                    for (int i = 0; i < n; ++i) { sum += raw[i]; ssq += raw[i]*raw[i]; }
                }
                std::cerr << "DEBUG PHYDLL INPUT rank=" << this->rank << " tensor=" << ti << " shape=";
                for (int d : t.dimensions()) std::cerr << d << " ";
                std::cerr << " numel=" << n << " sum=" << sum << " sumSq=" << ssq << " first=" << first << " last=" << last << std::endl;
            }
        }

#ifdef USE_SCOREP
        if (profile_details) {
        static bool scorep_phydll_metrics_initialized = false;
        if (!scorep_phydll_metrics_initialized) {
            SCOREP_USER_METRIC_INIT(bytes_sent_logical, "bytes_sent_logical", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            SCOREP_USER_METRIC_INIT(bytes_sent_actual, "bytes_sent_actual", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            SCOREP_USER_METRIC_INIT(bytes_recv_logical, "bytes_recv_logical", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            SCOREP_USER_METRIC_INIT(bytes_recv_actual, "bytes_recv_actual", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            scorep_phydll_metrics_initialized = true;
        }
        SCOREP_USER_METRIC_UINT64(bytes_sent_logical, sum_sizes(input_sizes_) * sizeof(float));
        // PhyDLL transmits the registered fixed-size field(s), not just the useful input.
        SCOREP_USER_METRIC_UINT64(bytes_sent_actual, static_cast<uint64_t>(field_size_) * static_cast<uint64_t>(phy_field_count_) * sizeof(double));
        SCOREP_USER_REGION_BEGIN(handle_phydll_send, "phydll_send", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        if (transport_layout_ == TransportLayout::UniformChunks)
        {
            char chunk_label[16] = {0};
            for (int i = 0; i < phy_field_count_; ++i)
            {
                double *data_ptr = chunk_buffers_[static_cast<size_t>(i)].data();
                std::snprintf(chunk_label, sizeof(chunk_label), "PHY-IN-%03d", i);
                phydll_set_field(&data_ptr, chunk_label);
            }
        }
        else
        {
            double *data_ptr = data_buffer_.data();
            char data_label[] = "PHY-DATA";
            phydll_set_field(&data_ptr, data_label);
        }
        phydll_send();
#ifdef USE_SCOREP
        if (profile_details) {
        SCOREP_USER_REGION_END(handle_phydll_send)

        SCOREP_USER_REGION_BEGIN(handle_phydll_recv, "phydll_recv", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif

        phydll_recv();

        if (transport_layout_ == TransportLayout::UniformChunks)
        {
            bool found_output = false;
            for (int i = 0; i < dl_field_count_; ++i)
            {
                double *recv_ptr = nullptr;
                char recv_label[64] = {0};
                phydll_get_field(&recv_ptr, recv_label);
                if (recv_ptr)
                {
                    found_output = true;
                    const size_t dst = static_cast<size_t>(i) * static_cast<size_t>(field_size_);
                    std::copy(recv_ptr, recv_ptr + static_cast<size_t>(field_size_),
                              recv_reassembly_.begin() + static_cast<std::ptrdiff_t>(dst));
                    free(recv_ptr);
                }
            }
            guarantee(found_output, "PhyDLL provider received no DL output fields.");
        }
        else
        {
            bool found_output = false;
            for (int i = 0; i < dl_field_count_; ++i)
            {
                double *recv_ptr = nullptr;
                char recv_label[64] = {0};
                std::memset(recv_label, 0, sizeof(recv_label));
                phydll_get_field(&recv_ptr, recv_label);
                if (std::string(recv_label) == "DL-OUT")
                {
                    const size_t per_rank = static_cast<size_t>(sum_sizes(output_sizes_));
                    if (recv_ptr && per_rank > 0)
                    {
                        std::copy(recv_ptr, recv_ptr + per_rank, data_buffer_.begin());
                    }
                    found_output = true;
                }
                if (recv_ptr)
                {
                    free(recv_ptr);
                }
            }
            guarantee(found_output, "PhyDLL provider received no DL-OUT field.");
        }
#ifdef USE_SCOREP
        if (profile_details) {
        SCOREP_USER_REGION_END(handle_phydll_recv)
        SCOREP_USER_METRIC_UINT64(bytes_recv_logical, sum_sizes(output_sizes_) * sizeof(float));
        // The response uses the same fixed-size field(s) and includes output padding in packed mode.
        SCOREP_USER_METRIC_UINT64(bytes_recv_actual, static_cast<uint64_t>(field_size_) * static_cast<uint64_t>(dl_field_count_) * sizeof(double));

        SCOREP_USER_REGION_BEGIN(handle_phydll_unpack, "phydll_unpack", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif

        unpack_output_buffer();
#ifdef USE_SCOREP
         if (profile_details) {
         SCOREP_USER_REGION_END(handle_phydll_unpack)
         SCOREP_USER_REGION_END(handle_phydll_library_static_step)
         }
#endif
        metadata_sent_ = true;
#endif
    }

private:
    enum class MetaPhase : int
    {
        Init = 1,
        Data = 2
    };

    enum class TransportLayoutConfig : int
    {
        Auto = 0,
        Packed = 1,
        UniformChunks = 2
    };

    enum class TransportLayout : int
    {
        Packed = 0,
        UniformChunks = 1
    };

    static constexpr double kMetaMagic = 424242.0;
    static constexpr int kMetaVersion = 1;
    static constexpr int kHeaderFixedCount = 14;
    static constexpr int kBcastMetaMagic = 0x4D4C434D; // "MLCM"
    static constexpr int kBcastMetaVersion = 3;
    static constexpr int kMaxFieldCount = 4096;

    std::string model_file;
    std::string backend;
    std::string device;
    int batch_size = 0;

    TransportLayoutConfig layout_config_ = TransportLayoutConfig::Auto;
    TransportLayout transport_layout_ = TransportLayout::Packed;

    int field_size_ = 0;          // per-field per-source size in doubles
    int phy_field_count_ = 1;     // fields this PHY rank sends
    int dl_field_count_ = 1;      // fields this PHY rank expects back
    bool initialized_ = false;
    bool phydll_initialized_ = false;
    bool metadata_sent_ = false;
    bool metadata_bcasted_ = false;

    MLCouplingData<In> *input_after_preprocessing = nullptr;
    MLCouplingData<Out> *output_before_postprocessing = nullptr;

    std::vector<int64_t> input_sizes_;
    std::vector<int64_t> output_sizes_;

    std::vector<double> meta_buffer_;
    std::vector<double> data_buffer_;                  // packed-mode transport buffer
    std::vector<std::vector<double>> chunk_buffers_;   // uniform-chunk input fields
    std::vector<double> recv_reassembly_;              // uniform-chunk output reassembly

    struct BcastMetaHeader
    {
        int32_t magic = kBcastMetaMagic;
        int32_t version = kBcastMetaVersion;
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
        int32_t layout_kind = 0;
        int32_t phy_count = 0;
        int32_t dl_count = 0;
        int64_t field_size = 0;
    };
    static_assert(sizeof(BcastMetaHeader) == 88,
                  "BcastMetaHeader size must be exactly 88 bytes to match dl_client wire protocol");

    std::vector<int64_t> input_dims_;
    std::vector<int64_t> output_dims_;

    template <typename T>
    std::vector<int64_t> collect_dims(const MLCouplingData<T> &data)
    {
        std::vector<int64_t> dims;
        for (size_t i = 0; i < data.size(); ++i)
        {
            auto tensor_dims = data[i].dimensions();
            dims.push_back(static_cast<int64_t>(tensor_dims.size()));
            for (auto d : tensor_dims) {
                dims.push_back(static_cast<int64_t>(d));
            }
        }
        return dims;
    }

    void broadcast_metadata_once()
    {
#ifdef WITH_PHYDLL
        if (metadata_bcasted_)
        {
            return;
        }

        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized)
        {
            return;
        }

        int world_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        if (world_rank == 0)
        {
            std::cout << "[PHYDLL:PHY] Rank 0 starting metadata handshake. Synchronizing with DL side..." << std::endl;
        }

        if (world_rank == 0)
        {
            std::cout << "[PHYDLL:PHY] Handshake synchronization complete." << std::endl;
        }

        BcastMetaHeader header;
        std::vector<unsigned char> payload;

        header.model_len = static_cast<int32_t>(model_file.size());
        header.backend_len = static_cast<int32_t>(backend.size());
        header.device_len = static_cast<int32_t>(device.size());
        header.batch_size = static_cast<int32_t>(batch_size);
        header.num_inputs = static_cast<int32_t>(input_sizes_.size());
        header.num_outputs = static_cast<int32_t>(output_sizes_.size());
        header.total_input = sum_sizes(input_sizes_);
        header.total_output = sum_sizes(output_sizes_);
        header.dtype = static_cast<int32_t>(to_ml_coupling_data_type<In>());
        header.layout = static_cast<int32_t>(MLCouplingMemLayoutContiguous);

        header.num_input_dims = static_cast<int32_t>(input_dims_.size());
        header.num_output_dims = static_cast<int32_t>(output_dims_.size());
        header.layout_kind = static_cast<int32_t>(transport_layout_);
        header.phy_count = static_cast<int32_t>(phy_field_count_);
        header.dl_count = static_cast<int32_t>(dl_field_count_);
        header.field_size = static_cast<int64_t>(field_size_);

        const size_t sizes_bytes = (input_dims_.size() + output_dims_.size()) * sizeof(int64_t);
        payload.resize(static_cast<size_t>(header.model_len + header.backend_len + header.device_len) + sizes_bytes);
        size_t offset = 0;
        if (header.model_len > 0)
        {
            std::memcpy(payload.data() + offset, model_file.data(), header.model_len);
            offset += static_cast<size_t>(header.model_len);
        }
        if (header.backend_len > 0)
        {
            std::memcpy(payload.data() + offset, backend.data(), header.backend_len);
            offset += static_cast<size_t>(header.backend_len);
        }
        if (header.device_len > 0)
        {
            std::memcpy(payload.data() + offset, device.data(), header.device_len);
            offset += static_cast<size_t>(header.device_len);
        }
        if (!input_dims_.empty())
        {
            std::memcpy(payload.data() + offset, input_dims_.data(), input_dims_.size() * sizeof(int64_t));
            offset += input_dims_.size() * sizeof(int64_t);
        }
        if (!output_dims_.empty())
        {
            std::memcpy(payload.data() + offset, output_dims_.data(), output_dims_.size() * sizeof(int64_t));
        }

        int ndest = phydll_get_ndest();
        int *dests = phydll_get_dest();
        for (int i = 0; i < ndest; ++i)
        {
            int dl_rank = dests[i];
            MPI_Send(&header, sizeof(header), MPI_BYTE, dl_rank, world_rank, MPI_COMM_WORLD);
            if (!payload.empty())
            {
                MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_BYTE, dl_rank, world_rank, MPI_COMM_WORLD);
            }
        }

        metadata_bcasted_ = true;
#endif
    }

    void initialize_if_needed()
    {
#ifdef WITH_PHYDLL
        if (initialized_ || !input_after_preprocessing || !output_before_postprocessing)
        {
            return;
        }

        input_sizes_ = collect_sizes(*input_after_preprocessing);
        output_sizes_ = collect_sizes(*output_before_postprocessing);
        input_dims_ = collect_dims(*input_after_preprocessing);
        output_dims_ = collect_dims(*output_before_postprocessing);

        const int64_t total_input = sum_sizes(input_sizes_);
        const int64_t total_output = sum_sizes(output_sizes_);
        const int header_len = compute_header_content_len();

        if (layout_config_ == TransportLayoutConfig::UniformChunks)
        {
            guarantee(total_input > 0, "PhyDLL uniform_chunks layout requires non-empty input tensors.");
            guarantee(total_output > 0, "PhyDLL uniform_chunks layout requires non-empty output tensors.");

            const int64_t g = static_cast<int64_t>(std::gcd(total_input, total_output));
            guarantee(g > 0, "PhyDLL uniform_chunks layout: gcd of input/output element counts must be positive.");

            phy_field_count_ = static_cast<int>(total_input / g);
            dl_field_count_ = static_cast<int>(total_output / g);
            field_size_ = static_cast<int>(g);
            guarantee(phy_field_count_ >= 1 && dl_field_count_ >= 1,
                      "PhyDLL uniform_chunks layout derived a non-positive field count.");
            guarantee(phy_field_count_ <= kMaxFieldCount && dl_field_count_ <= kMaxFieldCount,
                      ("PhyDLL uniform_chunks layout derived more than " + std::to_string(kMaxFieldCount) +
                          " fields (" + std::to_string(phy_field_count_) + "/" + std::to_string(dl_field_count_) +
                          "). Increase kMaxFieldCount or use the packed layout.").c_str());

            transport_layout_ = TransportLayout::UniformChunks;
            std::cerr << "[PHYDLL:PHY] uniform_chunks layout: total_input=" << total_input
                      << " total_output=" << total_output << " gcd=" << g
                      << " phy_count=" << phy_field_count_ << " dl_count=" << dl_field_count_ << std::endl;
        }
        else if (layout_config_ == TransportLayoutConfig::Auto)
        {
            bool can_uniform = false;
            if (total_input > 0 && total_output > 0)
            {
                const int64_t g = static_cast<int64_t>(std::gcd(total_input, total_output));
                if (g > 0)
                {
                    const int64_t p_count = total_input / g;
                    const int64_t d_count = total_output / g;
                    if (p_count >= 1 && d_count >= 1 && p_count <= kMaxFieldCount && d_count <= kMaxFieldCount)
                    {
                        phy_field_count_ = static_cast<int>(p_count);
                        dl_field_count_ = static_cast<int>(d_count);
                        field_size_ = static_cast<int>(g);
                        transport_layout_ = TransportLayout::UniformChunks;
                        can_uniform = true;
                        std::cerr << "[PHYDLL:PHY] auto layout selected uniform_chunks: total_input=" << total_input
                                  << " total_output=" << total_output << " gcd=" << g
                                  << " phy_count=" << phy_field_count_ << " dl_count=" << dl_field_count_ << std::endl;
                    }
                }
            }
            if (!can_uniform)
            {
                field_size_ = static_cast<int>(std::max<int64_t>({total_input, total_output, header_len}));
                phy_field_count_ = 1;
                dl_field_count_ = 1;
                transport_layout_ = TransportLayout::Packed;
                std::cerr << "[PHYDLL:PHY] auto layout falling back to packed: total_input=" << total_input
                          << " total_output=" << total_output << " field_size=" << field_size_ << std::endl;
            }
        }
        else
        {
            field_size_ = static_cast<int>(std::max<int64_t>({total_input, total_output, header_len}));
            phy_field_count_ = 1;
            dl_field_count_ = 1;
            transport_layout_ = TransportLayout::Packed;
        }

        initialize_phydll_if_needed();
        phydll_opt_enable_cpl_loop();
        std::cerr << "[PHYDLL:PHY] before phydll_define_phy count=" << phy_field_count_
                  << " field_size=" << field_size_ << std::endl;
        phydll_define_phy(phy_field_count_, field_size_);
        std::cerr << "[PHYDLL:PHY] after phydll_define_phy" << std::endl;

        broadcast_metadata_once();

        meta_buffer_.assign(static_cast<size_t>(field_size_), 0.0);
        if (transport_layout_ == TransportLayout::UniformChunks)
        {
            chunk_buffers_.assign(static_cast<size_t>(phy_field_count_),
                                  std::vector<double>(static_cast<size_t>(field_size_), 0.0));
            recv_reassembly_.assign(static_cast<size_t>(total_output), 0.0);
            data_buffer_.assign(static_cast<size_t>(field_size_), 0.0);
        }
        else
        {
            data_buffer_.assign(static_cast<size_t>(field_size_), 0.0);
        }

        initialized_ = true;
#endif
    }

    template <typename T>
    std::vector<int64_t> collect_sizes(const MLCouplingData<T> &data)
    {
        std::vector<int64_t> sizes;
        sizes.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            sizes.push_back(static_cast<int64_t>(data[i].numel()));
        }
        return sizes;
    }

    int64_t sum_sizes(const std::vector<int64_t> &sizes) const
    {
        return std::accumulate(sizes.begin(), sizes.end(), static_cast<int64_t>(0));
    }

    void initialize_phydll_if_needed()
    {
#ifdef WITH_PHYDLL
        if (phydll_initialized_)
        {
            return;
        }

        char mode[] = "physical";
        std::cerr << "[PHYDLL:PHY] before phydll_init" << std::endl;
        phydll_init(mode);
        std::cerr << "[PHYDLL:PHY] after phydll_init" << std::endl;
        phydll_initialized_ = true;
#endif
    }

    int compute_header_content_len() const
    {
        return kHeaderFixedCount + static_cast<int>(model_file.size() + backend.size() + device.size() +
                                                    input_sizes_.size() + output_sizes_.size());
    }

    void encode_string(const std::string &value, size_t &cursor)
    {
        for (char ch : value)
        {
            meta_buffer_[cursor++] = static_cast<double>(static_cast<unsigned char>(ch));
        }
    }

    void prepare_meta_buffer(MetaPhase phase)
    {
        std::fill(meta_buffer_.begin(), meta_buffer_.end(), 0.0);
        const int header_len = compute_header_content_len();
        guarantee(field_size_ >= header_len, "PhyDLL metadata header exceeds field size.");

        meta_buffer_[0] = kMetaMagic;
        meta_buffer_[1] = static_cast<double>(kMetaVersion);
        meta_buffer_[2] = static_cast<double>(static_cast<int>(phase));
        meta_buffer_[3] = static_cast<double>(this->rank);
        meta_buffer_[4] = static_cast<double>(input_sizes_.size());
        meta_buffer_[5] = static_cast<double>(output_sizes_.size());
        meta_buffer_[6] = static_cast<double>(sum_sizes(input_sizes_));
        meta_buffer_[7] = static_cast<double>(sum_sizes(output_sizes_));
        meta_buffer_[8] = static_cast<double>(static_cast<int>(to_ml_coupling_data_type<In>()));
        meta_buffer_[9] = static_cast<double>(static_cast<int>(MLCouplingMemLayoutContiguous));
        meta_buffer_[10] = static_cast<double>(model_file.size());
        meta_buffer_[11] = static_cast<double>(backend.size());
        meta_buffer_[12] = static_cast<double>(device.size());
        meta_buffer_[13] = static_cast<double>(field_size_);

        size_t cursor = kHeaderFixedCount;
        encode_string(model_file, cursor);
        encode_string(backend, cursor);
        encode_string(device, cursor);

        for (const auto size : input_sizes_)
        {
            meta_buffer_[cursor++] = static_cast<double>(size);
        }
        for (const auto size : output_sizes_)
        {
            meta_buffer_[cursor++] = static_cast<double>(size);
        }
    }

    void prepare_data_buffer()
    {
        if (transport_layout_ == TransportLayout::UniformChunks)
        {
            for (auto &chunk : chunk_buffers_)
            {
                std::fill(chunk.begin(), chunk.end(), 0.0);
            }
            size_t cursor = 0;
            for (size_t i = 0; i < input_after_preprocessing->size(); ++i)
            {
                const auto &tensor = (*input_after_preprocessing)[i];
                if (tensor.is_contiguous())
                {
                    const In *ptr = static_cast<const In *>(tensor.root());
                    for (size_t j = 0; j < tensor.numel(); ++j)
                    {
                        chunk_buffers_[cursor / static_cast<size_t>(field_size_)][cursor % static_cast<size_t>(field_size_)] =
                            static_cast<double>(ptr[j]);
                        ++cursor;
                    }
                }
                else
                {
                    const auto flat = tensor.as_flat_vector();
                    for (const auto value : flat)
                    {
                        chunk_buffers_[cursor / static_cast<size_t>(field_size_)][cursor % static_cast<size_t>(field_size_)] =
                            static_cast<double>(value);
                        ++cursor;
                    }
                }
            }
            return;
        }

        std::fill(data_buffer_.begin(), data_buffer_.end(), 0.0);
        size_t cursor = 0;
        for (size_t i = 0; i < input_after_preprocessing->size(); ++i)
        {
            const auto &tensor = (*input_after_preprocessing)[i];
            if (tensor.is_contiguous())
            {
                const In *ptr = static_cast<const In *>(tensor.root());
                for (size_t j = 0; j < tensor.numel(); ++j)
                {
                    data_buffer_[cursor++] = static_cast<double>(ptr[j]);
                }
            }
            else
            {
                const auto flat = tensor.as_flat_vector();
                for (const auto value : flat)
                {
                    data_buffer_[cursor++] = static_cast<double>(value);
                }
            }
        }
    }

    void unpack_output_buffer()
    {
        const std::vector<double> &src =
            (transport_layout_ == TransportLayout::UniformChunks) ? recv_reassembly_ : data_buffer_;
        size_t cursor = 0;
        for (size_t i = 0; i < output_before_postprocessing->size(); ++i)
        {
            auto &tensor = (*output_before_postprocessing)[i];
            if (tensor.is_contiguous())
            {
                Out *ptr = static_cast<Out *>(tensor.root());
                for (size_t j = 0; j < tensor.numel(); ++j)
                {
                    ptr[j] = static_cast<Out>(src[cursor++]);
                }
            }
            else
            {
                for (size_t j = 0; j < tensor.numel(); ++j)
                {
                    tensor.set_linear(j, static_cast<Out>(src[cursor++]));
                }
            }
        }
    }
};
