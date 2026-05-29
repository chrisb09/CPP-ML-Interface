#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

extern "C" {
#include "phydll.h"
    int* phydll_get_dest();
    int phydll_get_ndest();
}

namespace phydll_dl {

constexpr double kMetaMagic = 424242.0;
constexpr int kMetaVersion = 1;

enum class MetaPhase : int {
    Init = 1,
    Data = 2
};

struct MetaEntry {
    int phy_rank = -1;
    int num_inputs = 0;
    int num_outputs = 0;
    int64_t total_input_size = 0;
    int64_t total_output_size = 0;
    int dtype = 0;
    int layout = 0;
    std::string model_path;
    std::string backend;
    std::string device;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;
};

struct MetaBatch {
    MetaPhase phase = MetaPhase::Init;
    std::vector<MetaEntry> entries;
};

struct Frame {
    bool has_meta = false;
    MetaBatch meta;
    std::vector<double> data;
};

class DlRuntime {
public:
    explicit DlRuntime(int dl_count);
    ~DlRuntime();

    void initialize();
    bool is_running() const;

    Frame receive_frame();
    void send_output(const std::vector<double>& output);

    int field_size() const { return field_size_; }

private:
    int dl_count_ = 1;
    int phy_count_ = 1;
    int source_count_ = 1;
    int field_size_ = 0;
    bool initialized_ = false;

    std::vector<double> meta_buffer_;
    std::vector<double> data_buffer_;
    std::vector<double> meta_out_buffer_;
    std::vector<double> combined_data_;
    
    // Persistent buffers for output that are registered with PhyDLL.
    // We use raw malloc'd pointers because PhyDLL finalize will free them.
    std::vector<double*> output_ptrs_;

    void receive_fields();
    void reset_buffers();

    static MetaBatch parse_meta(const std::vector<double>& buffer);
};

std::vector<double> build_meta_buffer(const MetaEntry& entry, MetaPhase phase, int field_size);

} // namespace phydll_dl
