#pragma once

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ml_coupling_application.hpp"

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// @registry_name: MLCouplingApplicationFlowExtrapolator
// @registry_aliases: flow-extrapolator, flow_extrapolator, maia-flow-extrapolator
// @registry_description: Preprocesses three raw flow fields into cube-based model tensors and reconstructs model output back into the raw fields.
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class MLCouplingApplicationFlowExtrapolator
    : public MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>
{
public:
    MLCouplingApplicationFlowExtrapolator(MLCouplingData<CouplingInput> coupling_input,
                                          MLCouplingData<CouplingOutput> coupling_output,
                                          MLCouplingNormalization<LibraryInput, CouplingOutput> *normalization = nullptr,
                                          int cube_dimension = 8,
                                          int cube_overlap = 0,
                                          int input_sequence_length = 1,
                                          int forecast_window = 1,
                                          int n_ghost_layers = 0)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(std::move(coupling_input),
                                                                                              std::move(coupling_output),
                                          normalization)
    {
        this->library_input = make_input_buffer(this->coupling_input, cube_dimension, cube_overlap, input_sequence_length, n_ghost_layers);
        this->library_output = make_library_output_buffer(this->coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        working_output_ = make_working_output_buffer(this->coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        initialize(cube_dimension, cube_overlap, input_sequence_length, forecast_window, n_ghost_layers);
    }

    MLCouplingApplicationFlowExtrapolator(MLCouplingData<CouplingInput> coupling_input,
                                          MLCouplingData<LibraryInput> library_input,
                                          MLCouplingData<LibraryOutput> library_output,
                                          MLCouplingData<CouplingOutput> coupling_output,
                                          MLCouplingNormalization<LibraryInput, CouplingOutput> *normalization = nullptr,
                                          int cube_dimension = 8,
                                          int cube_overlap = 0,
                                          int input_sequence_length = 1,
                                          int forecast_window = 1,
                                          int n_ghost_layers = 0)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(std::move(coupling_input),
                                                                                              std::move(library_input),
                                                                                              std::move(library_output),
                                                                                              std::move(coupling_output),
                                          normalization)
    {
        this->library_input = make_input_buffer(this->coupling_input, cube_dimension, cube_overlap, input_sequence_length, n_ghost_layers);
        this->library_output = make_library_output_buffer(this->coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        working_output_ = make_working_output_buffer(this->coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        initialize(cube_dimension, cube_overlap, input_sequence_length, forecast_window, n_ghost_layers);
    }

protected:
    MLCouplingData<LibraryInput> preprocess_coupling_input(MLCouplingData<CouplingInput> coupling_input) override
    {
        validate_input_fields(coupling_input);

#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_flowex_extract_cubes)
        SCOREP_USER_REGION_BEGIN(handle_flowex_extract_cubes, "flowex_extract_cubes", SCOREP_USER_REGION_TYPE_COMMON)
#endif
        std::vector<std::vector<CouplingInput>> current_step(static_cast<size_t>(kFieldCount));
        for (int field = 0; field < kFieldCount; ++field)
        {
            current_step[static_cast<size_t>(field)] = extract_field_cubes(coupling_input[field]);
        }

        history_.push_back(std::move(current_step));
        while (history_.size() > static_cast<size_t>(input_sequence_length_))
        {
            history_.pop_front();
        }

        auto &tensor = this->library_input[0];
        LibraryInput *buffer = static_cast<LibraryInput *>(tensor.root());
        const size_t total_values = tensor.numel();
        std::fill(buffer, buffer + total_values, static_cast<LibraryInput>(0));

        for (int seq = 0; seq < input_sequence_length_; ++seq)
        {
            const size_t history_index = resolve_history_index(seq);
            const auto &step_fields = history_[history_index];
            for (int field = 0; field < kFieldCount; ++field)
            {
                const auto &field_cubes = step_fields[static_cast<size_t>(field)];
                for (int cube = 0; cube < num_cubes_; ++cube)
                {
                    const int batch_index = field * num_cubes_ + cube;
                    const size_t dst_offset = (static_cast<size_t>(batch_index) * static_cast<size_t>(input_sequence_length_) + static_cast<size_t>(seq)) * static_cast<size_t>(cube_size_);
                    const size_t src_offset = static_cast<size_t>(cube) * static_cast<size_t>(cube_size_);
                    std::copy_n(field_cubes.data() + src_offset, static_cast<size_t>(cube_size_), buffer + dst_offset);
                }
            }
        }
#ifdef USE_SCOREP
        SCOREP_USER_REGION_END(handle_flowex_extract_cubes)
#endif

        if (debug_enabled())
        {
            debug_assembled_input_.assign(buffer, buffer + total_values);
        }
        if (this->normalization)
        {
            this->normalization->normalize_input(this->library_input);
        }

        return this->library_input;
    }

    int ml_step(MLCouplingLibrary<LibraryInput, LibraryOutput>& library, MLCouplingBehavior& behavior) override
    {
        if (behavior.should_send_data())
        {
            this->prepare_library_input();
        }
        if (behavior.should_perform_inference())
        {
            debug_active_ = debug_begin_inference();
            if (debug_active_)
            {
                debug_dump_layout();
                debug_dump_fields("raw_fields", this->coupling_input);
                debug_dump_vector("assembled_input", debug_assembled_input_);
                debug_dump_tensor("normalized_input", this->library_input[0]);
            }
#ifdef USE_SCOREP
            SCOREP_USER_REGION_DEFINE(handle_app_provider_inference)
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
                SCOREP_USER_REGION_BEGIN(handle_app_provider_inference, "app_provider_inference", SCOREP_USER_REGION_TYPE_COMMON)
            }
#endif
            library.static_inference(&this->library_input, &this->library_output);
#ifdef USE_SCOREP
            if (ml_coupling_scorep::detailed_regions_are_enabled()) {
                SCOREP_USER_REGION_END(handle_app_provider_inference)
            }
#endif
            if (debug_active_)
            {
                debug_dump_tensor("raw_provider_output", this->library_output[0]);
            }
            this->finalize_coupling_output();
            debug_active_ = false;
            return behavior.time_step_delta();
        }
        return 0;
    }

    MLCouplingData<CouplingOutput> postprocess_library_output(MLCouplingData<LibraryOutput> library_output) override
    {
        validate_input_fields(this->coupling_input);
        validate_output_fields(this->coupling_output);
        validate_model_output(library_output);

        auto &working_tensor = working_output_[0];
        CouplingOutput *working_buffer = static_cast<CouplingOutput *>(working_tensor.root());
        const auto &library_tensor = library_output[0];
        const LibraryOutput *library_buffer = static_cast<const LibraryOutput *>(library_tensor.root());
        std::transform(library_buffer, library_buffer + library_tensor.numel(), working_buffer,
                       [](LibraryOutput value) { return static_cast<CouplingOutput>(value); });

        if (this->normalization)
        {
            this->normalization->denormalize_output(working_output_);
        }
        if (debug_active_)
        {
            debug_dump_tensor("denormalized_output", working_output_[0]);
        }

        clear_output_active_region();

#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_flowex_reconstruct_output)
        SCOREP_USER_REGION_BEGIN(handle_flowex_reconstruct_output, "flowex_reconstruct_output", SCOREP_USER_REGION_TYPE_COMMON)
#endif
        const CouplingOutput *buffer = static_cast<const CouplingOutput *>(working_tensor.root());

        // Widen denormalized model output to float64 and accumulate in double to match legacy CMI.
        // Legacy CMI (ml_coupling_maia_aix.cpp) uses a double* fullField accumulator.
        // Without this, float32 accumulation produces a ~1 ULP (~1e-8) divergence per cell,
        // which the CFD solver then amplifies over subsequent steps.
        std::vector<double> widened(static_cast<size_t>(working_tensor.numel()));
        for (size_t i = 0; i < working_tensor.numel(); ++i)
        {
            widened[i] = static_cast<double>(buffer[i]);
        }

        for (int field = 0; field < kFieldCount; ++field)
        {
            CouplingOutput *dst_field = static_cast<CouplingOutput *>(this->coupling_output[field].root());

            // Accumulate into a double temporary to preserve precision (matches legacy CMI).
            std::vector<double> accum(weight_.size(), 0.0);
            for (int cube = 0; cube < num_cubes_; ++cube)
            {
                const int batch_index = field * num_cubes_ + cube;
                const size_t src_offset = (static_cast<size_t>(batch_index) * static_cast<size_t>(forecast_window_) + static_cast<size_t>(forecast_window_ - 1)) * static_cast<size_t>(cube_size_);
                const std::vector<int> &mapping = cube_volume_indices_[static_cast<size_t>(cube)];
                for (int local = 0; local < cube_size_; ++local)
                {
                    accum[static_cast<size_t>(mapping[static_cast<size_t>(local)])] +=
                        widened[src_offset + static_cast<size_t>(local)];
                }
            }

            // Normalize in double, then cast to float32 for solver buffer.
            for (size_t i = 0; i < weight_.size(); ++i)
            {
                if (weight_[i] > 0.0)
                {
                    dst_field[i] = static_cast<CouplingOutput>(accum[i] / weight_[i]);
                }
            }
        }
        if (debug_active_)
        {
            debug_dump_fields("reconstructed_fields", this->coupling_output);
        }
#ifdef USE_SCOREP
        SCOREP_USER_REGION_END(handle_flowex_reconstruct_output)
#endif

        return this->coupling_output;
    }

private:
    static constexpr int kFieldCount = 3;

    std::vector<int> n_cells_;
    std::vector<int> active_cells_;
    int cube_dimension_ = 8;
    int cube_overlap_ = 0;
    int input_sequence_length_ = 1;
    int forecast_window_ = 1;
    int n_ghost_layers_ = 0;
    int cube_size_ = 0;
    int num_cubes_ = 0;
    int yz_stride_ = 0;
    int row_stride_ = 0;

    std::vector<int> xs_;
    std::vector<int> ys_;
    std::vector<int> zs_;
    std::vector<std::vector<int>> cube_volume_indices_;
    std::vector<double> weight_;
    std::deque<std::vector<std::vector<CouplingInput>>> history_;
    MLCouplingData<CouplingOutput> working_output_;
    std::vector<LibraryInput> debug_assembled_input_;
    int debug_inference_count_ = 0;
    bool debug_active_ = false;
    std::string debug_prefix_;

    static bool debug_enabled()
    {
        const char* enabled = std::getenv("MLCOUPLING_DEBUG_EXPORT");
        if (!enabled || std::string(enabled) != "1") return false;
        const char* rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
        if (!rank_env) rank_env = std::getenv("SLURM_PROCID");
        const int rank = rank_env ? std::atoi(rank_env) : 0;
        const char* all_ranks = std::getenv("MLCOUPLING_DEBUG_ALL_RANKS");
        if (all_ranks && std::string(all_ranks) == "1") return true;
        const char* requested_rank = std::getenv("MLCOUPLING_DEBUG_RANK");
        return rank == (requested_rank ? std::atoi(requested_rank) : 0);
    }

    bool debug_begin_inference()
    {
        if (!debug_enabled()) return false;
        const char* max_env = std::getenv("MLCOUPLING_DEBUG_MAX_INFERENCES");
        const int max_inferences = max_env ? std::atoi(max_env) : 1;
        if (++debug_inference_count_ > max_inferences) return false;
        const char* root_env = std::getenv("MLCOUPLING_DEBUG_EXPORT_DIR");
        const std::filesystem::path root = root_env ? root_env : "mlcoupling-debug";
        std::filesystem::create_directories(root);
        const char* rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
        if (!rank_env) rank_env = std::getenv("SLURM_PROCID");
        const int rank = rank_env ? std::atoi(rank_env) : 0;
        debug_prefix_ = (root / ("current_rank_" + std::to_string(rank) + "_inference_" + std::to_string(debug_inference_count_))).string();
        std::ofstream manifest(debug_prefix_ + "_manifest.txt");
        manifest << "implementation=current\n";
        manifest << "rank=" << rank << "\n";
        if (const char* n_cells = std::getenv("MLCOUPLING_DEBUG_NCELLS")) manifest << "n_cells=" << n_cells << "\n";
        if (const char* offsets = std::getenv("MLCOUPLING_DEBUG_OFFSETS")) manifest << "offsets=" << offsets << "\n";
        manifest << "cube_dimension=" << cube_dimension_ << "\n";
        manifest << "cube_overlap=" << cube_overlap_ << "\n";
        manifest << "input_sequence_length=" << input_sequence_length_ << "\n";
        manifest << "forecast_window=" << forecast_window_ << "\n";
        manifest << "n_ghost_layers=" << n_ghost_layers_ << "\n";
        // Self-describing layout fields so comparators can validate without guessing.
        manifest << "num_cubes=" << num_cubes_ << "\n";
        manifest << "cube_size=" << cube_size_ << "\n";
        manifest << "n_fields=" << kFieldCount << "\n";
        {
            const size_t ai_count = static_cast<size_t>(kFieldCount) * static_cast<size_t>(num_cubes_)
                                  * static_cast<size_t>(input_sequence_length_) * static_cast<size_t>(cube_size_);
            const size_t rpo_count = static_cast<size_t>(kFieldCount) * static_cast<size_t>(num_cubes_)
                                   * static_cast<size_t>(forecast_window_) * static_cast<size_t>(cube_size_);
            manifest << "assembled_input_dtype=float32\n";
            manifest << "assembled_input_count=" << ai_count << "\n";
            manifest << "raw_provider_output_dtype=float32\n";
            manifest << "raw_provider_output_count=" << rpo_count << "\n";
            manifest << "reconstructed_field_dtype=" << (sizeof(CouplingOutput) == sizeof(float) ? "float32" : "float64") << "\n";
            manifest << "reconstructed_field_count=" << active_cells_[0] * active_cells_[1] * active_cells_[2] << "\n";
        }
        if (const char* n_cells = std::getenv("MLCOUPLING_DEBUG_NCELLS"))
            manifest << "active_cells=" << n_cells << "\n";
        return true;
    }

    template <typename T>
    void debug_dump_vector(const std::string& stage, const std::vector<T>& values) const
    {
        if (!debug_active_ || values.empty()) return;
        std::ofstream output(debug_prefix_ + "_" + stage + ".bin", std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    }

    template <typename T>
    void debug_dump_tensor(const std::string& stage, const MLCouplingTensor<T>& tensor) const
    {
        const T* values = static_cast<const T*>(tensor.root());
        std::vector<T> copy(values, values + tensor.numel());
        debug_dump_vector(stage, copy);
    }

    template <typename T>
    void debug_dump_fields(const std::string& stage, const MLCouplingData<T>& fields) const
    {
        for (int field = 0; field < kFieldCount; ++field)
        {
            debug_dump_tensor(stage + "_field" + std::to_string(field), fields[field]);
        }
    }

    void debug_dump_layout() const
    {
        debug_dump_vector("cube_starts_x", xs_);
        debug_dump_vector("cube_starts_y", ys_);
        debug_dump_vector("cube_starts_z", zs_);
        debug_dump_vector("cube_weights", weight_);
        std::vector<int> mapping;
        mapping.reserve(cube_volume_indices_.size() * static_cast<size_t>(cube_size_));
        for (const auto& cube : cube_volume_indices_) mapping.insert(mapping.end(), cube.begin(), cube.end());
        debug_dump_vector("cube_volume_indices", mapping);
    }

    static MLCouplingData<LibraryInput> make_input_buffer(const MLCouplingData<CouplingInput> &coupling_input,
                                                           int cube_dimension,
                                                           int cube_overlap,
                                                           int input_sequence_length,
                                                           int n_ghost_layers)
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_flowex_make_input_buffer)
        SCOREP_USER_REGION_BEGIN(handle_flowex_make_input_buffer, "flowex_make_input_buffer", SCOREP_USER_REGION_TYPE_COMMON)
#endif
        const ShapeInfo shape = infer_shape(coupling_input, cube_dimension, cube_overlap, n_ghost_layers);
        const std::vector<int> dims = {kFieldCount * shape.num_cubes, input_sequence_length, shape.cube_size};
        MLCouplingData<LibraryInput> result(std::vector<MLCouplingTensor<LibraryInput>>{
            MLCouplingTensor<LibraryInput>::from_flat_copy(std::vector<LibraryInput>(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]), static_cast<LibraryInput>(0)), dims)});
#ifdef USE_SCOREP
        SCOREP_USER_REGION_END(handle_flowex_make_input_buffer)
#endif
        return result;
    }

    template <typename T>
    static MLCouplingData<T> make_output_buffer(const MLCouplingData<CouplingInput> &coupling_input,
                                                int forecast_window,
                                                int cube_dimension,
                                                int cube_overlap,
                                                int n_ghost_layers)
    {
#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(handle_flowex_make_output_buffer)
        SCOREP_USER_REGION_BEGIN(handle_flowex_make_output_buffer, "flowex_make_output_buffer", SCOREP_USER_REGION_TYPE_COMMON)
#endif
        const ShapeInfo shape = infer_shape(coupling_input, cube_dimension, cube_overlap, n_ghost_layers);
        const std::vector<int> dims = {kFieldCount * shape.num_cubes, forecast_window, shape.cube_size};
        MLCouplingData<T> result(std::vector<MLCouplingTensor<T>>{
            MLCouplingTensor<T>::from_flat_copy(std::vector<T>(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]), static_cast<T>(0)), dims)});
#ifdef USE_SCOREP
        SCOREP_USER_REGION_END(handle_flowex_make_output_buffer)
#endif
        return result;
    }

    static MLCouplingData<LibraryOutput> make_library_output_buffer(const MLCouplingData<CouplingInput> &coupling_input,
                                                                      int forecast_window,
                                                                      int cube_dimension,
                                                                      int cube_overlap,
                                                                      int n_ghost_layers)
    {
        return make_output_buffer<LibraryOutput>(coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
    }

    static MLCouplingData<CouplingOutput> make_working_output_buffer(const MLCouplingData<CouplingInput> &coupling_input,
                                                                       int forecast_window,
                                                                       int cube_dimension,
                                                                       int cube_overlap,
                                                                       int n_ghost_layers)
    {
        return make_output_buffer<CouplingOutput>(coupling_input, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
    }

    struct ShapeInfo
    {
        int cube_size = 0;
        int num_cubes = 0;
    };

    static ShapeInfo infer_shape(const MLCouplingData<CouplingInput> &input_data,
                                 int cube_dimension,
                                 int cube_overlap,
                                 int n_ghost_layers)
    {
        validate_input_fields(input_data);
        if (cube_dimension <= 0)
        {
            throw std::invalid_argument("FlowExtrapolator: cube_dimension must be positive");
        }
        if (cube_overlap < 0 || cube_overlap >= cube_dimension)
        {
            throw std::invalid_argument("FlowExtrapolator: cube_overlap must be in [0, cube_dimension)");
        }
        const std::vector<int> dims = input_data[0].dimensions();
        std::vector<int> active_cells(3);
        for (int axis = 0; axis < 3; ++axis)
        {
            active_cells[axis] = dims[static_cast<size_t>(axis)] - 2 * n_ghost_layers;
            if (active_cells[axis] <= 0)
            {
                throw std::invalid_argument("FlowExtrapolator: ghost layers leave no active cells");
            }
            if (active_cells[axis] < cube_dimension)
            {
                throw std::invalid_argument("FlowExtrapolator: cube_dimension exceeds active grid extent");
            }
        }

        const std::vector<int> xs = cube_overlap == 0
                                        ? linspace(0, active_cells[2] - cube_dimension, static_cast<int>(std::ceil(static_cast<double>(active_cells[2]) / static_cast<double>(cube_dimension))))
                                        : get_full_indices(active_cells[2], cube_dimension, cube_dimension - cube_overlap);
        const std::vector<int> ys = cube_overlap == 0
                                        ? linspace(0, active_cells[1] - cube_dimension, static_cast<int>(std::ceil(static_cast<double>(active_cells[1]) / static_cast<double>(cube_dimension))))
                                        : get_full_indices(active_cells[1], cube_dimension, cube_dimension - cube_overlap);
        const std::vector<int> zs = cube_overlap == 0
                                        ? linspace(0, active_cells[0] - cube_dimension, static_cast<int>(std::ceil(static_cast<double>(active_cells[0]) / static_cast<double>(cube_dimension))))
                                        : get_full_indices(active_cells[0], cube_dimension, cube_dimension - cube_overlap);

        ShapeInfo info;
        info.cube_size = cube_dimension * cube_dimension * cube_dimension;
        info.num_cubes = static_cast<int>(xs.size() * ys.size() * zs.size());
        return info;
    }

    void initialize(int cube_dimension,
                    int cube_overlap,
                    int input_sequence_length,
                    int forecast_window,
                    int n_ghost_layers)
    {
        cube_dimension_ = cube_dimension;
        cube_overlap_ = cube_overlap;
        input_sequence_length_ = input_sequence_length;
        forecast_window_ = forecast_window;
        n_ghost_layers_ = n_ghost_layers;
        cube_size_ = cube_dimension_ * cube_dimension_ * cube_dimension_;

        validate_input_fields(this->coupling_input);
        validate_output_fields(this->coupling_output);

        n_cells_ = this->coupling_input[0].dimensions();
        active_cells_.resize(3);
        for (int axis = 0; axis < 3; ++axis)
        {
            active_cells_[static_cast<size_t>(axis)] = n_cells_[static_cast<size_t>(axis)] - 2 * n_ghost_layers_;
            if (active_cells_[static_cast<size_t>(axis)] <= 0)
            {
                throw std::invalid_argument("FlowExtrapolator: ghost layers leave no active cells");
            }
            if (active_cells_[static_cast<size_t>(axis)] < cube_dimension_)
            {
                throw std::invalid_argument("FlowExtrapolator: cube_dimension exceeds active grid extent");
            }
        }

        if (cube_dimension_ <= 0)
        {
            throw std::invalid_argument("FlowExtrapolator: cube_dimension must be positive");
        }
        if (cube_overlap_ < 0 || cube_overlap_ >= cube_dimension_)
        {
            throw std::invalid_argument("FlowExtrapolator: cube_overlap must be in [0, cube_dimension)");
        }
        if (input_sequence_length_ <= 0)
        {
            throw std::invalid_argument("FlowExtrapolator: input_sequence_length must be positive");
        }
        if (forecast_window_ <= 0)
        {
            throw std::invalid_argument("FlowExtrapolator: forecast_window must be positive");
        }

        yz_stride_ = n_cells_[1] * n_cells_[2];
        row_stride_ = n_cells_[2];

        if (cube_overlap_ == 0)
        {
            xs_ = linspace(0, active_cells_[2] - cube_dimension_, static_cast<int>(std::ceil(static_cast<double>(active_cells_[2]) / static_cast<double>(cube_dimension_))));
            ys_ = linspace(0, active_cells_[1] - cube_dimension_, static_cast<int>(std::ceil(static_cast<double>(active_cells_[1]) / static_cast<double>(cube_dimension_))));
            zs_ = linspace(0, active_cells_[0] - cube_dimension_, static_cast<int>(std::ceil(static_cast<double>(active_cells_[0]) / static_cast<double>(cube_dimension_))));
        }
        else
        {
            xs_ = get_full_indices(active_cells_[2], cube_dimension_, cube_dimension_ - cube_overlap_);
            ys_ = get_full_indices(active_cells_[1], cube_dimension_, cube_dimension_ - cube_overlap_);
            zs_ = get_full_indices(active_cells_[0], cube_dimension_, cube_dimension_ - cube_overlap_);
        }

        num_cubes_ = static_cast<int>(xs_.size() * ys_.size() * zs_.size());
        build_cube_mappings();
        validate_model_buffers();
    }

    static void validate_input_fields(const MLCouplingData<CouplingInput> &raw_input)
    {
        if (raw_input.size() != static_cast<size_t>(kFieldCount))
        {
            throw std::invalid_argument("FlowExtrapolator: expected exactly three raw field tensors");
        }
        const std::vector<int> reference_dims = raw_input[0].dimensions();
        if (reference_dims.size() != 3)
        {
            throw std::invalid_argument("FlowExtrapolator: raw field tensors must be 3D");
        }
        for (int field = 0; field < kFieldCount; ++field)
        {
            if (!raw_input[static_cast<size_t>(field)].is_contiguous())
            {
                throw std::invalid_argument("FlowExtrapolator: raw field tensors must be contiguous");
            }
            if (raw_input[static_cast<size_t>(field)].dimensions() != reference_dims)
            {
                throw std::invalid_argument("FlowExtrapolator: raw field tensor shapes must match");
            }
        }
    }

    static void validate_output_fields(const MLCouplingData<CouplingOutput> &raw_output)
    {
        if (raw_output.size() != static_cast<size_t>(kFieldCount))
        {
            throw std::invalid_argument("FlowExtrapolator: expected exactly three raw output tensors");
        }
        const std::vector<int> reference_dims = raw_output[0].dimensions();
        if (reference_dims.size() != 3)
        {
            throw std::invalid_argument("FlowExtrapolator: raw output tensors must be 3D");
        }
        for (int field = 0; field < kFieldCount; ++field)
        {
            if (!raw_output[static_cast<size_t>(field)].is_contiguous())
            {
                throw std::invalid_argument("FlowExtrapolator: raw output tensors must be contiguous");
            }
            if (raw_output[static_cast<size_t>(field)].dimensions() != reference_dims)
            {
                throw std::invalid_argument("FlowExtrapolator: raw output tensor shapes must match");
            }
        }
    }

    void validate_model_buffers() const
    {
        if (this->library_input.size() != 1)
        {
            throw std::invalid_argument("FlowExtrapolator: preprocessed input must contain exactly one tensor");
        }
        if (this->library_output.size() != 1 || working_output_.size() != 1)
        {
            throw std::invalid_argument("FlowExtrapolator: model output buffer must contain exactly one tensor");
        }

        const std::vector<int> expected_input_dims = {kFieldCount * num_cubes_, input_sequence_length_, cube_size_};
        const std::vector<int> expected_output_dims = {kFieldCount * num_cubes_, forecast_window_, cube_size_};

        if (this->library_input[0].dimensions() != expected_input_dims)
        {
            throw std::invalid_argument("FlowExtrapolator: preprocessed input tensor has unexpected shape");
        }
        if (this->library_output[0].dimensions() != expected_output_dims ||
            working_output_[0].dimensions() != expected_output_dims)
        {
            throw std::invalid_argument("FlowExtrapolator: model output tensor has unexpected shape");
        }
    }

    void validate_model_output(const MLCouplingData<LibraryOutput> &model_output) const
    {
        if (model_output.size() != 1 || !model_output[0].is_contiguous())
        {
            throw std::invalid_argument("FlowExtrapolator: model output must contain one contiguous tensor");
        }
        const std::vector<int> expected_output_dims = {kFieldCount * num_cubes_, forecast_window_, cube_size_};
        if (model_output[0].dimensions() != expected_output_dims)
        {
            throw std::invalid_argument("FlowExtrapolator: model output tensor has unexpected shape");
        }
    }

    void build_cube_mappings()
    {
        cube_volume_indices_.clear();
        cube_volume_indices_.reserve(static_cast<size_t>(num_cubes_));
        weight_.assign(static_cast<size_t>(n_cells_[0] * n_cells_[1] * n_cells_[2]), 0.0);

        for (int z0 : zs_)
        {
            for (int y0 : ys_)
            {
                for (int x0 : xs_)
                {
                    std::vector<int> mapping(static_cast<size_t>(cube_size_));
                    int local = 0;
                    for (int dz = 0; dz < cube_dimension_; ++dz)
                    {
                        const int global_z = z0 + dz + n_ghost_layers_;
                        for (int dy = 0; dy < cube_dimension_; ++dy)
                        {
                            const int global_y = y0 + dy + n_ghost_layers_;
                            for (int dx = 0; dx < cube_dimension_; ++dx)
                            {
                                const int global_x = x0 + dx + n_ghost_layers_;
                                const int flat_index = global_z * yz_stride_ + global_y * row_stride_ + global_x;
                                mapping[static_cast<size_t>(local++)] = flat_index;
                                weight_[static_cast<size_t>(flat_index)] += 1.0;
                            }
                        }
                    }
                    cube_volume_indices_.push_back(std::move(mapping));
                }
            }
        }
    }

    std::vector<CouplingInput> extract_field_cubes(const MLCouplingTensor<CouplingInput> &tensor) const
    {
        const CouplingInput *src = static_cast<const CouplingInput *>(tensor.root());
        std::vector<CouplingInput> cubes(static_cast<size_t>(num_cubes_) * static_cast<size_t>(cube_size_));
        size_t cube_index = 0;
        for (const auto &mapping : cube_volume_indices_)
        {
            for (int local = 0; local < cube_size_; ++local)
            {
                cubes[cube_index * static_cast<size_t>(cube_size_) + static_cast<size_t>(local)] = src[static_cast<size_t>(mapping[static_cast<size_t>(local)])];
            }
            ++cube_index;
        }
        return cubes;
    }

    void clear_output_active_region()
    {
        for (int field = 0; field < kFieldCount; ++field)
        {
            CouplingOutput *dst = static_cast<CouplingOutput *>(this->coupling_output[static_cast<size_t>(field)].root());
            for (int z = n_ghost_layers_; z < n_cells_[0] - n_ghost_layers_; ++z)
            {
                const int base_z = z * yz_stride_;
                for (int y = n_ghost_layers_; y < n_cells_[1] - n_ghost_layers_; ++y)
                {
                    const int start = base_z + y * row_stride_ + n_ghost_layers_;
                    std::fill(dst + start, dst + start + active_cells_[2], static_cast<CouplingOutput>(0));
                }
            }
        }
    }

    size_t resolve_history_index(int sequence_slot) const
    {
        if (history_.empty())
        {
            throw std::logic_error("FlowExtrapolator: history is unexpectedly empty");
        }

        const int missing = input_sequence_length_ - static_cast<int>(history_.size());
        if (sequence_slot < missing)
        {
            return 0;
        }
        return static_cast<size_t>(sequence_slot - missing);
    }

    static std::vector<int> linspace(int start, int end, int count)
    {
        if (count <= 0)
        {
            return {};
        }
        if (count == 1)
        {
            return {start};
        }

        std::vector<int> values(static_cast<size_t>(count));
        const double step = static_cast<double>(end - start) / static_cast<double>(count - 1);
        for (int i = 0; i < count; ++i)
        {
            values[static_cast<size_t>(i)] = static_cast<int>(std::round(static_cast<double>(start) + static_cast<double>(i) * step));
        }
        return values;
    }

    static std::vector<int> get_full_indices(int length, int cube_dimension, int step)
    {
        if (step <= 0)
        {
            throw std::invalid_argument("FlowExtrapolator: cube stride must be positive");
        }

        std::vector<int> indices;
        for (int i = 0; i <= length - cube_dimension; i += step)
        {
            indices.push_back(i);
        }
        if (indices.empty() || indices.back() != length - cube_dimension)
        {
            indices.push_back(length - cube_dimension);
        }
        return indices;
    }
};
