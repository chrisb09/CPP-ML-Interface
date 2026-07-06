#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ml_coupling_application.hpp"

// @registry_name: MLCouplingApplicationFlowExtrapolator
// @registry_aliases: flow-extrapolator, flow_extrapolator, maia-flow-extrapolator
// @registry_description: Preprocesses three raw flow fields into cube-based model tensors and reconstructs model output back into the raw fields.
template <typename In, typename Out>
class MLCouplingApplicationFlowExtrapolator : public MLCouplingApplication<In, Out>
{
public:
    MLCouplingApplicationFlowExtrapolator(MLCouplingData<In> input_data,
                                          MLCouplingData<Out> output_data,
                                          MLCouplingNormalization<In, Out> *normalization = nullptr,
                                          int cube_dimension = 8,
                                          int cube_overlap = 0,
                                          int input_sequence_length = 1,
                                          int forecast_window = 1,
                                          int n_ghost_layers = 0)
        : MLCouplingApplication<In, Out>(std::move(input_data),
                                         std::move(output_data),
                                         normalization)
    {
        this->input_data_after_preprocessing = make_input_buffer(this->input_data, cube_dimension, cube_overlap, input_sequence_length, n_ghost_layers);
        this->output_data_before_postprocessing = make_output_buffer(this->input_data, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        initialize(cube_dimension, cube_overlap, input_sequence_length, forecast_window, n_ghost_layers);
    }

    MLCouplingApplicationFlowExtrapolator(MLCouplingData<In> input_data,
                                          MLCouplingData<In> input_data_after_preprocessing,
                                          MLCouplingData<Out> output_data_before_postprocessing,
                                          MLCouplingData<Out> output_data,
                                          MLCouplingNormalization<In, Out> *normalization = nullptr,
                                          int cube_dimension = 8,
                                          int cube_overlap = 0,
                                          int input_sequence_length = 1,
                                          int forecast_window = 1,
                                          int n_ghost_layers = 0)
        : MLCouplingApplication<In, Out>(std::move(input_data),
                                          std::move(input_data_after_preprocessing),
                                         std::move(output_data_before_postprocessing),
                                         std::move(output_data),
                                         normalization)
    {
        // Always recreate pre/post buffers unconditionally: the base class
        // constructor (MLCouplingApplication) copies input_data (3 raw tensors)
        // into input_data_after_preprocessing when the latter is empty, which
        // defeats the empty() check and causes "exactly one tensor" validation
        // failure. We must override with proper single-tensor buffers.
        this->input_data_after_preprocessing = make_input_buffer(this->input_data, cube_dimension, cube_overlap, input_sequence_length, n_ghost_layers);
        this->output_data_before_postprocessing = make_output_buffer(this->input_data, forecast_window, cube_dimension, cube_overlap, n_ghost_layers);
        initialize(cube_dimension, cube_overlap, input_sequence_length, forecast_window, n_ghost_layers);
    }

protected:
    MLCouplingData<In> preprocess(MLCouplingData<In> input_data) override
    {
        validate_input_fields(input_data);

        std::vector<std::vector<In>> current_step(static_cast<size_t>(kFieldCount));
        for (int field = 0; field < kFieldCount; ++field)
        {
            current_step[static_cast<size_t>(field)] = extract_field_cubes(input_data[field]);
        }

        history_.push_back(std::move(current_step));
        while (history_.size() > static_cast<size_t>(input_sequence_length_))
        {
            history_.pop_front();
        }

        auto &tensor = this->input_data_after_preprocessing[0];
        In *buffer = static_cast<In *>(tensor.root());
        const size_t total_values = tensor.numel();
        std::fill(buffer, buffer + total_values, static_cast<In>(0));

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

        if (this->normalization)
        {
            this->normalization->normalize_input(this->input_data_after_preprocessing);
        }

        return this->input_data_after_preprocessing;
    }

    int ml_step(MLCouplingProvider<In,Out>& provider, MLCouplingBehavior& behavior) override
    {
        if (behavior.should_send_data())
        {
            this->prepare_input();
        }
        if (behavior.should_perform_inference())
        {
            provider.static_inference(&this->input_data_after_preprocessing,
                                       &this->output_data_before_postprocessing);
            this->finalize_output();
            return behavior.time_step_delta();
        }
        return 0;
    }

    MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) override
    {
        validate_input_fields(this->input_data);
        validate_output_fields(this->output_data);
        validate_model_output(output_data_before_postprocessing);

        if (this->normalization)
        {
            this->normalization->denormalize_output(output_data_before_postprocessing);
        }

        clear_output_active_region();

        const auto &tensor = output_data_before_postprocessing[0];
        const Out *buffer = static_cast<const Out *>(tensor.root());

        for (int field = 0; field < kFieldCount; ++field)
        {
            Out *dst_field = static_cast<Out *>(this->output_data[field].root());
            for (int cube = 0; cube < num_cubes_; ++cube)
            {
                const int batch_index = field * num_cubes_ + cube;
                const size_t src_offset = (static_cast<size_t>(batch_index) * static_cast<size_t>(forecast_window_) + static_cast<size_t>(forecast_window_ - 1)) * static_cast<size_t>(cube_size_);
                const std::vector<int> &mapping = cube_volume_indices_[static_cast<size_t>(cube)];
                for (int local = 0; local < cube_size_; ++local)
                {
                    dst_field[static_cast<size_t>(mapping[static_cast<size_t>(local)])] += buffer[src_offset + static_cast<size_t>(local)];
                }
            }

            for (size_t i = 0; i < weight_.size(); ++i)
            {
                if (weight_[i] > 0.0)
                {
                    dst_field[i] = static_cast<Out>(dst_field[i] / static_cast<Out>(weight_[i]));
                }
            }
        }

        return this->output_data;
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
    std::deque<std::vector<std::vector<In>>> history_;

    static MLCouplingData<In> make_input_buffer(const MLCouplingData<In> &input_data,
                                                int cube_dimension,
                                                int cube_overlap,
                                                int input_sequence_length,
                                                int n_ghost_layers)
    {
        const ShapeInfo shape = infer_shape(input_data, cube_dimension, cube_overlap, n_ghost_layers);
        const std::vector<int> dims = {kFieldCount * shape.num_cubes, input_sequence_length, shape.cube_size};
        return MLCouplingData<In>(std::vector<MLCouplingTensor<In>>{
            MLCouplingTensor<In>::from_flat_copy(std::vector<In>(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]), static_cast<In>(0)), dims)});
    }

    static MLCouplingData<Out> make_output_buffer(const MLCouplingData<In> &input_data,
                                                  int forecast_window,
                                                  int cube_dimension,
                                                  int cube_overlap,
                                                  int n_ghost_layers)
    {
        const ShapeInfo shape = infer_shape(input_data, cube_dimension, cube_overlap, n_ghost_layers);
        const std::vector<int> dims = {kFieldCount * shape.num_cubes, forecast_window, shape.cube_size};
        return MLCouplingData<Out>(std::vector<MLCouplingTensor<Out>>{
            MLCouplingTensor<Out>::from_flat_copy(std::vector<Out>(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]), static_cast<Out>(0)), dims)});
    }

    struct ShapeInfo
    {
        int cube_size = 0;
        int num_cubes = 0;
    };

    static ShapeInfo infer_shape(const MLCouplingData<In> &input_data,
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

        validate_input_fields(this->input_data);
        validate_output_fields(this->output_data);

        n_cells_ = this->input_data[0].dimensions();
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

    static void validate_input_fields(const MLCouplingData<In> &raw_input)
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

    static void validate_output_fields(const MLCouplingData<Out> &raw_output)
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
        if (this->input_data_after_preprocessing.size() != 1)
        {
            throw std::invalid_argument("FlowExtrapolator: preprocessed input must contain exactly one tensor");
        }
        if (this->output_data_before_postprocessing.size() != 1)
        {
            throw std::invalid_argument("FlowExtrapolator: model output buffer must contain exactly one tensor");
        }

        const std::vector<int> expected_input_dims = {kFieldCount * num_cubes_, input_sequence_length_, cube_size_};
        const std::vector<int> expected_output_dims = {kFieldCount * num_cubes_, forecast_window_, cube_size_};

        if (this->input_data_after_preprocessing[0].dimensions() != expected_input_dims)
        {
            throw std::invalid_argument("FlowExtrapolator: preprocessed input tensor has unexpected shape");
        }
        if (this->output_data_before_postprocessing[0].dimensions() != expected_output_dims)
        {
            throw std::invalid_argument("FlowExtrapolator: model output tensor has unexpected shape");
        }
    }

    void validate_model_output(const MLCouplingData<Out> &model_output) const
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

    std::vector<In> extract_field_cubes(const MLCouplingTensor<In> &tensor) const
    {
        const In *src = static_cast<const In *>(tensor.root());
        std::vector<In> cubes(static_cast<size_t>(num_cubes_) * static_cast<size_t>(cube_size_));
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
            Out *dst = static_cast<Out *>(this->output_data[static_cast<size_t>(field)].root());
            for (int z = n_ghost_layers_; z < n_cells_[0] - n_ghost_layers_; ++z)
            {
                const int base_z = z * yz_stride_;
                for (int y = n_ghost_layers_; y < n_cells_[1] - n_ghost_layers_; ++y)
                {
                    const int start = base_z + y * row_stride_ + n_ghost_layers_;
                    std::fill(dst + start, dst + start + active_cells_[2], static_cast<Out>(0));
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
