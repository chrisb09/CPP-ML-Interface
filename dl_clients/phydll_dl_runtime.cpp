#include "phydll_dl_runtime.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace phydll_dl {
namespace {
constexpr int kHeaderMagicIndex = 0;
constexpr int kHeaderVersionIndex = 1;
constexpr int kHeaderPhaseIndex = 2;
constexpr int kHeaderPhyRankIndex = 3;
constexpr int kHeaderNumInputsIndex = 4;
constexpr int kHeaderNumOutputsIndex = 5;
constexpr int kHeaderTotalInputIndex = 6;
constexpr int kHeaderTotalOutputIndex = 7;
constexpr int kHeaderDtypeIndex = 8;
constexpr int kHeaderLayoutIndex = 9;
constexpr int kHeaderModelLenIndex = 10;
constexpr int kHeaderBackendLenIndex = 11;
constexpr int kHeaderDeviceLenIndex = 12;
constexpr int kHeaderLengthIndex = 13;
constexpr int kHeaderFixedCount = 14;

int64_t as_int64(double value) {
    return static_cast<int64_t>(value);
}

std::string decode_string(const std::vector<double>& buffer, size_t start, size_t length) {
    std::string out;
    out.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        const auto code = static_cast<int>(buffer[start + i]);
        out.push_back(static_cast<char>(code));
    }
    return out;
}

void encode_string(const std::string& value, std::vector<double>& buffer, size_t start) {
    for (size_t i = 0; i < value.size(); ++i) {
        buffer[start + i] = static_cast<double>(static_cast<unsigned char>(value[i]));
    }
}

MetaPhase to_phase(double value) {
    return static_cast<MetaPhase>(static_cast<int>(value));
}
} // namespace

DlRuntime::DlRuntime(int dl_count) : dl_count_(dl_count) {}

void DlRuntime::initialize() {
    if (initialized_) {
        return;
    }
    std::cerr << "[PHYDLL:DL] init" << std::endl;
    phydll_init(const_cast<char*>("dl"));
    phydll_define_dl(dl_count_);
    phydll_get_field_size(&field_size_);
    std::cerr << "[PHYDLL:DL] dl_count=" << dl_count_ << " field_size=" << field_size_ << std::endl;
    if (field_size_ <= 0) {
        throw std::runtime_error("PhyDLL returned invalid field size.");
    }
    meta_buffer_.resize(static_cast<size_t>(field_size_));
    data_buffer_.resize(static_cast<size_t>(field_size_));
    meta_out_buffer_.resize(static_cast<size_t>(field_size_));
    initialized_ = true;
}

bool DlRuntime::is_running() const {
    return phydll_is_phy_signal();
}

Frame DlRuntime::receive_frame() {
    if (!initialized_) {
        initialize();
    }

    reset_buffers();
    receive_fields();

    Frame frame;
    frame.data = data_buffer_;
    frame.meta = parse_meta(meta_buffer_);
    frame.has_meta = !frame.meta.entries.empty();
    std::fprintf(stderr, "[PHYDLL:DL] meta header magic=%.0f version=%.0f phase=%.0f header_len=%.0f entries=%zu\n",
                 meta_buffer_[0], meta_buffer_[1], meta_buffer_[2], meta_buffer_[13],
                 frame.meta.entries.size());
    std::fflush(stderr);
    return frame;
}

void DlRuntime::send_output(const std::vector<double>& output) {
    if (!initialized_) {
        initialize();
    }

    if (output.size() > data_buffer_.size()) {
        throw std::runtime_error("Output buffer larger than field size.");
    }

    std::fill(data_buffer_.begin(), data_buffer_.end(), 0.0);
    std::copy(output.begin(), output.end(), data_buffer_.begin());

    double* output_ptr = data_buffer_.data();
    double* meta_ptr = meta_out_buffer_.data();
    std::fill(meta_out_buffer_.begin(), meta_out_buffer_.end(), 0.0);
    char out_label[] = "DL-OUT";
    char meta_label[] = "DL-META";
    phydll_set_field(&output_ptr, out_label);
    phydll_set_field(&meta_ptr, meta_label);
    phydll_send();
}

void DlRuntime::receive_fields() {
    std::cerr << "[PHYDLL:DL] waiting recv" << std::endl;
    phydll_irecv();
    phydll_wait_irecv();
    std::cerr << "[PHYDLL:DL] recv done" << std::endl;

    double* buffer_ptr = nullptr;
    char label[64] = {0};

    for (int i = 0; i < 2; ++i) {
        buffer_ptr = data_buffer_.data();
        std::memset(label, 0, sizeof(label));
        phydll_get_field(&buffer_ptr, label);

        std::cerr << "[PHYDLL:DL] got label '" << label << "'" << std::endl;

        if (std::string(label) == "PHY-META") {
            std::copy(buffer_ptr, buffer_ptr + field_size_, meta_buffer_.begin());
        } else if (std::string(label) == "PHY-DATA") {
            std::copy(buffer_ptr, buffer_ptr + field_size_, data_buffer_.begin());
        }
    }
}

void DlRuntime::reset_buffers() {
    std::fill(meta_buffer_.begin(), meta_buffer_.end(), 0.0);
    std::fill(data_buffer_.begin(), data_buffer_.end(), 0.0);
}

MetaBatch DlRuntime::parse_meta(const std::vector<double>& buffer) {
    MetaBatch batch;
    size_t offset = 0;
    while (offset + kHeaderFixedCount <= buffer.size()) {
        if (buffer[offset + kHeaderMagicIndex] != kMetaMagic) {
            std::cerr << "[PHYDLL:DL] meta magic mismatch value="
                      << buffer[offset + kHeaderMagicIndex] << std::endl;
            break;
        }
        const int version = static_cast<int>(buffer[offset + kHeaderVersionIndex]);
        if (version != kMetaVersion) {
            std::cerr << "[PHYDLL:DL] meta version mismatch value=" << version << std::endl;
            break;
        }

        const int header_len = static_cast<int>(buffer[offset + kHeaderLengthIndex]);
        if (header_len <= 0 || offset + static_cast<size_t>(header_len) > buffer.size()) {
            std::cerr << "[PHYDLL:DL] meta header length invalid value=" << header_len
                      << " buffer_size=" << buffer.size() << std::endl;
            break;
        }

        MetaEntry entry;
        entry.phy_rank = static_cast<int>(buffer[offset + kHeaderPhyRankIndex]);
        entry.num_inputs = static_cast<int>(buffer[offset + kHeaderNumInputsIndex]);
        entry.num_outputs = static_cast<int>(buffer[offset + kHeaderNumOutputsIndex]);
        entry.total_input_size = as_int64(buffer[offset + kHeaderTotalInputIndex]);
        entry.total_output_size = as_int64(buffer[offset + kHeaderTotalOutputIndex]);
        entry.dtype = static_cast<int>(buffer[offset + kHeaderDtypeIndex]);
        entry.layout = static_cast<int>(buffer[offset + kHeaderLayoutIndex]);

        const size_t model_len = static_cast<size_t>(buffer[offset + kHeaderModelLenIndex]);
        const size_t backend_len = static_cast<size_t>(buffer[offset + kHeaderBackendLenIndex]);
        const size_t device_len = static_cast<size_t>(buffer[offset + kHeaderDeviceLenIndex]);

        size_t cursor = offset + kHeaderFixedCount;
        entry.model_path = decode_string(buffer, cursor, model_len);
        cursor += model_len;
        entry.backend = decode_string(buffer, cursor, backend_len);
        cursor += backend_len;
        entry.device = decode_string(buffer, cursor, device_len);
        cursor += device_len;

        entry.input_sizes.clear();
        entry.output_sizes.clear();
        entry.input_sizes.reserve(entry.num_inputs);
        entry.output_sizes.reserve(entry.num_outputs);

        for (int i = 0; i < entry.num_inputs; ++i) {
            entry.input_sizes.push_back(as_int64(buffer[cursor + i]));
        }
        cursor += static_cast<size_t>(entry.num_inputs);
        for (int i = 0; i < entry.num_outputs; ++i) {
            entry.output_sizes.push_back(as_int64(buffer[cursor + i]));
        }

        batch.entries.push_back(std::move(entry));
        batch.phase = to_phase(buffer[offset + kHeaderPhaseIndex]);
        offset += static_cast<size_t>(header_len);
    }
    return batch;
}

std::vector<double> build_meta_buffer(const MetaEntry& entry, MetaPhase phase, int field_size) {
    const size_t model_len = entry.model_path.size();
    const size_t backend_len = entry.backend.size();
    const size_t device_len = entry.device.size();

    const size_t header_len = kHeaderFixedCount + model_len + backend_len + device_len +
                              entry.input_sizes.size() + entry.output_sizes.size();
    if (field_size < static_cast<int>(header_len)) {
        throw std::runtime_error("Meta header larger than field size.");
    }

    std::vector<double> buffer(static_cast<size_t>(field_size), 0.0);
    buffer[kHeaderMagicIndex] = kMetaMagic;
    buffer[kHeaderVersionIndex] = static_cast<double>(kMetaVersion);
    buffer[kHeaderPhaseIndex] = static_cast<double>(static_cast<int>(phase));
    buffer[kHeaderPhyRankIndex] = static_cast<double>(entry.phy_rank);
    buffer[kHeaderNumInputsIndex] = static_cast<double>(entry.num_inputs);
    buffer[kHeaderNumOutputsIndex] = static_cast<double>(entry.num_outputs);
    buffer[kHeaderTotalInputIndex] = static_cast<double>(entry.total_input_size);
    buffer[kHeaderTotalOutputIndex] = static_cast<double>(entry.total_output_size);
    buffer[kHeaderDtypeIndex] = static_cast<double>(entry.dtype);
    buffer[kHeaderLayoutIndex] = static_cast<double>(entry.layout);
    buffer[kHeaderModelLenIndex] = static_cast<double>(model_len);
    buffer[kHeaderBackendLenIndex] = static_cast<double>(backend_len);
    buffer[kHeaderDeviceLenIndex] = static_cast<double>(device_len);
    buffer[kHeaderLengthIndex] = static_cast<double>(field_size);

    size_t cursor = kHeaderFixedCount;
    encode_string(entry.model_path, buffer, cursor);
    cursor += model_len;
    encode_string(entry.backend, buffer, cursor);
    cursor += backend_len;
    encode_string(entry.device, buffer, cursor);
    cursor += device_len;

    for (const auto size : entry.input_sizes) {
        buffer[cursor++] = static_cast<double>(size);
    }
    for (const auto size : entry.output_sizes) {
        buffer[cursor++] = static_cast<double>(size);
    }

    return buffer;
}

} // namespace phydll_dl
