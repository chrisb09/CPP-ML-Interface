# PhyDLL DL Client: Dynamic Tensor Shapes and Chunked Inference

The standalone C++ and Python DL clients used by the PhyDLL coupling provider
(`dl_clients/dl_client.cpp` and `dl_clients/phydll_dl_client.py`) used to make a
hardcoded assumption about the input tensor shape: they forced every input
into `[batch_size, features]` where `batch_size` was hard-replaced with the
number of physics ranks (`ndest`). That worked for tests with a single sample
per rank but broke at scale (e.g. `[96, 18] is invalid for input of size
671846400`) and produced wrong outputs whenever the PHY side wanted a real
batch dimension.

The DL clients are now fully dynamic and support per-sample chunking for OOM
prevention. This document describes the new metadata handshake and how the
clients use it.

---

## 1. P2P metadata header

The C++ provider sends a `BcastMetaHeader` (and a payload of variable-length
strings) to the DL client at startup:

```cpp
struct BcastMetaHeader {
    int32_t magic;          // 0x4D4C434D 'MLCM'
    int32_t version;        // 2 (v2 protocol)
    int32_t model_len;      // length of model path string
    int32_t backend_len;    // length of backend name string
    int32_t device_len;     // length of device name string
    int32_t batch_size;     // per-call chunk size, 0 = no chunking
    int32_t num_inputs;     // number of distinct input tensors
    int32_t num_outputs;    // number of distinct output tensors
    int64_t total_input;    // total input element count across all inputs
    int64_t total_output;   // total output element count across all outputs
    int64_t field_size;     // physical field size sent by this source rank
    int32_t dtype;          // element dtype tag
    int32_t layout;         // element layout tag
    int32_t num_input_dims; // total dim count across all input tensors
    int32_t num_output_dims;// total dim count across all output tensors
};
// (Total: 72 bytes)
```

The Python client unpacks the same layout:

```python
magic, version, m_len, b_len, d_len, batch_size_arg, n_in, n_out, \
    t_in, t_out, f_size, dtype, layout, n_in_dims, n_out_dims = \
    struct.unpack("=8i 3q 4i", header_buf)
```

`batch_size` is the **per-call chunk size** that the PHY side intends to
broadcast (e.g. the number of physics cells the solver wants to fit into a
single forward pass on the DL side). `0` disables chunking.

---

## 2. Per-sample reshaping

The clients compute the true number of samples from the metadata, not from
`ndest`:

```cpp
long long client_batch_size = 1;
if (!final_meta.input_shapes.empty() && !final_meta.input_shapes.front().empty()) {
    client_batch_size = final_meta.input_shapes.front().front();
}
const long long batch_size = std::max(1LL, static_cast<long long>(ndest) * client_batch_size);
std::vector<long long> rank_field_offsets(ndest + 1, 0);
for (int i = 0; i < ndest; ++i) {
    rank_field_offsets[i + 1] = rank_field_offsets[i] + meta_per_rank[i].field_size;
}
```

For each output element `b`, the client maps it to `(client_id, sample_id)`
and uses cumulative per-rank field offsets (`rank_field_offsets[client_id]`) to walk non-uniform rank partitions:

```cpp
long long client_id = b / client_batch_size;
long long sample_id = b % client_batch_size;
long long src_start = rank_field_offsets[client_id] + sample_id * input_per_rank_used;
```

The Python client mirrors the same math.

The result is a tensor of shape `[batch_size, ...remaining_dims]`, which is
exactly what the TorchScript model expects, regardless of how many
physics ranks or how many samples per rank the PHY side chose to send.

---

## 3. Chunked inference (OOM prevention)

For massive flat inputs (e.g. 37M cells through a width-4096 MLP) a single
`forward()` allocates hundreds of GiB of intermediate GPU memory and OOMs.
`final_meta.batch_size` is used as a chunk size to slice the input tensor
and run `forward()` in a loop, concatenating the per-chunk outputs:

```cpp
torch::NoGradGuard no_grad;
long long max_chunk_size = final_meta.batch_size > 0
    ? static_cast<long long>(final_meta.batch_size)
    : batch_size;
std::vector<torch::Tensor> outputs;
for (long long chunk_idx = 0; chunk_idx < batch_size; chunk_idx += max_chunk_size) {
    long long chunk_size = std::min(max_chunk_size, batch_size - chunk_idx);
    auto chunk_tensor = input_tensor.slice(0, chunk_idx, chunk_idx + chunk_size);
    outputs.push_back(model.forward({chunk_tensor}).toTensor());
}
auto output_tensor = torch::cat(outputs, 0);
```

The Python client mirrors the same loop under `torch.no_grad()`.

Setting `batch_size = 0` (in the TOML `[provider]` section) disables
chunking entirely, which is the right default for small models and for
verifying that the dynamic-reshape path produces identical outputs across
batch sizes.

---

## 4. Configuration

In the solver's TOML, the new option is a top-level `[provider]` field:

```toml
[provider]
class = "Phydll"
backend = "TORCH"
model_file = "./train_models/model_a/best_model_giant_mlp.pt"
device = "GPU"
batch_size = 500000   # chunk size; 0 disables chunking
```

For the 96-rank terrain_solver use case with a width-4096 MLP, a value of
`500000` is a good default (it slices the 37M-element buffer into ~75
chunks). Smaller models don't need it; set `0`.

---

## 5. Behaviour summary

| TOML `batch_size` | Per-call chunks | GPU memory | Speed |
|-------------------|-----------------|------------|-------|
| `0`               | 1 (full batch)  | High       | Fastest for small models |
| `500000`          | ~75 of 500k     | Bounded    | Negligible overhead |
| too small         | thousands       | Lowest     | Per-call overhead dominates |
