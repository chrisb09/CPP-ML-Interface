# PhyDLL Transport Layouts: `packed` vs `uniform_chunks`

The PhyDLL provider supports two wire layouts for moving data between the
physics solver ranks and the DL client. The layout is selected per solver via
the TOML `[provider]` option `transport_layout`.

- `auto` (default): automatically selects `uniform_chunks` when input and output
  shapes have a valid positive $\gcd$ deriving field counts $\le 4096$; otherwise
  safely falls back to `packed`.
- `uniform_chunks`: forces the many-fields transport that removes response padding when
  the input and output sizes differ strongly (e.g. `[B,18] -> [B]`).
- `packed`: forces the original single-field transport with zero padding.

No PhyDLL core changes are required for either layout; only how the provider
and the C++ DL client use the PhyDLL API differs.

---

## 1. How PhyDLL transports data

PhyDLL gives every coupled rank a *field size* and a *field count*:

```c
phydll_define_phy(int count, int size); // solver side
phydll_define_dl(int count);            // DL side
```

- The field **size** is a number of `double`s, and the DL side treats each PHY
  field as the concatenation of all coupled solver ranks' segments.
- The field **counts** may differ per direction: `count` on the PHY side and
  `count` on the DL side are independent.

Per inference frame, every PHY rank sends `phy_count` fields of `field_size`
doubles to the DL client, and the DL client sends `dl_count` fields back. PhyDLL
slices each DL field into per-source segments automatically.

## 2. `packed` layout (default)

The original behaviour:

- `phy_count = dl_count = 1`.
- `field_size = max(total_input, total_output, metadata_header)`.
- All input tensors are flattened into one field; the smaller logical direction
  is zero-padded on the wire.
- Used for everything before this change; byte-for-byte behaviour is preserved.

**Good for:** models where the input and output widths are similar, or small
batches where message count dominates.

## 3. `uniform_chunks` layout

The provider derives uniform chunks from the flattened per-rank logical element
counts:

```text
g          = gcd(total_input_elements, total_output_elements)
field_size = g
phy_count  = total_input_elements  / g
dl_count   = total_output_elements / g
```

For `[B, 18] -> [B]`:

```text
g          = B
phy_count  = 18     (18 input chunks of B doubles)
dl_count   = 1      (1  output field of B doubles)
```

Every PHY chunk is a consecutive `g`-element slice of the flattened input. The
DL client reconstructs the rank-major flattened input directly into the CPU
float tensor it hands to Torch, fusing the field/rank transpose with the
`double -> float` conversion. The output is built field-major and sent back;
PhyDLL slices each rank's segment automatically.

**Benefits:**
- The response is no longer padded: `[B,18] -> [B]` sends `B` doubles back per
  rank instead of `18*B`.
- Per-rank field sizes may differ (e.g. rank 0 `B=3`, rank 1 `B=2`); PhyDLL's
  source-size aggregation handles this.
- Reconstruction is fused with the transport cast, avoiding intermediate
  `vector<double>` copies in the DL client.

**Costs / constraints:**
- More MPI messages: `[B,18] -> [B]` with 24 solver ranks becomes 432 input
  fields plus 24 output fields per frame. PhyDLL posts these as non-blocking
  `MPI_Isend`s and waits once, so they are not transferred strictly serially,
  but message/label overhead and network contention can dominate for small `B`.
- The derived field counts must be identical across all solver ranks.
- `gcd` can be `1`, producing an impractical number of tiny messages; consider
  the packed layout for such shapes.
- Frame-synchronous like the packed layout: PhyDLL exposes only whole-frame
  completion, so partial/streaming inference is not possible.

## 4. Configuration

In the solver TOML:

```toml
[provider]
class = "Phydll"
backend = "TORCH"
model_file = "./train_models/model_a/best_model.pt"
device = "GPU"
batch_size = 500000   # DL-side chunk size; 0 disables chunking
transport_layout = "auto"   # "auto" (default), "uniform_chunks", or "packed"
```

The DL client must be launched with the matching output field count:

```bash
export PHYDLL_DL_FIELD_COUNT=1        # [B,18] -> [B]
# or
export PHYDLL_DL_FIELD_COUNT=18       # [B] -> [B,18]
```

The client validates this against the metadata sent by the provider and aborts
with a clear error on mismatch.

## 5. Support matrix

| Component              | `packed` | `uniform_chunks` |
|------------------------|:--------:|:----------------:|
| C++ DL client          | yes      | yes              |
| Python DL client       | yes      | yes              |
| Multi-tensor models    | yes      | one flattened input/output pair only |
| Differing local batch per rank | yes | yes |

## 6. Regression harness

An MPMD test covers both layouts in both directions with unequal local batch
sizes (rank 0 `B=3`, rank 1 `B=2`):

```bash
cd test/phydll_mpmd
sbatch submit_phydll_mpmd_test.sh 18to1 packed
sbatch submit_phydll_mpmd_test.sh 18to1 uniform_chunks
sbatch submit_phydll_mpmd_test.sh 1to18 packed
sbatch submit_phydll_mpmd_test.sh 1to18 uniform_chunks
```

Use the Python DL client instead of the C++ one by setting
`PHYDLL_TEST_DL_PYTHON=1`:

```bash
PHYDLL_TEST_DL_PYTHON=1 sbatch submit_phydll_mpmd_test.sh 18to1 uniform_chunks
```

- Deterministic TorchScript models: `scripts/create_phydll_asym_models.py`.
- Solver executable: `test/phydll_phy_test.cpp`.
- Env overrides: `PHYDLL_BATCH_CHUNK` (DL inference chunk size),
  `PHYDLL_TEST_DL_PYTHON=1` (use the Python DL client).
- Metadata-header decode unit test: `test/phydll_mpmd/test_metadata_decode.py`.

## 7. Choosing a layout

Benchmark both for the target model, rank count, and batch size. As a rule of
thumb:

- **Output width ≪ input width** (e.g. `[50k, 18] -> [50k]`): `uniform_chunks`
  is attractive; it reduces the response from `18*B` doubles to `B` doubles per
  rank per frame.
- **Similar widths, small batches, or `gcd == 1`**: stick with `packed` to keep
  the message count low.
