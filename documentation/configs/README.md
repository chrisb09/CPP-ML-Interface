# Configuration Examples

These examples use the current config parser, provider headers, and generated
registry in `build/generated_registry.hpp`. CMake generates the registry in the
selected build directory, not in `misc/`. The registered application is
`MLCouplingApplicationFlowExtrapolator`, not `MLCouplingApplicationTurbulenceClosure`.

| File | Purpose | Execution status |
| --- | --- | --- |
| [Root example](../../example.config.toml) | Explicit FlowExtrapolator settings with logging | Construction only with matching C++ buffers |
| [minimal.toml](minimal.toml) | Smallest example using application defaults | Construction only with matching C++ buffers |
| [flow_extrapolator.toml](flow_extrapolator.toml) | Explicit Dummy baseline | Construction only with matching C++ buffers |
| [phydll.toml](phydll.toml) | Phydll provider template | Requires model, MPI/MPMD launch, and DL ranks |
| [aixelerator.toml](aixelerator.toml) | Aixelerator provider template | Requires model, MPI, and compatible AIxelerator backend |
| [smartsim.toml](smartsim.toml) | Smartsim provider template | Requires model and provisioned inference database |

**Dummy is not an identity model or an inference implementation.** Construction
can succeed, but its `static_inference()` intentionally terminates the process
with exit status 1 and `Dummy provider does not implement anything.` This is
not a catchable C++ exception. With no `[behavior]` table the
parser selects the default behavior, so the first `step()` requests inference.
Do not use the Dummy examples to run inference.

## Buffer Contract

TOML selects components and constructor options; it does not allocate the raw
simulation fields. For every example here, the calling C++ code must provide:

- Exactly three contiguous rank-3 input field tensors with equal dimensions.
- Exactly three contiguous writable rank-3 output field tensors, with the same
  dimensions as the inputs for safe field reconstruction.
- Live backing storage throughout the coupling object's lifetime when using
  `wrap_flat`; the last dimension is contiguous (layout `[z, y, x]`).

Use `float` for the complete baseline. Each raw field in the examples is
`[8, 8, 8]` (512 values). The application allocates its own model-side buffers;
both constructors replace supplied intermediate buffers with its cube buffers.
With the settings shown, there is one spatial cube per field, so model input
and output each consist of one contiguous tensor shaped `[3, 1, 512]`.

For other geometries, let `N` be the number of spatial cubes per field and `C`
be `cube_dimension`. Model input is `[3*N, input_sequence_length, C*C*C]` and
model output is `[3*N, forecast_window, C*C*C]`. Batch order is field-major,
then cube; each cube is flattened with x varying fastest. Reconstruction uses
the last forecast slice. Model signatures, scalar types, and backend batching
must agree with these buffers; the placeholder `.pt` paths do not supply such
a model.

Keep `cube_dimension > 0`, `0 <= cube_overlap < cube_dimension`,
`input_sequence_length > 0`, `forecast_window > 0`, and nonnegative
`n_ghost_layers`. Each raw extent minus `2*n_ghost_layers` must be at least
`cube_dimension`. Inputs and outputs should have identical full-grid shapes,
including ghost layers. With longer input sequences, the application repeats
the oldest available snapshot until its history fills; configure sampling
behavior to match the model's training assumptions.

## Construction From C++

Run this from the repository root in a consumer built with the repository's
headers, toml++, the selected build directory's generated registry, and MPI
headers/libraries. This snippet constructs the baseline only, not inference:

```cpp
#include <array>
#include <iostream>
#include <memory>
#include <vector>
#include "ml_coupling.hpp"
#include "config.hpp"

std::array<std::vector<float>, 3> inputs, outputs;
MLCouplingData<float> input_data, output_data;
for (int field = 0; field < 3; ++field) {
    inputs[field].assign(512, 0.0f);
    outputs[field].assign(512, 0.0f);
    input_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        inputs[field].data(), {8, 8, 8}));
    output_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
        outputs[field].data(), {8, 8, 8}));
}
std::unique_ptr<MLCoupling<float, float>> coupling(
    create_mlcoupling_from_config_file<float, float>(
        "documentation/configs/flow_extrapolator.toml", input_data, output_data));
if (!coupling) {
    throw std::runtime_error("Coupling construction failed");
}
// Do not call coupling->step() with Dummy.
```

The existing `src/main.cpp` executable's `--config-file` path passes empty
`MLCouplingData<float>` containers. It cannot construct this application;
changing TOML cannot provide its missing fields. The generic application and
generic library require C++ callbacks (`std::function`), which plain TOML cannot
express. For a runnable callback-based pipeline, construct those components in
C++ as demonstrated in `test/test_generic_implementations.cpp`.

## Parser And Options

Put `coupling_type = "STATIC"` at the top level, before any table. Putting it
under `[general]` does not select the coupling type. In STATIC mode the parser
passes the application's model buffers to the provider. Do not encode buffer
pointers or MPI communicators as TOML integers or strings.

Normalization is optional and deliberately omitted here. FlowExtrapolator
calls `normalize_input()` during preprocessing and `denormalize_output()`
during reconstruction when a normalization component is present. To enable
MinMax, append the following table, replacing the illustrative bounds with the
model's training bounds. Verify that the selected generated registry records
the application's normalization dependency: the currently inspected `build/`
registry has empty dependency lists, so adding this table alone creates a
scaler but does not attach it to the application. Manual construction reliably
wires the scaler; the [coupling guide](../coupling_guide.md) describes this
limitation. The following is the illustrative normalization table:

```toml
[normalization]
class = "minmax"
input_min = -1.0
input_max = 1.0
output_min = -1.0
output_max = 1.0
```

No example specifies behavior: default behavior prepares input and requests
inference every step. If using `Periodic`, its required options are
`inference_interval`, `coupled_steps_before_inference`, `coupled_steps_stride`,
and `step_increment_after_inference`. Set scheduling deliberately for your
simulation and model; it does not make Dummy runnable.

## Backend Requirements

- **Phydll:** Enable `WITH_PHYDLL`, initialize the MPI runtime, and launch
  physics and matching DL processes together. The constructor initializes
  coupling and exchanges metadata, so even construction requires the peer
  processes. See `dl_clients/phydll_dl_client.py`, `dl_clients/dl_client.cpp`,
  and their launch scripts. Required option: `model_file`. Constructor defaults
  are `backend="TORCH"`, `device="GPU"`, `batch_size=0`, and
  `transport_layout="auto"`; this template explicitly selects CPU. Transport
  layout must be compatible with the DL runtime.
- **Aixelerator:** Enable `WITH_AIX` with a compatible compiled model backend,
  initialize MPI before constructing the coupling, and destroy it before MPI
  finalization. Required option: `model_file`. Defaults are `batchsize=1`,
  `app_comm=MPI_COMM_WORLD`, `enable_hybrid=false`, `host_fraction=std::nullopt`,
  and `communication_mode="collective"`. The alternative communication mode
  is `pipelined`. This provider requires equal library input/output scalar
  types, either float/float or double/double. The backend is selected by the
  installed service and model, not a `backend` TOML key.
- **Smartsim:** Enable `WITH_SMARTSIM` and provide a reachable SmartSim/RedisAI
  database supporting the chosen model backend and device. Required options
  for the file-based constructor are `device`, `model_backend`, and
  `model_path`; `model_name` defaults to `model`. The template explicitly
  selects one CPU database (`nodes=1`, `num_gpus=0`, `db_layout="shared"`).
  Replace host and port with the real endpoint, or remove both and set `SSDB`.
  Constructor defaults are `host=""`, `port=-1`, `nodes=-1`, `num_gpus=-1`,
  `first_gpu=0`, `batch_size=0`, `min_batch_size=0`, `min_batch_timeout=0`, and
  `db_layout="shared"`; timeout options default to `-1`. Provisioning the
  database is external to TOML, and construction connects to it and loads the
  model.

The backend templates are not runnable as shipped. Replace every
`REPLACE_WITH_*` placeholder and satisfy the launch, model, and buffer contracts
before construction or inference. Provider names appearing in the registry do
not imply that their optional dependencies were enabled in your build.
