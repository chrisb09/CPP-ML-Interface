# Coupling Guide

This guide follows `include/`, `src/c_api.cpp`, and the inspected `build/generated_registry.hpp`, not old registries under `misc/`. Constructor defaults come from the headers; generated-factory differences are called out below.
See also the [API guide](api_guide.md), [transport layouts](phydll_transport_layouts.md),
[construction diagram](tikz/coupling_construction.pdf), and
[static-step diagram](tikz/coupling_flow.pdf). The diagrams are separate
150 mm-wide thesis drafts, not yet integrated into the thesis.

## Data And Control

`MLCoupling<CI, CO, LI = CI, LO = CO>` separates four scalar types:

| Type | Boundary | Container |
| --- | --- | --- |
| `CI` | Solver input | `application.coupling_input` |
| `LI` | Preprocessed model input | `application.library_input` |
| `LO` | Raw model output | `application.library_output` |
| `CO` | Solver output | `application.coupling_output` |

Applications have the same template order; libraries are `MLCouplingLibrary<LI, LO>`. Application normalization is `MLCouplingNormalization<LI, CO>`, **not** `<LI, LO>`.
`MLCouplingData<T>` is a collection of tensors. Each `MLCouplingTensor<T>` records scalar type, dimensions, layout, and ownership. Row-major/column-major contiguous buffers and nested pointer trees are distinct layouts. A layout tag describes memory; it does not transpose or convert it.

The ordinary `step()` delegates to the application:

1. Ask behavior whether to infer.
2. Prepare library input using the application's preprocessing hook.
3. Call `library.static_inference(&library_input, &library_output)`.
4. Finalize solver output using the postprocessing hook.
5. Return `behavior.time_step_delta()`; return zero if inference was skipped.

**The C++ return value is a timestep delta, not a success flag.** Default behavior always infers and returns **zero**, so zero does not prove inference was skipped. The solver decides how to apply the delta alongside ordinary timestep advancement. C functions return a status and write the delta to an output argument.
Use exceptions, C statuses, and output checks for validation. `guarantee(false, ...)` terminates the process rather than throwing, so C bindings cannot catch every failure.

Base preprocessing/postprocessing returns the passed container for matching types, potentially rebinding a buffer rather than copying values. Mixed types require conversion hooks; the base throws `logic_error` otherwise. Four buffers alone do not implement conversion. Generic callbacks must return the desired container and write the intended destination. A Generic `ml_step_fn` replaces the entire orchestration.

**Normalization is stored/owned by the base application, but not automatically called by base or default Generic execution.** FlowExtrapolator explicitly calls it. Generic users must implement scaling in callbacks/custom application code or call a separate scaler. Some API comments describe automatic normalization or delta one; the implementations are authoritative.

FlowExtrapolator calls `should_send_data()` before `should_perform_inference()`, collecting history on send steps. Base/Generic ordinary execution does not query send. Scheduling behaviors advance counters on inference queries: do not query again merely to log a decision.

## Ownership

- `MLCoupling` takes raw library/application/behavior ownership through `unique_ptr`. Allocate transferred components with `new`, not on the stack; do not delete or transfer twice. Null behavior creates Default.
- Applications own raw normalization pointers. Provider and explicit coordinator buffer pointers are borrowed.
- Tensor/container copies share owned storage, not deep-copy values; external memory stays borrowed. Use `deep_copy()` or `from_flat_copy()` for independent storage.
- Wrapped memory must remain alive and stable throughout use, including staged operations/callbacks. Do not resize wrapped vectors, deallocate Fortran targets, or return temporary-backed views.
- Owned flat wrapping uses `delete[]`: never mark C `malloc`, stack, vector, or Fortran storage owned. Prefer copy creation for interface-owned memory.
- Initialize MPI before MPI-dependent providers; destroy providers before finalizing MPI. PhyDLL destruction finalizes its connection and frees its control communicator. Keep service buffers stable.

## Manual C++

This complete callback example needs no model/server. It wraps vectors, infers twice, and checks output independently of delta. The callback is a test operation, not a trained model.

```cpp
#include <iostream>
#include "ml_coupling.hpp"
#include "application/ml_coupling_application_generic.hpp"
#include "library/ml_coupling_library_generic.hpp"
#include <cassert>
#include <vector>

int main(int, char**) {
    std::vector<double> x{1, 2, 3}, y(3, 0);
    using Tensor = MLCouplingTensor<double>;
    using Data = MLCouplingData<double>;
    Data input(std::vector<Tensor>{Tensor::wrap_flat(x.data(), {3})});
    Data output(std::vector<Tensor>{Tensor::wrap_flat(y.data(), {3})});
    auto* library = new MLCouplingLibraryGeneric<double, double>(
        [](Data* in, Data* out) {
            for (size_t i = 0; i < (*in)[0].numel(); ++i)
                (*out)[0].set_linear(i, 2 * (*in)[0].at_linear(i));
        });
    auto* application = new MLCouplingApplicationGeneric<double, double>(
        input, output);
    MLCoupling<double, double> coupling(library, application);
    int delta = coupling.step();
    assert(delta == 0); // Default behavior still performed inference.
    assert((y == std::vector<double>{2, 4, 6}));
    x[1] = 5;
    delta = coupling.step();
    assert(delta == 0 && y[1] == 10);
} // Coupling dies before x/y; transferred components are deleted here.
```

Use the configured target's includes/definitions and link dependencies in solver builds. `ml_coupling.hpp` includes configuration and the generated registry: this example needs build includes, toml++, and MPI headers even with backends disabled. From the repository root:

```sh
mpicxx -std=c++17 -Iinclude -Ibuild -Ibuild/_deps/tomlplusplus-src/include generic.cpp -o generic
./generic
```

The leading `<iostream>` works around a library header using `std::cerr` without including it; `main(int, char**)` matches the coordinator's friend declaration.

For real providers construct the application first, obtain `auto [li, lo] = application->get_library_buffers()`, and pass the borrowed pointers to the provider. With three valid flow fields, a manual mixed-precision setup is:

```cpp
// raw_input/raw_output: MLCouplingData<double>, each with three matching 3D fields.
auto* app = new MLCouplingApplicationFlowExtrapolator<double, double, float, float>(
    raw_input, raw_output,
    new MLCouplingMinMaxNormalization<float, double>(-1.f, 1.f, -1., 1.),
    8, 0, 6, 24, 0);
auto [li, lo] = app->get_library_buffers();
auto* lib = new MLCouplingLibraryAixelerator<float, float>(
    "model.pt", 1, MPI_COMM_WORLD, false, std::nullopt, li, lo);
auto* behavior = new MLCouplingBehaviorFlowExtrapolator(
    50, 6, 1, 1000, 10000, 1.0, 24, 1, 6, 0);
MLCoupling<double, double, float, float> coupling(lib, app, behavior);
```

This fragment needs `application/ml_coupling_application_flow_extrapolator.hpp`, `behavior/ml_coupling_behavior_flow_extrapolator.hpp`, and `provider/ml_coupling_provider_aixelerator.hpp`, enabled AIxelerator, initialized MPI, and a compatible model. It is not standalone.

## Constructor Reference

**Required** means no C++ default; parameters are in constructor order. `Data<T>` abbreviates `MLCouplingData<T>`, `Norm` means `MLCouplingNormalization<LI, CO>`. Callbacks are `std::function` unless stated otherwise. Base Behavior/Library/Normalization are abstract.

### Coordinator And Data

| Constructor | Parameters and exact defaults |
| --- | --- |
| `MLCoupling<CI,CO,LI,LO>` ordinary | `MLCouplingLibrary<LI,LO>* library` required; `MLCouplingApplication<CI,CO,LI,LO>* application` required; `MLCouplingBehavior* behavior = nullptr`; `std::string log_level = ""`; `std::optional<bool> error_separate = std::nullopt` |
| `MLCoupling` explicit mode | Same required `library`, `application`; required `behavior`, `CouplingType coupling_type`, `Data<LI>* library_input`, `Data<LO>* library_output`; `log_level = ""`; `error_separate = std::nullopt` |
| `MLCouplingTensor<T>()` | No parameters; empty tensor |
| `MLCouplingTensor<T>(...)` | `void* root`, `std::vector<int> dimensions`, `MLCouplingMemoryLayout layout` required; `MLCouplingOwnership ownership = MLCouplingOwnershipExternal`; `std::function<void(void*)> deleter = {}` |
| `Tensor::wrap_flat` | `T* data`, `std::vector<int> dimensions` required; `layout = MLCouplingMemLayoutContiguous`; `ownership = MLCouplingOwnershipExternal` |
| `Tensor::wrap_nested` | `void* root`, `std::vector<int> dimensions` required; `layout = MLCouplingMemLayoutNested`; `ownership = MLCouplingOwnershipExternal`; `deleter = {}` |
| `Tensor::from_flat_copy` | `const std::vector<T>& values`, `const std::vector<int>& dimensions` required; `layout = MLCouplingMemLayoutContiguous` |
| `MLCouplingData<T>` | Either no parameters (empty), or required `std::vector<MLCouplingTensor<T>> tensors` |
| `MLCouplingScalar<T>` | Required `T value` (owned copy); alternatively required `T* ptr`, `ownership = MLCouplingOwnershipExternal` |
| `MLCouplingVector<T>` | Required `std::vector<T> values` (owned copy); alternatively required `T* ptr`, `int size`, `ownership = MLCouplingOwnershipExternal` |
| `MLCouplingMatrix<T>` | Required `T* ptr`, `int rows`, `int cols`, `ownership = MLCouplingOwnershipExternal`; alternatively required `std::vector<std::vector<T>> values` (owned row-major copy) |

Ordinary construction selects STATIC and application buffers. Explicit STATIC construction requires non-null pointers, not necessarily nonempty/model-compatible tensors. Empty logging arguments retain global settings.

### Applications

| Constructor | Parameters and exact defaults |
| --- | --- |
| `MLCouplingApplication<CI,CO,LI=CI,LO=CO>` two-buffer | `Data<CI> coupling_input`, `Data<CO> coupling_output` required; `Norm* normalization = nullptr` |
| Base four-buffer | `Data<CI> coupling_input`, `Data<LI> library_input`, `Data<LO> library_output`, `Data<CO> coupling_output` required; `normalization = nullptr` |
| `MLCouplingApplicationGeneric` two-buffer | Required `coupling_input`, `coupling_output`; `PreprocessFn preprocess_fn = nullptr`; `PostprocessFn postprocess_fn = nullptr`; `MlStepFn ml_step_fn = nullptr`; `normalization = nullptr` |
| Generic four-buffer | Required `coupling_input`, `library_input`, `library_output`, `coupling_output`; same four optional parameters, in the same order |
| `MLCouplingApplicationFlowExtrapolator` two-buffer | Required `coupling_input`, `coupling_output`; `normalization = nullptr`; `int cube_dimension = 8`; `int cube_overlap = 0`; `int input_sequence_length = 1`; `int forecast_window = 1`; `int n_ghost_layers = 0` |
| FlowExtrapolator four-buffer | Required `coupling_input`, `library_input`, `library_output`, `coupling_output`; same six optional parameters, in the same order |

Generic callback signatures are:

| Callback | Signature |
| --- | --- |
| `PreprocessFn` | `Data<LI>(Data<CI>)` |
| `PostprocessFn` | `Data<CO>(Data<LO>)` |
| `MlStepFn` | `int(MLCouplingLibrary<LI,LO>&, MLCouplingBehavior&)` |

Base construction aliases empty library buffers to coupling buffers only for matching types. FlowExtrapolator **allocates/replaces library buffers in both constructors**, even when four buffers were supplied.
It needs exactly three contiguous matching 3D input fields and three matching output fields. Provide identical input/output grid extents and row-major storage for raw indexing. Ghost layers are excluded from the active region and preserved in output. Cube/sequence/forecast sizes must be positive, active extents sufficient, overlap in `[0,cube_dimension)`, and ghost depth nonnegative.
Model shapes are `[3*num_cubes, input_sequence_length, cube_dimension^3]` and `[3*num_cubes, forecast_window, cube_dimension^3]`. Missing early history repeats the earliest snapshot. Reconstruction averages overlaps and uses the **last** forecast slice, not the entire forecast sequence.

### Behaviors

| Constructor | Parameters and exact defaults |
| --- | --- |
| `MLCouplingBehavior` | No explicit constructor parameters; abstract |
| `MLCouplingBehaviorDefault` | No parameters; infer/send true, delta zero |
| `MLCouplingBehaviorGeneric` | Required `std::function<bool()> should_infer`, `std::function<int()> time_step_delta`, `std::function<bool()> should_send_data`; all must be nonempty |
| `MLCouplingBehaviorPeriodic` | Required `int inference_interval`, `int coupled_steps_before_inference`, `int coupled_steps_stride`, `int step_increment_after_inference`; `std::function<bool(int)> prohibit_inference = allow_inference_at_all_steps` |
| `MLCouplingBehaviorFlowExtrapolator` | Required `int inference_interval`, `int coupled_steps_before_inference`, `int step_increment_after_inference`, `int hdf_output_interval`, `int total_timesteps`; `double scaling_factor = 1.0`; `int forecast_window = 1`; `int input_step_distance = 1`; `int inference_start_step = 0`; `int global_step_offset = 0` |

Periodic increments its counter on each inference query, starting at one. Inference needs a multiple of the positive interval, `count >= before*stride`, and a false prohibition predicate. Before/increment must be nonnegative and stride positive. The default prohibition returns false. Send uses the current counter and strict `steps_until_next < before*stride`; behavior itself does not collect history.

Flow behavior tracks logical calls and effective global time after jumps. Delta is `round(step_increment_after_inference * scaling_factor * forecast_window)`; history stride is at least one after rounding `input_step_distance * scaling_factor`. Interval/total must be positive and history count at least one; `hdf_output_interval <= 0` disables output-collision checks.
**Set a reachable positive `inference_start_step`: default zero never matches the counter starting at one.** Allow sufficient history and align application forecast/history lengths with behavior.

### Normalization

| Constructor | Parameters and exact defaults |
| --- | --- |
| `MLCouplingNormalization<In,Out>` | No explicit constructor parameters; abstract |
| `MLCouplingMinMaxNormalization<In,Out>` bounds | Required `In input_min`, `In input_max`, `Out output_min`, `Out output_max` |
| MinMax raw samples | Required `In* input_data`, `int input_data_size`, `Out* output_data`, `int output_data_size` |
| MinMax containers | Required `Data<In> input_data`, `Data<Out> output_data` |
| `MLCouplingNormalizationGeneric<In,Out>` | Required `std::function<void(Data<In>)> normalize_input_fn`, `std::function<void(Data<Out>)> denormalize_output_fn`; `std::function<void(std::ostream&)> print_fn = nullptr` |

MinMax computes `(x-input_min)/(input_max-input_min)` and `y*(output_max-output_min)+output_min` in place. Sample constructors compute fixed bounds at construction. Supply nonempty samples and nonzero input range; no constant-range guard or clipping exists. Floating types are normally appropriate. Generic requires both transformations; printing is optional. Application normalization uses `In=LI`, `Out=CO`.

### Libraries And Providers

Concrete providers derive from `MLCouplingLibrary<LI,LO>` (also named `In,Out`). Optional data pointers below are borrowed.

| Constructor | Parameters and exact defaults |
| --- | --- |
| `MLCouplingLibrary<LI,LO>` | No parameters (rank starts at zero, detects initialized MPI when enabled); alternatively required `int rank`; abstract |
| `MLCouplingLibraryGeneric` | Required `InferenceFn inference_fn`; `TrainFn train_fn = nullptr`; `SyncIterFn sync_iter_fn = nullptr` |
| Generic explicit rank | Required `int rank`, `InferenceFn inference_fn`; `train_fn = nullptr`; `sync_iter_fn = nullptr` |
| `MLCouplingLibraryDummy` | `Data<LI>* input_after_preprocessing = nullptr`; `Data<LO>* output_before_postprocessing = nullptr` |
| `MLCouplingLibraryAixelerator` | Required `std::string model_file`; `int batchsize = 1`; `MPI_Comm app_comm = MPI_COMM_WORLD`; `bool enable_hybrid = false`; `std::optional<float> host_fraction = std::nullopt`; `Data<LI>* input_after_preprocessing = nullptr`; `Data<LO>* output_before_postprocessing = nullptr`; `std::string communication_mode = "collective"` |
| `MLCouplingLibraryPhydll` | Required `std::string model_file`; `std::string backend = "TORCH"`; `std::string device = "GPU"`; `int batch_size = 0`; `std::string transport_layout = "auto"`; `Data<LI>* input_after_preprocessing = nullptr`; `Data<LO>* output_before_postprocessing = nullptr`; `bool solver_readiness_wait = false` |

Generic signatures: `InferenceFn = void(Data<LI>*, Data<LO>*)`, `TrainFn = std::map<std::string,double>(Data<LI>*, Data<LO>*)`, `SyncIterFn = std::size_t(std::size_t)`. Inference is required; absent training throws, absent synchronization returns local iterations.
**Dummy is a construction placeholder: inference terminates the process.** Use Generic for executable mocks.

AIxelerator needs `WITH_AIX`, service/model dependencies, and matching input/output scalar types (`float` or `double`). It uses the first tensor, assuming flat contiguous storage. Service buffers bind eagerly with both pointers supplied, or lazily at inference; keep them stable. Communication accepts `collective` (also `default`/empty) or `pipelined` (also `p2p`), case-insensitively.

PhyDLL needs `WITH_PHYDLL` and a matching MPMD DL client/launch, not a local model loader. Transport choices `auto`, `uniform_chunks`, `packed` are case-sensitive. `solver_readiness_wait` optionally polls with 100 microsecond sleeps; it is not a deadline and still completes the payload receive. See the transport guide.

SmartSim has **two public constructors**: one takes a model file (`std::string model_path`), the other serialized model bytes (`std::string_view model`). The table lists their parameters in exact order; the third parameter differs by overload, and all subsequent parameters/defaults are identical.

| Parameter type and name | C++ default | Meaning / constraint |
| --- | --- | --- |
| `std::string device` | Required | `CPU` or `GPU` |
| `std::string model_backend` | Required | `TF`, `ONNX`, `TFLITE`, `TORCH` |
| `std::string model_path` (file overload) or `std::string_view model` (serialized overload) | Required | Third parameter; file route preferred, serialized limitation below |
| `std::string model_name` | `"model"` | Nonempty database model key |
| `std::string host` | `""` | Single endpoint host |
| `int port` | `-1` | Single endpoint port |
| `int nodes` | `-1` | Resolve explicitly or through `MLCOUPLING_SMARTSIM_NODES`; must be positive |
| `int num_gpus` | `-1` | Resolve from environment; CPU resolves to zero |
| `int first_gpu` | `0` | Nonnegative, less than positive GPU count |
| `int batch_size` | `0` | Nonnegative backend batching setting |
| `int min_batch_size` | `0` | Nonnegative minimum batch size |
| `int min_batch_timeout` | `0` | Nonnegative batching timeout, milliseconds |
| `int command_timeout` | `-1` | SmartRedis command timeout, seconds; -1 leaves environment unchanged |
| `int socket_timeout` | `-1` | Socket timeout, seconds; -1 leaves environment unchanged |
| `int model_timeout` | `-1` | Model timeout, milliseconds; also controls follower model polling |
| `const std::vector<std::string>& tf_input_labels` | `{}` | Current validation requires empty labels |
| `const std::vector<std::string>& tf_output_labels` | `{}` | Current validation requires empty labels |
| `const std::vector<std::string>& tf_input_keys` | `{}` | Optional input tensor keys |
| `Data<LI>* input_after_preprocessing` | `nullptr` | Borrowed model input container |
| `Data<LO>* output_before_postprocessing` | `nullptr` | Borrowed model output container |
| `std::string db_layout` | `"shared"` | Shared DB or `per-ml-node` |

Both public constructors delegate to a **private shared initializer**, not a third user-callable constructor. Its additional `hosts`/`ports` vectors are internal-only; both public overloads pass empty vectors. They cannot be supplied through public manual construction or the inspected registry. Prefer explicitly typed `std::string` or `std::string_view` model arguments to distinguish the two public overloads.

SmartSim requires `WITH_SMARTSIM`, SmartRedis, and a live compatible database/backend. Public callers supply `host`/`port` or use `SSDB` (which can hold multiple endpoints); an explicit single endpoint wins. Shared layout permits one or at least three DB nodes, not two; per-ML-node assigns rank blocks to standalone DBs. Timeouts set environment variables only if not already set.
The private initializer passes an **empty model view to validation**, so the public serialized-model overload fails with its empty model path. Validation also requires both TF label vectors to be empty, even for TF/TFLITE, despite its diagnostic suggesting otherwise. Use file models and empty labels.

## Configuration

Companion templates: [FlowExtrapolator](configs/flow_extrapolator.toml), [PhyDLL](configs/phydll.toml), [AIxelerator](configs/aixelerator.toml), [SmartSim](configs/smartsim.toml). Supply model paths, buffers, services, and enabled builds; these are not self-contained tests.
Root [example.config.toml](../example.config.toml) is a construction-only Dummy/FlowExtrapolator example requiring three live 3D fields. It is not an inference recipe: Dummy fails when inference is reached. Older copies selecting TurbulenceClosure use an unregistered application.

### Schema

| Location | Values and behavior |
| --- | --- |
| Top-level `coupling_type` | String `STATIC` (default) or `FLEXIBLE`, case-insensitive; place **before** TOML table headers |
| `[library]` | Required `class` and constructor arguments; `[provider]` is an alternative spelling, not a second provider |
| `[application]` | Required `class` and application arguments; buffers are injected by the factory |
| `[behavior]` | Optional `class` and constructor arguments; absent class selects Default |
| `[normalization]` | Optional `class` and normalization arguments; wiring caveat below |
| `[logging]` | `level` string, `error_separate` boolean; omission retains settings |

`[general].coupling_type` is **not read**. Supported values are scalar integers (`int64_t`), floating values (`double`), strings, booleans, and homogeneous string arrays. Numeric arrays, nested schemas, callbacks, MPI communicators, and typed C++ objects are not ordinary TOML arguments.
Names match constructors exactly: AIxelerator `batchsize` versus PhyDLL/SmartSim `batch_size`. Matching defaults to Strict; unknown constructor keys fail. Cast mode defaults to Relaxed; `ConfigCastMode::Strict` limits coercions and `ConfigParameterMatchMode::Lenient` allows unused keys. Neither validates shapes or makes opaque pointers type-safe.

The inspected build registry recognizes full concrete class names and these aliases:

| Category | Names / aliases |
| --- | --- |
| Library | `Aixelerator`, `aixelerator`, `AIxelerator`, `aix`, `AIx`, `AIX`; `Dummy`, `dummy`; `Phydll`, `phydll`, `PhyDLL`; `Smartsim`, `smartsim`, `SmartSim` |
| Application | `MLCouplingApplicationFlowExtrapolator`, `flow-extrapolator`, `flow_extrapolator`, `maia-flow-extrapolator` |
| Behavior | `Default`, `default`; `Periodic`, `periodic`; `FlowExtrapolatorBehavior`, `flow-extrapolator-behavior`, `maia-flow-extrapolator-behavior` |
| Normalization | `MinMax`, `minmax`, `min-max`, `MinMaxNormalization` |

Generic classes use full names, e.g. `MLCouplingApplicationGeneric`. Registry membership does not imply a provider was compiled in. Inspect the selected build's registry, not `misc/`.
Inspected `build/` signatures omit PhyDLL `solver_readiness_wait`; that public parameter can be supplied manually. SmartSim `hosts`/`ports` belong to its private initializer and are unavailable both through the registry and public manual constructors; use `SSDB` for multiple endpoints. Registry dependency lists are empty: `[normalization]` creates a scaler but does **not** attach it to an application. Use manual construction for guaranteed wiring, or verify regenerated dependencies.

Construction order: normalization, application, library, behavior, coordinator. STATIC injects provider buffer pointers; Generic callback libraries may need Lenient matching because those injected names are not Generic arguments.
`MLCOUPLING_LOG_LEVEL` and `MLCOUPLING_LOG_ERROR_SEPARATE` override TOML logging. `ConfigOverrides` does not change separately extracted logging/coupling type. Malformed TOML and several construction errors call `exit(1)`; other paths return null. Not all failures are recoverable exceptions.

### C++ Factory Use

Adopt raw owning factory results into `unique_ptr`. This construction-only example uses the manual example's live `input`/`output` (do **not** call Dummy `step()`):

```toml
coupling_type = "STATIC"
[library]
class = "Dummy"
[application]
class = "MLCouplingApplicationGeneric"
[behavior]
class = "Default"
```

```cpp
std::unique_ptr<MLCoupling<double, double>> configured(
    MLCoupling<double, double>::create_from_config("coupling.toml", input, output));
assert(configured);
```

`create_from_config_string` parses text; free functions `create_mlcoupling_from_config`/`create_mlcoupling_from_config_file` provide corresponding forms. Same-type overloads also accept pre/post buffers.
**Most basic static factories return `MLCoupling<CI,CO>` even on `MLCoupling<CI,CO,LI,LO>`, discarding custom library types.** For mixed types use the file overload taking exactly `const ConfigOverrides&`:

```cpp
using Mixed = MLCoupling<double, double, float, float>;
ConfigOverrides overrides{{"library.model_file", std::string("model.pt")}};
std::unique_ptr<Mixed> configured(Mixed::create_from_config(
    "documentation/configs/aixelerator.toml", raw_input, raw_output, overrides));
```

This uses flow/backend configuration and live valid fields. For text use `create_mlcoupling_from_config_with_library_types<CI,CO,LI,LO>(text, input, output, overrides)`. The static string overload taking overrides still returns two types; adding cast-mode arguments to the file call also selects two types. The four-type path fixes Relaxed casts and Strict matching.

`ConfigOverrides` constructors: no arguments, required `ConfigSectionOverrides` map, required `ConfigDottedOverrides` map, or dotted key/value initializer list. Values: `int64_t`, `double`, `std::string`, `bool`, `std::vector<std::string>`, `void*`. Precedence: dotted entries, section map, TOML.
Opaque pointers must match the factory's exact C++ type/address (e.g. pointer to `std::function`, not a function address). Referred objects and callback captures require appropriate lifetimes.

## C Quickstart

The C and Fortran wrappers are implemented and pass basic callback smoke
tests. They have not yet been used or validated in an actual solver, so end-to-end
solver integration and validation remain outstanding. The tested snippets
below demonstrate wrapper functionality, not complete solver readiness.

This standalone C99 test uses manual Generic creation. `CHECK` executes even in
release builds; an assertion alone must not wrap calls with side effects.

```c
#include "c_api.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(call) do { if ((call) != CMI_SUCCESS) { \
    fprintf(stderr, "%s\n", cmi_get_last_error()); exit(1); } } while (0)

static void infer(cmi_data_t in, cmi_data_t out, void* context) {
    (void)context;
    cmi_tensor_t ti = NULL, to = NULL;
    void *ip = NULL, *op = NULL;
    CHECK(cmi_data_get_tensor(in, 0, &ti));
    CHECK(cmi_data_get_tensor(out, 0, &to));
    CHECK(cmi_tensor_get_data(ti, &ip));
    CHECK(cmi_tensor_get_data(to, &op));
    for (int i = 0; i < 3; ++i) ((double*)op)[i] = 2 * ((double*)ip)[i];
    CHECK(cmi_tensor_destroy(ti));
    CHECK(cmi_tensor_destroy(to));
}
int main(void) {
    double x[3] = {1, 2, 3}, y[3] = {0, 0, 0};
    int dims[1] = {3}, delta = -1;
    cmi_tensor_t ti = NULL, to = NULL;
    cmi_data_t in = NULL, out = NULL;
    cmi_library_t lib = NULL;
    cmi_application_t app = NULL;
    cmi_coupling_t coupling = NULL;
    CHECK(cmi_tensor_create_flat(&ti, x, dims, 1, CMI_DTYPE_DOUBLE,
                                CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL));
    CHECK(cmi_tensor_create_flat(&to, y, dims, 1, CMI_DTYPE_DOUBLE,
                                CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL));
    CHECK(cmi_data_create(&in, CMI_DTYPE_DOUBLE));
    CHECK(cmi_data_create(&out, CMI_DTYPE_DOUBLE));
    CHECK(cmi_data_add_tensor(in, ti));
    CHECK(cmi_data_add_tensor(out, to));
    CHECK(cmi_tensor_destroy(ti));
    CHECK(cmi_tensor_destroy(to));
    CHECK(cmi_library_create_generic(&lib, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE,
                                     infer, NULL));
    CHECK(cmi_application_create_generic(&app, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE,
        CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, in, out, NULL));
    CHECK(cmi_coupling_create(&coupling, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE,
        CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, lib, app, NULL));
    CHECK(cmi_library_destroy(lib)); // Empty wrapper after ownership transfer.
    CHECK(cmi_application_destroy(app));
    lib = NULL; app = NULL;
    CHECK(cmi_coupling_step(coupling, &delta));
    assert(delta == 0 && y[0] == 2 && y[1] == 4 && y[2] == 6);
    CHECK(cmi_coupling_destroy(coupling));
    CHECK(cmi_data_destroy(in));
    CHECK(cmi_data_destroy(out));
    return 0;
}
```

`cmi_data_add_tensor` copies the tensor view; the original tensor handle remains
separately destroyable. `cmi_data_get_tensor` allocates a new view handle, which
must be destroyed. Inference callback data handles are temporary stack wrappers:
never destroy or retain those input/output handles. Modify output **storage**;
replacing the temporary container does not propagate back to the application.

Successful `cmi_coupling_create` releases the component objects from their C
wrappers. Destroy the emptied wrappers, but never use them for operations again;
the coordinator now owns the actual objects. The same rule applies to normalization
transferred into `cmi_application_create_generic`. Destruction takes handles by
value and does not null caller variables. Data handles are copied, not consumed.
Callback context memory is borrowed and must outlive every callback.

The current C four-type dispatcher supports all-float, all-double, float solver/
double library, and double solver/float library. But the direct Generic application
creator supplies **no conversion callbacks**, so its mixed-type construction does
not yield a working mixed-type step. Use same-type Generic as above or a specialized
application. Supply matching declared types; component release uses raw casts.

`cmi_coupling_create_from_config` accepts a path and four type tags but constructs
with **empty input/output containers**. It has no buffer attachment argument or
coupling accessor to populate them afterward. FlowExtrapolator rejects these empty
fields; typical inference providers require actual tensors. It is therefore not
a fully working solver setup route today. The Fortran config interface inherits
this limitation. Use manual creation for live solver buffers.

## Fortran Quickstart

The current module is named `cmi`; it has no `private` declaration, so the
`c_cmi_*` low-level interfaces are accessible. Its derived handle component is
`%ptr`, **not `%handle`**. The following program creates a Generic coupling with a
`bind(C)` callback. Module callback placement keeps the procedure address stable.

```fortran
module demo_callbacks
  use cmi
  implicit none
contains
  subroutine check(st)
    integer(c_int), intent(in) :: st
    if (st /= CMI_SUCCESS) error stop "CMI operation failed"
  end subroutine
  subroutine infer(input, output, context) bind(C)
    type(c_ptr), value :: input, output, context
    type(c_ptr) :: ti, to, ip, op
    real(c_double), pointer :: x(:), y(:)
    call check(c_cmi_data_get_tensor(input, 0_c_int, ti))
    call check(c_cmi_data_get_tensor(output, 0_c_int, to))
    call check(c_cmi_tensor_get_data(ti, ip))
    call check(c_cmi_tensor_get_data(to, op))
    call c_f_pointer(ip, x, [3])
    call c_f_pointer(op, y, [3])
    y = 2.0_c_double * x
    call check(c_cmi_tensor_destroy(ti))
    call check(c_cmi_tensor_destroy(to))
  end subroutine
end module

program generic_fortran
  use demo_callbacks
  implicit none
  real(c_double), target :: x(3) = [1.0_c_double, 2.0_c_double, 3.0_c_double]
  real(c_double), target :: y(3) = 0.0_c_double
  type(cmi_tensor) :: ti, to
  type(cmi_data) :: input, output
  type(cmi_library) :: lib
  type(cmi_application) :: app
  type(cmi_coupling) :: coupling
  integer :: ierr
  integer(c_int) :: delta
  call cmi_tensor_wrap(ti, x, ierr)
  if (ierr /= 0) error stop "input wrap failed"
  call cmi_tensor_wrap(to, y, ierr)
  if (ierr /= 0) error stop "output wrap failed"
  call check(c_cmi_data_create(input%ptr, CMI_DTYPE_DOUBLE))
  call check(c_cmi_data_create(output%ptr, CMI_DTYPE_DOUBLE))
  call check(c_cmi_data_add_tensor(input%ptr, ti%ptr))
  call check(c_cmi_data_add_tensor(output%ptr, to%ptr))
  call check(c_cmi_tensor_destroy(ti%ptr))
  call check(c_cmi_tensor_destroy(to%ptr))
  ti%ptr = c_null_ptr; to%ptr = c_null_ptr
  call check(c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_DOUBLE, &
      CMI_DTYPE_DOUBLE, c_funloc(infer), c_null_ptr))
  call check(c_cmi_application_create_generic(app%ptr, CMI_DTYPE_DOUBLE, &
      CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, &
      input%ptr, output%ptr, c_null_ptr))
  call check(c_cmi_coupling_create(coupling%ptr, CMI_DTYPE_DOUBLE, &
      CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, CMI_DTYPE_DOUBLE, &
      lib%ptr, app%ptr, c_null_ptr))
  call check(c_cmi_library_destroy(lib%ptr))
  call check(c_cmi_application_destroy(app%ptr))
  lib%ptr = c_null_ptr; app%ptr = c_null_ptr
  call check(c_cmi_coupling_step(coupling%ptr, delta))
  if (delta /= 0 .or. any(y /= 2.0_c_double*x)) error stop "wrong result"
  call check(c_cmi_coupling_destroy(coupling%ptr))
  coupling%ptr = c_null_ptr
  call check(c_cmi_data_destroy(input%ptr))
  call check(c_cmi_data_destroy(output%ptr))
  input%ptr = c_null_ptr; output%ptr = c_null_ptr
end program
```

`cmi_tensor_wrap` supports rank 1/2/3 `real(c_float)` and `real(c_double)` arrays,
borrows `target` storage, and uses Fortran-contiguous layout. Pass whole contiguous
arrays, not strided sections or expressions that can create temporaries. Avoid
ordinary assignment of live handles: it copies the pointer, not ownership, and
there are no automatic finalizers. Destroy once and invalidate `%ptr` explicitly.
Raw strings for low-level interfaces require a trailing `c_null_char`.

## Build And Verify

C and Fortran clients link the C API shared library; Fortran also needs its wrapper library and a module compiled with the same compiler. Adapt library/runtime paths to the selected build and backend dependencies. From the repository root:

```sh
gcc -std=c99 -Iinclude generic.c -Lbuild -lcpp_ml_interface_library -Wl,-rpath,"$PWD/build" -o generic_c
gfortran -Ibuild/fortran generic.f90 -Lbuild -lcpp_ml_interface_fortran -lcpp_ml_interface_library -Wl,-rpath,"$PWD/build" -o generic_f
./generic_c
./generic_f
```

If the existing `.mod` is incompatible, compile `include/fortran/cmi_fortran.f90`
with the client compiler into a separate build directory and link that object
instead of the wrapper library. A Score-P-instrumented library additionally
requires the configured Score-P compiler/link wrapper; plain compiler linking
can fail with unresolved `scorep_subsystems` symbols. Generic smoke tests
validate orchestration and memory visibility, not external inference. Real tests
also need enabled providers, model-compatible shapes/dtypes, MPI/MPMD launch
setup where applicable, a live SmartSim database where applicable, and checks of
actual predicted values. For flexible coupling, `ordered()` and `keyed()` stage
inputs, infer, then retrieve/postprocess outputs; these calls do not run behavior
scheduling. Base fallback merges staged data (default strategy `Auto`) and calls
static inference, so do not assume native multi-key or asynchronous execution.
`train_step()` prepares input and trains against library output; it does not
automatically construct labels. Concrete inference providers do not override the
base unsupported training method. Generic can supply training; `track()` and
history/current queries record returned metrics.
