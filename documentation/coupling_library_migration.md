# Coupling Library Migration

## Terminology

- **Coupling library**: an interchangeable ML runtime integration such as
  AIXelerator, SmartSim, or PhyDLL. It executes inference or training and owns
  its transport/runtime details.
- **Application**: converts public coupling fields into coupling-library
  tensors and converts library results back into public coupling fields.
- **Behavior**: decides when a coupling step sends data or performs inference.
- **MLCoupling**: orchestrates an application, a coupling library, and a
  behavior.

`MLCouplingProvider` has been renamed to `MLCouplingLibrary`. Configuration
uses `[library]` instead of `[provider]`.

## Type Boundaries

The previous API used one input/output type pair for every boundary:

```cpp
MLCoupling<In, Out>
```

The new API distinguishes the public coupling boundary from the coupling
library boundary:

```cpp
MLCoupling<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>
```

`LibraryInput` defaults to `CouplingInput`, and `LibraryOutput` defaults to
`CouplingOutput`. Existing same-type applications can therefore continue to
use two template arguments.

The application interface follows the same order:

```cpp
MLCouplingApplication<CouplingInput, CouplingOutput,
                      LibraryInput, LibraryOutput>
```

The coupling-library interface only needs its own tensor boundary:

```cpp
MLCouplingLibrary<LibraryInput, LibraryOutput>
```

## Application Migration

Replace the ambiguous old buffers:

```text
input_data
input_data_after_preprocessing
output_data_before_postprocessing
output_data
```

with explicit boundary buffers:

```text
coupling_input
library_input
library_output
coupling_output
```

Replace overrides of `preprocess()` and `postprocess()` with:

```cpp
MLCouplingData<LibraryInput>
preprocess_coupling_input(MLCouplingData<CouplingInput>);

MLCouplingData<CouplingOutput>
postprocess_library_output(MLCouplingData<LibraryOutput>);
```

Use `prepare_library_input()` and `finalize_coupling_output()` in custom
application orchestration.

## FlowExtrapolator Example

MAIA stores CFD fields as `MFloat` (`double`), while the TorchScript model and
all coupling-library protocols use `float`:

```cpp
MLCoupling<float, double, float, float>
```

The application keeps cube assembly and provider transport in float32. It
promotes the float32 library result before denormalization, scatter, and
weight normalization, then returns double reconstructed fields to MAIA. This
matches the precision boundary used by the legacy AIX CMI implementation.
