# CPP-ML-Interface Reference Manual {#mainpage}

**CPP-ML-Interface** is a C++17 library designed to couple High-Performance Computing (HPC) simulations with Machine Learning (ML) inference frameworks (PyTorch, TorchScript, ONNX, TensorRT).

It provides native interfaces in **C++17**, **C99**, and **Fortran 2003+** with support for zero-copy array sharing and pluggable inference backends.

---

## Architecture at a Glance

The library decouples the coupling lifecycle into distinct components:

- **Orchestrator** (@ref MLCoupling): Manages the step lifecycle and coordinates components.
- **Data & Tensors** (@ref MLCouplingData, @ref MLCouplingTensor): Multi-dimensional tensor views with configurable layouts and zero-copy external buffer borrowing.
- **Provider / Library** (@ref MLCouplingLibrary): Communicates with the ML framework (PhyDLL, SmartSim, AIxelerator, Dummy, or Generic callbacks).
- **Application** (@ref MLCouplingApplication): Handles simulation data preparation, preprocessing, and postprocessing.
- **Behavior** (@ref MLCouplingBehavior): Dictates when inference is executed and how many simulation timesteps to advance.
- **Normalization** (@ref MLCouplingNormalization): In-place feature scaling and denormalization.

---

## Language APIs

### C++17 API

The primary object-oriented interface (see @ref cpp_api):

- **Core Classes**: @ref MLCoupling, @ref MLCouplingData, @ref MLCouplingTensor, @ref MLCouplingMatrix, @ref MLCouplingVector, @ref MLCouplingScalar
- **Base Components**: @ref MLCouplingLibrary, @ref MLCouplingApplication, @ref MLCouplingBehavior, @ref MLCouplingNormalization
- **Generic Adapters**: @ref MLCouplingLibraryGeneric, @ref MLCouplingApplicationGeneric, @ref MLCouplingBehaviorGeneric, @ref MLCouplingNormalizationGeneric

### C API

Procedural C99 interface using opaque handles and thread-local error reporting (see @ref c_api):

- **Errors**: @ref cmi_get_last_error(), @ref cmi_status_string(), @ref cmi_clear_last_error()
- **Tensors & Data**: @ref cmi_tensor_create_flat(), @ref cmi_tensor_destroy(), @ref cmi_data_create(), @ref cmi_data_add_tensor()
- **Components**: @ref cmi_behavior_create_default(), @ref cmi_behavior_create_periodic(), @ref cmi_behavior_create_generic(), @ref cmi_normalization_create_minmax(), @ref cmi_library_create_generic()
- **Pipeline Execution**: @ref cmi_coupling_create(), @ref cmi_coupling_create_from_config(), @ref cmi_coupling_step()
- **Introspection**: @ref cmi_get_class_count(), @ref cmi_get_class_name(), @ref cmi_is_class_registered()

### Fortran 2003+ API

Module `cmi` provides `iso_c_binding` interfaces and multi-dimensional array wrappers (see @ref fortran_api):

- **Derived Types**: `type(cmi_tensor)`, `type(cmi_data)`, `type(cmi_behavior)`, `type(cmi_normalization)`, `type(cmi_library)`, `type(cmi_application)`, `type(cmi_coupling)`
- **Zero-Copy Array Wrapping**: Overloaded `cmi_tensor_wrap()` supporting 1D, 2D, and 3D arrays of single and double precision floats using native Fortran column-major storage (`CMI_LAYOUT_FORTRAN_CONTIGUOUS`).
- **Error Handling**: `cmi_get_error(msg)` to inspect thread-local errors.

---

## Generic Callback Implementations

Instead of creating derived classes for simple use cases or non-C++ languages, you can instantiate generic callback adapters:

- **@ref MLCouplingBehaviorGeneric**: Accepts lambdas or function pointers for `should_infer`, `time_step_delta`, and `should_send_data`.
- **@ref MLCouplingNormalizationGeneric**: Accepts callbacks for `normalize_input` and `denormalize_output`.
- **@ref MLCouplingLibraryGeneric**: Accepts an `inference_fn` callback for custom or mock model execution.
- **@ref MLCouplingApplicationGeneric**: Accepts callbacks for `preprocess_fn`, `postprocess_fn`, and `ml_step_fn`.

---

## Inference Providers

- **PhyDLL** (@ref MLCouplingLibraryPhydll): In-situ and in-transit MPMD inference over MPI point-to-point channels using LibTorch / PyTorch. Supports `packed` and `uniform_chunks` wire layouts.
- **SmartSim** (@ref MLCouplingLibrarySmartsim): Distributed Redis database inference using the SmartRedis client, with automated hash slot key balancing across cluster nodes.
- **AIxelerator** (@ref MLCouplingLibraryAixelerator): Accelerated GPU/CPU inference engine with batching and collective MPI communication.
- **Dummy** (@ref MLCouplingLibraryDummy): Mock provider for interface sanity checks without GPU or ML dependencies.

---

## Known Quirks & Current Limitations

1. **Inference Only**: All built-in providers (`PhyDLL`, `SmartSim`, `AIxelerator`, `Dummy`) implement inference only. Training methods (`static_train()`, `cmi_coupling_train_step()`) throw a runtime error unless backed by a user-supplied callback via @ref MLCouplingLibraryGeneric.
2. **AIxelerator Single-Tensor Limitation**: @ref MLCouplingLibraryAixelerator currently binds only the first tensor (`[0]`) in an @ref MLCouplingData container. Any additional tensors are ignored.
3. **Contiguous Buffers Required**: All underlying inference providers require flat contiguous buffers (@ref CMI_LAYOUT_CONTIGUOUS or @ref CMI_LAYOUT_FORTRAN_CONTIGUOUS). Nested pointer trees (@ref CMI_LAYOUT_NESTED) must be converted using @ref MLCouplingTensor::flatten() prior to provider calls.
4. **C and Fortran Type Dispatch**: The C and Fortran interfaces support single-precision (`float` / `CMI_DTYPE_FLOAT`) and double-precision (`double` / `CMI_DTYPE_DOUBLE`) scalars.
5. **Turbulence Closure Application**: @ref MLCouplingApplicationTurbulenceClosure is currently a skeletal placeholder; custom domain transformations should be implemented using @ref MLCouplingApplicationGeneric.
