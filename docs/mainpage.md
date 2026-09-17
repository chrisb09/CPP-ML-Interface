# CPP-ML-Interface Documentation {#mainpage}

Welcome to the documentation for **CPP-ML-Interface**, a high-performance C++17 library designed to couple High-Performance Computing (HPC) simulations with Machine Learning (ML) inference frameworks.

---

## Key Features

- **Multi-Language Support**:
  - **C++ API**: Modern C++17 interface with static and flexible execution tiers, compile-time type safety, and zero-copy tensor sharing.
  - **C API**: Full C99-compatible procedural API (`cmi_*`) using opaque handles, thread-local error reporting, and dynamic type dispatch.
  - **Fortran API**: Fortran 2003/2008 `iso_c_binding` module (`cmi`) featuring zero-copy wrapping for multi-dimensional arrays with native column-major memory layout.
- **Pluggable Backends**:
  - **PhyDLL**: In-situ / in-transit MPMD inference via PyTorch and TorchScript.
  - **SmartSim / SmartRedis**: In-memory Redis-based orchestration and distributed inference.
  - **AIxelerator**: Accelerated inference engine supporting TensorRT, ONNX Runtime, and TensorFlow.
  - **Dummy Provider**: Mock provider for pipeline testing and verification without ML dependencies.
  - **Generic Callbacks**: Custom C/C++/Fortran functions plugged directly into the pipeline without subclassing.
- **Architectural Separation**:
  - **Library**: Manages model lifecycle and inference calls.
  - **Application**: Controls data staging, pre-processing, and post-processing.
  - **Behavior**: Controls execution frequency, periodic intervals, and simulation step advancement.
  - **Normalization**: Scales inputs/outputs (e.g. MinMax, z-score, or custom transformations).

---

## Architectural Overview

```
+-------------------------------------------------------------+
|                     Simulation Application                  |
|                 (C++ / C99 / Fortran 2003+)                 |
+-------------------------------------------------------------+
                              |
                              v
             +----------------------------------+
             |        MLCoupling (Pipeline)      |
             +----------------------------------+
               /              |               \
              v               v                v
     +----------------+ +----------------+ +----------------+
     |   Application  | |    Behavior    | | Normalization  |
     | Pre/Post-proc  | | Step Decision  | | MinMax/Custom  |
     +----------------+ +----------------+ +----------------+
              \               |               /
               v              v              v
             +----------------------------------+
             |        MLCouplingLibrary         |
             | (PhyDLL / SmartSim / AIx / etc.) |
             +----------------------------------+
```

---

## API References by Language

### 1. [C++ API](annotated.html)
The primary C++ interface consists of:
- \ref MLCoupling - Main orchestration class.
- \ref MLCouplingData and \ref MLCouplingTensor - Tensor management and multi-dimensional views.
- Base & Generic classes:
  - \ref MLCouplingLibrary and \ref MLCouplingLibraryGeneric
  - \ref MLCouplingApplication and \ref MLCouplingApplicationGeneric
  - \ref MLCouplingBehavior and \ref MLCouplingBehaviorGeneric
  - \ref MLCouplingNormalization and \ref MLCouplingNormalizationGeneric

### 2. [C API Reference](group__c__api.html)
A complete C API prefixed with `cmi_`:
- **Lifecycle & Error Handling**: `cmi_get_last_error()`, `cmi_clear_last_error()`
- **Tensors & Data**: `cmi_tensor_create_borrowed()`, `cmi_tensor_create_allocated()`, `cmi_data_create()`
- **Generic Modules**: `cmi_behavior_create_generic()`, `cmi_library_create_generic()`, `cmi_normalization_create_generic()`, `cmi_application_create_generic()`
- **Named / Factory Modules**: `cmi_behavior_create()`, `cmi_library_create()`, `cmi_normalization_create()`, `cmi_application_create()`
- **Coupling & Execution**: `cmi_coupling_create()`, `cmi_coupling_step()`
- **Introspection**: `cmi_get_registered_class_count()`, `cmi_get_registered_class_name()`

### 3. [Fortran 2003+ Module](namespacecmi.html)
A native Fortran module `cmi` wrapping the C API with automatic column-major layout:
- Zero-copy tensor creation from Fortran 1D, 2D, and 3D arrays (`cmi_tensor_create_1d`, `cmi_tensor_create_2d`, etc.).
- Convenient interfaces for behavior, normalization, library, application, and orchestrator.

---

## User Guides and Documentation

- [API Usage Guide](api_guide.html): In-depth guide for static and flexible execution tiers.
- [Migration Guide](migration_guide.html): Migration instructions between coupling versions.
- [PhyDLL Transport Layouts](phydll_transport_layouts.html): Details on tensor layouts and memory alignment for PhyDLL.
