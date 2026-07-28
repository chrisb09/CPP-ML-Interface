# Preliminary Merge & Unification Plan: `redesign/coupling-interface` ↔ `debug/current-prepost`

> **IMPORTANT DISCLAIMER**  
> **Date:** July 28, 2026  
> **Status:** Preliminary / Subject to Change  
>  
> This document outlines a preliminary merge and unification strategy between the `redesign/coupling-interface` branch (located in `/hpcwork/ro092286/smartsim/CPP-ML-Interface`) and the `debug/current-prepost` branch (located in `/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/CPP-ML-Interface`).  
>  
> **Notice:** The `smartsim` checkout is currently in active development and is **not yet ready for a merge**. This plan reflects the state of both codebases as of July 28, 2026. It may require **significant alterations, updates, and re-review** at a later date to accommodate further changes, new developments, and to rectify potential errors or unforeseen code conflicts.

---

## 1. Branch Context & Common Ancestor

- **Common Ancestor Commit**: `c500c7c` (July 11, 2026) — *"fix: align Python PhyDLL metadata ordering"*
- **Target Unified Branch**: `redesign/coupling-interface`

---

## 2. High-Level Summary of Divergence

### A. `redesign/coupling-interface` (SmartSim Sandbox)
- **Primary Goal**: Flexible API testing, Score-P performance profiling, DL client hardening.
- **Committed Ahead (`523f95b`)**:
  - PhyDLL DL client hardening (`dl_client.cpp`, `phydll_dl_runtime.*`, `dl_clients/CMakeLists.txt`, `phydll_dl_client.py`).
- **Uncommitted WIP (~320 lines)**:
  - New `include/scorep_profiling_state.hpp` for global runtime toggle (`detailed_regions_enabled`).
  - Score-P region wrappers gated behind `profile_details` in `ml_coupling_application.hpp`, `ml_coupling.hpp`, `ml_coupling_provider.hpp`, `ml_coupling_provider_phydll.hpp`, and `ml_coupling_provider_smartsim.hpp`.
  - Added build helper `build_aix.sh` and PAPI smoke test script `smoke_papi_720.sbatch`.

### B. `debug/current-prepost` (Hybrid CFD/ML Artifact)
- **Primary Goal**: Numerical correctness, bitwise provider determinism, PyTorch 2.6.0 alignment, double-precision solver output preservation.
- **Committed Ahead (8 commits through `0496407`)**:
  - **Renamed** `include/provider/ml_coupling_provider.hpp` → `include/library/ml_coupling_library.hpp`.
  - **Templated 4-type interface**: `MLCoupling<In, Out, PrepIn, PostOut>` allowing model inputs/outputs in `float32` while returning `MFloat` (double) fields to the MAIA solver.
  - Configuration section rename `[provider]` → `[library]` with backward compatibility.
  - Portable `python_runner.sh` and PyTorch 2.6.0 installation scripts.
  - Extensive debug dump exports and bit-perfect verification tooling.

---

## 3. Conflict Matrix & Resolution Strategies

| File Location | Conflict Type | Resolution Strategy |
|---|---|---|
| `include/application/ml_coupling_application.hpp` | 4-type parameter template refactor (artifact) vs Score-P region wrappers (redesign) | Retain 4-type template signature. Re-wrap Score-P profiling regions around the updated method bodies. |
| `include/ml_coupling.hpp` | 4-type template refactor vs `set_scorep_detailed_regions_enabled()` | Retain 4-type template signature. Add `set_scorep_detailed_regions_enabled()` setter into the class body. |
| `include/provider/ml_coupling_provider.hpp` → `include/library/ml_coupling_library.hpp` | **File Rename** (artifact) vs Score-P regions added to old file (redesign) | Accept file rename. Manually re-apply redesign's flex-merge Score-P regions onto `include/library/ml_coupling_library.hpp`. |
| `include/provider/ml_coupling_provider_phydll.hpp` | Base class rename (artifact) vs Score-P metric gating (redesign) | Re-apply Score-P region gating onto the updated PhyDLL provider implementation. |
| `include/provider/ml_coupling_provider_smartsim.hpp` | Both modified | Re-apply Score-P region gating (`smartsim_chunk_plan`, `put_tensor`, `run_model`, `unpack_tensor`) onto the updated provider. |
| `dl_clients/phydll_dl_client.py` | PyTorch 2.6 alignment (artifact) vs hardening (redesign) | Combine changes additively. Keep PyTorch 2.6 alignment and redesign hardening logic. |
| `dl_clients/dl_client.cpp`, `phydll_dl_runtime.*` | Modified only on redesign | Clean import (no conflict). |
| `include/config.hpp`, `src/c_api.cpp`, `test/*` | Modified only on artifact | Clean import (no conflict). |

---

## 4. Multi-Phase Unification Workflow

### Phase 0: Cleanup & Pre-Commit on `redesign/coupling-interface`
1. Remove junk dump files (e.g., `core.n23m0123...`). Keep legitimate scripts (`build_aix.sh`, `smoke_papi_720.sbatch`).
2. When ready, commit all uncommitted WIP on `redesign/coupling-interface` in a single focused commit:
   `"feat: add gated Score-P profiling regions and DL client hardening"`

### Phase 1: Branch Merge
1. Fetch latest refs and check out `redesign/coupling-interface`:
   ```bash
   git fetch origin debug/current-prepost
   git merge FETCH_HEAD
   ```
2. Manually resolve file rename and template parameter conflicts according to Section 3.
3. Re-apply Score-P regions onto `include/library/ml_coupling_library.hpp`.

### Phase 2: Submodule Alignment
- **`extern/AIxeleratorService`**: Reconcile commit references between redesign and artifact (ensure Score-P target guard is present).
- **`extern/phydll`**: Reconcile commit references (ensure stack overflow fix is preserved).
- **`extern/SmartRedis`**: Confirm dependency alignment.

### Phase 3: Post-Merge Verification Loop
1. **Build Verification**: Compile CMI using `build.sh` and `build_phydll.sh`.
2. **Matrix Test Suite**: Execute `test_matrix.py` in `smartsim/module_test` to verify `STATIC`, `ORDERED`, `KEYED`, `ORDERED_MULTI`, `KEYED_MULTI` API modes across all providers.
3. **Profiling Verification**: Verify Score-P regions trigger when `detailed_regions_enabled = true`.
4. **Numerical Verification**: Run `verification_report/verify_bit_perfect.sh` in the artifact repo to guarantee bitwise determinism and float32/MFloat conversion safety.

---

*Document generated on July 28, 2026. Keep in repository documentation for future reference.*
