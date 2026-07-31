# Merge Note: `scorep_instrument_target` conflict resolution

**Context:** Merge of `origin/redesign/coupling-interface` (smartsim WIP) into
`debug/current-prepost` (MMCP artifact), executed 2026-07-31.

**Affected files:** `dl_clients/CMakeLists.txt`, top-level `CMakeLists.txt`

---

## The conflict

**Our branch** (`debug/current-prepost`) removed all `scorep_instrument_target()`
calls from both CMakeLists files.

**Reason:** On CLAIX23 we build with the `scorep-mpicxx` compiler wrapper, which
already injects `scorep_init.o` (and defines the `scorep_subsystems` symbol) at link
time automatically. Calling `scorep_instrument_target()` additionally causes a
**duplicate-symbol linker error** (`scorep_subsystems` defined twice — once by the
wrapper's injected stub, once by the macro's own stub).

**SmartSim branch** (`redesign/coupling-interface`) still had `scorep_instrument_target()`
in `dl_clients/CMakeLists.txt`, and added `target_include_directories(...Scorep::Plugin)`
and `target_link_libraries(...Scorep::Plugin)` on top of it.

---

## Resolution applied in merge

- **Remove** `scorep_instrument_target()` (keep our side — required for CLAIX23 wrapper build).
- **Add** smartsim's `target_include_directories` + `target_link_libraries(Scorep::Plugin)`
  lines — these are still needed so the DL client can find `<scorep/SCOREP_User.h>`
  and link the plugin correctly, even without the CMake macro.

The merged `dl_clients/CMakeLists.txt` Score-P block becomes:

```cmake
if(SCOREP_FOUND)
    if(NOT CPPML_SCOREP_MPP)
        set(CPPML_SCOREP_MPP "mpi")
    endif()
    # scorep_instrument_target is NOT called here: the scorep-mpicxx wrapper already
    # injects scorep_init.o at link time. Calling scorep_instrument_target additionally
    # causes a duplicate-symbol linker error (scorep_subsystems defined twice).
    target_include_directories(phydll_dl_runtime PRIVATE ${SCOREP_ROOT_DIR}/include
                                                         ${SCOREP_ROOT_DIR}/include/scorep)
    target_include_directories(phydll_dl_client  PRIVATE ${SCOREP_ROOT_DIR}/include
                                                         ${SCOREP_ROOT_DIR}/include/scorep)
    target_link_libraries(phydll_dl_runtime PRIVATE Scorep::Plugin)
    target_link_libraries(phydll_dl_client  PRIVATE Scorep::Plugin)
    target_link_options(phydll_dl_client PRIVATE -no-pie)
endif()
```

The same logic applies to the top-level `CMakeLists.txt`: `scorep_instrument_target()`
is removed from the main library target, while `USE_SCOREP` compile definitions remain.

---

## Impact on smartsim if it ever takes this merged branch

The smartsim checkout (`/hpcwork/ro092286/smartsim/CPP-ML-Interface`) may be using
a different Score-P invocation path (direct `mpicxx` + manual `-lscorep` flags,
without the `scorep-mpicxx` wrapper). If `scorep_instrument_target()` is required
in that environment for compiler-level instrumentation, it will need to be re-added
after taking the merge.

**Action required before switching `../smartsim` to this merged branch:**
1. Verify whether `scorep_instrument_target()` is still needed in the smartsim build environment.
2. If needed: add a CMake option (e.g. `CPPML_SCOREP_USE_WRAPPER=ON/OFF`) to toggle
   between the wrapper approach (no `scorep_instrument_target`) and the direct approach
   (with `scorep_instrument_target`), rather than hard-coding either.
3. Confirm the `Scorep::Plugin` link lines do not conflict with how smartsim resolves
   Score-P in its own AIX benchmarking builds.
