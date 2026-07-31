# Local Score-P and PAPI Setup

This guide explains how to use the local Score-P/PAPI builds in this repo instead of loading the system modules.

## What gets installed

The local bootstrap installs everything into `CPP-ML-Interface/tmp/opencode/` by default:

- PAPI 7.2.0
- Score-P 8.4
- OTF2 3.1.1
- OPARI2 2.0.9
- CubeW 4.8.2
- CubeLib 4.8.2

## One-time bootstrap

From `CPP-ML-Interface/`:

```bash
git submodule update --init --recursive
./install_scorep.sh cuda-12
```

If you only want the profiling stack and not the full SmartSim install, run:

```bash
./build_scorep.sh
```

## Using the local stack

Do not load the `Score-P` or `PAPI` modules.

Instead, rely on the local env helper:

```bash
source ./set_env_claix23_cuda12.4.sh
```

If `USE_SCOREP=1` is set before sourcing it, the script adds the local Score-P and PAPI binaries and libraries to your environment.

Example:

```bash
USE_SCOREP=1 source ./set_env_claix23_cuda12.4.sh
```

That sets:

- `PATH`
- `LD_LIBRARY_PATH`
- `LIBRARY_PATH`
- `CPATH`
- `PAPI_ROOT`
- `PAPI_INC`
- `PAPI_LIB`
- `SCOREP_ROOT`
- `SCOREP_ROOT_DIR`

## Building with Score-P

Use the normal build script with `USE_SCOREP=1`:

```bash
USE_SCOREP=1 ./build.sh
```

This switches the compiler wrappers to:

- `CC=scorep-mpicc`
- `CXX=scorep-mpicxx`

and enables `-DWITH_SCOREP=ON` in CMake.

## Python bindings

The local Score-P install also satisfies the external Python binding package requirement.

After sourcing the local env, reinstall the bindings in the project venv so they pick up the local `scorep-config`:

```bash
source ./env_scorep.sh
./extern/python/smartsim_cuda-12/bin/python -m pip install --force-reinstall --no-cache-dir git+https://github.com/score-p/scorep_binding_python.git
```

Then run Python programs through Score-P with:

```bash
./extern/python/smartsim_cuda-12/bin/python -m scorep your_script.py
```

## Running jobs with PAPI counters

If you want network or hardware counters through Score-P, request the PAPI hardware counter resource in Slurm:

```bash
#SBATCH --hwctr=papi
```

This makes the PAPI measurement component active. The mini-app job script has an **auto-detection block** that queries `papi_native_avail` to find the correct active InfiniBand/network device prefixes (e.g. `mlx5_0_1_ext` and `ib0`), setting the metrics dynamically.

Alternatively, you can manually set the Score-P PAPI metric variables **before launching the application**:

```bash
export SCOREP_METRIC_PAPI_SEP=,
export SCOREP_METRIC_PAPI='net:::ib0:rx:byte,net:::ib0:tx:byte,infiniband:::mlx5_0_1_ext:port_rcv_data,infiniband:::mlx5_0_1_ext:port_xmit_data'
```

If you are not requesting the `--hwctr=papi` resource, keep the PAPI metric setting empty:

```bash
export SCOREP_METRIC_PAPI=""
```

## Python manual user regions

To profile user regions in Python (e.g., `py_recv`, `py_inference`) when auto-instrumentation is disabled (`--noinstrumenter`), you must:
1. Export `ENABLE_SCOREP_USER=1` in the environment.
2. Call `scorep.instrumenter.enable()` at Python startup:
   ```python
   import scorep.user
   import scorep.instrumenter
   scorep.instrumenter.enable()
   ```
3. Call `scorep.user.force_finalize()` before Python exits (especially if using `mpi4py` or other libraries registering atexit handlers):
   ```python
   scorep.user.force_finalize()
   ```

## CUDA instrumentation status

The current local custom Score-P build has CUPTI/CUDA instrumentation adapter disabled (`--disable-cuda`). GPU memory timing and usage should be profiled via NVML (`nvmlDeviceGetMemoryInfo`) or by spawning an `nvidia-smi dmon` sidecar process.

If one run cannot fit all requested counters, split them across multiple runs and compare the resulting traces/profiles.

## Quick checks

```bash
source ./env_scorep.sh
which scorep-config
which scorep-mpicxx
scorep-info config-summary | grep -A 12 'Score-P (backend):'
```

The backend summary should report shared libraries enabled.
