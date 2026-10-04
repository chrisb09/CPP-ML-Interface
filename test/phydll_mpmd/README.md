# Solver Readiness Protocol

The Phydll provider accepts a trailing constructor argument
`bool solver_readiness_wait = false`. The auto-generated registry exposes the
same option in TOML:

```toml
[library]
class = "Phydll"
model_file = "/absolute/path/model.pt"
backend = "TORCH"
device = "CPU"
solver_readiness_wait = true
```

Omitting the option or setting it to `false` retains the normal blocking payload
receive. This is a runtime protocol, not persisted metadata: rebuild the solver
registry/provider and C++ DL client together, or use the updated Python DL
client. All participants duplicate `MPI_COMM_WORLD` immediately after PhyDLL
initialization, before field definition, including when readiness is disabled.
Older binaries cannot participate in that collective and must not be mixed.

The one-time 88-byte metadata header is version 4, with the boolean at byte 76.
Each DL rank requires the same policy from all of its assigned solver sources;
mixed policies are rejected before inference. Independent DL groups negotiate
from their own assigned sources.

For each enabled frame, the solver posts its own control receives before sending
input, then posts the PhyDLL output receive. DL registers every output field and
asynchronously notifies every assigned source with a uint64 frame sequence on the
dedicated communicator before sending payload. The solver tests only its control
requests, sleeps 100 us while pending, validates sequence and message count, then
calls `phydll_wait_irecv()` before the existing unpack. Frame numbering starts at
zero, including the initialization frame. Tokens do not imply payload or label
completion. DL token storage remains stable and send requests are completed
before returning, starting another frame, or normal shutdown.

`FORWARD_CPU_DIAGNOSTIC` stays opt-in. Its existing `solver_recv` timer covers
receive posting, readiness polling/sleep, and payload completion. The detailed
Score-P `phydll_readiness_wait` region is nested inside `phydll_recv`; do not add
its inclusive time to its parent's inclusive time.

## Isolated Tests

Run commands from the project root. The following reuses only isolated artifacts
under `tmp/opencode/phydll-readiness`, not production builds or model data:

```bash
source set_env_claix23_cuda12.4.sh
source /hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/activate
python scripts/generate_registry.py tmp/opencode/phydll-readiness/generated_registry.hpp \
  MLCouplingLibrary,MLCouplingApplication,MLCouplingBehavior,MLCouplingNormalization \
  include/library/ml_coupling_library.hpp include/application/ml_coupling_application.hpp \
  include/behavior/ml_coupling_behavior.hpp include/normalization/ml_coupling_normalization.hpp \
  include/provider/ml_coupling_provider_phydll.hpp \
  include/application/ml_coupling_application_generic.hpp include/behavior/ml_coupling_behavior_default.hpp
cmake -S dl_clients -B tmp/opencode/phydll-readiness/config-client \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DPHYDLL_BUILD_DIR="$PWD/tmp/opencode/phydll-readiness/phydll" \
  -DLIBTORCH_DIR=/hpcwork/ro092286/smartsim/python/smartsim_cpu/lib64/python3.9/site-packages/torch \
  -DPHYDLL_TEST_CONFIG_DIR="$PWD/tmp/opencode/phydll-readiness" \
  -DPHYDLL_TEST_TOML_INCLUDE_DIR="$PWD/tmp/opencode/cmi-full-build/_deps/tomlplusplus-src/include" \
  -DPHYDLL_TEST_FAULT_CLIENT=ON
cmake --build tmp/opencode/phydll-readiness/config-client -j 4
python3 test/phydll_mpmd/test_metadata_decode.py
```

`PHYDLL_TEST_CONFIG_DIR` contains an auto-generated registry for the Phydll,
Generic application, and Default behavior, so the harness actually constructs
and runs a coupling from TOML, rather than merely checking strings. Without
that option it exercises the direct constructor; `PHYDLL_TEST_REGISTRY_DIR`
instead tests the generated typed factory.

Submit the bounded CPU matrix with an existing writable results directory:

```bash
export PHYDLL_TEST_PROJECT_ROOT="$PWD"
export PHYDLL_TEST_BUILD_DIR="$PWD/tmp/opencode/phydll-readiness/config-client"
export PHYDLL_TEST_PREFIX="$PWD/tmp/opencode/phydll-readiness/phydll"
export PHYDLL_TEST_PYTHON_ENV=/hpcwork/ro092286/smartsim/python/smartsim_cpu/bin/activate
export PHYDLL_TEST_FAULTS=1
sbatch --chdir="$PWD/tmp/opencode/phydll-readiness/final-results" \
  --output=smoke-%j.log test/phydll_mpmd/readiness_smoke.sbatch
```

The default matrix covers 2:1, 3:1, 4:2, and 3:2 solver:DL topologies, both
clients, packed/uniform_chunks, both asymmetric models, and omitted/false/true
policies. Eight numerically checked frames change inputs each time. Mixed source
policies must fail promptly, not time out. The optional fault executable injects
incorrect sequences and zero-length tokens through MPI interposition, without
modifying the provider, production client, or PhyDLL library.

For explicit shutdown-barrier coverage use
`PHYDLL_MPMD_SHUTDOWN_BARRIER=1`; the harness and both DL clients join the world
barrier only after provider destruction and control-communicator cleanup. Set
`PHYDLL_TEST_STEPS=1` when running the launcher directly to exercise immediate
shutdown after the initialization frame. The smoke matrix itself uses eight
frames.

`readiness_lifecycle.sbatch` runs both clients with readiness and the shutdown
barrier independently on/off, for one and eight frames, with a five-minute bound.
Use a separate output directory and a rebuilt harness when submitting it.

## Verified Runs

All of these isolated compute jobs completed with exit code 0:

| Job | Coverage | Results Directory |
| --- | --- | --- |
| 4694510 | 96 TOML numerical cases, both mixed-policy rejections, sequence/count faults | `tmp/opencode/phydll-readiness/final-results` |
| 4694767 | 16 shutdown/readiness/client/frame-count combinations | `tmp/opencode/phydll-readiness/lifecycle-results` |
| 4694868 | 8 diagnostic-enabled cases and mixed-policy rejection | `tmp/opencode/phydll-readiness/final-diag-results` |
| 4694948 | 8 Score-P cases and mixed-policy rejection | `tmp/opencode/phydll-readiness/final-scorep-results` |

The diagnostic and Score-P builds were rebuilt with these exact commands using
their preexisting isolated CMake caches:

```bash
bash -lc 'source set_env_claix23_cuda12.4.sh && cmake --build tmp/opencode/phydll-readiness/diag-client -j 4'
bash -lc 'export USE_SCOREP=1; source set_env_claix23_cuda12.4.sh && cmake --build tmp/opencode/phydll-readiness/scorep-client -j 4'
```

The diagnostic cache retains the preexisting opt-in flags
`-DFORWARD_CPU_DIAGNOSTIC -I/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark`.
Its job sets `FORWARD_DIAG_DIR` to the results directory. The Score-P cache uses
`scorep-mpicxx`, `WITH_SCOREP=ON`, and the generated typed registry. Both matrices
set `PHYDLL_TEST_TOPOLOGIES=2:1`, `PHYDLL_TEST_CLIENTS=0`, and
`PHYDLL_TEST_READINESS_VALUES='0 1'`; the Score-P job additionally sets
`USE_SCOREP=1`, `SCOREP_ENABLE_PROFILING=true`, and `SCOREP_ENABLE_TRACING=false`.

Inspect the verified readiness-enabled call tree with:

```bash
export USE_SCOREP=1
source set_env_claix23_cuda12.4.sh
cube_dump -w calltree tmp/opencode/phydll-readiness/final-scorep-results/scorep-20261003_1808_3096940739807645/profile.cubex
```

It places `phydll_readiness_wait` and its `MPI_Test` child under `phydll_recv`,
with payload `MPI_Waitall` as a sibling of readiness polling. Diagnostic records
include eight `solver_recv` measurements per solver, including the initial frame.
These small correctness smoke runs are not benchmark campaign measurements.
