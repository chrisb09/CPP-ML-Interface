#!/bin/bash
#SBATCH --job-name=scorep-papi-720
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --hwctr=papi
#SBATCH --output=scorep_papi_720_%j.out
#SBATCH --error=scorep_papi_720_%j.err

set -euo pipefail

PAPI_ROOT="/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/papi-7.2.0-install"
SCOREP_ROOT="/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install"
WORKDIR="${SLURM_TMPDIR:-/tmp}/scorep_papi_720_${SLURM_JOB_ID}"

source /opt/lmod/lmod/init/bash
export TERM=dumb
export LMOD_PAGER=cat
export PAGER=cat
module load gompi/2022a

export PATH="${SCOREP_ROOT}/bin:${PAPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LIBRARY_PATH:-}"
export CPATH="${SCOREP_ROOT}/include:${PAPI_ROOT}/include:${CPATH:-}"
export PKG_CONFIG_PATH="${PAPI_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_TOTAL_MEMORY=16000K
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=none --io=none --memory=malloc --thread=none --nocuda"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

cat > smoke.c <<'EOF'
#include <stdint.h>
#include <stdio.h>
#include <scorep/SCOREP_User.h>

SCOREP_USER_REGION_DEFINE(smoke_loop_region);

int main(void)
{
    SCOREP_USER_REGION_BEGIN(smoke_loop_region, "smoke_loop", SCOREP_USER_REGION_TYPE_COMMON);

    volatile double x = 1.0;
    for (uint64_t i = 0; i < 100000000ULL; ++i)
    {
        x = x * 1.0000000001 + 0.0000000001;
    }

    SCOREP_USER_REGION_END(smoke_loop_region);

    printf("x=%f\n", x);
    return 0;
}
EOF

scorep-gcc smoke.c -O2 -o smoke

echo "Linked libraries:"
ldd ./smoke | rg "libpapi|scorep" || true

echo "Local PAPI component availability:"
papi_component_avail | rg -n "(?i)(perf_event|infiniband|net|sysdetect)" || true

echo "Local PAPI network / IB events:"
papi_native_avail | rg -n "(?i)(net:::ib0:rx:byte|net:::ib0:tx:byte|infiniband:::mlx5_0_1_ext:port_rcv_data|infiniband:::mlx5_0_1_ext:port_xmit_data)" || true

run_case() {
    local metric="$1"
    local expect_success="$2"
    local safe_metric="${metric//:/_}"
    safe_metric="${safe_metric//\//_}"
    safe_metric="${safe_metric//,/__}"
    export SCOREP_EXPERIMENT_DIRECTORY="scorep_papi_720_local_${safe_metric:-empty}"
    export SCOREP_METRIC_PAPI="${metric}"

    rm -rf "${SCOREP_EXPERIMENT_DIRECTORY}"

    echo "Running with SCOREP_METRIC_PAPI='${SCOREP_METRIC_PAPI}'"
    set +e
    ./smoke
    local rc=$?
    set -e

    if [[ "${expect_success}" == "yes" && ${rc} -ne 0 ]]; then
        echo "ERROR: expected success but run failed (rc=${rc})" >&2
        exit 1
    fi

    if [[ "${expect_success}" == "no" && ${rc} -eq 0 ]]; then
        echo "WARNING: run succeeded unexpectedly" >&2
    fi

    if [[ "${expect_success}" == "yes" && ! -f "${SCOREP_EXPERIMENT_DIRECTORY}/profile.cubex" ]]; then
        echo "ERROR: missing ${SCOREP_EXPERIMENT_DIRECTORY}/profile.cubex" >&2
        exit 1
    fi

    if [[ -d "${SCOREP_EXPERIMENT_DIRECTORY}" ]]; then
        ls -la "${SCOREP_EXPERIMENT_DIRECTORY}"
    fi
}

run_case "net:::ib0:rx:byte,net:::ib0:tx:byte" "yes"
run_case "infiniband:::mlx5_0_1_ext:port_rcv_data,infiniband:::mlx5_0_1_ext:port_xmit_data" "yes"
