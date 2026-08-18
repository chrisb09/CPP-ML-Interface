#!/bin/bash
# Run the PhyDLL MPMD regression test (2 solver ranks + 1 DL rank).
#
# Usage:
#   ./run_phydll_mpmd_test.sh <mode> <transport_layout>
#   mode: 18to1 | 1to18 ; layout: packed | uniform_chunks
#
# Environment (optional):
#   PHYDLL_TEST_NTASKS  total MPI ranks (default 3 = 2 PHY + 1 DL)
#   PHYDLL_BATCH_CHUNK  DL inference chunk size (default 0)
#   PHYDLL_TEST_DL_PYTHON=1  use the Python DL client instead of the C++ one

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
source "${PROJECT_ROOT}/set_env_claix23_cuda12.4.sh"

MODE="${1:?usage: run_phydll_mpmd_test.sh <18to1|1to18> <packed|uniform_chunks>}"
LAYOUT="${2:?usage: run_phydll_mpmd_test.sh <18to1|1to18> <packed|uniform_chunks>}"

NP_PHY=2
NP_DL=1
NTASKS=$((NP_PHY + NP_DL))
BATCH_CHUNK="${PHYDLL_BATCH_CHUNK:-0}"

case "${MODE}" in
    18to1) DL_FIELD_COUNT=1  ;;
    1to18) DL_FIELD_COUNT=18 ;;
    *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac

export PHYDLL_DL_FIELD_COUNT="${DL_FIELD_COUNT}"
CUDA_ROOT_DIR="${CUDA_ROOT:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0}"
# Stub libcuda.so.1 lets the libtorch client load on CPU-only nodes; the
# client never calls into the CUDA driver because device=CPU.
export LD_LIBRARY_PATH="${PROJECT_ROOT}/extern/phydll/build/lib:${CUDA_ROOT_DIR}/lib64:${CUDA_ROOT_DIR}/lib/stubs:${LD_LIBRARY_PATH}"

mkdir -p "${PROJECT_ROOT}/test/phydll_mpmd/models"
MODEL_DIR="${PROJECT_ROOT}/test/phydll_mpmd/models"
if [[ ! -f "${MODEL_DIR}/model_18to1.pt" || ! -f "${MODEL_DIR}/model_1to18.pt" ]]; then
    python3 "${PROJECT_ROOT}/scripts/create_phydll_asym_models.py" "${MODEL_DIR}"
fi

MODEL=""
if [[ "${MODE}" == "18to1" ]]; then
    MODEL="${MODEL_DIR}/model_18to1.pt"
else
    MODEL="${MODEL_DIR}/model_1to18.pt"
fi
PHY_BIN="${PROJECT_ROOT}/build/dl_clients/phydll_phy_test"
if [[ "${PHYDLL_TEST_DL_PYTHON:-0}" == "1" ]]; then
    DL_BIN="${PROJECT_ROOT}/dl_clients/phydll_dl_client.py"
    # The Python client imports pyphydll from the PhyDLL source tree.
    export PYTHONPATH="${PROJECT_ROOT}/extern/phydll/src/python:${PYTHONPATH:-}"
    export PHYDLL_PY_SCOREP_WRAPPER=0
else
    DL_BIN="${PROJECT_ROOT}/build/dl_clients/phydll_dl_client"
fi
PHY_ARGS="${MODE} ${MODEL} ${LAYOUT} ${BATCH_CHUNK}"

CONF="${SCRIPT_DIR}/phydll_mpmd_${SLURM_JOB_ID:-$$}.conf"
{
    for ((i = 0; i < NP_PHY; i++)); do
        printf '%d\t%s %s\n' "$i" "${PHY_BIN}" "${PHY_ARGS}"
    done
    printf '%d\t%s\n' "${NP_PHY}" "${DL_BIN}"
} > "${CONF}"
echo "[run_phydll_mpmd_test] ${NP_PHY} PHY + ${NP_DL} DL, layout=${LAYOUT}, mode=${MODE}, DL_FIELD_COUNT=${DL_FIELD_COUNT}"
cat "${CONF}"

PARTITION_FLAGS=()
if [[ -n "${PHYDLL_TEST_PARTITION:-}" ]]; then
    PARTITION_FLAGS+=(--partition="${PHYDLL_TEST_PARTITION}")
fi

srun "${PARTITION_FLAGS[@]}" --ntasks="${NTASKS}" --kill-on-bad-exit --multi-prog "${CONF}"
rm -f "${CONF}"
