#!/bin/bash
#SBATCH --job-name=phydll_readiness
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
ROOT="${PHYDLL_TEST_PROJECT_ROOT:?Set PHYDLL_TEST_PROJECT_ROOT}"
export MLCOUPLING_INTRA_OP_THREADS=2
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_TEST_STEPS=8
RUN="${ROOT}/test/phydll_mpmd/run_phydll_mpmd_test.sh"

for topology in ${PHYDLL_TEST_TOPOLOGIES:-2:1 3:1 4:2 3:2}; do
    export PHYDLL_TEST_NP_PHY="${topology%:*}"
    export PHYDLL_TEST_NP_DL="${topology#*:}"
    for client in ${PHYDLL_TEST_CLIENTS:-0 1}; do
        export PHYDLL_TEST_DL_PYTHON="${client}"
        for readiness in ${PHYDLL_TEST_READINESS_VALUES:-default 0 1}; do
            export PHYDLL_TEST_READINESS="${readiness}"
            for mode in 18to1 1to18; do
                for layout in packed uniform_chunks; do
                    label="${topology}-${client}-${readiness}-${mode}-${layout}"
                    timeout 90 bash "${RUN}" "${mode}" "${layout}" > "${label}.log" 2>&1
                    printf 'PASS %s\n' "${label}"
                done
            done
        done
    done
done

export PHYDLL_TEST_NP_PHY=2 PHYDLL_TEST_NP_DL=1 PHYDLL_TEST_READINESS=mixed
for client in ${PHYDLL_TEST_CLIENTS:-0 1}; do
    export PHYDLL_TEST_DL_PYTHON="${client}"
    set +e
    timeout 90 bash "${RUN}" 18to1 packed > "mixed-${client}.log" 2>&1
    result=$?
    set -e
    if [[ "${result}" == 0 || "${result}" == 124 ]]; then
        printf 'FAIL mixed flags client=%s status=%s\n' "${client}" "${result}"
        exit 1
    fi
    if ! grep -q 'mixed solver_readiness_wait flags' "mixed-${client}.log"; then
        printf 'FAIL mixed flags client=%s failed for an unexpected reason\n' "${client}"
        exit 1
    fi
    printf 'PASS mixed flags rejected client=%s status=%s\n' "${client}" "${result}"
done
if [[ "${PHYDLL_TEST_FAULTS:-0}" == 1 ]]; then
    export PHYDLL_TEST_NP_PHY=2 PHYDLL_TEST_NP_DL=1 PHYDLL_TEST_READINESS=1
    export PHYDLL_TEST_DL_PYTHON=0 PHYDLL_TEST_DL_EXECUTABLE=phydll_dl_fault_client
    for fault in sequence count; do
        export PHYDLL_TEST_TOKEN_FAULT="${fault}"
        set +e
        timeout 90 bash "${RUN}" 18to1 packed > "fault-${fault}.log" 2>&1
        result=$?
        set -e
        if [[ "${result}" == 0 || "${result}" == 124 ]] ||
           ! grep -q 'PhyDLL readiness sequence mismatch' "fault-${fault}.log"; then
            printf 'FAIL token fault=%s status=%s\n' "${fault}" "${result}"
            exit 1
        fi
        printf 'PASS malformed token rejected fault=%s status=%s\n' "${fault}" "${result}"
    done
fi
printf 'ALL READINESS SMOKE TESTS PASS\n'
