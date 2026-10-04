#!/bin/bash
#SBATCH --job-name=phydll_lifecycle
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

set -euo pipefail
RUN="${PHYDLL_TEST_PROJECT_ROOT:?}/test/phydll_mpmd/run_phydll_mpmd_test.sh"
export PHYDLL_TEST_NP_PHY=2 PHYDLL_TEST_NP_DL=1
export MLCOUPLING_INTRA_OP_THREADS=2 MLCOUPLING_INTER_OP_THREADS=1
for barrier in 0 1; do
    export PHYDLL_MPMD_SHUTDOWN_BARRIER="${barrier}"
    for client in 0 1; do
        export PHYDLL_TEST_DL_PYTHON="${client}"
        for readiness in 0 1; do
            export PHYDLL_TEST_READINESS="${readiness}"
            for steps in 1 8; do
                export PHYDLL_TEST_STEPS="${steps}"
                label="shutdown-${barrier}-${client}-${readiness}-${steps}"
                timeout 45 bash "${RUN}" 18to1 packed > "${label}.log" 2>&1
                printf 'PASS %s\n' "${label}"
            done
        done
    done
done
printf 'ALL LIFECYCLE TESTS PASS\n'
