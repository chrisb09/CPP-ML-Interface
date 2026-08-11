#!/bin/bash
#SBATCH --job-name=phydll_mpmd_test
#SBATCH --time=00:30:00
#SBATCH --ntasks=3
#SBATCH --output=phydll_mpmd_test_%j.log
#SBATCH --error=phydll_mpmd_test_%j.err

# Run the full MPMD regression test as a batch job so nothing long-running
# stays on the login node. CPU-only, so a lightweight partition is preferred;
# the scheduler auto-selects when the user is over quota.
#
# Usage: sbatch submit_phydll_mpmd_test.sh <18to1|1to18> <packed|uniform_chunks>
# Environment overrides: PHYDLL_BATCH_CHUNK

MODE="${1:?usage: submit_phydll_mpmd_test.sh <18to1|1to18> <packed|uniform_chunks>}"
LAYOUT="${2:?usage: submit_phydll_mpmd_test.sh <18to1|1to18> <packed|uniform_chunks>}"

# Invoke from the directory containing this script (sbatch copies the script to
# the spool dir, so $0-based paths do not work; the batch job's cwd is the
# submission directory).
"${PWD}/run_phydll_mpmd_test.sh" "${MODE}" "${LAYOUT}"
