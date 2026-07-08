#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PAPI_ROOT="${SMARTSIM_PAPI_ROOT:-${SCRIPT_DIR}/tmp/opencode/papi-7.2.0-install}"
SCOREP_ROOT="${SMARTSIM_SCOREP_ROOT:-${SCRIPT_DIR}/tmp/opencode/scorep-8.4-papi72-install}"

export PAPI_ROOT
export SCOREP_ROOT
export SCOREP_ROOT_DIR="${SCOREP_ROOT}"
export PAPI_INC="${PAPI_ROOT}/include"
export PAPI_LIB="${PAPI_ROOT}/lib"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS:---nocompiler --user --mpp=none --io=none --memory=malloc --thread=none --nocuda}"

export PATH="${SCOREP_ROOT}/bin:${PAPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LIBRARY_PATH:-}"
export CPATH="${SCOREP_ROOT}/include:${PAPI_ROOT}/include:${CPATH:-}"
