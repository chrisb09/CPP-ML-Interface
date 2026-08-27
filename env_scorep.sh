#!/usr/bin/env bash

CPPML_INTERFACE_DIR="${CMI_DIR:-${CPP_ML_DIR:-}}"
if [[ -z "${CPPML_INTERFACE_DIR}" ]]; then
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    CPPML_INTERFACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  else
    CPPML_INTERFACE_DIR="$(cd "$(dirname "$0")" && pwd)"
  fi
fi

PAPI_ROOT="${SMARTSIM_PAPI_ROOT:-${CPPML_INTERFACE_DIR}/tmp/opencode/papi-7.2.0-install}"
SCOREP_ROOT="${SMARTSIM_SCOREP_ROOT:-${CPPML_INTERFACE_DIR}/tmp/opencode/scorep-8.4-papi72-install}"

export PAPI_ROOT
export SCOREP_ROOT
export SCOREP_ROOT_DIR="${SCOREP_ROOT}"
export PAPI_INC="${PAPI_ROOT}/include"
export PAPI_LIB="${PAPI_ROOT}/lib"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS:---nocompiler --user}"

export PATH="${SCOREP_ROOT}/bin:${PAPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${SCOREP_ROOT}/lib:${PAPI_ROOT}/lib:${LIBRARY_PATH:-}"
export CPATH="${SCOREP_ROOT}/include:${PAPI_ROOT}/include:${CPATH:-}"
