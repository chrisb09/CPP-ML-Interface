#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIX_DIR="${SCRIPT_DIR}/extern/AIxeleratorService"
TMP_DIR="${SCRIPT_DIR}/tmp/opencode"
LIBTORCH_DIR="${SMARTSIM_LIBTORCH_DIR:-${SCRIPT_DIR}/extern/libtorch}"
AIX_VENV_DIR="${SMARTSIM_AIX_VENV_DIR:-${SCRIPT_DIR}/extern/python/smartsim_cuda-12}"
TORCH_VERSION="${SMARTSIM_TORCH_VERSION:-2.4.0}"
BUILD_JOBS="${SMARTSIM_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-8}}"

usage() {
    echo "Usage: $0 [both|plain|noscorep|scorep]"
}

ensure_base_env() {
    source "${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
    export CC=gcc
    export CXX=g++
}

ensure_local_scorep() {
    if command -v scorep-config >/dev/null 2>&1 && command -v scorep-mpicxx >/dev/null 2>&1; then
        return 0
    fi

    echo "Local Score-P tools not found; bootstrapping the custom stack..."
    "${SCRIPT_DIR}/build_scorep.sh"
    source "${SCRIPT_DIR}/env_scorep.sh"

    if ! command -v scorep-config >/dev/null 2>&1 || ! command -v scorep-mpicxx >/dev/null 2>&1; then
        echo "Local Score-P bootstrap did not provide scorep-config/scorep-mpicxx." >&2
        exit 1
    fi
}

validate_install() {
    local variant="$1"
    local install_dir="$2"
    local lib_path="${install_dir}/lib/libAIxeleratorService.so"

    if [ ! -f "${lib_path}" ]; then
        echo "Missing expected library: ${lib_path}" >&2
        exit 1
    fi

    local dyn
    dyn="$(readelf -d "${lib_path}")"
    local deps
    deps="$(ldd "${lib_path}")"

    if [[ "${variant}" == "plain" ]]; then
        if grep -qE 'libscorep_|libpapi\.so' <<<"${dyn}"; then
            echo "Plain AIX build still links against Score-P/PAPI:" >&2
            grep -E 'libscorep_|libpapi\.so' <<<"${dyn}" >&2
            exit 1
        fi
        if grep -qE '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${dyn}"; then
            echo "Plain AIX build still has CVMFS Score-P/PAPI paths in RPATH:" >&2
            grep -E '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${dyn}" >&2
            exit 1
        fi
        if grep -qE 'libscorep_|libpapi\.so|/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${deps}"; then
            echo "Plain AIX build resolves Score-P/PAPI dependencies unexpectedly:" >&2
            grep -E 'libscorep_|libpapi\.so|/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${deps}" >&2
            exit 1
        fi
    else
        if ! grep -qE 'libscorep_|libpapi\.so' <<<"${dyn}"; then
            echo "Score-P AIX build does not link against Score-P/PAPI as expected." >&2
            exit 1
        fi
        if grep -qE '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${dyn}"; then
            echo "Score-P AIX build still references CVMFS Score-P/PAPI paths:" >&2
            grep -E '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/' <<<"${dyn}" >&2
            exit 1
        fi
        if grep -qE '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/|/lib64/libpapi\.so' <<<"${deps}"; then
            echo "Score-P AIX build resolves Score-P/PAPI outside the local stack:" >&2
            grep -E '/cvmfs/.*/Score-P/|/cvmfs/.*/PAPI/|/lib64/libpapi\.so' <<<"${deps}" >&2
            exit 1
        fi
    fi
}

build_variant() {
    local variant="$1"
    local with_scorep="$2"
    local build_dir="${TMP_DIR}/aix-${variant}-build"
    local install_dir

    if [[ "${with_scorep}" == "1" ]]; then
        install_dir="${AIX_DIR}/INSTALL-SCOREP"
        ensure_local_scorep
    else
        install_dir="${AIX_DIR}/INSTALL"
    fi

    if [ ! -d "${AIX_VENV_DIR}" ]; then
        echo "Expected Python env not found: ${AIX_VENV_DIR}" >&2
        exit 1
    fi

    rm -rf "${build_dir}" "${install_dir}"
    mkdir -p "${build_dir}"

    local -a cmake_args=(
        -S "${AIX_DIR}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${install_dir}"
        -DBUILD_TESTS=OFF
        -DWITH_TORCH=ON
        -DTORCH_VERSION="${TORCH_VERSION}"
        -DLIBTORCH_DIR="${LIBTORCH_DIR}"
        -DAIX_VENV_DIR="${AIX_VENV_DIR}"
        -DAIX_SKIP_VENV_CREATION=ON
        -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF
    )

    if [[ "${with_scorep}" == "1" ]]; then
        cmake_args+=(
            -DWITH_SCOREP=ON
            -DCMAKE_SHARED_LINKER_FLAGS=-L${SCOREP_ROOT}/lib\ -L${PAPI_ROOT}/lib\ -Wl,-rpath,${SCOREP_ROOT}/lib\ -Wl,-rpath,${PAPI_ROOT}/lib
            -DCMAKE_EXE_LINKER_FLAGS=-L${SCOREP_ROOT}/lib\ -L${PAPI_ROOT}/lib\ -Wl,-rpath,${SCOREP_ROOT}/lib\ -Wl,-rpath,${PAPI_ROOT}/lib
        )
    else
        cmake_args+=(-DWITH_SCOREP=OFF)
    fi

    echo "Configuring AIX ${variant} build..."
    cmake "${cmake_args[@]}"

    echo "Building AIX ${variant} variant with -j${BUILD_JOBS}..."
    cmake --build "${build_dir}" -j "${BUILD_JOBS}"

    echo "Installing AIX ${variant} variant to ${install_dir}..."
    cmake --install "${build_dir}"

    git -C "${AIX_DIR}" rev-parse HEAD > "${install_dir}/.git_commit"

    validate_install "${variant}" "${install_dir}"
    echo "AIX ${variant} install ready at ${install_dir}"
}

mode="${1:-both}"

case "${mode}" in
    both|all)
        ensure_base_env
        build_variant plain 0
        build_variant scorep 1
        ;;
    plain|noscorep)
        ensure_base_env
        build_variant plain 0
        ;;
    scorep)
        ensure_base_env
        build_variant scorep 1
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
