#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${CPPML_ENV_READY:-0}" != "1" ]; then
    source "${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
fi

PAPI_SRC_DIR="${SCRIPT_DIR}/extern/papi"
PAPI_BUILD_DIR="${SMARTSIM_PAPI_BUILD_DIR:-${SCRIPT_DIR}/tmp/opencode/papi-7.2.0-build}"
PAPI_INSTALL_DIR="${SMARTSIM_PAPI_ROOT:-${SCRIPT_DIR}/tmp/opencode/papi-7.2.0-install}"

SCOREP_SRC_DIR="${SCRIPT_DIR}/extern/scorep"
SCOREP_BUILD_DIR="${SMARTSIM_SCOREP_BUILD_DIR:-${SCRIPT_DIR}/tmp/opencode/scorep-8.4-build-papi72-noshmem}"
SCOREP_INSTALL_DIR="${SMARTSIM_SCOREP_ROOT:-${SCRIPT_DIR}/tmp/opencode/scorep-8.4-papi72-install}"
SCOREP_TAG="${SMARTSIM_SCOREP_TAG:-v8.4}"

build_jobs="${SMARTSIM_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-8}}"

ensure_submodule() {
    local submodule_path="$1"
    if ! git -C "${SCRIPT_DIR}/${submodule_path}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "${SCRIPT_DIR}" submodule update --init --recursive "${submodule_path}"
    fi
}

is_valid_scorep_install() {
    [ -x "${SCOREP_INSTALL_DIR}/bin/scorep-config" ] || return 1
    "${SCOREP_INSTALL_DIR}/bin/scorep-info" config-summary | grep -A 12 'Score-P (backend):' | grep -q 'Link mode:.*shared=yes' || return 1
    "${SCOREP_INSTALL_DIR}/bin/scorep-config" --libs | grep -q -- '-lpapi' || return 1
    ! "${SCOREP_INSTALL_DIR}/bin/scorep-config" --libs | grep -q -- '-lunwind'
}

if is_valid_scorep_install; then
    echo "Score-P already installed at ${SCOREP_INSTALL_DIR}"
    exit 0
fi

ensure_submodule "extern/papi"
ensure_submodule "extern/scorep"

"${SCRIPT_DIR}/build_papi.sh"

git -C "${SCOREP_SRC_DIR}" fetch --tags --force origin >/dev/null 2>&1 || true
git -C "${SCOREP_SRC_DIR}" checkout --detach "${SCOREP_TAG}"
git -C "${SCOREP_SRC_DIR}" submodule update --init --recursive

"${SCRIPT_DIR}/build_scorep_deps.sh"

export PATH="${SCOREP_INSTALL_DIR}/bin:${PAPI_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCOREP_INSTALL_DIR}/lib:${PAPI_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${SCOREP_INSTALL_DIR}/lib:${PAPI_INSTALL_DIR}/lib:${LIBRARY_PATH:-}"
export CPATH="${SCOREP_INSTALL_DIR}/include:${PAPI_INSTALL_DIR}/include:${CPATH:-}"

if [ ! -x "${SCOREP_SRC_DIR}/configure" ]; then
    (cd "${SCOREP_SRC_DIR}" && ./bootstrap)
fi

mkdir -p "${SCOREP_BUILD_DIR}" "${SCOREP_INSTALL_DIR}"
rm -rf "${SCOREP_BUILD_DIR}"
mkdir -p "${SCOREP_BUILD_DIR}"

(cd "${SCOREP_BUILD_DIR}" && \
    CPPFLAGS="-I${PAPI_INSTALL_DIR}/include -I${SCOREP_INSTALL_DIR}/include" \
    LDFLAGS="-L${PAPI_INSTALL_DIR}/lib -L${SCOREP_INSTALL_DIR}/lib -Wl,-rpath,${PAPI_INSTALL_DIR}/lib -Wl,-rpath,${SCOREP_INSTALL_DIR}/lib" \
    PAPI_INC="${PAPI_INSTALL_DIR}/include" \
    PAPI_LIB="${PAPI_INSTALL_DIR}/lib" \
        "${SCOREP_SRC_DIR}/configure" \
            --prefix="${SCOREP_INSTALL_DIR}" \
            --with-nocross-compiler-suite=gcc \
            --with-mpi=openmpi3 \
            --with-shmem=no \
            --with-otf2="${SCOREP_INSTALL_DIR}/bin" \
            --with-opari2="${SCOREP_INSTALL_DIR}/bin" \
            --with-cubew="${SCOREP_INSTALL_DIR}/bin" \
            --with-cubelib="${SCOREP_INSTALL_DIR}/bin" \
            --disable-cuda \
            --disable-rocm-adapter \
            --disable-dependency-tracking \
            --disable-gcc-plugin \
            --disable-llvm-plugin \
            --disable-xray \
            --disable-libwrap-generator \
            --disable-fortran \
            --enable-backend-test-runs=no \
            --enable-shared \
            --with-libunwind=no) \
    > "${SCOREP_BUILD_DIR}/configure.log" 2>&1

make -C "${SCOREP_BUILD_DIR}" -j "${build_jobs}"
make -C "${SCOREP_BUILD_DIR}" install

if ! is_valid_scorep_install; then
    echo "Score-P install verification failed at ${SCOREP_INSTALL_DIR}" >&2
    exit 1
fi

echo "Score-P installed at ${SCOREP_INSTALL_DIR}"
