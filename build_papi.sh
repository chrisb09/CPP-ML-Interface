#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${CPPML_ENV_READY:-0}" != "1" ]; then
    source "${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
fi

PAPI_SRC_DIR="${SCRIPT_DIR}/extern/papi"
PAPI_BUILD_DIR="${SMARTSIM_PAPI_BUILD_DIR:-${SCRIPT_DIR}/tmp/opencode/papi-7.2.0-build}"
PAPI_INSTALL_DIR="${SMARTSIM_PAPI_ROOT:-${SCRIPT_DIR}/tmp/opencode/papi-7.2.0-install}"
PAPI_TAG="${SMARTSIM_PAPI_TAG:-papi-7-2-0-t}"

build_jobs="${SMARTSIM_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-8}}"

ensure_submodule() {
    local submodule_path="$1"
    if ! git -C "${SCRIPT_DIR}/${submodule_path}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "${SCRIPT_DIR}" submodule update --init --recursive "${submodule_path}"
    fi
}

is_valid_papi_install() {
    [ -x "${PAPI_INSTALL_DIR}/bin/papi_component_avail" ] || return 1
    [ -f "${PAPI_INSTALL_DIR}/lib/libpapi.so.7.2" ] || return 1
}

if is_valid_papi_install; then
    echo "PAPI already installed at ${PAPI_INSTALL_DIR}"
    exit 0
fi

ensure_submodule "extern/papi"

mkdir -p "${PAPI_BUILD_DIR}" "${PAPI_INSTALL_DIR}"

git -C "${PAPI_SRC_DIR}" fetch --tags --force origin >/dev/null 2>&1 || true
git -C "${PAPI_SRC_DIR}" checkout --detach "${PAPI_TAG}"

(cd "${PAPI_BUILD_DIR}" && \
    CPPFLAGS="-I${PAPI_INSTALL_DIR}/include" \
    LDFLAGS="-L${PAPI_INSTALL_DIR}/lib -Wl,-rpath,${PAPI_INSTALL_DIR}/lib" \
        "${PAPI_SRC_DIR}/src/configure" \
            --prefix="${PAPI_INSTALL_DIR}" \
            --with-components="infiniband,net") \
    > "${PAPI_BUILD_DIR}/configure.log" 2>&1

make -C "${PAPI_BUILD_DIR}" -j "${build_jobs}"
make -C "${PAPI_BUILD_DIR}" install

if ! is_valid_papi_install; then
    echo "PAPI install verification failed at ${PAPI_INSTALL_DIR}" >&2
    exit 1
fi

echo "PAPI installed at ${PAPI_INSTALL_DIR}"
