#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${CPPML_ENV_READY:-0}" != "1" ]; then
    source "${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
fi

SCOREP_INSTALL_DIR="${SMARTSIM_SCOREP_ROOT:-${SCRIPT_DIR}/tmp/opencode/scorep-8.4-papi72-install}"
SCOREP_DEPS_BUILD_DIR="${SMARTSIM_SCOREP_DEPS_BUILD_DIR:-${SCRIPT_DIR}/tmp/opencode/scorep-8.4-deps-build}"
SCOREP_DOWNLOAD_DIR="${SMARTSIM_SCOREP_DOWNLOAD_DIR:-${SCRIPT_DIR}/tmp/opencode/scorep-downloads}"

OTF2_TAG="${SMARTSIM_OTF2_TAG:-otf2-3.1.1}"
OPARI2_TAG="${SMARTSIM_OPARI2_TAG:-opari2-2.0.9}"
CUBEW_TAG="${SMARTSIM_CUBEW_TAG:-cubew-4.8.2}"
CUBELIB_TAG="${SMARTSIM_CUBELIB_TAG:-cubelib-4.8.2}"

build_jobs="${SMARTSIM_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-8}}"
TMPDIR="${TMPDIR:-${SCRIPT_DIR}/tmp}"
mkdir -p "${TMPDIR}"

download_file() {
    local url="$1"
    local dest="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "${dest}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "${dest}" "${url}"
    else
        echo "Neither wget nor curl is available for downloading ${url}" >&2
        exit 1
    fi
}

archive_root_dir() {
    local archive="$1"
    local first_entry
    read -r first_entry < <(tar -tf "${archive}")
    printf '%s\n' "${first_entry%%/*}"
}

install_dependency() {
    local project="$1"
    local tag="$2"
    local config_tool="$3"
    local build_dir="$4"

    if [ -x "${SCOREP_INSTALL_DIR}/bin/${config_tool}" ]; then
        echo "${project} already installed at ${SCOREP_INSTALL_DIR}"
        return 0
    fi

    local download_dir="${SCOREP_DOWNLOAD_DIR}/${project}"
    local tarball="${download_dir}/${tag}.tar.gz"
    local tmp_dir src_root src_dir

    mkdir -p "${download_dir}" "${SCOREP_INSTALL_DIR}"
    if [ ! -f "${tarball}" ]; then
        echo "Downloading ${project} ${tag}..."
        download_file "https://perftools.pages.jsc.fz-juelich.de/cicd/${project}/tags/${tag}/${tag}.tar.gz" "${tarball}"
    fi

    tmp_dir="$(mktemp -d "${TMPDIR}/scorep-${project}.XXXXXX")"
    src_root="$(archive_root_dir "${tarball}")"
    src_dir="${tmp_dir}/${src_root}"
    tar -xf "${tarball}" -C "${tmp_dir}"

    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"

    echo "Configuring ${project} ${tag}..."
    (
        cd "${build_dir}" &&
        PATH="${SCOREP_INSTALL_DIR}/bin:${PATH}" \
        LD_LIBRARY_PATH="${SCOREP_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}" \
        "${src_dir}/configure" \
            --prefix="${SCOREP_INSTALL_DIR}"
    ) > "${build_dir}/configure.log" 2>&1

    echo "Building ${project} ${tag}..."
    make -C "${build_dir}" -j "${build_jobs}"
    make -C "${build_dir}" install

    rm -rf "${tmp_dir}"
}

install_dependency "cubelib" "${CUBELIB_TAG}" "cubelib-config" "${SCOREP_DEPS_BUILD_DIR}/cubelib"
install_dependency "cubew" "${CUBEW_TAG}" "cubew-config" "${SCOREP_DEPS_BUILD_DIR}/cubew"
install_dependency "otf2" "${OTF2_TAG}" "otf2-config" "${SCOREP_DEPS_BUILD_DIR}/otf2"
install_dependency "opari2" "${OPARI2_TAG}" "opari2-config" "${SCOREP_DEPS_BUILD_DIR}/opari2"

echo "Score-P dependency stack installed at ${SCOREP_INSTALL_DIR}"
