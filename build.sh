#!/bin/bash

# Source the main environment script from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/..")"
source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"

if [[ "${USE_SCOREP}" == "1" ]]; then
    if [ -f "${SCRIPT_DIR}/env_scorep.sh" ]; then
        source "${SCRIPT_DIR}/env_scorep.sh"
    else
        echo "USE_SCOREP=1 but ${SCRIPT_DIR}/env_scorep.sh is missing. Run ./install_scorep.sh first." >&2
        exit 1
    fi
    if ! command -v scorep-mpicxx >/dev/null 2>&1 || ! command -v scorep-config >/dev/null 2>&1; then
        echo "USE_SCOREP=1 but local Score-P is not on PATH. Run ./install_scorep.sh first." >&2
        exit 1
    fi
fi

echo "DEBUG: SCOREP_ROOT_DIR = $SCOREP_ROOT_DIR"
echo "DEBUG: PATH = $PATH"
echo "DEBUG: MODULEPATH = $MODULEPATH"
type module

# Usage:
#   ./build.sh          - build the project (tests excluded)

#   ./build.sh test     - build the project including tests, then run them
#   ./build.sh test <name>  - build and run only the named test (e.g. test_flexible_fallback)

mode="${1:-}"
test_filter="${2:-}"

run_registry_tests="${CPPML_RUN_REGISTRY_TESTS:-OFF}"
with_smartsim="${WITH_SMARTSIM:-ON}"
with_aix="${WITH_AIX:-ON}"
with_phydll="${WITH_PHYDLL:-ON}"
with_torch="${WITH_TORCH:-ON}"
with_tensorflow="${WITH_TENSORFLOW:-OFF}"
with_onnx="${WITH_ONNX:-OFF}"
force_aix_rebuild="${FORCE_AIX_REBUILD:-OFF}"

# Only compile tests when explicitly requested
if [ "$mode" = "test" ]; then
    build_testing="ON"
else
    build_testing="OFF"
fi

# Set up environment for clang and CUDA
USER_PYTHON_ENV="${PWD}/extern/python/smartsim_cuda-12"
USER_PYTHON="$USER_PYTHON_ENV/bin/python"
CUDA_ROOT="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0"

# Explicitly set compilers to ensure consistency and CUDA compatibility
if [[ "${USE_SCOREP}" == "1" ]]; then
    export CC=scorep-mpicc
    export CXX=scorep-mpicxx
    scorep_flag="-DWITH_SCOREP=ON"
    aix_use_prebuilt="${AIX_USE_PREBUILT:-ON}"
else
    export CC=gcc
    export CXX=g++
    scorep_flag="-DWITH_SCOREP=OFF"
    aix_use_prebuilt="${AIX_USE_PREBUILT:-ON}"
fi
unset LD # Remove LD if set by install.sh to prevent CMake compiler checks from failing

# CRITICAL: Prioritize CVMFS CUDA paths to avoid using "stripped" Python wheel libraries
export LD_LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$LIBRARY_PATH"
export PATH="$CUDA_ROOT/bin:$PATH"

# configure
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCUDAToolkit_ROOT="$CUDA_ROOT" \
	-DCMAKE_CXX_FLAGS="-I$CUDA_ROOT/include -g -O0" \
	-DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF \
	-DAIX_SKIP_VENV_CREATION=ON \
	-DTorch_DIR="" \
	-DTORCH_VERSION="2.4.0" \
	-DTORCH_CUDA_ARCH_LIST="9.0" \
	-DCPPML_RUN_REGISTRY_TESTS="${run_registry_tests}" \
	-DWITH_SMARTSIM="${with_smartsim}" \
	-DWITH_AIX="${with_aix}" \
	-DAIX_USE_PREBUILT="${aix_use_prebuilt}" \
	-DWITH_PHYDLL="${with_phydll}" \
	-DWITH_TORCH="${with_torch}" \
	-DWITH_TENSORFLOW="${with_tensorflow}" \
	-DWITH_ONNX="${with_onnx}" \
	-DFORCE_AIX_REBUILD="${force_aix_rebuild}" \
	-DBUILD_TESTING="${build_testing}" \
	${scorep_flag} \
	-DTEST_PYTHON_EXECUTABLE="${USER_PYTHON}" || { echo "CMake configuration failed"; exit 1; }

# build (generator-agnostic; passes -j to underlying tool)
build_jobs="${SLURM_CPUS_ON_NODE:-8}"
echo "Building with -j${build_jobs} parallel jobs..."
cmake --build build -j ${build_jobs} || { echo "Build failed"; exit 1; }

# run tests if requested
if [ "$mode" = "test" ]; then
    # Ensure bundled GCC 13 runtime libs are on the path
    . ./env.sh

    echo ""
    echo "=== Running tests ==="
    if [ -n "$test_filter" ]; then
        ctest --test-dir build --output-on-failure -R "$test_filter"
    else
        ctest --test-dir build --output-on-failure
    fi
fi
