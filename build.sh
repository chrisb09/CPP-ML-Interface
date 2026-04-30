#!/bin/sh

run_registry_tests="${CPPML_RUN_REGISTRY_TESTS:-OFF}"
with_smartsim="${WITH_SMARTSIM:-ON}"
with_aix="${WITH_AIX:-ON}"
aix_use_prebuilt="${AIX_USE_PREBUILT:-ON}"
with_phydll="${WITH_PHYDLL:-OFF}"
with_torch="${WITH_TORCH:-ON}"
with_tensorflow="${WITH_TENSORFLOW:-OFF}"
with_onnx="${WITH_ONNX:-OFF}"
force_aix_rebuild="${FORCE_AIX_REBUILD:-OFF}"
# Set up environment for clang and CUDA
USER_PYTHON_ENV="${PWD}/extern/python/smartsim_cuda-12"
USER_PYTHON="$USER_PYTHON_ENV/bin/python"
CUDA_ROOT="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0"

# Explicitly set compilers to ensure consistency and CUDA compatibility
export CC=gcc
export CXX=g++
unset LD # Remove LD if set by install.sh to prevent CMake compiler checks from failing

# CRITICAL: Prioritize CVMFS CUDA paths to avoid using "stripped" Python wheel libraries
export LD_LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$LIBRARY_PATH"
export PATH="$CUDA_ROOT/bin:$PATH"

# configure
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCUDAToolkit_ROOT="$CUDA_ROOT" \
	-DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF \
	-DTorch_DIR="" \
	-DTORCH_VERSION="2.4.0" \
	-DCPPML_RUN_REGISTRY_TESTS="${run_registry_tests}" \
	-DWITH_SMARTSIM="${with_smartsim}" \
	-DWITH_AIX="${with_aix}" \
	-DAIX_USE_PREBUILT="${aix_use_prebuilt}" \
	-DWITH_PHYDLL="${with_phydll}" \
	-DWITH_TORCH="${with_torch}" \
	-DWITH_TENSORFLOW="${with_tensorflow}" \
	-DWITH_ONNX="${with_onnx}" \
	-DFORCE_AIX_REBUILD="${force_aix_rebuild}" || { echo "CMake configuration failed"; exit 1; }
##cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release

# build (generator-agnostic; passes -j to underlying tool)
cmake --build build -- -j8 || { echo "Build failed"; exit 1; }
##cmake --build build_release -- -j"$(nproc)"


# Example of how to run the registry parser tests if they are enabled
# CPPML_RUN_REGISTRY_TESTS=ON WITH_SMARTSIM=ON WITH_AIX=ON WITH_TORCH=ON ./build.sh