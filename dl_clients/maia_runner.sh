#!/bin/bash
set -eu

dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "${dir}/../.." && pwd)"
if [ -f "${project_dir}/setup_env_claix23.sh" ]; then
    source "${project_dir}/setup_env_claix23.sh" >/dev/null 2>&1 || true
fi
cuda_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib"
cuda_stubs="${cuda_lib}/stubs"
cupti_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/extras/CUPTI/lib64"
gcc13_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/13.2.0/lib64"

# Check if real GPU driver is available on the current node
is_gpu_node=0
if [[ -f "/usr/lib64/libnvidia-ml.so.1" ]] || [[ -f "/lib64/libnvidia-ml.so.1" ]] || [[ -e "/dev/nvidia0" ]] || [[ -e "/dev/nvidiactl" ]]; then
    if [[ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" && "${CUDA_VISIBLE_DEVICES:-}" != "" ]]; then
        is_gpu_node=1
    elif [[ -e "/dev/nvidia0" ]] || [[ -e "/dev/nvidiactl" ]]; then
        is_gpu_node=1
    fi
fi

if [[ "${is_gpu_node}" -eq 1 ]]; then
    export LD_LIBRARY_PATH="${project_dir}/maia/build_gnu_production/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}:${cupti_lib}:${gcc13_lib}:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${project_dir}/maia/build_gnu_production/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}:${cuda_stubs}:${cupti_lib}:${gcc13_lib}:${LD_LIBRARY_PATH:-}"
fi

exec "$@"
