
# Script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export smartredis_DIR="${SCRIPT_DIR}/extern/SmartRedis/cmake"
export smartredis_INCLUDE_DIR="${SCRIPT_DIR}/extern/SmartRedis/include"

# Prepend runtime_libs directories (bundled GCC 13 libstdc++/libgcc_s) to
# LD_LIBRARY_PATH so that binaries built against GCC 13 work regardless of
# which GCCcore environment module happens to be loaded at run time.
for _runtime_libs_dir in "${SCRIPT_DIR}/extern/python"/smartsim_*/runtime_libs; do
    if [ -d "$_runtime_libs_dir" ]; then
        case ":${LD_LIBRARY_PATH}:" in
            *":${_runtime_libs_dir}:"*) ;;   # already present – skip
            *) export LD_LIBRARY_PATH="${_runtime_libs_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
unset _runtime_libs_dir