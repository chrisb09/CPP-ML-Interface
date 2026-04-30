
# Script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export smartredis_DIR="${SCRIPT_DIR}/extern/SmartRedis/cmake"
export smartredis_INCLUDE_DIR="${SCRIPT_DIR}/extern/SmartRedis/include"