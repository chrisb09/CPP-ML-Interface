#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMARTSIM_BUILD_SCOREP=1 exec "${SCRIPT_DIR}/install.sh" "$@"
