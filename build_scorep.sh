#!/bin/sh

# Set Score-P environment flags
export USE_SCOREP=1

# Use standard Score-P instrumenter wrapper flags (defaults to mpp=none to avoid MPMD python deadlocks)
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=none --io=none --memory=malloc --thread=none --nocuda"

# Forward all arguments (e.g. test, test <name>) directly to the main build script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/build.sh" "$@"
