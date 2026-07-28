#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/phydll_py_venv/bin/activate" ]; then
    . "$PROJECT_ROOT/phydll_py_venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/CPP-ML-Interface/extern/phydll/src/python:${PYTHONPATH:-}"
exec python3 "$@"
