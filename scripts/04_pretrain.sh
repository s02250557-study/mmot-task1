#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/session.sh"
exec "$MMOT_PYTHON" -B src/training/train.py "$@"
