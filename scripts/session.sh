#!/usr/bin/env bash
# Используется Python активированного окружения в Linux и Git Bash.
MMOT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MMOT_PYTHON="${MMOT_PYTHON:-python}"
export MMOT_PROJECT_ROOT MMOT_PYTHON
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
cd -- "$MMOT_PROJECT_ROOT"
