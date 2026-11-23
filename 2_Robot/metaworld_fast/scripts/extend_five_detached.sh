#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SOURCE=${1:?Original completed five-seed run required}
RUN_DIR=${2:?New ten-seed output directory required}
mkdir -p "$RUN_DIR/logs"
RUN_DIR=$(realpath "$RUN_DIR")
if [[ -f "$RUN_DIR/supervisor.pid" ]] && kill -0 "$(cat "$RUN_DIR/supervisor.pid")" 2>/dev/null; then
    printf 'Already running: PID %s\n' "$(cat "$RUN_DIR/supervisor.pid")"
    exit 0
fi
nohup setsid "$ROOT/.venv/bin/python" -u -m metaworld_fast.extension \
    --source "$SOURCE" --run-dir "$RUN_DIR" \
    </dev/null >>"$RUN_DIR/logs/supervisor.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_DIR/launcher.pid"
printf 'Started PID %s\nRun directory: %s\n' "$!" "$RUN_DIR"
