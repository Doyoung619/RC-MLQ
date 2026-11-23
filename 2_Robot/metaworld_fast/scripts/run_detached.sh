#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MUJOCO_GL=egl
RUN_DIR=${1:-"metaworld_fast/outputs/rcmlq_$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$RUN_DIR/logs"
RUN_DIR=$(realpath "$RUN_DIR")
if [[ -f "$RUN_DIR/supervisor.pid" ]] && kill -0 "$(cat "$RUN_DIR/supervisor.pid")" 2>/dev/null; then
    printf 'Already running: PID %s\n' "$(cat "$RUN_DIR/supervisor.pid")"
    exit 0
fi
nohup setsid "$ROOT/.venv/bin/python" -u -m metaworld_fast.cli run \
    --config metaworld_fast/config/final.yaml --run-dir "$RUN_DIR" --resume \
    </dev/null >>"$RUN_DIR/logs/supervisor.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf 'Started launcher PID %s\nRun directory: %s\nLog: %s/logs/supervisor.log\n' "$PID" "$RUN_DIR" "$RUN_DIR"
