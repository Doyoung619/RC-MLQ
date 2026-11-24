#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${1:?Usage: run_e01_detached.sh RUN_DIR [ROBOT_E01_OPTIONS]}
shift
mkdir -p -- "$run_dir/logs"
run_dir=$(realpath -- "$run_dir")
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

nohup setsid bash -c '
    root=$1
    run_dir=$2
    shift 2
    exec 9>"$run_dir/launcher.lock"
    flock -n 9 || exit 1
    printf "%s\n" "$$" > "$run_dir/launcher.pid"
    printf "%s\n" running > "$run_dir/exit_code"
    "$root/.venv/bin/python" -u "$root/robot_e01.py" run --run-dir "$run_dir" "$@"
    result=$?
    printf "%s\n" "$result" > "$run_dir/exit_code.tmp"
    mv -- "$run_dir/exit_code.tmp" "$run_dir/exit_code"
    exit "$result"
' bash "$root" "$run_dir" "$@" </dev/null >>"$run_dir/logs/supervisor.log" 2>&1 &

printf 'Launcher PID %s\nLog %s/logs/supervisor.log\nExit marker %s/exit_code\n' "$!" "$run_dir" "$run_dir"
