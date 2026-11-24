#!/usr/bin/env bash
set -eu

script=$(realpath -- "${BASH_SOURCE[0]}")
project=$(dirname -- "$(dirname -- "$script")")

if [[ ${1:-} == --supervise ]]; then
    config=$2
    run_dir=$3
    workers=$4
    exec 9>"$run_dir/launcher.lock"
    if ! flock -n 9; then
        printf 'Another launcher owns %s\n' "$run_dir" >&2
        exit 1
    fi

    printf '%s\n' "$$" > "$run_dir/launcher.pid"
    exit_marker="$run_dir/launcher.exit"
    printf 'running\n' > "$exit_marker"
    cd -- "$project"
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    result=0
    "$project/.venv/bin/python" -m rcmlq_synthetic run \
        --config "$config" --run-dir "$run_dir" --workers "$workers" || result=$?
    printf '%s\n' "$result" > "$exit_marker.tmp"
    mv -- "$exit_marker.tmp" "$exit_marker"
    exit "$result"
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s CONFIG RUN_DIR [WORKERS]\n' "$0" >&2
    exit 2
fi
workers=${3:-4}
if [[ ! $workers =~ ^[1-9][0-9]*$ || ! -x "$project/.venv/bin/python" ]]; then
    printf 'Provide a positive worker count and install the project .venv first.\n' >&2
    exit 2
fi
config=$(realpath -e -- "$1")
mkdir -p -- "$2"
run_dir=$(realpath -- "$2")
nohup setsid --wait "$script" --supervise "$config" "$run_dir" "$workers" \
    </dev/null >> "$run_dir/launcher.log" 2>&1 &
printf 'Run directory %s\nLog %s/launcher.log\nExit status %s/launcher.exit\n' "$run_dir" "$run_dir" "$run_dir"
