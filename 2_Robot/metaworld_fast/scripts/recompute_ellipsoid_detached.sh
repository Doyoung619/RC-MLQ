#!/usr/bin/env bash
set -euo pipefail
cd /data1/doyoung/Ideal_Control
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
reevaluation_dir="${1:-metaworld_fast/outputs/ellipsoid_t128_s10_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$reevaluation_dir/logs"
reevaluation_dir="$(realpath "$reevaluation_dir")"
.venv/bin/python - "$reevaluation_dir" <<'PY'
import os
from pathlib import Path
import subprocess
import sys
root=Path(sys.argv[1])
args=[sys.executable,'-u','-m','metaworld_fast.ellipsoid_recompute',
      '--source','metaworld_fast/outputs/rcmlq_t128_s10_20261004T2008Z',
      '--manuscript','/home/compu/.codex/attachments/8a76e776-da90-4bbd-b048-f33aea924769/Pasted text.txt',
      '--run-dir',str(root)]
with (root/'logs/supervisor.log').open('a') as log:
    child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,close_fds=True,env=os.environ.copy())
Path('metaworld_fast/ELLIPSOID_CURRENT_RUN.txt').write_text(str(root)+'\n')
print(f'PID={child.pid}\nRUN_DIR={root}\nDetached session; SSH disconnect does not stop the process.')
PY
