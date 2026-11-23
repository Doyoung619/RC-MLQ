"""Small atomic artifact and provenance helpers (no CUDA imports)."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_json(path, value):
    def default(obj):
        if hasattr(obj, 'tolist'): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=default, allow_nan=False) + '\n')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def source_manifest():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.sh') and 'outputs' not in p.parts and '__pycache__' not in p.parts}

def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return {'returncode': 127, 'stdout': '', 'stderr': str(exc)}
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

def hardware():
    affinity = sorted(os.sched_getaffinity(0))
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    cgpath = Path('/proc/self/cgroup').read_text().split('::', 1)[-1].strip()
    cg = Path('/sys/fs/cgroup') / cgpath.lstrip('/')
    limits = {}
    for ancestor in [cg, *cg.parents]:
        if not str(ancestor).startswith('/sys/fs/cgroup'): continue
        for name in ('cpu.max','memory.max','cpuset.cpus.effective'):
            p = ancestor / name
            if p.exists(): limits[str(p)] = p.read_text().strip()
    return {'observed_at': utcnow(), 'hostname': platform.node(), 'python': sys.executable,
            'python_version': sys.version, 'cpu_affinity': affinity,
            'lscpu': command(['lscpu', '-J']), 'memory': mem,
            'cgroup_limits': limits, 'gpus': command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu', '--format=csv']),
            'gpu_processes': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv']),
            'load_average': list(os.getloadavg())}

def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())
