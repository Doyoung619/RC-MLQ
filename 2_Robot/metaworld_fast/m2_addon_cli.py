"""Inspect, test, queue, resume and report the incremental binary addon."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Set isolation before importing numerical/CUDA libraries in this process.
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from .util import atomic_json, atomic_text, load_config, utcnow


def run_tests(root):
    from .m2_addon_runner import source_code_inventory
    root = Path(root)
    (root/'tests').mkdir(parents=True, exist_ok=True)
    before = source_code_inventory()
    paths = sorted(str(p.relative_to(Path.cwd())) for p in (Path(__file__).parent/'tests').glob('test_m2_addon_*.py'))
    paths.append('metaworld_fast/tests/test_statistics.py')
    paths.append('metaworld_fast/tests/test_robot.py')
    command = [sys.executable,'-m','pytest','-q',*paths,'-k',
               'not adaptive_marginal_doubled_resolution_frozen_history']
    log = root/'tests'/'correctness.log'
    started = time.time()
    with log.open('w') as stream:
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                stdout=stream, stderr=subprocess.STDOUT)
    after = source_code_inventory()
    record = dict(status='passed' if result.returncode==0 and before==after else 'failed',
                  returncode=result.returncode, command=command, log=str(log),
                  source_files=after, source_stable_during_tests=before==after,
                  cuda_visible_devices='', production_answers_generated=0,
                  completed_at=utcnow(), elapsed_seconds=time.time()-started)
    atomic_json(root/'tests'/'correctness.json', record)
    if record['status'] != 'passed':
        raise RuntimeError(f'Correctness checks failed or source changed during tests: {log}')
    return record


def mark_ready(root):
    from .m2_addon_runner import source_code_inventory
    from .m2_addon_lock import verify_lock
    root = Path(root)
    record = json.loads((root/'tests'/'correctness.json').read_text())
    code = source_code_inventory()
    if record['status'] != 'passed' or record['source_files'] != code:
        raise RuntimeError('Run tests against the final code before freezing READY')
    checked = verify_lock(root)
    if not checked['unchanged']:
        raise RuntimeError('Baseline artifacts changed')
    # Snapshot executable implementation, tests and the exact task specification.
    package = Path(__file__).resolve().parent
    for name in code:
        target = root/'source_snapshot'/'metaworld_fast'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package/name, target)
    for name in ('metaworld_m2_addon_codex_prompt.md','metaworld_m2_addon_codex_prompt.txt'):
        shutil.copy2(package.parent/name, root/'source_provenance'/name)
    config_path = root/'frozen_config.yaml'
    ready = dict(status='queue_ready', tests_passed=True, source_files=code,
                 frozen_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 frozen_at=utcnow(), correctness_log=record['log'],
                 baseline_inventory_unchanged=True, production_started=False,
                 gpu_numerical_equivalence='required_after_queue_admission_before_production',
                 not_before_utc=load_config(config_path)['addon']['not_before_utc'])
    atomic_json(root/'READY.json', ready)
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(status='queue_ready',source_files=code,code_hash=hashlib.sha256(
        json.dumps(code,sort_keys=True).encode()).hexdigest(),readiness=str(root/'READY.json'))
    atomic_json(manifest_path,manifest)
    return ready


def require_admission(root):
    config = load_config(Path(root)/'frozen_config.yaml')
    not_before = datetime.fromisoformat(config['addon']['not_before_utc'])
    if datetime.now(timezone.utc) < not_before:
        raise RuntimeError(f'Production cannot begin before {not_before.isoformat()}')
    # The queue daemon sets this only while holding shared host GPU leases and
    # after the predecessor-completion gate. A direct CLI cannot bypass it.
    if os.environ.get('M2_ADDON_QUEUE_ADMITTED') != str(Path(root).resolve()):
        raise RuntimeError('Start/resume through m2_addon_queue; predecessor/lease admission required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect','plan','prepare','test','ready','profile','run','resume','validate','plot','report','videos'])
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if args.action in ('inspect','plan','prepare'):
        from .m2_addon_lock import prepare_lock, discover
        if args.action=='inspect':
            base, ellipsoid, config = discover()
            result=dict(source_run=str(base),ellipsoid_source_run=str(ellipsoid),
                        interactions=config['experiment']['interaction_ids'])
        else:
            prepare_lock(root)
            result=json.loads((root/'REUSE_PLAN.json').read_text())['counts']
    elif args.action=='test':
        result=run_tests(root)
    elif args.action=='ready':
        result=mark_ready(root)
    elif args.action in ('run','resume','profile'):
        require_admission(root)
        from .m2_addon_runner import run_pipeline, profile, check_ready
        check_ready(root)
        result = profile(root) if args.action=='profile' else run_pipeline(root)
    elif args.action=='validate':
        require_admission(root)
        from .m2_addon_validation import validate_run
        result=validate_run(root, device='cuda:0', numerical=True)
    elif args.action in ('plot','report'):
        from .m2_addon_reporting import generate_report
        result=generate_report(root)
    else:
        require_admission(root)
        from .m2_addon_runner import render_videos
        result=render_videos(root)
    # Do not print full inventories or entire validation/plot arrays.
    if isinstance(result, dict):
        result={k:v for k,v in result.items() if k not in ('source_files','clips','command','source_code','profiles')}
    print(json.dumps(result,default=str), flush=True)
    if args.action == 'videos' and (result.get('failed', 1) or result.get('planned') != 18):
        raise SystemExit(1)


if __name__=='__main__':
    main()
