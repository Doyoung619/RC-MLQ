"""Persistent delayed admission for the incremental addon; never edits prior queues."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .m2_addon_lock import DEFAULT_ROOT, NOT_BEFORE, PROJECT, PACKAGE, sha256
from .util import atomic_json, utcnow

GLM_QUEUE = Path('/data1/doyoung/GLM/experiments/continuous_ofat/runs/20261006T000156Z/status.json')
ADV_QUEUE = Path('/data1/doyoung/ADV-AL/Experiment/campaign_APGD_FINAL108_20260922/state/status.json')
LEASE_DIR = Path('/data1/doyoung/ADV-AL/Experiment/campaign_B/host_gpu_leases')


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def predecessor_complete(kind, data):
    """A paused/inactive process is not evidence that its queued work is finished."""
    if kind == 'continuous_ofat':
        p = data.get('physical',{})
        expected = int(p.get('expected',0))
        return (expected > 0 and int(p.get('completed',0)) == expected
                and int(p.get('pending',1)) == 0 and int(p.get('running',1)) == 0
                and int(p.get('failed',0)) == 0)
    if kind == 'real_a':
        jobs = data.get('jobs',{})
        return bool(jobs) and all(row.get('status') in {'succeeded','retired'} for row in jobs.values())
    raise ValueError(f'Unknown predecessor schema: {kind}')


def gate_decision(plan, now=None):
    now = time.time() if now is None else now
    if now < timestamp(plan['not_before_utc']):
        return dict(stage='delayed',eligible=False,
                    seconds_until_queue_admission=timestamp(plan['not_before_utc'])-now)
    pending=[]
    for prior in plan['predecessors']:
        try:
            data=json.loads(Path(prior['status_path']).read_text())
            if not predecessor_complete(prior['kind'],data):
                pending.append(dict(status_path=prior['status_path'],state=data.get('state'),
                                    physical=data.get('physical'),counts=data.get('counts'),
                                    reason='Preexisting work is unfinished, including stopped queues.'))
        except (OSError,ValueError,TypeError) as exc:
            pending.append(dict(status_path=prior['status_path'],reason=repr(exc)))
    return dict(stage='waiting_prior_queue' if pending else 'waiting_resources',
                eligible=not pending,pending_predecessors=pending)


def prepare_queue(root=DEFAULT_ROOT, predecessors=None):
    root=Path(root).resolve()
    path=root/'QUEUE_PLAN.json'
    if path.exists(): return json.loads(path.read_text())
    prior = predecessors if predecessors is not None else [
        dict(kind='continuous_ofat',status_path=str(GLM_QUEUE)),
        dict(kind='real_a',status_path=str(ADV_QUEUE))]
    plan=dict(schema_version=1,prepared_at=utcnow(),not_before_utc=NOT_BEFORE,
              user_delay_minutes=40,delay_anchor_utc='2026-10-06T00:17:46+00:00',
              predecessors=prior,physical_gpus=[0,1,2],forbidden_physical_gpus=[3],
              predecessor_priority='All existing unfinished work precedes this addon; STOPPED is not completion.',
              lease_dir=str(LEASE_DIR),poll_seconds=30,
              command=[str(PROJECT/'.venv/bin/python'),'-m','metaworld_fast.m2_addon_cli',
                       'run','--run-dir',str(root)],cwd=str(PROJECT),
              mutates_existing_queues=False,resource_admission='No compute processes on GPU0/1/2; acquire shared whole-GPU and three slot leases for each GPU.')
    atomic_json(path,plan)
    atomic_json(root/'QUEUE_STATUS.json',dict(updated_at=utcnow(),stage='prepared_not_launched',
                                            not_before_utc=NOT_BEFORE,daemon_running=False))
    return plan


def verify_ready(root):
    root=Path(root)
    path=root/'READY.json'
    if not path.is_file(): return False, 'READY.json absent; correctness gate is not complete'
    data=json.loads(path.read_text())
    if not (data.get('status')=='queue_ready' and data.get('tests_passed') is True):
        return False, 'READY.json is not marked passed/ready'
    expected=data.get('frozen_config_sha256')
    if expected and sha256(root/'frozen_config.yaml')!=expected:
        return False, 'Frozen configuration changed after readiness validation'
    for filename,expected in data.get('source_files',{}).items():
        path=Path(filename)
        if not path.is_absolute():path=PACKAGE/path
        if not path.is_file() or sha256(path)!=expected:
            return False, f'Source changed after readiness validation: {path}'
    return True,None


def _nvidia(args):
    result=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=15,check=True)
    return result.stdout.strip().splitlines()


def resource_snapshot(plan):
    inventory={}
    for line in _nvidia(['--query-gpu=index,uuid','--format=csv,noheader,nounits']):
        index,uuid=[p.strip() for p in line.split(',')]
        inventory[int(index)]=uuid
    allowed=plan['physical_gpus']
    if set(allowed)!={0,1,2}: raise ValueError('Addon GPU authorization must remain exactly physical 0,1,2')
    if not set(allowed)<=set(inventory): raise RuntimeError('Authorized GPUs unavailable')
    uuids={inventory[i] for i in allowed}
    busy=[]
    for line in _nvidia(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits']):
        if line.split(',')[0].strip() in uuids:busy.append(line)
    return inventory,busy


def completion_stage(returncode, run_status):
    """Preserve scientific/operational completion distinctions from the runner."""
    stage=run_status.get('stage')
    if returncode:
        return 'run_failed'
    if stage in {'complete','complete_with_unresolved','stopped_resumable'}:
        return stage
    return 'run_finished_without_completion_status'


def _lock(stack,path):
    stream=stack.enter_context(Path(path).open('a+b'))
    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return stream


def run_queue(root):
    root=Path(root).resolve()
    plan=json.loads((root/'QUEUE_PLAN.json').read_text())
    with ExitStack() as ownership:
        _lock(ownership,root/'queue.lock')
        def status(stage,daemon_running=True,**extra):
            atomic_json(root/'QUEUE_STATUS.json',dict(updated_at=utcnow(),pid=os.getpid(),
                daemon_running=daemon_running,stage=stage,not_before_utc=plan['not_before_utc'],**extra))
        while True:
            # An explicit user clarification may change predecessor selection;
            # reread it without requiring restart of the waiting supervisor.
            plan=json.loads((root/'QUEUE_PLAN.json').read_text())
            if (root/'STOP_QUEUE').exists():
                status('stopped_by_request',daemon_running=False);return 0
            decision=gate_decision(plan)
            if not decision.pop('eligible'):
                stage=decision.pop('stage');status(stage,**decision)
                time.sleep(min(30,plan['poll_seconds']));continue
            ready,reason=verify_ready(root)
            if not ready:
                status('waiting_correctness_gate',reason=reason)
                time.sleep(min(30,plan['poll_seconds']));continue
            try:
                inventory,busy=resource_snapshot(plan)
                if busy:
                    status('waiting_resources',compute_processes=busy)
                    time.sleep(min(30,plan['poll_seconds']));continue
                with ExitStack() as gpu_leases:
                    dispatch=Path(plan['lease_dir'])/'dispatch.lock'
                    with ExitStack() as dispatch_lock:
                        _lock(dispatch_lock,dispatch)
                        fds=[]
                        for gpu in plan['physical_gpus']:
                            uuid=inventory[gpu]
                            for suffix in ['.lock','.slot0.lock','.slot1.lock','.slot2.lock']:
                                stream=_lock(gpu_leases,Path(plan['lease_dir'])/(uuid+suffix))
                                fds.append(stream.fileno())
                        # Recheck after admission locks so a predecessor cannot win a race.
                        if not gate_decision(plan)['eligible']:continue
                        _,busy=resource_snapshot(plan)
                        if busy:continue
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                             NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',M2_ADDON_QUEUE_ADMITTED=str(root))
                    status('running',command=plan['command'],admitted_at=utcnow())
                    with (root/'logs/queued_run.log').open('a',buffering=1) as log:
                        child=subprocess.Popen(plan['command'],cwd=plan['cwd'],env=env,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds))
                        status('running',command=plan['command'],runner_pid=child.pid,admitted_at=utcnow())
                        returncode=child.wait()
                    try:run_status=json.loads((root/'status.json').read_text())
                    except (OSError,ValueError):run_status={}
                    status(completion_stage(returncode,run_status),daemon_running=False,
                           returncode=returncode,runner_stage=run_status.get('stage'))
                    return returncode
            except BlockingIOError:
                status('waiting_resources',reason='Another queue holds a shared GPU or dispatch lease')
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                status('waiting_resources',reason=repr(exc))
            time.sleep(min(30,plan['poll_seconds']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('--run-dir',default=str(DEFAULT_ROOT))
    args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare_queue(args.run_dir),indent=2))
    elif args.action=='run':sys.exit(run_queue(args.run_dir))
    else:
        plan=json.loads((Path(args.run_dir)/'QUEUE_PLAN.json').read_text())
        print(json.dumps(gate_decision(plan),indent=2))
