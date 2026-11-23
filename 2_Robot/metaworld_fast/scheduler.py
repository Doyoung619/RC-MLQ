"""Detached, resumable statistical producer and concurrent realization consumer."""
from __future__ import annotations
import concurrent.futures as cf
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from .util import atomic_json, atomic_text, digest, hardware, source_manifest, utcnow

STOP = False

def stop_handler(signum, frame):
    global STOP
    STOP = True
    print('Stop requested: draining active runs to durable checkpoints.', flush=True)

def job_id(job):
    return f"{job['setting_id']}__{job['method']}__i{job['interaction_id']}"

def jobs_from_config(config):
    e = config['experiment']
    jobs = []
    # M=16 first so expensive jobs cannot all pile up at the tail.
    settings = sorted(e['settings'], key=lambda s: (-s['choices'], -s['beta']))
    for interaction in e['interaction_ids']:
        for setting in settings:
            for method in e['methods']:
                jobs.append(dict(setting_id=setting['id'], M=setting['choices'], beta=setting['beta'], method=method,
                                 interaction_id=interaction, root_seed=e['root_seed'], answers=e['answers_per_run'],
                                 particles=config['inference']['smc_particles'], dimension=e['dimension']))
    return jobs

def statistical_worker(job, run_dir):
    from .statistical_runner import run_statistical
    logfile = Path(run_dir) / 'logs' / (job_id(job) + '.log')
    with logfile.open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f'{utcnow()} start {job}', flush=True)
        try:
            result = run_statistical(job, run_dir, resume=True)
            print(f'{utcnow()} finished {result}', flush=True)
            return result
        except Exception:
            traceback.print_exc()
            raise

def make_requests(path, run_dir):
    import numpy as np
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        q = z['q'].copy()
        refs = z['reference_queries'].copy()
    sid, method, interaction = path.stem.split('__')
    interaction = int(interaction.removeprefix('i'))
    requests = [dict(request_id=f'{path.stem}__t{t:03d}', context_id=interaction,
                     q=q[t], kind='selected', setting_id=sid, method=method,
                     interaction_id=interaction, t=t) for t in range(len(q))]
    if interaction == 0:
        for t in (0,127):
            for r in range(7):
                requests.append(dict(request_id=f'{path.stem}__reference_t{t:03d}_s{r}', context_id=interaction,
                                     q=refs[t,r], kind='reference', setting_id=sid, method=method,
                                     interaction_id=interaction, t=t, scale_index=r))
    atomic_json(Path(run_dir)/'rollout_requests'/f'{path.stem}.json',
                {'status':'pending','source':str(path), 'requests':requests})
    return requests

def rollout_one(pool, path, run_dir):
    path = Path(path)
    requests = make_requests(path, run_dir)
    start = time.monotonic()
    result = pool.execute_queries(requests, Path(run_dir)/'rollouts'/path.stem)
    summary = dict(statistical_run_id=path.stem, selected_logical=sum(len(r['q']) for r in requests if r['kind']=='selected'),
                   reference_logical=sum(len(r['q']) for r in requests if r['kind']=='reference'),
                   wall_seconds=time.monotonic()-start, result=result, completed_at=utcnow())
    atomic_json(Path(run_dir)/'rollouts'/(path.stem+'.done.json'),summary)
    return summary

def read_progress(run_dir):
    progress = []
    for path in (Path(run_dir)/'checkpoints').glob('*.json'):
        try: progress.append(json.loads(path.read_text()))
        except (ValueError, OSError): pass
    return progress

def status_snapshot(run_dir, stage, started, profile=None, error=None):
    import numpy as np
    root = Path(run_dir)
    stats = list((root/'statistical').glob('*.npz'))
    answers = 0
    completed_ids = set()
    for path in stats:
        try:
            with np.load(path, allow_pickle=False) as z:
                n = len(z['y']); answers += n
                if n==128: completed_ids.add(path.stem)
        except (OSError,ValueError,KeyError): continue
    pending_answers = 0
    for p in read_progress(root):
        if p.get('run_key',p.get('run_id',p.get('job_id'))) not in completed_ids:
            pending_answers += int(p.get('answers_completed',p.get('completed_answers',0)))
    done=[]
    for p in (root/'rollouts').glob('*.done.json'):
        try: done.append(json.loads(p.read_text()))
        except (ValueError,OSError): pass
    logical = sum(x.get('selected_logical',0) for x in done)
    elapsed = time.time()-started
    profile = profile or {}
    expected = profile.get('estimated_pipeline_seconds')
    remaining = None
    live={}
    if expected:
        # Conservative task-progress estimate. Validation/videos have their own reserve.
        stat_fraction=min(1.,(answers+pending_answers)/26880)
        rollout_fraction=min(1.,logical/161280)
        remaining=max(profile.get('estimated_statistical_seconds',0)*(1-stat_fraction),
                      profile.get('estimated_rollout_seconds',0)*(1-rollout_fraction))
        remaining+=profile.get('estimated_postprocessing_seconds',0)
        # Full-run durations include competition from active rollout consumers.
        # Use completed interaction timings, with three prior profile samples
        # to avoid a single fast interaction dominating the early ETA.
        groups={'joint':[],'marginal':[]}
        for p in (root/'statistical').glob('*.json'):
            try:
                meta=json.loads(p.read_text())
                if meta.get('answers_completed')==128:
                    kind='joint' if meta['method'] in ('PS','GC','SG') else 'marginal'
                    groups[kind].append(float(meta['elapsed_seconds']))
            except (ValueError,KeyError,OSError):pass
        if stage=='inference_and_rollouts' and groups['joint']:
            resources=profile['resources']
            parts=[]
            for kind,workers in [('joint',resources['joint_workers']),('marginal',resources['marginal_workers'])]:
                observed=groups[kind]
                prior=[r['wall_seconds'] for r in profile['statistical_profiles'] if (r['job']['method']=='SG')==(kind=='joint')]
                baseline=sum(prior)/len(prior)
                average=(sum(observed)+3*baseline)/(len(observed)+3)
                parts.append(average*max(0,105-len(observed))/workers)
                live[f'{kind}_mean_completed_run_seconds']=sum(observed)/len(observed) if observed else None
                live[f'{kind}_completed_runs']=len(observed)
            inference_remaining=max(parts)
            wall=sum(item.get('wall_seconds',0) for item in done)
            total_logical=sum(item.get('selected_logical',0)+item.get('reference_logical',0) for item in done)
            if wall and total_logical:
                rate=total_logical/wall
                rollout_remaining=max(0,164808-total_logical)/rate
                live['logical_rollout_clips_per_second_including_cache_and_storage']=rate
            else: rollout_remaining=profile['estimated_rollout_seconds']*(1-rollout_fraction)
            remaining=max(inference_remaining,rollout_remaining)*1.10+profile.get('estimated_postprocessing_seconds',0)
            live['measurement_scope']='Completed final interactions and realization shards while both producers and consumers run; 10 percent remaining-work margin.'
        if stage.startswith('complete'): remaining=0.
    if remaining is not None and stage=='numerical_validation':
        vp=root/'validation'/'progress.json'
        if vp.exists():
            try:
                progress=json.loads(vp.read_text());n=progress.get('completed',0)
                if n: remaining=progress.get('elapsed_seconds',0)/n*max(0,18-n)+180
            except (ValueError,OSError):pass
    if stage=='figures_and_videos': remaining=180.
    data=dict(updated_at=utcnow(),stage=stage,pid=os.getpid(),elapsed_seconds=elapsed,
              statistical_runs_complete=len(completed_ids),statistical_runs_total=210,
              answers_complete=answers+pending_answers,answers_total=26880,
              selected_rollout_logical_completed=logical,selected_rollout_logical_total=161280,
              rollout_run_shards_completed=len(done),eta_remaining_seconds=remaining,live_measurements=live,error=error)
    if remaining is not None:
        data['estimated_finish_new_york']=(datetime.now(ZoneInfo('America/New_York'))+timedelta(seconds=remaining)).isoformat()
    atomic_json(root/'status.json',data)
    text=f"# Run status\n\nUpdated: {data['updated_at']}\n\nStage: **{stage}**; PID {os.getpid()}.\n\nStatistical runs: {len(completed_ids)}/210. Answers: {answers+pending_answers}/26,880.\n\nSelected logical realization requests completed: {logical:,}/161,280.\n"
    if error: text+=f'\nError: `{error}`\n'
    atomic_text(root/'STATUS.md',text)
    eta='# Measured ETA\n\n'
    if remaining is None: eta+='Pending actual throughput measurements; no hardware-only estimate is asserted.\n'
    else:
        eta+=f"Estimated remaining: **{remaining/60:.1f} minutes** (planning range {remaining*.7/60:.1f}–{remaining*1.5/60:.1f} minutes). Approximate finish: {data['estimated_finish_new_york']} (America/New_York).\n\n"
        eta+='Estimate uses local profiling, full selected workloads and validation/video reserve. Statistical inference and realization overlap. Early estimates remain uncertain and are updated from completed work.\n'
        eta+=f"\nProfiling details: `PROFILE.json`. Current stage: {stage}. Initial compute estimate: {(expected or 0)/3600:.2f} hours.\n"
    atomic_text(root/'ETA.md',eta)
    estimate='측정 중' if remaining is None else f'{remaining/60:.1f}분'
    print(f"{data['updated_at']} stage={stage} runs={len(completed_ids)}/210 answers={answers+pending_answers}/26880 rollouts={logical}/161280 ETA={estimate}",flush=True)
    return data

@contextlib.contextmanager
def heartbeat(run_dir,stage,started,profile):
    stop=threading.Event()
    def tick():
        while not stop.wait(15):
            try:status_snapshot(run_dir,stage,started,profile)
            except Exception:traceback.print_exc()
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join()

def prepare(run_dir, config):
    from .artifact_reuse import inspect_reuse
    import yaml
    root=Path(run_dir)
    for folder in ('logs','checkpoints','statistical','rollout_requests','rollouts','diagnostics','validation','figures','videos','tables','contexts'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    source=source_manifest(); config_hash=digest(config); code_hash=digest(source)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if m['config_hash']!=config_hash or m['code_hash']!=code_hash:
            raise RuntimeError('Config/code differs from frozen manifest; use a new run directory.')
        return m
    m=dict(created_at=utcnow(),config_hash=config_hash,code_hash=code_hash,source_files=source,
           reuse=inspect_reuse(),statistical_jobs=jobs_from_config(config),
           physical_context_ids=[0,1,2,3,4],validation_selection={'interaction_ids':[0],'settings':['m04_b04','m16_b04','m04_b16']},
           expected_counts={'runs':210,'answers':26880,'selected_candidate_requests':161280,'selected_environment_action_steps':8064000},
           runtime_environment_note='Task-isolated venv with read-only numeric packages from existing ADV-AL environment; pinned simulator installed locally.')
    atomic_json(manifest_path,m)
    atomic_text(root/'config_frozen.yaml',yaml.safe_dump(config,sort_keys=False))
    atomic_json(root/'HARDWARE.json',hardware())
    env=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    atomic_text(root/'ENVIRONMENT.txt',f'Python: {sys.executable}\n{sys.version}\n\n'+env.stdout)
    src=root/'source_snapshot'
    for rel in source:
        dst=src/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/rel,dst)
    return m

def run_pipeline(config, run_dir):
    global STOP
    root=Path(run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Run already has an active supervisor')
    atomic_text(root/'supervisor.pid',str(os.getpid())+'\n')
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    started=time.time();profile=None
    try:
        # Prepare directories before testing; freeze statistical settings before outcomes.
        manifest=prepare(root,config)
        status_snapshot(root,'tests',started)
        testlog=root/'logs'/'tests.log'
        with testlog.open('w') as log:
            tested=subprocess.run([sys.executable,'-m','pytest','-q','metaworld_fast/tests'],stdout=log,stderr=subprocess.STDOUT)
        if tested.returncode: raise RuntimeError(f'Correctness tests failed; inspect {testlog}')
        status_snapshot(root,'profiling',started)
        from .profiling import profile_workload
        profile=profile_workload(root,config)
        atomic_json(root/'PROFILE.json',profile)
        atomic_json(root/'resources_frozen.json',profile['resources'])
        status_snapshot(root,'inference_and_rollouts',started,profile)
        resources=profile['resources']
        from .rollout_pool import RolloutPool
        ctx=mp.get_context('spawn')
        marginal_pool=cf.ProcessPoolExecutor(max_workers=resources['marginal_workers'],mp_context=ctx)
        joint_pool=cf.ProcessPoolExecutor(max_workers=resources['joint_workers'],mp_context=ctx)
        rollout_thread=cf.ThreadPoolExecutor(max_workers=1)
        pool=RolloutPool(str(root/'contexts'),workers=resources['rollout_workers'])
        try:
            pending_marginal=[];pending_joint=[]
            jobs=jobs_from_config(config)
            for job in jobs:
                job.update(code_hash=manifest['code_hash'],config_hash=manifest['config_hash'])
                job['device']=resources['joint_device'] if job['method'] in ('PS','GC','SG') else 'cpu'
                if (root/'statistical'/(job_id(job)+'.npz')).exists(): continue
                (pending_joint if job['method'] in ('PS','GC','SG') else pending_marginal).append(job)
            active={};rollout_future=None;rollout_known=set()
            for p in (root/'rollouts').glob('*.done.json'): rollout_known.add(p.name.removesuffix('.done.json'))
            last_status=0
            while pending_marginal or pending_joint or active or rollout_future is not None or any(p.stem not in rollout_known for p in (root/'statistical').glob('*.npz')):
                if not STOP:
                    for pending,executor,limit,kind in ((pending_marginal,marginal_pool,resources['marginal_workers'],'marginal'),(pending_joint,joint_pool,resources['joint_workers'],'joint')):
                        while pending and sum(v[0]==kind for v in active.values())<limit:
                            job=pending.pop(0); f=executor.submit(statistical_worker,job,str(root));active[f]=(kind,job)
                for f in list(active):
                    if f.done():
                        _,job=active.pop(f);f.result()
                        print(f'{utcnow()} statistical completed {job_id(job)}',flush=True)
                if rollout_future is not None and rollout_future.done():
                    result=rollout_future.result();print(f"{utcnow()} realization completed {result['statistical_run_id']}",flush=True);rollout_future=None
                if rollout_future is None and not STOP:
                    paths=sorted(p for p in (root/'statistical').glob('*.npz') if p.stem not in rollout_known)
                    if paths:
                        path=paths[0];rollout_known.add(path.stem)
                        rollout_future=rollout_thread.submit(rollout_one,pool,path,str(root))
                if time.monotonic()-last_status>15:
                    status_snapshot(root,'draining' if STOP else 'inference_and_rollouts',started,profile);last_status=time.monotonic()
                if STOP and not active and rollout_future is None: break
                time.sleep(1)
        finally:
            marginal_pool.shutdown(wait=True,cancel_futures=True);joint_pool.shutdown(wait=True,cancel_futures=True)
            rollout_thread.shutdown(wait=True,cancel_futures=True)
            pool.close()
        if STOP:
            status_snapshot(root,'stopped_resumable',started,profile)
            return
        status_snapshot(root,'numerical_validation',started,profile)
        from .validation import validate_run
        with heartbeat(root,'numerical_validation',started,profile):
            validation=validate_run(str(root),config=config,device=resources['joint_device'])
        atomic_json(root/'validation'/'pipeline_validation.json',validation)
        status_snapshot(root,'figures_and_videos',started,profile)
        from .renderer import render_qualitative_videos
        with heartbeat(root,'figures_and_videos',started,profile):
            videos=render_qualitative_videos(root)
            atomic_json(root/'videos'/'pipeline_summary.json',videos)
            from .reporting import generate_report
            report=generate_report(str(root),config=config,make_plots=True)
        stage='complete' if validation.get('status')=='within_reference_uncertainty' and videos.get('status')=='completed' else 'complete_with_unresolved'
        status_snapshot(root,stage,started,profile)
        print(f'{utcnow()} pipeline complete: {root}',flush=True)
    except Exception as exc:
        traceback.print_exc()
        status_snapshot(root,'failed',started,profile,error=str(exc))
        atomic_json(root/'failure.json',dict(time=utcnow(),error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.close()
