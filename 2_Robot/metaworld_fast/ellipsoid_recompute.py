"""Reevaluate an ellipsoidal information reference on unchanged saved histories."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import contextlib
import fcntl
import hashlib
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
from .util import atomic_json, atomic_text, digest, hardware, source_manifest, utcnow

STOP = threading.Event()

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def default_protocol():
    return dict(alpha=.5, radius=16., starts=8, steps=160, learning_rate=.002,
                marginal_samples=4096, batch_size=16, root_seed=20261005,
                dtype='float64', gpu_ids=[0,1,2,3], global_optimum_certified=False,
                original_feasible_domain='Questions contained in a radius32 ball; E^M is a subset since rhoE16<=32.',
                validation_settings=['m04_b04','m16_b04','m04_b16'],
                validation_interactions=[0,5], validation_t=[0,64,127])

def prepare(source, manuscript, root, protocol):
    import numpy as np
    root,source=Path(root).resolve(),Path(source).resolve()
    if root==source:raise ValueError('A new evaluation output directory is required')
    for name in ('evaluated','checkpoints','logs','validation','figures','tables','source_snapshot'):
        (root/name).mkdir(parents=True,exist_ok=True)
    paths=sorted((source/'statistical').glob('*.npz'))
    if len(paths)!=420:raise ValueError(f'Expected the complete ten-seed dataset; found{len(paths)}')
    source_hashes={p.name:file_sha(p) for npz in paths for p in (npz,npz.with_suffix('.json'))}
    state=dict(source_run=str(source),manuscript_sha256=file_sha(manuscript),
               protocol=protocol,source_artifacts_sha256=source_hashes,
               evaluation_source_hash=digest(source_manifest()))
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for key,value in state.items():
            if previous[key]!=value:raise RuntimeError(f'Frozen reevaluation changed:{key}; use a new output directory')
        return previous
    d=4;max_m=16;rmax=2.
    required_radius=np.sqrt(d)+(1+np.sqrt(2*max_m-1))*rmax
    if not 0<protocol['alpha']<=1 or not required_radius<=protocol['radius']<=32:
        raise ValueError('Invalid common reference geometry or original-domain containment')
    state.update(created_at=utcnow(),expected_runs=420,expected_preanswer_histories=53760,
                 required_minimum_radius=float(required_radius),actual_answers_generated=0,
                 simulator_rollouts_generated=0,
                 statement='Achieved numerical reference from full-dimensional constrained multistart optimization; not a certified global maximum.')
    atomic_json(manifest_path,state);atomic_json(root/'oracle_config.json',protocol)
    atomic_json(root/'HARDWARE.json',hardware())
    shutil.copy2(manuscript,root/'manuscript_source.txt')
    for rel in source_manifest():
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/rel,target)
    return state

def _optimizer_inputs(snapshots):
    import numpy as np
    return dict(points=np.stack([s.points for s in snapshots]),weights=np.stack([s.weights for s in snapshots]),
                mu=np.stack([s.mu for s in snapshots]),cov=np.stack([s.cov for s in snapshots]),
                beta=snapshots[0].beta,actual_q=np.stack([s.actual_q for s in snapshots]),
                grid_q=np.stack([s.grid_q for s in snapshots]))

def optimize_batch(snapshots,protocol,device,seed_key,*,warm_q=None,refined=False):
    import numpy as np
    from .ellipsoid_optimizer import optimize_ellipsoid
    from .model import stable_seed
    seeds=[stable_seed(protocol['root_seed'],seed_key,s.t,s.method,'ellipsoid_search') for s in snapshots]
    return optimize_ellipsoid(**_optimizer_inputs(snapshots),device=device,
                              alpha=protocol['alpha'],radius=protocol['radius'],
                              starts=protocol['starts']*(2 if refined else 1),
                              steps=protocol['steps']*(2 if refined else 1),
                              learning_rate=protocol['learning_rate'],seed=np.asarray(seeds,dtype=np.uint64),warm_q=warm_q)

def completed_artifact(path):
    path=Path(path)
    return path.exists() and path.with_suffix('.json').exists() and json.loads(path.with_suffix('.json').read_text()).get('complete',False)

def evaluate_history(path,root,protocol,device,rounds=None):
    import numpy as np
    from .ellipsoid_posterior import iter_posterior_snapshots
    from .statistical_runner import _atomic_npz
    path,root=Path(path),Path(root);key=path.stem
    output=root/'evaluated'/f'{key}.npz'
    if completed_artifact(output):return json.loads(output.with_suffix('.json').read_text())
    partial=root/'checkpoints'/f'{key}.npz'
    data={};done=set();previous_elapsed=0.
    if partial.exists():
        with np.load(partial) as z:data={k:z[k].copy() for k in z.files}
        done=set(map(int,data['t']))
        if partial.with_suffix('.json').exists():
            previous_elapsed=json.loads(partial.with_suffix('.json').read_text()).get('elapsed_seconds',0.)
    requested=[t for t in (range(128) if rounds is None else rounds) if t not in done]
    with np.load(path) as z:original_eig=z['eig'].copy()
    started=time.perf_counter();batch=[];iterations=0
    def consume(snapshots):
        nonlocal data,iterations
        result=optimize_batch(snapshots,protocol,device,key)
        t=np.array([s.t for s in snapshots])
        actual=np.atleast_1d(result['actual_eig'])
        achieved=np.atleast_1d(result['achieved_eig'])
        upper=np.atleast_1d(result['upper_bound_estimate'])
        grid=np.max(np.atleast_2d(result['grid_eigs']),axis=-1)
        ratio=actual/achieved
        if not np.all(np.isfinite(ratio)) or np.any(achieved<grid-1e-11):
            raise FloatingPointError('Invalid achieved reference or missing grid witness')
        inside=np.atleast_1d(result['actual_inside'])
        if np.any(ratio[inside]>1+1e-9):raise FloatingPointError('Feasible actual question exceeds achieved witness set')
        chunk=dict(t=t,eig_actual=actual,eig_grid=grid,eig_achieved=achieved,eig_upper=upper,
                   eta_ellipsoid_approx=ratio,eta_grid_recomputed=actual/grid,
                   eta_lower_plugin=actual/upper,eta_upper_plugin=ratio,
                   achieved_q=np.asarray(result['best_q']),actual_inside=inside,
                   max_normz_actual=np.atleast_1d(result['actual_max_normalized_radius']),
                   max_normz_achieved=np.atleast_1d(result['best_max_normalized_radius']),
                   best_step=np.atleast_1d(result['best_step']),
                   best_eig_trace=np.asarray(result['best_eig_trace']),
                   initialization_seed=np.atleast_1d(result['initialization_seed']),
                   optimization_gain=achieved-np.atleast_1d(result['initial_best_eig']),
                   integration_original_actual_discrepancy=actual-original_eig[t])
        for name,value in chunk.items():
            data[name]=np.concatenate([data[name],value],axis=0) if name in data else value
        order=np.argsort(data['t']);data={k:v[order] for k,v in data.items()}
        _atomic_npz(partial,data)
        iterations+=1
        atomic_json(root/'checkpoints'/f'{key}.json',dict(run_key=key,rounds_complete=len(data['t']),
                    elapsed_seconds=previous_elapsed+time.perf_counter()-started,device=device,updated_at=utcnow()))
    replay=iter_posterior_snapshots(path,marginal_samples=protocol['marginal_samples'],device=device,rounds=requested)
    for snapshot in replay:
        batch.append(snapshot)
        if len(batch)>=protocol['batch_size']:
            consume(batch);batch=[]
            if STOP.is_set():break
    if batch:consume(batch)
    expected=set(range(128) if rounds is None else rounds)
    complete=set(map(int,data.get('t',[])))==expected
    summary=dict(run_key=key,rounds=len(data.get('t',[])),elapsed_seconds=previous_elapsed+time.perf_counter()-started,
                 source=str(path),source_sha256=file_sha(path),device=device,
                 complete=complete,global_optimum_certified=False,protocol=protocol,
                 posterior_replay=replay.summary())
    if complete:
        _atomic_npz(output,data);atomic_json(output.with_suffix('.json'),summary)
    return summary

def worker(path,root,protocol,device,rounds=None):
    logfile=Path(root)/'logs'/f'{Path(path).stem}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:return evaluate_history(path,root,protocol,device,rounds)
        except Exception:traceback.print_exc();raise

def status(root,stage,started,profile=None,error=None):
    root=Path(root)
    complete={p.stem:json.loads(p.read_text()) for p in (root/'evaluated').glob('*.json')}
    rounds=sum(r['rounds'] for r in complete.values())
    for p in (root/'checkpoints').glob('*.json'):
        if p.stem not in complete:rounds+=json.loads(p.read_text()).get('rounds_complete',0)
    remaining=None
    if profile:
        remaining=profile['seconds_per_history']*(53760-rounds)/len(profile['gpu_ids'])*1.2+profile['validation_reserve_seconds']
    if stage.startswith('complete'):remaining=0
    if stage=='figures_and_report':remaining=60
    result=dict(updated_at=utcnow(),stage=stage,pid=os.getpid(),runs_complete=len(complete),runs_total=420,
                preanswer_histories_complete=rounds,preanswer_histories_total=53760,
                eta_remaining_seconds=remaining,elapsed_seconds=time.time()-started,error=error,
                new_actual_answers=0,new_simulator_rollouts=0)
    atomic_json(root/'status.json',result)
    atomic_text(root/'STATUS.md',f'# Ellipsoidal information reevaluation\n\nStage:{stage}, PID:{os.getpid()}.\n\n'
                f'Histories:{rounds:,}/53,760, complete runs:{len(complete)}/420. No new preference answers or robot executions.\n\nError:{error}\n')
    estimate='Profiling in progress' if remaining is None else f'{remaining/60:.1f} minutes (planning range {remaining*.8/60:.1f}–{remaining*1.5/60:.1f})'
    atomic_text(root/'ETA.md',f'# Measured ETA\n\n{estimate}\n\nUpdated:{utcnow()}\n')
    print(f'{utcnow()} stage={stage} histories={rounds}/53760 runs={len(complete)}/420 ETA={estimate}',flush=True)
    return result

def profile_work(source,root,protocol):
    ctx=mp.get_context('spawn');rows=[]
    # These histories include the hardest M and beta, early/middle/late posteriors.
    cases=[f'{s}__{m}__i0' for s in ('m04_b04','m16_b04','m04_b16') for m in ('RC','SG')]
    folder=Path(root)/'profiling'
    for name in ('evaluated','checkpoints','logs'):(folder/name).mkdir(parents=True,exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=1,mp_context=ctx) as pool:
        for case in cases:
            result=pool.submit(worker,str(Path(source)/'statistical'/f'{case}.npz'),str(folder),protocol,'cuda:3').result()
            rows.append(result);print(f'PROFILE {case}: {result["elapsed_seconds"]:.3f}s for128 histories and frozen replay',flush=True)
    seconds_per_history=sum(r['elapsed_seconds'] for r in rows)/sum(r['rounds'] for r in rows)
    profile=dict(measured_at=utcnow(),cases=rows,seconds_per_history=seconds_per_history,
                 gpu_ids=protocol['gpu_ids'],validation_reserve_seconds=240.,
                 estimate_note='Measured full128-round pilots include frozen replay and optimization; will refine from completed runs.')
    atomic_json(Path(root)/'PROFILE.json',profile)
    return profile

def run(source,manuscript,root,protocol):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_text(root/'supervisor.pid',f'{os.getpid()}\n')
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    started=time.time();profile=None
    try:
        manifest=prepare(source,manuscript,root,protocol)
        status(root,'tests',started)
        with (root/'logs/tests.log').open('w') as log:
            tests=subprocess.run([sys.executable,'-m','pytest','-q','metaworld_fast/tests/test_ellipsoid_optimizer.py',
                                  'metaworld_fast/tests/test_ellipsoid_posterior.py','metaworld_fast/tests/test_ellipsoid_reporting.py',
                                  'metaworld_fast/tests/test_ellipsoid_validation.py'],stdout=log,stderr=subprocess.STDOUT)
        if tests.returncode:raise RuntimeError('Ellipsoid validation tests failed; see logs/tests.log')
        status(root,'profiling',started)
        profile_path=root/'PROFILE.json'
        profile=json.loads(profile_path.read_text()) if profile_path.exists() else profile_work(source,root,protocol)
        paths=sorted((Path(source)/'statistical').glob('*.npz'))
        paths.sort(key=lambda p:(-int(p.stem.split('_b')[0][1:]),p.name))
        pending=[p for p in paths if not completed_artifact(root/'evaluated'/p.name)]
        ctx=mp.get_context('spawn')
        pools={g:cf.ProcessPoolExecutor(max_workers=1,mp_context=ctx) for g in protocol['gpu_ids']}
        active={};last=0
        try:
            while pending or active:
                for gpu,pool in pools.items():
                    if not STOP.is_set() and pending and gpu not in active:
                        path=pending.pop(0);active[gpu]=pool.submit(worker,str(path),str(root),protocol,f'cuda:{gpu}')
                for gpu in list(active):
                    if active[gpu].done():
                        result=active.pop(gpu).result();print(f'Finished {result["run_key"]}: {result["elapsed_seconds"]:.1f}s',flush=True)
                if time.monotonic()-last>15:
                    completed=[json.loads(p.read_text()) for p in (root/'evaluated').glob('*.json')]
                    if len(completed)>=4:
                        profile['seconds_per_history']=sum(r['elapsed_seconds'] for r in completed)/sum(r['rounds'] for r in completed)
                        atomic_json(root/'PROFILE.json',profile)
                    status(root,'recomputing',started,profile);last=time.monotonic()
                if STOP.is_set() and not active:break
                time.sleep(.2)
        finally:
            for pool in pools.values():pool.shutdown(wait=True,cancel_futures=True)
        if STOP.is_set():status(root,'stopped_resumable',started,profile);return
        status(root,'numerical_validation',started,profile)
        from .ellipsoid_validation import validate_evaluation
        validation=validate_evaluation(root)
        status(root,'figures_and_report',started,profile)
        from .ellipsoid_reporting import generate_ellipsoid_report
        generate_ellipsoid_report(root)
        if any(file_sha(Path(source)/'statistical'/name)!=h for name,h in manifest['source_artifacts_sha256'].items()):
            raise RuntimeError('Original source data changed during reevaluation')
        status(root,'complete_with_numerical_approximation',started,profile)
    except Exception as exc:
        traceback.print_exc();status(root,'failed',started,profile,error=str(exc));raise
    finally:lock.close()

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--manuscript',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--protocol',help='Frozen engineering protocol JSON; defaults follow the supplied revision')
    args=parser.parse_args();protocol=json.loads(Path(args.protocol).read_text()) if args.protocol else default_protocol()
    run(args.source,args.manuscript,args.run_dir,protocol)

if __name__=='__main__':main()
