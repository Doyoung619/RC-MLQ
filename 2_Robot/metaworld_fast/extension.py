"""Append exactly five interactions, preserving the original five artifacts."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from .util import atomic_json, atomic_text, digest, hardware, load_config, source_manifest, utcnow

OLD_IDS = list(range(5))
NEW_IDS = list(range(5, 10))
CORE_FILES = ('model.py', 'metrics.py', 'line_template.py', 'selectors.py',
              'posterior_marginal.py', 'posterior_smc.py', 'statistical_runner.py',
              'contexts.py', 'renderer.py')
STOP = threading.Event()
STATUS_LOCK = threading.Lock()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extension_config(source):
    config = copy.deepcopy(load_config(Path(source)/'config_frozen.yaml'))
    if config['experiment']['interaction_ids'] != OLD_IDS:
        raise ValueError('Expected the original five interactions 0..4')
    config['experiment']['name'] = 'rcmlq_metaworld_realization_t128_s10'
    config['experiment']['interaction_ids'] = OLD_IDS + NEW_IDS
    config['robot_audit']['context_count'] = 10
    config['output']['confidence_display'] = 'individual_ten_interactions_and_mean'
    config['extension'] = dict(source_run=str(Path(source).resolve()), reused_interaction_ids=OLD_IDS,
                               added_interaction_ids=NEW_IDS,
                               additional_validation_interaction_ids=[5],
                               reason='User requested exactly five more seeds, independent of method rankings.')
    return config

def verify_source(source):
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    current = source_manifest()
    for name in CORE_FILES:
        if current[name] != manifest['source_files'][name]:
            raise RuntimeError(f'Core semantics changed: {name}; refusing mixed experiment')
    summary = json.loads((source/'tables/completion_summary.json').read_text())
    if summary['completed_runs'] != 210 or summary['covered_selected_requests'] != 161280:
        raise RuntimeError('Source five-seed experiment is not complete')
    return manifest

def prepare(source, root):
    import numpy as np
    import yaml
    from .scheduler import jobs_from_config
    from .model import stable_seed
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root: raise ValueError('Use a new output directory to preserve the original run')
    original = verify_source(source)
    config = extension_config(source)
    code = source_manifest()
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['code_hash'] != digest(code) or manifest['config_hash'] != digest(config):
            raise RuntimeError('Frozen extension code/config changed; a new run directory is required')
        if manifest.get('preparation_complete'): return config, manifest
    for folder in ('statistical', 'rollouts', 'rollout_requests', 'validation', 'videos', 'contexts',
                   'diagnostics', 'checkpoints', 'logs', 'figures', 'tables', 'source_snapshot'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    manifest = dict(created_at=utcnow(), code_hash=digest(code), source_files=code,
                    config_hash=digest(config), source_run=str(source),
                    source_manifest_sha256=sha(source/'manifest.json'),
                    source_code_hash=original['code_hash'],
                    reused_interaction_ids=OLD_IDS, added_interaction_ids=NEW_IDS,
                    expected_counts=dict(runs=420, new_runs=210, answers=53760,
                                         selected_candidate_requests=322560,
                                         new_selected_candidate_requests=161280),
                    preparation_complete=False)
    atomic_json(manifest_path, manifest)
    atomic_text(root/'config_frozen.yaml', yaml.safe_dump(config, sort_keys=False))
    old_hashes = {}
    for folder in ('statistical', 'rollouts', 'rollout_requests', 'validation', 'videos'):
        for path in sorted((source/folder).iterdir()):
            if not path.is_file(): continue
            target = root/folder/path.name
            shutil.copy2(path, target)
            # Preserve source metadata and hash; only add explicit reuse provenance.
            if folder == 'statistical' and path.suffix == '.json':
                metadata = json.loads(target.read_text())
                metadata['reused_from'] = str(path.with_suffix('.npz'))
                metadata['source_metadata_sha256'] = sha(path)
                metadata['source_npz_sha256'] = sha(path.with_suffix('.npz'))
                atomic_json(target, metadata)
            if folder == 'statistical' and path.suffix == '.npz':
                old_hashes[path.name] = sha(path)
                if sha(target) != old_hashes[path.name]: raise RuntimeError('Copy checksum mismatch')
    from .contexts import load_contexts, prepare_contexts
    old_contexts = load_contexts(source/'contexts')
    # Helper creates deterministic physical contexts only; no preference interaction
    # is run here. The original 0..4 snapshots remain byte-for-byte authoritative.
    generated = prepare_contexts(root/'context_generation', count=10)
    contexts = old_contexts + [c for c in generated if c['context_id'] in NEW_IDS]
    with (root/'contexts/contexts.pkl.tmp').open('wb') as f: pickle.dump(contexts, f, protocol=5)
    os.replace(root/'contexts/contexts.pkl.tmp', root/'contexts/contexts.pkl')
    atomic_json(root/'contexts/manifest.json', dict(reused_from=str(source/'contexts'),
                context_ids=OLD_IDS+NEW_IDS,
                contexts=[dict(context_id=c['context_id'], context_hash=c['context_hash'], seed=c['seed']) for c in contexts]))
    from .renderer import calibrate
    old_calibration=json.loads((source/'diagnostics/engineering_calibration.json').read_text())
    additional = calibrate([c for c in contexts if c['context_id'] in NEW_IDS],
                           root/'diagnostics/additional_context_calibration.json')
    if additional['physical_interpretation'] != 'finite_audit_passed':
        raise RuntimeError('Additional physical context calibration failed')
    combined = copy.deepcopy(old_calibration)
    for key in ('fixed_command_audit', 'motion_sensitivity', 'restore_order_checks'):
        combined[key] += additional[key]
    combined['reused_context_ids'] = OLD_IDS
    combined['new_context_ids'] = NEW_IDS
    atomic_json(root/'diagnostics/engineering_calibration.json', combined)
    all_jobs = jobs_from_config(config)
    seeds = []
    for job in all_jobs:
        root_seed=config['experiment']['root_seed']
        purpose_seeds={purpose:stable_seed(root_seed, job['setting_id'], job['interaction_id'],
                                         None if purpose=='theta' else job['method'], purpose)
                       for purpose in ('theta','response','selector','smc','diagnostics')}
        theta=np.random.default_rng(purpose_seeds['theta']).uniform(-1,1,4)
        seeds.append(dict(setting_id=job['setting_id'], method=job['method'],
                          interaction_id=job['interaction_id'], seeds=purpose_seeds, theta_star=theta.tolist()))
    atomic_json(root/'seed_manifest.json', seeds)
    atomic_json(root/'reused_statistical_sha256.json', old_hashes)
    resources = json.loads((source/'resources_frozen.json').read_text())
    atomic_json(root/'resources_frozen.json', resources)
    atomic_json(root/'HARDWARE.json', hardware())
    env = subprocess.run([sys.executable,'-m','pip','freeze'], capture_output=True, text=True)
    atomic_text(root/'ENVIRONMENT.txt',f'Python: {sys.executable}\n'+env.stdout)
    previous_profile=json.loads((source/'PROFILE.json').read_text())
    previous_status=json.loads((source/'status.json').read_text())
    durations={kind:[] for kind in ('joint','marginal')}
    for p in (source/'statistical').glob('*.json'):
        record=json.loads(p.read_text())
        durations['joint' if record['method'] in ('PS','GC','SG') else 'marginal'].append(record['elapsed_seconds'])
    profile=dict(measured_source_run=str(source), resources=resources,
                 prior_full_pipeline_seconds=previous_status['elapsed_seconds'],
                 prior_mean_run_seconds={k:sum(v)/len(v) for k,v in durations.items()},
                 prior_validation_seconds=json.loads((source/'validation/summary.json').read_text())['elapsed_seconds'],
                 initial_eta_seconds=previous_status['elapsed_seconds']*1.15,
                 notes=['Measured on the identical workload with seeds0..4; new seeds5..9 only.',
                        'Original statistical and physical artifacts, reference audits, validation results and36 videos reused.',
                        'New seed5 numerical replay and calibration of contexts5..9 added; all ten seeds used in figures.'])
    atomic_json(root/'PROFILE.json',profile)
    shutil.copy2(source/'PROFILE.json', root/'PROFILE.source.json')
    for rel in code:
        dst=root/'source_snapshot'/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/rel,dst)
    for name in ('PROVENANCE.md','LIMITATIONS.md'):
        shutil.copy2(Path(__file__).parent/name,root/name)
    atomic_text(root/'EXTENSION.md',
        f'Exactly five additional interactions: 5,6,7,8,9. Root seed20261004 and statistical/robot definitions unchanged.\n\n'
        f'Original interactions0..4 reused from {source}; NPZ checksums retained. Source output is unmodified.\n\n'
        'Total420 runs,53,760 answers,322,560 selected requests. Exactly210 runs newly computed.\n\n'
        'Same four figure layouts/colors/axes, ten thin curves and their arithmetic mean. No smoothing or seed selection.\n\n'
        'Inherited numerical unresolved cases remain; independently validate first added interaction5 at the same three prespecified settings.\n')
    manifest['preparation_complete']=True
    atomic_json(manifest_path, manifest)
    return config, manifest

def status(root, stage, started, profile=None, error=None):
    with STATUS_LOCK:
        return _status(root, stage, started, profile, error)

def _status(root, stage, started, profile=None, error=None):
    root=Path(root)
    files=list((root/'statistical').glob('*.npz'))
    completed={p.stem for p in files}
    new_completed={name for name in completed if int(name.rsplit('__i',1)[1]) in NEW_IDS}
    answers=len(completed)*128
    for p in (root/'checkpoints').glob('*.progress.json'):
        row=json.loads(p.read_text())
        if row['run_key'] not in completed: answers+=row['answers_completed']
    done=[json.loads(p.read_text()) for p in (root/'rollouts').glob('*.done.json')]
    logical=sum(r['selected_logical'] for r in done)
    new_rollouts=[r for r in done if int(r['statistical_run_id'].rsplit('__i',1)[1]) in NEW_IDS]
    elapsed=time.time()-started
    remaining=max(0,(profile or {}).get('initial_eta_seconds',700)-elapsed)
    if profile and stage=='inference_and_rollouts':
        joint=[p for p in new_completed if p.split('__')[1] in ('PS','GC','SG')]
        # Early estimate uses the previous end-to-end measured wall time;
        # once enough new interactions finish, use current pipeline throughput.
        if len(joint)>=10:
            durations=[]
            for key in joint: durations.append(json.loads((root/'statistical'/f'{key}.json').read_text())['elapsed_seconds'])
            compute=(sum(durations)/len(durations)+.2)*(105-len(joint))
            wall=sum(r['wall_seconds'] for r in new_rollouts)
            count=sum(r['selected_logical'] for r in new_rollouts)
            replay=(161280-count)*wall/count if count else 180
            remaining=max(compute,replay)*1.25+profile['prior_validation_seconds']*1.5+60
    if stage=='numerical_validation':
        progress_file=root/'validation/extension_progress.json'
        if progress_file.exists():
            row=json.loads(progress_file.read_text());n=row['completed']
            remaining=(row['elapsed_seconds']/n*(18-n) if n else 90)+60
        else: remaining=150
    if stage=='figures_and_report':remaining=60
    if stage.startswith('complete'): remaining=0
    data=dict(updated_at=utcnow(),pid=os.getpid(),stage=stage,error=error,elapsed_seconds=elapsed,
              reused_statistical_runs=210,additional_statistical_runs_complete=len(new_completed),
              additional_statistical_runs_total=210,statistical_runs_complete=len(completed),
              statistical_runs_total=420,answers_complete=answers,answers_total=53760,
              selected_rollout_logical_completed=logical,selected_rollout_logical_total=322560,
              eta_remaining_seconds=remaining)
    atomic_json(root/'status.json',data)
    atomic_text(root/'STATUS.md',f'# Ten-seed extension\n\nStage: {stage}. PID: {os.getpid()}.\n\n'
                f'Original210 runs reused. Added {len(new_completed)}/210 runs (IDs5..9). Total {len(completed)}/420.\n\n'
                f'Answers {answers}/53,760. Selected clips {logical}/322,560.\n\nError: {error}\n')
    atomic_text(root/'ETA.md',f'# Measured ETA\n\nRemaining approximately **{remaining/60:.1f} minutes** '
                f'(planning range {remaining*.8/60:.1f}–{remaining*1.5/60:.1f} minutes).\n\n'
                f'Updated {utcnow()}; stage {stage}. Based on original actual608.6-second workload and new completed interactions.\n')
    print(f"{utcnow()} {stage}: new {len(new_completed)}/210, total {len(completed)}/420, clips {logical}/322560, ETA {remaining/60:.1f} min",flush=True)
    return data

def validate_additional(root, config, device):
    from .reporting import load_runs, write_csv
    from .validation import FROZEN_SETTINGS, validate_history
    root=Path(root);start=time.monotonic()
    runs=[r for r in load_runs(root) if r['interaction_id']==5 and r['setting_id'] in FROZEN_SETTINGS]
    old=json.loads((root/'validation/summary.json').read_text())
    results=[];errors=[]
    for r in runs:
        try:
            result=validate_history(r,root,device=device);results.append(result)
            print(f"additional validation {result['identity']}: {result['status']}",flush=True)
        except Exception as exc:
            errors.append(dict(identity=[r['setting_id'],r['method'],5],error=str(exc)))
        atomic_json(root/'validation/extension_progress.json',dict(completed=len(results),errors=errors,elapsed_seconds=time.monotonic()-start))
    unresolved=[r['identity'] for r in results if r['status']=='unresolved']
    # Reload only the prespecified original0 and added5 per-history results.
    all_results=[]
    for r in load_runs(root):
        if r['interaction_id'] in (0,5) and r['setting_id'] in FROZEN_SETTINGS:
            path=root/'validation'/f"{r['setting_id']}_{r['method']}_{r['interaction_id']}.json"
            if path.exists():all_results.append(json.loads(path.read_text()))
    rows=[]
    for r in all_results:
        for comparison in r.get('comparisons',[])+r.get('reference_refinement_comparisons',[]):
            row=dict(setting_id=r['identity'][0],method=r['identity'][1],interaction_id=r['identity'][2],**comparison)
            rows.append({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    write_csv(root/'tables/numerical_discrepancies.csv',rows)
    summary=dict(old)
    summary.update(status='unresolved' if old['status']=='unresolved' or unresolved or errors or len(results)!=18 else 'within_reference_uncertainty',
                   selection_rule='Reused interaction0 checks; added interaction5, all six methods, same three settings, chosen before new outcomes.',
                   selected_histories=36,expected_histories=36,completed_histories=len(all_results),missing_histories=36-len(all_results),
                   unresolved_histories=[r['identity'] for r in all_results if r['status']=='unresolved'],
                   errors=old.get('errors',[])+errors,additional_elapsed_seconds=time.monotonic()-start,
                   reused_validation_histories=18,additional_validation_histories=len(results))
    atomic_json(root/'validation/summary.json',summary)
    return summary

def run(source,root):
    from .scheduler import jobs_from_config, job_id, statistical_worker, rollout_one
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_text(root/'supervisor.pid',f'{os.getpid()}\n')
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    started=time.time();profile=None
    try:
        status(root,'preparation',started)
        config,manifest=prepare(source,root)
        profile=json.loads((root/'PROFILE.json').read_text())
        with (root/'logs/tests.log').open('w') as log:
            checked=subprocess.run([sys.executable,'-m','pytest','-q','metaworld_fast/tests'],stdout=log,stderr=subprocess.STDOUT)
        if checked.returncode:raise RuntimeError('Correctness tests failed; see logs/tests.log')
        resources=profile['resources'];ctx=mp.get_context('spawn')
        from .rollout_pool import RolloutPool
        with cf.ProcessPoolExecutor(max_workers=4,mp_context=ctx) as cpus,cf.ProcessPoolExecutor(max_workers=1,mp_context=ctx) as gpu,cf.ThreadPoolExecutor(max_workers=1) as consumer,RolloutPool(root/'contexts',workers=resources['rollout_workers']) as pool:
            pending={'marginal':[],'joint':[]}
            for job in jobs_from_config(config):
                if job['interaction_id'] not in NEW_IDS:continue
                if (root/'statistical'/f'{job_id(job)}.npz').exists():continue
                kind='joint' if job['method'] in ('PS','GC','SG') else 'marginal'
                job.update(device=resources['joint_device'] if kind=='joint' else 'cpu',code_hash=manifest['code_hash'],config_hash=manifest['config_hash'])
                pending[kind].append(job)
            active={};replay=None
            processed={p.name.removesuffix('.done.json') for p in (root/'rollouts').glob('*.done.json')}
            last_status=0
            while True:
                for kind,executor,count in [('marginal',cpus,4),('joint',gpu,1)]:
                    while not STOP.is_set() and pending[kind] and sum(k==kind for k in active.values())<count:
                        job=pending[kind].pop(0);active[executor.submit(statistical_worker,job,str(root))]=kind
                for future in list(active):
                    if future.done():future.result();active.pop(future)
                if replay is not None and replay.done():
                    result=replay.result()
                    # Update queue descriptor as well as the authoritative candidate records.
                    path=root/'rollout_requests'/f"{result['statistical_run_id']}.json"
                    request=json.loads(path.read_text());request['status']='completed';request['result']=result['result'];atomic_json(path,request)
                    replay=None
                available=sorted(p for p in (root/'statistical').glob('*.npz') if p.stem not in processed)
                if replay is None and available and not STOP.is_set():
                    p=available[0];processed.add(p.stem);replay=consumer.submit(rollout_one,pool,p,str(root))
                if time.monotonic()-last_status>15:
                    status(root,'inference_and_rollouts',started,profile);last_status=time.monotonic()
                if not active and replay is None and (STOP.is_set() or not any(pending.values()) and not available):break
                time.sleep(.2)
        if STOP.is_set():status(root,'stopped_resumable',started,profile);return
        stage=['numerical_validation'];finished=threading.Event()
        def heartbeat():
            while not finished.wait(15):status(root,stage[0],started,profile)
        monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
        try:
            status(root,stage[0],started,profile)
            validation=validate_additional(root,config,resources['joint_device'])
            stage[0]='figures_and_report';status(root,stage[0],started,profile)
            from .reporting import generate_report
            report=generate_report(root,config=config)
            if report['completed_runs']!=420 or report['covered_selected_requests']!=322560:
                raise RuntimeError('Final total does not match exactly ten interactions')
            for name,expected in json.loads((root/'reused_statistical_sha256.json').read_text()).items():
                if sha(root/'statistical'/name)!=expected or sha(Path(source)/'statistical'/name)!=expected:
                    raise RuntimeError('Original statistical data changed')
        finally:finished.set();monitor.join()
        status(root,'complete_with_unresolved' if validation['status']=='unresolved' else 'complete',started,profile)
    except Exception as exc:
        traceback.print_exc();status(root,'failed',started,profile,error=str(exc))
        raise
    finally:lock.close()

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run(args.source,args.run_dir)

if __name__=='__main__':main()
