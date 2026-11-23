"""Incremental M=2 producer, exact-history checkpoints, and physical replay.

The six original selectors/inference engines are never edited by this module.
Only nine new method IDs are accepted by this producer.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import copy
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from .line_template import make_template
from .metrics import information_ratio
from .model import coordinate_of_query, probabilities, stable_seed
from .posterior_marginal import MarginalPosterior
from .posterior_smc import SMCPosterior
from .statistical_runner import _atomic_bytes, _atomic_npz, run_key
from .util import atomic_json, atomic_text, digest, load_config, utcnow

DIAGNOSTIC_T = (0, 31, 63, 95, 127)
STOP = threading.Event()


def plain(value):
    """Strict JSON diagnostics: missing/nonfinite values stay missing."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class AddonSMCPosterior(SMCPosterior):
    """Analytic uniform-prior moments, followed by the unchanged joint engine.

Initial particle integration still has Monte Carlo error; its empirical moments
are recorded separately. This convention applies only to new addon trajectories.
"""
    def moments(self):
        if not self.history:
            return np.zeros(self.d), np.eye(self.d) / 3
        return super().moments()

    def projected_quantiles(self, direction, probabilities):
        if not self.history and np.count_nonzero(np.abs(direction) > 1e-12) == 1:
            return (2 * np.asarray(probabilities) - 1) * np.linalg.norm(direction)
        return super().projected_quantiles(direction, probabilities)


def make_state(job):
    from .m2_addon_selectors import AddonSelectorState, MARGINAL_METHODS, NEW_METHODS
    if job['method'] not in NEW_METHODS:
        raise ValueError('Addon producer must not rerun controls or SG-legacy')
    if (job['setting_id'], job['M'], job['dimension'], float(job['beta']), job['answers']) != ('m02_b04', 2, 4, 4., 128):
        raise ValueError('Only frozen M2/d4/beta4/T128 is authorized')
    root, method, iid = job['root_seed'], job['method'], job['interaction_id']
    seeds = {p: stable_seed(root, 'm02_b04', iid, None if p == 'theta' else method, p)
             for p in ('theta', 'response', 'selector', 'smc', 'diagnostics')}
    rng = {p: np.random.default_rng(s) for p, s in seeds.items()}
    theta = np.asarray(job['theta_star'], dtype=np.float64)
    if theta.shape != (4,) or not np.isfinite(theta).all() or np.any(np.abs(theta) > 1):
        raise ValueError('Invalid frozen theta_star')
    # Compare seed derivation but use the saved values as the authority.
    if not np.array_equal(rng['theta'].uniform(-1, 1, 4), theta):
        raise ValueError('Frozen theta/seed mapping mismatch')
    posterior = (MarginalPosterior(4, 4., job.get('nodes', 257)) if method in MARGINAL_METHODS
                 else AddonSMCPosterior(4, 4., job.get('particles', 4096), rng['smc'], job.get('device', 'cpu')))
    empirical = None
    if isinstance(posterior, SMCPosterior):
        em, ec = SMCPosterior.moments(posterior)
        empirical = dict(mean=em, covariance=ec, risk=float(np.trace(ec)))
    mean, cov = posterior.moments()
    names = ('q', 'q_raw', 'y', 'eig', 'reference_eig', 'eta', 'reference_queries',
             'reference_eigs', 'fisher_scores', 'ratio_status', 'diag_t', 'diag_eig',
             'diag_eta', 'diag_se', 'update_diagnostics', 'selector_stage',
             'selector_center', 'selector_std', 'selector_metadata', 'cap_activated')
    data = {key: [] for key in names}
    data.update(mu=[mean], cov=[cov], risk=[float(np.trace(cov))],
                error=[float(np.sum((mean - theta) ** 2))])
    return dict(posterior=posterior, rng=rng, selector_state=AddonSelectorState(),
                theta=theta.copy(), seeds=seeds, data=data, job=copy.deepcopy(job),
                answers_completed=0, elapsed_seconds=0., pending_rollout_request_ids=[],
                initial_particle_moments=empirical)


def arrays(state):
    data = state['data']
    result = {k: np.asarray(v) for k, v in data.items()
              if k not in ('update_diagnostics', 'ratio_status', 'selector_metadata')}
    result['ratio_status'] = np.asarray(data['ratio_status'], dtype='U64')
    result['selector_metadata'] = np.asarray([json.dumps(plain(x), sort_keys=True) for x in data['selector_metadata']])
    result.update(theta=state['theta'], answers_completed=np.asarray(state['answers_completed']),
                  eta_grid=np.asarray(data['eta']), posterior_risk=np.asarray(data['risk']),
                  realized_sq_error=np.asarray(data['error']))
    return result


def run_statistical(job, output_dir, *, resume=True, max_answers=None, diagnostic_draws=64):
    from .m2_addon_selectors import select_query
    job = copy.deepcopy(job)
    root = Path(output_dir)
    for folder in ('statistical', 'checkpoints', 'randomized_diagnostics', 'logs'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    key = run_key(job)
    checkpoint = root / 'checkpoints' / f'{key}.pkl'
    artifact = root / 'statistical' / f'{key}.npz'
    with (root / 'checkpoints' / f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if artifact.exists() and artifact.with_suffix('.json').exists():
            meta = json.loads(artifact.with_suffix('.json').read_text())
            if meta['job'] != job or meta['answers_completed'] != 128:
                raise ValueError('Existing completed artifact has different frozen job')
            return dict(run_key=key, complete=True, artifact=str(artifact), reused=True)
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(checkpoint)
            with checkpoint.open('rb') as stream:
                state = pickle.load(stream)
            if state['job'] != job:
                raise ValueError('Checkpoint code/config/base/horizon mismatch')
        else:
            state = make_state(job)
        posterior, rng, data = state['posterior'], state['rng'], state['data']
        target = min(128, max_answers) if max_answers is not None else 128
        before_time = state['elapsed_seconds']
        started = time.monotonic()
        randomized = job['method'] == 'sm_continuous' or job['method'].startswith('gausscloud_paper_')
        for t in range(state['answers_completed'], target):
            template = make_template(posterior, 2)
            old_stage = state['selector_state'].stage
            raw, q = select_query(job['method'], posterior, 2, t, rng['selector'],
                                  state['selector_state'], template, horizon=128)
            if q.shape != (2, 4) or not np.isfinite(q).all() or np.max(np.abs(q)) > 33 + 1e-10:
                raise ValueError('Invalid displayed query/adapter envelope')
            if posterior.factorized:
                # Inspect actual displayed geometry; method name alone is insufficient.
                coordinate_of_query(q)
            eig = float(posterior.eig(q))
            refs = np.asarray([posterior.eig(question) for question in template.queries])
            reference = float(refs.max())
            eta, ratio_status = information_ratio(eig, reference)
            metadata = copy.deepcopy(state['selector_state'].diagnostics)
            if randomized and t in DIAGNOSTIC_T:
                frozen_state = copy.deepcopy(state['selector_state'])
                diagnostic_rng = copy.deepcopy(rng['diagnostics'])
                values = []
                for _ in range(diagnostic_draws):
                    _, dq = select_query(job['method'], posterior, 2, t, diagnostic_rng,
                                         frozen_state, template, horizon=128)
                    values.append(float(posterior.eig(dq)))
                # Only the independent diagnostics stream advances.
                rng['diagnostics'].bit_generator.state = diagnostic_rng.bit_generator.state
                values = np.asarray(values)
                data['diag_t'].append(t)
                data['diag_eig'].append(values)
                data['diag_eta'].append(float(values.mean() / reference) if reference > 0 else np.nan)
                data['diag_se'].append(float(values.std(ddof=1) / np.sqrt(len(values)) / reference) if reference > 0 and len(values) > 1 else np.nan)
            y = int(rng['response'].choice(2, p=probabilities(state['theta'], q, 4.)))
            for name, value in (('q', q.copy()), ('q_raw', raw.copy()), ('y', y), ('eig', eig),
                                ('reference_eig', reference), ('eta', eta), ('reference_queries', template.queries),
                                ('reference_eigs', refs), ('fisher_scores', template.fisher_scores), ('ratio_status', ratio_status)):
                data[name].append(value)
            selector = state['selector_state']
            data['selector_stage'].append(selector.stage if selector.stage is not None else -1)
            data['selector_center'].append(selector.center.copy() if selector.center is not None else np.full(4, np.nan))
            data['selector_std'].append(selector.std if selector.std is not None else np.nan)
            data['selector_metadata'].append(metadata)
            data['cap_activated'].append(np.linalg.norm(q - raw, axis=1) > 1e-12)
            posterior.update(q, y)
            data['update_diagnostics'].append(copy.deepcopy(posterior.last_diagnostics))
            mean, cov = posterior.moments()
            data['mu'].append(mean); data['cov'].append(cov)
            data['risk'].append(float(np.trace(cov)))
            data['error'].append(float(np.sum((mean - state['theta']) ** 2)))
            state['answers_completed'] = t + 1
            state['pending_rollout_request_ids'].append(f'{key}__t{t:03d}')
            if (t + 1) % 16 == 0 or t + 1 == target or selector.stage != old_stage:
                state['elapsed_seconds'] = before_time + time.monotonic() - started
                _atomic_bytes(checkpoint, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
                _atomic_npz(root / 'checkpoints' / f'{key}.partial.npz', arrays(state))
                atomic_json(root / 'checkpoints' / f'{key}.progress.json',
                            dict(run_key=key, answers_completed=t+1, answers_target=128,
                                 elapsed_seconds=state['elapsed_seconds'], pid=os.getpid(), time_utc=utcnow()))
        if state['answers_completed'] == 128:
            _atomic_npz(artifact, arrays(state))
            meta = dict(**job, job=job, run_key=key, seeds=state['seeds'], answers_completed=128,
                        elapsed_seconds=state['elapsed_seconds'],
                        reference_scope='own_posterior_seven_scale_grid', eta_field_alias='eta_grid',
                        initial_moments='analytic_uniform_prior', initial_exact_prior_risk=4/3,
                        initial_particle_moments=state['initial_particle_moments'],
                        numerical_engine='adaptive_sinh_simpson_positive_continuous_pchip_cdf' if posterior.factorized else 'tempered_resample_move_exact_delayed_acceptance',
                        exact_eig_formula_used=True, exact_posterior_computed=False,
                        selector_metadata=data['selector_metadata'], update_diagnostics=data['update_diagnostics'],
                        artifact=str(artifact), checkpoint=str(checkpoint))
            atomic_json(artifact.with_suffix('.json'), plain(meta))
            if randomized:
                atomic_json(root/'randomized_diagnostics'/f'{key}.json', plain(dict(
                    t=data['diag_t'], eig_draws=data['diag_eig'], mean_eta_grid=data['diag_eta'],
                    ratio_monte_carlo_se=data['diag_se'], actual_answers_generated=0,
                    label='conditional next-query mean at frozen posterior', draws=diagnostic_draws)))
        return dict(run_key=key, answers_completed=state['answers_completed'], complete=state['answers_completed']==128,
                    elapsed_seconds=state['elapsed_seconds'], artifact=str(artifact) if artifact.exists() else None)


def source_code_inventory():
    root = Path(__file__).parent
    paths = set(root.glob('*.py')) | set((root/'tests').glob('test_m2_addon_*.py'))
    paths |= {root/'tests'/'test_statistics.py', root/'tests'/'test_robot.py'}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def check_ready(root):
    root = Path(root)
    ready = json.loads((root/'READY.json').read_text())
    if ready.get('status') != 'queue_ready' or not ready.get('tests_passed'):
        raise RuntimeError('Correctness gate has not passed')
    if ready['source_files'] != source_code_inventory():
        raise RuntimeError('Implementation changed since readiness tests; revalidate')
    if ready['frozen_config_sha256'] != hashlib.sha256((root/'frozen_config.yaml').read_bytes()).hexdigest():
        raise RuntimeError('Frozen scientific configuration changed')
    return ready


def jobs_for_run(root):
    from .m2_addon_selectors import NEW_METHODS, MARGINAL_METHODS
    root = Path(root)
    config = load_config(root/'frozen_config.yaml')
    source = Path(config['addon']['source_run'])
    jobs = []
    lock = json.loads((root/'BASELINE_LOCK.json').read_text())
    context_ids = {row['interaction_id']: row['context_id'] for row in lock['interactions']}
    for iid in config['experiment']['interaction_ids']:
        with np.load(source/'statistical'/f'm02_b04__RC__i{iid}.npz', allow_pickle=False) as data:
            theta = data['theta'].tolist()
        for method in NEW_METHODS:
            jobs.append(dict(setting_id='m02_b04', method=method, interaction_id=iid,
                             M=2, dimension=4, beta=4., answers=128, particles=4096, nodes=257,
                             root_seed=config['experiment']['root_seed'], theta_star=theta,
                             context_id=context_ids[iid], device='cpu', code_hash=digest(source_code_inventory()),
                             config_hash=hashlib.sha256((root/'frozen_config.yaml').read_bytes()).hexdigest(),
                             base_lock_hash=hashlib.sha256((root/'BASELINE_LOCK.json').read_bytes()).hexdigest(),
                             kind='marginal' if method in MARGINAL_METHODS else 'joint'))
    return jobs


def worker_init():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def statistical_worker(job, root):
    worker_init()
    with (Path(root)/'logs'/f'{run_key(job)}.log').open('a', buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            print(f'{utcnow()} start {job}', flush=True)
            try:
                result = run_statistical(job, root)
                print(f'{utcnow()} {result}', flush=True)
                return result
            except Exception:
                traceback.print_exc()
                raise


def replay_one(pool, artifact, root):
    artifact, root = Path(artifact), Path(root)
    meta = json.loads(artifact.with_suffix('.json').read_text())
    with np.load(artifact, allow_pickle=False) as data:
        questions = data['q'].copy()
    requests = [dict(request_id=f'{artifact.stem}__t{t:03d}', context_id=meta['context_id'],
                     q=q, kind='selected', setting_id='m02_b04', method=meta['method'],
                     interaction_id=meta['interaction_id'], t=t) for t, q in enumerate(questions)]
    descriptor = dict(status='pending', source=str(artifact), requests=requests)
    atomic_json(root/'rollout_requests'/f'{artifact.stem}.json', descriptor)
    result = pool.execute_queries(requests, root/'rollouts'/artifact.stem)
    descriptor.update(status='completed' if result['failed']==0 else 'completed_with_failures', result=result)
    atomic_json(root/'rollout_requests'/f'{artifact.stem}.json', descriptor)
    atomic_json(root/'rollouts'/f'{artifact.stem}.done.json',
                dict(statistical_run_id=artifact.stem, selected_logical=2*len(questions), reference_logical=0,
                     result=result, completed_at=utcnow()))
    return result


def replay_is_complete(root, key):
    root = Path(root)
    done = root/'rollouts'/f'{key}.done.json'
    shard = root/'rollouts'/f'{key}.json'
    if not (done.exists() and shard.exists() and shard.with_suffix('.npz').exists()):
        return False
    try:
        result = json.loads(done.read_text())['result']
        records = json.loads(shard.read_text())['per_candidate']
        return (result.get('failed') == 0 and result.get('logical_requests') == 256
                and result.get('simulation_valid') == 256 and len(records) == 256
                and all(r.get('simulation_valid') and r.get('command_reconstruction_ok')
                        and r.get('action_bounds_ok') and r.get('actual_steps') == 50 for r in records))
    except (ValueError, KeyError, OSError):
        return False


def write_status(root, stage, started, **extra):
    root = Path(root)
    completed = len(list((root/'statistical').glob('m02_b04__*.npz')))
    replayed = len(list((root/'rollouts').glob('*.done.json')))
    status = dict(stage=stage, updated_at=utcnow(), new_runs_completed=completed, new_runs_expected=90,
                  reused_controls=50, replayed_new_runs=replayed, elapsed_seconds=time.time()-started, **extra)
    atomic_json(root/'status.json', plain(status))
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update(status=stage, production_started=True, operational_status=str(root/'status.json'))
        atomic_json(manifest_path, manifest)
    atomic_text(root/'STATUS.md', f'# M=2 addon\n\nStatus: {stage}\n\nNew statistical runs: {completed}/90; reused controls: 50.\n'
                f'Physical replay shards: {replayed}/90.\n\nUpdated: {status["updated_at"]}\n')
    if completed and stage == 'inference_and_rollouts':
        remaining = (90-completed) * max(0., time.time()-started) / completed
        atomic_text(root/'ETA.md', f'Observed statistical throughput estimate: {remaining/60:.1f} minutes remaining for statistical runs.\n'
                    'Physical replay, validation and videos are additional; this is not a completion guarantee.\n')


def profile(root):
    """Short M2-only physical throughput profile, after queue resource admission."""
    from .rollout_pool import RolloutPool
    root = Path(root)
    saved = root/'PROFILE.json'
    if saved.exists():
        prior = json.loads(saved.read_text())
        if prior.get('status') == 'measured':
            return prior
    config = load_config(root/'frozen_config.yaml')
    cid = config['experiment']['interaction_ids'][0]
    rng = np.random.default_rng(917243)
    qs = rng.uniform(-1, 1, (128, 2, 4))
    records = []
    started = time.monotonic()
    affinity = len(os.sched_getaffinity(0))
    for workers in (16, 24, 32):
        if workers > affinity or (records and time.monotonic()-started > 480):
            continue
        requests = [dict(request_id=f'profile_w{workers}_{i}', context_id=cid, q=q,
                         kind='engineering_profile') for i, q in enumerate(qs)]
        before = time.monotonic()
        with RolloutPool(config['addon']['context_dir'], workers=workers, chunk_size=16) as pool:
            result = pool.execute_queries(requests, root/'profiling'/f'workers_{workers}')
        wall = time.monotonic()-before
        if result['failed']:
            raise RuntimeError('M2 physical profiling failed')
        records.append(dict(wall_seconds=wall, clips_per_second=256/wall,
                            initialization_included=True, **result))
    fastest = max(records, key=lambda r:r['clips_per_second'])
    result = dict(status='measured', workload='M2 only, 128 engineering pairs per worker count',
                  profiles=records, rollout_workers=fastest['workers'], marginal_workers=4,
                  allowed_physical_gpus=[0,1,2], forbidden_compute_gpus=[3],
                  scientific_budget_changes=False)
    atomic_json(saved, result)
    return result


def preflight(root):
    """Fixed extreme addon candidates and restore checks, never outcome selected."""
    from .contexts import load_contexts
    from .renderer import RobotAdapter
    root = Path(root)
    config = load_config(root/'frozen_config.yaml')
    path = root/'validation'/'physical_preflight.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous.get('status') == 'passed'
            and len(previous.get('records', [])) == 16*len(config['experiment']['interaction_ids'])
            and all(r.get('simulation_valid') and r.get('command_reconstruction_ok') for r in previous['records'])):
            return previous
    adapter = RobotAdapter(load_contexts(config['addon']['context_dir']))
    records = []
    try:
        for cid in config['experiment']['interaction_ids']:
            for amplitude in (20., 32.):
                for axis in range(4):
                    q = amplitude*np.eye(4)[axis]
                    for sign in (-1, 1):
                        result = adapter.rollout(sign*q, cid)
                        records.append(dict(q=(sign*q).tolist(), **result['diagnostics']))
            q = np.array([.2, -.3, .4, -.5])
            first = adapter.rollout(q, cid)
            adapter.rollout(-q, cid)
            second = adapter.rollout(q, cid)
            if not np.array_equal(first['positions'], second['positions']):
                raise RuntimeError('Physical restore/candidate order invariance failed')
    finally:
        adapter.close()
    result = dict(status='passed' if all(r['simulation_valid'] and r['command_reconstruction_ok'] for r in records) else 'unresolved',
                  records=records, newly_selected_queries=False)
    atomic_json(path, result)
    if result['status'] != 'passed':
        raise RuntimeError('Addon physical preflight unresolved; adapter not changed')
    return result


def render_videos(root):
    """Runs in an EGL-isolated process only after quantitative phases."""
    import imageio.v2 as imageio
    from .contexts import load_contexts
    from .renderer import RobotAdapter
    root = Path(root)
    config = load_config(root/'frozen_config.yaml')
    iid = config['experiment']['interaction_ids'][0]
    source = Path(config['addon']['source_run'])
    output = root/'selected_videos'; output.mkdir(exist_ok=True)
    records = []
    adapter = RobotAdapter(load_contexts(config['addon']['context_dir']), render_mode='rgb_array')
    try:
        for method in ('RC', 'sm_continuous', 'gausscloud_paper_L3'):
            base = source if method == 'RC' else root
            path = base/'statistical'/f'm02_b04__{method}__i{iid}.npz'
            with np.load(path, allow_pickle=False) as data:
                questions = data['q'].copy()
            for t in (0, 63, 127):
                for slot, q in enumerate(questions[t]):
                    video = output/f'{path.stem}__t{t:03d}__c{slot}.mp4'
                    record = dict(method=method, interaction_id=iid, t=t, candidate_slot=slot, path=str(video))
                    if (video.exists() and video.stat().st_size > 0 and video.with_suffix('.json').exists()
                        and json.loads(video.with_suffix('.json').read_text()).get('simulation_valid') is True):
                        record.update(status='reused')
                    else:
                        try:
                            result = adapter.rollout(q, iid, collect_frames=True)
                            if not result['diagnostics']['simulation_valid']:
                                raise RuntimeError(result['diagnostics'])
                            with imageio.get_writer(video, fps=50, codec='libx264', quality=7,
                                                    ffmpeg_params=['-threads','1']) as writer:
                                for frame in result['frames']:
                                    writer.append_data(frame)
                            atomic_json(video.with_suffix('.json'), result['diagnostics'])
                            record.update(status='executed')
                        except Exception as exc:
                            record.update(status='failed', error=str(exc))
                    records.append(record)
                    atomic_json(output/'manifest.json', dict(precommitted_subset=True, planned=18, clips=records))
    finally:
        adapter.close()
    return dict(planned=18, failed=sum(r['status']=='failed' for r in records), clips=records)


def run_pipeline(root):
    from .m2_addon_lock import verify_lock
    from .m2_addon_validation import validate_run, validate_devices
    from .m2_addon_reporting import generate_report
    from .rollout_pool import RolloutPool
    root = Path(root).resolve()
    check_ready(root)
    # Child logical CUDA indices map strictly to these three physical cards.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2':
        raise RuntimeError('Runner requires CUDA_VISIBLE_DEVICES=0,1,2 before startup')
    if not verify_lock(root)['unchanged']:
        raise RuntimeError('Baseline inventory changed before execution')
    config = load_config(root/'frozen_config.yaml')
    started = time.time()
    STOP.clear()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_text(root/'supervisor.pid', str(os.getpid())+'\n')
        try:
            write_status(root, 'device_equivalence', started)
            parity = validate_devices(root, devices=('cuda:0','cuda:1','cuda:2'))
            if parity.get('status') != 'passed':
                raise RuntimeError('CPU/GPU arithmetic gate failed before production; see device_equivalence.json')
            write_status(root, 'physical_preflight', started)
            preflight(root)
            write_status(root, 'profiling', started)
            measured = profile(root)
            jobs = jobs_for_run(root)
            pending = {'marginal':[], 'joint':[]}
            for job in jobs:
                artifact = root/'statistical'/f'{run_key(job)}.npz'
                if artifact.exists() and artifact.with_suffix('.json').exists():
                    metadata = json.loads(artifact.with_suffix('.json').read_text())
                    expected = {k:v for k,v in job.items() if k != 'device'}
                    actual = {k:v for k,v in metadata.get('job', {}).items() if k != 'device'}
                    if expected != actual or metadata.get('answers_completed') != 128:
                        raise RuntimeError(f'Completed addon job is incompatible: {artifact}')
                else:
                    cp = root/'checkpoints'/f'{run_key(job)}.pkl'
                    if cp.exists():
                        with cp.open('rb') as f:
                            old = pickle.load(f)
                        job['device'] = old['job']['device']
                        job['_resume_device'] = job['device']
                    pending[job['kind']].append(job)
            ctx = mp.get_context('spawn')
            with contextlib.ExitStack() as stack:
                cpu = stack.enter_context(cf.ProcessPoolExecutor(max_workers=measured['marginal_workers'],
                                          mp_context=ctx, initializer=worker_init))
                gpus = [stack.enter_context(cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx,
                                           initializer=worker_init)) for _ in range(3)]
                consumer = stack.enter_context(cf.ThreadPoolExecutor(max_workers=1))
                pool = stack.enter_context(RolloutPool(config['addon']['context_dir'], workers=measured['rollout_workers'],
                                                       chunk_size=64, cache_dir=Path(config['addon']['source_run'])/'rollouts'))
                pool.scan_cache(root/'rollouts')
                active = {}; replay = None
                processed = {p.name.removesuffix('.done.json') for p in (root/'rollouts').glob('*.done.json')
                             if replay_is_complete(root, p.name.removesuffix('.done.json'))}
                last_status = 0.
                while True:
                    for slot, executor, kind, limit in [('cpu',cpu,'marginal',measured['marginal_workers'])] + [(f'cuda:{i}',g,'joint',1) for i,g in enumerate(gpus)]:
                        while not STOP.is_set() and pending[kind] and sum(s==slot for s in active.values()) < limit:
                            eligible = next((i for i,j in enumerate(pending[kind])
                                             if j.get('_resume_device', slot) == slot), None)
                            if eligible is None:
                                break
                            job = pending[kind].pop(eligible)
                            job.pop('_resume_device', None)
                            job['device'] = slot
                            active[executor.submit(statistical_worker, job, str(root))] = slot
                    for future in list(active):
                        if future.done():
                            future.result(); del active[future]
                    if replay is not None and replay.done():
                        replay.result(); replay = None
                    available = sorted(p for p in (root/'statistical').glob('m02_b04__*.npz')
                                       if p.stem not in processed and p.with_suffix('.json').exists())
                    if replay is None and available and not STOP.is_set():
                        artifact = available[0]; processed.add(artifact.stem)
                        replay = consumer.submit(replay_one, pool, artifact, root)
                    if time.monotonic()-last_status > 15:
                        write_status(root, 'inference_and_rollouts', started); last_status=time.monotonic()
                    if not active and replay is None and (STOP.is_set() or not any(pending.values()) and not available):
                        break
                    time.sleep(.25)
            if STOP.is_set():
                write_status(root, 'stopped_resumable', started)
                return
            stage = ['numerical_validation']
            done = threading.Event()
            def heartbeat():
                while not done.wait(15):
                    write_status(root, stage[0], started)
            monitor = threading.Thread(target=heartbeat, daemon=True); monitor.start()
            try:
                validation = validate_run(root, config=config, device='cuda:0', numerical=True)
                stage[0] = 'qualitative_videos'
                env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
                with (root/'logs'/'videos.log').open('a') as stream:
                    video = subprocess.run([sys.executable,'-m','metaworld_fast.m2_addon_cli','videos','--run-dir',str(root)],
                                           env=env, stdout=stream, stderr=subprocess.STDOUT)
                stage[0] = 'figures_tables_report'
                check_ready(root)
                if not verify_lock(root)['unchanged']:
                    raise RuntimeError('Baseline inventory changed during execution')
                report = generate_report(root, config=config)
            finally:
                done.set(); monitor.join()
            if (report['completed_runs'] != 140 or report['reused_completed_runs'] != 50
                or report['new_actual_answers'] != 11520 or report['covered_selected_clips'] != 35840
                or report['failed_selected_clips'] or report['pending_selected_clips']
                or not report['physical_complete'] or not report['videos_complete']
                or validation.get('artifact_status') != 'passed' or video.returncode):
                raise RuntimeError('Final coverage/artifact/video gate failed; see completion_summary and validation')
            write_status(root, 'complete_with_unresolved' if validation.get('status') != 'passed' else 'complete',
                         started, validation=plain(validation), report=plain(report), videos_returncode=video.returncode)
        except Exception as exc:
            traceback.print_exc()
            write_status(root, 'failed_resumable', started, error=f'{type(exc).__name__}: {exc}')
            raise
