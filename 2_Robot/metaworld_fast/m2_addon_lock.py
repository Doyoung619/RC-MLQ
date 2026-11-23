"""Read-only baseline discovery and immutable M=2 addon preparation.

No simulator, CUDA worker, scientific run, or predecessor queue is started here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np
import yaml

from .util import atomic_json, atomic_text, digest, utcnow

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
DEFAULT_ROOT = PACKAGE / 'outputs/m2_addon_t128_s10_20261006T0017Z'
NOT_BEFORE = '2026-10-06T00:57:46+00:00'
CONTROLS = ['RC', 'PQ', 'QI', 'PS', 'GC']
NEW_METHODS = ['sm_continuous', 'mem_line', 'gmem_line', '2mem_line_2sweeps',
               'infosynth_trace', 'infosynth_bounded', 'gausscloud_paper_L1',
               'gausscloud_paper_L3', 'gausscloud_paper_L5']
SOURCE_COMMIT = '9e621462af6b7ca502b362b1415c781dc10eb404'
SOURCE_BLOB = '1969bfe73ca95027060b22ab0d2421cc446d7941'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _source_snapshot(root):
    """Fetch the pinned research source, check git's object ID, record license audit."""
    directory = root / 'source_provenance'
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f'https://raw.githubusercontent.com/siplab-gt/pairsearch/{SOURCE_COMMIT}/'
    path = directory / 'nusvm_sim.m'
    if not path.exists():
        data = urllib.request.urlopen(prefix + 'enusvm/nusvm_sim.m', timeout=30).read()
        path.write_bytes(data)
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError(f'Pinned author code blob mismatch: {blob}')
    listing_path = directory / 'repository_root_listing.json'
    if not listing_path.exists():
        url = f'https://api.github.com/repos/siplab-gt/pairsearch/contents?ref={SOURCE_COMMIT}'
        listing = json.load(urllib.request.urlopen(url, timeout=30))
        atomic_json(listing_path, listing)
    listing = _json(listing_path)
    if not (directory / 'README.upstream.md').exists():
        (directory / 'README.upstream.md').write_bytes(
            urllib.request.urlopen(prefix + 'README.md', timeout=30).read())
    lines = data.decode().splitlines()
    anchors = {needle: [i + 1 for i, line in enumerate(lines) if needle in line]
               for needle in ['sig = max', 'measperstage =', 'measperstage(end)',
                              'sigstage =', 'Z = sigstage', 'xhat = enusvm_solve']}
    license_names = [r['name'] for r in listing if any(
        token in r['name'].lower() for token in ('license', 'copying'))]
    result = dict(repository='siplab-gt/pairsearch', commit=SOURCE_COMMIT,
                  source_path='enusvm/nusvm_sim.m', blob_sha1=blob,
                  sha256=sha256(path), local_snapshot=str(path), url=prefix+'enusvm/nusvm_sim.m',
                  line_anchors=anchors, root_license_files=license_names,
                  license_status='No LICENSE/COPYING at pinned repository root; no permission inferred.',
                  reuse_scope='Single source file retained as research evidence; independently implemented schedule; no repository copied into executable product code.')
    atomic_json(directory / 'SOURCE_LOCK.json', result)
    return result


def discover(base_run=None, ellipsoid_run=None):
    base = Path(base_run) if base_run else Path((PACKAGE / 'CURRENT_RUN.txt').read_text().strip())
    ellipsoid = Path(ellipsoid_run) if ellipsoid_run else Path((PACKAGE / 'ELLIPSOID_CURRENT_RUN.txt').read_text().strip())
    if not base.is_absolute(): base = PROJECT / base
    if not ellipsoid.is_absolute(): ellipsoid = PROJECT / ellipsoid
    base, ellipsoid = base.resolve(), ellipsoid.resolve()
    config = yaml.safe_load((base / 'config_frozen.yaml').read_text())
    e = config['experiment']
    settings = [s for s in e['settings'] if s['id'] == 'm02_b04']
    if len(settings) != 1 or settings[0]['choices'] != 2 or float(settings[0]['beta']) != 4:
        raise ValueError('Baseline is not the required M=2, beta=4 setting')
    if e['dimension'] != 4 or e['answers_per_run'] != 128 or len(e['interaction_ids']) != 10:
        raise ValueError('Baseline dimension/horizon/interaction count does not match the addon')
    return base, ellipsoid, config


def _run_record(base, method, interaction, theta):
    key = f'm02_b04__{method}__i{interaction}'
    artifact, metadata = base/'statistical'/f'{key}.npz', base/'statistical'/f'{key}.json'
    if not artifact.exists() or not metadata.exists():
        return dict(run_key=key, method=method, interaction_id=interaction, status='missing')
    meta = _json(metadata)
    with np.load(artifact, allow_pickle=False) as z:
        ok = (z['q'].shape == (128,2,4) and z['y'].shape == (128,)
              and z['cov'].shape == (129,4,4) and np.array_equal(z['theta'], theta)
              and int(z['answers_completed']) == 128 and np.isfinite(z['q']).all()
              and np.isfinite(z['cov']).all() and np.isfinite(z['eig']).all())
    ok = ok and all(meta[k] == v for k,v in [('M',2),('beta',4.0),('dimension',4),('answers_completed',128)])
    if not ok: raise ValueError(f'Incompatible completed control: {key}')
    checkpoint = base/'checkpoints'/f'{key}.pkl'
    if not checkpoint.exists(): checkpoint = Path(meta['checkpoint'])
    rollout = base/'rollouts'/f'{key}.done.json'
    return dict(run_key=key, method=method, interaction_id=interaction, status='reused',
                artifact=str(artifact), metadata=str(metadata), checkpoint=str(checkpoint),
                rollout_done=str(rollout), rollout_complete=rollout.exists(), seeds=meta['seeds'],
                numerical_engine=meta['numerical_engine'], metadata_config_hash=meta['config_hash'])


def verify_lock(run_dir):
    root = Path(run_dir)
    lock = _json(root/'BASELINE_LOCK.json')
    changed = []
    for filename, expected in lock['immutable_inventory'].items():
        path = Path(filename)
        if not path.is_file(): changed.append(dict(path=filename, error='missing'))
        elif sha256(path) != expected: changed.append(dict(path=filename, error='sha256_changed'))
    result = dict(checked_at=utcnow(), unchanged=not changed,
                  checked_files=len(lock['immutable_inventory']), changed=changed)
    atomic_json(root/'validation/baseline_immutability.json', result)
    return result


def prepare_lock(run_dir=DEFAULT_ROOT, base_run=None, ellipsoid_run=None):
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'BASELINE_LOCK.json').exists():
        verification = verify_lock(root)
        if not verification['unchanged']: raise ValueError('Previously locked baseline changed')
        return _json(root/'BASELINE_LOCK.json')
    base, ellipsoid, config = discover(base_run, ellipsoid_run)
    for directory in ['statistical','checkpoints','oracle_sidecars','randomized_diagnostics',
                      'rollout_requests','rollouts','validation','tests','figures','tables',
                      'selected_videos','logs','source_provenance','source_snapshot']:
        (root/directory).mkdir(exist_ok=True)
    interactions, records = [], []
    ids = config['experiment']['interaction_ids']
    seeds = _json(base/'seed_manifest.json')
    contexts = pickle.loads((base/'contexts/contexts.pkl').read_bytes())
    context_by_id = {c['context_id']:c for c in contexts}
    for interaction in ids:
        match = [s for s in seeds if s['setting_id']=='m02_b04' and s['interaction_id']==interaction]
        if len(match) != 6: raise ValueError(f'Missing baseline RNG mapping for interaction {interaction}')
        theta = np.asarray(match[0]['theta_star'])
        if not all(np.array_equal(theta,s['theta_star']) for s in match):
            raise ValueError('Baseline methods do not share theta values')
        ctx = context_by_id[interaction]
        interactions.append(dict(interaction_id=interaction,theta_star=theta.tolist(),
                                 theta_seed=match[0]['seeds']['theta'],context_id=interaction,
                                 context_hash=ctx['context_hash'],context_seed=ctx['seed'],
                                 control_seeds={s['method']:s['seeds'] for s in match}))
        records.extend(_run_record(base, method, interaction, theta) for method in CONTROLS+['SG'])
    from .renderer import adapter_parameters
    from .contexts import source_hash, hash_value
    eval_manifest = _json(ellipsoid/'manifest.json')
    if Path(eval_manifest['source_run']).resolve() != base:
        raise ValueError('Ellipsoid history source differs from completed base')
    oracle = dict(alpha_E=float(eval_manifest['protocol']['alpha']),
                  rho_E=float(eval_manifest['protocol']['radius']),
                  source_manifest=str(ellipsoid/'manifest.json'),
                  inherited=True, optimize_ellipsoid_lower=False,
                  note='Inherited rho_E=16; specification fallback radius is not used.')
    inventory = {}
    # Includes *all* M>2 numeric histories and original figures, not just reused M=2.
    # Completed output trees are read only; transient PID/lock/logs are excluded.
    for tree in (base, ellipsoid):
        for path in sorted(tree.rglob('*')):
            if not path.is_file() or path.suffix in ('.lock','.pid') or 'logs' in path.relative_to(tree).parts:
                continue
            inventory[str(path)] = sha256(path)
    for path in [PROJECT/'metaworld_m2_addon_codex_prompt.md',PROJECT/'metaworld_m2_addon_codex_prompt.txt']:
        inventory[str(path)] = sha256(path)
    for name in ['model.py','selectors.py','posterior_marginal.py','posterior_smc.py','renderer.py',
                 'contexts.py','metrics.py','line_template.py','rollout_pool.py','requirements-frozen.txt']:
        inventory[str(PACKAGE/name)] = sha256(PACKAGE/name)
    source = _source_snapshot(root)
    lock = dict(schema_version=1,locked_at=utcnow(),source_run=str(base),ellipsoid_source_run=str(ellipsoid),
                manifest_path=str(base/'manifest.json'),config_frozen_path=str(base/'config_frozen.yaml'),
                seed_manifest_path=str(base/'seed_manifest.json'),interaction_ids=ids,interactions=interactions,
                setting=dict(id='m02_b04',choices=2,beta=4.,dimension=4,answers=128),
                controls=[r for r in records if r['method']!='SG'],legacy_sg=[r for r in records if r['method']=='SG'],
                method_registry=CONTROLS+NEW_METHODS,oracle=oracle,
                latent_interpretation='Ten distinct baseline latent draws; shared exactly across methods within m02_b04.',
                context_mapping='interaction_id equals existing context_id; ten preexisting serialized contexts preserved.',
                context_dir=str(base/'contexts'),context_manifest=str(base/'contexts/manifest.json'),
                adapter=dict(parameters=adapter_parameters(50),code_hash=source_hash(),
                             versions=contexts[0]['versions'],contexts=[dict(context_id=c['context_id'],
                             H=c['H'],nominal_actions_hash=hash_value(c['nominal_actions']),
                             restore_state_hash=hash_value(c['state'])) for c in contexts]),
                inference=config['inference'],validation_source=str(base/'validation/summary.json'),
                baseline_validation_status=_json(base/'validation/summary.json')['status'],
                metrics=dict(eta_grid='stored eta; displayed-query integrated EIG / own-posterior seven-scale maximum',
                             posterior_risk='stored risk; trace of full covariance (129 post-answer states)',
                             realized_sq_error='stored error; squared mean minus locked theta',
                             ellipsoid='separate inherited rho16 ellipsoid lower/upper bound sidecars'),
                source_code=source,immutable_inventory=inventory,inventory_digest=digest(inventory))
    atomic_json(root/'BASELINE_LOCK.json', lock)
    required = [dict(run_key=f'm02_b04__{m}__i{i}',method=m,interaction_id=i)
                for i in ids for m in CONTROLS+NEW_METHODS]
    reused_keys = {r['run_key'] for r in records if r['method'] in CONTROLS and r['status']=='reused'}
    reused = [r for r in required if r['run_key'] in reused_keys]
    new = [r for r in required if r['run_key'] not in reused_keys]
    plan = dict(schema_version=1,created_at=utcnow(),source_run=str(base),allowed_settings=['m02_b04'],
                required=required,reused=reused,new=new,resumed=[],missing=[],
                counts=dict(required=len(required),reused=len(reused),new=len(new),resumed=0,missing=0),
                new_actual_answers=128*len(new),new_selected_candidate_clips=256*len(new),
                new_logical_action_steps=12800*len(new),
                original_SG_preserved=True,SG_reporting_alias='SG-legacy',
                rerun_completed_controls=False,rerun_m_gt_2=False,
                new_compatible_artifact_search='No addon artifacts existed at preparation; existing output manifests inspected.')
    atomic_json(root/'REUSE_PLAN.json',plan)
    frozen = copy.deepcopy(config)
    frozen['experiment'].update(name='metaworld_m2_binary_addon_s10',methods=CONTROLS+NEW_METHODS,
                                settings=[dict(id='m02_b04',choices=2,beta=4.0)],incremental_only=True)
    frozen['inference'].update(marginal_methods=['RC','PQ','QI']+NEW_METHODS[:6],
                               joint_methods=['PS','GC']+NEW_METHODS[6:],
                               addon_initial_moments='analytic_uniform_prior',
                               reused_control_initial_moments='preserved_original_numerical_representation')
    frozen['measurements']['random_query_diagnostic_methods'] = ['PS','GC','sm_continuous']+NEW_METHODS[6:]
    frozen['resources'].update(allowed_physical_gpus=[0,1,2],forbidden_compute_gpus=[3],
                               persistent_gpu_workers='one_per_allowed_gpu',cpu_worker_candidates=[16,24,32],
                               rollout_worker_blas_threads=1,dynamic_queue=True,gpu_processes=3)
    frozen['oracle'] = oracle
    frozen['addon'] = dict(source_run=str(base),ellipsoid_source_run=str(ellipsoid),context_dir=str(base/'contexts'),
                           controls=CONTROLS,new_methods=NEW_METHODS,not_before_utc=NOT_BEFORE,
                           original_sg_reporting_alias='SG-legacy',representative_interaction_ids=[ids[0]])
    frozen['gausscloud_paper'] = dict(stage_counts=[1,3,5],source_commit=SOURCE_COMMIT,
        source_blob_sha=SOURCE_BLOB,allocation='floor_each_remainder_to_final_stage',std_shrink_factor=0.5,
        initial_std=float(1/np.sqrt(3)),initial_scale_is_an_adaptation=True,
        initial_scale_source_rule='max_coordinate_sd_of_fixed_catalog',
        initial_scale_transfer='analytic_max_coordinate_sd_of_initial_uniform_prior',
        center_mode='shared_posterior_mean_at_stage_entry',intra_stage_recenter=False,
        original_catalog_retrieval=False,original_enusvm_estimator=False,cap_radius=32.)
    frozen['mem_family'] = dict(geometry='coordinate_line',center_domain=[-1.,1.],
        marginal_solver='weighted_sorted_partition',moving_center_solver='piecewise_quadratic_global',
        two_mem_slot_order=[0,1,0,1],until_convergence=False)
    frozen['infosynth_bounded'] = dict(sigma0=.25,relative_bounds=[.01,10.],physical_halfspacing_cap=32.,
        log_grid_points=25,sobol_points=1024,reference_sobol_points=4096,
        refinement_discrepancy_tolerance=.0005,log_scale_tolerance=.00001,
        acquisition='auxiliary_gaussian_binary_conditional_entropy',replace_response_model=False)
    frozen['output'].update(root=str(root),qualitative_setting='m02_b04',
                           qualitative_interaction_ids=[ids[0]],qualitative_methods=['RC','sm_continuous','gausscloud_paper_L3'])
    frozen.pop('extension',None)
    atomic_text(root/'frozen_config.yaml',yaml.safe_dump(frozen,sort_keys=False))
    (root/'config_frozen.yaml').symlink_to('frozen_config.yaml')
    (root/'contexts').symlink_to(base/'contexts', target_is_directory=True)
    atomic_json(root/'manifest.json',dict(schema_version=1,created_at=utcnow(),
        status='prepared_not_started',source_run=str(base),ellipsoid_source_run=str(ellipsoid),
        interaction_ids=ids,methods=CONTROLS+NEW_METHODS,allowed_settings=['m02_b04'],
        expected_counts=plan['counts'],config_hash=digest(frozen),
        not_before_utc=NOT_BEFORE,baseline_lock_sha256=sha256(root/'BASELINE_LOCK.json')))
    atomic_text(root/'CONFIG_DIFF.md',
        '# Fixed addon differences\n\nOnly m02_b04 (M=2,d=4,beta=4,T=128), same ten interactions and contexts. '
        'Five completed controls are reused; nine new configurations give 90 new runs. SG is preserved read only. '
        'GCQ-L uses the pinned author-code schedule with analytic initial SD 1/sqrt(3), shared Bayes, '
        'continuous pairs, and inherited radius-32 cap. The latest ellipsoid protocol is inherited: '
        'alpha_E=0.5, rho_E=16 (the fallback radius is not substituted). GPU3 is forbidden for addon compute, '
        'even though prior experiments used it. Production profiling starts only after delayed queue admission.\n')
    atomic_text(root/'PROVENANCE.md',
        '# M=2 addon provenance\n\n'
        '[BASE] Completed baseline: '+str(base)+'. Frozen manifest/config, all numeric histories, '
        'contexts, checkpoints, rollout/cache records, figures and table artifacts are SHA-256 locked. '
        'The base numerical validation status is unresolved; reuse does not relabel it passed. '
        'Each of ten original interaction IDs has a distinct theta draw and its corresponding existing context.\n\n'
        '[SOURCE] GaussCloud stage allocation and standard-deviation halving are independently implemented '
        'from siplab-gt/pairsearch commit '+SOURCE_COMMIT+', enusvm/nusvm_sim.m, git blob '+SOURCE_BLOB+'. '
        'See source_provenance/SOURCE_LOCK.json for verified hash and line anchors. '
        'The pinned repository root contains no LICENSE/COPYING; no broad reuse permission is inferred. '
        'Only one source file and README are archived as research evidence. No full upstream repository is copied into executable code.\n\n'
        '[TRANSFER] GaussCloud-Q (author-code schedule; continuous/shared-Bayes transfer): '
        'initial_scale_source_rule: max_coordinate_sd_of_fixed_catalog; '
        'initial_scale_transfer: analytic_max_coordinate_sd_of_initial_uniform_prior; '
        'initial_std: 1/sqrt(3); initial_scale_is_an_adaptation: true. '
        'Centers are shared-posterior means at stage entry. Continuous independent pairs replace catalog retrieval '
        'and enusvm estimation. The existing current-mean radius-32 cap is retained. '
        'SG-legacy differs in initial scale as well as schedule, so their comparison is not an isolated causal schedule effect. '
        'Finite-horizon GCQ-L is not asserted to satisfy an infinite-policy vanishing-radius theorem.\n\n'
        '[TRANSFER] SM uses principal-hyperplane Gaussian reflection; MEM family uses the binary coordinate line; '
        '2MEM stops after slots 1,2,1,2; IT/IB are specified trace-scale/bounded component transfers, '
        'not end-to-end reproductions. All actual answers use fixed-beta squared-distance logit and the shared posterior target.\n\n'
        '[ENGINEERING] New output root, stable method-purpose RNG streams, bitwise existing latent reuse, '
        'atomic checkpoints, delayed queue admission, GPU0/1/2 only. Main eta_grid preserves the baseline '
        'seven-scale denominator; inherited full-dimensional ellipsoid bounds use alpha_E=.5,rho_E=16 separately. '
        'All methodological definitions, seeds, scales and stage counts are frozen before results; outcomes cannot trigger tuning.\n')
    verify_lock(root)
    return lock


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','verify','inspect'])
    parser.add_argument('--run-dir',default=str(DEFAULT_ROOT))
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare_lock(args.run_dir)
        print(json.dumps(dict(source_run=result['source_run'],interactions=result['interaction_ids'],
                              locked_files=len(result['immutable_inventory']))))
    elif args.action=='verify': print(json.dumps(verify_lock(args.run_dir)))
    else:
        b,e,c=discover();print(json.dumps(dict(source_run=str(b),ellipsoid_source_run=str(e),
                                               interaction_ids=c['experiment']['interaction_ids'])))
