"""Measured engineering workload, disjoint from final interaction streams."""
from __future__ import annotations
import concurrent.futures as cf
import copy
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import time
from .util import atomic_json, digest, source_manifest, utcnow

def _profile_statistical(job, output):
    from .statistical_runner import run_statistical
    start=time.perf_counter()
    result=run_statistical(job,output,resume=True)
    return {'job':job,'wall_seconds':result['elapsed_seconds'],'call_wall_seconds':time.perf_counter()-start,'result':result}

def _backend_benchmark(checkpoint, cuda_device):
    import numpy as np
    import torch
    with open(checkpoint,'rb') as f: saved=pickle.load(f)
    p=saved['posterior']
    if not hasattr(p,'target_logdensity'): return None
    points=p.particles.copy()
    measurements={};values={}
    for device in ('cpu', cuda_device):
        p.device=device;p._torch_coefficients=None
        p.target_logdensity(points)
        timings=[]
        for _ in range(3):
            start=time.perf_counter();values[device]=p.target_logdensity(points)
            if device == cuda_device: torch.cuda.synchronize(cuda_device)
            timings.append(time.perf_counter()-start)
        measurements[device]=float(np.median(timings))
    discrepancy=float(np.max(np.abs(values['cpu']-values[cuda_device])))
    if not np.allclose(values['cpu'],values[cuda_device],rtol=1e-11,atol=1e-10):
        raise RuntimeError(f'CPU/CUDA frozen full-history target discrepancy {discrepancy}')
    return {'history_length':len(p.history),'particles':p.n,'timings_seconds':measurements,'max_abs_difference':discrepancy}

def profile_workload(run_dir,config):
    root=Path(run_dir);root.mkdir(parents=True,exist_ok=True)
    profile_path=root/'PROFILE.json'
    code=digest(source_manifest())
    if profile_path.exists():
        saved=json.loads(profile_path.read_text())
        if saved.get('code_hash')==code and saved.get('complete'): return saved
    import torch
    cuda_count = torch.cuda.device_count()
    cuda_device = ('cuda:3' if cuda_count >= 4 else 'cuda:0') if cuda_count else None
    available_cpus = len(os.sched_getaffinity(0))
    cpu_workers = min(8, max(1, available_cpus // 4))
    marginal_workers = min(4, available_cpus)
    from .contexts import prepare_contexts
    contexts=prepare_contexts(root/'contexts')
    from .renderer import calibrate
    calibration=calibrate(contexts,root/'diagnostics'/'engineering_calibration.json')
    from .rollout_pool import benchmark
    worker_counts = [4, 8, 12, 16, 20, 24] if available_cpus >= 4 else [available_cpus]
    robot=benchmark(root/'contexts',root/'profiling'/'rollouts',candidates=worker_counts,clips_per_worker=256)
    ctx=mp.get_context('spawn')
    directory=root/'profiling'/'statistics'
    directory.mkdir(parents=True,exist_ok=True)
    from .statistical_runner import _default_code_hash
    statistical_hash=_default_code_hash()
    jobs=[]
    for sid,m,b in [('profile_m16_b04',16,4.),('profile_m04_b16',4,16.)]:
        for method in ('RC','QI','SG'):
            jobs.append(dict(setting_id=sid,M=m,beta=b,method=method,interaction_id=0,
                             root_seed=2026100401,answers=128,particles=4096,dimension=4,
                             device=(cuda_device or 'cpu') if method=='SG' else 'cpu',code_hash=statistical_hash,config_hash=digest(config)))
    # Isolate joint profiles from marginal profiles, including on CPU.
    statistics=[]
    with cf.ProcessPoolExecutor(max_workers=marginal_workers,mp_context=ctx) as cpus,cf.ProcessPoolExecutor(max_workers=1,mp_context=ctx) as joint:
        futures=[(joint if j['method']=='SG' else cpus).submit(_profile_statistical,j,str(directory)) for j in jobs]
        for f in cf.as_completed(futures):
            result=f.result();statistics.append(result)
            print(f"{utcnow()} profile completed {result['job']['setting_id']} {result['job']['method']} in {result['wall_seconds']:.2f}s",flush=True)
            atomic_json(root/'PROFILE.partial.json',{'robot':robot,'statistics':statistics})
    comparisons=[]
    if cuda_device is not None:
        with cf.ProcessPoolExecutor(max_workers=1,mp_context=ctx) as evaluator:
            for p in (directory/'checkpoints').glob('*SG*.pkl'):
                result=evaluator.submit(_backend_benchmark,p,cuda_device).result()
                if result: comparisons.append(result)
    # Per-run exact targets favor GPU only if faster even after allowing CPU concurrency.
    ratios=[c['timings_seconds']['cpu']/c['timings_seconds'][cuda_device] for c in comparisons]
    cpu_ratio = sum(ratios)/len(ratios) if ratios else 1.
    choose_gpu = cuda_device is not None and (not ratios or cpu_ratio > cpu_workers)
    resources={'marginal_workers':marginal_workers,'joint_workers':1 if choose_gpu else cpu_workers,
               'joint_device':cuda_device if choose_gpu else 'cpu',
               'rollout_workers':robot['recommended_workers'],'rollout_chunk_candidates':64,
               'gpu_inference_processes':1 if choose_gpu else 0}
    joint_times=[r['wall_seconds'] for r in statistics if r['job']['method']=='SG']
    marginal_times=[r['wall_seconds'] for r in statistics if r['job']['method']!='SG']
    joint_seconds=sum(joint_times)/len(joint_times)*105
    if not choose_gpu: joint_seconds*=cpu_ratio/cpu_workers
    marginal_seconds=sum(marginal_times)/len(marginal_times)*105/marginal_workers
    inference_seconds=max(joint_seconds,marginal_seconds)
    rollout_seconds=(161280+3528)/robot['measured_clips_per_second']
    # Two8192 references for nine joint frozen histories, bounded optional16384;
    # extra margin accounts for small quadrature cases, reports and36 videos.
    postprocessing_seconds=(sum(joint_times)/len(joint_times))*60+600
    # Unmeasured overlap contention is explicitly reserved; live updates will refine it.
    combined=max(inference_seconds,rollout_seconds)*1.25
    result={'complete':True,'measured_at':utcnow(),'code_hash':code,'resources':resources,
            'robot':robot,'statistical_profiles':statistics,'frozen_cpu_gpu_comparisons':comparisons,
            'engineering_calibration':calibration,
            'estimated_statistical_seconds':inference_seconds*1.25,
            'estimated_rollout_seconds':rollout_seconds*1.25,
            'estimated_postprocessing_seconds':postprocessing_seconds,
            'estimated_pipeline_seconds':combined+postprocessing_seconds,
            'estimate_assumptions':['Full128 histories at M16,beta4 and M4,beta16 measured; RC,QI,SG timings cover representative paths.',
                'SG timing is a proxy for PS/GC and may differ by resampling frequency.',
                'CPU/GPU comparison uses identical final frozen history; CPU parallel scaling is an initial estimate.',
                '25 percent resource contention reserve; no measured overlap throughput claimed by isolated profile.',
                'Validation reserve assumes two8192 independent replays and bounded16384 refinements; videos/report add10min.'],
            'profile_root_seed':2026100401,'profile_not_final_data':True}
    atomic_json(profile_path,result)
    return result
