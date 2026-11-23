"""One statistically independent interaction, with atomic resumable artifacts."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
from .model import stable_seed, probabilities
from .line_template import make_template
from .metrics import information_ratio
from .posterior_marginal import MarginalPosterior
from .posterior_smc import SMCPosterior
from .selectors import SelectorState, select_query


def run_key(job):
    return f"{job['setting_id']}__{job['method']}__i{int(job['interaction_id'])}"


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with open(temporary,"wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def _atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with open(temporary,"wb") as stream:
        np.savez(stream,**arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def _default_code_hash():
    digest = hashlib.sha256()
    for name in ("model.py","metrics.py","line_template.py","selectors.py","posterior_marginal.py","posterior_smc.py","statistical_runner.py"):
        digest.update((Path(__file__).parent/name).read_bytes())
    return digest.hexdigest()


def _make_state(job):
    root = int(job.get("root_seed",20261004))
    setting,method,interaction = job["setting_id"],job["method"],int(job["interaction_id"])
    seeds = {purpose:stable_seed(root,setting,interaction,None if purpose=="theta" else method,purpose) for purpose in ("theta","response","selector","smc","diagnostics")}
    rng = {purpose:np.random.default_rng(seed) for purpose,seed in seeds.items()}
    d = int(job.get("dimension",4))
    theta = rng["theta"].uniform(-1,1,d)
    if method in {"RC","PQ","QI"}:
        posterior = MarginalPosterior(d,float(job["beta"]),int(job.get("nodes",257)))
    else:
        posterior = SMCPosterior(d,float(job["beta"]),int(job.get("particles",4096)),rng["smc"],job.get("device","cpu"))
    mean,cov = posterior.moments()
    data = {key:[] for key in ("q","q_raw","y","eig","reference_eig","eta","reference_queries","reference_eigs","fisher_scores","ratio_status","diag_t","diag_eig","diag_eta","diag_se","update_diagnostics","selector_stage","selector_center")}
    data.update(mu=[mean],cov=[cov],risk=[float(np.trace(cov))],error=[float(np.sum((mean-theta)**2))])
    return {"posterior":posterior,"rng":rng,"selector_state":SelectorState(),"theta":theta,"seeds":seeds,"data":data,"elapsed_seconds":0.,"job":copy.deepcopy(job),"answers_completed":0,"pending_rollout_request_ids":[]}


def _arrays(state):
    data = state["data"]
    arrays = {k:np.asarray(v) for k,v in data.items() if k not in {"update_diagnostics","ratio_status"}}
    arrays["theta"] = state["theta"]
    arrays["ratio_status"] = np.asarray(data["ratio_status"],dtype="U64")
    arrays["answers_completed"] = np.asarray(state["answers_completed"])
    return arrays


def run_statistical(job, output_dir, resume=True, max_answers=None, on_checkpoint=None):
    """Run/restart one job. ``max_answers`` supports interrupted-run validation.

    Required job fields: setting_id, M, beta, method, interaction_id.
    Optional: root_seed, answers(128), dimension(4), particles(4096),
    nodes(257), device(cpu), code_hash, config_hash, adapter_hash.
    """
    job = copy.deepcopy(job)
    job.setdefault("answers",128)
    job.setdefault("root_seed",20261004)
    job.setdefault("dimension",4)
    job.setdefault("particles",4096)
    job.setdefault("nodes",257)
    job.setdefault("device","cpu")
    job.setdefault("code_hash",_default_code_hash())
    output_dir=Path(output_dir)
    for directory in ("statistical","checkpoints"):
        (output_dir/directory).mkdir(parents=True,exist_ok=True)
    key=run_key(job)
    checkpoint=output_dir/"checkpoints"/(key+".pkl")
    progress_path=output_dir/"checkpoints"/(key+".progress.json")
    final_path=output_dir/"statistical"/(key+".npz")
    metadata_path=output_dir/"statistical"/(key+".json")
    # A lock prevents two workers from writing the same run concurrently.
    import fcntl
    lock=open(output_dir/"checkpoints"/(key+".lock"),"a+b")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Run already owned by another process: {key}")
    try:
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint}")
            with open(checkpoint,"rb") as stream:
                state=pickle.load(stream)
            if state["job"] != job:
                raise ValueError(f"Checkpoint job/code/config differs for {key}; use a new output run directory")
        else:
            state=_make_state(job)
        target=int(job["answers"] if max_answers is None else min(job["answers"],max_answers))
        posterior,rng,data=state["posterior"],state["rng"],state["data"]
        start=time.monotonic()
        prior_elapsed=float(state["elapsed_seconds"])
        for t in range(state["answers_completed"],target):
            template=make_template(posterior,int(job["M"]))
            raw,q=select_query(job["method"],posterior,int(job["M"]),t,rng["selector"],state["selector_state"],template)
            reference_eigs=np.array([posterior.eig(question) for question in template.queries])
            eig=posterior.eig(q)
            reference=float(reference_eigs.max())
            eta,status=information_ratio(eig,reference)
            if job["method"]=="RC" and eta > 1+1e-10:
                raise FloatingPointError("RC ratio exceeds its own seven-scale reference")
            if job["method"] in {"PS","GC","SG"} and t in (0,31,63,95,127):
                diagnostic_state=copy.deepcopy(state["selector_state"])
                values=[]
                for _ in range(64):
                    _,diagnostic_q=select_query(job["method"],posterior,int(job["M"]),t,rng["diagnostics"],diagnostic_state,template)
                    values.append(posterior.eig(diagnostic_q))
                values=np.asarray(values)
                data["diag_t"].append(t)
                data["diag_eig"].append(values)
                data["diag_eta"].append(float(values.mean()/reference) if status=="resolved" else float("nan"))
                data["diag_se"].append(float(values.std(ddof=1)/np.sqrt(64)/reference) if status=="resolved" else float("nan"))
            y=int(rng["response"].choice(int(job["M"]),p=probabilities(state["theta"],q,float(job["beta"]))))
            for name,value in (("q",q),("q_raw",raw),("y",y),("eig",eig),("reference_eig",reference),("eta",eta),("reference_queries",template.queries),("reference_eigs",reference_eigs),("fisher_scores",template.fisher_scores),("ratio_status",status)):
                data[name].append(value)
            data["selector_stage"].append(state["selector_state"].stage if state["selector_state"].stage is not None else -1)
            data["selector_center"].append(state["selector_state"].center.copy() if state["selector_state"].center is not None else np.full(job["dimension"],np.nan))
            posterior.update(q,y)
            data["update_diagnostics"].append(copy.deepcopy(posterior.last_diagnostics))
            mean,cov=posterior.moments()
            data["mu"].append(mean)
            data["cov"].append(cov)
            data["risk"].append(float(np.trace(cov)))
            data["error"].append(float(np.sum((mean-state["theta"])**2)))
            state["answers_completed"]=t+1
            state["pending_rollout_request_ids"].extend(f"{key}:t{t}:m{m}" for m in range(int(job["M"])))
            if (t+1)%16==0 or t+1==target:
                state["elapsed_seconds"]=prior_elapsed+time.monotonic()-start
                _atomic_bytes(checkpoint,pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL))
                progress={"run_key":key,"job":job,"answers_completed":t+1,"answers_target":job["answers"],"elapsed_seconds":state["elapsed_seconds"],"stage":"complete" if t+1==job["answers"] else "running","mean_seconds_per_answer":state["elapsed_seconds"]/(t+1),"latest_update_seconds":posterior.last_diagnostics.get("seconds",0.),"pid":os.getpid(),"time_unix":time.time()}
                _atomic_bytes(progress_path,json.dumps(progress,indent=2).encode())
                _atomic_npz(output_dir/"checkpoints"/(key+".partial.npz"),_arrays(state))
                if on_checkpoint is not None:
                    on_checkpoint(progress)
        if state["answers_completed"]==job["answers"]:
            _atomic_npz(final_path,_arrays(state))
            metadata={"run_key":key,**job,"seeds":state["seeds"],"answers_completed":state["answers_completed"],"elapsed_seconds":state["elapsed_seconds"],"update_diagnostics":data["update_diagnostics"],"coordinate_model_oracle":True,"reference_scope":"own_posterior_seven_scale_grid","initial_exact_prior_risk":job["dimension"]/3,"eigenspace_rtol":1e-10,"numerical_engine":"adaptive_sinh_simpson_positive_continuous_pchip_cdf" if posterior.factorized else "tempered_resample_move_exact_delayed_acceptance","sg_actual_stages":sorted(set(data["selector_stage"])) if job["method"]=="SG" else [],"checkpoint":str(checkpoint),"artifact":str(final_path)}
            _atomic_bytes(metadata_path,json.dumps(metadata,indent=2).encode())
        return {"run_key":key,"answers_completed":state["answers_completed"],"elapsed_seconds":state["elapsed_seconds"],"complete":state["answers_completed"]==job["answers"],"artifact":str(final_path) if final_path.exists() else None,"checkpoint":str(checkpoint)}
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def replay_history(q,y,beta,kind,nodes=513,particles=8192,seed=1001,checkpoints=(0,32,64,128),device="cpu"):
    q=np.asarray(q,dtype=np.float64)
    posterior=MarginalPosterior(q.shape[2],beta,nodes) if kind in {"marginal","RC","PQ","QI"} else SMCPosterior(q.shape[2],beta,particles,np.random.default_rng(seed),device)
    result={}
    if 0 in checkpoints:
        result[0]=copy.deepcopy(posterior)
    for t,(question,answer) in enumerate(zip(q,y),start=1):
        posterior.update(question,int(answer))
        if t in checkpoints:
            result[t]=copy.deepcopy(posterior)
    return result
