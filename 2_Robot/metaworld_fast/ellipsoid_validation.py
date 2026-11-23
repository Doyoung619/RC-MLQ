"""Prespecified independent numerical and optimizer-sensitivity audits.

Every query and answer remains frozen. Larger integration/replay budgets and
stronger optimization are diagnostic only; main reevaluated curves are untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from .util import atomic_json, digest, utcnow
from .model import stable_seed

SETTINGS = ("m04_b04", "m16_b04", "m04_b16")
METHODS = ("RC", "PQ", "QI", "PS", "GC", "SG")
INTERACTIONS = (0, 5)
ROUNDS = (0, 64, 127)
VERSION = "ellipsoid-independent-references-v1"


def numerical_comparison(original, references, *,relative_tolerance=.005, absolute_tolerance=None):
    values=np.asarray(references,dtype=np.float64)
    mean=float(np.mean(values))
    se=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    error=abs(float(original)-mean)
    tolerance=float(absolute_tolerance) if absolute_tolerance is not None else relative_tolerance*abs(mean)
    allowance=tolerance+3*(se or 0.)
    return dict(original=float(original),reference_values=values.tolist(),reference_mean=mean,
                reference_se=se,absolute_difference=error,
                relative_difference=error/abs(mean) if mean else None,tolerance=tolerance,
                decision="within_reference_uncertainty" if np.isfinite(error) and error<=allowance else "unresolved",
                interpretation="Descriptive 3-SE numerical comparison, not a confidence certificate; only two independent references.")


def optimizer_comparison(original, refined, upper):
    improvement=float(refined-original)
    relative=improvement/abs(float(original)) if original else None
    return dict(original_achieved=float(original),refined_achieved=float(refined),
                absolute_improvement=improvement,relative_improvement=relative,
                analytic_upper=float(upper),remaining_gap=float(upper-refined),
                warm_witness_retained=bool(improvement>=-1e-10),
                budget_sensitive=bool(relative is not None and relative>.005),
                global_optimum_certified=False)


class Progress:
    def __init__(self,path):
        self.path=path;self.started=time.monotonic();self.lock=threading.Lock();self.stop=threading.Event()
        self.state=dict(stage="starting",histories_completed=0,expected_histories=36,
                        snapshots_completed=0,expected_snapshots=108)
        self.thread=threading.Thread(target=self._heartbeat,daemon=True)
    def _write(self):
        atomic_json(self.path,dict(self.state,updated_at=utcnow(),elapsed_seconds=time.monotonic()-self.started))
    def update(self,**values):
        with self.lock:self.state.update(values);self._write()
    def _heartbeat(self):
        while not self.stop.wait(10):
            with self.lock:self._write()
    def __enter__(self):self.update();self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.update()


def _evaluate(points,weights,queries,beta,device):
    from .ellipsoid_optimizer import eig_value_gradient
    values,_=eig_value_gradient(points,weights,np.asarray(queries),beta,device=device,gradient=False)
    return np.asarray(values)


def _moments_error(mu,cov,reference_moments):
    means=np.stack([r[0] for r in reference_moments]);covs=np.stack([r[1] for r in reference_moments])
    mean_mu=means.mean(axis=0);mean_cov=covs.mean(axis=0)
    risk=np.trace(cov)
    result=dict(mean_l2_difference=float(np.linalg.norm(mu-mean_mu)),
                covariance_frobenius_difference=float(np.linalg.norm(cov-mean_cov)),
                original_mean=mu.tolist(),reference_mean=mean_mu.tolist(),
                original_covariance=cov.tolist(),reference_covariance=mean_cov.tolist(),
                risk=numerical_comparison(risk,[np.trace(c) for c in covs]),
                coordinate_variance=[numerical_comparison(cov[i,i],covs[:,i,i]) for i in range(len(mu))])
    return result


def _reference_particle_replays(data,beta,identity,protocol,device,progress):
    from .statistical_runner import replay_history
    results=[]
    for repeat in (0,1):
        progress.update(stage="independent_8192_particle_replay",current_history=identity,reference_replica=repeat)
        seed=stable_seed(protocol.get("root_seed",20261005),identity,repeat,None,"ellipsoid_independent_smc_reference")
        results.append(replay_history(data["q"],data["y"],beta,"joint",particles=8192,
                                      seed=seed,checkpoints=ROUNDS,device=device))
    return results


def _validate_history(source_path,evaluated_path,protocol,device,progress):
    from .ellipsoid_posterior import PosteriorReplay, stack_snapshots
    from .ellipsoid_optimizer import optimize_ellipsoid
    identity=source_path.stem;interaction=int(identity.split("__i")[-1])
    with np.load(evaluated_path,allow_pickle=False) as archive:
        evaluated={k:archive[k] for k in archive.files}
    indices={int(t):i for i,t in enumerate(evaluated["t"])}
    missing=[t for t in ROUNDS if t not in indices]
    if missing:raise ValueError(f"Required saved evaluation rounds missing: {missing}")
    progress.update(stage="original_posterior_replay",current_history=identity)
    replay=PosteriorReplay(source_path,marginal_samples=int(protocol.get("marginal_samples",4096)),device=device,rounds=ROUNDS)
    snapshots=list(replay)
    with np.load(source_path,allow_pickle=False) as archive:
        source={k:archive[k] for k in ("q","y")}
    marginal=snapshots[0].marginal_laws is not None
    references=None if marginal else _reference_particle_replays(source,snapshots[0].beta,identity,protocol,device,progress)
    comparisons=[]
    for snapshot in snapshots:
        t=snapshot.t;i=indices[t]
        achieved_q=evaluated["achieved_q"][i]
        questions=np.stack([snapshot.actual_q,achieved_q])
        original_values=_evaluate(snapshot.points,snapshot.weights,questions,snapshot.beta,device)
        saved_values=np.array([evaluated["eig_actual"][i],evaluated["eig_achieved"][i]])
        matches=bool(np.allclose(original_values,saved_values,rtol=2e-10,atol=2e-12))
        estimates=[];moments=[]
        for repeat in (0,1):
            if marginal:
                points,weights=snapshot.integration(count=8192,replicate=100+repeat)
            else:
                reference=references[repeat][t]
                points,weights=reference.particles,reference.weights
                moments.append(reference.moments())
            estimates.append(_evaluate(points,weights,questions,snapshot.beta,device))
        estimates=np.asarray(estimates)
        base_ratio=float(saved_values[0]/saved_values[1])
        rows=dict(t=t,kind="independent_joint_product_integration" if marginal else "independent_posterior_approximation",
                  reference_sample_count=8192,independent_reference_replicates=2,
                  original_same_bank_reproduced=matches,
                  original_same_bank_max_abs_difference=float(np.max(np.abs(original_values-saved_values))),
                  actual_eig=numerical_comparison(saved_values[0],estimates[:,0]),
                  fixed_achieved_query_eig=numerical_comparison(saved_values[1],estimates[:,1]),
                  fixed_query_ratio=numerical_comparison(base_ratio,estimates[:,0]/estimates[:,1],absolute_tolerance=.005),
                  note="Both saved queries are fixed. The reference query is not reoptimized on the held-out law; these ratios diagnose numerical sensitivity and do not replace main curves.")
        if not marginal:
            rows["posterior_moments"]=_moments_error(snapshot.mu,snapshot.cov,moments)
            rows["integration_note"]="Weighted-particle EIG is summed exactly for each numerical particle law. Differences across independent SMC replays measure posterior approximation, not a second Monte Carlo quadrature on the same particle set."
        else:
            rows["integration_note"]="Independent scrambled Sobol banks target the same saved continuous product posterior; its marginal quadrature approximation is unchanged."
        comparisons.append(rows)
    optimizer=[]
    if interaction==0:
        progress.update(stage="doubled_optimizer_budget",current_history=identity)
        batch=stack_snapshots(snapshots)
        seeds=[stable_seed(protocol.get("root_seed",20261005),identity,s.t,s.method,"ellipsoid_optimizer_sensitivity") for s in snapshots]
        warm=np.stack([evaluated["achieved_q"][indices[s.t]] for s in snapshots])
        result=optimize_ellipsoid(**{k:batch[k] for k in ("points","weights","mu","cov","beta","actual_q","grid_q")},
                                  device=device,alpha=protocol.get("alpha",.5),radius=protocol.get("radius",16.),
                                  starts=2*int(protocol.get("starts",8)),steps=2*int(protocol.get("steps",80)),
                                  learning_rate=protocol.get("learning_rate",.002),seed=seeds,warm_q=warm)
        for n,snapshot in enumerate(snapshots):
            i=indices[snapshot.t]
            optimizer.append(dict(t=snapshot.t,**optimizer_comparison(evaluated["eig_achieved"][i],result["achieved_eig"][n],evaluated["eig_upper"][i]),
                                  refined_query=result["best_q"][n].tolist(),starts=result["starts"],steps=result["steps"]))
    unresolved=sum(c[k]["decision"]=="unresolved" for c in comparisons for k in ("actual_eig","fixed_achieved_query_eig","fixed_query_ratio"))
    moment_unresolved=sum(m["decision"]=="unresolved" for c in comparisons if "posterior_moments" in c
                          for m in [c["posterior_moments"]["risk"],*c["posterior_moments"]["coordinate_variance"]])
    same_bank_failures=sum(not c["original_same_bank_reproduced"] for c in comparisons)
    warm_failures=sum(not r["warm_witness_retained"] for r in optimizer)
    sensitive=sum(r["budget_sensitive"] for r in optimizer)
    return dict(run_key=identity,status="unresolved" if unresolved or moment_unresolved or same_bank_failures or warm_failures or sensitive else "within_reference_uncertainty",
                marginal=marginal,comparisons=comparisons,optimization_sensitivity=optimizer,
                unresolved_eig_or_ratio_checks=unresolved,same_bank_reproduction_failures=same_bank_failures,
                unresolved_posterior_moment_checks=moment_unresolved,
                optimization_budget_sensitive_snapshots=sensitive,warm_witness_failures=warm_failures,
                original_replay_summary=replay.summary(),main_evaluation_modified=False)


def validate_evaluation(run_dir,device="cuda:3"):
    root=Path(run_dir);manifest=json.loads((root/"manifest.json").read_text())
    source=Path(manifest["source_run"]);protocol=manifest["protocol"]
    prior_audit_path=source/"validation/summary.json"
    prior_audit=json.loads(prior_audit_path.read_text()) if prior_audit_path.exists() else {}
    output=root/"validation";output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();results=[];errors=[]
    own_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Progress(output/"progress.json") as progress:
        for setting in SETTINGS:
            for method in METHODS:
                for interaction in INTERACTIONS:
                    key=f"{setting}__{method}__i{interaction}"
                    source_path=source/"statistical"/f"{key}.npz"
                    evaluated_path=root/"evaluated"/f"{key}.npz"
                    path=output/f"{key}.json"
                    try:
                        signature=digest(dict(source=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                              evaluated=hashlib.sha256(evaluated_path.read_bytes()).hexdigest(),
                                              protocol=protocol,validation_code_hash=own_hash,version=VERSION))
                        previous=json.loads(path.read_text()) if path.exists() else {}
                        if previous.get("signature")==signature:
                            result=previous
                        else:
                            result=_validate_history(source_path,evaluated_path,protocol,device,progress)
                            result.update(signature=signature,validation_version=VERSION)
                            atomic_json(path,result)
                        results.append(result)
                        print(f"ELLIPSOID VALIDATION {key}: {result['status']}",flush=True)
                    except Exception as exc:
                        error=dict(run_key=key,error=f"{type(exc).__name__}: {exc}")
                        errors.append(error);atomic_json(output/f"{key}.error.json",error)
                        print(f"ELLIPSOID VALIDATION {key}: {error['error']}",flush=True)
                    progress.update(histories_completed=len(results),snapshots_completed=3*len(results),errors=len(errors))
        unresolved=[r["run_key"] for r in results if r["status"]=="unresolved"]
        summary=dict(status="unresolved" if unresolved or errors or len(results)!=36 else "within_reference_uncertainty",
                     validation_version=VERSION,expected_histories=36,completed_histories=len(results),
                     expected_snapshots=108,completed_snapshots=3*len(results),
                     selection=dict(settings=list(SETTINGS),methods=list(METHODS),interaction_ids=list(INTERACTIONS),preanswer_t=list(ROUNDS)),
                     unresolved_histories=unresolved,errors=errors,
                     independent_reference_count=8192,independent_replicates=2,
                     optimization_sensitivity_snapshots=sum(len(r["optimization_sensitivity"]) for r in results),
                     optimization_budget_sensitive_snapshots=sum(r["optimization_budget_sensitive_snapshots"] for r in results),
                     same_bank_reproduction_failures=sum(r["same_bank_reproduction_failures"] for r in results),
                     unresolved_posterior_moment_checks=sum(r["unresolved_posterior_moment_checks"] for r in results),
                     original_numerical_audit_status=prior_audit.get("status","not_available"),
                     original_numerical_unresolved_histories=prior_audit.get("unresolved_histories",[]),
                     original_numerical_audit_path=str(prior_audit_path),
                     posterior_approximation_note="Independent SMC references change only the numerical posterior approximation on exactly the saved q/y history; coordinate-marginal approximation uncertainty remains documented in the original numerical audit.",
                     integration_note="Marginal references use two independent 8192-point joint Sobol banks. SMC sums weighted EIG on each independent8192-particle posterior. Two-replicate reference SE is imprecise and descriptive.",
                     optimization_note="Only interaction0 has doubled starts/steps on the original integration bank, retaining the original achieved query as a warm witness. These diagnostic improvements do not modify any main curve or certify a global optimum.",
                     main_evaluation_modified=False,global_optimum_certified=False,
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(output/"summary.json",summary)
        progress.update(stage="complete",status=summary["status"])
    return summary
