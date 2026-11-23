"""Frozen-history posterior replay and joint integration for ellipsoid audits.

Only actual saved (q,y) pairs update a posterior. No selectors, response draws,
robot environments, or optimization variables enter the replay. Existing
posterior engines remain untouched. A product posterior is integrated using
four-dimensional scrambled Sobol points transformed through each continuous
PCHIP CDF; arbitrary reference geometries never use a coordinate shortcut.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Iterator
import numpy as np
from scipy.stats import qmc
from .model import stable_seed
from .posterior_marginal import MarginalPosterior
from .posterior_smc import SMCPosterior

REPLAY_VERSION = "frozen-history-joint-integral-v1"
MOMENT_RTOL = 2e-10
MOMENT_ATOL = 2e-12


@lru_cache(maxsize=32)
def sobol_bank(dimension: int, count: int, seed: int):
    """A separate integration stream. Rows target the joint product law."""
    if count < 2 or count & (count-1):
        raise ValueError("Scrambled Sobol integration count must be a power of two")
    bank = qmc.Sobol(d=int(dimension), scramble=True, seed=int(seed)).random_base2(int(count).bit_length()-1)
    bank.setflags(write=False)
    return bank


@dataclass(frozen=True)
class MarginalInverseCDF:
    """An immutable copy of a continuous marginal CDF, without its history."""
    knots: np.ndarray
    coefficients: np.ndarray
    cdf_knots: np.ndarray
    total: float
    origin: float
    center: float
    scale: float | None

    @classmethod
    def from_marginal(cls, marginal):
        integral=marginal._cdf_integral
        arrays=[np.asarray(marginal.u).copy(),np.asarray(integral.c).copy(),
                (np.asarray(integral(marginal.u))-marginal._cdf_origin)/marginal._cdf_total]
        for a in arrays:a.setflags(write=False)
        return cls(*arrays,float(marginal._cdf_total),float(marginal._cdf_origin),
                   float(marginal.center),None if marginal.scale is None else float(marginal.scale))

    def quantiles(self, probabilities):
        """Vectorized bracketed Newton inversion of each quartic CDF piece.

        Interval-local polynomial coefficients avoid scipy/root calls per
        sample. A bounded bisection fallback handles zero-density tails.
        """
        p=np.asarray(probabilities,dtype=np.float64)
        original_shape=p.shape;p=p.ravel()
        if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
            raise ValueError("CDF probabilities must lie in [0,1]")
        index=np.searchsorted(self.cdf_knots,p,side="right")-1
        index=np.clip(index,0,len(self.knots)-2)
        coefficients=self.coefficients[:,index]
        left=self.knots[index]
        width=self.knots[index+1]-left
        low=np.zeros(len(p));high=width.copy()
        cp0=self.cdf_knots[index];cp1=self.cdf_knots[index+1]
        fraction=np.divide(p-cp0,cp1-cp0,out=np.full(len(p),.5),where=cp1>cp0)
        x=np.clip(fraction,0,1)*width
        target=self.origin+p*self.total
        active=(p>0)&(p<1)
        # Degree four antiderivative of the cubic PCHIP density.
        c0,c1,c2,c3,c4=coefficients
        for _ in range(18):
            value=((((c0*x+c1)*x+c2)*x+c3)*x+c4)-target
            derivative=((4*c0*x+3*c1)*x+2*c2)*x+c3
            correction=np.divide(value,derivative,out=np.full(len(p),np.inf),where=derivative>0)
            converged=(value==0)|(np.abs(correction)<=2e-14*np.maximum(1,np.abs(left+x)))
            active &= ~converged
            if not np.any(active):break
            low=np.where(active&(value<0),x,low)
            high=np.where(active&(value>=0),x,high)
            candidate=x-correction
            use_newton=np.isfinite(candidate)&(candidate>low)&(candidate<high)
            candidate=np.where(use_newton,candidate,(low+high)*.5)
            x=np.where(active,candidate,x)
        for _ in range(42):
            if not np.any(active):break
            value=((((c0*x+c1)*x+c2)*x+c3)*x+c4)-target
            low=np.where(active&(value<0),x,low)
            high=np.where(active&(value>=0),x,high)
            converged=(value==0)|((high-low)<=4e-14*np.maximum(1,np.abs(left+x)))
            active &= ~converged
            x=np.where(active,(low+high)*.5,x)
        if np.any(active):raise FloatingPointError("Vectorized continuous CDF inversion did not converge")
        u=left+x
        result=u if self.scale is None else self.center+self.scale*np.sinh(u)
        result=np.where(p<=0,-1.,np.where(p>=1,1.,result))
        return result.reshape(original_shape)


def joint_product_points(laws, count=4096, seed=0):
    bank=sobol_bank(len(laws),int(count),int(seed))
    return np.column_stack([law.quantiles(bank[:,axis]) for axis,law in enumerate(laws)])


def _weighted_moments(points,weights):
    mu=weights@points;delta=points-mu
    cov=delta.T@(weights[:,None]*delta)
    return mu,(cov+cov.T)*.5


@dataclass
class PosteriorSnapshot:
    t: int
    points: np.ndarray
    weights: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    actual_q: np.ndarray
    grid_q: np.ndarray
    beta: float
    method: str
    run_key: str
    integration_seed: int
    original_eig: float
    original_grid_eigs: np.ndarray
    diagnostics: dict=field(default_factory=dict)
    marginal_laws: tuple | None=field(default=None,repr=False)

    def integration(self,count=8192,replicate=1):
        """Independent product-QMC bank; joint SMC particle law stays fixed.

        Reintegrating SMC with a larger/different particle posterior would be
        a separate posterior approximation audit, never integration refinement.
        """
        if self.marginal_laws is None:
            return self.points.copy(),self.weights.copy()
        seed=stable_seed(self.integration_seed,self.run_key,int(replicate),self.method,"ellipsoid_joint_integral_replica")
        points=joint_product_points(self.marginal_laws,int(count),seed)
        return points,np.full(int(count),1./int(count))

    def eig_comparison(self,queries=None,count=None,replicate=1):
        """Report same-bank EIG and independent integration discrepancies.

        Defaults to actual + all seven grid questions so the old one-coordinate
        integral can be compared without mixing it into the new ratio.
        """
        from .metrics import integrated_eig
        questions=np.concatenate((self.actual_q[None],self.grid_q),axis=0) if queries is None else np.asarray(queries)
        base=np.array([integrated_eig(self.points,self.weights,q,self.beta) for q in questions])
        report={"same_joint_bank_eigs":base.tolist(),"numerical_integration":"product_scrambled_sobol" if self.marginal_laws else "fixed_weighted_smc_particles"}
        if queries is None:
            original=np.r_[self.original_eig,self.original_grid_eigs]
            report.update(original_eigs=original.tolist(),joint_minus_original=(base-original).tolist(),
                          max_abs_original_discrepancy=float(np.max(np.abs(base-original))))
        if count is not None:
            points,weights=self.integration(count,replicate)
            alternate=np.array([integrated_eig(points,weights,q,self.beta) for q in questions])
            report.update(alternate_eigs=alternate.tolist(),alternate_count=len(points),
                          alternate_replicate=int(replicate),max_abs_alternate_discrepancy=float(np.max(np.abs(base-alternate))))
        return report


def stack_snapshots(snapshots):
    """Optimizer schema: B,N,d; B,N; B,d; B,d,d; B,M,d; B,7,M,d."""
    snapshots=list(snapshots)
    if not snapshots:raise ValueError("Cannot stack an empty posterior batch")
    payload={name:np.stack([getattr(s,name) for s in snapshots]) for name in ("points","weights","mu","cov","actual_q","grid_q")}
    payload.update(beta=np.array([s.beta for s in snapshots]),t=np.array([s.t for s in snapshots],dtype=np.int64),
                   snapshots=snapshots)
    return payload


class PosteriorReplay(Iterator[PosteriorSnapshot]):
    """Stateful, forward-only pre-answer snapshots with complete replay checks.

    On return of snapshot t, the saved answer t has already been replayed and
    checked. Thus receiving all 128 snapshots verifies all 129 stored moments,
    even if a consumer does not request one more item to exhaust an iterator.
    """
    def __init__(self,statistical_npz,marginal_samples=4096,device=None,integration_seed=None,
                 rounds=None,moment_rtol=MOMENT_RTOL,moment_atol=MOMENT_ATOL,strict=True):
        self.path=Path(statistical_npz).resolve()
        self.metadata=json.loads(self.path.with_suffix(".json").read_text())
        with np.load(self.path,allow_pickle=False) as archive:
            self.data={key:archive[key].copy() for key in ("q","y","mu","cov","reference_queries","eig","reference_eigs")}
        self.T=len(self.data["y"])
        if self.data["mu"].shape[0]!=self.T+1 or self.data["cov"].shape[0]!=self.T+1:
            raise ValueError("Saved posterior moments do not cover every actual history")
        self.method=self.metadata["method"];self.run_key=self.metadata.get("run_key",self.path.stem)
        self.dimension=int(self.metadata.get("dimension",4));self.beta=float(self.metadata["beta"])
        self.factorized=self.method in ("RC","PQ","QI")
        self.marginal_samples=int(marginal_samples)
        self.original_device=self.metadata.get("device","cpu")
        self.device=self.original_device if device is None else str(device)
        self.strict=bool(strict);self.moment_rtol=float(moment_rtol);self.moment_atol=float(moment_atol)
        self.rounds=None if rounds is None else frozenset(int(t) for t in rounds)
        if self.rounds is not None and any(t<0 or t>=self.T for t in self.rounds):raise ValueError("Requested pre-answer round outside history")
        self.integration_seed=int(integration_seed if integration_seed is not None else stable_seed(
            int(self.metadata.get("root_seed",20261004)),self.metadata["setting_id"],int(self.metadata["interaction_id"]),self.method,"ellipsoid_joint_integral_sobol_v1"))
        self.engine_hashes=self._check_engine_sources()
        if self.factorized:
            self.posterior=MarginalPosterior(self.dimension,self.beta,int(self.metadata.get("nodes",257)))
            self._bank=sobol_bank(self.dimension,self.marginal_samples,self.integration_seed)
            self._product_points=np.empty((self.marginal_samples,self.dimension))
            self._marginal_versions=[None]*self.dimension;self._laws=[None]*self.dimension
            self._uniform_weights=np.full(self.marginal_samples,1./self.marginal_samples)
        else:
            seed=int(self.metadata["seeds"]["smc"])
            self.posterior=SMCPosterior(self.dimension,self.beta,int(self.metadata.get("particles",4096)),np.random.default_rng(seed),self.device)
        self.t=0;self.started=time.perf_counter();self.replay_seconds=0.;self.integration_seconds=0.
        self.max_mean_error=0.;self.max_cov_error=0.;self.moment_checks=[];self.terminal_state_check=None
        self._check_moments(0)

    def _check_engine_sources(self):
        root=self.path.parent.parent
        source=root/"source_snapshot"
        output={}
        for name in ("posterior_smc.py","posterior_marginal.py","model.py","metrics.py"):
            current=Path(__file__).with_name(name).read_bytes()
            original=(source/name).read_bytes() if (source/name).exists() else None
            equal=original is None or current==original
            output[name]={"used_sha256":hashlib.sha256(current).hexdigest(),
                          "frozen_sha256":None if original is None else hashlib.sha256(original).hexdigest(),
                          "matches_available_snapshot":equal}
            if self.strict and not equal:raise RuntimeError(f"Replay engine differs from frozen source: {name}")
        return output

    def _check_moments(self,t):
        mean,cov=self.posterior.moments()
        expected_mean,expected_cov=self.data["mu"][t],self.data["cov"][t]
        mean_error=float(np.max(np.abs(mean-expected_mean)))
        cov_error=float(np.max(np.abs(cov-expected_cov)))
        ok=bool(np.allclose(mean,expected_mean,rtol=self.moment_rtol,atol=self.moment_atol)
                and np.allclose(cov,expected_cov,rtol=self.moment_rtol,atol=self.moment_atol))
        self.max_mean_error=max(self.max_mean_error,mean_error);self.max_cov_error=max(self.max_cov_error,cov_error)
        check={"answers":int(t),"mean_max_abs_error":mean_error,"cov_max_abs_error":cov_error,"passed":ok}
        self.moment_checks.append(check)
        if self.strict and not ok:
            raise RuntimeError(f"Frozen-history posterior mismatch {self.run_key} at {t}: {check}")
        return mean,cov,check

    def _verify_terminal_state(self):
        import pickle
        checkpoint=Path(self.metadata.get("checkpoint",""))
        if not checkpoint.is_file():
            checkpoint=self.path.parent.parent/"checkpoints"/(self.path.stem+".pkl")
        if not checkpoint.is_file():
            self.terminal_state_check={"available":False,"reason":"terminal checkpoint unavailable; all saved moments checked"}
            return
        with checkpoint.open("rb") as stream:original=pickle.load(stream)["posterior"]
        pairs=[]
        if self.factorized:
            for i,(observed,expected) in enumerate(zip(self.posterior.marginals,original.marginals)):
                pairs.extend([(f"marginal_{i}_nodes",observed.x,expected.x),
                              (f"marginal_{i}_weights",observed.weights,expected.weights)])
            rng_equal=True
        else:
            pairs=[("particles",self.posterior.particles,original.particles),
                   ("logweights",self.posterior.logweights,original.logweights),
                   ("log_target",self.posterior.log_target,original.log_target),
                   ("proposal_scale",np.asarray(self.posterior.proposal_scale),np.asarray(original.proposal_scale))]
            rng_equal=self.posterior.rng.bit_generator.state==original.rng.bit_generator.state
        values={}
        for name,observed,expected in pairs:
            values[name]={"max_abs_error":float(np.max(np.abs(observed-expected))),
                          "bitwise_equal":bool(np.array_equal(observed,expected)),
                          "within_tolerance":bool(np.allclose(observed,expected,rtol=self.moment_rtol,atol=self.moment_atol))}
        passed=all(v["within_tolerance"] for v in values.values()) and rng_equal
        self.terminal_state_check={"available":True,"source":str(checkpoint),"arrays":values,
                                   "smc_rng_state_exact":bool(rng_equal),"passed":bool(passed)}
        if self.strict and not passed:raise RuntimeError(f"Terminal posterior state mismatch: {self.terminal_state_check}")

    def _snapshot(self):
        started=time.perf_counter();mean,cov=self.posterior.moments()
        if self.factorized:
            for axis,marginal in enumerate(self.posterior.marginals):
                version=(len(marginal.history),marginal.remesh_count)
                if version!=self._marginal_versions[axis]:
                    law=MarginalInverseCDF.from_marginal(marginal)
                    self._product_points[:,axis]=law.quantiles(self._bank[:,axis])
                    self._laws[axis]=law;self._marginal_versions[axis]=version
            points=self._product_points.copy();weights=self._uniform_weights.copy();laws=tuple(self._laws)
        else:
            points=self.posterior.particles.copy();weights=self.posterior.weights.copy();laws=None
        sample_mu,sample_cov=_weighted_moments(points,weights)
        elapsed=time.perf_counter()-started;self.integration_seconds+=elapsed
        diagnostics={"integration_count":len(points),"integration_seconds":elapsed,
                     "integration_method":"joint_product_scrambled_sobol_continuous_cdf" if laws else "original_weighted_smc_particles",
                     "integration_seed":self.integration_seed if laws else None,
                     "sample_mean_max_abs_error":float(np.max(np.abs(sample_mu-mean))),
                     "sample_cov_max_abs_error":float(np.max(np.abs(sample_cov-cov))),
                     "replay_mean_max_abs_error":self.max_mean_error,"replay_cov_max_abs_error":self.max_cov_error,
                     "original_device":self.original_device,"replay_device":self.device,
                     "all_queries_share_integration_bank":True}
        return PosteriorSnapshot(t=self.t,points=points,weights=weights,mu=mean.copy(),cov=cov.copy(),
                                 actual_q=self.data["q"][self.t].copy(),grid_q=self.data["reference_queries"][self.t].copy(),
                                 beta=self.beta,method=self.method,run_key=self.run_key,integration_seed=self.integration_seed,
                                 original_eig=float(self.data["eig"][self.t]),original_grid_eigs=self.data["reference_eigs"][self.t].copy(),
                                 diagnostics=diagnostics,marginal_laws=laws)

    def __iter__(self):return self

    def __next__(self):
        while self.t<self.T:
            selected=self.rounds is None or self.t in self.rounds
            snapshot=self._snapshot() if selected else None
            start=time.perf_counter()
            self.posterior.update(self.data["q"][self.t],int(self.data["y"][self.t]))
            self.replay_seconds+=time.perf_counter()-start
            self.t+=1;self._check_moments(self.t)
            if self.t==self.T:self._verify_terminal_state()
            if selected:return snapshot
        raise StopIteration

    def next_batch(self,batch_size=8):
        if batch_size<1:raise ValueError("batch_size must be positive")
        snapshots=[]
        for _ in range(int(batch_size)):
            try:snapshots.append(next(self))
            except StopIteration:break
        return None if not snapshots else stack_snapshots(snapshots)

    def batches(self,batch_size=8):
        while True:
            batch=self.next_batch(batch_size)
            if batch is None:return
            yield batch

    def finish(self):
        """Replay any unconsumed observations without constructing extra banks."""
        self.rounds=frozenset()
        for _ in self:pass
        return self.summary()

    def summary(self):
        return {"replay_version":REPLAY_VERSION,"source":str(self.path),"run_key":self.run_key,
                "method":self.method,"answers_replayed":self.t,"answers_total":self.T,"complete":self.t==self.T,
                "moment_checks":len(self.moment_checks),"moment_checks_passed":all(c["passed"] for c in self.moment_checks),
                "max_mean_error":self.max_mean_error,"max_cov_error":self.max_cov_error,
                "moment_rtol":self.moment_rtol,"moment_atol":self.moment_atol,
                "original_smc_seed":self.metadata.get("seeds",{}).get("smc"),
                "original_device":self.original_device,"replay_device":self.device,
                "integration_seed":self.integration_seed,"marginal_samples":self.marginal_samples,
                "replay_seconds":self.replay_seconds,"integration_seconds":self.integration_seconds,
                "elapsed_seconds":time.perf_counter()-self.started,"engine_hashes":self.engine_hashes,
                "terminal_state_check":self.terminal_state_check,
                "semantics":"Saved questions and answers only; numerical integration/optimizer streams do not change inference RNG or history."}


def iter_posterior_snapshots(statistical_npz,**kwargs):
    return PosteriorReplay(statistical_npz,**kwargs)


def load_snapshot_batches(statistical_npz,batch_size=8,**kwargs):
    replay=PosteriorReplay(statistical_npz,**kwargs)
    yield from replay.batches(batch_size)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("path");parser.add_argument("--device");parser.add_argument("--samples",type=int,default=4096)
    args=parser.parse_args();replay=PosteriorReplay(args.path,marginal_samples=args.samples,device=args.device)
    checks=[]
    for snapshot in replay:
        if snapshot.t in (0,64,127):checks.append({"t":snapshot.t,**snapshot.eig_comparison(count=8192)})
    print(json.dumps({"summary":replay.summary(),"eig_checks":checks},indent=2))
