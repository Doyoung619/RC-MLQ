"""Full-dimensional, finite-budget ellipsoid EIG search.

This module computes an achieved numerical reference, never a certified global
maximum. All seven saved grid questions are retained as witnesses. Search and
integration do not generate observations or change a learner's posterior.
"""
from __future__ import annotations

import time
import numpy as np

DEFAULT_ALPHA = .5
DEFAULT_RADIUS = 16.
DEFAULT_STARTS = 8
DEFAULT_STEPS = 160
DEFAULT_LEARNING_RATE = .002


def ellipsoid_geometry(mu, cov, alpha=DEFAULT_ALPHA, radius=DEFAULT_RADIUS):
    """Return symmetric root/inverse root and normalized shape (NumPy)."""
    mu, cov = np.asarray(mu, dtype=np.float64), np.asarray(cov, dtype=np.float64)
    if not 0 < alpha <= 1 or radius <= 0:
        raise ValueError("Require alpha in (0,1] and a positive physical radius")
    values, vectors = np.linalg.eigh((cov + np.swapaxes(cov, -1, -2)) / 2)
    if np.any(values[..., -1] <= 0) or np.any(values[..., 0] < -1e-12):
        raise ValueError("Invalid covariance; no spread floor or ridge is added")
    normalized = alpha + (1-alpha)*values/values[..., -1, None]
    root = (vectors*np.sqrt(normalized)[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    inverse = (vectors/np.sqrt(normalized)[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    shape = (vectors*normalized[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    return root, inverse, shape


def normalized_candidate_norms(q, mu, inverse_root, radius=DEFAULT_RADIUS):
    q, mu = np.asarray(q, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    if mu.ndim == 1:
        normalized = (q-mu) @ inverse_root / radius
    else:
        # q [B,S,M,D] or [B,M,D].
        shift = mu.reshape((len(mu),)+(1,)*(q.ndim-2)+(mu.shape[-1],))
        normalized = np.einsum("b...d,bdk->b...k",q-shift,inverse_root)/radius
    return np.linalg.norm(normalized,axis=-1)


def _torch_eig_gradient(points, weights, q, beta, gradient=True):
    """Points [B,N,D], q [B,S,M,D], beta [B]; mean-centered coordinates.

    dI/dq_m = 2 beta E[p_m (log(p_m/pbar_m)-KL(p||pbar))
                          (theta-q_m)].
    The formula includes both candidate-position and candidate-norm terms.
    """
    import torch
    logits = 2*beta[:,None,None,None]*torch.einsum("bnd,bsmd->bsnm",points,q)
    logits -= beta[:,None,None,None]*torch.sum(q*q,dim=-1)[:,:,None,:]
    logp = torch.log_softmax(logits,dim=-1)
    logw = torch.log(weights)
    logbar = torch.logsumexp(logp+logw[:,None,:,None],dim=2)
    a = logp-logbar[:,:,None,:]
    p,pbar = torch.exp(logp),torch.exp(logbar)[:,:,None,:]
    direct = p*a-(p-pbar)
    small = torch.abs(a)<1e-3
    # Only bounded arguments enter the polynomial; large ratios can otherwise
    # overflow even in an unselected torch.where branch.
    x = torch.where(small,a,torch.zeros_like(a))
    series=x*x*(.5+x*(1/3+x*(1/8+x*(1/30+x/144))))
    contribution=torch.where(small,pbar*series,direct)
    values=torch.sum(weights[:,None,:]*torch.sum(contribution,dim=-1),dim=-1)
    if not gradient:
        return values,None
    conditional_kl=torch.sum(p*a,dim=-1,keepdim=True)
    sensitivity=weights[:,None,:,None]*p*(a-conditional_kl)
    derivative=2*beta[:,None,None,None]*(
        torch.einsum("bsnm,bnd->bsmd",sensitivity,points)
        -q*torch.sum(sensitivity,dim=2)[...,None])
    return values,derivative


def eig_value_gradient(points,weights,q,beta,*,device="cpu",gradient=True):
    """Public numerical objective/gradient for a single or batched posterior.

    A single posterior takes points[N,D], weights[N], q[S,M,D] (or q[M,D]).
    Batched input takes points[B,N,D], weights[B,N], q[B,S,M,D].
    """
    import torch
    points,weights,q=np.asarray(points,dtype=np.float64),np.asarray(weights,dtype=np.float64),np.asarray(q,dtype=np.float64)
    single=points.ndim==2
    single_q=single and q.ndim==2
    if single:
        points,weights=points[None],weights[None]
        q=q[None,None] if single_q else q[None]
    weights=weights/weights.sum(axis=1,keepdims=True)
    centers=np.einsum("bn,bnd->bd",weights,points)
    b=np.broadcast_to(np.asarray(beta,dtype=np.float64),(len(points),)).copy()
    with torch.inference_mode():
        v,g=_torch_eig_gradient(torch.as_tensor(points-centers[:,None,:],device=device),torch.as_tensor(weights,device=device),
             torch.as_tensor(q-centers[:,None,None,:],device=device),torch.as_tensor(b,device=device),gradient)
        values=v.cpu().numpy()
        deriv=None if g is None else g.cpu().numpy()
    if single:
        values=values[0]
        if deriv is not None:deriv=deriv[0]
    if single_q:
        values=values[0]
        if deriv is not None:deriv=deriv[0]
    return values,deriv


def _balanced_directions(rng,M,d):
    """Rotated simplex when possible, otherwise balanced spherical directions.

    A four-candidate start spans a three-dimensional affine simplex rather than
    a two-dimensional antipodal square. No direction constraint is maintained
    during optimization.
    """
    if M<=d+1:
        centered=np.eye(M)-np.ones((M,M))/M
        simplex=np.linalg.qr(centered[:,:M-1],mode="reduced")[0]
        embedding=np.linalg.qr(rng.standard_normal((d,M-1)),mode="reduced")[0]
        return simplex@embedding.T/np.sqrt((M-1)/M)
    half=(M+1)//2
    directions=rng.standard_normal((half,d))
    if half<=d:
        directions=np.linalg.qr(directions.T,mode="reduced")[0].T
    else:
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    return np.concatenate([directions,-directions],axis=0)[:M]


def optimize_ellipsoid(points,weights,mu,cov,beta,actual_q,grid_q,*,device="cuda:3",
                       alpha=DEFAULT_ALPHA,radius=DEFAULT_RADIUS,starts=DEFAULT_STARTS,
                       steps=DEFAULT_STEPS,seed=20261005,warm_q=None,
                       learning_rate=DEFAULT_LEARNING_RATE):
    """Batch independent histories and starts; return achieved EIG and metadata.

    All inputs optionally have leading batch B. Single-history return arrays
    omit that leading dimension. ``seed`` can be a B-vector for scheduling-
    independent initialization. A feasible warm question is both retained as
    a witness and used as a search start. Global optimality is not certified.
    Original feasibility is the radius-32 enclosing-ball domain: radius<=32
    makes every searched ellipsoid tuple feasible. No robot rollout is needed.
    """
    import torch
    started=time.perf_counter()
    points,weights,mu,cov,actual_q,grid_q=[np.asarray(x,dtype=np.float64) for x in (points,weights,mu,cov,actual_q,grid_q)]
    single=points.ndim==2
    if single:
        points,weights,mu,cov,actual_q,grid_q=[x[None] for x in (points,weights,mu,cov,actual_q,grid_q)]
        if warm_q is not None:warm_q=np.asarray(warm_q,dtype=np.float64)[None]
    B,N,d=points.shape
    M=actual_q.shape[-2]
    if actual_q.shape!=(B,M,d) or grid_q.shape!=(B,7,M,d) or weights.shape!=(B,N):
        raise ValueError("Incorrect joint quadrature, actual, or seven-grid shapes")
    if starts<2 or steps<0 or learning_rate<=0:
        raise ValueError("At least two starts, nonnegative steps and positive learning rate required")
    required_radius=np.sqrt(d)+2*(1+np.sqrt(2*M-1))
    if radius < required_radius-1e-12 or radius>32:
        raise ValueError(f"Common reference radius must be >= {required_radius} and <= original domain radius32")
    if np.any(weights<0) or np.any(weights.sum(axis=1)<=0) or not all(np.isfinite(x).all() for x in (points,weights,mu,cov,actual_q,grid_q)):
        raise ValueError("Nonfinite inputs or invalid integration weights")
    weights=weights/weights.sum(axis=1,keepdims=True)
    beta=np.broadcast_to(np.asarray(beta,dtype=np.float64),(B,)).copy()
    if np.any(beta<=0):raise ValueError("beta must be positive")
    # Never infer an array dtype from mixed small/large Python integers:
    # NumPy can choose float64 and silently round stable 64-bit seeds.
    seed_values=[seed] if np.isscalar(seed) else list(seed)
    if any(not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_))
           or int(value)<0 or int(value)>=2**64 for value in seed_values):
        raise ValueError("Seeds must be exact nonnegative 64-bit integers")
    seeds=np.asarray(seed,dtype=np.uint64)
    if seeds.ndim==0:
        seeds=np.array([(int(seed)+i) % 2**64 for i in range(B)],dtype=np.uint64)
    if seeds.shape!=(B,):raise ValueError("seed must be scalar or one integer per history")
    root,inverse,shape=ellipsoid_geometry(mu,cov,alpha,radius)
    grid_norm=normalized_candidate_norms(grid_q,mu,inverse,radius)
    if np.any(grid_norm>1+1e-10):raise ValueError("Saved grid is outside reference ellipsoid")
    actual_norm=normalized_candidate_norms(actual_q,mu,inverse,radius)
    actual_inside=np.max(actual_norm,axis=1)<=1+1e-12
    witnesses=[grid_q,actual_q[:,None,:,:]]
    eligible=[np.ones((B,7),dtype=bool),actual_inside[:,None]]
    warm_inside=np.zeros(B,dtype=bool)
    if warm_q is not None:
        warm_q=np.asarray(warm_q,dtype=np.float64)
        if warm_q.shape!=(B,M,d) or not np.isfinite(warm_q).all():raise ValueError("Invalid warm question")
        warm_inside=np.max(normalized_candidate_norms(warm_q,mu,inverse,radius),axis=1)<=1+1e-12
        witnesses.append(warm_q[:,None,:,:]);eligible.append(warm_inside[:,None])
    witnesses=np.concatenate(witnesses,axis=1)
    eligible=np.concatenate(eligible,axis=1)
    qstart=np.empty((B,starts,M,d),dtype=np.float64)
    # Equal *physical* radius within each start avoids norm-offset saturation.
    radius_factors=np.geomspace(.06,1.,max(2,starts))
    radius_factors[-2:]=1.  # Independent broad starts close to the boundary.
    for b in range(B):
        rng=np.random.default_rng(int(seeds[b]))
        for s in range(starts):
            direction=_balanced_directions(rng,M,d)
            directional_norm=np.linalg.norm(direction@inverse[b],axis=1)
            common_radius=radius/np.max(directional_norm)*radius_factors[s]
            qstart[b,s]=mu[b]+common_radius*direction
        if warm_q is not None and warm_inside[b]:qstart[b,0]=warm_q[b]
    with torch.inference_mode():
        tx=torch.as_tensor(points-mu[:,None,:],dtype=torch.float64,device=device)
        tw=torch.as_tensor(weights,dtype=torch.float64,device=device)
        tb=torch.as_tensor(beta,dtype=torch.float64,device=device)
        tr=torch.as_tensor(root,dtype=torch.float64,device=device)
        ti=torch.as_tensor(inverse,dtype=torch.float64,device=device)
        tmu=torch.as_tensor(mu,dtype=torch.float64,device=device)
        witness_q=torch.as_tensor(witnesses-mu[:,None,None,:],dtype=torch.float64,device=device)
        witness_eig,_=_torch_eig_gradient(tx,tw,witness_q,tb,False)
        mask=torch.as_tensor(eligible,device=device)
        masked=torch.where(mask,witness_eig,torch.full_like(witness_eig,-torch.inf))
        best,indices=torch.max(masked,dim=1)
        rows=torch.arange(B,device=device)
        best_q=witness_q[rows,indices].clone()
        q=torch.as_tensor(qstart-mu[:,None,None,:],dtype=torch.float64,device=device)
        z=torch.einsum("bsmd,bdk->bsmk",q,ti)/radius
        first_moment=torch.zeros_like(z);second_moment=torch.zeros_like(z)
        radial_first=torch.zeros_like(z[...,:1]);radial_second=torch.zeros_like(z[...,:1])
        sigma=torch.as_tensor(np.sqrt(np.linalg.eigvalsh(cov)[:,-1]),dtype=torch.float64,device=device)[:,None,None,None]
        best_step=torch.full((B,),-1,dtype=torch.int64,device=device)
        trace=[]
        initial_best=None
        for step in range(steps+1):
            q=radius*torch.einsum("bsmd,bdk->bsmk",z,tr)
            values,gradient=_torch_eig_gradient(tx,tw,q,tb,gradient=step<steps)
            start_best,start_index=torch.max(values,dim=1)
            improve=start_best>best
            best_q=torch.where(improve[:,None,None],q[rows,start_index],best_q)
            best=torch.maximum(best,start_best)
            best_step=torch.where(improve,torch.full_like(best_step,step),best_step)
            if step==0:initial_best=best.clone()
            if step==0 or step==steps or step%10==0:trace.append(best.clone())
            if step==steps:break
            # Candidate-norm offsets make radial directions much stiffer than
            # angular directions when posterior width << physical query radius.
            # Separate Adam moments and retract angular moves at fixed *current*
            # radius, then apply a nonzero radial move. All Md variables remain
            # free; radii are not fixed or tied between candidate slots.
            qradius=torch.linalg.vector_norm(q,dim=-1,keepdim=True)
            unit=q/torch.clamp(qradius,min=torch.finfo(torch.float64).tiny)
            grad_radial=torch.sum(gradient*unit,dim=-1,keepdim=True)
            grad_tangent=gradient-grad_radial*unit
            first_moment=.9*first_moment+.1*grad_tangent
            second_moment=.999*second_moment+.001*grad_tangent*grad_tangent
            radial_first=.9*radial_first+.1*grad_radial
            radial_second=.999*radial_second+.001*grad_radial*grad_radial
            corrected_m=first_moment/(1-.9**(step+1))
            corrected_v=second_moment/(1-.999**(step+1))
            rate=learning_rate*(.1+.9*.5*(1+np.cos(np.pi*step/max(1,steps-1))))
            angular_delta=rate*radius*corrected_m/(torch.sqrt(corrected_v)+1e-12)
            angular_delta-=torch.sum(angular_delta*unit,dim=-1,keepdim=True)*unit
            radial_delta=rate*radius*(radial_first/(1-.9**(step+1)))/(torch.sqrt(radial_second/(1-.999**(step+1)))+1e-12)
            radial_delta*=sigma/torch.maximum(qradius,sigma)
            rotated=q+angular_delta
            rotated_norm=torch.linalg.vector_norm(rotated,dim=-1,keepdim=True)
            new_radius=torch.clamp(qradius+radial_delta,min=0.)
            qnext=rotated/torch.clamp(rotated_norm,min=torch.finfo(torch.float64).tiny)*new_radius
            qnext=torch.where(qradius>0,qnext,angular_delta)
            z=torch.einsum("bsmd,bdk->bsmk",qnext,ti)/radius
            norm=torch.linalg.vector_norm(z,dim=-1,keepdim=True)
            z/=torch.clamp(norm,min=1.)
        absolute_q=best_q+tmu[:,None,:]
        achieved=best.cpu().numpy()
        bestq=absolute_q.cpu().numpy()
        all_witness=witness_eig.cpu().numpy()
        initial=initial_best.cpu().numpy()
        trajectory=torch.stack(trace,dim=1).cpu().numpy()
        beststep=best_step.cpu().numpy()
    if not np.isfinite(achieved).all() or np.any(achieved < -1e-10) or np.any(achieved>np.log(M)+1e-10):
        raise FloatingPointError("Achieved EIG failed finite/information bounds")
    final_norm=normalized_candidate_norms(bestq,mu,inverse,radius)
    if np.any(final_norm>1+1e-10):raise FloatingPointError("Optimizer returned an infeasible question")
    risk=np.trace(cov,axis1=1,axis2=2)
    upper=np.minimum(np.log(M),2*beta**2*radius**2*risk)
    numerator=all_witness[:,7]
    with np.errstate(divide="ignore",invalid="ignore"):
        ratio=numerator/achieved
        lower_ratio=numerator/upper
    result={
        "achieved_eig":achieved,"best_q":bestq,"initial_best_eig":initial,
        "improvement":achieved-initial,"grid_eigs":all_witness[:,:7],
        "actual_eig":numerator,"actual_inside":actual_inside,
        "actual_max_normalized_radius":np.max(actual_norm,axis=1),
        "best_max_normalized_radius":np.max(final_norm,axis=1),
        "upper_bound_estimate":upper,"global_gap_estimate":upper-achieved,
        "ratio_approx":ratio,"ratio_lower_estimate":lower_ratio,
        "best_step":beststep,"best_eig_trace":trajectory,
        "initialization_seed":seeds,"warm_inside":warm_inside,
        "posterior_sample_count":N,"alpha":float(alpha),"radius":float(radius),
        "starts":int(starts),"steps":int(steps),"learning_rate":float(learning_rate),
        "seconds":time.perf_counter()-started,"batch_size":B,"device":str(device),
        "global_certified":False,"integration_certified":False,
        "reference_kind":"achieved_full_dimensional_ellipsoid_eig",
        "optimizer":"projected_polar_adam_analytic_gradient_cosine_decay",
        "preconditioning":"separate radial/angular Adam moments; radial step damped by sigma/max(candidate_radius,sigma); angular retraction and ellipsoid projection; all Md variables free",
        "optimization_limitations":"finite randomized starts and steps; local stationary points, saturated responses and conservative analytic upper gaps can remain; doubled-budget diagnostics are not a global certificate",
        "convergence_claim":"fixed_budget_only; retained best iterate and global gap proxy reported",
        "ratio_interpretation":"approximate upper estimate of exact ellipsoid ratio; numerical integrals uncertified",
    }
    if single:
        for key,value in list(result.items()):
            if isinstance(value,np.ndarray) and value.shape and value.shape[0]==1:result[key]=value[0]
    return result
