"""Family baselines from published rules; only B-EMCM receives observed labels.

Sources/equation mapping and adaptations are frozen in the campaign audit.
All pool arguments here mean the same capped, available candidates as the
existing benchmark. No reference/evaluation objects cross this interface.
"""
import copy
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import expit, xlogy
from .numerics import fit_glm
from .selectors import stable_priorities, _winner, curvature

FAMILY_METHODS = {'b_emcm':'gaussian', 'lcmd':'gaussian', 'regvol':'gaussian',
                  'entropy':'bernoulli', 'firal':'bernoulli'}
FAMILY_NAMES = {'b_emcm':'B-EMCM', 'lcmd':'LCMD', 'regvol':'RegVol',
                'entropy':'Entropy', 'firal':'FIRAL'}
VERSION = 'family_baselines_v1'


def inverse(A):
    result = cho_solve(cho_factor((A+A.T)*.5, lower=True, check_finite=False),
                       np.eye(len(A)), check_finite=False)
    return (result+result.T)*.5


def bemcm_select(C, theta, bootstrap_theta, m, alpha, priority):
    """Eqs19-25: bootstrap scenarios carry their own virtual SGD history."""
    predictions = C @ bootstrap_theta.T
    virtual = np.tile(theta, (len(bootstrap_theta), 1))
    norms = np.linalg.norm(C, axis=1)
    valid = np.ones(len(C), bool); selected = []; trace = []
    for slot in range(m):
        residual = C @ virtual.T - predictions
        scores = np.mean(np.abs(residual), axis=1) * norms
        j, ties = _winner(scores, valid, priority)
        gradients = residual[j, :, None] * C[j]
        trace.append(dict(slot=slot, candidate_position=j, score=float(scores[j]),
                          scenario_gradients=gradients.tolist(), numerical_ties=ties))
        virtual -= alpha * gradients
        valid[j] = False; selected.append(j)
    return np.asarray(selected), dict(slot_trace=trace, virtual_theta_final=virtual.tolist())


def lcmd_select(C, centers, m, priority):
    # A constant intercept cancels exactly; use existing coordinates unchanged.
    distance = cdist(C, centers, metric='sqeuclidean')
    assignments = distance.argmin(axis=1)
    nearest = distance[np.arange(len(C)), assignments].copy()
    valid = np.ones(len(C), bool); selected = []; trace = []
    center_count = len(centers)
    for slot in range(m):
        masses = np.bincount(assignments[valid], weights=nearest[valid], minlength=center_count)
        maximum = float(masses.max())
        eligible_centers = np.abs(masses-maximum) <= 1e-11*max(1.,abs(maximum))
        eligible = valid & eligible_centers[assignments]
        j, ties = _winner(nearest, eligible, priority)
        trace.append(dict(slot=slot,candidate_position=j,center_index=int(assignments[j]),
            cluster_squared_distance_mass=float(masses[assignments[j]]),
            squared_distance=float(nearest[j]),numerical_ties=ties))
        valid[j] = False; selected.append(j)
        new_distance = np.sum((C-C[j])**2, axis=1)
        changed = new_distance < nearest  # fixed earlier-center tie convention
        nearest[changed] = new_distance[changed]; assignments[changed] = center_count
        center_count += 1
    return np.asarray(selected), dict(slot_trace=trace,distance='squared_euclidean_original_query',
        cluster_mass='sum_squared_distances',initial_centers=len(centers),feature_map='linear_query')


def regvol_select(C, m, rho, rng):
    """FastRegVol Algorithm2: uniform proposal, h_i acceptance, inverse downdate."""
    active = np.arange(len(C)); count = len(C); Z = inverse(C.T@C+rho*np.eye(C.shape[1]))
    attempts = removed = clips = rebuilds = 0; min_h = 1.; max_h = 0.
    removal_ids = []; acceptance = []
    while count > m:
        position = int(rng.integers(count)); j = int(active[position])
        v = Z@C[j]; h = float(1.-C[j]@v); attempts += 1
        if not np.isfinite(h) or h < -1e-10 or h > 1.+1e-10:
            raise FloatingPointError(f'RegVol invalid h={h}; no substantial clipping allowed')
        if h < 0. or h > 1.:clips += 1; h = float(np.clip(h,0.,1.))
        min_h = min(min_h,h);max_h=max(max_h,h)
        if rng.random() >= h:continue
        if h <= 0.:raise FloatingPointError('RegVol accepted zero probability')
        Z += np.outer(v,v)/h; Z = (Z+Z.T)*.5
        active[position] = active[count-1]; count -= 1; removed += 1
        removal_ids.append(j);acceptance.append(h)
        if removed % 32 == 0:
            Z = inverse(C[active[:count]].T@C[active[:count]]+rho*np.eye(C.shape[1])); rebuilds += 1
    return np.sort(active[:count]), dict(method_name='RegVol_roundwise',lambda_regvol=rho,
        proposal_count=attempts,reverse_removal_iterations=removed,roundoff_clip_count=clips,
        inverse_rebuilds=rebuilds,min_h=min_h,max_h=max_h,
        removed_candidate_positions=removal_ids,accepted_h=acceptance,slot_trace=[])


def fir_relax(C0, U, F, m, relative_gap=1e-7, max_iter=2000):
    """Convex capped-simplex FIR, pairwise exchange with exact 1D line search.

    A valid convex lower bound f-gap certifies the solve. This is an optimizer
    for Eq14, not the discrete selection rule (which is FTRL below).
    """
    k = len(U); z = np.full(k,m/k); history=[]
    A=inverse(C0+U.T@(z[:,None]*U)); AU=U@A
    grad=-np.einsum('ij,ij->i',AU@F,AU)
    vertex=np.zeros(k);vertex[np.argsort(grad,kind='stable')[:m]]=1.
    if np.trace(inverse(C0+U.T@(vertex[:,None]*U))@F) < np.trace(A@F):z=vertex
    for iteration in range(max_iter):
        S = C0+U.T@(z[:,None]*U); A=inverse(S); AU=U@A
        value=float(np.trace(A@F)); gradient=-np.einsum('ij,ij->i',AU@F,AU)
        vertex=np.zeros(k);vertex[np.argsort(gradient,kind='stable')[:m]]=1.
        gap=max(0.,float(gradient@(z-vertex)))
        if iteration % 25 == 0:history.append([iteration,value,gap])
        if gap <= relative_gap*max(abs(value),1e-15):break
        add=np.flatnonzero(z < 1.-1e-13);remove=np.flatnonzero(z > 1e-13)
        i=int(add[np.argmin(gradient[add])]);j=int(remove[np.argmax(gradient[remove])])
        if i==j:raise FloatingPointError('FIR relaxation stalled above certificate tolerance')
        bound=min(1.-z[i],z[j])
        delta=np.outer(U[i],U[i])-np.outer(U[j],U[j])
        # Matrix pencil diagonalization makes line-search derivatives O(d^2).
        lam,Q=np.linalg.eigh(S);S_inv_half=(Q*lam**-.5)@Q.T
        eig,V=np.linalg.eigh(S_inv_half@delta@S_inv_half)
        target=V.T@S_inv_half@F@S_inv_half@V;weight=np.diag(target)
        def derivative(gamma):return float(-np.sum(weight*eig/(1.+gamma*eig)**2))
        step=bound if derivative(bound) <= 0 else brentq(derivative,0.,bound,xtol=1e-13)
        z[i] += step;z[j] -= step
    else:
        raise RuntimeError(f'FIRAL convex relaxation not certified: relative gap {gap/value:g}')
    if abs(z.sum()-m)>1e-8 or z.min() < -1e-10 or z.max()>1.+1e-10:
        raise FloatingPointError('FIRAL capped simplex violated')
    return z,S,dict(relaxed_objective=value,relaxed_duality_gap=gap,
        relaxed_relative_gap=gap/max(abs(value),1e-15),relax_iterations=iteration+1,
        relax_optimizer='pairwise_exchange_exact_line_search_FW_certificate',relax_history=history,
        relaxed_sum=float(z.sum()),relaxed_min=float(z.min()),relaxed_max=float(z.max()))


def firal_ftrl(U, C0, S, m, eta, priority):
    """Binary rank-one specialization of Eq17,21,23 and Algorithm1."""
    eig,V=np.linalg.eigh(S)
    if eig[0] <= 0:raise FloatingPointError('FIRAL normalization not SPD')
    inverse_half=(V*eig**-.5)@V.T
    P=U@inverse_half;D=inverse_half@(C0/m)@inverse_half
    d=len(S);cumulative=np.zeros_like(S);valid=np.ones(len(U),bool)
    selected=[];trace=[]
    for slot in range(m):
        if slot==0:root=np.sqrt(d)*np.eye(d);nu=np.sqrt(d)
        else:
            ev,Q=np.linalg.eigh(eta*cumulative);shift=ev-ev[0]
            if d==1:delta=1.
            else:delta=brentq(lambda x:float(np.sum((x+shift)**-2)-1.),1.,np.sqrt(d),xtol=1e-12)
            nu=delta-ev[0];root=(Q*(delta+shift))@Q.T
        Bhalf=inverse(root+eta*D);PB=P@Bhalf
        score=np.einsum('ij,ij->i',PB,PB)/(1.+eta*np.einsum('ij,ij->i',PB,P))
        j,ties=_winner(score,valid,priority);valid[j]=False;selected.append(j)
        trace.append(dict(slot=slot,candidate_position=j,score=float(score[j]),
                          ftrl_nu=float(nu),numerical_ties=ties))
        cumulative += D+np.outer(P[j],P[j]);cumulative=(cumulative+cumulative.T)*.5
    return np.asarray(selected),float(np.linalg.eigvalsh(cumulative)[0]),trace


def select_family_batch(method,X_pool,labeled_ids,candidates,theta,family,m,rho,rng,
                        config=None,batch_index=1,observed_y=None):
    config={} if config is None else config
    if FAMILY_METHODS.get(method)!=family:raise ValueError('Ineligible family/method')
    X=np.asarray(X_pool,dtype=np.float64);old=np.asarray(labeled_ids,dtype=np.int64)
    ids=np.asarray(candidates,dtype=np.int64);C=X[ids]
    if len(np.unique(ids))!=len(ids) or np.intersect1d(ids,old).size or not 0<m<=len(ids):
        raise ValueError('Invalid candidate availability or batch size')
    if method!='b_emcm' and observed_y is not None:raise ValueError('This method must not receive labels')
    priority=stable_priorities(ids,'tie',config.get('dataset',''),int(config.get('seed',0)),int(batch_index))
    started=time.perf_counter()
    audit=dict(method=method,method_version=VERSION,batch_index=batch_index,batch_size=m,
        candidate_ids=ids.tolist(),candidate_count=len(ids),rng_state_before=copy.deepcopy(rng.bit_generator.state),
        uses_theta_ref=False,uses_future_labels=False,feature_normalization='none_added',
        fallback_used=False,fallback_reason=None,jitter_events=[])
    if method=='b_emcm':
        y=np.asarray(observed_y,dtype=np.float64)
        if y.shape!=(len(old),):raise ValueError('B-EMCM receives only an observed-label vector')
        Z=int(config.get('bemcm_Z',4));alpha=float(config.get('bemcm_alpha',.05))
        seeds=rng.integers(0,2**63-1,size=Z);bootstrap=[];fit_audits=[]
        for seed in seeds:
            positions=np.random.default_rng(int(seed)).integers(len(old),size=len(old))
            th,fit=fit_glm(X[old[positions]],y[positions],'gaussian',rho=rho)
            if not fit['fit_converged']:raise RuntimeError('Bootstrap fit failed')
            bootstrap.append(th);fit_audits.append(fit)
        bootstrap=np.asarray(bootstrap)
        chosen,details=bemcm_select(C,np.asarray(theta),bootstrap,m,alpha,priority)
        details.update(Z=Z,alpha=alpha,bootstrap_seeds=seeds.tolist(),bootstrap_theta=bootstrap.tolist(),
            bootstrap_fit_audits=fit_audits,observed_count=len(old),
            virtual_response_convention='fixed_bootstrap_member_per_scenario')
    elif method=='lcmd':chosen,details=lcmd_select(C,X[old],m,priority)
    elif method=='regvol':chosen,details=regvol_select(C,m,rho,rng)
    elif method=='entropy':
        p=expit(C@theta);score=-xlogy(p,p)-xlogy(1.-p,1.-p)
        chosen=np.lexsort((priority,-score))[:m]
        details=dict(slot_trace=[dict(slot=s,candidate_position=int(j),score=float(score[j])) for s,j in enumerate(chosen)],
                     probability=p.tolist(),entropy=score.tolist())
    elif method=='firal':
        weights=curvature(C@theta,family);old_weights=curvature(X[old]@theta,family)
        U=C*np.sqrt(weights)[:,None]
        C0=rho*np.eye(X.shape[1])+X[old].T@(old_weights[:,None]*X[old])
        F=U.T@U/len(U)
        z,S,details=fir_relax(C0,U,F,m,relative_gap=float(config.get('firal_relative_gap',1e-7)),
                            max_iter=int(config.get('firal_relax_max_iter',2000)))
        etas=config.get('firal_eta_candidates',[2.,20.,200.,2000.,20000.])
        alternatives=[];batches=[];traces=[]
        for eta in etas:
            selected,min_eigen,trace=firal_ftrl(U,C0,S,m,float(eta),priority)
            objective=float(np.trace(inverse(C0+U[selected].T@U[selected])@F))
            alternatives.append(dict(eta=float(eta),normalized_min_eigenvalue=min_eigen,internal_FIR_objective=objective,
                                     selected_candidate_positions=selected.tolist()))
            batches.append(selected);traces.append(trace)
        winner=int(np.argmax([a['normalized_min_eigenvalue'] for a in alternatives]));chosen=batches[winner]
        details.update(relaxed_z=z.tolist(),Sigma_relaxed=S.tolist(),labeled_information_with_ridge=C0.tolist(),
            pool_Fisher_target=F.tolist(),eta_candidates=list(etas),eta_diagnostics=alternatives,
            selected_eta=float(etas[winner]),eta_selection='published_G2_max_normalized_min_eigenvalue',
            eta_grid_source='predeclared_log_grid_around_author_code_default_200',
            slot_trace=traces[winner],ridge_prior_adaptation='rho_I_in_labeled_information',
            pool_target_scope='current_capped_unqueried_candidates',curvature_floor=0.)
    chosen=np.asarray(chosen,dtype=np.int64);result=ids[chosen]
    assert len(result)==m and len(np.unique(result))==m
    audit.update(details);audit.update(selected_ids=result.tolist(),selected_candidate_positions=chosen.tolist(),
        rng_state_after=copy.deepcopy(rng.bit_generator.state),acquisition_seconds=time.perf_counter()-started)
    return result,audit
