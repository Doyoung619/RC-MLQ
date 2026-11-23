import itertools
from collections import Counter
import numpy as np
import pytest
from scipy.optimize import minimize, brentq
from pscore_harness.spec_suite.family_selectors import (FAMILY_METHODS, bemcm_select, lcmd_select,
    regvol_select, fir_relax, firal_ftrl, select_family_batch)
from pscore_harness.spec_suite.selectors import select_batch


def test_bemcm_sequential_correlation_and_zero_alpha_limit():
    X=np.array([[1.,0.],[.99,.01],[0.,.98]])
    theta=np.ones(2);bootstrap=np.zeros((4,2));priority=np.arange(3)
    selected,a=bemcm_select(X,theta,bootstrap,2,.05,priority)
    naive,_=bemcm_select(X,theta,bootstrap,2,0.,priority)
    np.testing.assert_array_equal(naive,[0,1]);np.testing.assert_array_equal(selected,[0,2])
    virtual=np.tile(theta,(4,1));available=list(range(3))
    for j,row in zip(selected,a['slot_trace']):
        G=np.array([[(X[i]@virtual[z]-X[i]@bootstrap[z])*X[i] for z in range(4)] for i in available])
        scores=np.linalg.norm(G,axis=2).mean(axis=1)
        assert j==available[int(np.argmax(scores))]
        assert row['score']==pytest.approx(scores.max())
        virtual-=.05*G[available.index(j)];available.remove(j)


def test_lcmd_cluster_mass_not_farthest_first():
    centers=np.array([[0.,0.],[10.,0.]])
    candidates=np.array([[2.,0.],[2.,.1],[2.,-.1],[13.,0.]])
    selected,a=lcmd_select(candidates,centers,2,np.arange(4))
    assert selected[0] in [1,2] and selected[1]==3
    assert a['slot_trace'][0]['cluster_squared_distance_mass']==pytest.approx(12.02)
    assert a['slot_trace'][1]['cluster_squared_distance_mass']==pytest.approx(9.)


def test_regvol_matches_exact_reverse_transition_distribution():
    X=np.array([[2.,0.],[0.,1.],[.2,.3],[1.,1.]])
    rho=1.;m=2;distribution={tuple(range(len(X))):1.}
    while len(next(iter(distribution)))>m:
        nxt=Counter()
        for ids,mass in distribution.items():
            A=np.linalg.inv(X[list(ids)].T@X[list(ids)]+rho*np.eye(2))
            h=np.array([1.-X[j]@A@X[j] for j in ids]);h/=h.sum()
            for j,prob in zip(ids,h):nxt[tuple(i for i in ids if i!=j)]+=mass*prob
        distribution=nxt
    rng=np.random.default_rng(400);observed=Counter()
    for _ in range(7000):
        selected,a=regvol_select(X,m,rho,rng);observed[tuple(selected)]+=1
        assert len(set(selected))==m and a['reverse_removal_iterations']==2
    for ids,p in distribution.items():assert abs(observed[ids]/7000.-p)<.025
    assert distribution[(0,1)]>distribution[(1,2)]
    np.testing.assert_array_equal(regvol_select(X,m,rho,np.random.default_rng(17))[0],
                                  regvol_select(X,m,rho,np.random.default_rng(17))[0])


def test_entropy_extreme_logits_and_uniform_center():
    X=np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,1.]])
    ids,a=select_family_batch('entropy',X,[0],[1,2,3],np.array([1000.,0.]),'bernoulli',1,1.,np.random.default_rng(1))
    assert ids.tolist()==[1]
    assert np.isfinite(a['entropy']).all() and a['entropy'][0]==pytest.approx(np.log(2.))


def test_firal_relaxation_vs_independent_constrained_optimizer():
    rng=np.random.default_rng(941);U=rng.normal(size=(14,3));C0=np.diag([.5,1.,1.5]);F=U.T@U/len(U);m=4
    z,S,a=fir_relax(C0,U,F,m,relative_gap=1e-7,max_iter=5000)
    def value(x):return np.trace(np.linalg.inv(C0+U.T@(x[:,None]*U))@F)
    def grad(x):
        A=np.linalg.inv(C0+U.T@(x[:,None]*U));return -np.einsum('ij,ij->i',(U@A)@F,U@A)
    r=minimize(value,np.full(len(U),m/len(U)),jac=grad,method='SLSQP',bounds=[(0,1)]*len(U),
        constraints={'type':'eq','fun':lambda x:x.sum()-m,'jac':lambda x:np.ones(len(x))},options={'ftol':1e-12,'maxiter':1000})
    assert r.success and abs(z.sum()-m)<1e-9 and z.min()>=-1e-12 and z.max()<=1.+1e-12
    assert value(z)==pytest.approx(r.fun,rel=2e-7)
    assert a['relaxed_objective']-a['relaxed_duality_gap']<=r.fun+1e-10


def test_firal_ftrl_rank_one_rule_matches_direct_matrix_objective():
    rng=np.random.default_rng(151);U=rng.normal(size=(11,3));C0=np.eye(3);m=4;eta=20.
    z,S,_=fir_relax(C0,U,U.T@U/len(U),m,max_iter=5000)
    selected,smallest,_=firal_ftrl(U,C0,S,m,eta,np.arange(len(U)))
    ev,Q=np.linalg.eigh(S);invhalf=(Q*ev**-.5)@Q.T
    H=np.array([invhalf@(C0/m+np.outer(u,u))@invhalf for u in U]);F=np.zeros_like(S);remaining=list(range(len(U)))
    for slot,j in enumerate(selected):
        if slot==0:root=np.sqrt(3)*np.eye(3)
        else:
            vals,V=np.linalg.eigh(eta*F);shift=vals-vals[0]
            delta=brentq(lambda x:sum((x+shift)**-2)-1.,1.,np.sqrt(3))
            root=(V*(delta+shift))@V.T
        objectives=[np.trace(np.linalg.inv(root+eta*H[i])) for i in remaining]
        assert j==remaining[int(np.argmin(objectives))]
        remaining.remove(j);F+=H[j]
    assert smallest==pytest.approx(np.linalg.eigvalsh(F)[0])


@pytest.mark.parametrize('method',FAMILY_METHODS)
def test_replay_only_observed_labels_and_eligible_family(method):
    rng=np.random.default_rng(391);X=rng.normal(size=(35,4))/3;old=np.arange(9);cand=np.arange(9,35);theta=rng.normal(size=4)
    kwargs=dict(method=method,X_pool=X,labeled_ids=old,candidates=cand,theta=theta,
        family=FAMILY_METHODS[method],m=4,rho=1.,config={'dataset':'toy','seed':1})
    if method=='b_emcm':kwargs['observed_y']=rng.normal(size=len(old))
    a,meta=select_family_batch(**kwargs,rng=np.random.default_rng(93))
    kwargs['config'].update(theta_ref=np.ones(4)*1e6,future_labels=np.zeros(35))
    b,_=select_family_batch(**kwargs,rng=np.random.default_rng(93));np.testing.assert_array_equal(a,b)
    assert len(set(a))==4 and set(a).issubset(cand) and not set(a)&set(old)
    if method=='b_emcm':assert len(meta['bootstrap_theta'])==meta['Z']==4 and meta['alpha']==.05
    if method in ('lcmd','regvol'):
        kwargs['theta']=theta*100
        c,_=select_family_batch(**kwargs,rng=np.random.default_rng(93));np.testing.assert_array_equal(a,c)
    if method=='firal':
        assert len(meta['eta_candidates'])==5 and meta['relaxed_relative_gap']<=1e-7
        winner=np.argmax([v['normalized_min_eigenvalue'] for v in meta['eta_diagnostics']])
        assert meta['selected_eta']==meta['eta_candidates'][winner]
    kwargs['family']='poisson'
    with pytest.raises(ValueError):select_family_batch(**kwargs,rng=np.random.default_rng(93))


def test_firal_is_not_bait_selection():
    differences=0
    for seed in [0,1,2]:
        rng=np.random.default_rng(seed);X=rng.normal(size=(100,5));X[60:,0]*=8
        old=np.arange(12);cand=np.arange(12,60);theta=rng.normal(size=5)
        a,_=select_family_batch('firal',X,old,cand,theta,'bernoulli',6,1.,np.random.default_rng(73))
        b,_=select_batch('bait_glm',X,old,cand,theta,'bernoulli',6,1.,np.random.default_rng(73))
        differences+=not np.array_equal(np.sort(a),np.sort(b))
    assert differences>0
