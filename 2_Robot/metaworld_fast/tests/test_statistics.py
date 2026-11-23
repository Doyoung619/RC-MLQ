"""Behavioral and numerical checks; reduced SMC N only for plumbing tests."""
import copy
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import logsumexp
from metaworld_fast.model import log_probabilities, probabilities, principal_direction, coordinate_of_query, stable_seed
from metaworld_fast.metrics import integrated_eig, information_ratio
from metaworld_fast.line_template import make_template, SCALE_GRID
from metaworld_fast.posterior_marginal import MarginalPosterior
from metaworld_fast.posterior_smc import SMCPosterior, effective_sample_size
from metaworld_fast.selectors import SelectorState, select_query, sg_stage, radial_cap
from metaworld_fast.statistical_runner import run_statistical, replay_history


def test_squared_distance_model_norm_offset_translation_and_permutation():
    rng=np.random.default_rng(4)
    x,q=rng.normal(size=(40,4)),rng.normal(size=(8,4))
    direct=-4*np.sum((x[:,None,:]-q[None,:,:])**2,axis=2)
    direct-=logsumexp(direct,axis=1,keepdims=True)
    actual=log_probabilities(x,q,4)
    np.testing.assert_allclose(actual,direct,atol=1e-12)
    np.testing.assert_allclose(np.exp(actual).sum(axis=1),1,atol=2e-14)
    shift=np.array([100,-20,40,.5])
    np.testing.assert_allclose(log_probabilities(x+shift,q+shift,4),actual,atol=2e-12)
    perm=rng.permutation(len(q))
    np.testing.assert_allclose(log_probabilities(x,q[perm],4),actual[:,perm],atol=1e-12)
    w=np.full(len(x),1/len(x))
    assert abs(integrated_eig(x,w,q,4)-integrated_eig(x,w,q[perm],4))<1e-12
    incorrect=4*x@q.T
    incorrect-=logsumexp(incorrect,axis=1,keepdims=True)
    assert np.max(np.abs(incorrect-actual))>1


def test_uniform_binary_template_full_risk_and_identical_candidates():
    p=MarginalPosterior(d=1,beta=4)
    t=make_template(p,2)
    np.testing.assert_allclose(t.mean,0,atol=1e-15)
    np.testing.assert_allclose(t.sigma,1/np.sqrt(3),atol=1e-14)
    np.testing.assert_allclose(t.quantiles,[-.5,.5],atol=1e-13)
    np.testing.assert_allclose(t.b,[-np.sqrt(3)/2,np.sqrt(3)/2],atol=1e-13)
    np.testing.assert_allclose(t.fisher_scores,3*16*SCALE_GRID**2,rtol=1e-13)
    assert t.rc_index==6
    _,q=select_query("PQ",p,2,0,np.random.default_rng(0),SelectorState(),t)
    np.testing.assert_allclose(q[:,0],[-.5,.5],atol=1e-13)
    full=MarginalPosterior()
    assert abs(np.trace(full.moments()[1])-4/3)<1e-14
    same=np.ones((4,4))
    np.testing.assert_allclose(probabilities(np.zeros(4),same,16),.25,atol=1e-15)
    assert abs(full.eig(same))<1e-25


def test_small_eig_stable_and_no_denominator_floor():
    p=MarginalPosterior(d=1,beta=1)
    for amplitude in (1e-4,1e-7,1e-9):
        q=np.array([[-amplitude],[amplitude]])
        eig=p.eig(q)
        expected=2/3*amplitude**2
        assert eig>0
        assert abs(eig/expected-1)<2e-6
    assert np.isnan(information_ratio(0.,0.)[0])
    assert information_ratio(.4,.2)[0]==2.


def test_deterministic_full_covariance_direction_and_factorization_guard():
    v,s=principal_direction(np.eye(4)/3)
    np.testing.assert_allclose(v,[1,0,0,0],atol=1e-14)
    # Repeated top eigenspace, whose canonical projection is not an eigenvector column.
    normal=np.array([1.,1.,1.])/np.sqrt(3)
    covariance=2*np.eye(3)-np.outer(normal,normal)
    v,_=principal_direction(covariance)
    projector=np.eye(3)-np.outer(normal,normal)
    expected=projector[:,0]/np.linalg.norm(projector[:,0])
    np.testing.assert_allclose(v,expected,atol=1e-14)
    with pytest.raises(ValueError,match="factorized"):
        coordinate_of_query([[0,0],[1,1]])
    with pytest.raises(ValueError,match="factorized"):
        MarginalPosterior(d=2).update([[0,0],[1,1]],0)


def test_qi_one_frozen_transformation_and_own_reference():
    p=MarginalPosterior()
    p.update(np.array([[-.5,0,0,0],[.5,0,0,0]]),0)
    before=pickle.dumps(p)
    t=make_template(p,4)
    expected=p.hypothetical_means(t.queries[t.rc_index])
    _,q=select_query("QI",p,4,1,np.random.default_rng(0),SelectorState(),t)
    np.testing.assert_array_equal(q,expected)
    assert pickle.dumps(p)==before
    assert len(p.history)==1
    prior_template=make_template(MarginalPosterior(),4)
    assert not np.array_equal(t.queries,prior_template.queries)
    for index in range(7):
        eig=p.eig(t.queries[index])
        assert 0<=eig<=np.log(4)
    assert p.eig(t.queries[t.rc_index])/max(p.eig(q) for q in t.queries)<=1+1e-12


def test_ps_joint_weighted_with_replacement():
    p=SMCPosterior(d=2,particles=4,rng=np.random.default_rng(0))
    p.particles=np.array([[-1,-1],[-.5,-.5],[.5,.5],[1,1]],float)
    p.logweights=np.log([.01,.02,.07,.9])
    samples=p.sample(10000,np.random.default_rng(1))
    np.testing.assert_array_equal(samples[:,0],samples[:,1])
    assert np.mean(samples[:,0]==1)>.88
    assert len(np.unique(samples,axis=0))<len(samples)


def test_weighted_quantile_first_strict_cdf_exceedance():
    p=SMCPosterior(d=1,particles=4,rng=np.random.default_rng(0))
    p.particles=np.array([[-.75],[-.25],[.25],[.75]])
    p.logweights=np.full(4,np.log(.25))
    np.testing.assert_array_equal(p.projected_quantiles(np.array([1.]),[.0,.25,.5,.75,1.]),[-.75,-.25,.25,.75,.75])


def test_gaussian_cloud_uses_full_covariance():
    cov=np.array([[.1,.08],[.08,.2]])
    template=SimpleNamespace(mean=np.zeros(2),covariance=cov)
    _,q=select_query("GC",None,100000,0,np.random.default_rng(14),SelectorState(),template)
    np.testing.assert_allclose(np.cov(q.T),cov,atol=.002)


def test_sg_boundaries_saved_center_clone_and_current_mean_cap():
    assert [sg_stage(t) for t in (0,127,128,383,384)]==[0,0,1,1,2]
    state=SelectorState()
    rng=np.random.default_rng(2)
    first=SimpleNamespace(mean=np.array([.1,.2]),covariance=np.eye(2))
    select_query("SG",None,4,0,rng,state,first)
    changed=SimpleNamespace(mean=np.array([.8,-.2]),covariance=np.eye(2))
    select_query("SG",None,4,127,rng,state,changed)
    np.testing.assert_array_equal(state.center,first.mean)
    clone=copy.deepcopy(state)
    select_query("SG",None,4,128,rng,clone,changed)
    np.testing.assert_array_equal(clone.center,changed.mean)
    np.testing.assert_array_equal(state.center,first.mean)
    capped=radial_cap(np.array([[100.,100.],[.8,-.2]]),changed.mean)
    np.testing.assert_allclose(np.linalg.norm(capped[0]-changed.mean),32,atol=1e-13)
    np.testing.assert_array_equal(capped[1],changed.mean)


def test_tempering_exact_target_and_ten_sweeps_per_resampling():
    rng=np.random.default_rng(12)
    p=SMCPosterior(d=2,beta=16,particles=256,rng=rng)
    for t in range(8):
        q=rng.uniform(-2,2,(4,2))
        p.update(q,t%4)
        diag=p.last_diagnostics
        assert diag["final_exponent"]==1.
        assert len(diag["sweeps"])==10*diag["resamplings"]
        assert np.all(np.abs(p.particles)<=1)
        assert effective_sample_size(p.logweights)>=.5*p.n-1e-7
        exact=sum(log_probabilities(p.particles,hq,p.beta)[:,hy] for hq,hy in p.history)
        np.testing.assert_allclose(p.log_target,exact,atol=2e-11)
        np.testing.assert_allclose(p.target_logdensity(p.particles),exact,atol=2e-11)
    half=sum(log_probabilities(p.particles,hq,p.beta)[:,hy] for hq,hy in p.history[:-1])+.5*log_probabilities(p.particles,*[p.history[-1][0],p.beta])[:,p.history[-1][1]]
    np.testing.assert_allclose(p.target_logdensity(p.particles,.5),half,atol=2e-11)


@pytest.mark.parametrize("method",["RC","PQ","QI","PS","GC","SG"])
def test_resume_exact_and_theta_shared(tmp_path,method):
    job=dict(setting_id="resume",M=4,beta=8,method=method,interaction_id=2,answers=18,particles=128)
    full=run_statistical(job,tmp_path/"full")
    run_statistical(job,tmp_path/"resume",max_answers=7)
    resumed=run_statistical(job,tmp_path/"resume")
    a,b=np.load(full["artifact"]),np.load(resumed["artifact"])
    for key in ("q","q_raw","y","theta","mu","cov","risk","error","eta","eig","reference_eig","diag_eig","reference_queries"):
        np.testing.assert_array_equal(a[key],b[key])
    theta_seed=stable_seed(20261004,"resume",2,None,"theta")
    np.testing.assert_array_equal(a["theta"],np.random.default_rng(theta_seed).uniform(-1,1,4))


def test_diagnostic_rng_cannot_change_actual_queries():
    p=SMCPosterior(particles=128,rng=np.random.default_rng(3))
    t=make_template(p,4)
    a,b=np.random.default_rng(10),np.random.default_rng(10)
    diagnostic=np.random.default_rng(11)
    state=SelectorState()
    for _ in range(64):
        select_query("PS",p,4,0,diagnostic,copy.deepcopy(state),t)
    _,qa=select_query("PS",p,4,0,a,state,t)
    _,qb=select_query("PS",p,4,0,b,state,t)
    np.testing.assert_array_equal(qa,qb)


def test_adaptive_marginal_doubled_resolution_frozen_history(tmp_path):
    job=dict(setting_id="resolution",M=16,beta=16,method="RC",interaction_id=0,answers=128)
    result=run_statistical(job,tmp_path)
    data=np.load(result["artifact"])
    refs=replay_history(data["q"],data["y"],16,"marginal",nodes=513,checkpoints=(32,64,128))
    for t,p in refs.items():
        mean,cov=p.moments()
        np.testing.assert_allclose(mean,data["mu"][t],atol=1e-5)
        np.testing.assert_allclose(np.diag(cov),np.diag(data["cov"][t]),rtol=.005)
        if t<128:
            eig=p.eig(data["q"][t])
            assert abs(eig/data["eig"][t]-1)<.005


def test_cuda_cpu_frozen_history_target_agree():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng=np.random.default_rng(31)
    p=SMCPosterior(d=4,beta=16,particles=64,rng=rng)
    p.history=[(rng.normal(size=(16,4)),int(rng.integers(16))) for _ in range(128)]
    p._refresh_coefficients()
    points=rng.uniform(-1,1,(400,4))
    cpu=p.target_logdensity(points,.375)
    p.device="cuda:3" if torch.cuda.device_count()>=4 else "cuda:0"
    gpu=p.target_logdensity(points,.375)
    np.testing.assert_allclose(gpu,cpu,rtol=1e-13,atol=1e-9)
