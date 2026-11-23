"""Analytic-gradient, domain and witness tests for the new reference only."""
import numpy as np
import pytest
from metaworld_fast.ellipsoid_optimizer import (
    ellipsoid_geometry, normalized_candidate_norms, eig_value_gradient,
    optimize_ellipsoid,
)
from metaworld_fast.metrics import integrated_eig


def case(seed=100,M=4,N=128):
    rng=np.random.default_rng(seed)
    points=rng.uniform(-1,1,(N,4))
    weights=rng.random(N);weights/=weights.sum()
    mu=weights@points
    dx=points-mu
    cov=dx.T@(dx*weights[:,None])
    q=mu+rng.normal(scale=.6,size=(M,4))
    b=np.linspace(-1.5,1.5,M)
    grid=mu[None,None,:]+np.array([.125,.2,.32,.5,.8,1.25,2.])[:,None,None]*b[None,:,None]*np.array([1,0,0,0])
    return points,weights,mu,cov,q,grid


def test_analytic_gradient_matches_finite_difference_and_original_eig():
    points,weights,mu,cov,q,grid=case()
    eig,gradient=eig_value_gradient(points,weights,q,4.)
    assert abs(eig-integrated_eig(points,weights,q,4))<2e-13
    finite=np.empty_like(q)
    eps=1e-6
    for m in range(len(q)):
        for d in range(4):
            plus,minus=q.copy(),q.copy()
            plus[m,d]+=eps;minus[m,d]-=eps
            finite[m,d]=(integrated_eig(points,weights,plus,4)-integrated_eig(points,weights,minus,4))/(2*eps)
    np.testing.assert_allclose(gradient,finite,rtol=2e-6,atol=2e-8)
    # Common translation leaves objective and its derivative unchanged.
    shifted,gshift=eig_value_gradient(points+200,weights,q+200,4.)
    np.testing.assert_allclose([shifted],eig,atol=2e-12)
    np.testing.assert_allclose(gshift,gradient,atol=2e-12)


def test_geometry_preserves_full_covariance_and_constraints():
    mu=np.array([.1,-.2,.3,.4])
    cov=np.array([[.2,.15,0,0],[.15,.2,0,0],[0,0,.1,.02],[0,0,.02,.03]])
    root,inverse,shape=ellipsoid_geometry(mu,cov)
    np.testing.assert_allclose(root@root,shape,atol=1e-14)
    np.testing.assert_allclose(root@inverse,np.eye(4),atol=1e-14)
    assert .5<=np.linalg.eigvalsh(shape)[0]<=1
    assert abs(np.linalg.eigvalsh(shape)[-1]-1)<1e-14
    q=mu+16*np.eye(4)@root
    np.testing.assert_allclose(normalized_candidate_norms(q,mu,inverse),1,atol=1e-14)


def test_grid_and_feasible_actual_are_retained_and_search_can_improve():
    points,weights,mu,cov,q,grid=case(M=4)
    result=optimize_ellipsoid(points,weights,mu,cov,4,q,grid,device="cpu",starts=4,steps=12,seed=5)
    assert result["actual_inside"]
    assert result["achieved_eig"]>=max(result["actual_eig"],max(result["grid_eigs"]))-1e-12
    assert result["achieved_eig"]>=result["initial_best_eig"]-1e-12
    assert result["best_max_normalized_radius"]<=1+1e-12
    assert abs(integrated_eig(points,weights,result["best_q"],4)-result["achieved_eig"])<1e-12
    assert not result["global_certified"] and not result["integration_certified"]
    assert result["ratio_approx"]<=1+1e-12
    assert result["global_gap_estimate"]>=-1e-12
    assert np.linalg.matrix_rank(result["best_q"]-result["best_q"].mean(axis=0))>1


def test_outside_actual_is_not_reference_witness_and_numerator_unmodified():
    points,weights,mu,cov,q,grid=case(M=2)
    outside=mu+np.array([[-100.,0,0,0],[100.,0,0,0]])
    result=optimize_ellipsoid(points,weights,mu,cov,4,outside,grid,device="cpu",starts=2,steps=0,seed=6)
    assert not result["actual_inside"]
    assert abs(result["actual_eig"]-integrated_eig(points,weights,outside,4))<1e-12
    assert result["best_max_normalized_radius"]<=1+1e-12
    badgrid=grid.copy();badgrid[0]=outside
    with pytest.raises(ValueError,match="grid"):
        optimize_ellipsoid(points,weights,mu,cov,4,q,badgrid,device="cpu",steps=0)


def test_per_history_seeds_preserve_batching_and_feasible_warm_witness():
    points,weights,mu,cov,q,grid=case(M=4)
    single=optimize_ellipsoid(points,weights,mu,cov,4,q,grid,device="cpu",starts=3,steps=4,seed=17)
    batch=optimize_ellipsoid(np.stack([points,points]),np.stack([weights,weights]),np.stack([mu,mu]),np.stack([cov,cov]),4,
                            np.stack([q,q]),np.stack([grid,grid]),device="cpu",starts=3,steps=4,seed=[17,19])
    np.testing.assert_allclose(single["achieved_eig"],batch["achieved_eig"][0],atol=1e-13)
    np.testing.assert_allclose(single["best_q"],batch["best_q"][0],atol=1e-12)
    warm=optimize_ellipsoid(points,weights,mu,cov,4,q,grid,device="cpu",starts=3,steps=0,seed=17,warm_q=single["best_q"])
    assert warm["warm_inside"]
    assert warm["achieved_eig"]>=single["achieved_eig"]-1e-12


def test_small_eig_and_identical_candidates():
    points=np.linspace(-1,1,257)[:,None]
    weights=np.ones(len(points))/len(points)
    q=np.array([[-1e-8],[1e-8]])
    value,gradient=eig_value_gradient(points,weights,q,1.)
    assert value>0
    assert abs(value/integrated_eig(points,weights,q,1)-1)<1e-12
    identical,g=eig_value_gradient(points,weights,np.zeros((4,1)),16.)
    assert abs(identical)<1e-25
    np.testing.assert_allclose(g,0,atol=1e-25)


def test_cuda_cpu_objective_gradient_with_large_offsets():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    points,weights,mu,cov,q,grid=case(M=16,N=512)
    q=mu+12*(q-mu)/np.linalg.norm(q-mu,axis=1,keepdims=True)
    a,da=eig_value_gradient(points,weights,q,16.,device="cpu")
    b,db=eig_value_gradient(points,weights,q,16.,device="cuda:0")
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(da,db,rtol=1e-10,atol=2e-11)


def test_large_integer_seed_list_is_exact_and_batch_independent():
    points,weights,mu,cov,q,grid=case(M=4,N=32)
    seeds=[17151354684409484346,17]
    batch=optimize_ellipsoid(np.stack([points,points]),np.stack([weights,weights]),np.stack([mu,mu]),np.stack([cov,cov]),4,
                            np.stack([q,q]),np.stack([grid,grid]),device="cpu",starts=3,steps=2,seed=seeds)
    assert batch["initialization_seed"].dtype==np.uint64
    assert [int(s) for s in batch["initialization_seed"]]==seeds
    for i,seed in enumerate(seeds):
        single=optimize_ellipsoid(points,weights,mu,cov,4,q,grid,device="cpu",starts=3,steps=2,seed=seed)
        np.testing.assert_allclose(batch["best_q"][i],single["best_q"],atol=1e-12)
        np.testing.assert_allclose(batch["achieved_eig"][i],single["achieved_eig"],atol=1e-13)
    with pytest.raises(ValueError,match="exact"):
        optimize_ellipsoid(points,weights,mu,cov,4,q,grid,device="cpu",steps=0,seed=float(seeds[0]))
