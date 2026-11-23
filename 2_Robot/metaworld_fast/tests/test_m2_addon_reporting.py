import json
from pathlib import Path

import numpy as np
import pytest

from metaworld_fast.m2_addon_reporting import (
    METHODS, generate_report, load_runs, numerical_tables, physical_tables, report_context, randomized_tables, schedule_tables,
)
from metaworld_fast.m2_addon_validation import artifact_checks, oracle_bounds, _history_kind


def make_fixture(tmp_path, method="RC", interactions=(0, 1)):
    base, root = tmp_path / "base", tmp_path / "addon"
    (base / "statistical").mkdir(parents=True)
    root.mkdir()
    (root / "BASELINE_LOCK.json").write_text(json.dumps(dict(
        source_run=str(base), interaction_ids=list(interactions), oracle=dict(alpha_E=.5, rho_E=16.))))
    for interaction in interactions:
        stem = f"m02_b04__{method}__i{interaction}"
        q = np.zeros((128,2,4)); q[:,0,0] = -.5; q[:,1,0] = .5
        mu = np.zeros((129,4)); cov = np.tile(np.eye(4)/3,(129,1,1))
        data = dict(q=q, y=np.zeros(128,dtype=int), mu=mu, cov=cov, risk=np.full(129,4/3),
                    error=np.zeros(129), theta=np.zeros(4), eta=np.full(128,1.5),
                    eig=np.full(128,.3), reference_eig=np.full(128,.2),
                    reference_queries=np.tile(q[:,None,:,:],(1,7,1,1)))
        np.savez(base / "statistical" / f"{stem}.npz", **data)
        (base / "statistical" / f"{stem}.json").write_text(json.dumps(dict(
            setting_id="m02_b04",method=method,interaction_id=interaction,M=2,beta=4,dimension=4,
            answers=128,reference_scope="own_posterior_seven_scale_grid")))
    return base, root


def test_all_registry_rows_missing_not_zero_and_ratio_not_clipped(tmp_path):
    _, root = make_fixture(tmp_path, method="PS")
    runs = load_runs(root)
    runs[1]["data"]["eta"][31] = np.nan
    _, endpoints, status = numerical_tables(root,runs,[0,1])
    row = next(r for r in endpoints if r["method"]=="PS" and r["metric"]=="eta_grid" and r["t"]==31)
    assert row["mean"] == 1.5 and row["n_effective"] == 1
    assert row["sample_sd"] is None and row["descriptive_se"] is None
    missing = next(r for r in endpoints if r["method"]=="mem_line" and r["metric"]=="posterior_risk")
    assert missing["mean"] is None and missing["n_effective"] == 0
    assert len(status) == 2*len(METHODS)


def test_report_partial_cannot_claim_complete(tmp_path):
    _, root = make_fixture(tmp_path, method="PS")
    result = generate_report(root,make_plots=False)
    assert result["completed_runs"] == 2
    assert result["expected_runs"] == 28
    assert not result["scientific_complete"]
    assert result["numerical_validation_status"] == "not_run"
    assert result["pending_selected_clips"] == 512
    assert "GCQ-5" in (root / "REPORT.md").read_text()


def test_reject_unknown_eta_denominator(tmp_path):
    base, root = make_fixture(tmp_path)
    p = base / "statistical/m02_b04__RC__i0.json"
    meta = json.loads(p.read_text()); meta["reference_scope"] = "ellipsoid_achieved"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="denominator"):
        load_runs(root)


def test_ellipsoid_interval_direction_and_outside_ratio_above_one():
    mu=np.zeros(4); cov=np.eye(4)/3
    grid=np.zeros((7,2,4)); grid[:,0,0]=-.5; grid[:,1,0]=.5
    actual=np.zeros((2,4)); actual[:,0]=[-2,2]
    result=oracle_bounds(mu,cov,.4,.2,actual,grid,radius=1)
    assert result["resolved"]
    assert not result["actual_inside_ellipsoid"]
    assert result["eig_ellipsoid_lower"]==.2
    assert result["eta_ellipsoid_upper"]==2.
    assert result["eta_ellipsoid_lower"]==pytest.approx(.4/np.log(2))
    inside=oracle_bounds(mu,cov,.4,.2,grid[0],grid,radius=1)
    assert inside["eig_ellipsoid_lower"]==.4
    assert inside["eta_ellipsoid_upper"]==1.
    assert not inside["global_optimum_certified"]


def test_oracle_rejects_outside_grid_and_zero_denominator():
    mu=np.zeros(4);cov=np.eye(4)/3
    grid=np.ones((7,2,4))*4;actual=np.zeros((2,4))
    result=oracle_bounds(mu,cov,.1,.2,actual,grid,radius=1)
    assert not result["resolved"]
    assert result["eig_ellipsoid_lower"] is None
    assert result["eta_ellipsoid_upper"] is None
    grid[:]=0
    result=oracle_bounds(mu,cov,0.,0.,actual,grid,radius=1)
    assert not result["resolved"] and result["eta_ellipsoid_lower"] is None


def test_factorization_is_checked_on_displayed_candidates(tmp_path):
    _,root=make_fixture(tmp_path)
    run=load_runs(root)[0]
    assert _history_kind(run)=="marginal"
    run["data"]["q"][0,1,1]=.1
    with pytest.raises(ValueError,match="coordinate-marginal"):
        _history_kind(run)


def test_physical_deduplicates_and_keeps_reference_separate(tmp_path):
    base,root=make_fixture(tmp_path,interactions=(0,))
    (base/"rollouts").mkdir()
    record=dict(per_query=[dict(kind="selected"),dict(kind="reference")],per_candidate=[
        dict(request_index=0,candidate_slot=0,status="executed",simulation_valid=True),
        dict(request_index=0,candidate_slot=1,status="cache_hit",simulation_valid=True),
        dict(request_index=1,candidate_slot=0,status="executed",simulation_valid=True)])
    path=base/"rollouts/m02_b04__RC__i0.json";path.write_text(json.dumps(record))
    rows=physical_tables(root,load_runs(root))
    assert rows[0]["covered"]==2 and rows[0]["pending"]==254
    assert rows[0]["executed"]==1 and rows[0]["cache_hits"]==1
    record["per_candidate"].append(record["per_candidate"][0]);path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="Duplicate logical"):
        physical_tables(root,load_runs(root))


def test_recorded_risk_and_error_are_distinct(tmp_path):
    _,root=make_fixture(tmp_path)
    run=load_runs(root)[0]
    # Use a valid RC EIG ratio before checking the moment audit.
    run["data"]["eta"][:]=1.;run["data"]["eig"][:]=.2
    checked=artifact_checks(run,report_context(root))
    assert checked["status"]=="passed"
    run["data"]["error"][:]=run["data"]["risk"]
    checked=artifact_checks(run,report_context(root))
    assert checked["status"]=="failed"
    assert any("squared error" in error for error in checked["errors"])


def test_randomized_means_keep_conditional_scope_and_no_ratio_clip(tmp_path):
    _,root=make_fixture(tmp_path,method="PS",interactions=(0,))
    run=load_runs(root)[0]
    run["data"]["diag_t"]=np.array([31])
    run["data"]["diag_eig"]=np.full((1,64),.4)
    (root/"oracle_sidecars").mkdir()
    (root/"oracle_sidecars/m02_b04__PS__i0.json").write_text(json.dumps(dict(rounds=[dict(
        t=31,resolved=True,eig_ellipsoid_upper=.6,eig_ellipsoid_lower=.2)])))
    row=randomized_tables(root,[run])[0]
    assert row["mean_eta_grid"]==pytest.approx(2)
    assert row["mean_eta_ellipsoid_upper"]==pytest.approx(2)
    assert row["finite_draws"]==64 and row["actual_answers_added"]==0


def test_device_checks_are_explicit_and_preserve_failures(tmp_path,monkeypatch):
    from metaworld_fast import m2_addon_validation as module
    _,root=make_fixture(tmp_path,method="GC",interactions=(0,))
    called=[]
    def fake_check(run,device):
        called.append(device)
        return dict(device=device,status="unresolved" if device=="cuda:1" else "passed")
    monkeypatch.setattr(module,"cpu_gpu_fixed_history_check",fake_check)
    result=module.validate_devices(root)
    assert called==["cuda:0","cuda:1","cuda:2"]
    assert result["status"]=="unresolved"
    assert result["actual_histories_modified"] is False


def test_schedule_cap_flags_are_per_candidate(tmp_path):
    data=dict(y=np.zeros(2),selector_stage=np.zeros(2),selector_center=np.zeros((2,4)),
              selector_std=np.full(2,1/np.sqrt(3)),cap_activated=np.array([[False,False],[True,False]]))
    run=dict(method="gausscloud_paper_L1",label="GCQ-1",interaction_id=0,data=data)
    rows=schedule_tables(tmp_path,[run])
    assert not rows[0]["cap_activated"] and rows[1]["cap_activated"]
    assert json.loads(rows[1]["cap_candidate_flags"])==[True,False]


def test_real_runner_arrays_all_nine_methods_roundtrip_as_partial(tmp_path):
    """Actual CPU engineering histories exercise the reporting schema, not results."""
    import pickle
    from metaworld_fast.m2_addon_reporting import NEW_METHODS
    from metaworld_fast.m2_addon_runner import arrays, run_statistical
    from metaworld_fast.model import stable_seed
    from metaworld_fast.statistical_runner import run_key

    _,root=make_fixture(tmp_path,interactions=(0,))
    root_seed=20261004
    theta=np.random.default_rng(stable_seed(root_seed,"m02_b04",0,None,"theta")).uniform(-1,1,4)
    for method in NEW_METHODS:
        job=dict(setting_id="m02_b04",method=method,interaction_id=0,M=2,dimension=4,beta=4.,
                 answers=128,nodes=257,particles=32,root_seed=root_seed,theta_star=theta.tolist(),
                 device="cpu",kind="engineering",config_hash="test",code_hash="test",base_lock_hash="test")
        run_statistical(job,root,max_answers=2,diagnostic_draws=2)
        with (root/"checkpoints"/f"{run_key(job)}.pkl").open("rb") as handle:
            state=pickle.load(handle)
        np.savez(root/"statistical"/f"{run_key(job)}.npz",**arrays(state))
        (root/"statistical"/f"{run_key(job)}.json").write_text(json.dumps(dict(
            **job,answers_completed=2,reference_scope="own_posterior_seven_scale_grid")))
    result=generate_report(root,make_plots=False)
    assert result["completed_runs"]==1
    assert result["new_completed_runs"]==0
    assert result["new_actual_answers"]==18
    assert result["expected_runs"]==14
    assert result["randomized_diagnostic_checkpoints"]==4
    assert not result["scientific_complete"]
