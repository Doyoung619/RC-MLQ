"""Small CPU engineering histories: RNG, stage-boundary resume, and scope."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from metaworld_fast.m2_addon_runner import make_state, run_statistical
from metaworld_fast.model import probabilities, stable_seed
from metaworld_fast.statistical_runner import run_key


def engineering_job(method="sm_continuous", iid=3):
    root = 20261004
    seed = stable_seed(root, "m02_b04", iid, None, "theta")
    theta = np.random.default_rng(seed).uniform(-1, 1, 4).tolist()
    return dict(setting_id="m02_b04", method=method, interaction_id=iid,
                M=2, dimension=4, beta=4., answers=128, nodes=257, particles=32,
                root_seed=root, theta_star=theta, context_id=iid, device="cpu",
                config_hash="engineering_test", code_hash="engineering_test",
                base_lock_hash="engineering_test", kind="engineering_test")


def checkpoint_data(root, job):
    with (root/"checkpoints"/f"{run_key(job)}.pkl").open("rb") as stream:
        return pickle.load(stream)


@pytest.mark.parametrize("cutoff", [41, 42, 43])
def test_gcq_joint_exact_resume_around_actual_stage_entry(tmp_path, cutoff):
    job = engineering_job("gausscloud_paper_L3")
    full, resume = tmp_path/"full", tmp_path/"resume"
    run_statistical(job, full, max_answers=44, diagnostic_draws=2)
    run_statistical(job, resume, max_answers=cutoff, diagnostic_draws=2)
    run_statistical(job, resume, max_answers=44, diagnostic_draws=2)
    a, b = checkpoint_data(full, job), checkpoint_data(resume, job)
    for key in ("q", "q_raw", "y", "eig", "eta", "mu", "cov", "risk", "error",
                "selector_center", "selector_stage", "selector_std", "reference_queries"):
        np.testing.assert_array_equal(a["data"][key], b["data"][key])
    assert a["answers_completed"] == b["answers_completed"] == 44
    assert a["data"]["selector_stage"][41] == 0
    assert a["data"]["selector_stage"][42] == 1
    np.testing.assert_array_equal(a["data"]["selector_center"][42], a["data"]["mu"][42])
    np.testing.assert_array_equal(a["data"]["selector_center"][43], a["data"]["mu"][42])
    assert a["data"]["selector_std"][42] == .5*a["data"]["selector_std"][41]
    for purpose in ("response", "selector", "smc", "diagnostics"):
        assert a["rng"][purpose].bit_generator.state == b["rng"][purpose].bit_generator.state
    # The particles implement the entire saved actual likelihood history.
    assert len(a["posterior"].history) == 44
    np.testing.assert_allclose(a["posterior"].target_logdensity(a["posterior"].particles),
                               a["posterior"].log_target, atol=2e-12)


def test_sm_diagnostic_draw_count_does_not_change_actual_history(tmp_path):
    job = engineering_job()
    for name, count in (("few", 2), ("more", 5)):
        run_statistical(job, tmp_path/name, max_answers=33, diagnostic_draws=count)
    a = checkpoint_data(tmp_path/"few", job)
    b = checkpoint_data(tmp_path/"more", job)
    for key in ("q", "q_raw", "y", "eig", "mu", "cov", "risk", "error"):
        np.testing.assert_array_equal(a["data"][key], b["data"][key])
    assert np.asarray(a["data"]["diag_eig"]).shape == (2, 2)
    assert np.asarray(b["data"]["diag_eig"]).shape == (2, 5)
    for purpose in ("response", "selector", "smc"):
        assert a["rng"][purpose].bit_generator.state == b["rng"][purpose].bit_generator.state


def test_scope_guards_frozen_latent_registry_independence_and_analytic_prior():
    methods = ("gausscloud_paper_L3", "mem_line", "sm_continuous")
    forward = {method: make_state(engineering_job(method)) for method in methods}
    backward = {method: make_state(engineering_job(method)) for method in reversed(methods)}
    for method in methods:
        a, b = forward[method], backward[method]
        assert a["seeds"] == b["seeds"]
        np.testing.assert_array_equal(a["theta"], b["theta"])
        np.testing.assert_array_equal(a["theta"], forward[methods[0]]["theta"])
        np.testing.assert_allclose(a["data"]["mu"][0], 0., atol=1e-16)
        np.testing.assert_allclose(a["data"]["cov"][0], np.eye(4)/3, atol=1e-15)
        assert abs(a["data"]["risk"][0]-4/3) < 1e-14
        assert abs(a["data"]["error"][0]-np.sum(a["theta"]**2)) < 1e-14
    for key, value in (("M", 4), ("answers", 4096), ("beta", 8.),
                       ("dimension", 3), ("setting_id", "m04_b04"), ("method", "SG"), ("method", "RC")):
        job = engineering_job(); job[key] = value
        with pytest.raises(ValueError):
            make_state(job)
    job = engineering_job(); job["theta_star"][0] += .001
    with pytest.raises(ValueError, match="theta/seed"):
        make_state(job)


def test_actual_response_uses_displayed_queries_and_own_frozen_latent(tmp_path):
    job = engineering_job("infosynth_bounded")
    run_statistical(job, tmp_path, max_answers=3, diagnostic_draws=2)
    state = checkpoint_data(tmp_path, job)
    response = np.random.default_rng(state["seeds"]["response"])
    for q, saved_y in zip(state["data"]["q"], state["data"]["y"]):
        expected_y = response.choice(2, p=probabilities(state["theta"], q, 4.))
        assert saved_y == expected_y
    assert len(state["posterior"].history) == 3
    for (history_q, history_y), q, y in zip(state["posterior"].history, state["data"]["q"], state["data"]["y"]):
        np.testing.assert_array_equal(history_q, q)
        assert history_y == y
    for text in state["data"]["selector_metadata"]:
        assert text["sigma0"] == .25
    assert state["posterior"].beta == 4.


def test_full_completion_atomic_publication_recovery_and_completed_skip(tmp_path, monkeypatch):
    """Engineering nodes=65 only: crash after final NPZ and before JSON marker."""
    from metaworld_fast import m2_addon_runner as runner
    job = engineering_job("mem_line")
    job["nodes"] = 65
    artifact = tmp_path/"statistical"/f"{run_key(job)}.npz"
    metadata = artifact.with_suffix(".json")
    real_atomic_json = runner.atomic_json
    def fail_final_metadata(path, value):
        if Path(path) == metadata:
            raise OSError("engineering injected publication crash")
        return real_atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "atomic_json", fail_final_metadata)
        with pytest.raises(OSError, match="injected publication crash"):
            run_statistical(job, tmp_path, diagnostic_draws=2)
    assert artifact.exists() and not metadata.exists()
    saved = checkpoint_data(tmp_path, job)
    assert saved["answers_completed"] == 128
    with np.load(artifact, allow_pickle=False) as data:
        original = {key: data[key].copy() for key in data.files}
    result = run_statistical(job, tmp_path, diagnostic_draws=2)
    assert result["complete"] and metadata.exists()
    summary = json.loads(metadata.read_text())
    assert summary["answers_completed"] == 128 and summary["job"] == job
    assert summary["exact_eig_formula_used"] and not summary["exact_posterior_computed"]
    assert summary["eta_field_alias"] == "eta_grid"
    with np.load(artifact, allow_pickle=False) as data:
        for key, expected in original.items():
            np.testing.assert_array_equal(data[key], expected)
        assert data["q"].shape == (128, 2, 4)
        assert data["risk"].shape == data["realized_sq_error"].shape == (129,)
        assert data["reference_queries"].shape == (128, 7, 2, 4)
        assert data["eta_grid"].shape == (128,)
        assert float(data["risk"][0]) == pytest.approx(4/3)
    stamp = artifact.stat().st_mtime_ns
    reused = run_statistical(job, tmp_path, diagnostic_draws=2)
    assert reused["reused"] and artifact.stat().st_mtime_ns == stamp
    changed = copy.deepcopy(job); changed["config_hash"] = "different"
    with pytest.raises(ValueError, match="different frozen job"):
        run_statistical(changed, tmp_path, diagnostic_draws=2)
