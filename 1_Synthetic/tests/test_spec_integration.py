from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm
from scipy.special import expit

from pscore_harness.config import RunSpec, expand_config, load_yaml
from pscore_harness.datasets import make_synthetic_bundle
from pscore_harness.experiment import load_run_trajectories, run_spec
from pscore_harness.glm import GLMFamily
from pscore_harness.plotting import METHOD_STYLES
from pscore_harness.spec_suite import (GAUSSIAN_METHODS, BERNOULLI_METHODS,
    information_matrix, select_batch, selector_seed)
from pscore_harness.spec_suite.numerics import derivatives, fit_glm
from pscore_harness.spec_suite.selectors import fb_opt2_scale, stable_priorities, shared_initial_ids
from pscore_harness.spec_suite.runner import validate_completed_run


def tiny_spec(method="fb_opt2", family="gaussian", seed=0):
    return RunSpec("pytest_spec", "synthetic", {
        "name": "anisotropic_ball", "dimension": 3, "representation_dimension": 3,
        "n_candidates": 90, "radius": 1., "anisotropy": 1., "weak_rank": 1,
        "weak_fraction": .1, "signal_std": .6, "with_replacement": False,
        "max_common_randomness": 200}, {"family": family},
        {"name": "batch", "algorithm_suite": "spec_v1", "n_init": 12,
         "budget": 8, "batch_size": 4, "candidate_limit": 24},
        {"name": method}, {"ridge": 1., "tolerance": 1e-8}, {}, seed, "results", "cache")


def test_main_registry_and_future_plot_convention():
    root = Path(__file__).resolve().parents[1]
    specs = expand_config(load_yaml(root / "configs/synthetic_main.yaml"))
    assert len(specs) == 2160
    for family, names in (("gaussian", GAUSSIAN_METHODS), ("logistic", BERNOULLI_METHODS)):
        assert {s.method["name"] for s in specs if s.model["family"] == family} == set(names)
    assert METHOD_STYLES["fb_opt2"] == {"color": "#0057B8", "linestyle": "-"}
    assert METHOD_STYLES["ours"] == METHOD_STYLES["fb_opt2"]


def test_initialization_shared_across_nested_dimensions():
    spec = tiny_spec()
    a = make_synthetic_bundle({**spec.dataset, "dimension": 3, "representation_dimension": 6}, GLMFamily("gaussian"), 7)
    b = make_synthetic_bundle({**spec.dataset, "dimension": 6, "representation_dimension": 6}, GLMFamily("gaussian"), 7)
    np.testing.assert_array_equal(a.X, b.X[:, :3])
    np.testing.assert_array_equal(shared_initial_ids(a.initialization_X, "toy", 7, 12),
                                  shared_initial_ids(b.initialization_X, "toy", 7, 12))


def test_ours_matches_dense_gibbs_formula_and_final_temperature():
    rng = np.random.default_rng(231)
    X = rng.normal(size=(22, 3)) / 3
    old = np.arange(5, dtype=np.int64)
    ids = np.arange(5, 22, dtype=np.int64)
    theta = np.array([.2, -.7, .4])
    picked, audit = select_batch("ours", X, old, ids, theta, "logistic", 4, 1.,
        rng, "toy", 5, 3, planned_rounds=3)
    scale = fb_opt2_scale("bernoulli", 3, 4, 3)
    assert audit["temperature"] == scale["tau_range"]
    W = information_matrix(X[old], theta, "bernoulli")
    w = expit(X[ids] @ theta) * expit(-X[ids] @ theta)
    U = np.sqrt(w)[:, None] * X[ids]
    valid = np.ones(len(ids), dtype=bool)
    priority = stable_priorities(ids, "tie", "toy", 5, 3)
    for chosen in picked:
        P = expm(-(W - np.linalg.eigvalsh(W)[0] * np.eye(3)) / scale["tau_range"])
        P /= np.trace(P)
        scores = np.einsum("ij,jk,ik->i", U, P, U)
        best = scores[valid].max()
        ties = np.flatnonzero(valid & (np.abs(scores-best) <= 1e-11 * max(1., abs(best))))
        j = ties[np.argmin(priority[ties])]
        assert chosen == ids[j]
        W += np.outer(U[j], U[j])
        valid[j] = False


def test_exact_logistic_likelihood_and_evaluation_do_not_clip_or_floor():
    X = np.eye(2)
    theta = np.array([1000., -1000.])
    y = np.array([0., 1.])
    f, g, H = derivatives(theta, X, y, "bernoulli")
    assert f == pytest.approx(1002000.)
    np.testing.assert_array_equal(g, [1001., -1001.])
    np.testing.assert_array_equal(H, np.eye(2))
    np.testing.assert_array_equal(information_matrix(X, theta, "logistic"), np.eye(2))
    fitted, audit = fit_glm(X, y, "bernoulli", theta0=theta)
    assert audit["fit_converged"]
    assert audit["fit_grad_relative"] <= 1e-8
    assert np.isfinite(fitted).all()


@pytest.mark.parametrize("family,method", [("gaussian", n) for n in GAUSSIAN_METHODS]
                         + [("logistic", n) for n in BERNOULLI_METHODS])
def test_each_eligible_method_runs_and_saves_full_batch_contract(tmp_path, family, method):
    spec = tiny_spec(method, family)
    path = run_spec(spec, tmp_path)
    validate_completed_run(path)
    ids = np.load(path / "selected_indices.npy")
    initial = np.load(path / "initial_indices.npy")
    assert len(ids) == len(np.unique(ids)) == 8
    assert not np.intersect1d(initial, ids).size
    frame = pd.read_csv(path / "trajectory.csv")
    assert frame["acquired"].tolist() == [0, 4, 8]
    assert np.isfinite(frame[["lambda_min", "plugin_lambda_min", "squared_error"]]).all().all()
    for t in (1, 2):
        record = json.loads((path / "checkpoints" / f"round_{t:06d}.json").read_text())
        audit = record["acquisition"]
        with np.load(path / "checkpoints" / f"round_{t-1:06d}.npz") as previous:
            np.testing.assert_array_equal(audit["frozen_theta"], previous["theta_hat"])
            assert not np.intersect1d(audit["candidate_ids"], previous["observed_ids"]).size
        assert len(audit["selected_ids"]) == 4
        assert record["fit"]["fit_converged"]
        assert "rng_state_before" in audit and "rng_state_after" in audit
        if method == "e_sdp":
            assert not audit["fallback_used"], audit.get("sdp_attempts")
    if method == "fb_opt2":
        assert audit["temperature"] == audit["fb_opt2"]["tau_range"]


def test_interrupted_stochastic_run_resumes_exact_rng_and_parameter_state(tmp_path, monkeypatch):
    from pscore_harness.spec_suite import runner
    spec = tiny_spec("ts_glm", "logistic")
    baseline = run_spec(replace(spec, output_root="baseline"), tmp_path)
    original = runner.select_batch
    count = 0
    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected interruption after one committed batch")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "select_batch", interrupted)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_spec(spec, tmp_path)
    monkeypatch.setattr(runner, "select_batch", original)
    resumed = run_spec(spec, tmp_path)
    np.testing.assert_array_equal(np.load(baseline / "selected_indices.npy"), np.load(resumed / "selected_indices.npy"))
    with np.load(baseline / "checkpoints/round_000002.npz") as a, np.load(resumed / "checkpoints/round_000002.npz") as b:
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    assert not (resumed / "failed.json").exists()


def test_completed_checkpoint_corruption_is_detected(tmp_path):
    path = run_spec(tiny_spec(), tmp_path)
    with (path / "checkpoints/round_000001.npz").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(RuntimeError, match="corrupted checkpoint"):
        run_spec(tiny_spec(), tmp_path)


def test_family_and_label_boundaries():
    X = np.eye(3)
    args = (X, np.array([0]), np.array([1, 2]), np.zeros(3), "gaussian", 1, 1.,
            np.random.default_rng(9), "toy", 0, 1)
    with pytest.raises(ValueError, match="not applicable"):
        select_batch("firal", *args)
    with pytest.raises(ValueError, match="Only B-EMCM"):
        select_batch("a_opt", *args, observed_y=np.array([1.]))
    with pytest.raises(ValueError, match="forbidden"):
        select_batch("a_opt", *args, settings={"theta_star": np.zeros(3)})
    assert selector_seed("toy", 3, 0, "ours") == selector_seed("toy", 3, 0, "fb_opt2")


def test_report_root_risks_use_uncentered_error_second_moment(tmp_path):
    from pscore_harness.spec_suite.reporting import generate_suite_report
    paths = [run_spec(tiny_spec(seed=seed), tmp_path) for seed in (0, 1)]
    frame = load_run_trajectories(tmp_path / "results")
    outputs = generate_suite_report(frame, tmp_path / "paper")
    endpoints = pd.read_csv(outputs["suite_endpoints"])
    errors = []
    for path in paths:
        with np.load(path / "checkpoints/round_000002.npz") as z:
            errors.append(z["signed_error"].copy())
    errors = np.asarray(errors)
    Q = errors.T @ errors / 2
    assert endpoints.iloc[0]["R2"] == pytest.approx(np.sqrt(np.trace(Q)))
    assert endpoints.iloc[0]["Rdir"] == pytest.approx(np.sqrt(np.linalg.eigvalsh(Q)[-1]))
    manifest = json.loads(outputs["suite_manifest"].read_text())
    assert manifest["ours_style"] == {"color": "#0057B8", "linestyle": "-"}
