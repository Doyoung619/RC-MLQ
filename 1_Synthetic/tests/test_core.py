from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pscore_harness.acquisition.base import AcquisitionContext
from pscore_harness.acquisition.optimal_design import a_opt_scores, d_opt_scores, e_opt_scores
from pscore_harness.acquisition.proposed import hard_scores, soft_scores
from pscore_harness.config import RunSpec
from pscore_harness.experiment import _candidate_subset, run_spec
from pscore_harness.glm import GLMFamily
from pscore_harness.information import minimum_eigenvalue, rank_one_update


def test_information_minimum_eigenvalue_is_monotone() -> None:
    rng = np.random.default_rng(7)
    V = np.eye(5)
    previous = minimum_eigenvalue(V)
    for _ in range(20):
        x = rng.normal(size=5)
        V = rank_one_update(V, x, weight=float(rng.uniform(0.01, 0.25)))
        current = minimum_eigenvalue(V)
        assert current >= previous - 1e-10
        previous = current


def test_projected_and_oed_scores_are_finite() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)
    context = AcquisitionContext(
        X=X,
        candidate_ids=np.arange(len(X)),
        theta_hat=rng.normal(size=6),
        V=np.eye(6) + X[:10].T @ X[:10],
        family=GLMFamily("logistic"),
        rng=rng,
        method={"name": "test", "batch_size": 4, "radius": 1.0, "eopt_exact_limit": 100},
        batch_position=0,
    )
    for scorer in [hard_scores, soft_scores, a_opt_scores, d_opt_scores, e_opt_scores]:
        scores = scorer(context)
        assert scores.shape == (len(X),)
        assert np.isfinite(scores).all()


def test_poisson_family_response_curvature_and_ridge_fit_are_finite() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(250, 5))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)
    theta_star = np.asarray([0.5, -0.25, 0.15, 0.1, -0.2])
    family = GLMFamily("poisson", poisson_intercept=-0.4)
    mu = family.mean(X @ theta_star)
    y = family.sample_from_common_randomness(X @ theta_star, rng.uniform(size=len(X)))
    theta_hat = family.fit_ridge(X, y, ridge=0.2, max_iter=100, tol=1e-9)
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)
    assert np.all(np.isfinite(family.curvature(X @ theta_hat)))
    assert np.all(y >= 0) and np.all(y == np.floor(y))
    assert np.all(np.isfinite(theta_hat))


def test_poisson_soft_score_uses_candidate_curvature_scale() -> None:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(60, 4))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)
    context = AcquisitionContext(
        X=X,
        candidate_ids=np.arange(len(X)),
        theta_hat=rng.normal(size=4),
        V=np.eye(4) + X[:12].T @ X[:12],
        family=GLMFamily("poisson", poisson_intercept=-0.5),
        rng=rng,
        method={"name": "soft", "batch_size": 8, "radius": 1.0, "tau_mode": "theory"},
        batch_position=0,
    )
    scores = soft_scores(context)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)


def test_tiny_run_writes_only_prespecified_metrics(tmp_path: Path) -> None:
    spec = RunSpec(
        experiment_name="pytest_smoke",
        source="synthetic",
        dataset={
            "name": "anisotropic_ball",
            "dimension": 5,
            "n_candidates": 80,
            "radius": 1.0,
            "anisotropy": 1.0,
            "weak_fraction": 0.1,
            "weak_rank": 2,
            "theta_regime": "weak",
            "signal_std": 0.7,
            "with_replacement": True,
            "max_common_randomness": 200,
        },
        model={"family": "gaussian", "noise_std": 1.0},
        protocol={
            "name": "sequential",
            "n_init": 10,
            "initialization": "d_opt",
            "budget": 4,
            "batch_size": 1,
            "candidate_limit": 60,
        },
        method={"name": "hard", "label": "Hard-PScore"},
        estimator={"ridge": 1.0, "max_iter": 20, "tolerance": 1e-8},
        metrics={"names": ["l2_error", "lambda_min"]},
        seed=0,
        output_root="results",
        cache_root="cache",
    )
    run_dir = run_spec(spec, tmp_path)
    trajectory = pd.read_csv(run_dir / "trajectory.csv")
    assert {"l2_error", "lambda_min"}.issubset(trajectory.columns)
    assert trajectory["acquired"].tolist() == [0, 1, 2, 3, 4]
    assert np.isfinite(trajectory[["l2_error", "lambda_min"]].to_numpy()).all()
    assert np.all(np.diff(trajectory["lambda_min"].to_numpy()) >= -1e-9)
    assert (run_dir / "complete.json").exists()



def test_candidate_subsampling_has_global_id_priorities() -> None:
    full = np.arange(200, dtype=int)
    reduced = full[~np.isin(full, [3, 7, 19, 41, 88])]
    chosen_full = _candidate_subset(full, 40, seed=5, batch_index=9)
    chosen_reduced = _candidate_subset(reduced, 40, seed=5, batch_index=9)

    # Removing unavailable rows must not change the priority order of rows that
    # remain.  The new subset may replace removed winners, but it cannot drop a
    # common winner in favor of a lower-priority common row.
    common_winners = np.intersect1d(chosen_full, reduced)
    assert set(common_winners).issubset(set(chosen_reduced))
    assert np.array_equal(
        chosen_full,
        _candidate_subset(full, 40, seed=5, batch_index=9),
    )
