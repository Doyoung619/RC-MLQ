from __future__ import annotations

import json
import math
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .acquisition import AcquisitionContext, choose_one
from .acquisition.proposed import hard_scores, soft_scores
from .config import RUN_SCHEMA_VERSION, RunSpec
from .datasets import DatasetBundle, make_dataset_bundle
from .glm import GLMFamily, make_family
from .information import (
    d_opt_initial_indices,
    information_matrix,
    minimum_eigenvalue,
    random_initial_indices,
    rank_one_update,
)
from .utils import (
    atomic_write_json,
    environment_manifest,
    git_commit,
    jsonable,
    seed_everything,
    stable_hash,
)


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _method_rng(seed: int, method_name: str) -> np.random.Generator:
    method_code = sum((i + 1) * ord(c) for i, c in enumerate(method_name))
    return np.random.default_rng(seed * 1_000_003 + method_code + 77)


def _candidate_subset(
    available: np.ndarray,
    candidate_limit: int | None,
    seed: int,
    batch_index: int,
) -> np.ndarray:
    """Return a method-independent candidate subset.

    Each global candidate id receives a deterministic pseudo-random priority
    that depends only on the response seed and acquisition batch.  Methods
    that have removed different rows therefore retain the same ordering on
    every candidate that remains common to their pools.
    """
    available = np.asarray(available, dtype=np.int64)
    if candidate_limit is None or len(available) <= candidate_limit:
        return available.copy()

    # SplitMix64-style integer mixing.  Using candidate ids rather than
    # sampling positions in ``available`` prevents method-specific pool
    # histories from changing the random priority of common rows.
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    batch_key = np.uint64(
        (int(seed) * 1_000_003 + int(batch_index) * 9_176 + 12_347)
        & int(mask)
    )
    values = available.astype(np.uint64, copy=False) ^ batch_key
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    priorities = values ^ (values >> np.uint64(31))
    chosen = np.argpartition(priorities, candidate_limit - 1)[:candidate_limit]
    # Sort by global id for stable vectorized scoring; the priority determines
    # membership, not the acquisition tie-break order.
    return np.sort(available[chosen].astype(int))


def _observe(
    bundle: DatasetBundle,
    family: GLMFamily,
    indices: list[int],
    observation_offset: int,
) -> np.ndarray:
    index_array = np.asarray(indices, dtype=int)
    z = bundle.X[index_array] @ bundle.theta_star
    if bundle.with_replacement:
        stop = observation_offset + len(indices)
        if stop > len(bundle.common_randomness):
            raise ValueError("Synthetic common-randomness buffer is too short")
        common = bundle.common_randomness[observation_offset:stop]
    else:
        common = bundle.common_randomness[index_array]
    return family.sample_from_common_randomness(z, common)


def _initial_indices(
    bundle: DatasetBundle,
    spec: RunSpec,
    ridge: float,
) -> list[int]:
    n_init = int(spec.protocol.get("n_init", max(2 * bundle.d, 10)))
    init_mode = str(spec.protocol.get("initialization", "d_opt"))
    rng = np.random.default_rng(spec.seed + 4_271)
    if init_mode == "d_opt":
        return d_opt_initial_indices(
            bundle.X,
            n_init=n_init,
            ridge=ridge,
            with_replacement=bundle.with_replacement,
            candidate_limit=spec.protocol.get("initial_candidate_limit"),
            rng=rng,
        )
    if init_mode == "random":
        return random_initial_indices(bundle.n, n_init, bundle.with_replacement, rng)
    raise ValueError(f"Unknown initialization: {init_mode}")


def _record_metrics(
    rows: list[dict[str, Any]],
    acquired: int,
    total_observations: int,
    theta_hat: np.ndarray,
    theta_star: np.ndarray,
    V_oracle: np.ndarray,
    batch_index: int,
) -> None:
    rows.append(
        {
            "acquired": int(acquired),
            "total_observations": int(total_observations),
            "batch_index": int(batch_index),
            "l2_error": float(np.linalg.norm(theta_hat - theta_star)),
            "lambda_min": minimum_eigenvalue(V_oracle),
        }
    )


def _topb_frozen_indices(
    base_acquisition: str,
    X: np.ndarray,
    candidate_ids: np.ndarray,
    theta: np.ndarray,
    V: np.ndarray,
    family: GLMFamily,
    rng: np.random.Generator,
    method_config: dict[str, Any],
    batch_size: int,
) -> list[int]:
    context = AcquisitionContext(
        X=X,
        candidate_ids=candidate_ids,
        theta_hat=theta,
        V=V,
        family=family,
        rng=rng,
        method=method_config,
        batch_position=0,
    )
    if base_acquisition == "hard":
        scores = hard_scores(context)
    elif base_acquisition == "soft":
        scores = soft_scores(context)
    else:
        raise ValueError(f"Frozen top-b is not implemented for {base_acquisition}")
    order = np.argsort(-scores, kind="mergesort")
    return order[: min(batch_size, len(order))].astype(int).tolist()


def _select_batch(
    bundle: DatasetBundle,
    family: GLMFamily,
    spec: RunSpec,
    theta_hat: np.ndarray,
    V_plugin: np.ndarray,
    V_oracle: np.ndarray,
    available: np.ndarray,
    batch_index: int,
    batch_size: int,
    method_rng: np.random.Generator,
) -> list[int]:
    candidate_limit_raw = spec.protocol.get("candidate_limit", spec.dataset.get("candidate_limit"))
    candidate_limit = int(candidate_limit_raw) if candidate_limit_raw is not None else None
    candidate_ids = _candidate_subset(available, candidate_limit, spec.seed, batch_index)
    if len(candidate_ids) == 0:
        return []

    method_config = dict(spec.method)
    method_config["batch_size"] = batch_size
    method_config["radius"] = float(spec.dataset.get("radius", bundle.metadata.get("radius", 1.0)))
    acquisition_name = str(method_config.get("acquisition", method_config["name"]))
    batch_mode = str(method_config.get("batch_mode", "pseudo_update"))
    oracle = bool(method_config.get("oracle", False))
    acquisition_theta = bundle.theta_star if oracle else theta_hat
    V_work = V_oracle.copy() if oracle else V_plugin.copy()

    X_candidates = bundle.X[candidate_ids]
    if batch_mode == "topb_frozen":
        local_indices = _topb_frozen_indices(
            base_acquisition=acquisition_name,
            X=X_candidates,
            candidate_ids=candidate_ids,
            theta=acquisition_theta,
            V=V_work,
            family=family,
            rng=method_rng,
            method_config=method_config,
            batch_size=batch_size,
        )
        return candidate_ids[np.asarray(local_indices, dtype=int)].astype(int).tolist()

    selected: list[int] = []
    local_ids = candidate_ids.copy()
    allow_batch_duplicates = bool(method_config.get("allow_batch_duplicates", False))
    for position in range(batch_size):
        if len(local_ids) == 0:
            break
        local_X = bundle.X[local_ids]
        context = AcquisitionContext(
            X=local_X,
            candidate_ids=local_ids,
            theta_hat=acquisition_theta,
            V=V_work,
            family=family,
            rng=method_rng,
            method=method_config,
            batch_position=position,
        )
        local_choice = choose_one(acquisition_name, context)
        global_choice = int(local_ids[local_choice])
        selected.append(global_choice)

        x = bundle.X[global_choice]
        weight = float(family.curvature(np.asarray([x @ acquisition_theta]))[0])
        V_work = rank_one_update(V_work, x, weight)
        if not allow_batch_duplicates:
            local_ids = np.delete(local_ids, local_choice)
    return selected


def _setting_label(spec: RunSpec, bundle: DatasetBundle) -> str:
    parts = [f"d={bundle.d}", f"theta={spec.dataset.get('theta_regime', 'weak')}"]
    aliases = [
        ("anisotropy", "alpha"),
        ("weak_fraction", "weak-frac"),
        ("weak_rank", "weak-rank"),
        ("duplicate_noise", "dup-noise"),
        ("ray_levels", "ray-levels"),
        ("signal_std", "signal-std"),
    ]
    for key, label in aliases:
        if key in spec.dataset:
            parts.append(f"{label}={spec.dataset[key]}")
    return ", ".join(parts)


def run_spec(
    spec: RunSpec,
    project_root: Path,
    overwrite: bool = False,
) -> Path:
    if spec.protocol.get("algorithm_suite") == "spec_v1":
        from .spec_suite.runner import run_spec_suite
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return run_spec_suite(spec, project_root, overwrite=overwrite)
    seed_everything(spec.seed)
    output_root = _resolve_path(project_root, spec.output_root) / spec.experiment_name
    run_dir = output_root / spec.run_id
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not overwrite:
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        requested_metrics = set(spec.metrics.get("names", ["l2_error", "lambda_min"]))
        required_metrics = {"l2_error", "lambda_min"}
        if requested_metrics != required_metrics:
            raise ValueError(
                "This paper harness intentionally reports exactly "
                f"{sorted(required_metrics)}, got {sorted(requested_metrics)}"
            )
        family = make_family(spec.model)
        cache_root = _resolve_path(project_root, spec.cache_root)
        bundle = make_dataset_bundle(
            source=spec.source,
            config=spec.dataset,
            family=family,
            seed=spec.seed,
            cache_root=cache_root,
        )
        ridge = float(spec.estimator.get("ridge", 1.0))
        max_iter = int(spec.estimator.get("max_iter", 80))
        tolerance = float(spec.estimator.get("tolerance", 1e-8))
        budget = int(spec.protocol.get("budget", 20 * bundle.d))
        batch_size = int(spec.protocol.get("batch_size", 1))
        if batch_size <= 0 or budget <= 0:
            raise ValueError("budget and batch_size must be positive")
        if not bundle.with_replacement:
            n_init_requested = int(spec.protocol.get("n_init", max(2 * bundle.d, 10)))
            if n_init_requested + budget > bundle.n:
                budget = max(1, bundle.n - n_init_requested)

        init_indices = _initial_indices(bundle, spec, ridge)
        observed_indices = list(init_indices)
        y_observed = _observe(bundle, family, init_indices, observation_offset=0).tolist()
        X_observed = bundle.X[np.asarray(observed_indices, dtype=int)].copy()
        y_array = np.asarray(y_observed, dtype=np.float64)
        theta_hat = family.fit_ridge(
            X_observed,
            y_array,
            ridge=ridge,
            theta_init=None,
            max_iter=max_iter,
            tol=tolerance,
        )
        V_plugin = information_matrix(family, X_observed, theta_hat, ridge)
        V_oracle = information_matrix(family, X_observed, bundle.theta_star, ridge)

        if bundle.with_replacement:
            available = np.arange(bundle.n, dtype=int)
        else:
            mask = np.ones(bundle.n, dtype=bool)
            mask[np.unique(np.asarray(observed_indices, dtype=int))] = False
            available = np.flatnonzero(mask)

        rows: list[dict[str, Any]] = []
        _record_metrics(
            rows,
            acquired=0,
            total_observations=len(observed_indices),
            theta_hat=theta_hat,
            theta_star=bundle.theta_star,
            V_oracle=V_oracle,
            batch_index=0,
        )

        method_rng = _method_rng(spec.seed, str(spec.method["name"]))
        acquired = 0
        batch_index = 0
        selections: list[int] = []
        while acquired < budget:
            batch_index += 1
            current_batch_size = min(batch_size, budget - acquired)
            selected = _select_batch(
                bundle=bundle,
                family=family,
                spec=spec,
                theta_hat=theta_hat,
                V_plugin=V_plugin,
                V_oracle=V_oracle,
                available=available,
                batch_index=batch_index,
                batch_size=current_batch_size,
                method_rng=method_rng,
            )
            if not selected:
                break
            if not bundle.with_replacement:
                selected_unique = np.unique(np.asarray(selected, dtype=int))
                if len(selected_unique) != len(selected):
                    raise RuntimeError("Without-replacement batch selected a duplicate row")
                available = available[~np.isin(available, selected_unique)]

            new_y = _observe(
                bundle,
                family,
                selected,
                observation_offset=len(observed_indices),
            )
            selections.extend(selected)
            observed_indices.extend(selected)
            X_observed = np.vstack([X_observed, bundle.X[np.asarray(selected, dtype=int)]])
            y_array = np.concatenate([y_array, new_y])
            theta_hat = family.fit_ridge(
                X_observed,
                y_array,
                ridge=ridge,
                theta_init=theta_hat,
                max_iter=max_iter,
                tol=tolerance,
            )
            V_plugin = information_matrix(family, X_observed, theta_hat, ridge)
            V_oracle = information_matrix(family, X_observed, bundle.theta_star, ridge)
            acquired += len(selected)
            _record_metrics(
                rows,
                acquired=acquired,
                total_observations=len(observed_indices),
                theta_hat=theta_hat,
                theta_star=bundle.theta_star,
                V_oracle=V_oracle,
                batch_index=batch_index,
            )

        trajectory = pd.DataFrame(rows)
        for key, value in {
            "experiment": spec.experiment_name,
            "source": spec.source,
            "dataset": bundle.name,
            "data_id": spec.dataset.get("data_id"),
            "model": family.name,
            "protocol": spec.protocol.get("name"),
            "batch_size": batch_size,
            "method": spec.method.get("name"),
            "method_label": spec.method.get("label", spec.method.get("name")),
            "seed": spec.seed,
            "dimension": bundle.d,
            "theta_regime": spec.dataset.get("theta_regime", "weak"),
            "dataset_variant": stable_hash(spec.dataset, length=10),
            "setting_label": _setting_label(spec, bundle),
        }.items():
            trajectory[key] = value
        trajectory.to_csv(run_dir / "trajectory.csv", index=False)
        np.save(run_dir / "selected_indices.npy", np.asarray(selections, dtype=int))
        np.save(run_dir / "theta_star.npy", bundle.theta_star)

        metadata = {
            "run_spec": jsonable(asdict(spec)),
            "run_id": spec.run_id,
            "run_schema_version": RUN_SCHEMA_VERSION,
            "dataset_metadata": jsonable(bundle.metadata),
            "effective_budget": acquired,
            "initial_observations": len(init_indices),
            "final_total_observations": len(observed_indices),
            "final_l2_error": float(trajectory.iloc[-1]["l2_error"]),
            "final_lambda_min": float(trajectory.iloc[-1]["lambda_min"]),
            "environment": environment_manifest(),
            "git_commit": git_commit(project_root),
        }
        atomic_write_json(run_dir / "metadata.json", metadata)
        atomic_write_json(complete_path, {"status": "complete", "rows": len(trajectory)})
        failed_path = run_dir / "failed.json"
        if failed_path.exists():
            failed_path.unlink()
        return run_dir
    except Exception as exc:
        atomic_write_json(
            run_dir / "failed.json",
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "run_spec": jsonable(asdict(spec)),
            },
        )
        raise


def load_run_trajectories(results_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results_root.rglob("trajectory.csv")):
        try:
            frame = pd.read_csv(path)
            frame["run_dir"] = str(path.parent)
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No trajectory.csv files under {results_root}")
    return pd.concat(frames, ignore_index=True)
