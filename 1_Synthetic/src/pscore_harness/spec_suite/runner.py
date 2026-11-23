"""Synthetic execution of the frozen 15-method specification.

Synthetic truth is used exclusively for generating responses and evaluation.
Each completed batch is committed atomically and can be replayed or resumed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import RUN_SCHEMA_VERSION, RunSpec
from ..datasets import make_dataset_bundle
from ..glm import make_family
from ..utils import atomic_write_json, environment_manifest, jsonable, stable_hash
from . import (SUITE_VERSION, METHOD_LABELS, canonical_method, information_matrix,
               methods_for_family, normalize_family, select_batch, selector_seed)
from .numerics import fit_glm
from .selectors import shared_candidates, shared_initial_ids


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    names = ("__init__.py", "selectors.py", "family_selectors.py", "numerics.py", "runner.py")
    return {name: _hash(Path(__file__).parent / name) for name in names}


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def _resolve(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def _read_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    data = path.with_suffix(".npz")
    if not data.exists() or _hash(data) != record["array_sha256"]:
        raise RuntimeError(f"Missing or corrupted checkpoint: {data}")
    with np.load(data, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError(f"Nonfinite checkpoint: {data}")
    return record, arrays


def validate_completed_run(run_dir: Path) -> None:
    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["implementation_sha256"] != _implementation_hashes():
        raise RuntimeError("Completed run was produced by a different scientific implementation")
    for name, fingerprint in complete["file_sha256"].items():
        path = run_dir / name
        if not path.exists() or _hash(path) != fingerprint:
            raise RuntimeError(f"Missing or corrupted completed output: {path}")
    commits = sorted((run_dir / "checkpoints").glob("round_*.json"))
    if len(commits) != complete["rows"]:
        raise RuntimeError("Completed run has a missing/extra checkpoint commit")
    for t, path in enumerate(commits):
        if path.name != f"round_{t:06d}.json":
            raise RuntimeError("Completed checkpoints are not contiguous")
        record, _ = _read_checkpoint(path)
        if record["round"] != t or record["suite_version"] != SUITE_VERSION:
            raise RuntimeError("Checkpoint version/round mismatch")


def run_spec_suite(spec: RunSpec, project_root: Path, overwrite: bool = False) -> Path:
    run_dir = _resolve(project_root, spec.output_root) / spec.experiment_name / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    try:
        if overwrite:
            # Only this run's recognized commits are reset; user files remain.
            for path in checkpoints.glob("round_*"):
                if path.suffix in (".json", ".npz"):
                    path.unlink()
            (run_dir / "complete.json").unlink(missing_ok=True)
        elif (run_dir / "complete.json").exists():
            validate_completed_run(run_dir)
            (run_dir / "failed.json").unlink(missing_ok=True)
            return run_dir
        family = normalize_family(str(spec.model["family"]))
        method = canonical_method(str(spec.method["name"]))
        if method not in methods_for_family(family):
            raise ValueError(f"{method} is not applicable to {family}")
        if float(spec.estimator.get("ridge", 1.)) != 1.:
            raise ValueError("The specified suite uses ridge rho=1, including any intercept")
        if family == "gaussian" and float(spec.model.get("noise_std", 1.)) != 1.:
            raise ValueError("The specified Gaussian working model has variance 1")
        m = int(spec.protocol.get("batch_size", 10))
        budgets = spec.protocol.get("budget_by_family", {"gaussian": 400, "bernoulli": 4000})
        budget = int(spec.protocol.get("budget", budgets[family]))
        n0 = int(spec.protocol.get("n_init", 80))
        K = int(spec.protocol.get("candidate_limit", 256 if family == "gaussian" else 512))
        if m <= 0 or budget <= 0 or budget % m or K < m or not 0 < n0 <= 2000:
            raise ValueError("Require positive full batches, K >= m and 0 < n_init <= 2000")
        T = budget // m
        glm = make_family({**spec.model, "family": "logistic" if family == "bernoulli" else "gaussian"})
        bundle = make_dataset_bundle(spec.source, spec.dataset, glm, spec.seed,
                                     _resolve(project_root, spec.cache_root))
        if bundle.with_replacement:
            raise ValueError("The spec suite requires a fixed pool without replacement")
        if n0 + budget > bundle.n:
            raise ValueError("Pool too small for initialization plus full planned budget")
        if len(bundle.common_randomness) < bundle.n:
            raise ValueError("Need one synthetic response random variate per global row")
        X = np.asarray(bundle.X, dtype=np.float64)
        dataset = str(spec.dataset.get("selector_dataset", bundle.name))
        init_X = bundle.initialization_X if bundle.initialization_X is not None else X
        init_ids = shared_initial_ids(init_X, dataset, spec.seed, n0=n0).astype(np.int64)
        # Evaluation truth is deliberately never passed into select_batch.
        eta_true = X @ bundle.theta_star
        if family == "gaussian":
            y_pool = eta_true + bundle.common_randomness[:bundle.n]
        else:
            from scipy.special import expit
            y_pool = (bundle.common_randomness[:bundle.n] < expit(eta_true)).astype(np.float64)
        rng_seed = selector_seed(dataset, bundle.d, spec.seed, method)
        rng = np.random.default_rng(rng_seed)
        pool_hash = hashlib.sha256(X.tobytes() + bundle.theta_star.tobytes()
                                   + y_pool.tobytes() + init_ids.tobytes()).hexdigest()
        implementation = _implementation_hashes()
        identity = stable_hash({"spec": asdict(spec), "pool": pool_hash,
                                "implementation": implementation}, length=64)
        settings = dict(spec.method.get("settings", {}))
        observed = init_ids.copy()
        theta = None
        records: list[dict[str, Any]] = []
        next_round = 0
        for path in sorted(checkpoints.glob("round_*.json")):
            if path.name != f"round_{next_round:06d}.json":
                raise RuntimeError("Committed checkpoints have a gap; cannot resume")
            record, arrays = _read_checkpoint(path)
            if record["identity"] != identity or record["round"] != next_round:
                raise RuntimeError("Checkpoint belongs to a different configuration/implementation")
            expected = n0 + next_round * m
            observed = arrays["observed_ids"].astype(np.int64)
            if len(observed) != expected or len(np.unique(observed)) != expected:
                raise RuntimeError("Checkpoint violates cumulative unique observation budget")
            if not np.array_equal(observed[:n0], init_ids) or not np.array_equal(arrays["observed_y"], y_pool[observed]):
                raise RuntimeError("Checkpoint observation prefix does not match generated data")
            theta = arrays["theta_hat"]
            rng.bit_generator.state = record["rng_state_after"]
            records.append(record)
            next_round += 1
        if next_round > T + 1:
            raise RuntimeError("Checkpoint exceeds planned horizon")
        for t in range(next_round, T + 1):
            audit = None
            if t:
                candidates = shared_candidates(bundle.n, observed, dataset, spec.seed, t, K)
                selected, audit = select_batch(method, X, observed, candidates, theta, family,
                    m, 1., rng, dataset, spec.seed, t,
                    observed_y=y_pool[observed].copy() if method == "b_emcm" else None,
                    planned_rounds=T, settings=settings)
                observed = np.concatenate((observed, selected))
            theta, fit = fit_glm(X[observed], y_pool[observed], family, rho=1., theta0=theta,
                max_iter=spec.estimator.get("max_iter", 50),
                tolerance=spec.estimator.get("tolerance", 1e-8))
            if not fit["fit_converged"]:
                raise RuntimeError(f"Nonconverged {family} fit at batch {t}: {fit['fit_grad_relative']}")
            V_true = information_matrix(X[observed], bundle.theta_star, family)
            V_hat = information_matrix(X[observed], theta, family)
            error = theta - bundle.theta_star
            evaluation = {"acquired": t * m, "total_observations": len(observed),
                "batch_index": t, "l2_error": float(np.linalg.norm(error)),
                "squared_error": float(error @ error),
                "lambda_min": float(np.linalg.eigvalsh(V_true)[0]),
                "plugin_lambda_min": float(np.linalg.eigvalsh(V_hat)[0])}
            data_path = checkpoints / f"round_{t:06d}.npz"
            _atomic_npz(data_path, observed_ids=observed, observed_y=y_pool[observed],
                theta_hat=theta, signed_error=error, V_true=V_true, V_plugin=V_hat,
                eigenvalues_true=np.linalg.eigvalsh(V_true), eigenvalues_plugin=np.linalg.eigvalsh(V_hat))
            record = {"round": t, "identity": identity, "suite_version": SUITE_VERSION,
                "array_sha256": _hash(data_path), "evaluation": evaluation, "fit": fit,
                "acquisition": audit, "rng_state_after": rng.bit_generator.state}
            atomic_write_json(data_path.with_suffix(".json"), jsonable(record))
            records.append(record)
        trajectory = pd.DataFrame([r["evaluation"] for r in records])
        fields = {"experiment": spec.experiment_name, "source": spec.source, "dataset": bundle.name,
            "data_id": spec.dataset.get("data_id"), "model": glm.name,
            "protocol": spec.protocol.get("name", "batch"), "batch_size": m,
            "method": method, "method_label": METHOD_LABELS[method], "seed": spec.seed,
            "dimension": bundle.d, "theta_regime": spec.dataset.get("theta_regime", "weak"),
            "dataset_variant": stable_hash(spec.dataset, length=10),
            "setting_label": f"d={bundle.d}, suite={SUITE_VERSION}, theta={spec.dataset.get('theta_regime', 'weak')}",
            "algorithm_suite": SUITE_VERSION}
        for key, value in fields.items():
            trajectory[key] = value
        tmp = run_dir / "trajectory.csv.tmp"
        trajectory.to_csv(tmp, index=False)
        os.replace(tmp, run_dir / "trajectory.csv")
        np.save(run_dir / "selected_indices.npy", observed[n0:])
        np.save(run_dir / "initial_indices.npy", init_ids)
        np.save(run_dir / "theta_star.npy", bundle.theta_star)
        atomic_write_json(run_dir / "metadata.json", {"run_spec": asdict(spec), "run_id": spec.run_id,
            "run_schema_version": RUN_SCHEMA_VERSION, "suite_version": SUITE_VERSION,
            "dataset_metadata": bundle.metadata, "selector_seed": rng_seed,
            "planned_rounds": T, "effective_budget": budget, "initial_observations": n0,
            "final_total_observations": len(observed), "pool_sha256": pool_hash,
            "implementation_sha256": implementation,
            "environment": {**environment_manifest(), **{name: importlib.metadata.version(name)
                for name in ("cvxpy", "scs", "clarabel", "threadpoolctl")}},
            "synthetic_adaptation": "Known synthetic theta_star and generated responses; original query geometry retained; no added intercept; no parameter box; without replacement."})
        names = ("trajectory.csv", "selected_indices.npy", "initial_indices.npy", "theta_star.npy", "metadata.json")
        fingerprints = {name: _hash(run_dir / name) for name in names}
        fingerprints.update({p.relative_to(run_dir).as_posix(): _hash(p) for p in checkpoints.glob("round_*.json")})
        atomic_write_json(run_dir / "complete.json", {"status": "complete", "rows": T + 1,
            "suite_version": SUITE_VERSION, "identity": identity,
            "file_sha256": fingerprints})
        (run_dir / "failed.json").unlink(missing_ok=True)
        return run_dir
    except Exception as exc:
        atomic_write_json(run_dir / "failed.json", {"status": "failed", "error": repr(exc),
            "traceback": traceback.format_exc(), "run_spec": jsonable(asdict(spec))})
        raise
