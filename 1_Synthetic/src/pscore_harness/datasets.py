from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .glm import GLMFamily
from .utils import atomic_write_json, ensure_finite, stable_hash


OPENML_PREPROCESS_VERSION = "2026-08-05.1"


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    theta_star: np.ndarray
    source: str
    with_replacement: bool
    metadata: dict[str, Any]
    common_randomness: np.ndarray
    initialization_X: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


def _row_normalize(X: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    scale = np.maximum(norms / radius, 1.0)
    return X / scale[:, None]


def _covariance_eigenvectors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = (X.T @ X) / max(1, X.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    return values, vectors


def _make_theta_star(
    X: np.ndarray,
    regime: str,
    rng: np.random.Generator,
    signal_std: float,
    theta_norm_cap: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values, vectors = _covariance_eigenvectors(X)
    d = X.shape[1]
    if regime == "weak":
        direction = vectors[:, 0]
    elif regime == "strong":
        direction = vectors[:, -1]
    elif regime == "random":
        direction = rng.normal(size=d)
        direction /= max(np.linalg.norm(direction), 1e-12)
    elif regime == "mixed":
        random_direction = rng.normal(size=d)
        random_direction /= max(np.linalg.norm(random_direction), 1e-12)
        direction = 0.8 * vectors[:, 0] + 0.2 * random_direction
        direction /= max(np.linalg.norm(direction), 1e-12)
    else:
        raise ValueError(f"Unknown theta regime: {regime}")

    projected = X @ direction
    current_std = float(np.std(projected))
    if current_std <= 1e-12:
        raise ValueError("Chosen theta direction has zero empirical signal variance")
    scale = signal_std / current_std
    scale = min(scale, theta_norm_cap)
    theta_star = scale * direction
    metadata = {
        "cov_min": float(values[0]),
        "cov_max": float(values[-1]),
        "condition_number": float(values[-1] / max(values[0], 1e-12)),
        "theta_norm": float(np.linalg.norm(theta_star)),
        "signal_std": float(np.std(X @ theta_star)),
    }
    return theta_star, metadata


def _sample_sphere_pool(
    rng: np.random.Generator,
    n_candidates: int,
    d: int,
    radius: float,
    anisotropy: float,
    weak_fraction: float,
    weak_rank: int,
) -> np.ndarray:
    # Draw an anisotropic pool, then project every row to the Euclidean ball.
    scales = np.power(np.arange(1, d + 1, dtype=np.float64), -anisotropy / 2.0)
    Z = rng.normal(size=(n_candidates, d)) * scales[None, :]

    # Ensure every coordinate and several weak-subspace mixtures are represented.
    basis = np.vstack([np.eye(d), -np.eye(d)]) * radius
    n_weak = int(max(0, round(weak_fraction * n_candidates)))
    if n_weak > 0:
        weak = np.zeros((n_weak, d), dtype=np.float64)
        weak[:, -weak_rank:] = rng.normal(size=(n_weak, weak_rank))
        weak /= np.maximum(np.linalg.norm(weak, axis=1, keepdims=True), 1e-12)
        weak *= radius
        Z[:n_weak] = weak
    Z = _row_normalize(Z, radius)
    X = np.vstack([basis, Z])
    return X


def _make_multiplicity_pool(
    rng: np.random.Generator,
    n_candidates: int,
    d: int,
    radius: float,
    weak_rank: int,
    duplicate_noise: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    per_direction = max(4, n_candidates // max(weak_rank, 1))
    for j in range(weak_rank):
        direction = np.zeros(d)
        direction[j] = radius
        for _ in range(per_direction):
            x = direction + duplicate_noise * rng.normal(size=d)
            x = radius * x / max(np.linalg.norm(x), 1e-12)
            rows.append(x)
    while len(rows) < n_candidates:
        x = rng.normal(size=d)
        x = radius * x / max(np.linalg.norm(x), 1e-12)
        rows.append(x)
    X = np.asarray(rows[:n_candidates], dtype=np.float64)
    return np.vstack([np.eye(d) * radius, -np.eye(d) * radius, X])


def make_synthetic_bundle(
    config: dict[str, Any], family: GLMFamily, seed: int
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    name = str(config.get("name", "anisotropic_ball"))
    requested_d = int(config.get("dimension", 20))
    d = int(config.get("representation_dimension", requested_d))
    if not 0 < requested_d <= d:
        raise ValueError("Need 0 < dimension <= representation_dimension")
    n_candidates = int(config.get("n_candidates", 2000))
    radius = float(config.get("radius", 1.0))
    if name in {"anisotropic_ball", "curvature_alignment"}:
        X = _sample_sphere_pool(
            rng=rng,
            n_candidates=n_candidates,
            d=d,
            radius=radius,
            anisotropy=float(config.get("anisotropy", 1.5)),
            weak_fraction=float(config.get("weak_fraction", 0.10)),
            weak_rank=int(config.get("weak_rank", max(2, d // 5))),
        )
    elif name == "multiplicity":
        X = _make_multiplicity_pool(
            rng=rng,
            n_candidates=n_candidates,
            d=d,
            radius=radius,
            weak_rank=int(config.get("weak_rank", 4)),
            duplicate_noise=float(config.get("duplicate_noise", 0.02)),
        )
    elif name == "orthogonal_rays":
        amplitudes = np.linspace(0.2 * radius, radius, int(config.get("ray_levels", 5)))
        rows = []
        for i in range(d):
            for amplitude in amplitudes:
                x = np.zeros(d)
                x[i] = amplitude
                rows.extend([x.copy(), -x.copy()])
        X = np.asarray(rows, dtype=np.float64)
    else:
        raise ValueError(f"Unknown synthetic dataset generator: {name}")

    initialization_X = X.copy() if "representation_dimension" in config else None
    X = X[:, :requested_d].copy()
    d = requested_d
    theta_star, theta_meta = _make_theta_star(
        X=X,
        regime=str(config.get("theta_regime", "weak")),
        rng=rng,
        signal_std=float(config.get("signal_std", 1.0)),
        theta_norm_cap=float(config.get("theta_norm_cap", 8.0)),
    )

    max_observations = int(config.get("max_common_randomness", 100000))
    if family.name == "gaussian":
        common = rng.normal(size=max_observations)
    else:
        common = rng.uniform(size=max_observations)
    metadata: dict[str, Any] = {
        "generator": name,
        "radius": radius,
        "n_candidates": int(X.shape[0]),
        "dimension": d,
        "representation_dimension": int(config.get("representation_dimension", d)),
        "theta_regime": str(config.get("theta_regime", "weak")),
        **theta_meta,
    }
    ensure_finite("synthetic X", X)
    return DatasetBundle(
        name=name,
        X=X,
        theta_star=theta_star,
        source="synthetic",
        with_replacement=bool(config.get("with_replacement", True)),
        metadata=metadata,
        common_randomness=common,
        initialization_X=initialization_X,
    )


def fetch_and_preprocess_openml(
    config: dict[str, Any], cache_root: Path, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    data_id = int(config["data_id"])
    # Keep the unlabeled feature geometry fixed across response seeds. The run
    # seed controls theta/noise, whereas preprocess_seed controls row sampling,
    # one-hot vocabulary, and PCA/SVD. This avoids conflating acquisition
    # variance with a changing covariate matrix.
    preprocess_seed = int(config.get("preprocess_seed", 0))
    preprocessing_signature = {
        "preprocess_code_version": OPENML_PREPROCESS_VERSION,
        "data_id": data_id,
        "preprocess_seed": preprocess_seed,
        "dimension": int(config.get("dimension", 20)),
        "max_rows": int(config.get("max_rows", 10000)),
        "radius": float(config.get("radius", 1.0)),
        "min_category_frequency": config.get("min_category_frequency", 2),
        "max_categories_per_feature": config.get("max_categories_per_feature", 32),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = stable_hash(preprocessing_signature, length=16)
    processed_path = cache_root / f"openml_{data_id}_{cache_key}.npz"
    metadata_path = processed_path.with_suffix(".json")
    if processed_path.exists() and metadata_path.exists():
        with np.load(processed_path) as archive:
            X_cached = np.asarray(archive["X"], dtype=np.float64)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata_cached = json.load(handle)
        return X_cached, metadata_cached

    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        cache=True,
        data_home=str(cache_root),
        parser="auto",
    )
    X_frame = bunch.data
    if not isinstance(X_frame, pd.DataFrame):
        X_frame = pd.DataFrame(X_frame, columns=getattr(bunch, "feature_names", None))

    max_rows = int(config.get("max_rows", 10000))
    if len(X_frame) > max_rows:
        X_frame = X_frame.sample(n=max_rows, random_state=preprocess_seed, replace=False)
    X_frame = X_frame.reset_index(drop=True)

    categorical_columns = [
        c
        for c in X_frame.columns
        if str(X_frame[c].dtype) in {"category", "object", "string", "bool"}
    ]
    numeric_columns = [c for c in X_frame.columns if c not in categorical_columns]

    transformers = []
    if numeric_columns:
        numeric_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=config.get("min_category_frequency", 2),
                        max_categories=config.get("max_categories_per_feature", 32),
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_columns))
    if not transformers:
        raise ValueError(f"OpenML dataset {data_id} has no usable columns")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    encoded = preprocessor.fit_transform(X_frame)
    target_dimension = int(config.get("dimension", 20))
    max_dimension = min(target_dimension, encoded.shape[0] - 1, encoded.shape[1])
    if max_dimension < 2:
        raise ValueError(f"OpenML dataset {data_id} has effective dimension {max_dimension}")

    if sparse.issparse(encoded):
        reducer = TruncatedSVD(n_components=max_dimension, random_state=preprocess_seed)
        X = reducer.fit_transform(encoded)
        explained = float(np.sum(reducer.explained_variance_ratio_))
    else:
        dense = np.asarray(encoded, dtype=np.float64)
        reducer = PCA(n_components=max_dimension, random_state=preprocess_seed)
        X = reducer.fit_transform(dense)
        explained = float(np.sum(reducer.explained_variance_ratio_))

    # Geometry is available before labels. Standardizing the reduced pool and
    # bounding every row makes lambda_min comparable across OpenML datasets.
    X = np.asarray(X, dtype=np.float64)
    X -= np.mean(X, axis=0, keepdims=True)
    column_scale = np.std(X, axis=0, keepdims=True)
    X /= np.maximum(column_scale, 1e-8)
    radius = float(config.get("radius", 1.0))
    X = _row_normalize(X, radius)

    # Drop rows that became numerically zero after preprocessing.
    keep = np.linalg.norm(X, axis=1) > 1e-10
    X = X[keep]
    ensure_finite("OpenML X", X)

    details = getattr(bunch, "details", {}) or {}
    metadata: dict[str, Any] = {
        "data_id": data_id,
        "openml_name": details.get("name", config.get("name", str(data_id))),
        "openml_version": details.get("version"),
        "n_rows_raw": int(len(X_frame)),
        "n_rows_processed": int(X.shape[0]),
        "n_features_raw": int(X_frame.shape[1]),
        "dimension": int(X.shape[1]),
        "categorical_columns": int(len(categorical_columns)),
        "numeric_columns": int(len(numeric_columns)),
        "explained_variance_ratio": explained,
        "radius": radius,
        "preprocess_seed": preprocess_seed,
        "preprocessing_signature": preprocessing_signature,
        "cache_key": cache_key,
    }
    temporary_npz = processed_path.with_name(
        f"{processed_path.name}.{os.getpid()}.tmp"
    )
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, X=X)
    os.replace(temporary_npz, processed_path)
    atomic_write_json(metadata_path, metadata)
    return X, metadata


def make_openml_bundle(
    config: dict[str, Any], family: GLMFamily, seed: int, cache_root: Path
) -> DatasetBundle:
    X, metadata = fetch_and_preprocess_openml(config, cache_root, seed)
    rng = np.random.default_rng(seed)
    theta_star, theta_meta = _make_theta_star(
        X=X,
        regime=str(config.get("theta_regime", "weak")),
        rng=rng,
        signal_std=float(config.get("signal_std", 1.0)),
        theta_norm_cap=float(config.get("theta_norm_cap", 8.0)),
    )
    metadata.update(theta_meta)
    metadata["theta_regime"] = str(config.get("theta_regime", "weak"))
    if family.name == "poisson":
        mu = family.mean(X @ theta_star)
        metadata.update(
            {
                "poisson_intercept": family.poisson_intercept,
                "poisson_mean_count": float(np.mean(mu)),
                "poisson_median_count": float(np.median(mu)),
                "poisson_q99_count": float(np.quantile(mu, 0.99)),
                "poisson_max_count": float(np.max(mu)),
            }
        )

    if family.name == "gaussian":
        common = rng.normal(size=X.shape[0])
    else:
        common = rng.uniform(size=X.shape[0])
    name = str(config.get("name", metadata.get("openml_name", config["data_id"])))
    return DatasetBundle(
        name=name,
        X=X,
        theta_star=theta_star,
        source="openml",
        with_replacement=False,
        metadata=metadata,
        common_randomness=common,
    )


def make_dataset_bundle(
    source: str,
    config: dict[str, Any],
    family: GLMFamily,
    seed: int,
    cache_root: Path,
) -> DatasetBundle:
    if source == "synthetic":
        return make_synthetic_bundle(config, family, seed)
    if source == "openml":
        return make_openml_bundle(config, family, seed, cache_root)
    raise ValueError(f"Unknown source: {source}")
