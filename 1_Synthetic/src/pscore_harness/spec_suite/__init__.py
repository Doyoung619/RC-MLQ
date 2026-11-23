"""The 15 acquisition algorithms in ALGORITHM_IMPLEMENTATION_SPEC.md.

The numerical kernels are preserved from the supplied source snapshot.
This interface keeps response generation and evaluation outside selectors.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

from .family_selectors import FAMILY_METHODS, select_family_batch
from .selectors import curvature, select_batch as _select_common

SUITE_VERSION = "algorithm_spec_20261005_v1"
COMMON_METHODS = ("fb_opt2", "a_opt", "d_opt", "e_greedy", "e_sdp",
                  "bait_glm", "ucb_glm", "ts_glm", "random")
GAUSSIAN_METHODS = COMMON_METHODS + ("b_emcm", "lcmd", "regvol")
BERNOULLI_METHODS = COMMON_METHODS + ("entropy", "badge", "firal")
ALL_METHODS = COMMON_METHODS + ("b_emcm", "lcmd", "regvol", "entropy", "badge", "firal")
METHOD_LABELS = dict(zip(ALL_METHODS, (
    "Ours", "A-opt", "D-opt", "E-greedy", "E-SDP", "BAIT", "UCB", "TS",
    "Random", "B-EMCM", "LCMD", "RegVol", "Entropy", "BADGE", "FIRAL")))
ALIASES = {"ours": "fb_opt2", "bait": "bait_glm", "ucb": "ucb_glm",
           "ts": "ts_glm", "thompson": "ts_glm", "e_opt": "e_greedy"}


def normalize_family(family: str) -> str:
    if family in ("bernoulli", "logistic"):
        return "bernoulli"
    if family in ("gaussian", "normal"):
        return "gaussian"
    raise ValueError("The specified algorithm suite supports Gaussian and Bernoulli only")


def canonical_method(method: str) -> str:
    method = ALIASES.get(method, method)
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown spec-suite method: {method}")
    return method


def methods_for_family(family: str) -> tuple[str, ...]:
    return GAUSSIAN_METHODS if normalize_family(family) == "gaussian" else BERNOULLI_METHODS


def default_methods() -> list[dict[str, Any]]:
    result = []
    for name in ALL_METHODS:
        item: dict[str, Any] = {"name": name, "label": METHOD_LABELS[name]}
        if name in ("b_emcm", "lcmd", "regvol"):
            item["allow_models"] = ["gaussian"]
        elif name in ("entropy", "badge", "firal"):
            item["allow_models"] = ["logistic", "bernoulli"]
        result.append(item)
    return result


def selector_seed(dataset: str, d: int, seed: int, method: str) -> int:
    text = "|".join(map(str, ("selector", dataset, d, seed, canonical_method(method))))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def information_matrix(X: np.ndarray, theta: np.ndarray, family: str, rho: float = 1.) -> np.ndarray:
    """Exact evaluation curvature: no acquisition floor or predictor clipping."""
    X = np.asarray(X, dtype=np.float64)
    w = curvature(X @ theta, normalize_family(family))
    V = rho * np.eye(X.shape[1]) + X.T @ (w[:, None] * X)
    return (V + V.T) * .5


def select_batch(method: str, X_pool: np.ndarray, observed_ids: np.ndarray,
                 candidate_ids: np.ndarray, theta_hat: np.ndarray, family: str,
                 m: int, rho: float, rng: np.random.Generator, dataset: str,
                 seed: int, round_index: int, observed_y: np.ndarray | None = None,
                 planned_rounds: int | None = None,
                 settings: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Select distinct global IDs with frozen theta and no evaluation inputs."""
    method = canonical_method(method)
    family = normalize_family(family)
    if method not in methods_for_family(family):
        raise ValueError(f"{method} is not applicable to {family}")
    X = np.asarray(X_pool, dtype=np.float64)
    theta = np.asarray(theta_hat, dtype=np.float64)
    observed = np.asarray(observed_ids, dtype=np.int64)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("Expected a finite, nonempty float64 query matrix")
    if theta.shape != (X.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("Invalid fitted parameter")
    if observed.ndim != 1 or candidates.ndim != 1 or len(np.unique(observed)) != len(observed):
        raise ValueError("Observed and candidate IDs must be one-dimensional and unique")
    if np.any(observed < 0) or np.any(observed >= len(X)) or np.any(candidates < 0) or np.any(candidates >= len(X)):
        raise ValueError("Global row IDs outside query pool")
    if int(m) != m or not 0 < m <= len(candidates) or rho != 1. or round_index < 1:
        raise ValueError("Require positive integer m <= K, ridge rho=1 and round >= 1")
    if len(np.unique(candidates)) != len(candidates) or np.intersect1d(observed, candidates).size:
        raise ValueError("Candidates must be unique, unqueried global IDs")
    if observed_y is not None and method != "b_emcm":
        raise ValueError("Only B-EMCM may receive observed labels")
    config = copy.deepcopy(settings or {})
    forbidden = {"theta_ref", "theta_star", "V_ref", "V_oracle", "y_pool", "evaluation", "metrics"}
    if forbidden.intersection(config):
        raise ValueError("Evaluation and hidden-label inputs are forbidden in selector settings")
    config.update(dataset=str(dataset), seed=int(seed), acquisition_floor=1e-12 if family == "bernoulli" else 0.)
    if method == "fb_opt2":
        if planned_rounds is None or int(planned_rounds) != planned_rounds or planned_rounds < 1:
            raise ValueError("Ours requires a fixed, positive planned horizon")
        config.update(fb_opt2_Tmax=int(planned_rounds), temperature_multiplier=1.)
    if method in FAMILY_METHODS:
        selected, audit = select_family_batch(method, X, observed, candidates, theta, family,
                                             int(m), rho, rng, config, round_index, observed_y)
    else:
        selected, audit = _select_common(method, X, observed, candidates, theta, family,
                                         int(m), rho, rng, config, round_index)
    audit.update(suite_version=SUITE_VERSION, display_name=METHOD_LABELS[method],
                 method_version="fb_opt2_v1_range_anchored" if method == "fb_opt2" else audit.get("method_version", SUITE_VERSION),
                 frozen_theta=theta.tolist(), uses_theta_ref=False, uses_future_labels=False)
    if method == "badge":
        from scipy.special import expit
        p = expit(X[candidates] @ theta)
        G = (p - (p >= .5))[:, None] * X[candidates]
        audit["gradient_embedding_squared_norms"] = np.einsum("ij,ij->i", G, G).tolist()
    return np.asarray(selected, dtype=np.int64), audit
