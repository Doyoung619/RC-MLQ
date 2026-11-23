from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..glm import GLMFamily


@dataclass
class AcquisitionContext:
    X: np.ndarray
    candidate_ids: np.ndarray
    theta_hat: np.ndarray
    V: np.ndarray
    family: GLMFamily
    rng: np.random.Generator
    method: dict[str, Any]
    batch_position: int


def quadratic_forms(X: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk,ik->i", X, matrix, X, optimize=True)


def safe_inverse(V: np.ndarray) -> np.ndarray:
    try:
        inverse = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(V)
    return 0.5 * (inverse + inverse.T)


def stable_argmax(scores: np.ndarray, candidate_ids: np.ndarray, rng: np.random.Generator) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.shape[0] != candidate_ids.shape[0]:
        raise ValueError("Scores and candidate ids have incompatible shapes")
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise FloatingPointError("No finite acquisition score")
    best = np.nanmax(scores)
    tolerance = 1e-12 * max(1.0, abs(float(best)))
    ties = np.flatnonzero(finite & (scores >= best - tolerance))
    # Seeded random tie-breaking avoids a fixed coordinate artifact.
    return int(rng.choice(ties))
