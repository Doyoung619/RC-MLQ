from __future__ import annotations

import numpy as np

from .glm import GLMFamily
from .utils import ensure_finite


def fisher_weights(family: GLMFamily, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return family.curvature(np.asarray(X, dtype=np.float64) @ np.asarray(theta, dtype=np.float64))


def information_matrix(
    family: GLMFamily,
    X: np.ndarray,
    theta: np.ndarray,
    ridge: float,
) -> np.ndarray:
    d = X.shape[1]
    if X.shape[0] == 0:
        return ridge * np.eye(d)
    weights = fisher_weights(family, X, theta)
    V = ridge * np.eye(d) + X.T @ (weights[:, None] * X)
    V = 0.5 * (V + V.T)
    ensure_finite("information matrix", V)
    return V


def rank_one_update(V: np.ndarray, x: np.ndarray, weight: float) -> np.ndarray:
    updated = V + float(weight) * np.outer(x, x)
    return 0.5 * (updated + updated.T)


def minimum_eigenvalue(V: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(V)[0])


def minimum_eigenspace(V: np.ndarray, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(V)
    threshold = values[0] + tolerance * max(1.0, abs(values[-1]))
    count = int(np.sum(values <= threshold))
    return values, vectors[:, :count]


def d_opt_initial_indices(
    X: np.ndarray,
    n_init: int,
    ridge: float,
    with_replacement: bool,
    candidate_limit: int | None,
    rng: np.random.Generator,
) -> list[int]:
    """Label-free greedy D-optimal warm start using unit curvature."""
    n, d = X.shape
    if not with_replacement and n_init > n:
        raise ValueError("n_init exceeds the without-replacement pool")
    V_inv = np.eye(d) / ridge
    available = np.arange(n, dtype=int)
    selected: list[int] = []
    for _ in range(n_init):
        if candidate_limit is not None and len(available) > candidate_limit:
            considered = rng.choice(available, size=candidate_limit, replace=False)
        else:
            considered = available
        Xc = X[considered]
        leverage = np.einsum("ij,jk,ik->i", Xc, V_inv, Xc, optimize=True)
        local = int(np.argmax(leverage))
        idx = int(considered[local])
        x = X[idx]
        denom = 1.0 + float(x @ V_inv @ x)
        vx = V_inv @ x
        V_inv = V_inv - np.outer(vx, vx) / max(denom, 1e-12)
        selected.append(idx)
        if not with_replacement:
            available = available[available != idx]
    return selected


def random_initial_indices(
    n: int, n_init: int, with_replacement: bool, rng: np.random.Generator
) -> list[int]:
    return rng.choice(n, size=n_init, replace=with_replacement).astype(int).tolist()


def condition_number(V: np.ndarray) -> float:
    values = np.linalg.eigvalsh(V)
    return float(values[-1] / max(values[0], 1e-12))
