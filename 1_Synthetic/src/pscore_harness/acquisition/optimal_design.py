from __future__ import annotations

import numpy as np

from ..information import rank_one_update
from .base import AcquisitionContext, quadratic_forms, safe_inverse, stable_argmax
from .proposed import hard_scores


def a_opt_scores(context: AcquisitionContext) -> np.ndarray:
    """Exact rank-one reduction of tr(V^{-1}). Larger is better."""
    V_inv = safe_inverse(context.V)
    XV = context.X @ V_inv
    leverage = np.einsum("ij,ij->i", XV, context.X)
    squared_inverse_norm = np.einsum("ij,ij->i", XV, XV)
    curvature = context.family.curvature(context.X @ context.theta_hat)
    return curvature * squared_inverse_norm / np.maximum(1.0 + curvature * leverage, 1e-12)


def d_opt_scores(context: AcquisitionContext) -> np.ndarray:
    """Exact rank-one log-determinant gain."""
    V_inv = safe_inverse(context.V)
    leverage = quadratic_forms(context.X, V_inv)
    curvature = context.family.curvature(context.X @ context.theta_hat)
    return np.log1p(np.maximum(curvature * leverage, 0.0))


def e_opt_scores(context: AcquisitionContext) -> np.ndarray:
    """Exact one-step minimum eigenvalue on a controlled shortlist."""
    n = context.X.shape[0]
    exact_limit = int(context.method.get("eopt_exact_limit", 512))
    shortlist_size = int(context.method.get("eopt_shortlist", 256))
    curvature = context.family.curvature(context.X @ context.theta_hat)

    if n <= exact_limit:
        shortlist = np.arange(n)
    else:
        first_order = hard_scores(context)
        top_count = min(shortlist_size, n)
        shortlist = np.argpartition(first_order, -top_count)[-top_count:]
        random_count = min(int(context.method.get("eopt_random_shortlist", 32)), n)
        random_extra = context.rng.choice(n, size=random_count, replace=False)
        shortlist = np.unique(np.concatenate([shortlist, random_extra]))

    scores = np.full(n, -np.inf, dtype=np.float64)
    for local in shortlist:
        updated = rank_one_update(context.V, context.X[local], float(curvature[local]))
        scores[local] = float(np.linalg.eigvalsh(updated)[0])
    return scores


def g_opt_scores(context: AcquisitionContext) -> np.ndarray:
    V_inv = safe_inverse(context.V)
    return quadratic_forms(context.X, V_inv)


def choose_a_opt(context: AcquisitionContext) -> int:
    return stable_argmax(a_opt_scores(context), context.candidate_ids, context.rng)


def choose_d_opt(context: AcquisitionContext) -> int:
    return stable_argmax(d_opt_scores(context), context.candidate_ids, context.rng)


def choose_e_opt(context: AcquisitionContext) -> int:
    return stable_argmax(e_opt_scores(context), context.candidate_ids, context.rng)


def choose_g_opt(context: AcquisitionContext) -> int:
    return stable_argmax(g_opt_scores(context), context.candidate_ids, context.rng)
