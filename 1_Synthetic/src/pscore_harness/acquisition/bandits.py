from __future__ import annotations

import numpy as np

from .base import AcquisitionContext, quadratic_forms, safe_inverse, stable_argmax


def _confidence_scale(context: AcquisitionContext) -> float:
    return float(context.method.get("beta", 1.0))


def ucb_scores(context: AcquisitionContext) -> np.ndarray:
    V_inv = safe_inverse(context.V)
    uncertainty = np.sqrt(np.maximum(quadratic_forms(context.X, V_inv), 0.0))
    mean = context.family.mean(context.X @ context.theta_hat)
    return mean + _confidence_scale(context) * uncertainty


def thompson_scores(context: AcquisitionContext) -> np.ndarray:
    V_inv = safe_inverse(context.V)
    scale = float(context.method.get("posterior_scale", 1.0))
    values, vectors = np.linalg.eigh(V_inv)
    values = np.maximum(values, 0.0)
    sample = context.theta_hat + scale * vectors @ (np.sqrt(values) * context.rng.normal(size=len(values)))
    return context.family.mean(context.X @ sample)


def epsilon_greedy_scores(context: AcquisitionContext) -> np.ndarray:
    epsilon = float(context.method.get("epsilon", 0.1))
    if context.rng.uniform() < epsilon:
        return context.rng.uniform(size=context.X.shape[0])
    return context.family.mean(context.X @ context.theta_hat)


def lingape_scores(context: AcquisitionContext) -> np.ndarray:
    """LinGapE sampling rule for Gaussian linear arms.

    The current estimated best arm is compared with the challenger having the
    largest optimistic gap. The query maximizes the one-step reduction of the
    variance of that contrast. This baseline is enabled only for Gaussian runs.
    """
    if context.family.name != "gaussian":
        raise ValueError("LinGapE is implemented only for the Gaussian linear model")
    V_inv = safe_inverse(context.V)
    predicted = context.X @ context.theta_hat
    best_local = int(np.argmax(predicted))
    best_x = context.X[best_local]
    contrasts = context.X - best_x[None, :]
    contrast_var = np.sqrt(np.maximum(quadratic_forms(contrasts, V_inv), 0.0))
    optimistic_gap = predicted - predicted[best_local] + _confidence_scale(context) * contrast_var
    optimistic_gap[best_local] = -np.inf
    challenger = int(np.argmax(optimistic_gap))
    contrast = best_x - context.X[challenger]

    Vc = V_inv @ contrast
    Vx = context.X @ V_inv
    numerator = np.square(context.X @ Vc)
    leverage = np.einsum("ij,ij->i", Vx, context.X)
    return numerator / np.maximum(1.0 + leverage, 1e-12)


def choose_ucb(context: AcquisitionContext) -> int:
    return stable_argmax(ucb_scores(context), context.candidate_ids, context.rng)


def choose_thompson(context: AcquisitionContext) -> int:
    return stable_argmax(thompson_scores(context), context.candidate_ids, context.rng)


def choose_epsilon_greedy(context: AcquisitionContext) -> int:
    return stable_argmax(epsilon_greedy_scores(context), context.candidate_ids, context.rng)


def choose_lingape(context: AcquisitionContext) -> int:
    return stable_argmax(lingape_scores(context), context.candidate_ids, context.rng)
