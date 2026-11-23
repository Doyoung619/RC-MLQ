from __future__ import annotations

import numpy as np

from .base import AcquisitionContext, stable_argmax


def hard_scores(context: AcquisitionContext) -> np.ndarray:
    values, vectors = np.linalg.eigh(context.V)
    v = vectors[:, 0]
    weights = context.family.curvature(context.X @ context.theta_hat)
    return weights * np.square(context.X @ v)


def soft_temperature(context: AcquisitionContext) -> float:
    mode = str(context.method.get("tau_mode", "theory"))
    if mode == "fixed":
        tau = float(context.method.get("tau", 0.1))
    elif mode == "theory":
        batch_size = int(context.method.get("batch_size", 1))
        dimension = context.V.shape[0]
        radius = float(context.method.get("radius", 1.0))
        if context.family.name == "poisson":
            curvature = context.family.curvature(context.X @ context.theta_hat)
            row_norm_sq = np.einsum("ij,ij->i", context.X, context.X)
            R = float(np.max(curvature * row_norm_sq))
        else:
            R = context.family.max_curvature * radius * radius
        tau = R * np.sqrt(batch_size / max(8.0 * np.log(max(dimension, 2)), 1e-12))
        tau *= float(context.method.get("tau_multiplier", 1.0))
    else:
        raise ValueError(f"Unknown tau_mode: {mode}")
    return max(tau, 1e-8)


def soft_scores(context: AcquisitionContext) -> np.ndarray:
    values, vectors = np.linalg.eigh(context.V)
    tau = soft_temperature(context)
    logits = -(values - values[0]) / tau
    logits -= np.max(logits)
    spectral_weights = np.exp(np.clip(logits, -700.0, 0.0))
    spectral_weights /= np.sum(spectral_weights)
    projections = context.X @ vectors
    directional_energy = np.square(projections) @ spectral_weights
    curvature = context.family.curvature(context.X @ context.theta_hat)
    return curvature * directional_energy


def choose_hard(context: AcquisitionContext) -> int:
    return stable_argmax(hard_scores(context), context.candidate_ids, context.rng)


def choose_soft(context: AcquisitionContext) -> int:
    return stable_argmax(soft_scores(context), context.candidate_ids, context.rng)
