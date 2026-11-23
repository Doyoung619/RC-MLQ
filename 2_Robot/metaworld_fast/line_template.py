"""Frozen-posterior line template shared by RC, PQ, QI and EIG references."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from .model import principal_direction

SCALE_GRID = np.array([.125, .2, .32, .5, .8, 1.25, 2.], dtype=np.float64)


@dataclass
class LineTemplate:
    mean: np.ndarray
    covariance: np.ndarray
    direction: np.ndarray
    sigma: float
    median: float
    quantiles: np.ndarray
    b: np.ndarray
    queries: np.ndarray
    fisher_scores: np.ndarray
    rc_index: int


def make_template(posterior, M, scale_grid=SCALE_GRID):
    mean, covariance = posterior.moments()
    direction, sigma = principal_direction(covariance, posterior.factorized)
    levels = np.r_[.5, (np.arange(M) + .5) / M]
    vals = posterior.projected_quantiles(direction, levels) - direction @ mean
    median, quantiles = float(vals[0]), vals[1:]
    b = (quantiles - median) / sigma
    if np.ptp(b) <= 0:
        raise FloatingPointError("Repeated posterior quantiles: zero empirical template width")
    r = np.asarray(scale_grid, dtype=np.float64)
    omega = softmax(-posterior.beta * r[:, None]**2 * b[None, :]**2, axis=1)
    bmean = omega @ b
    variance = np.sum(omega * (b[None, :] - bmean[:, None])**2, axis=1)
    scores = 4 * posterior.beta**2 * r**2 * variance
    queries = mean[None, None, :] + (median + r[:, None] * b[None, :])[:, :, None] * direction
    return LineTemplate(mean, covariance, direction, sigma, median, quantiles, b, queries, scores, int(np.argmax(scores)))
