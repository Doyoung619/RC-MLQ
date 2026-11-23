"""Integrated posterior EIG, retaining a log-domain predictive distribution."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .model import log_probabilities


def integrated_eig_from_loglik(logp, weights):
    logp = np.asarray(logp, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    with np.errstate(divide="ignore"):
        logw = np.log(weights)
    logbar = logsumexp(logw[:, None] + logp, axis=0)
    a = logp - logbar
    p = np.exp(logp)
    pbar = np.exp(logbar)
    contribution = p * a - (p - pbar)
    small = np.abs(a) < 1e-3
    x = a[small]
    # f(x)=exp(x)x-expm1(x), avoiding cancellation for small information.
    f = x*x * (0.5 + x * (1/3 + x * (1/8 + x * (1/30 + x / 144))))
    contribution[small] = np.broadcast_to(pbar, a.shape)[small] * f
    value = float(weights @ contribution.sum(axis=1))
    tolerance = 2e-13 * max(1., np.log(logp.shape[1]))
    if value < -tolerance or value > np.log(logp.shape[1]) + tolerance:
        raise FloatingPointError(f"EIG outside mathematical bounds: {value}")
    return value


def integrated_eig(theta, weights, q, beta):
    return integrated_eig_from_loglik(log_probabilities(theta, q, beta), weights)


def information_ratio(actual, reference):
    if not np.isfinite(reference) or reference <= 0:
        return float("nan"), "unresolved_nonpositive_reference"
    if reference < 1e-13:
        return float("nan"), "unresolved_reference_below_numerical_resolution"
    return float(actual / reference), "resolved"
