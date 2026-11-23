"""The specified squared-distance multinomial logit, in float64."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.special import logsumexp


def stable_seed(root: int, setting: str, interaction: int, method: str | None, purpose: str) -> int:
    data = json.dumps([int(root), setting, int(interaction), method, purpose], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], "little")


def log_probabilities(theta, q, beta):
    theta, q = np.asarray(theta, dtype=np.float64), np.asarray(q, dtype=np.float64)
    # Centering improves translation invariance without removing the candidate norm term.
    origin = q[0]
    qc = q - origin
    logits = 2 * float(beta) * ((theta - origin) @ qc.T) - float(beta) * np.sum(qc * qc, axis=1)
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def probabilities(theta, q, beta):
    return np.exp(log_probabilities(theta, q, beta))


def coordinate_of_query(q, tolerance=1e-12):
    q = np.asarray(q, dtype=np.float64)
    span = np.ptp(q, axis=0)
    varying = np.flatnonzero(span > tolerance * max(1., float(np.max(np.abs(q)))))
    if len(varying) > 1:
        raise ValueError("Actual displayed query is not coordinate-factorized")
    return int(varying[0]) if len(varying) else 0


def principal_direction(covariance, factorized=False, eigenspace_rtol=1e-10):
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) / 2
    if factorized:
        values = np.diag(covariance)
        top = values.max()
        idx = int(np.flatnonzero(top - values <= eigenspace_rtol * max(top, np.finfo(float).tiny))[0])
        direction = np.eye(len(values))[idx]
    else:
        values, vectors = np.linalg.eigh(covariance)
        top = values[-1]
        if values[0] < -1e-12 * max(1., top):
            raise FloatingPointError("Significantly negative posterior covariance eigenvalue")
        top_vectors = vectors[:, top - values <= eigenspace_rtol * max(top, np.finfo(float).tiny)]
        projector = top_vectors @ top_vectors.T
        for col in projector.T:
            norm = np.linalg.norm(col)
            if norm > eigenspace_rtol:
                direction = col / norm
                break
    if not top > 0:
        raise FloatingPointError("Posterior has zero principal width")
    return direction, float(np.sqrt(top))
