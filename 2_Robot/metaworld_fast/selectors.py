"""Six selectors, with random and schedule state independent of inference."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .line_template import make_template


def radial_cap(q, mean, radius=32.):
    delta = np.asarray(q) - mean
    norms = np.linalg.norm(delta, axis=1)
    scale = np.ones_like(norms)
    outside = norms > radius
    scale[outside] = radius / norms[outside]
    return mean + delta * scale[:, None]


def sg_stage(t):
    return int(np.floor(np.log2(1 + int(t) / 128)))


@dataclass
class SelectorState:
    stage: int | None = None
    center: np.ndarray | None = None


def select_query(method, posterior, M, t, rng, state, template=None):
    template = template if template is not None else make_template(posterior, M)
    mean, cov = template.mean, template.covariance
    if method == "RC":
        raw = template.queries[template.rc_index].copy()
    elif method == "PQ":
        raw = mean + template.quantiles[:, None] * template.direction
    elif method == "QI":
        raw = posterior.hypothetical_means(template.queries[template.rc_index])
    elif method == "PS":
        raw = posterior.sample(M, rng)
    elif method == "GC":
        values, vectors = np.linalg.eigh((cov + cov.T) / 2)
        if values[0] < -1e-12:
            raise FloatingPointError("GC covariance has a negative eigenvalue")
        root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
        raw = mean + rng.standard_normal((M, len(mean))) @ root.T
    elif method == "SG":
        stage = sg_stage(t)
        if stage != state.stage:
            state.stage, state.center = stage, mean.copy()
        raw = state.center + 2.**(-stage) * rng.standard_normal((M, len(mean)))
    else:
        raise ValueError(method)
    q = radial_cap(raw, mean) if method in {"GC", "SG"} else raw.copy()
    if not np.isfinite(q).all() or np.max(np.abs(q)) > 33 + 1e-10:
        raise FloatingPointError("Displayed query violates finite global coordinate envelope")
    return raw, q
