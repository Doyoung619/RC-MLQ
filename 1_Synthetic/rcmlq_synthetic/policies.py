"""Policy variants and binary transfers for the synthetic simulation."""
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import entr, expit
from scipy.stats import qmc

from .config import BINARY
from metaworld_fast.selectors import radial_cap, select_query

RADIUS = 32.0


def _squared_loss(weights, sums, squares, center):
    return squares - 2 * sums * center + weights * center**2


def _partitions(nodes, weights):
    stats = np.stack((weights, weights * nodes, weights * nodes**2))
    left = np.column_stack((np.zeros(3), np.cumsum(stats, axis=1)))
    return left, left[:, -1:] - left


def _centers(parts, fallback):
    mass, first, _ = parts
    return np.divide(first, mass, out=np.full_like(first, fallback), where=mass > 0)


def _update_center(left, right, fixed):
    """Minimize nearest-center loss over a single free scalar center."""
    left_center, right_center = _centers(left, fixed), _centers(right, fixed)
    scores = np.concatenate((
        _squared_loss(*left, left_center) + _squared_loss(*right, fixed),
        _squared_loss(*left, fixed) + _squared_loss(*right, right_center),
    ))
    return float(np.r_[left_center, right_center][scores.argmin()])


def _mem_centers(method, marginal):
    left, right = _partitions(marginal.x, marginal.weights)
    if method == "MEM":
        a, b = _centers(left, marginal.mean), _centers(right, marginal.mean)
        best = (_squared_loss(*left, a) + _squared_loss(*right, b)).argmin()
        return np.asarray([a[best], b[best]])
    centers = np.array([marginal.mean, _update_center(left, right, marginal.mean)])
    if method == "2M":
        for slot in (0, 1, 0, 1):
            centers[slot] = _update_center(left, right, centers[1 - slot])
    return centers


@lru_cache(maxsize=16)
def _normal_bank(dimension, count, seed):
    return qmc.MultivariateNormalQMC(np.zeros(dimension), seed=seed).random(count)


def _entropy_search(projection, squared_norm, search_max):
    def objective(log_spacing):
        x = np.exp(log_spacing)
        denom = 0.25 * np.sqrt(2 * (squared_norm + x * x)**2 + 8 * (x * projection)**2)
        p = expit(4 * x * projection / denom)
        return float(np.mean(entr(p) + entr(1 - p)))

    search_grid = np.linspace(np.log(0.01), np.log(search_max), 25)
    values = np.array([objective(x) for x in search_grid])
    candidates = [(values[0], search_grid[0]), (values[-1], search_grid[-1])]
    for i in range(1, len(values) - 1):
        if values[i] <= min(values[i - 1], values[i + 1]):
            result = minimize_scalar(objective, bounds=(search_grid[i - 1], search_grid[i + 1]),
                                     method="bounded", options={"xatol": 1e-5})
            candidates.extend(((values[i], search_grid[i]), (result.fun, result.x)))
    return min(candidates)[1], objective


def _info_synth_spacing(template):
    values = np.diag(template.covariance)
    trace = float(np.sum(values))
    absolute_scale = np.sqrt(trace)
    search_max = min(10.0, RADIUS / absolute_scale)
    axis = int(np.argmax(template.direction))

    def search(count, seed):
        points = _normal_bank(len(values), count, seed) * np.sqrt(values / trace)
        return _entropy_search(points[:, axis], np.sum(points**2, axis=1), search_max)

    coarse, first_score = search(1024, 1)
    refined, second_score = search(4096, 1)
    tolerance = 5e-4
    if max(abs(first_score(coarse) - second_score(coarse)),
           second_score(coarse) - second_score(refined)) <= tolerance:
        return float(np.exp(coarse) * absolute_scale)
    final, _ = search(16384, 1)
    return float(np.exp(final) * absolute_scale)


def choose(job, posterior, template, t, rng, state, evalues=None):
    """Return raw/displayed candidates and the physical scale if applicable."""
    method, m = job["method"], job["M"]
    mean, direction = template.mean, template.direction
    if method in BINARY and m != 2:
        raise ValueError(f"{method} requires M=2")
    index = None
    if method == "RC" or job["fisher_grid_variant"]:
        index = template.rc_index
    elif method == "RC-EIG":
        index = int(np.argmax(evalues))
    elif method.startswith("RC-Fixed-"):
        scale = float(method.removeprefix("RC-Fixed-"))
        index = job["grid"].index(scale)

    if index is not None:
        raw = template.queries[index].copy()
        return raw, raw.copy(), job["grid"][index]
    if method == "SM":
        offset = rng.standard_normal(len(mean)) * np.sqrt(np.diag(template.covariance))
        raw = mean + np.stack((offset, offset - 2 * (offset @ direction) * direction))
        return raw, radial_cap(raw, mean, RADIUS), None
    if method in ("MEM", "GM", "2M"):
        axis = int(np.argmax(direction))
        centers = _mem_centers(method, posterior.marginals[axis])
        raw = mean + (centers - mean[axis])[:, None] * direction
        return raw, raw.copy(), None
    if method in ("IT", "IB"):
        scale = (float(np.sqrt(np.trace(template.covariance))) if method == "IT"
                 else _info_synth_spacing(template))
        raw = mean + np.array([-scale, scale])[:, None] * direction
        return raw, raw.copy(), scale
    raw, displayed = select_query(method, posterior, m, t, rng, state, template)
    return raw, displayed, None
