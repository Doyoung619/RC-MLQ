"""Frozen-posterior binary selectors for the incremental M=2 experiment.

These are the transfers specified in ``metaworld_m2_addon_codex_prompt.md``;
none claims end-to-end reproduction of an original catalog algorithm. The
legacy selector module, posterior engines, and response likelihood are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtri
from scipy.stats import qmc

from .model import principal_direction
from .selectors import radial_cap


MARGINAL_METHODS = (
    "sm_continuous", "mem_line", "gmem_line", "2mem_line_2sweeps",
    "infosynth_trace", "infosynth_bounded",
)
GCQ_STAGES = {"gausscloud_paper_L1": 1, "gausscloud_paper_L3": 3,
              "gausscloud_paper_L5": 5}
NEW_METHODS = MARGINAL_METHODS + tuple(GCQ_STAGES)
INITIAL_STD = 1. / np.sqrt(3.)
CAP_RADIUS = 32.
IB_SIGMA0 = .25
IB_POINTS = 1024
IB_REFERENCE_POINTS = 4096
IB_DISCREPANCY_TOLERANCE = .0005
IB_LOG_X_TOLERANCE = .00001


@dataclass
class AddonSelectorState:
    """Checkpointable state; diagnostic draws must use a deep copy."""
    stage: int | None = None
    center: np.ndarray | None = None
    std: float | None = None
    method: str | None = None
    horizon: int | None = None
    diagnostics: dict = field(default_factory=dict)


def stage_schedule(horizon=128, stages=3):
    """Author-code allocation: floor each budget, all remainder in final stage."""
    if int(horizon) != horizon or int(stages) != stages or not 1 <= stages <= horizon:
        raise ValueError("Require integer horizon >= stages >= 1")
    horizon, stages = int(horizon), int(stages)
    lengths = np.full(stages, horizon // stages, dtype=np.int64)
    lengths[-1] += horizon - int(lengths.sum())
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    return lengths, starts


def stage_at(t, horizon=128, stages=3):
    lengths, starts = stage_schedule(horizon, stages)
    if int(t) != t or not 0 <= t < horizon:
        raise ValueError("Selector requires pre-answer 0 <= t < horizon; no dummy final query")
    return int(np.searchsorted(starts, int(t), side="right") - 1)


def _weighted_nodes(nodes, weights):
    x, w = np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or w.shape != x.shape or not len(x):
        raise ValueError("Expected nonempty scalar integration nodes and matching weights")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("MEM requires finite nodes and nonnegative positive-total weights")
    if np.any((x < -1.-1e-12) | (x > 1.+1e-12)):
        raise ValueError("MEM marginal nodes must have support [-1,1]")
    keep = w > 0
    x, w = np.clip(x[keep], -1., 1.), w[keep]
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    return x, w / w.sum()


def mem_objective(nodes, weights, a, b):
    """Frozen discrete expected best quadratic loss (not information gain)."""
    x, w = _weighted_nodes(nodes, weights)
    return float(w @ np.minimum((x-float(a))**2, (x-float(b))**2))


def _deterministic_min(values, keys):
    """Numerically tied minima choose the first lexicographic sorted key."""
    values = np.asarray(values, dtype=np.float64)
    tolerance = 16 * np.finfo(float).eps * max(float(np.max(np.abs(values))), 1e-15)
    candidates = np.flatnonzero(values <= values.min() + tolerance)
    return min(candidates, key=lambda i: tuple(np.atleast_1d(keys[i])))


def solve_mem(nodes, weights):
    """Global optimum of the frozen discrete binary coordinate-line objective."""
    x, w = _weighted_nodes(nodes, weights)
    if len(x) == 1 or x[0] == x[-1]:
        return np.array([x[0], x[0]]), 0.
    # Translate the moments before subtracting them for concentrated posteriors.
    anchor = float(w @ x)
    z = x-anchor
    mass = np.r_[0., np.cumsum(w)]
    first = np.r_[0., np.cumsum(w*z)]
    second = np.r_[0., np.cumsum(w*z*z)]
    splits = np.flatnonzero(x[:-1] < x[1:]) + 1
    suffix_mass = np.r_[np.cumsum(w[::-1])[::-1], 0.]
    suffix_first = np.r_[np.cumsum((w*z)[::-1])[::-1], 0.]
    suffix_second = np.r_[np.cumsum((w*z*z)[::-1])[::-1], 0.]
    left, right = mass[splits], suffix_mass[splits]
    first_left, first_right = first[splits], suffix_first[splits]
    a = np.clip(anchor+first_left/left, -1., 1.)
    b = np.clip(anchor+first_right/right, -1., 1.)
    costs = (second[splits]-first_left**2/left
             + suffix_second[splits]-first_right**2/right)
    keys = np.column_stack([a, b])
    chosen = _deterministic_min(costs, keys)
    pair = keys[chosen].copy()
    return pair, float(w @ np.minimum((x-pair[0])**2, (x-pair[1])**2))


def solve_moving_center(nodes, weights, fixed):
    """Piecewise-quadratic global solve with one fixed center on [-1,1].

    All breakpoints, interval quadratic minima, and endpoints are evaluated.
    Prefix moments compute interval minimizers without repeated node loops.
    """
    x, w = _weighted_nodes(nodes, weights)
    fixed = float(fixed)
    if not np.isfinite(fixed) or not -1. <= fixed <= 1.:
        raise ValueError("Fixed MEM center must lie in [-1,1]")
    breaks = np.unique(np.r_[-1., fixed, 1., np.clip(2*x-fixed, -1., 1.)])
    lo, hi = breaks[:-1], breaks[1:]
    mid = (lo+hi)/2
    partition = np.searchsorted(x, (mid+fixed)/2, side="right")
    mass = np.r_[0., np.cumsum(w)]
    first = np.r_[0., np.cumsum(w*x)]
    moving_left = mid < fixed
    suffix_mass = np.r_[np.cumsum(w[::-1])[::-1], 0.]
    suffix_first = np.r_[np.cumsum((w*x)[::-1])[::-1], 0.]
    active_mass = np.where(moving_left, mass[partition], suffix_mass[partition])
    active_first = np.where(moving_left, first[partition], suffix_first[partition])
    means = np.divide(active_first, active_mass, out=lo.copy(), where=active_mass > 0)
    candidates = np.unique(np.r_[breaks, np.clip(means, lo, hi)])
    # Actual min-distance evaluation avoids cancellation in small cluster losses.
    losses = np.minimum((x[None, :]-fixed)**2, (x[None, :]-candidates[:, None])**2) @ w
    chosen = _deterministic_min(losses, candidates)
    return float(candidates[chosen]), float(losses[chosen])


def solve_gmem(nodes, weights, mean=None):
    x, w = _weighted_nodes(nodes, weights)
    fixed = float(w @ x) if mean is None else float(mean)
    moving, objective = solve_moving_center(x, w, fixed)
    return np.array([fixed, moving]), objective


def solve_two_mem(nodes, weights, mean=None):
    x, w = _weighted_nodes(nodes, weights)
    pair, initial = solve_gmem(x, w, mean)
    objectives = [initial]
    centers = [pair.tolist()]
    for slot in (0, 1, 0, 1):
        value, objective = solve_moving_center(x, w, pair[1-slot])
        if objective > objectives[-1] + 5e-13:
            raise FloatingPointError("Frozen 2MEM one-center update increased its objective")
        pair[slot] = value
        objectives.append(objective)
        centers.append(pair.tolist())
    return pair, {"slot_order": [0, 1, 0, 1], "objectives": objectives,
                  "centers": centers, "sweeps": 2}


def binary_entropy_from_logits(logits):
    """Positive entropy in nats; stable for arbitrarily large finite logits."""
    magnitude = np.abs(np.asarray(logits, dtype=np.float64))
    return np.logaddexp(0., -magnitude) + magnitude*expit(-magnitude)


def ib_conditional_entropy(relative, normalized_draws, direction, sigma0=IB_SIGMA0):
    """Auxiliary Gaussian acquisition only; never a response likelihood."""
    relative = float(relative)
    draws = np.asarray(normalized_draws, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if not relative > 0 or not sigma0 > 0:
        raise ValueError("IB requires positive relative spacing and sigma0")
    projection = draws @ direction
    norm_sq = np.sum(draws*draws, axis=1)
    denominator = float(sigma0)*np.sqrt(2*(norm_sq+relative**2)**2 + 8*relative**2*projection**2)
    logits = 4*relative*projection/denominator
    return float(np.mean(binary_entropy_from_logits(logits)))


def ib_raw_conditional_entropy(radius, centered_gaussian_draws, direction, sigma0=IB_SIGMA0):
    """Equivalent unnormalized acquisition, exposed for translation checks."""
    return ib_conditional_entropy(radius, centered_gaussian_draws, direction, sigma0)


def minimize_bounded_log_objective(objective, lower, upper, grid_points=25,
                                   log_tolerance=IB_LOG_X_TOLERANCE):
    """Refine every grid local minimum and explicitly retain both boundaries."""
    if not 0 < lower <= upper or grid_points < 3:
        raise ValueError("Expected a positive nonempty interval and >=3 grid points")
    grid = np.linspace(np.log(lower), np.log(upper), grid_points)
    grid_relative = np.clip(np.exp(grid), lower, upper)
    grid_relative[0], grid_relative[-1] = lower, upper
    values = np.array([objective(float(x)) for x in grid_relative])
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite IB auxiliary acquisition")
    trials = [(float(z), float(value)) for z, value in zip(grid, values)]
    local = [i for i in range(1, grid_points-1)
             if values[i] <= values[i-1] and values[i] <= values[i+1]]
    # Edge cells can contain a minimum even when the endpoint is grid-best.
    brackets = [(grid[i-1], grid[i+1]) for i in local]
    if values[0] <= values[1]:
        brackets.append((grid[0], grid[1]))
    if values[-1] <= values[-2]:
        brackets.append((grid[-2], grid[-1]))
    statuses = []
    if upper > lower:
        for left, right in brackets:
            result = minimize_scalar(lambda z: objective(float(np.clip(np.exp(z), lower, upper))),
                                     bounds=(left, right), method="bounded",
                                     options={"xatol": log_tolerance})
            if not result.success or not np.isfinite(result.fun):
                raise FloatingPointError("IB bounded log-scale refinement failed")
            trials.append((float(result.x), float(result.fun)))
            statuses.append({"success": bool(result.success), "evaluations": int(result.nfev)})
    trials.sort(key=lambda item: item[0])
    best = _deterministic_min([v for _, v in trials], [z for z, _ in trials])
    log_x, loss = trials[best]
    x = float(lower if log_x <= grid[0] else upper if log_x >= grid[-1]
              else np.clip(np.exp(log_x), lower, upper))
    return x, {"objective": loss, "relative_bounds": [float(lower), float(upper)],
               "grid_relative": grid_relative.tolist(), "grid_objective": values.tolist(),
               "local_minimum_indices": local, "refinement_status": statuses,
               "boundary_selected": bool(x == lower or x == upper)}


def select_ib(covariance, direction, rng):
    covariance = np.asarray(covariance, dtype=np.float64)
    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace <= 0:
        raise FloatingPointError("IB requires positive finite covariance trace")
    s = float(np.sqrt(trace))
    upper = min(10., CAP_RADIUS/s)
    if upper < .01:
        raise ValueError("IB physical cap and relative lower bound have empty intersection")
    values, vectors = np.linalg.eigh((covariance+covariance.T)/(2*trace))
    if values[0] < -1e-12:
        raise FloatingPointError("IB covariance is not positive semidefinite")
    root = (vectors*np.sqrt(np.maximum(values, 0))) @ vectors.T
    sobol_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    uniform = qmc.Sobol(len(direction), scramble=True, seed=sobol_seed).random_base2(12)
    # The same nested Gaussian draws are used in every grid/refinement evaluation.
    gaussian = ndtri(np.clip(uniform, np.finfo(float).eps, 1-np.finfo(float).eps)) @ root.T
    primary = lambda x: ib_conditional_entropy(x, gaussian[:IB_POINTS], direction)
    reference = lambda x: ib_conditional_entropy(x, gaussian, direction)
    selected, details = minimize_bounded_log_objective(primary, .01, upper)
    check_points = np.r_[details["grid_relative"], selected]
    discrepancy = float(max(abs(primary(x)-reference(x)) for x in check_points))
    primary_selected = selected
    refined = discrepancy > IB_DISCREPANCY_TOLERANCE
    if refined:
        selected, details = minimize_bounded_log_objective(reference, .01, upper)
    details.update({"relative_halfspacing": selected, "halfspacing": s*selected,
                    "trace_scale": s, "sigma0": IB_SIGMA0, "sobol_seed": sobol_seed,
                    "primary_points": IB_POINTS, "reference_points": IB_REFERENCE_POINTS,
                    "selected_integration_points": IB_REFERENCE_POINTS if refined else IB_POINTS,
                    "refined": refined, "max_reference_discrepancy": discrepancy,
                    "primary_relative_halfspacing": primary_selected,
                    "reference_objective_at_selected": reference(selected),
                    "objective_kind": "auxiliary_gaussian_conditional_entropy"})
    return s*selected, details


def select_query(method, posterior, M, t, rng, state, template=None, *, horizon=128):
    """Return raw/displayed pairs, changing selector state but never posterior.

    ``template`` may supply frozen mean/covariance/direction. The caller owns
    independent actual/diagnostic RNG streams and must clone state for diagnostics.
    The six coordinate methods are candidates for marginal inference; the runner
    must check the actual displayed geometry before choosing the update backend.
    """
    if M != 2:
        raise ValueError("All M=2 addon selectors require M=2")
    if method not in NEW_METHODS:
        raise ValueError(f"Unknown addon method: {method}")
    stage_at(t, horizon, 1)
    if template is None:
        mean, cov = posterior.moments()
        direction, _ = principal_direction(cov, bool(posterior.factorized))
    else:
        mean, cov = template.mean, template.covariance
        direction = getattr(template, "direction", None)
        if direction is None:
            direction, _ = principal_direction(cov, bool(getattr(posterior, "factorized", False)))
    mean, cov = np.asarray(mean, dtype=np.float64), np.asarray(cov, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    details = {"method": method, "preanswer_t": int(t), "horizon": int(horizon)}
    if method in GCQ_STAGES:
        stages = GCQ_STAGES[method]
        lengths, starts = stage_schedule(horizon, stages)
        stage = stage_at(t, horizon, stages)
        if state.method not in (None, method) or state.horizon not in (None, horizon):
            raise ValueError("GCQ checkpoint method/horizon mismatch")
        if stage != state.stage:
            if t != starts[stage]:
                raise ValueError("GCQ midstage resume requires restored stage/center state")
            if state.stage is not None and stage != state.stage+1:
                raise ValueError("GCQ stage sequence skipped or moved backward")
            state.center = mean.copy()
            state.stage = stage
        if state.center is None or np.asarray(state.center).shape != mean.shape:
            raise ValueError("GCQ checkpoint has no valid frozen stage center")
        state.method, state.horizon = method, int(horizon)
        state.std = INITIAL_STD*2.**(-stage)
        raw = state.center[None, :] + state.std*rng.standard_normal((2, len(mean)))
        details.update({"stage": stage, "stage_center": state.center.tolist(),
                        "std": state.std, "initial_std": INITIAL_STD,
                        "stage_lengths": lengths.tolist(), "stage_starts": starts.tolist(),
                        "stage_entry": bool(t == starts[stage])})
    elif method == "sm_continuous":
        values, vectors = np.linalg.eigh((cov+cov.T)/2)
        if values[0] < -1e-12:
            raise FloatingPointError("SM covariance is not positive semidefinite")
        root = (vectors*np.sqrt(np.maximum(values, 0))) @ vectors.T
        z = root @ rng.standard_normal(len(mean))
        reflected = z-2*float(direction @ z)*direction
        raw = mean+np.stack([z, reflected])
        details.update({"direction": direction.tolist(), "gaussian_vector": z.tolist(),
                        "geometry": "principal_hyperplane_reflection"})
    elif method in {"mem_line", "gmem_line", "2mem_line_2sweeps"}:
        if not bool(getattr(posterior, "factorized", False)) or not hasattr(posterior, "marginals"):
            raise ValueError("MEM-line family requires product posterior marginal nodes")
        direction, _ = principal_direction(cov, True)
        coordinate = int(np.argmax(direction))
        marginal = posterior.marginals[coordinate]
        if method == "mem_line":
            pair, objective = solve_mem(marginal.x, marginal.weights)
            details["objective"] = objective
        elif method == "gmem_line":
            pair, objective = solve_gmem(marginal.x, marginal.weights, mean[coordinate])
            details["objective"] = objective
        else:
            pair, refinement = solve_two_mem(marginal.x, marginal.weights, mean[coordinate])
            details.update(refinement)
            details["objective"] = refinement["objectives"][-1]
        raw = np.broadcast_to(mean, (2, len(mean))).copy()
        raw[:, coordinate] = pair
        details.update({"coordinate": coordinate, "scalar_centers": pair.tolist(),
                        "objective_kind": "frozen_discrete_expected_best_quadratic_loss",
                        "center_domain": [-1., 1.]})
    elif method == "infosynth_trace":
        spacing = float(np.sqrt(np.trace(cov)))
        raw = mean+np.array([-spacing, spacing])[:, None]*direction
        details.update({"halfspacing": spacing, "trace_scale": spacing,
                        "direction": direction.tolist()})
    else:
        spacing, acquisition = select_ib(cov, direction, rng)
        raw = mean+np.array([-spacing, spacing])[:, None]*direction
        details.update(acquisition)
        details["direction"] = direction.tolist()
    capped = method in GCQ_STAGES or method == "sm_continuous"
    displayed = radial_cap(raw, mean, CAP_RADIUS) if capped else raw.copy()
    raw_norms = np.linalg.norm(raw-mean, axis=1)
    details.update({"raw_centered_norms": raw_norms.tolist(),
                    "displayed_centered_norms": np.linalg.norm(displayed-mean, axis=1).tolist(),
                    "cap_active": (raw_norms > CAP_RADIUS).tolist() if capped else [False, False]})
    if not np.isfinite(displayed).all() or np.max(np.abs(displayed)) > 33+1e-10:
        raise FloatingPointError("Displayed addon query violates finite global coordinate envelope")
    state.diagnostics = details
    return raw, displayed
