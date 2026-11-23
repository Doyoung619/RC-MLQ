"""Label-free frozen-batch acquisition for real-response protocol v2.

BAIT is the rank-one, matched-sum specialization of Ash et al. (2021),
not the author's neural normalization. E-SDP uses a capped continuous
relaxation with custom dependent rounding, not the Allen-Zhu sparsifier.
All computations are float64; no target/reference labels enter this module.
"""
from __future__ import annotations

import copy
import hashlib
import time
import warnings
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit

METHODS = ("fb_sample", "fb_range", "a_opt", "d_opt", "e_greedy", "ucb_glm",
           "ts_glm", "random", "bait_glm", "e_sdp", "badge", "fb_opt", "fb_opt2")
METHOD_NAMES = dict(zip(METHODS, (
    "FisherBalancing-Sample", "FisherBalancing-Range", "A-opt", "D-opt",
    "E-greedy", "UCB-GLM plug-in", "Thompson-GLM Laplace", "Random",
    "BAIT-GLM, forward–backward", "E-SDP + rounding", "BADGE-logistic", "FB-opt", "FB-opt2")))
METHOD_NAMES.update({'b_emcm':'B-EMCM','lcmd':'LCMD','regvol':'RegVol',
                     'entropy':'Entropy','firal':'FIRAL'})
SOURCES = {
    "bait_paper": "https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf",
    "bait_code": "https://raw.githubusercontent.com/JordanAsh/badge/master/query_strategies/bait_sampling.py",
    "badge_paper": "https://arxiv.org/abs/1906.03671",
    "badge_code": "https://raw.githubusercontent.com/JordanAsh/badge/master/query_strategies/badge_sampling.py",
    "e_relaxation": "https://proceedings.mlr.press/v70/allen-zhu17e.html",
    "dpp": "https://www.cvxpy.org/tutorial/dpp/index.html",
}


def stable_priorities(ids, *key):
    """Vectorized SplitMix64 permutation keyed by SHA256; no Python hash()."""
    salt = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")
    z = np.asarray(ids, dtype=np.uint64) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def shared_candidates(n_pool, labeled_ids, dataset, seed, batch_index, K):
    available = np.ones(n_pool, dtype=bool)
    available[np.asarray(labeled_ids, dtype=np.int64)] = False
    ids = np.flatnonzero(available)
    priority = stable_priorities(ids, "candidate", dataset, int(seed), int(batch_index))
    return ids[np.argsort(priority, kind="stable")[:min(K, len(ids))]]


def _winner(scores, valid, priority, maximize=True, rtol=1e-11):
    ii = np.flatnonzero(valid)
    if not len(ii):
        raise ValueError("Empty selection set")
    values = np.asarray(scores)[ii]
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite acquisition scores")
    best = values.max() if maximize else values.min()
    tied = ii[np.abs(values - best) <= rtol * max(1., abs(float(best)))]
    j = tied[np.argmin(priority[tied])]
    return int(j), int(len(tied))


def curvature(eta, family):
    eta = np.asarray(eta, dtype=np.float64)
    if family in ("bernoulli", "logistic"):
        # More stable than p*(1-p) when eta is large and positive.
        return expit(eta) * expit(-eta)
    if family in ("gaussian", "normal"):
        return np.ones_like(eta)
    if family == "poisson":
        with np.errstate(over="raise", invalid="raise"):
            return np.exp(eta)
    raise ValueError(f"Unknown GLM family: {family}")


def fb_range_temperature(family, m, d, config=None, U=None, valid=None):
    """Original FB-Range expression, shared without changing its arithmetic."""
    config = {} if config is None else config
    R = 1. if family in ("gaussian", "normal") else .25
    if family == "poisson":
        R = float(np.max(np.sum(U[valid]**2, axis=1)))
    tau = max(1e-8, float(config.get("temperature_multiplier", config.get("fb_range_multiplier", 1.))) * R
              * np.sqrt(m / (8. * np.log(max(2, d)))))
    return tau, R


def fb_opt2_scale(family, d, m, Tmax, config=None):
    """Use the exact FB-Range path as the final temperature; no new multiplier."""
    if family not in ('gaussian', 'normal', 'bernoulli', 'logistic'):
        raise ValueError('FB-opt2 supports Gaussian and Bernoulli only')
    if d <= 1 or int(m) != m or int(Tmax) != Tmax or m <= 0 or Tmax <= 0:
        raise ValueError('FB-opt2 requires d > 1 and positive integer m and Tmax')
    m, Tmax = int(m), int(Tmax)
    tau_range, R = fb_range_temperature(family, m, d, config)
    Nmax = m * Tmax
    return dict(method_name='FB-opt2', method_version='fb_opt2_v1_range_anchored',
                d=int(d), m=m, Tmax=Tmax, Nmax=Nmax, range_R=R,
                tau_range=float(tau_range), tau0_opt2=float(tau_range/Nmax**.75),
                range_temperature_definition='existing fb_range_temperature, unchanged R/floor/multiplier',
                temperature_formula='tau_t = tau_range * ((m*t)/Nmax)**0.75; n0 excluded; fixed within batch',
                temperature_has_tuned_multiplier=False)


def fb_opt2_temperature(scale, batch_index):
    if not 1 <= batch_index <= scale['Tmax']:
        raise ValueError('FB-opt2 adaptive round outside planned horizon')
    # Ratio form preserves exact equality with tau_range at the final batch.
    return scale['tau_range'] * ((scale['m'] * batch_index) / scale['Nmax'])**.75


def fb_opt_scale(X_pool, family, m, Tmax):
    """Exact finite-horizon calibration; only full-pool features and design counts."""
    X = np.asarray(X_pool, dtype=np.float64)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("FB-opt requires a finite nonempty full query pool")
    d = X.shape[1]
    if d <= 1:
        raise ValueError("Finite-horizon FB-opt requires query dimension d > 1")
    if int(m) != m or int(Tmax) != Tmax or m <= 0 or Tmax <= 0:
        raise ValueError("FB-opt requires positive integer batch size m and horizon Tmax")
    m, Tmax = int(m), int(Tmax)
    if family in ("gaussian", "normal"):
        w_bar = 1.
    elif family in ("bernoulli", "logistic"):
        w_bar = .25
    else:
        raise ValueError("FB-opt is defined only for Gaussian and Bernoulli")
    L = float(np.sqrt(np.max(np.einsum('ij,ij->i', X, X))))
    if L <= 0:
        raise ValueError("FB-opt requires a positive query radius")
    B = w_bar * L**2
    Nmax = m * Tmax
    S_T = np.sum((np.float64(m) * np.arange(1, Tmax + 1, dtype=np.float64))**(-0.75), dtype=np.float64)
    tau0 = np.float64(B) * np.sqrt(np.float64(m) * S_T / (np.float64(Nmax)**0.75 * np.log(np.float64(d))))
    previous = B / np.sqrt(np.log(np.float64(d))) # Diagnostic only; never used for scores.
    return dict(method_name="FB-opt", method_version="fb_opt_v2_finite_horizon",
                L=L, w_bar=w_bar, B=B, d=d, m=m, Tmax=Tmax, Nmax=Nmax, S_T=float(S_T),
                tau0_opt=float(tau0), tau0_previous=float(previous),
                tau0_ratio_new_previous=float(tau0/previous),
                L_scope="original_full_query_pool_including_warm_start",
                calibration_formula="B * sqrt(m * S_T / ((m*Tmax)**0.75 * log(d)))",
                temperature_formula="tau_t = tau0_opt * (m*t)**0.75; n0 excluded; fixed within batch",
                temperature_has_tuned_multiplier=False)


def _inverse(W):
    ans = cho_solve(cho_factor(W, lower=True, check_finite=False),
                    np.eye(len(W)), check_finite=False)
    return (ans + ans.T) * .5


class InverseCache:
    """Rank-one inverse and cached Z,q,r, checked against Cholesky."""
    def __init__(self, W, U, F=None, check_every=32):
        self.W = np.array(W, dtype=float, copy=True)
        self.U = U
        self.F = np.eye(W.shape[0]) if F is None else F
        self.check_every = max(1, int(check_every))
        self.updates = self.rebuilds = 0
        self.max_inverse_error = 0.
        self.rebuild()
        self.rebuilds = 0

    def rebuild(self):
        self.A = _inverse(self.W)
        self.Z = self.U @ self.A
        self.q = np.einsum("ij,ij->i", self.Z, self.U)
        self.r = np.einsum("ij,ij->i", self.Z @ self.F, self.Z)
        self.rebuilds += 1
        self._nonnegative()

    def _nonnegative(self):
        for key in ("q", "r"):
            a = getattr(self, key)
            tolerance = 2e-10 * max(1., float(np.max(np.abs(a))))
            if np.min(a) < -tolerance:
                raise FloatingPointError(f"Negative {key} exceeds roundoff tolerance")
            np.maximum(a, 0., out=a)

    def update(self, j, sign=1):
        u = self.U[j]
        v = self.A @ u
        denom = 1. + sign * float(u @ v)
        self.W += sign * np.outer(u, u)
        self.W = (self.W + self.W.T) * .5
        self.updates += 1
        if denom <= 1e-10:
            # Especially for BAIT removal: never clip a bad denominator.
            self.rebuild()
            return
        coeff = sign / denom
        p = self.U @ v
        Fv = self.F @ v
        # RHS uses the OLD Z, before its rank-one update.
        self.r = (self.r - 2. * coeff * p * (self.Z @ Fv)
                  + coeff**2 * p**2 * float(v @ Fv))
        self.q -= coeff * p**2
        self.Z -= coeff * np.outer(p, v)
        self.A -= coeff * np.outer(v, v)
        self.A = (self.A + self.A.T) * .5
        try:
            self._nonnegative()
        except FloatingPointError:
            self.rebuild()
        if self.updates % self.check_every == 0:
            exact = _inverse(self.W)
            error = float(np.linalg.norm(self.A - exact) / max(1., np.linalg.norm(exact)))
            self.max_inverse_error = max(self.max_inverse_error, error)
            if error > 1e-9:
                self.rebuild()


def shared_initial_ids(X_pool, dataset, seed, n0=80):
    """Call on the fixed MAXIMUM-d representation, once per dataset/seed."""
    X = np.asarray(X_pool, dtype=np.float64)
    if len(X) < n0:
        raise ValueError("Pool smaller than shared warm start")
    ids = np.arange(len(X))
    priority = stable_priorities(ids, "initialization", dataset, int(seed))
    candidates = ids[np.argsort(priority, kind="stable")[:min(2000, len(X))]]
    U = X[candidates]
    cache = InverseCache(np.eye(X.shape[1]), U)
    valid = np.ones(len(U), dtype=bool)
    selected = []
    for _ in range(n0):
        j, _ = _winner(cache.q, valid, priority[candidates])
        valid[j] = False
        selected.append(j)
        cache.update(j)
    return candidates[selected]


def e_scores_batched(W, U, chunk_size=512, backend="batched"):
    """Exact batched lambda_min, optional CUDA with transfers included."""
    if backend.startswith("cuda"):
        import torch
        device = backend if ":" in backend else "cuda:0"
        with torch.no_grad():
            wt = torch.as_tensor(W, dtype=torch.float64, device=device)
            ut = torch.as_tensor(U, dtype=torch.float64, device=device)
            pieces = []
            for start in range(0, len(U), chunk_size):
                v = ut[start:start + chunk_size]
                pieces.append(torch.linalg.eigvalsh(wt + v[:, :, None] * v[:, None, :])[:, 0])
            return torch.cat(pieces).cpu().numpy()
    scores = np.empty(len(U))
    for start in range(0, len(U), chunk_size):
        v = U[start:start + chunk_size]
        scores[start:start + len(v)] = np.linalg.eigvalsh(W + v[:, :, None] * v[:, None, :])[:, 0]
    return scores


def e_scores_secular(W, U, rtol=1e-12, max_iters=48):
    """Vectorized rank-one minimum eigenvalue with deflation/pole handling."""
    lam, Q = np.linalg.eigh(W)
    if len(lam) == 1:
        return lam[0] + U[:, 0]**2
    gap = lam[1] - lam[0]
    tol = 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(lam))))
    if gap <= tol:
        return np.full(len(U), lam[0])
    Z2 = (U @ Q)**2
    # Translate to t=s-lambda_min, which avoids cancellation at large lambda.
    shifted = lam - lam[0]
    lo = np.zeros(len(U))
    hi = np.full(len(U), gap)
    active = Z2[:, 0] > 0.
    for _ in range(max_iters):
        mid = (lo + hi) * .5
        denom = shifted[None, :] - mid[:, None]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms = np.divide(Z2, denom, out=np.zeros_like(Z2), where=Z2 != 0.)
            f = 1. + terms.sum(axis=1)
        below = (f < 0.) & active
        lo = np.where(below, mid, lo)
        hi = np.where((~below) & active, mid, hi)
        if np.max(hi - lo) <= rtol * max(1., abs(float(lam[0])), gap):
            break
    return lam[0] + np.where(active, (lo + hi) * .5, 0.)


def _e_greedy(W, U, m, priority, config):
    W = W.copy()
    valid = np.ones(len(U), dtype=bool)
    selected, trace = [], []
    fallbacks = 0
    backend = config.get("e_backend", "secular")
    for slot in range(m):
        ii = np.flatnonzero(valid)
        if backend == "secular":
            values = e_scores_secular(W, U[ii])
        else:
            values = e_scores_batched(W, U[ii], backend=backend)
        scores = np.zeros(len(U))
        scores[ii] = values
        j, ties = _winner(scores, valid, priority)
        # Independently check winner and a few nearest competitors each slot.
        near = ii[np.argsort(values, kind="stable")[-min(4, len(ii)):]]
        check = np.unique(np.append(near, j))
        direct = e_scores_batched(W, U[check])
        error = float(np.max(np.abs(direct - scores[check])))
        tolerance = 1e-9 * max(1., abs(float(np.max(direct))))
        if error > tolerance:
            scores[ii] = e_scores_batched(W, U[ii])
            j, ties = _winner(scores, valid, priority)
            fallbacks += 1
        valid[j] = False
        selected.append(j)
        trace.append(dict(slot=slot, candidate_position=j, score=float(scores[j]),
                          numerical_ties=ties, direct_check_error=error))
        W += np.outer(U[j], U[j])
        W = (W + W.T) * .5
    return np.asarray(selected, dtype=np.int64), dict(slot_trace=trace,
                e_backend=backend, e_exact_fallbacks=fallbacks)


def project_capped_simplex(a, m):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)) or not 0 <= m <= len(a):
        raise ValueError("Invalid capped-simplex input")
    if m == 0:
        return np.zeros_like(a)
    if m == len(a):
        return np.ones_like(a)
    lo, hi = float(a.min() - 1.), float(a.max())
    for _ in range(80):
        mid = .5 * (lo + hi)
        if np.clip(a - mid, 0., 1.).sum() > m:
            lo = mid
        else:
            hi = mid
    answer = np.clip(a - .5 * (lo + hi), 0., 1.)
    if abs(answer.sum() - m) > 1e-9:
        raise FloatingPointError("Capped-simplex projection lost budget")
    return answer


def dependent_round(a, rng):
    p = np.asarray(a, dtype=float).copy()
    m = int(round(float(p.sum())))
    if abs(p.sum() - m) > 1e-8 or p.min() < -1e-10 or p.max() > 1 + 1e-10:
        raise ValueError("Dependent-rounding input must be on capped simplex")
    p = np.clip(p, 0., 1.)
    # Pair fractional entries in fixed candidate order; stochastic branch is
    # exactly the marginal-preserving pairwise dependent-rounding transition.
    for _ in range(len(p) + 2):
        fractional = np.flatnonzero((p > 1e-12) & (p < 1. - 1e-12))
        if len(fractional) < 2:
            break
        i, j = fractional[:2]
        alpha, beta = min(1. - p[i], p[j]), min(p[i], 1. - p[j])
        if rng.random() < beta / (alpha + beta):
            p[i] += alpha
            p[j] -= alpha
        else:
            p[i] -= beta
            p[j] += beta
    rounded = p > .5
    if int(rounded.sum()) != m or np.max(np.minimum(p, 1. - p)) > 1e-8:
        raise FloatingPointError("Dependent rounding failed exact cardinality")
    return np.flatnonzero(rounded)


def dual_upper_bound(W, U, m, raw_dual=None):
    d = len(W)
    fallback = raw_dual is None or not np.all(np.isfinite(raw_dual))
    if fallback:
        P = np.eye(d) / d
        info = dict(dual_fallback=True, dual_raw_min_eigenvalue=None,
                    dual_negative_mass=None, dual_projection_fro=None, dual_trace_before_normalization=None)
    else:
        raw = (raw_dual + raw_dual.T) * .5
        vals, Q = np.linalg.eigh(raw)
        positive = np.maximum(vals, 0.)
        total = float(positive.sum())
        projected = (Q * positive) @ Q.T
        P = projected / total if total > 1e-15 else np.eye(d) / d
        info = dict(dual_fallback=total <= 1e-15, dual_raw_min_eigenvalue=float(vals[0]),
                    dual_negative_mass=float(-np.minimum(vals, 0.).sum()),
                    dual_projection_fro=float(np.linalg.norm(projected - raw)),
                    dual_trace_before_normalization=total)
    P = (P + P.T) * .5
    scores = np.einsum("ij,ij->i", U @ P, U)
    ub = float(np.trace(P @ W) + np.sort(scores)[-m:].sum()) if m else float(np.trace(P @ W))
    info.update(feasible_dual_P=P.tolist(), dual_trace=float(np.trace(P)),
                dual_min_eigenvalue=float(np.linalg.eigvalsh(P)[0]))
    return ub, P, info


@lru_cache(maxsize=18)
def _sdp_template(d, K, solver="SCS"):
    import cvxpy as cp
    a, z = cp.Variable(K), cp.Variable()
    Wp, Qp, mp = cp.Parameter((d, d), symmetric=True), cp.Parameter((d*d, K)), cp.Parameter(nonneg=True)
    psd = Wp + cp.reshape(Qp @ a, (d, d), order="F") - z * np.eye(d) >> 0
    problem = cp.Problem(cp.Maximize(z), [psd, a >= 0, a <= 1, cp.sum(a) == mp])
    if not problem.is_dcp(dpp=True):
        raise RuntimeError("E-SDP template is not DPP compliant")
    return problem, a, z, Wp, Qp, mp, psd


def _e_sdp(W, U, m, priority, rng, config):
    sdp = config.get("sdp", {})
    greedy, greedy_audit = _e_greedy(W, U, m, priority, config)
    greedy_lb = float(np.linalg.eigvalsh(W + U[greedy].T @ U[greedy])[0])
    attempts = []
    raw_a = raw_dual = None
    status = "FAILED"
    scale = max(1., float(np.trace(W) / len(W)), float(np.max(np.sum(U**2, axis=1))))
    # A common external primal and valid-dual gap gate applies to each solver.
    feasibility_tol = float(sdp.get("feasibility_tolerance", 5e-4))
    gap_tol = float(sdp.get("objective_gap_tolerance", 2e-3))
    primary = sdp.get("primary_solver", "SCS").upper()
    plan = [(primary, False), (primary, True)]
    if primary != "CLARABEL":
        plan.append(("CLARABEL", True))
    try:
        # U_i U_i^T are symmetric, so row/column flattening coincide per i.
        fixed_Q = np.einsum("ki,kj->kij", U, U).reshape(len(U), -1).T / scale
        for solver, strict in plan:
            start = time.perf_counter()
            attempt = dict(solver=solver, strict_retry=strict)
            try:
                # Separate symbolic caches for each solver avoid invalidating
                # SCS canonicalization when a CLARABEL retry is necessary.
                problem, av, zv, Wp, Qp, mp, psd = _sdp_template(len(W), len(U), solver)
                Wp.value, Qp.value, mp.value = W / scale, fixed_Q, m
                # Candidate meanings change: fresh feasible initialization,
                # no old a-vector or numeric-factorization reuse claim.
                av.value = np.full(len(U), m / len(U))
                zv.value = float(np.linalg.eigvalsh(Wp.value)[0])
                if sdp.get("force_failure", False):
                    raise RuntimeError("Explicit failure-injection test")
                if solver == "SCS":
                    options = dict(max_iters=int(sdp.get("max_iters", 5000)) * (4 if strict else 1),
                        eps_abs=float(sdp.get("eps_abs", 1e-4)) / (20 if strict else 1),
                        eps_rel=float(sdp.get("eps_rel", 1e-4)) / (20 if strict else 1))
                elif solver == "CLARABEL":
                    options = dict(max_iter=1000, tol_gap_abs=1e-7, tol_feas=1e-7, tol_gap_rel=1e-7)
                else:
                    raise ValueError(f"Unsupported SDP solver: {solver}")
                with warnings.catch_warnings(record=True) as caught:
                    problem.solve(solver=solver, warm_start=False, enforce_dpp=True, **options)
                attempt["warnings"] = [str(w.message) for w in caught]
                attempt.update(status=problem.status, iterations=problem.solver_stats.num_iters,
                               solve_time=problem.solver_stats.solve_time,
                               compilation_time=problem.compilation_time)
                extra = problem.solver_stats.extra_stats
                info = extra.get("info", {}) if isinstance(extra, dict) else {}
                attempt.update(primal_residual=info.get("res_pri"), dual_residual=info.get("res_dual"),
                               solver_gap=info.get("gap"))
                if av.value is None or zv.value is None:
                    raise RuntimeError("Solver returned no primal solution")
                aa = np.asarray(av.value).reshape(-1)
                z = float(zv.value) * scale
                if not np.all(np.isfinite(aa)) or not np.isfinite(z):
                    raise FloatingPointError("Nonfinite SDP solution")
                equality = abs(float(aa.sum()) - m)
                bounds = max(0., -float(aa.min()), float(aa.max()) - 1.)
                fractional_matrix = W + U.T @ (aa[:, None] * U)
                psd_residual = max(0., z - float(np.linalg.eigvalsh(fractional_matrix)[0])) / scale
                projected = project_capped_simplex(aa, m)
                fractional_lb = float(np.linalg.eigvalsh(W + U.T @ (projected[:, None] * U))[0])
                dual = None if psd.dual_value is None else np.asarray(psd.dual_value)
                ub, _, _ = dual_upper_bound(W, U, m, dual)
                objective_gap = ub - fractional_lb
                external_gap = objective_gap / max(1., abs(ub), abs(fractional_lb))
                ok = (problem.status == "optimal" and equality <= feasibility_tol * max(1, m)
                      and bounds <= feasibility_tol and psd_residual <= feasibility_tol
                      and external_gap <= gap_tol and external_gap >= -1e-9)
                attempt.update(equality_residual=equality, bound_residual=bounds,
                    psd_residual_scaled=psd_residual, primal_value=z,
                    fractional_feasible_value=fractional_lb, valid_upper_bound=ub,
                    external_objective_gap=objective_gap, external_relative_gap=external_gap,
                    projection_l2=float(np.linalg.norm(projected - aa)), quality_gate_passed=ok)
                if ok:
                    raw_a, raw_dual, status = aa.copy(), dual, problem.status
            except Exception as exc:
                attempt.update(status=attempt.get("status", "ERROR"), error=f"{type(exc).__name__}: {exc}",
                               quality_gate_passed=False)
            attempt["wall_seconds"] = time.perf_counter() - start
            attempts.append(attempt)
            if raw_a is not None:
                break
    except Exception as exc:
        attempts.append(dict(status="TEMPLATE_ERROR", error=f"{type(exc).__name__}: {exc}", quality_gate_passed=False))
    if raw_a is None:
        ub, _, dual_info = dual_upper_bound(W, U, m)
        last = attempts[-1] if attempts else {}
        return greedy, dict(sdp_status="FAILED_FALLBACK_E_GREEDY", fallback_used=True,
            fallback_reason="All SDP attempts failed the external quality gate", sdp_attempts=attempts,
            sdp_scale=scale, sdp_ub=ub, sdp_lb=greedy_lb, sdp_gap=ub-greedy_lb,
            sdp_solve_iterations=last.get("iterations"),
            sdp_primal_residual=last.get("primal_residual"), sdp_dual_residual=last.get("dual_residual"),
            sdp_psd_residual=last.get("psd_residual_scaled"),
            sdp_inclusion_residual=max(last.get("bound_residual", 0.), last.get("equality_residual", 0.) / max(1, m))
                if "bound_residual" in last else None,
            sdp_eigenvalue_clipping=False,
            slot_trace=greedy_audit["slot_trace"], **dual_info)
    a = project_capped_simplex(raw_a, m)
    largest = np.lexsort((priority, -a))[:m]
    batches = [largest] + [dependent_round(a, rng) for _ in range(int(sdp.get("dependent_rounding_draws", 8)))] + [greedy]
    values = [float(np.linalg.eigvalsh(W + U[b].T @ U[b])[0]) for b in batches]
    best_index = int(np.argmax(values))
    selected = np.asarray(batches[best_index], dtype=np.int64)
    best_value = values[best_index]
    before_swap = best_value
    proposals = 0
    swap = None
    max_proposals = min(128, int(sdp.get("swap_max_proposals", 128)))
    if max_proposals and int(sdp.get("swap_max_passes", 1)):
        inside = selected[np.lexsort((priority[selected], a[selected]))[:8]]
        outside_mask = np.ones(len(U), dtype=bool)
        outside_mask[selected] = False
        outside = np.flatnonzero(outside_mask)
        outside = outside[np.lexsort((priority[outside], -a[outside]))[:16]]
        current = W + U[selected].T @ U[selected]
        for i in inside:
            for j in outside:
                if proposals >= max_proposals:
                    break
                value = float(np.linalg.eigvalsh(current - np.outer(U[i], U[i]) + np.outer(U[j], U[j]))[0])
                proposals += 1
                if value > best_value + 1e-11 * max(1., abs(best_value)):
                    best_value, swap = value, (int(i), int(j))
        if swap is not None:
            selected = selected.copy()
            selected[np.flatnonzero(selected == swap[0])[0]] = swap[1]
    lb = float(np.linalg.eigvalsh(W + U[selected].T @ U[selected])[0])
    ub, _, dual_info = dual_upper_bound(W, U, m, raw_dual)
    if lb + 1e-8 * max(1., abs(lb)) < greedy_lb:
        raise FloatingPointError("E-SDP rounding worse than its E-greedy candidate")
    if ub < lb - 1e-8 * max(1., abs(lb), abs(ub)):
        raise FloatingPointError("E-SDP valid upper bound below integer objective")
    last = attempts[-1]
    return selected, dict(sdp_status=status, fallback_used=False, fallback_reason=None,
        sdp_attempts=attempts, sdp_scale=scale, sdp_dpp=True,
        sdp_eigenvalue_clipping=False,
        sdp_fractional_weights_raw=raw_a.tolist(), sdp_fractional_weights=a.tolist(),
        sdp_projection_l2=float(np.linalg.norm(a - raw_a)),
        sdp_feasibility_tolerance=feasibility_tol, sdp_objective_gap_tolerance=gap_tol,
        sdp_rounding_values=values, sdp_rounding_winner=best_index,
        sdp_rounding_candidate_positions=[b.tolist() for b in batches],
        sdp_swap_proposals=proposals, sdp_swap=swap, sdp_before_swap_lb=before_swap,
        sdp_ub=ub, sdp_lb=lb, sdp_gap=ub-lb,
        sdp_fractional_value=last["fractional_feasible_value"],
        sdp_rounding_gap=last["fractional_feasible_value"]-lb,
        sdp_solve_iterations=last.get("iterations"),
        sdp_primal_residual=last.get("primal_residual"), sdp_dual_residual=last.get("dual_residual"),
        sdp_psd_residual=last["psd_residual_scaled"],
        sdp_inclusion_residual=max(last["bound_residual"], last["equality_residual"] / max(1, m)),
        slot_trace=[], **dual_info)


def select_batch(method, X_pool, labeled_ids, candidates, theta, family, m, rho, rng,
                 config=None, batch_index=1):
    """Return selected row IDs and replayable audit; reads no response values."""
    config = {} if config is None else config
    if method not in METHODS and method not in ("fb_range_legacy_m1", "fb_no_virtual"):
        raise ValueError(f"Unknown selector {method}")
    if method == "badge" and family not in ("bernoulli", "logistic"):
        raise ValueError("BADGE-logistic is NOT_APPLICABLE outside Bernoulli")
    X = np.asarray(X_pool, dtype=float)
    ids = np.asarray(candidates, dtype=np.int64)
    labeled = np.asarray(labeled_ids, dtype=np.int64)
    theta = np.asarray(theta, dtype=float)
    if not 0 < m <= len(ids) or len(np.unique(ids)) != len(ids) or len(np.intersect1d(labeled, ids)):
        raise ValueError("Candidates must be unique unqueried rows, with m <= K")
    if ids.min() < 0 or ids.max() >= len(X) or rho <= 0 or theta.shape != (X.shape[1],):
        raise ValueError("Invalid pool/parameter/ridge input")
    t0 = time.perf_counter()
    floor = config.get("acquisition_floor", 0.) or 0.
    if isinstance(floor, bool):
        floor = 1e-12 if floor else 0.
    floor = float(floor)
    if floor < 0:
        raise ValueError("Negative acquisition curvature floor")
    raw = curvature(X[ids] @ theta, family)
    weights = np.maximum(raw, floor)
    old_raw = curvature(X[labeled] @ theta, family)
    old_weights = np.maximum(old_raw, floor)
    U = X[ids] * np.sqrt(weights)[:, None]
    W = rho * np.eye(X.shape[1]) + X[labeled].T @ (old_weights[:, None] * X[labeled])
    W = (W + W.T) * .5
    opt_scale = opt_tau = None
    if method == "fb_opt":
        # The runner caches this feature-only constant once per original pool.
        opt_scale = config.get("_fb_opt_fixed_scale")
        if opt_scale is None:
            if 'fb_opt_Tmax' not in config:
                raise ValueError("FB-opt needs the planned finite horizon fb_opt_Tmax")
            opt_scale = fb_opt_scale(X, family, m, config['fb_opt_Tmax'])
        if opt_scale['method_version'] != 'fb_opt_v2_finite_horizon':
            raise ValueError("Obsolete FB-opt calibration cannot be used for acquisition")
        if opt_scale['m'] != m or opt_scale['d'] != X.shape[1] or not 1 <= batch_index <= opt_scale['Tmax']:
            raise ValueError("FB-opt batch/dimension/round disagrees with its fixed horizon")
        opt_tau = float(opt_scale['tau0_opt']) * (m * batch_index)**.75
    elif method == "fb_opt2":
        opt_scale = config.get('_fb_opt2_fixed_scale')
        if opt_scale is None:
            if 'fb_opt2_Tmax' not in config:
                raise ValueError('FB-opt2 needs the planned horizon fb_opt2_Tmax')
            opt_scale = fb_opt2_scale(family, X.shape[1], m, config['fb_opt2_Tmax'], config)
        if opt_scale['method_version'] != 'fb_opt2_v1_range_anchored' or opt_scale['m'] != m or opt_scale['d'] != X.shape[1]:
            raise ValueError('FB-opt2 calibration disagrees with the method/batch/dimension')
        opt_tau = fb_opt2_temperature(opt_scale, batch_index)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(U)):
        raise FloatingPointError("Nonfinite acquisition curvature/information")
    priority = stable_priorities(ids, "tie", config.get("dataset", ""),
                                 int(config.get("seed", 0)), int(batch_index))
    audit = dict(method=method, batch_index=int(batch_index), candidate_ids=ids.tolist(),
        candidate_count=len(ids), batch_size=int(m), frozen_theta=theta.tolist(),
        candidate_weights_raw=raw.tolist(), candidate_weights_acquisition=weights.tolist(),
        W_start=W.tolist(), rng_state_before=copy.deepcopy(rng.bit_generator.state),
        curvature_floor=floor, candidate_curvature_floor_count=int(np.sum(raw < floor)),
        labeled_curvature_floor_count=int(np.sum(old_raw < floor)),
        fallback_used=False, fallback_reason=None, backend=config.get("e_backend", "secular")
        if method in ("e_greedy", "e_sdp") else "numpy_cpu", slot_trace=[])
    valid = np.ones(len(ids), dtype=bool)
    selected = []
    if method == "random":
        selected = rng.choice(len(ids), size=m, replace=False)
    elif method == "ts_glm":
        L = np.linalg.cholesky(W)
        standard = rng.standard_normal(X.shape[1])
        draw = theta + float(config.get("alpha", 1.)) * solve_triangular(L.T, standard, lower=False)
        scores = X[ids] @ draw
        selected = np.lexsort((priority, -scores))[:m]
        audit.update(theta_tilde=draw.tolist(), thompson_draws_per_batch=1,
                     thompson_standard_draw=standard.tolist())
    elif method == "badge":
        p = expit(X[ids] @ theta)
        G = (p - (p >= .5))[:, None] * X[ids]
        norm = np.einsum("ij,ij->i", G, G)
        nearest = np.full(len(ids), np.inf)
        zero_fallbacks = 0
        for slot in range(m):
            uniform_draw = None
            if slot == 0:
                j, ties = _winner(norm, valid, priority)
                mass = float(norm[j])
            else:
                nearest[~valid] = 0.
                mass = float(nearest.sum())
                if mass <= 0.:
                    j = int(rng.choice(np.flatnonzero(valid)))
                    zero_fallbacks += 1
                else:
                    uniform_draw = float(rng.random())
                    j = int(np.searchsorted(np.cumsum(nearest), uniform_draw * mass, side="right"))
                    if j >= len(ids) or not valid[j]:
                        raise FloatingPointError("Invalid D-squared sampling draw")
                ties = 0
            selected.append(j)
            valid[j] = False
            distance = np.einsum("ij,ij->i", G - G[j], G - G[j])
            nearest = np.minimum(nearest, distance)
            audit["slot_trace"].append(dict(slot=slot, candidate_position=j, distance_mass=mass,
                                            d2_uniform_draw=uniform_draw, numerical_ties=ties))
        audit.update(badge_zero_distance_fallbacks=zero_fallbacks,
                     badge_seeding="single_draw_D_squared_kmeansplusplus")
    elif method == "e_greedy":
        selected, extra = _e_greedy(W, U, m, priority, config)
        audit.update(extra)
    elif method == "e_sdp":
        selected, extra = _e_sdp(W, U, m, priority, rng, config)
        audit.update(extra)
    elif method == "bait_glm":
        full_weights = curvature(X @ theta, family)
        F = X.T @ (full_weights[:, None] * X) / len(X)
        cache = InverseCache(W, U, F)
        provisional, removed = [], []
        for slot in range(min(2*m, len(ids))):
            scores = cache.r / (1. + cache.q)
            j, ties = _winner(scores, valid, priority)
            valid[j] = False
            provisional.append(j)
            audit["slot_trace"].append(dict(phase="forward", slot=slot, candidate_position=j,
                                            objective_change=-float(scores[j]), numerical_ties=ties))
            cache.update(j)
        keep = ~valid
        while int(keep.sum()) > m:
            denominator = 1. - cache.q
            safe = denominator > 1e-10
            scores = np.full(len(U), np.inf)
            scores[safe] = cache.r[safe] / denominator[safe]
            # Direct objective differences are mandatory near singular downdates.
            for j in np.flatnonzero(keep & ~safe):
                reduced = W + U[np.flatnonzero(keep & (np.arange(len(U)) != j))].T @ U[np.flatnonzero(keep & (np.arange(len(U)) != j))]
                scores[j] = float(np.trace((_inverse(reduced) - cache.A) @ F))
            j, ties = _winner(scores, keep, priority, maximize=False)
            keep[j] = False
            removed.append(j)
            audit["slot_trace"].append(dict(phase="backward", slot=len(removed)-1, candidate_position=j,
                                            objective_change=float(scores[j]), numerical_ties=ties))
            cache.update(j, sign=-1)
        selected = np.asarray([j for j in provisional if keep[j]])
        audit.update(bait_normalization="matched_sum_information", F_pool=F.tolist(),
            bait_target_curvature="exact_working_model",
            bait_provisional_ids=ids[provisional].tolist(), bait_removed_ids=ids[removed].tolist(),
            bait_final_objective=float(np.trace(_inverse(W + U[selected].T @ U[selected]) @ F)),
            inverse_rebuilds=cache.rebuilds, inverse_max_error=cache.max_inverse_error)
    else:
        cache = InverseCache(W, U) if method in ("a_opt", "d_opt", "ucb_glm") else None
        virtual = W.copy()
        natural = X[ids] @ theta
        for slot in range(m):
            entropy = tau = None
            if method == "a_opt":
                scores = cache.r / (1. + cache.q)
            elif method == "d_opt":
                scores = np.log1p(cache.q)
            elif method == "ucb_glm":
                uncertainty = np.einsum("ij,ij->i", X[ids] @ cache.A, X[ids])
                if uncertainty.min() < -1e-9:
                    raise FloatingPointError("Negative UCB variance")
                scores = natural + float(config.get("alpha", 1.)) * np.sqrt(np.maximum(uncertainty, 0.))
            else:
                lam, Q = np.linalg.eigh(virtual)
                if method in ("fb_opt", "fb_opt2"):
                    tau = opt_tau
                elif method in ("fb_sample", "fb_no_virtual"):
                    # len(labeled)+actual m handles partial final batches too.
                    tau = float(config.get("tau0", config.get("fb_tau0", .01))) * (len(labeled) + m)**.75
                elif method == "fb_range_legacy_m1" and m == 1:
                    tau = 0.
                else:
                    tau, _ = fb_range_temperature(family, m, X.shape[1], config, U, valid)
                if tau == 0. and method == "fb_range_legacy_m1":
                    spectral = np.zeros(len(lam))
                    spectral[0] = 1.
                else:
                    if tau <= 0:
                        raise ValueError("FisherBalancing temperature must be positive")
                    spectral = np.exp(-(lam - lam[0]) / tau)
                    spectral /= spectral.sum()
                positive = spectral > 0
                entropy = float(-np.sum(spectral[positive] * np.log(spectral[positive])))
                scores = (U @ Q)**2 @ spectral
            j, ties = _winner(scores, valid, priority)
            valid[j] = False
            selected.append(j)
            row = dict(slot=slot, candidate_position=j, score=float(scores[j]), numerical_ties=ties)
            if tau is not None:
                row.update(temperature=tau, spectral_entropy=entropy, eigenvalues=lam.tolist())
            audit["slot_trace"].append(row)
            if cache is not None:
                cache.update(j)
            elif method != "fb_no_virtual":
                virtual += np.outer(U[j], U[j])
                virtual = (virtual + virtual.T) * .5
        if cache is not None:
            audit.update(inverse_rebuilds=cache.rebuilds, inverse_max_error=cache.max_inverse_error)
    selected = np.asarray(selected, dtype=np.int64)
    result = ids[selected]
    if method == "fb_opt":
        audit.update(fb_opt=opt_scale, temperature=opt_tau,
                     effective_response_count=m * batch_index)
    elif method == "fb_opt2":
        audit.update(fb_opt2=opt_scale, temperature=opt_tau,
                     effective_response_count=m * batch_index)
    if len(result) != m or len(np.unique(result)) != m:
        raise RuntimeError("Selector violated exact unique label budget")
    audit.update(selected_ids=result.tolist(), selected_candidate_positions=selected.tolist(),
                 selected_weights_raw=raw[selected].tolist(), selected_weights_acquisition=weights[selected].tolist(),
                 W_end=(W + U[selected].T @ U[selected]).tolist(),
                 rng_state_after=copy.deepcopy(rng.bit_generator.state),
                 acquisition_seconds=time.perf_counter() - t0)
    return result, audit
