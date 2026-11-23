"""Independent post-hoc reconstruction of frozen acquisition decisions.

No responses or evaluation reference enter replay. Direct small-matrix
formulas deliberately avoid the selector's inverse caches and secular solver.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.special import expit


def replay_batch(X_pool, audit, family, rho=1., labeled_ids=None, config=None):
    config = config or {}
    X = np.asarray(X_pool, dtype=float)
    ids = np.asarray(audit["candidate_ids"], dtype=int)
    selected = np.asarray(audit["selected_ids"], dtype=int)
    theta = np.asarray(audit.get("frozen_theta", audit.get("pre_batch_theta")))
    m = len(selected)
    checked = 0
    max_error = 0.

    def check(actual, expected, label, atol=2e-8, rtol=2e-8):
        nonlocal checked, max_error
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(f"Replay {label}: invalid shapes or nonfinite values")
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        max_error = max(max_error, error)
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            raise AssertionError(f"Replay {label}: mismatch, maximum error {error}")
        checked += 1

    def weights(features):
        eta = features @ theta
        if family in ("bernoulli", "logistic"):
            return expit(eta) * expit(-eta)
        if family in ("gaussian", "normal"):
            return np.ones(len(features))
        if family == "poisson":
            return np.exp(eta)
        raise ValueError(f"Unknown family {family}")

    if len(np.unique(ids)) != len(ids) or len(np.unique(selected)) != m:
        raise AssertionError("Replay duplicate candidate/selected ID")
    lookup = {int(row_id): j for j, row_id in enumerate(ids)}
    positions = np.array([lookup[int(i)] for i in selected], dtype=int)
    if labeled_ids is not None and set(selected) & set(labeled_ids):
        raise AssertionError("Replay repeated query of old row")
    raw = weights(X[ids])
    acquired = np.maximum(raw, float(audit.get("curvature_floor", 0.)))
    check(audit["candidate_weights_raw"], raw, "exact frozen candidate curvature")
    check(audit["candidate_weights_acquisition"], acquired, "acquisition curvature")
    U = X[ids] * np.sqrt(acquired)[:, None]
    base = np.asarray(audit["W_start"], dtype=float)
    if labeled_ids is not None:
        old = X[np.asarray(labeled_ids, dtype=int)]
        expected = rho*np.eye(X.shape[1]) + old.T @ (np.maximum(weights(old), float(audit.get("curvature_floor", 0.)))[:, None] * old)
        check(base, expected, "all old rows reevaluated at frozen theta")
    end = base + U[positions].T @ U[positions]
    check(audit["W_end"], end, "end information reconstruction")
    if "selected_candidate_positions" in audit:
        check(audit["selected_candidate_positions"], positions, "selected position correspondence", 0., 0.)
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(audit["rng_state_before"])
    method = audit["method"]
    effective = "e_greedy" if method == "e_sdp" and audit.get("fallback_used") else method
    trace = audit.get("slot_trace", [])
    W = base.copy()
    remaining = np.ones(len(ids), dtype=bool)
    provisional = np.zeros(len(ids), dtype=bool)
    replay_positions = []
    if effective == "random":
        expected = rng.choice(len(ids), size=m, replace=False)
        check(positions, expected, "uniform sample and RNG replay", 0., 0.)
    elif effective == "ts_glm":
        normal = rng.standard_normal(X.shape[1])
        draw = theta + float(config.get("alpha", 1.)) * np.linalg.solve(np.linalg.cholesky(base).T, normal)
        check(audit["thompson_standard_draw"], normal, "Thompson standard-normal draw")
        check(audit["theta_tilde"], draw, "Thompson covariance orientation")
        score = X[ids] @ draw
        if np.min(score[positions]) < np.sort(score)[-m] - 1e-9:
            raise AssertionError("Replay Thompson selected rows are not top m")
    elif effective == "e_sdp":
        P = np.asarray(audit["feasible_dual_P"])
        check(np.trace(P), 1., "SDP dual trace")
        if np.linalg.eigvalsh(P)[0] < -1e-9:
            raise AssertionError("Replay SDP dual not PSD")
        dual_scores = np.sum((U @ P)*U, axis=1)
        upper = float(np.trace(P @ base) + np.sort(dual_scores)[-m:].sum())
        lower = float(np.linalg.eigvalsh(end)[0])
        check(audit["sdp_ub"], upper, "SDP valid dual upper bound")
        check(audit["sdp_lb"], lower, "SDP queried integer objective")
        check(audit["sdp_gap"], upper-lower, "SDP valid bound gap")
        if lower > upper + 1e-8:
            raise AssertionError("Replay SDP lower bound exceeds upper bound")
        a = np.asarray(audit["sdp_fractional_weights"])
        check(a.sum(), m, "SDP capped-simplex budget")
        if a.min() < -1e-8 or a.max() > 1+1e-8:
            raise AssertionError("Replay SDP weights violate caps")
        check(audit["sdp_fractional_value"], np.linalg.eigvalsh(base + U.T @ (a[:,None]*U))[0], "SDP fractional objective")
        batches = audit["sdp_rounding_candidate_positions"]
        for k, batch in enumerate(batches):
            b = np.asarray(batch, dtype=int)
            if len(b) != m or len(set(b)) != m:
                raise AssertionError("Replay SDP rounding batch cardinality")
            check(audit["sdp_rounding_values"][k], np.linalg.eigvalsh(base + U[b].T @ U[b])[0], "SDP rounding candidate objective")
        if lower < max(audit["sdp_rounding_values"]) - 1e-8:
            raise AssertionError("Replay SDP rounding failed best-candidate dominance")
    elif effective == "badge":
        p = expit(X[ids] @ theta)
        G = (p - (p >= .5))[:, None] * X[ids]
        nearest = np.full(len(ids), np.inf)
        for slot, row in enumerate(trace):
            j = int(row["candidate_position"])
            if not remaining[j]:
                raise AssertionError("Replay BADGE duplicate center")
            if slot == 0:
                norm = np.sum(G*G, axis=1)
                check(norm[j], norm.max(), "BADGE maximum first gradient norm")
                check(row["distance_mass"], norm[j], "BADGE first score")
            else:
                nearest[~remaining] = 0.
                mass = float(nearest.sum())
                check(row["distance_mass"], mass, "BADGE D-squared total mass")
                if mass <= 0.:
                    expected = int(rng.choice(np.flatnonzero(remaining)))
                else:
                    draw = float(rng.random())
                    check(row["d2_uniform_draw"], draw, "BADGE random variate")
                    expected = int(np.searchsorted(np.cumsum(nearest), draw*mass, side="right"))
                check(j, expected, "BADGE stochastic center replay", 0., 0.)
            remaining[j] = False
            replay_positions.append(j)
            nearest = np.minimum(nearest, np.sum((G-G[j])**2, axis=1))
    else:
        for row in trace:
            j = int(row["candidate_position"])
            phase = row.get("phase", "forward")
            if phase == "backward":
                if not provisional[j]:
                    raise AssertionError("Replay BAIT removes nonprovisional row")
                F = np.asarray(audit["F_pool"])
                before = float(np.trace(np.linalg.solve(W, F)))
                score = np.full(len(U), np.inf)
                for k in np.flatnonzero(provisional):
                    score[k] = np.trace(np.linalg.solve(W-np.outer(U[k],U[k]), F)) - before
                check(row["objective_change"], score[j], "BAIT backward exact objective change")
                check(score[j], score[provisional].min(), "BAIT best removal")
                provisional[j] = False
                W -= np.outer(U[j],U[j])
                continue
            if not remaining[j]:
                raise AssertionError("Replay repeated forward candidate")
            if effective in ("a_opt", "d_opt", "ucb_glm", "bait_glm"):
                solved = np.linalg.solve(W, U.T).T
                leverage = np.sum(solved*U, axis=1)
                if effective == "a_opt":
                    score = np.sum(solved*solved, axis=1)/(1+leverage)
                elif effective == "d_opt":
                    score = np.log1p(leverage)
                elif effective == "ucb_glm":
                    xx = X[ids]
                    score = xx@theta + float(config.get("alpha",1.))*np.sqrt(np.maximum(np.sum(np.linalg.solve(W,xx.T).T*xx,axis=1),0.))
                else:
                    F = X.T @ (weights(X)[:,None]*X) / len(X)
                    check(audit["F_pool"], F, "BAIT full-pool label-free target")
                    score = np.sum((solved@F)*solved,axis=1)/(1+leverage)
            elif effective == "e_greedy":
                score = np.linalg.eigvalsh(W + U[:,:,None]*U[:,None,:])[:,0]
            elif effective.startswith("fb_"):
                lam,Q = np.linalg.eigh(W)
                n_old = len(labeled_ids) if labeled_ids is not None else int(config.get("n0",80)) + (int(audit["batch_index"])-1)*m
                if effective == "fb_opt2":
                    R = 1. if family in ('gaussian','normal') else .25
                    tau_range = max(1e-8,float(config.get('temperature_multiplier',config.get('fb_range_multiplier',1.)))*R*np.sqrt(m/(8.*np.log(max(2,X.shape[1])))))
                    Nmax = m*int(audit['fb_opt2']['Tmax'])
                    check(audit['fb_opt2']['tau_range'],tau_range,'FB-opt2 existing range anchor')
                    check(audit['fb_opt2']['tau0_opt2'],tau_range/Nmax**.75,'FB-opt2 coefficient')
                    tau = tau_range*((m*int(audit['batch_index']))/Nmax)**.75
                elif effective == "fb_opt":
                    w_bar = 1. if family in ("gaussian", "normal") else .25
                    radius = float(np.linalg.norm(X, axis=1).max())
                    B = w_bar * radius**2
                    d = X.shape[1]
                    if d <= 1:raise ValueError('FB-opt replay requires d > 1')
                    Tmax = int(audit['fb_opt']['Tmax'])
                    # Independent Python finite sum; no use of selector calibration helper.
                    finite_sum = sum((m*t)**(-.75) for t in range(1,Tmax+1))
                    tau0 = B * np.sqrt(m*finite_sum/((m*Tmax)**.75*np.log(d)))
                    check(audit['fb_opt']['S_T'], finite_sum, 'FB-opt exact finite sum')
                    check(audit['fb_opt']['L'], radius, 'FB-opt full-pool radius')
                    check(audit['fb_opt']['B'], B, 'FB-opt information bound')
                    check(audit['fb_opt']['tau0_opt'], tau0, 'FB-opt coefficient')
                    tau = tau0 * (m * int(audit['batch_index']))**.75
                elif effective in ("fb_sample", "fb_no_virtual"):
                    tau = float(config.get("tau0",config.get("fb_tau0",.01)))*(n_old+m)**.75
                elif effective == "fb_range_legacy_m1" and m == 1:
                    tau = 0.
                else:
                    R = 1. if family in ("gaussian","normal") else .25
                    if family == "poisson":
                        R = float(np.max(np.sum(U[remaining]**2,axis=1)))
                    tau = max(1e-8,float(config.get("temperature_multiplier",config.get("fb_range_multiplier",1.)))*R*np.sqrt(m/(8*np.log(max(2,X.shape[1])))))
                check(row["temperature"],tau,"FB temperature schedule")
                if tau == 0.:
                    p=np.zeros(len(lam));p[0]=1.
                else:
                    p=np.exp(-(lam-lam[0])/tau);p/=p.sum()
                positive = p>0
                check(row["spectral_entropy"],-np.sum(p[positive]*np.log(p[positive])),"FB Gibbs entropy")
                check(row["eigenvalues"],lam,"FB virtual information spectrum")
                score = (U@Q)**2@p
            else:
                raise ValueError(f"Replay unsupported method {effective}")
            saved = -row["objective_change"] if effective=="bait_glm" else row["score"]
            check(saved,score[j],"selected score")
            check(score[j],score[remaining].max(),"exact best available objective")
            remaining[j]=False;provisional[j]=True;replay_positions.append(j)
            if effective != "fb_no_virtual":
                W+=np.outer(U[j],U[j])
                W=(W+W.T)*.5
        if effective == "bait_glm":
            replay_positions=[j for j in replay_positions if provisional[j]]
            check(audit["bait_final_objective"],np.trace(np.linalg.solve(W,np.asarray(audit["F_pool"]))),"BAIT final objective")
    if effective not in ("random","ts_glm","e_sdp"):
        check(positions,np.asarray(replay_positions,dtype=int),"ordered selected IDs",0.,0.)
    if effective in ("random","ts_glm","badge"):
        if rng.bit_generator.state != audit["rng_state_after"]:
            raise AssertionError("Replay random stream end state differs")
        checked += 1
    return dict(verified=True,method=method,batch_index=int(audit["batch_index"]),
                comparisons=checked,maximum_absolute_error=max_error,
                slot_records=len(trace),queried_rows=m)
