"""Float64, exact sum-likelihood GLMs with a penalized intercept.

Only acquisition code may floor working curvature. Neither this module's
objectives nor its derivatives clip linear predictors or responses.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, gammaln


def quantities(eta, family):
    eta = np.asarray(eta, dtype=np.float64)
    if family == "bernoulli":
        mu = expit(eta)
        return mu, mu * expit(-eta)
    if family == "gaussian":
        return eta, np.ones_like(eta)
    if family == "poisson":
        with np.errstate(over="ignore", under="ignore"):
            mu = np.exp(eta)
        return mu, mu
    raise ValueError(f"Unknown GLM family: {family}")


def loss_values(eta, y, family, constants=False):
    eta, y = np.asarray(eta, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if family == "bernoulli":
        # This equivalent form avoids cancellation for correctly classified
        # large positive predictors, without clipping the likelihood.
        return np.logaddexp(0.0, np.where(y == 1, -eta, eta))
    if family == "gaussian":
        out = 0.5 * (eta - y) ** 2
        return out + (0.5 * np.log(2.0 * np.pi) if constants else 0.0)
    if family == "poisson":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            out = np.exp(eta) - y * eta
        return out + gammaln(y + 1.0) if constants else out
    raise ValueError(f"Unknown GLM family: {family}")


def objective(theta, X, y, family, rho=1.0):
    return float(np.sum(loss_values(X @ theta, y, family)) + 0.5 * rho * np.dot(theta, theta))


def derivatives(theta, X, y, family, rho=1.0):
    eta = X @ theta
    mu, w = quantities(eta, family)
    f = float(np.sum(loss_values(eta, y, family)) + 0.5 * rho * np.dot(theta, theta))
    g = X.T @ (mu - y) + rho * theta
    H = (X.T * w) @ X + rho * np.eye(X.shape[1])
    return f, g, (H + H.T) * 0.5


def _scaled_gradient(g, Xty, theta, rho):
    numerator = float(np.linalg.norm(g, ord=np.inf))
    denominator = float(1.0 + np.linalg.norm(Xty, ord=np.inf) + rho * np.linalg.norm(theta, ord=np.inf))
    return numerator, denominator, numerator / denominator


def fit_audit_at_theta(theta, X, y, family, rho=1.0):
    """Recompute objective/stationarity at saved theta without optimization.

    Original solver iteration/timing fields cannot be recovered from theta;
    callers should overlay these recomputed fields on the saved fit audit.
    """
    theta, X, y = map(lambda a: np.asarray(a, dtype=np.float64), [theta, X, y])
    f, g, H = derivatives(theta, X, y, family, rho)
    gi, gs, gr = _scaled_gradient(g, X.T @ y, theta, rho)
    eta = X @ theta
    _, w = quantities(eta, family)
    ev = np.linalg.eigvalsh(H)
    loss = float(np.sum(loss_values(eta, y, family)))
    penalty = float(0.5 * rho * (theta @ theta))
    try:
        decrement = float(g @ cho_solve(cho_factor(H, lower=True, check_finite=False), g, check_finite=False))
    except (np.linalg.LinAlgError, ValueError):
        decrement = None
    return dict(
        fit_n=len(y), rho_sum=float(rho), rho_effective_mean=float(rho / len(y)),
        train_loss_sum=loss, train_loss_mean=loss / len(y),
        train_nll_mean_with_constants=float(np.mean(loss_values(eta, y, family, constants=True))),
        ridge_penalty=penalty, train_penalized_objective_sum=f,
        fit_grad_inf=gi, fit_grad_scale=gs, fit_grad_relative=gr,
        fit_newton_decrement_squared=decrement, fit_converged=bool(gr <= 1e-8),
        fit_status="STATIONARY" if gr <= 1e-8 else "SAVED_PARAMETER_NOT_STATIONARY",
        fit_eta_min=float(eta.min()), fit_eta_max=float(eta.max()),
        fit_curvature_min=float(w.min()), fit_curvature_max=float(w.max()),
        fit_hessian_condition=float(ev[-1] / ev[0]), theta_norm_l2=float(np.linalg.norm(theta)),
        loss_convention="squared_residual_half" if family == "gaussian" else "canonical_constants_omitted",
        intercept_penalized=True,
    )


def fit_glm(X, y, family, rho=1.0, theta0=None, reference=False,
            max_iter=None, tolerance=None):
    """Fit the exact stated objective; convergence requires gradient residual.

    Returns a float64 vector and a JSON-serializable fit audit. Failed Newton
    attempts are retained in ``fit_attempts`` before same-objective retries.
    A false ``fit_converged`` must be treated as a failed scientific fit.
    """
    started = time.perf_counter()
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (len(X),) or not len(X):
        raise ValueError("Expected nonempty X[n,d] and y[n]")
    if rho <= 0 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Positive ridge and finite features/responses are required")
    if family == "bernoulli" and not np.all((y == 0) | (y == 1)):
        raise ValueError("Bernoulli responses must be recorded 0/1 labels")
    if family == "poisson" and (np.any(y < 0) or np.any(y != np.floor(y))):
        raise ValueError("Poisson responses must be nonnegative integer counts")
    limit = int(max_iter if max_iter is not None else (200 if reference else 50))
    tol = float(tolerance if tolerance is not None else (1e-9 if reference else 1e-8))
    d, Xty = X.shape[1], X.T @ y
    theta = np.zeros(d) if theta0 is None else np.array(theta0, dtype=np.float64, copy=True)
    if theta.shape != (d,) or not np.all(np.isfinite(theta)):
        raise ValueError("Invalid starting parameter")
    iterations = trials = retries = 0
    final_step = 0.0
    attempts = []
    solver = "cholesky" if family == "gaussian" else "damped_newton_cholesky_armijo"

    def newton(start, budget):
        nonlocal iterations, trials, final_step
        th = start.copy()
        status = "MAX_ITERATIONS"
        for _ in range(budget):
            f, g, H = derivatives(th, X, y, family, rho)
            if _scaled_gradient(g, Xty, th, rho)[2] <= tol:
                return th, "STATIONARY"
            if not np.isfinite(f) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(H)):
                return th, "NONFINITE_CURRENT_STATE"
            try:
                direction = cho_solve(cho_factor(H, lower=True, check_finite=False), g, check_finite=False)
            except (np.linalg.LinAlgError, ValueError):
                return th, "CHOLESKY_FAILED"
            descent = float(g @ direction)
            if not np.isfinite(descent) or descent < 0:
                return th, "INVALID_NEWTON_DIRECTION"
            step = 1.0
            accepted = False
            # A tiny rounding allowance is used only in objective comparison;
            # stationarity tolerance is never relaxed.
            roundoff = 8 * np.finfo(float).eps * (1 + abs(f))
            for _line in range(64):
                trials += 1
                candidate = th - step * direction
                candidate_f = objective(candidate, X, y, family, rho)
                if np.isfinite(candidate_f) and candidate_f <= f - 1e-4 * step * descent + roundoff:
                    final_step = float(np.linalg.norm(step * direction))
                    th = candidate
                    accepted = True
                    break
                step *= 0.5
            iterations += 1
            if not accepted:
                return th, "LINE_SEARCH_FAILED"
        return th, status

    if family == "gaussian":
        H = X.T @ X + rho * np.eye(d)
        theta = cho_solve(cho_factor(H, lower=True, check_finite=False), Xty, check_finite=False)
        iterations = 1
        status = "STATIONARY"
    else:
        theta, status = newton(theta, limit)
        f, g, _ = derivatives(theta, X, y, family, rho)
        attempts.append({"solver": solver, "status": status, "iterations": iterations,
                         "objective": f, "grad_relative": _scaled_gradient(g, Xty, theta, rho)[2]})
        if _scaled_gradient(g, Xty, theta, rho)[2] > tol:
            retries = 1
            # BFGS uses the identical objective and analytic gradient. A final
            # Newton polish verifies the prescribed scaled stationarity gate.
            restart = theta if np.isfinite(f) else np.zeros(d)
            def fun(th):
                ff, gg, _ = derivatives(th, X, y, family, rho)
                return ff, gg
            result = minimize(fun, restart, jac=True, method="BFGS",
                              options={"gtol": tol * 0.01, "maxiter": max(200, limit * 4)})
            iterations += int(result.nit)
            attempts.append({"solver": "scipy_BFGS", "status": str(result.message),
                             "iterations": int(result.nit), "objective": float(result.fun)})
            theta, status = newton(np.asarray(result.x), limit)
            solver += "+BFGS_retry+Newton_polish"

    f, g, H = derivatives(theta, X, y, family, rho)
    grad_inf, grad_scale, grad_relative = _scaled_gradient(g, Xty, theta, rho)
    converged = bool(np.isfinite(f) and np.all(np.isfinite(theta)) and grad_relative <= tol)
    eta = X @ theta
    _, w = quantities(eta, family)
    ev = np.linalg.eigvalsh(H)
    try:
        decrement = float(g @ cho_solve(cho_factor(H, lower=True, check_finite=False), g, check_finite=False))
        decrement_reason = None
    except (np.linalg.LinAlgError, ValueError):
        decrement, decrement_reason = None, "CHOLESKY_FAILED"
    loss_sum = float(np.sum(loss_values(eta, y, family)))
    penalty = float(0.5 * rho * (theta @ theta))
    audit = dict(
        fit_n=len(y), rho_sum=float(rho), rho_effective_mean=float(rho / len(y)),
        train_loss_sum=loss_sum, train_loss_mean=loss_sum / len(y),
        train_nll_mean_with_constants=float(np.mean(loss_values(eta, y, family, constants=True))),
        ridge_penalty=penalty, train_penalized_objective_sum=loss_sum + penalty,
        fit_grad_inf=grad_inf, fit_grad_scale=grad_scale, fit_grad_relative=grad_relative,
        fit_newton_decrement_squared=decrement,
        fit_newton_decrement_na_reason=decrement_reason,
        fit_final_step_norm=final_step, fit_iterations=iterations,
        fit_line_search_trials=trials, fit_retry_count=retries,
        fit_converged=converged, fit_status="STATIONARY" if converged else status,
        fit_solver=solver, fit_seconds=time.perf_counter() - started,
        fit_eta_min=float(np.min(eta)), fit_eta_max=float(np.max(eta)),
        fit_curvature_min=float(np.min(w)), fit_curvature_max=float(np.max(w)),
        fit_hessian_condition=float(ev[-1] / ev[0]), theta_norm_l2=float(np.linalg.norm(theta)),
        fit_attempts=attempts, fit_gradient_tolerance=tol,
        loss_convention="squared_residual_half" if family == "gaussian" else "canonical_constants_omitted",
        intercept_penalized=True,
    )
    return theta, audit
