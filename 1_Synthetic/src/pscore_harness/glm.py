from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import expit
from scipy.stats import poisson as poisson_distribution

from .utils import ensure_finite


FamilyName = Literal["gaussian", "logistic", "poisson"]


@dataclass(frozen=True)
class GLMFamily:
    name: FamilyName
    noise_std: float = 1.0
    logistic_clip: float = 30.0
    poisson_intercept: float = -0.5
    poisson_clip: float = 30.0

    @property
    def max_curvature(self) -> float:
        if self.name == "gaussian":
            return 1.0 / (self.noise_std**2)
        if self.name == "logistic":
            return 0.25
        if self.name == "poisson":
            # Unbounded globally; Soft-PScore uses the current candidate scale.
            return float("inf")
        raise ValueError(f"Unsupported family: {self.name}")

    def mean(self, linear_predictor: np.ndarray) -> np.ndarray:
        z = np.asarray(linear_predictor, dtype=np.float64)
        if self.name == "gaussian":
            return z
        if self.name == "logistic":
            return expit(np.clip(z, -self.logistic_clip, self.logistic_clip))
        if self.name == "poisson":
            return np.exp(np.clip(self.poisson_intercept + z, -self.poisson_clip, self.poisson_clip))
        raise ValueError(f"Unsupported family: {self.name}")

    def curvature(self, linear_predictor: np.ndarray) -> np.ndarray:
        z = np.asarray(linear_predictor, dtype=np.float64)
        if self.name == "gaussian":
            return np.full_like(z, 1.0 / (self.noise_std**2), dtype=np.float64)
        if self.name == "logistic":
            p = expit(np.clip(z, -self.logistic_clip, self.logistic_clip))
            return np.maximum(p * (1.0 - p), 1e-10)
        if self.name == "poisson":
            return np.maximum(self.mean(z), 1e-12)
        raise ValueError(f"Unsupported family: {self.name}")

    def sample_from_common_randomness(
        self,
        linear_predictor: np.ndarray,
        common_randomness: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(linear_predictor, dtype=np.float64)
        u = np.asarray(common_randomness, dtype=np.float64)
        if self.name == "gaussian":
            return z + self.noise_std * u
        if self.name == "logistic":
            return (u < self.mean(z)).astype(np.float64)
        if self.name == "poisson":
            uniform = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            sampled = poisson_distribution.ppf(uniform, self.mean(z)).astype(np.float64)
            ensure_finite("poisson response", sampled)
            return sampled
        raise ValueError(f"Unsupported family: {self.name}")

    def fit_ridge(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ridge: float,
        theta_init: np.ndarray | None = None,
        max_iter: int = 80,
        tol: float = 1e-8,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        d = X.shape[1]
        if theta_init is None or theta_init.shape != (d,):
            theta = np.zeros(d, dtype=np.float64)
        else:
            theta = np.asarray(theta_init, dtype=np.float64).copy()

        if self.name == "gaussian":
            H = (X.T @ X) / (self.noise_std**2) + ridge * np.eye(d)
            b = (X.T @ y) / (self.noise_std**2)
            try:
                theta = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(H, b, rcond=None)[0]
            ensure_finite("gaussian theta", theta)
            return theta

        if self.name == "poisson":
            def objective(current: np.ndarray) -> float:
                eta = self.poisson_intercept + X @ current
                if np.max(eta) > self.poisson_clip:
                    return float("inf")
                return float(np.sum(np.exp(eta) - y * eta) + 0.5 * ridge * current @ current)

            old_obj = objective(theta)
            for _ in range(max_iter):
                eta = self.poisson_intercept + X @ theta
                if np.max(eta) > self.poisson_clip:
                    raise FloatingPointError("Poisson iterate exceeded configured predictor guard")
                mu = np.exp(eta)
                grad = X.T @ (mu - y) + ridge * theta
                grad_norm = float(np.linalg.norm(grad, ord=np.inf))
                scaled_tolerance = tol * (1.0 + float(np.linalg.norm(theta, ord=np.inf)))
                if grad_norm <= scaled_tolerance:
                    break
                H = X.T @ (mu[:, None] * X) + ridge * np.eye(d)
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(H, grad, rcond=None)[0]
                if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(theta)):
                    break
                directional = float(grad @ step)
                step_scale = 1.0
                accepted = False
                for _ in range(30):
                    candidate = theta - step_scale * step
                    new_obj = objective(candidate)
                    # At machine precision the Armijo decrease can round away,
                    # even though the Newton iterate has converged.  Permit only
                    # a relative round-off-sized slack; materially uphill steps
                    # remain rejected.
                    objective_slack = 32.0 * np.finfo(np.float64).eps * (1.0 + abs(old_obj))
                    if (
                        np.isfinite(new_obj)
                        and new_obj <= old_obj - 1e-4 * step_scale * directional + objective_slack
                    ):
                        theta = candidate
                        old_obj = new_obj
                        accepted = True
                        break
                    step_scale *= 0.5
                if not accepted:
                    # A failed search with a tiny Newton step is a numerical
                    # convergence condition, not a model-fit failure.
                    if np.linalg.norm(step) <= np.sqrt(tol) * (1.0 + np.linalg.norm(theta)):
                        break
                    raise FloatingPointError(
                        f"Poisson Newton line search failed (gradient_inf={grad_norm:.3e})"
                    )
            ensure_finite("poisson theta", theta)
            return theta

        if self.name != "logistic":
            raise ValueError(f"Unsupported family: {self.name}")

        def objective(current: np.ndarray) -> float:
            z = np.clip(X @ current, -self.logistic_clip, self.logistic_clip)
            # Stable log(1 + exp(z)) - y z.
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * current @ current)

        old_obj = objective(theta)
        for _ in range(max_iter):
            z = np.clip(X @ theta, -self.logistic_clip, self.logistic_clip)
            p = expit(z)
            w = np.maximum(p * (1.0 - p), 1e-8)
            grad = X.T @ (p - y) + ridge * theta
            H = X.T @ (w[:, None] * X) + ridge * np.eye(d)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(theta)):
                break

            step_scale = 1.0
            directional = float(grad @ step)
            accepted = False
            for _ in range(25):
                candidate = theta - step_scale * step
                new_obj = objective(candidate)
                if new_obj <= old_obj - 1e-4 * step_scale * directional:
                    theta = candidate
                    old_obj = new_obj
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                theta = theta - 1e-3 * grad / max(1.0, np.linalg.norm(grad))
                old_obj = objective(theta)
            if np.linalg.norm(grad, ord=np.inf) <= tol:
                break
        ensure_finite("logistic theta", theta)
        return theta


def make_family(config: dict) -> GLMFamily:
    name = str(config.get("family", "gaussian")).lower()
    if name == "bernoulli":
        name = "logistic"
    if name not in {"gaussian", "logistic", "poisson"}:
        raise ValueError(f"Only gaussian, logistic, and poisson are supported, got {name}")
    return GLMFamily(
        name=name,  # type: ignore[arg-type]
        noise_std=float(config.get("noise_std", 1.0)),
        logistic_clip=float(config.get("logistic_clip", 30.0)),
        poisson_intercept=float(config.get("poisson_intercept", -0.5)),
        poisson_clip=float(config.get("poisson_clip", 30.0)),
    )
