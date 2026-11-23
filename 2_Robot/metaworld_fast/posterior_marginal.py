"""Adaptive 257-node sinh quadrature with full-history remeshing.

Moments and EIG use positive Simpson weights in the uniform transformed
coordinate. Quantiles invert the antiderivative of a positive PCHIP density,
not a discrete-node CDF. This continuous-CDF numerical choice is checked
against doubled-resolution frozen histories.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import logsumexp
from .model import coordinate_of_query, log_probabilities
from .metrics import integrated_eig


class Marginal1D:
    def __init__(self, beta, nodes):
        if nodes < 5 or nodes % 2 != 1:
            raise ValueError("Composite Simpson quadrature requires an odd node count")
        self.beta, self.n = float(beta), int(nodes)
        self.history = []
        self.remesh_count = 0
        self.center, self.scale = 0., None
        self._mesh()

    def _mesh(self):
        if self.scale is None:
            self.u = np.linspace(-1., 1., self.n)
            self.x = self.u.copy()
            jac = np.ones(self.n)
        else:
            self.u = np.linspace(np.arcsinh((-1-self.center)/self.scale), np.arcsinh((1-self.center)/self.scale), self.n)
            self.x = self.center + self.scale * np.sinh(self.u)
            self.x[0], self.x[-1] = -1., 1.
            jac = self.scale * np.cosh(self.u)
        self.jac = jac
        quadrature = np.ones(self.n)
        quadrature[1:-1:2], quadrature[2:-1:2] = 4., 2.
        self.quadrature = quadrature * (self.u[1]-self.u[0])/3
        logdensity = np.zeros(self.n)
        for q, y in self.history:
            logits = 2*self.beta*self.x[:,None]*q - self.beta*q*q
            logdensity += logits[:, y]-logsumexp(logits, axis=1)
        self._set_density(logdensity)

    def _set_density(self, logdensity):
        self.logdensity = logdensity
        transformed = np.exp(logdensity - logdensity.max()) * self.jac
        self.weights = transformed * self.quadrature
        self.weights /= self.weights.sum()
        self.mean = float(self.weights @ self.x)
        self.variance = float(self.weights @ (self.x-self.mean)**2)
        if self.variance <= 0 or not np.isfinite(self.variance):
            raise FloatingPointError("Zero or invalid marginal variance")
        self._cdf_integral = PchipInterpolator(self.u, transformed, extrapolate=False).antiderivative()
        self._cdf_origin = float(self._cdf_integral(self.u[0]))
        self._cdf_total = float(self._cdf_integral(self.u[-1]) - self._cdf_origin)

    def update(self, q, y):
        q = np.asarray(q, dtype=np.float64).copy()
        self.history.append((q, int(y)))
        logits = 2*self.beta*self.x[:, None]*q - self.beta*q*q
        self._set_density(self.logdensity + logits[:, y] - logsumexp(logits, axis=1))
        sd = np.sqrt(self.variance)
        if self.scale is None or sd/self.scale < .8 or sd/self.scale > 1.25 or abs(self.mean-self.center) > .5*sd:
            self.center, self.scale = self.mean, sd
            self.remesh_count += 1
            self._mesh()  # Always replay every actual likelihood for this coordinate.

    def quantiles(self, probabilities):
        result = []
        for p in np.asarray(probabilities):
            if p <= 0:
                result.append(-1.); continue
            if p >= 1:
                result.append(1.); continue
            target = self._cdf_origin + float(p)*self._cdf_total
            u = brentq(lambda z: self._cdf_integral(z)-target, self.u[0], self.u[-1], xtol=2e-14)
            result.append(u if self.scale is None else self.center+self.scale*np.sinh(u))
        return np.asarray(result)


class MarginalPosterior:
    factorized = True

    def __init__(self, d=4, beta=4., nodes=257):
        self.d, self.beta, self.nodes = int(d), float(beta), int(nodes)
        self.marginals = [Marginal1D(beta, nodes) for _ in range(d)]
        self.history = []
        self.last_diagnostics = {}

    def moments(self):
        return np.array([m.mean for m in self.marginals]), np.diag([m.variance for m in self.marginals])

    def projected_quantiles(self, direction, probabilities):
        idx = np.flatnonzero(np.abs(direction) > 1e-12)
        if len(idx) != 1 or direction[idx[0]] != 1:
            raise ValueError("Factorized quantiles require a positive coordinate direction")
        return self.marginals[int(idx[0])].quantiles(probabilities)

    def _integration(self, q):
        idx = coordinate_of_query(q)
        marginal = self.marginals[idx]
        mean, _ = self.moments()
        points = np.broadcast_to(mean, (self.nodes, self.d)).copy()
        points[:, idx] = marginal.x
        return idx, points, marginal.weights

    def eig(self, q):
        _, points, weights = self._integration(q)
        return integrated_eig(points, weights, q, self.beta)

    def hypothetical_means(self, q):
        idx, points, weights = self._integration(q)
        logp = log_probabilities(points, q, self.beta)
        with np.errstate(divide="ignore"):
            joint = np.log(weights)[:,None] + logp
        normalizer = logsumexp(joint, axis=0)
        conditional = np.exp(joint-normalizer)
        # All other coordinate contributions cancel exactly.
        mean, _ = self.moments()
        output = np.broadcast_to(mean, (len(q), self.d)).copy()
        output[:,idx] = points[:,idx] @ conditional
        return output

    def update(self, q, y):
        idx = coordinate_of_query(q)
        self.marginals[idx].update(np.asarray(q)[:,idx], y)
        self.history.append((np.asarray(q).copy(), int(y)))
        self.last_diagnostics = {"coordinate":idx,"nodes":self.nodes,"remesh_count":sum(m.remesh_count for m in self.marginals)}

    def sample(self, count, rng):
        return np.column_stack([m.quantiles(rng.random(count)) for m in self.marginals])
