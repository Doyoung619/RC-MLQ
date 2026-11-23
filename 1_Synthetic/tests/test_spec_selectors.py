"""Independent scientific tests: matrix formulas, enumeration and distributions."""
import itertools
import copy
import unittest

import numpy as np

from pscore_harness.spec_suite.selectors import (METHODS, InverseCache, curvature, dependent_round,
    dual_upper_bound, e_scores_batched, e_scores_secular, project_capped_simplex,
    select_batch, shared_candidates, shared_initial_ids, stable_priorities)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(82)
        self.X = self.rng.normal(size=(55, 5)) / 3
        self.old = np.arange(10)
        self.candidates = np.arange(10, 30)
        self.theta = self.rng.normal(size=5)

    def select(self, method, family="gaussian", m=5, config=None):
        config = dict(config or {})
        if method == 'fb_opt':config.setdefault('fb_opt_Tmax',20)
        if method == 'fb_opt2':config.setdefault('fb_opt2_Tmax',20)
        return select_batch(method, self.X, self.old, self.candidates, self.theta,
                            family, m, 1., np.random.default_rng(12), config)

    def test_all_methods_cardinality_and_replay(self):
        for method in METHODS:
            family = "bernoulli" if method == "badge" else "gaussian"
            ids, audit = self.select(method, family)
            again, audit2 = self.select(method, family)
            np.testing.assert_array_equal(ids, again)
            self.assertEqual(len(set(ids)), 5)
            self.assertFalse(set(ids) & set(self.old))
            self.assertEqual(audit["rng_state_after"], audit2["rng_state_after"])
            w = curvature(self.X[ids] @ self.theta, family)
            np.testing.assert_allclose(audit["W_end"], np.asarray(audit["W_start"]) + self.X[ids].T @ (w[:, None] * self.X[ids]))

    def test_cached_scores_and_downdate_against_inverse(self):
        U = self.rng.normal(size=(18, 5))
        H = self.rng.normal(size=(5, 5))
        F = H.T @ H
        cache = InverseCache(2 * np.eye(5), U, F, check_every=3)
        for j, sign in [(0, 1), (2, 1), (3, 1), (4, 1), (2, -1), (4, -1)]:
            cache.update(j, sign)
            A = np.linalg.inv(cache.W)
            np.testing.assert_allclose(cache.A, A, atol=1e-12)
            np.testing.assert_allclose(cache.Z, U @ A, atol=1e-12)
            np.testing.assert_allclose(cache.q, np.diag(U @ A @ U.T), atol=1e-12)
            np.testing.assert_allclose(cache.r, np.diag(U @ A @ F @ A @ U.T), atol=1e-11)

    def test_secular_matches_bruteforce_and_deflation(self):
        for d in [1, 2, 5, 20]:
            for spectrum in [np.linspace(1., 4., d), np.ones(d), 1. + np.arange(d)*1e-10]:
                Q, _ = np.linalg.qr(self.rng.normal(size=(d, d)))
                W = (Q * spectrum) @ Q.T
                U = self.rng.normal(size=(100, d))
                U[0] = 0
                np.testing.assert_allclose(e_scores_secular(W, U),
                    [np.linalg.eigvalsh(W + np.outer(u, u))[0] for u in U], atol=5e-10, rtol=5e-10)
        W = np.diag([1., 2., 5.])
        U = np.array([[0., 4., 5.], [10., 0., 0.], [1., 0., 0.], [1e-10, 1., 0.]])
        np.testing.assert_allclose(e_scores_secular(W, U), e_scores_batched(W, U), atol=1e-10)

    def test_design_winners_exact(self):
        for method in ["a_opt", "d_opt", "e_greedy"]:
            ids, _ = self.select(method)
            W = np.eye(5) + self.X[self.old].T @ self.X[self.old]
            remaining = list(self.candidates)
            for chosen in ids:
                if method == "a_opt":
                    value = lambda i: -np.trace(np.linalg.inv(W + np.outer(self.X[i], self.X[i])))
                elif method == "d_opt":
                    value = lambda i: np.linalg.slogdet(W + np.outer(self.X[i], self.X[i]))[1]
                else:
                    value = lambda i: np.linalg.eigvalsh(W + np.outer(self.X[i], self.X[i]))[0]
                self.assertAlmostEqual(value(chosen), max(value(i) for i in remaining), places=9)
                remaining.remove(chosen)
                W += np.outer(self.X[chosen], self.X[chosen])

    def test_bait_forward_backward_exact_objectives(self):
        _, audit = self.select("bait_glm", "poisson")
        F = np.asarray(audit["F_pool"])
        U = self.X[self.candidates] * np.sqrt(curvature(self.X[self.candidates] @ self.theta, "poisson"))[:, None]
        W = np.asarray(audit["W_start"])
        provisional = set()
        for row in audit["slot_trace"]:
            j = row["candidate_position"]
            before = np.trace(np.linalg.inv(W) @ F)
            if row["phase"] == "forward":
                available = set(range(len(U))) - provisional
                values = {i: np.trace(np.linalg.inv(W + np.outer(U[i], U[i])) @ F) - before for i in available}
                self.assertAlmostEqual(values[j], min(values.values()), places=9)
                provisional.add(j)
                W += np.outer(U[j], U[j])
            else:
                values = {i: np.trace(np.linalg.inv(W - np.outer(U[i], U[i])) @ F) - before for i in provisional}
                self.assertAlmostEqual(values[j], min(values.values()), places=9)
                provisional.remove(j)
                W -= np.outer(U[j], U[j])
            self.assertAlmostEqual(values[j], row["objective_change"], places=9)
        self.assertEqual(len(provisional), 5)
        identity_cache = InverseCache(np.eye(5), U, np.eye(5))
        np.testing.assert_allclose(identity_cache.r, (identity_cache.Z**2).sum(axis=1))

    def test_dependent_round_marginals_and_budget(self):
        a = np.array([.1, .7, .4, .8, 0., 1.])
        counts = np.zeros(len(a))
        for _ in range(4000):
            ids = dependent_round(a, self.rng)
            self.assertEqual(len(ids), 3)
            counts[ids] += 1
        np.testing.assert_allclose(counts / 4000., a, atol=.035)
        projected = project_capped_simplex(np.array([-.01, .1, .6, 1.01]), 2)
        self.assertAlmostEqual(projected.sum(), 2)
        self.assertTrue(np.all((projected >= 0) & (projected <= 1)))

    def test_sdp_enumerated_optimum_and_bound(self):
        self.candidates = np.arange(10, 18)
        selected, audit = self.select("e_sdp", m=3)
        self.assertFalse(audit["fallback_used"], audit.get("sdp_attempts"))
        W = np.asarray(audit["W_start"])
        best = max(np.linalg.eigvalsh(W + self.X[list(s)].T @ self.X[list(s)])[0]
                   for s in itertools.combinations(self.candidates, 3))
        self.assertLessEqual(audit["sdp_lb"], best + 1e-9)
        self.assertGreaterEqual(audit["sdp_ub"], best - 1e-9)
        self.assertGreaterEqual(audit["sdp_lb"], audit["sdp_rounding_values"][-1] - 1e-9)
        P = np.asarray(audit["feasible_dual_P"])
        self.assertAlmostEqual(np.trace(P), 1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(P)[0], -1e-12)
        _, bad = self.select("e_sdp", m=3, config={"sdp": {"force_failure": True}})
        self.assertTrue(bad["fallback_used"])
        self.assertIn("FAILED", bad["sdp_status"])
        self.assertEqual(len(bad["sdp_attempts"]), 3)

    def test_dual_bound_with_indefinite_projection(self):
        W = np.eye(3)
        U = self.rng.normal(size=(7, 3))
        ub, P, info = dual_upper_bound(W, U, 3, np.diag([-1., .1, 2.]))
        best = max(np.linalg.eigvalsh(W + U[list(s)].T @ U[list(s)])[0]
                   for s in itertools.combinations(range(7), 3))
        self.assertGreaterEqual(ub + 1e-10, best)
        self.assertAlmostEqual(info["dual_negative_mass"], 1.)

    def test_badge_distances_and_zero_fallback(self):
        p = 1 / (1 + np.exp(-(self.X @ self.theta)))
        G = (p - (p >= .5))[:, None] * self.X
        full = np.concatenate([G, -G], axis=1)
        np.testing.assert_allclose(((full - full[0])**2).sum(axis=1), 2*((G - G[0])**2).sum(axis=1))
        self.X[:] = 0.
        _, audit = self.select("badge", "bernoulli")
        self.assertEqual(audit["badge_zero_distance_fallbacks"], 4)
        with self.assertRaises(ValueError):
            self.select("badge", "poisson")

    def test_thompson_covariance_orientation(self):
        _, audit = self.select("ts_glm")
        W = np.asarray(audit["W_start"])
        standard = np.asarray(audit["thompson_standard_draw"])
        difference = np.asarray(audit["theta_tilde"]) - self.theta
        np.testing.assert_allclose(np.linalg.cholesky(W).T @ difference, standard, atol=1e-12)
        L = np.linalg.cholesky(W)
        transform = np.linalg.solve(L.T, np.eye(len(W)))
        np.testing.assert_allclose(transform @ transform.T, np.linalg.inv(W), atol=1e-12)

    def test_fb_freeze_temperature_and_past_reweighting(self):
        ids, audit = self.select("fb_sample", "bernoulli", m=1)
        tau = .01 * (len(self.old) + 1)**.75
        self.assertAlmostEqual(audit["slot_trace"][0]["temperature"], tau)
        old_weights = curvature(self.X[self.old] @ self.theta, "bernoulli")
        W = np.eye(5) + self.X[self.old].T @ (old_weights[:, None] * self.X[self.old])
        np.testing.assert_allclose(audit["W_start"], W)
        self.old = np.array([], dtype=int)
        _, cold = self.select("fb_sample", "bernoulli", m=5)
        self.assertAlmostEqual(cold["slot_trace"][0]["temperature"], .01 * 5**.75)
        self.assertFalse(np.allclose(cold["W_start"], cold["W_end"]))

    def test_shared_initial_and_candidates(self):
        ids = shared_initial_ids(self.X, "toy", 7, n0=12)
        np.testing.assert_array_equal(ids, shared_initial_ids(self.X, "toy", 7, n0=12))
        a = shared_candidates(len(self.X), [1, 2], "toy", 1, 3, 12)
        b = shared_candidates(len(self.X), [2, 3], "toy", 1, 3, 12)
        common = set(a) & set(b)
        self.assertEqual([x for x in a if x in common], [x for x in b if x in common])
        self.assertEqual(len(set(ids)), 12)

    def test_independent_audit_replay_and_tamper_detection(self):
        from pscore_harness.spec_suite.audit_replay import replay_batch
        for method in METHODS:
            family = "bernoulli" if method in ("badge", "fb_opt", "fb_opt2") else "poisson"
            _, audit = self.select(method, family)
            result = replay_batch(self.X, audit, family, labeled_ids=self.old)
            self.assertTrue(result["verified"])
            corrupt = copy.deepcopy(audit)
            corrupt["W_end"][0][0] += .1
            with self.assertRaises(AssertionError):
                replay_batch(self.X, corrupt, family, labeled_ids=self.old)
            if audit["slot_trace"] and "score" in audit["slot_trace"][0]:
                corrupt = copy.deepcopy(audit)
                corrupt["slot_trace"][0]["score"] += .1
                with self.assertRaises(AssertionError):
                    replay_batch(self.X, corrupt, family, labeled_ids=self.old)


if __name__ == "__main__":
    unittest.main()
