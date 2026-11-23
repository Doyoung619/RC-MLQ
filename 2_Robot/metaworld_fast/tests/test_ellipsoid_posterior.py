"""Joint-integration and replay checks for the new offline ellipsoid audit."""
from pathlib import Path
import unittest
import numpy as np
from metaworld_fast.posterior_marginal import Marginal1D
from metaworld_fast.ellipsoid_posterior import (
    MarginalInverseCDF, PosteriorReplay, joint_product_points, sobol_bank,
)

ROOT=Path(__file__).resolve().parents[1]/"outputs"/"rcmlq_t128_s10_20261004T2008Z"


class InverseCDFTests(unittest.TestCase):
    def test_vectorized_quantiles_match_independent_scalar_roots(self):
        marginal=Marginal1D(beta=16.,nodes=257)
        rng=np.random.default_rng(5532)
        for step in range(24):
            marginal.update(np.array([-.5,0.,.5,1.]),int(rng.integers(4)))
        probabilities=np.r_[0.,1e-8,np.linspace(.001,.999,127),1-1e-8,1.]
        law=MarginalInverseCDF.from_marginal(marginal)
        expected=marginal.quantiles(probabilities)
        np.testing.assert_allclose(law.quantiles(probabilities),expected,rtol=0,atol=2e-9)

    def test_joint_uniform_law_and_independent_replication(self):
        laws=tuple(MarginalInverseCDF.from_marginal(Marginal1D(4,257)) for _ in range(4))
        first=joint_product_points(laws,4096,11)
        np.testing.assert_allclose(first,2*sobol_bank(4,4096,11)-1,atol=1e-14)
        np.testing.assert_allclose(first.mean(axis=0),0,atol=3e-5)
        np.testing.assert_allclose(np.cov(first,rowvar=False,bias=True),np.eye(4)/3,atol=3e-4)
        self.assertFalse(np.array_equal(first,joint_product_points(laws,4096,12)))
        with self.assertRaises(ValueError):sobol_bank(4,4095,11)


@unittest.skipUnless(ROOT.exists(),"Saved empirical histories unavailable")
class SavedReplayTests(unittest.TestCase):
    def test_all_marginal_snapshots_terminal_check_and_same_bank_eig(self):
        replay=PosteriorReplay(ROOT/"statistical/m04_b16__RC__i0.npz")
        first=next(replay);first_points=first.points.copy()
        actual=first.eig_comparison(count=8192)
        self.assertLess(actual["max_abs_original_discrepancy"],1e-5)
        snapshots=[first]+list(replay)
        self.assertEqual([s.t for s in snapshots],list(range(128)))
        self.assertEqual(replay.summary()["moment_checks"],129)
        self.assertEqual(replay.summary()["max_mean_error"],0)
        self.assertEqual(replay.summary()["max_cov_error"],0)
        np.testing.assert_array_equal(first.points,first_points)
        self.assertEqual(replay.summary()["answers_replayed"],128)
        selected=snapshots[64]
        alternate,_=selected.integration(8192,replicate=2)
        self.assertEqual(alternate.shape,(8192,4))
        self.assertFalse(np.array_equal(alternate[:4096],selected.points))
        with self.assertRaises(StopIteration):next(replay)

    def test_next_batch_does_not_restart_or_duplicate(self):
        replay=PosteriorReplay(ROOT/"statistical/m04_b04__PQ__i0.npz",rounds=(0,31,127))
        first=replay.next_batch(2);last=replay.next_batch(2)
        np.testing.assert_array_equal(first["t"],[0,31])
        np.testing.assert_array_equal(last["t"],[127])
        self.assertEqual(first["points"].shape,(2,4096,4))
        self.assertEqual(first["grid_q"].shape,(2,7,4,4))
        self.assertIsNone(replay.next_batch(2))
        self.assertTrue(replay.summary()["complete"])

    def test_smc_exact_history_seed_and_noninterference(self):
        # Preserve the exact original CUDA history and SMC random stream.
        replay=PosteriorReplay(ROOT/"statistical/m04_b16__SG__i0.npz",device="cuda:3",rounds=(0,64,127))
        selected=list(replay)
        self.assertTrue(replay.summary()["moment_checks_passed"])
        self.assertEqual(replay.summary()["max_mean_error"],0)
        for snapshot in selected:
            comparison=snapshot.eig_comparison(count=8192)
            self.assertEqual(comparison["max_abs_original_discrepancy"],0)
            points,weights=snapshot.integration(8192,replicate=7)
            np.testing.assert_array_equal(points,snapshot.points)
            np.testing.assert_array_equal(weights,snapshot.weights)
            self.assertEqual(len(points),4096)


if __name__=="__main__":unittest.main()
