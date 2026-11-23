import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from metaworld_fast.reporting import endpoint_tables, load_runs, physical_tables
from metaworld_fast.validation import compare_snapshots


class ReportingTests(unittest.TestCase):
    def test_endpoint_indexing_counts_and_unclipped_ratio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "statistical").mkdir()
            for interaction in range(5):
                name = root / "statistical" / f"m04_b04__PS__i{interaction}"
                eta = np.arange(128, dtype=float) / 10
                if interaction == 4:
                    eta[31] = np.nan
                np.savez(str(name) + ".npz", q=np.zeros((128, 4, 4)), y=np.zeros(128),
                         risk=np.arange(129, dtype=float), error=np.arange(129, dtype=float) * 2, eta=eta)
                name.with_suffix(".json").write_text(json.dumps(dict(setting_id="m04_b04", method="PS", interaction_id=interaction)))
            runs = load_runs(root)
            individuals, means = endpoint_tables(runs, root)
            eta_row = next(r for r in means if r["metric"] == "realized_information_ratio" and r["answer_count"] == 32)
            self.assertEqual(eta_row["n_effective"], 4)
            self.assertFalse(eta_row["complete_five"])
            self.assertAlmostEqual(eta_row["mean"], 3.1)
            risk_row = next(r for r in means if r["metric"] == "posterior_covariance_trace" and r["answer_count"] == 32)
            self.assertEqual(risk_row["mean"], 32)
            self.assertEqual(risk_row["n_effective"], 5)

    def test_physical_cache_hit_and_reference_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "rollouts").mkdir()
            record = dict(per_query=[dict(setting_id="m02_b04", method="RC", interaction_id=0, kind="selected"),
                                    dict(setting_id="m02_b04", method="RC", interaction_id=0, kind="reference")],
                          per_candidate=[dict(request_index=0, status="executed", simulation_valid=True),
                                         dict(request_index=0, status="cache_hit", simulation_valid=True),
                                         dict(request_index=1, status="executed", simulation_valid=True)])
            (root / "rollouts/shard.json").write_text(json.dumps(record))
            runs = [dict(setting_id="m02_b04", method="RC", data=dict(q=np.zeros((1, 2, 4))))]
            coverage, _, _ = physical_tables(root, runs)
            self.assertEqual(coverage[0]["covered"], 2)
            self.assertEqual(coverage[0]["executed"], 1)
            self.assertEqual(coverage[0]["cache_hit"], 1)
            self.assertEqual(coverage[0]["valid"], 2)

    def test_reference_uncertainty_is_not_labeled_exact(self):
        base = dict(mu=np.zeros(2), cov=np.eye(2), risk=2., eig=.1, reference_eig=.2, eta=.5)
        reference = dict(base, cov=np.eye(2) * 1.1, risk=2.2)
        rows = compare_snapshots(base, [reference, reference])
        self.assertEqual(next(r for r in rows if r["metric"] == "risk")["decision"], "unresolved")
        self.assertTrue(all(r["reference_se"] == 0 for r in rows if r["metric"].startswith("variance")))


if __name__ == "__main__":
    unittest.main()
