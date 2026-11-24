"""Check the compact follow-up against the established Robot policies."""
import numpy as np
import pytest

from metaworld_fast.line_template import SCALE_GRID
from metaworld_fast.statistical_runner import run_statistical
from robot_e01 import FIXED_SCALES, infer_one, jobs_for, protocol


@pytest.mark.parametrize("method", ("RC", "PQ"))
def test_existing_controls_keep_the_original_histories(tmp_path, method):
    job = next(job for job in jobs_for(protocol(1, 6, 20261004, False)) if job["method"] == method)
    root = tmp_path / "compact"
    (root / "statistical").mkdir(parents=True)
    path = infer_one(job, root)
    original = run_statistical(job, tmp_path / "original")
    with np.load(path, allow_pickle=False) as reduced, np.load(original["artifact"], allow_pickle=False) as baseline:
        for field in ("theta", "q", "y", "mu", "cov", "risk", "error"):
            np.testing.assert_array_equal(reduced[field], baseline[field])


def test_direct_eig_and_fixed_scales_use_the_same_admissible_grid(tmp_path):
    (tmp_path / "statistical").mkdir()
    jobs = jobs_for(protocol(1, 6, 20261004, False))
    latent = {}
    for job in jobs:
        path = infer_one(job, tmp_path)
        method = job["method"]
        with np.load(path, allow_pickle=False) as archive:
            latent[method] = archive["theta"]
            if method == "DirectEIG":
                np.testing.assert_array_equal(archive["selected_scale"], SCALE_GRID[archive["reference_eigs"].argmax(axis=1)])
                np.testing.assert_array_equal(archive["eta_grid"], np.ones(6))
            if method in FIXED_SCALES:
                scale = FIXED_SCALES[method]
                index = int(np.flatnonzero(SCALE_GRID == scale)[0])
                np.testing.assert_array_equal(archive["selected_scale"], np.full(6, scale))
                np.testing.assert_array_equal(archive["q"], archive["reference_queries"][:, index])
    for theta in latent.values():
        np.testing.assert_array_equal(theta, latent["RC"])
