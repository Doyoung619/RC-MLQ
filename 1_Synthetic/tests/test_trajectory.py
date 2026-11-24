from pathlib import Path

import numpy as np
import pytest

from rcmlq_synthetic.config import load_campaign
from rcmlq_synthetic.runner import code_hash, initial_seeds, run_interaction


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_campaign_reuses_matching_settings_and_restricts_binary_methods():
    campaign = load_campaign(CONFIGS / "paper_sweeps.yaml")
    jobs = campaign["jobs"]
    assert len(jobs) == 984
    assert len({job["key"] for job in jobs}) == len(jobs)
    assert len({job["group"] for job in jobs}) == 19
    assert {job["M"] for job in jobs if job["method"] == "MEM"} == {2}


def test_methods_share_theta_but_other_streams_are_separate():
    jobs = load_campaign(CONFIGS / "paper_main.yaml", interactions=1)["jobs"]
    rc, pq = map(initial_seeds, jobs[:2])
    assert rc["theta"] == pq["theta"]
    assert all(rc[name] != pq[name] for name in rc if name != "theta")


def test_resume_preserves_full_closed_loop_path(tmp_path):
    job = load_campaign(CONFIGS / "smoke.yaml", interactions=1, methods=["PS"])["jobs"][0]
    one, split = tmp_path / "one", tmp_path / "split"
    digest = code_hash()
    run_interaction(job, one, digest)
    assert not run_interaction(job, split, digest, max_answers=2)["complete"]
    assert len(list((split / "checkpoints").glob("*.pkl"))) == 1
    assert run_interaction(job, split, digest)["complete"]
    assert run_interaction(job, split, digest)["reused"]
    assert not list((split / "checkpoints").glob("*.pkl"))
    name = f"{job['key']}.npz"
    with np.load(one / "statistical" / name) as expected, np.load(split / "statistical" / name) as actual:
        for key in actual.files:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert len(actual["q"]) == len(actual["y"]) + 1 == 5
    with pytest.raises(ValueError, match="different job"):
        run_interaction(job, split, "different digest")


def test_direct_eig_matches_grid_max_and_fixed_scale_is_preserved(tmp_path):
    methods = ["RC-EIG", "RC-Fixed-0.5"]
    jobs = load_campaign(CONFIGS / "smoke.yaml", interactions=1, methods=methods)["jobs"]
    for job in jobs:
        run_interaction(job, tmp_path, code_hash())
        with np.load(tmp_path / "statistical" / f"{job['key']}.npz") as data:
            if job["method"] == "RC-EIG":
                np.testing.assert_allclose(data["eig"], data["grid_eig"], rtol=0, atol=0)
                np.testing.assert_allclose(data["eta"], 1.0, rtol=0, atol=0)
            else:
                np.testing.assert_allclose(data["selected_scale"], 0.5, rtol=0, atol=0)
