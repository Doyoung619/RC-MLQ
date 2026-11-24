import csv
import json

import numpy as np
import pytest

from rcmlq_synthetic.report import report_campaign


def test_report_preserves_matched_interactions_and_missing_jobs(tmp_path):
    statistical = tmp_path / "statistical"
    statistical.mkdir()
    jobs = [
        {
            "key": f"case__{method}__i{interaction}", "group": "case", "method": method,
            "interaction": interaction, "d": 4, "M": 4, "beta": 4, "answers": 1,
        }
        for method in ("RC", "RC-Fixed-0.125")
        for interaction in (0, 1, 2)
    ]
    campaign = {"schema": 1, "jobs": jobs, "bootstrap": 200}
    (tmp_path / "manifest.json").write_text(json.dumps({**campaign, "code_hash": "test"}))
    for job in jobs:
        fixed = job["method"] != "RC"
        if fixed and job["interaction"] == 2:
            continue
        metric = np.array([1.0, 3.0]) + job["interaction"] + 2 * fixed
        key = job["key"]
        np.savez(statistical / f"{key}.npz", **dict.fromkeys(("risk", "error", "eig", "grid_eig", "eta"), metric))
        (statistical / f"{key}.json").write_text(json.dumps(
            {"job": job, "code_hash": "test", "answers_completed": 1}
        ))

    result = report_campaign(campaign, tmp_path)

    assert result["complete"] == 5
    assert result["expected"] == 6
    with (tmp_path / "paired_vs_rc.csv").open() as stream:
        row = next(row for row in csv.DictReader(stream) if row["metric"] == "risk" and row["t"] == "1")
    assert row["method"] == "RC-Fixed-0.125"
    assert [float(row[column]) for column in ("mean", "ci_low", "ci_high")] == [2, 2, 2]
    assert [int(row[column]) for column in ("valid_interactions", "expected_interactions")] == [2, 3]
    with (tmp_path / "missing_jobs.csv").open() as stream:
        assert [row["job"] for row in csv.DictReader(stream)] == ["case__RC-Fixed-0.125__i2"]

    with pytest.raises(ValueError, match="differs from the frozen run manifest"):
        report_campaign({**campaign, "bootstrap": 201}, tmp_path)
