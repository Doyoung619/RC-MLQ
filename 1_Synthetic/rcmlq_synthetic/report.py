"""Aggregate matched synthetic interactions with pointwise bootstrap intervals."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

TRAJECTORIES = (
    "risk",
    "error",
    "eig",
    "grid_eig",
    "eta",
    "cumulative_error",
)
COLUMNS = (
    "group", "d", "M", "beta", "method", "reference_method", "metric", "t",
    "mean", "ci_low", "ci_high", "valid_interactions", "expected_interactions",
)


def _load_trace(run_dir, job, code_hash):
    directory = run_dir / "statistical"
    archive_path = directory / f"{job['key']}.npz"
    metadata_path = directory / f"{job['key']}.json"
    if not archive_path.is_file() or not metadata_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text())
    if metadata["job"] != job or metadata["code_hash"] != code_hash:
        raise ValueError(f"Run metadata differs from the manifest for {job['key']}")
    if metadata["answers_completed"] != job["answers"]:
        raise ValueError(f"Incomplete history stored as final for {job['key']}")
    count = job["answers"] + 1
    with np.load(archive_path, allow_pickle=False) as archive:
        trace = {
            metric: np.asarray(archive[metric], dtype=float)
            for metric in TRAJECTORIES
            if metric != "cumulative_error"
        }
    for metric, values in trace.items():
        if values.shape != (count,):
            raise ValueError(f"Invalid {metric} shape for {job['key']}: {values.shape}")
    trace["cumulative_error"] = np.cumsum(trace["error"]) / np.arange(1, count + 1)
    return trace


def _bootstrap_weights(group, n, draws):
    seed = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws).astype(float)


def _statistics(values, weights):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    numeric = np.where(valid, values, 0)
    mean = np.divide(numeric.sum(axis=0), count, out=np.full(len(count), np.nan), where=count > 0)
    intervals = np.full((2, len(count)), np.nan)

    # Reuse the same interaction weights across methods, time, and contrasts.
    # Chunk columns to keep memory bounded at the 4,096-answer budget.
    indices = np.flatnonzero(count >= 2)
    for offset in range(0, len(indices), 256):
        segment = indices[offset:offset + 256]
        numerator = weights @ numeric[:, segment]
        denominator = weights @ valid[:, segment]
        sample = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
        )
        sampled = np.any(denominator > 0, axis=0)
        intervals[:, segment[sampled]] = np.nanpercentile(sample[:, sampled], [2.5, 97.5], axis=0)
    return mean, intervals, count


def _optional_number(value):
    return float(value) if np.isfinite(value) else ""


def _write_rows(writer, condition, method, reference, metric, values, weights):
    mean, intervals, count = _statistics(values, weights)
    for t, estimate in enumerate(mean):
        writer.writerow((
            *condition, method, reference, metric, t, _optional_number(estimate),
            _optional_number(intervals[0, t]), _optional_number(intervals[1, t]),
            int(count[t]), len(values),
        ))


def _report_group(jobs, run_dir, code_hash, draws, curves, contrasts, missing):
    first = jobs[0]
    condition = tuple(first[key] for key in ("group", "d", "M", "beta"))
    interactions = sorted({job["interaction"] for job in jobs})
    positions = {interaction: index for index, interaction in enumerate(interactions)}
    weights = _bootstrap_weights(first["group"], len(interactions), draws)
    methods = tuple(dict.fromkeys(job["method"] for job in jobs))
    shape = (len(interactions), first["answers"] + 1)
    traces = {
        method: {metric: np.full(shape, np.nan) for metric in TRAJECTORIES}
        for method in methods
    }
    observed = 0
    for job in jobs:
        trace = _load_trace(run_dir, job, code_hash)
        if trace is None:
            missing.writerow((job["key"], job["group"], job["method"], job["interaction"]))
            continue
        observed += 1
        for metric, values in trace.items():
            traces[job["method"]][metric][positions[job["interaction"]]] = values

    for method, trace in traces.items():
        for metric, values in trace.items():
            _write_rows(curves, condition, method, "", metric, values, weights)
            if method != "RC" and "RC" in traces:
                _write_rows(
                    contrasts, condition, method, "RC", metric,
                    values - traces["RC"][metric], weights,
                )
    return observed


def report_campaign(campaign, run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for key in ("schema", "jobs", "bootstrap"):
        if campaign[key] != manifest[key]:
            raise ValueError(f"Requested {key} differs from the frozen run manifest")
    groups = {}
    for job in campaign["jobs"]:
        groups.setdefault(job["group"], []).append(job)

    completed = 0
    files = ("curves.csv", "paired_vs_rc.csv", "missing_jobs.csv")
    with threadpool_limits(limits=1), TemporaryDirectory(prefix=".report-", dir=run_dir) as directory:
        temporary = Path(directory)
        with (
            (temporary / files[0]).open("w", newline="") as curve_file,
            (temporary / files[1]).open("w", newline="") as contrast_file,
            (temporary / files[2]).open("w", newline="") as missing_file,
        ):
            curves, contrasts, missing = map(csv.writer, (curve_file, contrast_file, missing_file))
            curves.writerow(COLUMNS)
            contrasts.writerow(COLUMNS)
            missing.writerow(("job", "group", "method", "interaction"))
            for jobs in groups.values():
                completed += _report_group(
                    jobs, run_dir, manifest["code_hash"], campaign["bootstrap"],
                    curves, contrasts, missing,
                )
        for filename in files:
            (temporary / filename).replace(run_dir / filename)
    return {
        "complete": completed,
        "expected": len(campaign["jobs"]),
        "bootstrap_resamples": campaign["bootstrap"],
        "outputs": list(files),
    }
