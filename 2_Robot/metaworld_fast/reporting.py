"""Read experiment artifacts and publish unsmoothed, count-aware summaries."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METHODS = ("RC", "PQ", "QI", "PS", "GC", "SG")
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555")))
SETTINGS = (("m02_b04", 2, 4), ("m04_b04", 4, 4), ("m08_b04", 8, 4),
            ("m16_b04", 16, 4), ("m04_b02", 4, 2), ("m04_b08", 4, 8), ("m04_b16", 4, 16))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def read_calibration(root):
    for rel in ("diagnostics/engineering_calibration.json", "engineering/robot_calibration.json", "diagnostics/robot_calibration.json"):
        result = read_json(Path(root) / rel)
        if result:
            return result
    return {}


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def load_runs(run_dir):
    """Ignore auxiliary npz files; never infer a completed run from its filename."""
    runs = []
    for path in sorted((Path(run_dir) / "statistical").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if not {"q", "y", "risk", "eta"}.issubset(archive.files):
                continue
            data = {key: archive[key] for key in archive.files}
        meta = read_json(path.with_suffix(".json"))
        flattened = "_".join(path.parts)
        setting = meta.get("setting_id", meta.get("setting"))
        if isinstance(setting, dict):
            setting = setting.get("id")
        setting = setting or next((s for s, _, _ in SETTINGS if s in flattened), None)
        method = meta.get("method") or next((m for m in METHODS if m in path.stem.split("_")), None)
        if method is None:
            method = next((m for m in METHODS if m in path.parts), None)
        interaction = meta.get("interaction_id", meta.get("interaction"))
        if interaction is None:
            import re
            match = re.search(r"(?:interaction|seed|i)[_-]?(\d+)", path.stem)
            interaction = int(match.group(1)) if match else None
        if setting is None or method is None or interaction is None:
            raise ValueError(f"Missing run identity in {path.with_suffix('.json')}")
        runs.append(dict(path=path, setting_id=setting, method=method,
                         interaction_id=int(interaction), data=data, meta=meta))
    identities = [(r["setting_id"], r["method"], r["interaction_id"]) for r in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate statistical run identities; refusing a biased average")
    return runs


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_config(run_dir, config=None):
    if config is not None:
        return config
    from .util import load_config
    frozen = Path(run_dir) / "config_frozen.yaml"
    return load_config(frozen if frozen.exists() else Path(__file__).parent / "config/final.yaml")


def endpoint_tables(runs, run_dir, expected_interactions=None):
    expected_interactions = expected_interactions or len(report_config(run_dir)["experiment"]["interaction_ids"])
    rows = []
    for run in runs:
        data = run["data"]
        for count in (32, 64, 128):
            for metric, key, index in (("posterior_covariance_trace", "risk", count),
                                       ("realized_squared_error", "error", count),
                                       ("realized_information_ratio", "eta", count - 1)):
                values = data.get(key, np.empty(0))
                observed = len(data["y"]) >= count and len(values) > index
                value = finite_float(values[index]) if observed else None
                rows.append(dict(setting_id=run["setting_id"], method=run["method"],
                                 interaction_id=run["interaction_id"], answer_count=count,
                                 metric=metric, index=index, value=value,
                                 available=value is not None, source=str(run["path"])))
    means = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["setting_id"], row["method"], row["answer_count"], row["metric"])].append(row)
    for (setting, method, count, metric), group in sorted(grouped.items()):
        values = [r["value"] for r in group if r["value"] is not None]
        n = len(values)
        means.append(dict(setting_id=setting, method=method, answer_count=count, metric=metric,
                          mean=float(np.mean(values)) if n else None, n_effective=n,
                          n_expected=expected_interactions, complete_expected=n == expected_interactions,
                          **({"complete_five": n == 5} if expected_interactions == 5 else {}),
                          descriptive_between_interaction_se=float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None))
    write_csv(Path(run_dir) / "tables/endpoints_per_interaction.csv", rows)
    write_csv(Path(run_dir) / "tables/endpoints_mean.csv", means)
    return rows, means


def plot_run(run_dir, runs=None, config=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = load_runs(run_dir) if runs is None else runs
    expected_interactions = len(report_config(run_dir, config)["experiment"]["interaction_ids"])
    target = Path(run_dir) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    curve_counts = []
    for sweep, setting_ids in (("M", ["m02_b04", "m04_b04", "m08_b04", "m16_b04"]),
                               ("beta", ["m04_b02", "m04_b04", "m04_b08", "m04_b16"])):
        for key, ylabel in (("eta", "Realized integrated-EIG ratio"),
                            ("risk", "Full posterior covariance trace")):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharey=True, constrained_layout=True)
            for ax, setting in zip(axes, setting_ids):
                _, choices, beta = next(s for s in SETTINGS if s[0] == setting)
                ax.set_title(f"M={choices}, β={beta}, d=4")
                for method in METHODS:
                    selected = [r for r in runs if r["setting_id"] == setting and r["method"] == method]
                    arrays = []
                    for run in selected:
                        observed_length = len(run["data"]["y"]) + (key == "risk")
                        values = run["data"][key][:observed_length].astype(float)
                        arrays.append(values)
                        ax.plot(np.arange(len(values)), values, color=COLORS[method], lw=.55, alpha=.23)
                    if arrays:
                        stack = np.full((len(arrays), max(map(len, arrays))), np.nan)
                        for i, values in enumerate(arrays):
                            stack[i, :len(values)] = values
                        counts = np.sum(np.isfinite(stack), axis=0)
                        mean = np.divide(np.nansum(stack, axis=0), counts,
                                         out=np.full(stack.shape[1], np.nan), where=counts > 0)
                        ax.plot(np.arange(len(mean)), mean, color=COLORS[method], lw=1.8,
                                linestyle="-" if np.all(counts == expected_interactions) else "--")
                        curve_counts.extend(dict(sweep=sweep, metric=key, setting_id=setting,
                                                 method=method, t=t, n_effective=int(n), n_expected=expected_interactions)
                                            for t, n in enumerate(counts))
                ax.set_xlim(0, 127 if key == "eta" else 128)
                ax.set_xlabel("Answers already received, t")
                ax.grid(alpha=.16)
            axes[0].set_ylabel(ylabel)
            handles = [Line2D([], [], color=COLORS[m], lw=2, label=m) for m in METHODS]
            fig.legend(handles=handles, loc="outside upper center", ncol=6, frameon=False)
            fig.supxlabel(f"Thin: individual interactions. Thick: arithmetic mean. Dashed mean: fewer than {expected_interactions} finite values.\n"
                          "Raw curves; SG scale and stage center are fixed for all questions (t=0,…,127).", fontsize=9)
            for extension in ("pdf", "png"):
                path = target / f"{sweep.lower()}_sweep_{key}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                outputs.append(str(path))
            plt.close(fig)
    write_csv(Path(run_dir) / "tables/curve_effective_counts.csv", curve_counts)
    return outputs


def physical_tables(run_dir, runs):
    """Aggregate compact worker records while keeping requested/executed distinct."""
    root = Path(run_dir)
    rows = []
    query_rows = []
    runtime_rows = []
    for path in sorted((root / "rollouts").rglob("*.json")):
        record = read_json(path)
        if record.get("summary"):
            runtime_rows.append(dict(artifact=str(path), **record["summary"]))
        requests = record.get("requests", record.get("per_query", []))
        candidates = record.get("per_candidate", record.get("candidates", []))
        for c in candidates:
            item = dict(c)
            request_index = item.get("request_index")
            if request_index is not None and isinstance(requests, list) and request_index < len(requests):
                request = requests[request_index]
                for field in ("setting_id", "method", "interaction_id", "t", "kind"):
                    item.setdefault(field, request.get(field))
            item["artifact"] = str(path)
            rows.append(item)
        for q in record.get("per_query", []):
            item = dict(q)
            item["artifact"] = str(path)
            query_rows.append(item)
    aggregated = defaultdict(Counter)
    for run in runs:
        aggregate = aggregated[(run["setting_id"], run["method"])]
        aggregate["logical_selected_requests"] += int(np.prod(run["data"]["q"].shape[:2]))
    for item in rows:
        if item.get("kind", "selected") != "selected":
            continue
        key = item.get("setting_id"), item.get("method")
        if None in key:
            continue
        counter = aggregated[key]
        status = item.get("status", "unknown")
        counter[status] += 1
        if status in ("executed", "cache_hit"):
            counter["covered"] += 1
        if item.get("simulation_valid", item.get("valid", item.get("finite", False))):
            counter["valid"] += 1
    coverage = []
    for (setting, method), counts in sorted(aggregated.items()):
        logical = counts["logical_selected_requests"]
        valid_candidates = [r for r in rows if r.get("setting_id") == setting and r.get("method") == method
                            and r.get("kind", "selected") == "selected"]
        reconstruction = [r["command_reconstruction_error"] for r in valid_candidates
                          if r.get("command_reconstruction_error") is not None]
        saturation = [r["workspace_saturation_rate"] for r in valid_candidates
                      if r.get("workspace_saturation_rate") is not None]
        coverage.append(dict(setting_id=setting, method=method, **counts,
                             coverage_fraction=counts["covered"] / logical if logical else None,
                             pending=max(0, logical - counts["covered"] - counts["failed"]),
                             max_command_reconstruction_error=max(reconstruction) if reconstruction else None,
                             mean_workspace_saturation_rate=float(np.mean(saturation)) if saturation else None,
                             failed_fraction=counts["failed"] / logical if logical else None,
                             simulation_valid_fraction=counts["valid"] / logical if logical else None))
    # Structured fields remain lossless JSON cells in CSV rather than being flattened misleadingly.
    for collection in (rows, query_rows):
        for item in collection:
            for key, value in list(item.items()):
                if isinstance(value, (dict, list)):
                    item[key] = json.dumps(value)
    write_csv(root / "tables/physical_coverage.csv", coverage)
    write_csv(root / "tables/physical_candidates.csv", rows)
    write_csv(root / "tables/motion_coordinate_separation.csv", query_rows)
    write_csv(root / "tables/rollout_runtime_counts.csv", runtime_rows)
    reference_rows = [r for r in rows if r.get("kind") == "reference"]
    reference_counts = Counter(r.get("status", "unknown") for r in reference_rows)
    atomic_json(root / "tables/reference_grid_coverage.json", dict(
        logical_candidate_records=len(reference_rows), statuses=dict(reference_counts),
        required_interaction_ids=[0], required_preanswer_t=[0, 127],
        expected_reference_candidates=3528,
        note="Seven saved reference scales for every method and setting at the preset interaction/rounds. Exact cache hits count as covered, and are distinct from unique executions."))
    calibration = read_calibration(root)
    for key in ("motion_sensitivity", "fixed_command_audit", "restore_order_checks"):
        records = calibration.get(key, [])
        if isinstance(records, dict):
            records = [records]
        if isinstance(records, list):
            converted = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
                         for r in records if isinstance(r, dict)]
            write_csv(root / "tables" / f"robot_{key}.csv", converted)
    return coverage, len(rows), len(query_rows)


LIMITATIONS = """The oracle is defined on the original q coordinates, not on trajectory features. Simulator replay does not change statistical histories or curves under the same statistical random streams. The clips are short, open-loop realizations; no RL policy is learned and no RL return comparison is made. Command reconstruction does not establish human perceptual identifiability, trajectory-feature geometry, collision freedom, real-robot safety, or task success. Five finite interactions with T=128 do not establish an asymptotic limit or rate; rounds within an interaction are not independent replicates. The beta sweep is correctly specified (oracle beta equals inference beta), not a misspecification experiment. SG remains in stage 0, with scale 1 and fixed stage center, for every measured query. Numerical posterior and EIG approximations retain finite-resolution and finite-particle error; unresolved comparisons are reported separately."""


def generate_report(run_dir, config=None, make_plots=True):
    root = Path(run_dir)
    config = report_config(root, config)
    experiment = config["experiment"]
    n_interactions = len(experiment["interaction_ids"])
    expected_runs = len(experiment["settings"]) * len(experiment["methods"]) * n_interactions
    expected_answers = expected_runs * experiment["answers_per_run"]
    expected_requests = sum(s["choices"] for s in experiment["settings"]) * len(experiment["methods"]) * n_interactions * experiment["answers_per_run"]
    runs = load_runs(root)
    _, means = endpoint_tables(runs, root, n_interactions)
    figures = plot_run(root, runs, config=config) if make_plots and runs else []
    coverage, candidates, query_count = physical_tables(root, runs)
    complete = sum(len(r["data"]["y"]) == 128 for r in runs)
    answers = sum(len(r["data"]["y"]) for r in runs)
    logical = sum(int(np.prod(r["data"]["q"].shape[:2])) for r in runs)
    validation = read_json(root / "validation/summary.json")
    calibration = read_calibration(root)
    write_csv(root / "tables/statistical_runtime.csv", [dict(setting_id=r["setting_id"], method=r["method"],
              interaction_id=r["interaction_id"], actual_answers=len(r["data"]["y"]),
              wall_seconds=r["meta"].get("elapsed_seconds"), device=r["meta"].get("device")) for r in runs])
    reused = [r for r in runs if r["meta"].get("reused_from")]
    covered = sum(c.get("covered", 0) for c in coverage)
    failed = sum(c.get("failed", 0) for c in coverage)
    unknown = max(0, logical - covered - failed)
    videos = sorted(str(p.relative_to(root)) for p in (root / "videos").rglob("*.mp4"))
    references = read_json(root / "tables/reference_grid_coverage.json")
    reference_covered = sum(references.get("statuses", {}).get(s, 0) for s in ("executed", "cache_hit"))
    sensitivity = calibration.get("motion_sensitivity", [])
    ranks = [r.get("numerical_rank") for r in sensitivity]
    sensitivities = [v for r in sensitivity for v in r.get("coordinate_rms_sensitivity_meters_per_q", [])]
    sensitivity_text = (f"Finite-difference motion checks: {len(sensitivity)}; observed ranks: {sorted(set(ranks))}. "
                        f"RMS motion sensitivity per unit q ranges from {min(sensitivities):.6g} to {max(sensitivities):.6g} meters. "
                        if sensitivities else "Motion sensitivity results are not yet available. ")
    numerical_status = validation.get("status", "not_run")
    summary = dict(expected_runs=expected_runs, observed_runs=len(runs), completed_runs=complete,
                   actual_answers=answers, expected_answers=expected_answers, logical_selected_requests=logical,
                   expected_logical_selected_requests=expected_requests, covered_selected_requests=covered,
                   failed_selected_requests=failed, pending_or_unidentified_requests=unknown,
                   numerical_validation_status=numerical_status, videos=len(videos),
                   expected_videos=36, reference_covered_candidates=reference_covered,
                   expected_reference_candidates=3528,
                   reused_statistical_runs=len(reused), figures=figures)
    atomic_json(root / "tables/completion_summary.json", summary)
    lines = ["# RC-MLQ × Meta-World realization audit", "",
             f"Observed artifacts: **{complete}/{expected_runs} complete statistical runs**, {answers:,}/{expected_answers:,} actual answers. "
             f"Saved selected-candidate requests: {logical:,}/{expected_requests:,}. This report is an artifact snapshot, not a claim that unfinished work has completed.", "",
             "## Statistical results", "",
             f"The experiment uses d=4, 128 answers, seven settings and {n_interactions} preassigned interaction IDs. "
             "The M=4, beta=4 anchor is shared between sweeps and is counted once. "
             "Risk is the trace of the entire posterior covariance. Realized squared error is stored separately. "
             "EIG ratios use each method's own pre-answer posterior and its seven-scale line reference; values above one are retained.", "",
             f"The four figures use unsmoothed individual curves and arithmetic means. Dashed means indicate incomplete {n_interactions}-interaction coverage. "
             "Per-round valid counts are in tables/curve_effective_counts.csv. Endpoint tables use posterior answer counts 32/64/128 and eta indices 31/63/127. "
             "Any reported SE is descriptive between-interaction standard error, not a claim of significance.", "",
             "## Physical audit", "",
             f"Selected requests with executed/cache-hit records: {covered:,}; failed: {failed:,}; pending or not yet identified: {unknown:,}. "
             f"Loaded {candidates:,} candidate records and {query_count:,} query diagnostic records. "
             "tables/physical_coverage.csv separates requested, executed, cache-hit and failed counts by method and setting. "
             "tables/physical_candidates.csv records reconstruction, validity, saturation and step diagnostics; "
             "tables/motion_coordinate_separation.csv preserves q-distance and executed-motion comparisons.", "",
             f"Engineering calibration artifact: {'available' if calibration else 'not yet available'}. "
             + sensitivity_text + "Its motion-Jacobian singular values and sensitivity diagnostics must be interpreted alongside the realized trajectory differences. "
             f"Command rank alone does not imply full-rank physical motion. Reference-grid audit coverage is {reference_covered:,}/3,528 candidate records, separate from selected-query coverage.", "",
             f"Qualitative robot videos saved: {len(videos)}. " + (", ".join(videos) if videos else "Required preset clips remain incomplete until video artifacts exist."), "",
             "## Numerical checks and provenance", "",
             f"Frozen-history validation status: **{numerical_status}**. See validation/summary.json and numerical discrepancies in tables/. "
             "Independent particle references report reference uncertainty separately; a noisy replay is not exact ground truth. "
             "Unresolved discrepancies do not disappear by increasing only the reference particle count.", "",
             f"Reused statistical prefixes: {len(reused)} runs. "
             "No reuse is claimed unless its source is present in run metadata. Simulator execution is a realization replay of stored selected questions. "
             "The supplied execution specification defines the implementation; the named source manuscript was absent from the initial repository, "
             "so no independent manuscript-to-code audit is claimed. See PROVENANCE.md and LIMITATIONS.md in the implementation directory.", "",
             "Runtime, hardware, environment, seeds, frozen configuration and code hashes are recorded in manifest.json, HARDWARE.json, ENVIRONMENT.txt, PROFILE.json, ETA.md and STATUS.md when generated by the scheduler.", "",
             "## Required interpretation limits", "", LIMITATIONS.replace("Five finite", f"{n_interactions} finite"), "",
             "## Completion gaps", "",
             f"Unfinished statistical runs: {expected_runs-complete}. Uncovered or unidentified selected requests: {unknown:,}. "
             f"Numerical status: {numerical_status}. Preset videos missing: {max(0, 36-len(videos))}; reference candidates uncovered: {max(0,3528-reference_covered)}. "
             "No algorithm is required to win; transient advantages and unfavorable interactions are preserved.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary
