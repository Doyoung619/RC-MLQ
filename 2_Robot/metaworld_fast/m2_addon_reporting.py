"""Count-aware, read-only merge of the frozen M=2 controls and addon results.

The legacy ``eta`` field is exposed as eta_grid, never as an ellipsoid optimum.
Missing runs/values remain missing. SG is only used by the schedule appendix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .reporting import atomic_json, finite_float, read_json, write_csv

CONTROLS = ("RC", "PQ", "QI", "PS", "GC")
NEW_METHODS = ("sm_continuous", "mem_line", "gmem_line", "2mem_line_2sweeps",
               "infosynth_trace", "infosynth_bounded", "gausscloud_paper_L1",
               "gausscloud_paper_L3", "gausscloud_paper_L5")
METHODS = CONTROLS + NEW_METHODS
LABELS = dict(zip(METHODS, ("RC", "PQ", "QI-1", "PS", "GC", "SM", "MEM-line",
              "GMEM-line", "2MEM-line (2 sweeps)", "IT", "IB", "GCQ-1", "GCQ-3", "GCQ-5")))
LABELS["SG"] = "SG-legacy"
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
              "#542788", "#1b7837", "#a6611a", "#d73027", "#008B8B", "#e7298a",
              "#4d4d4d", "#756bb1", "#2166ac")))
COLORS["SG"] = "#999999"
METRICS = (("eta_grid", "eta", 128, "EIG(displayed Q) / max EIG(seven grid Q)"),
           ("posterior_risk", "risk", 129, "Full posterior covariance trace"),
           ("realized_sq_error", "error", 129, "Squared parameter error"))


def report_context(run_dir, config=None):
    root = Path(run_dir)
    lock = read_json(root / "BASELINE_LOCK.json")
    manifest = read_json(root / "manifest.json")
    if config is None:
        from .util import load_config
        paths = [root / "frozen_config.yaml", root / "config_frozen.yaml"]
        config = next((load_config(p) for p in paths if p.exists()), {})
    experiment = config.get("experiment", {})
    ids = lock.get("interaction_ids", experiment.get("interaction_ids", manifest.get("interaction_ids")))
    if ids is None:
        raise ValueError("Frozen interaction_ids are required; reporting never assumes ten IDs")
    source = lock.get("source_run", manifest.get("source_run", experiment.get("base_run_dir")))
    if source is None:
        raise ValueError("Frozen baseline source_run is required")
    return dict(root=root, config=config, lock=lock, manifest=manifest,
                interaction_ids=list(ids), source_run=Path(source),
                oracle=lock.get("oracle", dict(alpha_E=.5, rho_E=4 + 2*np.sqrt(3))))


def load_runs(run_dir, config=None, include_legacy=False):
    context = report_context(run_dir, config)
    roots = ((context["source_run"], "reused"), (context["root"], "new"))
    runs = {}
    allowed = METHODS + (("SG",) if include_legacy else ())
    for origin, disposition in roots:
        for path in sorted((origin / "statistical").glob("m02_b04__*.npz")):
            meta = read_json(path.with_suffix(".json"))
            if not meta:
                raise ValueError(f"Missing statistical metadata: {path}")
            method = meta.get("method")
            if method not in allowed or disposition == "reused" and method not in CONTROLS + ("SG",):
                continue
            interaction = meta.get("interaction_id")
            if interaction not in context["interaction_ids"]:
                raise ValueError(f"Unexpected interaction in {path}: {interaction}")
            if (meta.get("setting_id") != "m02_b04" or int(meta.get("M", 2)) != 2
                    or float(meta.get("beta", 4)) != 4 or int(meta.get("dimension", 4)) != 4
                    or int(meta.get("answers", 128)) != 128):
                raise ValueError(f"Incompatible scientific setting: {path}")
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
            if not {"q", "y", "risk", "eta"}.issubset(data):
                raise ValueError(f"Missing mandatory numerical arrays: {path}")
            if meta.get("reference_scope", "own_posterior_seven_scale_grid") not in (
                    "own_posterior_seven_scale_grid", "same_method_same_posterior_seven_scale_max"):
                raise ValueError(f"Cannot relabel an unknown denominator eta_grid: {path}")
            identity = method, interaction
            if identity in runs:
                old = runs[identity]
                if not all(k in data and np.array_equal(v, data[k], equal_nan=True)
                           for k, v in old["data"].items() if np.issubdtype(v.dtype, np.number)):
                    raise ValueError(f"Conflicting reused/addon run: {identity}")
                # A copied control is counted once and remains a reused result.
                continue
            runs[identity] = dict(path=path, root=origin, setting_id="m02_b04", method=method,
                                  interaction_id=interaction, data=data, meta=meta,
                                  disposition=disposition, label=LABELS[method])
    order = {method: i for i, method in enumerate(allowed)}
    return sorted(runs.values(), key=lambda r: (order[r["method"]], r["interaction_id"]))


def _summary(values, expected):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(values)
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    return dict(mean=float(values.mean()) if n else None, sample_sd=sd,
                descriptive_se=sd / np.sqrt(n) if sd is not None else None,
                n_effective=n, n_expected=expected, complete_expected=n == expected)


def numerical_tables(run_dir, runs, interaction_ids):
    root = Path(run_dir)
    lookup = {(r["method"], r["interaction_id"]): r for r in runs}
    per_round, means, endpoints, endpoint_means, statuses = [], [], [], [], []
    for method in METHODS:
        for interaction in interaction_ids:
            run = lookup.get((method, interaction))
            count = len(run["data"]["y"]) if run else 0
            statuses.append(dict(method=method, label=LABELS[method], interaction_id=interaction,
                                 answers_completed=count, status="complete" if count == 128 else "partial" if count else "missing",
                                 disposition=run["disposition"] if run else "pending_new" if method in NEW_METHODS else "missing_control",
                                 source=str(run["path"]) if run else None))
        for metric, key, length, _ in METRICS:
            for t in range(length):
                values = []
                for interaction in interaction_ids:
                    run = lookup.get((method, interaction))
                    data = run["data"] if run else {}
                    available = key in data and t < len(data[key]) and t < len(data.get("y", [])) + (key != "eta")
                    value = finite_float(data[key][t]) if available else None
                    values.append(value)
                    row = dict(method=method, label=LABELS[method], interaction_id=interaction,
                               metric=metric, t=t, indexing="pre_answer" if key == "eta" else "post_answer_state",
                               value=value, available=value is not None,
                               source=str(run["path"]) if run else None)
                    per_round.append(row)
                    if t in ((31, 63, 127) if key == "eta" else (32, 64, 128)):
                        endpoints.append(dict(row, answer_count=t + (key == "eta")))
                mean = dict(method=method, label=LABELS[method], metric=metric, t=t,
                            **_summary(values, len(interaction_ids)))
                means.append(mean)
                if t in ((31, 63, 127) if key == "eta" else (32, 64, 128)):
                    endpoint_means.append(dict(mean, answer_count=t + (key == "eta")))
    for name, rows in (("all_metrics_per_interaction", per_round), ("all_metrics_mean", means),
                       ("endpoints_per_interaction", endpoints), ("endpoints_mean", endpoint_means),
                       ("method_status", statuses)):
        write_csv(root / "tables" / f"{name}.csv", rows)
    return means, endpoint_means, statuses


def schedule_tables(run_dir, runs):
    rows = []
    for run in runs:
        if run["method"] not in ("SG", "gausscloud_paper_L1", "gausscloud_paper_L3", "gausscloud_paper_L5"):
            continue
        data = run["data"]
        for t in range(len(data["y"])):
            stage = int(data["selector_stage"][t]) if "selector_stage" in data else None
            center = data["selector_center"][t] if "selector_center" in data else None
            if "selector_std" in data:
                std = finite_float(data["selector_std"][t]); scale_source = "recorded_selector_std"
            elif run["method"] == "SG" and stage is not None:
                std = 2.**(-stage); scale_source = "derived_from_frozen_legacy_formula_and_recorded_stage"
            else:
                std = None; scale_source = "missing"
            change = None if center is None else 0. if t == 0 else finite_float(np.linalg.norm(center - data["selector_center"][t-1]))
            rows.append(dict(method=run["method"], label=run["label"], interaction_id=run["interaction_id"],
                             t=t, stage=stage, std=std, std_source=scale_source,
                             center_change_l2=change, center=json.dumps(center.tolist()) if center is not None else None,
                             cap_activated=bool(np.any(data["cap_activated"][t])) if "cap_activated" in data else None,
                             cap_candidate_flags=json.dumps(np.asarray(data["cap_activated"][t]).tolist()) if "cap_activated" in data else None))
    write_csv(Path(run_dir) / "tables/schedule_audit.csv", rows)
    return rows


def randomized_tables(run_dir, runs):
    """Conditional query-randomization averages are separate from realized eta."""
    root = Path(run_dir); rows = []
    for run in runs:
        data = run["data"]
        if run["method"] == "SG" or "diag_t" not in data or "diag_eig" not in data:
            continue
        sidecar = read_json(root/"oracle_sidecars"/f"{run['path'].stem}.json")
        bounds = {r["t"]:r for r in sidecar.get("rounds",[])}
        observations = []
        for j,t in enumerate(data["diag_t"]):
            t = int(t); draws = np.asarray(data["diag_eig"][j],dtype=float)
            finite = draws[np.isfinite(draws)]; count = len(finite)
            reference = float(data["reference_eig"][t])
            mean = float(np.mean(finite)) if count else None
            se = float(np.std(finite,ddof=1)/np.sqrt(count)) if count > 1 else None
            bound = bounds.get(t,{})
            resolved = bound.get("resolved",False) and mean is not None
            item = dict(method=run["method"],interaction_id=run["interaction_id"],t=t,
                        draws_expected=64,draws_observed=len(draws),finite_draws=count,
                        mean_eig=mean,conditional_query_eig_se=se,
                        mean_eta_grid=mean/reference if mean is not None and reference>0 else None,
                        conditional_query_eta_grid_se=se/reference if se is not None and reference>0 else None,
                        mean_eta_ellipsoid_lower=mean/bound["eig_ellipsoid_upper"] if resolved else None,
                        mean_eta_ellipsoid_upper=mean/bound["eig_ellipsoid_lower"] if resolved else None,
                        posterior_and_history_frozen=True,actual_answers_added=0,
                        interpretation="Conditional average over query randomization on one frozen posterior; not an interaction-level uncertainty estimate. Ellipsoid ratio endpoints are not clipped because per-draw containment is not certified.")
            rows.append(item)
            observations.append(dict(item,eig_draws=[finite_float(v) for v in draws]))
        if observations:
            atomic_json(root/"randomized_diagnostics"/f"{run['path'].stem}.report.json",dict(
                source=str(run["path"]),method=run["method"],interaction_id=run["interaction_id"],rounds=observations))
    write_csv(root/"tables/randomized_query_diagnostics.csv",rows)
    return rows


def physical_tables(run_dir, runs):
    """Read only selected logical clips for the exact merged run identities."""
    root = Path(run_dir)
    coverage, candidates, queries, runtime = [], [], [], []
    for run in runs:
        if run["method"] == "SG":
            continue
        path = run["root"] / "rollouts" / f"{run['path'].stem}.json"
        record = read_json(path)
        selected = []
        requests = record.get("per_query", record.get("requests", []))
        seen = set()
        for index, item in enumerate(record.get("per_candidate", [])):
            request_index = item.get("request_index")
            request = requests[request_index] if request_index is not None and request_index < len(requests) else {}
            if request.get("kind", item.get("kind", "selected")) != "selected":
                continue
            identity = item.get("request_id", request_index), item.get("candidate_slot", index)
            if identity in seen:
                raise ValueError(f"Duplicate logical clip record: {path}: {identity}")
            seen.add(identity)
            row = dict(item, setting_id="m02_b04", method=run["method"], interaction_id=run["interaction_id"],
                       disposition=run["disposition"], artifact=str(path))
            selected.append(row); candidates.append(row)
        counts = Counter(item.get("status", "unknown") for item in selected)
        logical = 2 * len(run["data"]["y"])
        covered = counts["executed"] + counts["cache_hit"]
        valid = sum(item.get("simulation_valid", False) for item in selected)
        reconstructed = sum(item.get("command_reconstruction_ok") is True for item in selected)
        reconstruction_failed = sum(item.get("command_reconstruction_ok") is False for item in selected)
        reconstruction_missing = sum(item.get("command_reconstruction_ok") is None for item in selected)
        command_bounds_ok = sum(item.get("action_bounds_ok") is True for item in selected)
        full_length = sum(item.get("actual_steps") == 50 for item in selected)
        if len(selected) > logical:
            raise ValueError(f"Rollout records exceed selected candidate count: {path}")
        errors = [item["command_reconstruction_error"] for item in selected if item.get("command_reconstruction_error") is not None]
        saturation = [item["workspace_saturation_rate"] for item in selected if item.get("workspace_saturation_rate") is not None]
        coverage.append(dict(method=run["method"], interaction_id=run["interaction_id"],
                             disposition=run["disposition"], logical_selected_requests=logical,
                             executed=counts["executed"], cache_hits=counts["cache_hit"], failed=counts["failed"],
                             covered=covered, pending=max(0, logical-covered-counts["failed"]), valid=valid,
                             reconstructed=reconstructed, reconstruction_failed=reconstruction_failed,
                             reconstruction_missing=reconstruction_missing, command_bounds_ok=command_bounds_ok,
                             full_length_clips=full_length,
                             coverage_fraction=covered/logical if logical else None,
                             validity_fraction=valid/logical if logical else None,
                             max_command_reconstruction_error=max(errors) if errors else None,
                             mean_workspace_saturation_rate=float(np.mean(saturation)) if saturation else None,
                             artifact=str(path) if path.exists() else None))
        queries.extend(dict(item, method=run["method"], interaction_id=run["interaction_id"], artifact=str(path))
                       for item in requests if item.get("kind", "selected") == "selected")
        if record.get("summary"):
            runtime.append(dict(method=run["method"], interaction_id=run["interaction_id"],
                                disposition=run["disposition"], **record["summary"]))
    for name, rows in (("physical_coverage", coverage), ("physical_candidates", candidates),
                       ("motion_coordinate_separation", queries), ("rollout_runtime", runtime)):
        serializable = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
        write_csv(root / "tables" / f"{name}.csv", serializable)
    return coverage


def plot_run(run_dir, runs, means, schedules, expected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(run_dir); (root / "figures").mkdir(parents=True, exist_ok=True)
    outputs = []
    names = {"eta_grid": "binary_eta_grid", "posterior_risk": "binary_posterior_risk",
             "realized_sq_error": "binary_realized_sq_error"}

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = root / "figures" / f"{name}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight"); outputs.append(str(path))
        plt.close(fig)

    for group, methods in (("all", METHODS), ("controls", CONTROLS), ("binary", NEW_METHODS[:6]), ("gaussian", NEW_METHODS[6:])):
        for metric, key, length, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(13, 7))
            fig.subplots_adjust(bottom=.30)
            for method in methods:
                for run in runs:
                    if run["method"] == method and key in run["data"]:
                        values = run["data"][key][:len(run["data"]["y"]) + (key != "eta")]
                        ax.plot(np.arange(len(values)), values, color=COLORS[method], alpha=.17, lw=.45)
                series = [r for r in means if r["method"] == method and r["metric"] == metric]
                values = [r["mean"] if r["mean"] is not None else np.nan for r in series]
                ax.plot(range(length), values, color=COLORS[method], lw=1.8,
                        linestyle="-" if all(r["complete_expected"] for r in series) else "--")
            ax.set(xlim=(0, length-1), xlabel="Answers already received, t", ylabel=ylabel,
                   title="M=2, d=4, β=4; independent histories per method")
            ax.grid(alpha=.15)
            ax.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in methods],
                      loc="upper center", bbox_to_anchor=(.5, -.19), ncol=min(5, len(methods)), frameon=False)
            fig.text(.5, .02, f"Thin: individuals; thick: arithmetic mean; dashed: fewer than {expected} finite values. No smoothing or ratio clipping.",
                     ha="center", fontsize=9)
            save(fig, f"{names[metric]}_{group}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for method in (*NEW_METHODS[6:], "SG"):
        for interaction in sorted({r["interaction_id"] for r in schedules if r["method"] == method}):
            rows = [r for r in schedules if r["method"] == method and r["interaction_id"] == interaction]
            for ax, field in zip(axes, ("std", "center_change_l2")):
                ax.plot([r["t"] for r in rows], [r[field] for r in rows], color=COLORS[method], alpha=.4, lw=.9)
    axes[0].set_ylabel("Proposal standard deviation"); axes[1].set_ylabel("Stage center change, L2")
    axes[1].set_xlabel("Pre-answer t")
    axes[0].legend(handles=[Line2D([], [], color=COLORS[m], label=LABELS[m]) for m in (*NEW_METHODS[6:], "SG")], ncol=4)
    fig.suptitle("Recorded schedules; GCQ and SG also differ in initial scale")
    save(fig, "binary_schedule_audit")
    sidecars = []
    for path in sorted((root / "oracle_sidecars").glob("*.json")):
        record = read_json(path)
        sidecars.extend(dict(method=record.get("method"), interaction_id=record.get("interaction_id"), **r)
                        for r in record.get("rounds", []) if r.get("t") in (0,31,63,95,127))
    write_csv(root / "tables/ellipsoid_checkpoints.csv", sidecars)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for m_index, method in enumerate(METHODS):
        rows = [r for r in sidecars if r["method"] == method]
        for t in (0,31,63,95,127):
            subset = [r for r in rows if r["t"] == t and r.get("resolved", False)]
            if not subset:
                continue
            x = m_index + .14*((0,31,63,95,127).index(t)-2)
            for ax, lo, hi in ((axes[0], "eig_ellipsoid_lower", "eig_ellipsoid_upper"),
                               (axes[1], "eta_ellipsoid_lower", "eta_ellipsoid_upper")):
                low = np.mean([r[lo] for r in subset]); high = np.mean([r[hi] for r in subset])
                ax.plot([x,x], [low,high], color=COLORS[method], alpha=.65, lw=2)
                ax.scatter([x,x], [low,high], color=COLORS[method], s=9)
    axes[0].set_ylabel("Mean feasible lower / analytic upper EIG")
    axes[1].set_ylabel("Mean ratio interval endpoints")
    axes[1].set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=45, ha="right")
    fig.suptitle("Ellipsoid bound intervals, t=0/31/63/95/127 (left to right within each method)\nNumerical-posterior bounds; no global optimization or confidence intervals")
    fig.tight_layout(); save(fig, "binary_ellipsoid_diagnostics")
    return outputs


def generate_report(run_dir, config=None, make_plots=True):
    context = report_context(run_dir, config); root = context["root"]
    runs = load_runs(root, config, include_legacy=True)
    main = [r for r in runs if r["method"] != "SG"]
    means, endpoints, statuses = numerical_tables(root, main, context["interaction_ids"])
    schedules = schedule_tables(root, runs)
    randomized = randomized_tables(root, main)
    coverage = physical_tables(root, main)
    figures = plot_run(root, main, means, schedules, len(context["interaction_ids"])) if make_plots else []
    write_csv(root / "tables/statistical_runtime.csv", [dict(method=r["method"], interaction_id=r["interaction_id"],
              disposition=r["disposition"], answers=len(r["data"]["y"]),
              elapsed_seconds=r["meta"].get("elapsed_seconds"), device=r["meta"].get("device")) for r in main])
    validation = read_json(root / "validation/summary.json")
    base_validation = read_json(context["source_run"] / "validation/summary.json")
    interactions = context["lock"].get("interactions", [])
    distinct_latents = len({json.dumps(r.get("theta_star")) for r in interactions if r.get("theta_star") is not None})
    distinct_contexts = len({r["context_id"] for r in interactions if r.get("context_id") is not None})
    expected = len(METHODS)*len(context["interaction_ids"])
    completed = [r for r in statuses if r["status"] == "complete"]
    video_manifest = read_json(root / "selected_videos/manifest.json")
    video_records = video_manifest.get("clips", [])
    expected_videos = {(method, context["interaction_ids"][0], t, candidate)
                       for method in ("RC","sm_continuous","gausscloud_paper_L3")
                       for t in (0,63,127) for candidate in (0,1)}
    covered_videos = {(r.get("method"),r.get("interaction_id"),r.get("t"),r.get("candidate_slot"))
                      for r in video_records if r.get("status") in ("reused","executed")
                      and r.get("path") and Path(r["path"]).is_file() and Path(r["path"]).stat().st_size > 0}
    videos_complete = covered_videos == expected_videos and len(video_records) == 18
    physical_complete = len(completed) == expected and len(coverage) == expected and all(
        row["covered"] == row["valid"] == row["reconstructed"] == row["command_bounds_ok"]
        == row["full_length_clips"] == row["logical_selected_requests"] and row["failed"] == 0 for row in coverage)
    summary = dict(expected_runs=expected, completed_runs=len(completed),
                   reused_completed_runs=sum(r["disposition"] == "reused" for r in completed),
                   new_completed_runs=sum(r["disposition"] == "new" for r in completed),
                   new_actual_answers=sum(len(r["data"]["y"]) for r in main if r["disposition"] == "new"),
                   expected_interaction_ids=context["interaction_ids"],
                   observed_answers=sum(len(r["data"]["y"]) for r in main),
                   covered_selected_clips=sum(r["covered"] for r in coverage),
                   failed_selected_clips=sum(r["failed"] for r in coverage),
                   pending_selected_clips=sum(r["pending"] for r in coverage),
                   numerical_validation_status=validation.get("numerical_status", "not_run"),
                   validation_status=validation.get("status", "not_run"), figures=figures,
                   baseline_numerical_validation_status=base_validation.get("status", "not_available"),
                   randomized_diagnostic_checkpoints=len(randomized),
                   physical_complete=physical_complete, videos_complete=videos_complete,
                   covered_precommitted_videos=len(covered_videos), expected_precommitted_videos=18,
                   invalid_selected_clips=sum(r["covered"]-r["valid"] for r in coverage),
                   reconstruction_failed_selected_clips=sum(r["reconstruction_failed"] for r in coverage),
                   reconstruction_missing_selected_clips=sum(r["reconstruction_missing"] for r in coverage),
                   scientific_complete=len(completed) == expected and validation.get("status") == "passed"
                       and physical_complete and videos_complete)
    atomic_json(root / "tables/completion_summary.json", summary)
    lines = ["# M=2 incremental benchmark report", "",
             f"Completed statistical runs: **{len(completed)}/{expected}**; reused controls: **{summary['reused_completed_runs']}**; new runs: **{summary['new_completed_runs']}**.",
             f"Setting: d=4, M=2, response/inference β=4, T=128. Frozen interaction IDs: {context['interaction_ids']}.",
             f"Statistical baseline: `{context['source_run']}`. Legacy SG is supplementary only; the main registry has all 14 configurations.", "",
             "## Completion and reuse", "", "| Method | Reused | New complete | Missing/partial |", "|---|---:|---:|---:|"]
    for method in METHODS:
        rows = [r for r in statuses if r["method"] == method]
        lines.append(f"| {LABELS[method]} | {sum(r['status']=='complete' and r['disposition']=='reused' for r in rows)} | {sum(r['status']=='complete' and r['disposition']=='new' for r in rows)} | {sum(r['status']!='complete' for r in rows)} |")
    lines += ["", f"New actual answers observed: {summary['new_actual_answers']}. Numerical replay and diagnostic EIG calls do not add preference answers.",
              f"The frozen mapping records {distinct_latents} distinct latent value vectors and {distinct_contexts} simulator context IDs across {len(interactions)} interaction records. Context IDs and actual theta values are inherited without remapping; counts alone are not treated as proof of independent random draws.",
              "Missing runs and nonfinite values are blank in tables, never zero-filled. Every mean records its effective n, sample SD and descriptive SE. Individual curves and unsmoothed arithmetic means are provided; no p-values or asymptotic rate claims are made.",
              "", "## Metrics and numerical limits", "",
              "`eta_grid` is the saved `eta` field: displayed-query integrated EIG divided by the maximum integrated EIG among seven frozen line-grid queries on that method's own pre-answer posterior. Values above one remain visible. Fisher scores are not substituted for EIG.",
              "`posterior_risk` is the full covariance trace at post-answer states 0…128. `realized_sq_error` is squared distance from the saved posterior mean to the shared interaction's theta_star. These quantities are distinct.",
              "`tables/randomized_query_diagnostics.csv` separately records the 64-draw conditional EIG/eta averages at preset checkpoints. Its SE measures query randomization at a frozen posterior, not between-interaction or posterior approximation uncertainty. No diagnostic query adds an actual answer, and no ellipsoid ratio is clipped without per-draw containment evidence.",
              f"Ellipsoid protocol: alpha_E={context['oracle'].get('alpha_E',.5)}, rho_E={context['oracle'].get('rho_E',4+2*np.sqrt(3))}. Each sidecar retains a feasible-grid lower EIG and the analytic upper min(log 2, 2 β² rho_E² trace). The displayed query strengthens the lower bound only after containment is verified. Ratio intervals use N/U…N/L; the right endpoint is capped at one only for an inside query. No full ellipsoid optimization is required or claimed.",
              f"Artifact validation: **{validation.get('artifact_status','not_run')}**. Independent frozen-history numerical validation: **{validation.get('numerical_status','not_run')}**. Unresolved numerical comparisons remain in validation and integration-accuracy tables; no numerical-reference result replaces a main adaptive history.",
              f"Original baseline numerical report status: **{base_validation.get('status','not_available')}**. That broader report is preserved. The addon adds the prespecified M=2 checks and does not silently relabel the original validation as passed. CPU/GPU fixed-history arithmetic status: **{validation.get('cpu_gpu_arithmetic_status','not_run')}**.",
              "The analytic uniform prior has mean zero and covariance I/3, trace 4/3. Reused SMC controls preserve their empirical initial particle moments; their initial numerical deviations are recorded rather than overwritten. The analytic target is not claimed to have been computed exactly by a finite-particle method.",
              "", "## Physical coverage and runtime", "",
              f"Covered selected candidate clips: {summary['covered_selected_clips']}; failed: {summary['failed_selected_clips']}; pending among observed histories: {summary['pending_selected_clips']}. Required new clips for the complete nine-configuration addon: {9*len(context['interaction_ids'])*128*2}.",
              f"Physical completeness (coverage, validity, command reconstruction/bounds, all 50 steps): **{physical_complete}**. Precommitted qualitative clips present: **{len(covered_videos)}/18**; required subset complete: **{videos_complete}**.",
              "Physical tables distinguish reused/executed/cache-hit/failed clips, submitted-command reconstruction, workspace saturation and motion-coordinate separation. They audit the existing H=50 realization adapter; response utility remains a function of q coordinates, not trajectories. They do not establish task success, perceptual identifiability, or RL performance.",
              "Runtime tables separate reused wall times from newly executed statistical/rollout work. PROFILE.json/ETA.md retain the measured scheduling assumptions. Missing runtime is not estimated from answer counts.",
              "", "## Source fidelity and interpretation", "",
              "GC remains the posterior-covariance Gaussian baseline. GCQ-1/3/5 use the pinned author's finite-stage allocation (remainder in the final stage), halved standard deviation and stage-entry recentering. Continuous pairs, prior-SD initial scale 1/sqrt(3), shared Bayes mean, and radius-32 cap are transfers. Original catalog retrieval and ν-SVM estimation are not reproduced.",
              "GCQ configurations are finite-horizon competitors; no automatic U_C membership or vanishing-ratio/rate assertion applies. SG-legacy retains scale one and initial center for all 128 queries. Its differences from GCQ include initial scale as well as schedule, so performance differences do not isolate a schedule effect.",
              "SM transfers SAMPLEMIRROR reflection geometry. MEM-line is a coordinate-line numerical specialization, not arbitrary-dimensional global MEM; 2MEM uses the prescribed two sweeps. IT/IB transfer components of Info-Synth, not its full end-to-end implementation. Exact-model membership claims require each manuscript's policy/feasibility assumptions and are separate from numerical approximation.",
              "`PROVENANCE.md` and `METHOD_SOURCE_AUDIT.md` distinguish [BASE], [SOURCE], [TRANSFER] and [ENGINEERING]. No result-dependent method, seed or schedule adjustment is applied.",
              "", "## Artifacts", "",
              "Full numerical values: `tables/all_metrics_per_interaction.csv`, `tables/all_metrics_mean.csv`; endpoints: `tables/endpoints_*.csv`; status: `tables/method_status.csv`; schedule: `tables/schedule_audit.csv`; ellipsoid checkpoints: `tables/ellipsoid_checkpoints.csv`; validation: `validation/summary.json`.",
              "All-method figures accompany grouped figures. Existing M>2 data/figures and all original SG artifacts remain in the baseline output root.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary
