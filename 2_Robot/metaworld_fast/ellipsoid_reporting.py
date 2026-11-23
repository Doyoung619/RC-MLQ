"""Report full-dimensional achieved EIG references without claiming global solves.

All writes belong to a new reevaluation directory. Original histories and plots
are read only. A ten-interaction mean always carries its effective sample count.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .reporting import COLORS, METHODS, SETTINGS, atomic_json, finite_float, read_json, write_csv

EXPECTED_SEEDS = 10
EXPECTED_RUNS = 420
EXPECTED_ROUNDS = 128
METRICS = ("eig_actual", "eig_grid", "eig_achieved", "eig_upper", "eta_ellipsoid_approx",
           "eta_grid_recomputed", "eta_lower_plugin", "eta_upper_plugin")
SWEEPS = (("m", ("m02_b04", "m04_b04", "m08_b04", "m16_b04")),
          ("beta", ("m04_b02", "m04_b04", "m04_b08", "m04_b16")))


def identity(path):
    setting, method, interaction = Path(path).stem.split("__")
    if method not in METHODS or setting not in {s[0] for s in SETTINGS} or not interaction.startswith("i"):
        raise ValueError(f"Unrecognized statistical identity: {path}")
    interaction = int(interaction[1:])
    if interaction not in range(EXPECTED_SEEDS):
        raise ValueError(f"Expected preassigned interaction IDs 0..9, got {interaction}")
    return setting, method, interaction


def load_artifacts(run_dir):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    source = manifest.get("source_run")
    if not source:
        raise ValueError("Reevaluation manifest must identify source_run")
    source = Path(source)
    if not source.is_absolute():
        source = (root / source).resolve()
    if source.resolve() == root:
        raise ValueError("Reevaluation must use a separate output directory")
    runs = []
    source_keys = ("q", "y", "mu", "cov", "risk", "error", "eta", "eig", "reference_eig")
    for path in sorted((source / "statistical").glob("*.npz")):
        setting, method, interaction = identity(path)
        with np.load(path, allow_pickle=False) as archive:
            original = {k: archive[k] for k in source_keys if k in archive.files}
        evaluated = None
        target = root / "evaluated" / path.name
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                evaluated = {k: archive[k] for k in archive.files}
            t = np.asarray(evaluated.get("t", np.arange(len(evaluated["eig_actual"]))), dtype=int)
            if len(set(t.tolist())) != len(t) or np.any((t < 0) | (t >= EXPECTED_ROUNDS)):
                raise ValueError(f"Invalid or repeated pre-answer round indices in {target}")
            if not np.array_equal(t, np.sort(t)):
                raise ValueError(f"Round indices must be increasing: {target}")
            evaluated["t"] = t
            for name in METRICS:
                if name not in evaluated or evaluated[name].shape != t.shape:
                    raise ValueError(f"Missing/misaligned {name} in {target}")
            evaluated["relative_global_gap"] = (evaluated["eig_upper"] - evaluated["eig_achieved"]) / evaluated["eig_upper"]
            evaluated["absolute_global_gap"] = evaluated["eig_upper"] - evaluated["eig_achieved"]
            evaluated["reference_gain_over_grid"] = evaluated["eig_achieved"] - evaluated["eig_grid"]
        runs.append(dict(setting_id=setting, method=method, interaction_id=interaction, source_path=path,
                         evaluated_path=target if evaluated is not None else None,
                         original=original, evaluated=evaluated))
    expected_names = {r["source_path"].name for r in runs}
    orphaned = [p.name for p in (root / "evaluated").glob("*.npz") if p.name not in expected_names]
    if orphaned:
        raise ValueError(f"Evaluated artifacts without an original history: {orphaned[:3]}")
    return root, source, manifest, runs


def _series(run, metric):
    if metric == "risk":
        values = run["original"]["risk"]
        return np.arange(len(values)), values
    if metric == "eta_grid_original":
        values = run["original"]["eta"]
        return np.arange(len(values)), values
    if run["evaluated"] is None:
        return np.empty(0, dtype=int), np.empty(0)
    return run["evaluated"]["t"], run["evaluated"][metric]


def mean_curves(runs, setting, method, metric):
    selected = [r for r in runs if r["setting_id"] == setting and r["method"] == method]
    width = 129 if metric == "risk" else 128
    values = np.full((len(selected), width), np.nan)
    for index, run in enumerate(selected):
        t, series = _series(run, metric)
        values[index, t] = series
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.divide(np.nansum(values, axis=0), count, out=np.full(width, np.nan), where=count > 0)
    return values, mean, count


def plot_figures(root, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    generated, counts = [], []
    specifications = (
        ("eta_ellipsoid_approx", "Approx. ellipsoidal-reference ratio", "ellipsoid_eta",
         "Denominator: achieved integrated EIG from feasible full-dimensional search; global optimum not certified."),
        ("risk", "Full posterior covariance trace", "risk",
         "Original posterior-risk values are unchanged; reevaluation introduces no new answers."),
        ("relative_global_gap", "(Analytic upper − achieved EIG) / upper", "global_gap",
         "Upper is log M in these data. This is the unresolved optimization gap for estimated EIG, not a numerical error certificate."),
        ("eta_grid_original", r"Original auxiliary grid ratio $\eta_{t,\mathrm{grid}}$", "grid_eta_original",
         "Original seven-scale ratios are retained and relabeled; they are distinct from ellipsoidal-reference ratios."),
    )
    for metric, label, suffix, interpretation in specifications:
        for sweep, settings in SWEEPS:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True, constrained_layout=True)
            for ax, setting in zip(axes, settings):
                _, choices, beta = next(s for s in SETTINGS if s[0] == setting)
                ax.set_title(f"M={choices}, β={beta}, d=4")
                for method in METHODS:
                    individuals, mean, n = mean_curves(runs, setting, method, metric)
                    for values in individuals:
                        ax.plot(np.arange(len(values)), values, color=COLORS[method], alpha=.20, lw=.5)
                    ax.plot(np.arange(len(mean)), mean, color=COLORS[method], lw=1.8,
                            linestyle="-" if np.all(n == EXPECTED_SEEDS) else "--")
                    counts.extend(dict(sweep=sweep, metric=metric, setting_id=setting, method=method,
                                       t=t, n_effective=int(count), n_expected=EXPECTED_SEEDS)
                                  for t, count in enumerate(n))
                ax.set_xlim(0, 128 if metric == "risk" else 127)
                ax.set_xlabel("Answers already received, t")
                ax.grid(alpha=.15)
            axes[0].set_ylabel(label)
            fig.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label=m) for m in METHODS],
                       loc="outside upper center", ncol=6, frameon=False)
            fig.supxlabel("Thin: ten individual interactions. Thick: arithmetic mean; dashed if fewer than ten finite values.\n"
                          + interpretation + "\n"
                          + "α=0.5, ρE=16. Raw curves, no ratio clipping. SG stays in stage 0 for t=0,…,127.", fontsize=8.5)
            for extension in ("pdf", "png"):
                path = output / f"{sweep}_sweep_{suffix}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                generated.append(str(path))
            plt.close(fig)
    write_csv(root / "tables/curve_effective_counts.csv", counts)
    return generated


def endpoint_tables(root, runs):
    rows = []
    derived = ("relative_global_gap", "absolute_global_gap", "reference_gain_over_grid")
    for run in runs:
        base = {k: run[k] for k in ("setting_id", "method", "interaction_id")}
        evaluated = run["evaluated"]
        index = {int(t): i for i, t in enumerate(evaluated["t"])} if evaluated is not None else {}
        for answer_count in (32, 64, 128):
            t = answer_count - 1
            for metric in METRICS + derived + ("eta_grid_original", "risk", "error"):
                if metric in ("risk", "error"):
                    source = run["original"].get(metric, [])
                    value = source[answer_count] if len(source) > answer_count else None
                    metric_index = answer_count
                elif metric == "eta_grid_original":
                    value = run["original"]["eta"][t] if len(run["original"]["eta"]) > t else None
                    metric_index = t
                else:
                    value = evaluated[metric][index[t]] if t in index else None
                    metric_index = t
                value = finite_float(value) if value is not None else None
                rows.append(dict(**base, answer_count=answer_count, metric=metric, index=metric_index,
                                 value=value, available=value is not None))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["setting_id"], row["method"], row["answer_count"], row["metric"])].append(row["value"])
    means = []
    for (setting, method, answer_count, metric), values in sorted(grouped.items()):
        finite = [x for x in values if x is not None]
        n = len(finite)
        means.append(dict(setting_id=setting, method=method, answer_count=answer_count, metric=metric,
                          mean=float(np.mean(finite)) if n else None, n_effective=n, n_expected=EXPECTED_SEEDS,
                          complete_expected=n == EXPECTED_SEEDS,
                          descriptive_between_interaction_se=float(np.std(finite, ddof=1)/np.sqrt(n)) if n > 1 else None))
    write_csv(root / "tables/endpoints_per_interaction.csv", rows)
    write_csv(root / "tables/endpoints_mean.csv", means)
    return rows, means


def diagnostic_tables(root, runs):
    rows = []
    for run in runs:
        e, original = run["evaluated"], run["original"]
        if e is None:
            continue
        t = e["t"]
        covariance = original["cov"][t]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        shape_values = .5 + .5 * eigenvalues / eigenvalues[:, -1, None]
        mu = original["mu"][t]
        actual_q = original["q"][t]
        actual_normalized = np.einsum("tmi,tij->tmj", actual_q - mu[:, None, :], eigenvectors) / (16*np.sqrt(shape_values[:, None, :]))
        max_actual = np.linalg.norm(actual_normalized, axis=-1).max(axis=-1)
        max_achieved = np.full(len(t), np.nan)
        if "achieved_q" in e:
            z = np.einsum("tmi,tij->tmj", e["achieved_q"]-mu[:, None, :], eigenvectors) / (16*np.sqrt(shape_values[:, None, :]))
            max_achieved = np.linalg.norm(z, axis=-1).max(axis=-1)
        for i, round_index in enumerate(t):
            numerator = float(e["eig_actual"][i])
            achieved = float(e["eig_achieved"][i])
            grid = float(e["eig_grid"][i])
            upper = float(e["eig_upper"][i])
            inside = bool(max_actual[i] <= 1+1e-8)
            record = dict(setting_id=run["setting_id"], method=run["method"], interaction_id=run["interaction_id"], t=int(round_index),
                          actual_inside=inside, max_normz_actual=float(max_actual[i]),
                          max_normz_achieved=finite_float(max_achieved[i]),
                          achieved_feasible=bool(max_achieved[i] <= 1+1e-7) if np.isfinite(max_achieved[i]) else None,
                          achieved_contains_grid_value=achieved >= grid-1e-9,
                          achieved_contains_actual_value=achieved >= numerator-1e-9 if inside else None,
                          achieved_below_entropy_upper=achieved <= upper+1e-9,
                          eig_actual=numerator, eig_achieved=achieved, eig_upper=upper,
                          global_gap=upper-achieved, relative_global_gap=(upper-achieved)/upper,
                          actual_eig_minus_original=numerator-float(original["eig"][round_index]),
                          actual_eig_absolute_discrepancy=abs(numerator-float(original["eig"][round_index])),
                          eta_lower_plugin=float(e["eta_lower_plugin"][i]), eta_upper_plugin=float(e["eta_upper_plugin"][i]))
            # Keep arbitrary per-round optimizer/integration diagnostics available without reinterpreting them.
            for key, values in e.items():
                if key not in record and key not in METRICS and key not in ("t", "achieved_q"):
                    if np.asarray(values).ndim == 1 and len(values) == len(t):
                        value = values[i]
                        record[key] = value.item() if hasattr(value, "item") else value
            rows.append(record)
    write_csv(root / "tables/oracle_diagnostics_per_round.csv", rows)
    coverage = []
    for setting, _, _ in SETTINGS:
        for method in METHODS:
            group = [r for r in rows if r["setting_id"] == setting and r["method"] == method]
            identities = {r["interaction_id"] for r in group}
            coverage.append(dict(setting_id=setting, method=method, evaluated_rounds=len(group), expected_rounds=1280,
                                 interactions_present=len(identities), expected_interactions=EXPECTED_SEEDS,
                                 actual_outside_rounds=sum(not r["actual_inside"] for r in group),
                                 achieved_infeasible_rounds=sum(r["achieved_feasible"] is False for r in group),
                                 absent_achieved_geometry_rounds=sum(r["achieved_feasible"] is None for r in group),
                                 grid_inclusion_violations=sum(not r["achieved_contains_grid_value"] for r in group),
                                 actual_inclusion_violations=sum(r["achieved_contains_actual_value"] is False for r in group),
                                 analytic_upper_violations=sum(not r["achieved_below_entropy_upper"] for r in group),
                                 mean_relative_global_gap=float(np.mean([r["relative_global_gap"] for r in group])) if group else None,
                                 max_relative_global_gap=max((r["relative_global_gap"] for r in group), default=None),
                                 mean_actual_eig_absolute_discrepancy=float(np.mean([r["actual_eig_absolute_discrepancy"] for r in group])) if group else None))
    write_csv(root / "tables/oracle_coverage_and_gaps.csv", coverage)
    return rows, coverage


def generate_ellipsoid_report(run_dir, make_plots=True):
    root, source, manifest, runs = load_artifacts(run_dir)
    endpoint_tables(root, runs)
    diagnostics, coverage = diagnostic_tables(root, runs)
    figures = plot_figures(root, runs) if make_plots else []
    complete = sum(r["evaluated"] is not None and np.array_equal(r["evaluated"]["t"], np.arange(128))
                   and all(np.isfinite(r["evaluated"][name]).all() for name in METRICS) for r in runs)
    violations = sum(c[key] for c in coverage for key in ("achieved_infeasible_rounds", "grid_inclusion_violations", "actual_inclusion_violations", "analytic_upper_violations"))
    errors = []
    for path in sorted((root / "validation").glob("*.json")):
        document = read_json(path)
        errors.append(dict(path=str(path.relative_to(root)), status=document.get("status", "see_artifact")))
    summary = dict(source_run=str(source), source_histories=len(runs), expected_histories=EXPECTED_RUNS,
                   complete_evaluated_histories=int(complete), evaluated_rounds=len(diagnostics), expected_rounds=53760,
                   new_actual_answers=0, new_robot_rollouts=0, original_risk_values_changed=False,
                   actual_outside_rounds=sum(c["actual_outside_rounds"] for c in coverage),
                   diagnostic_violations=violations, global_optimum_certified=False,
                   posterior_error_certified=False, status="complete_approximate" if complete == 420 and not violations else "incomplete_or_unresolved",
                   validation_artifacts=errors, figures=figures)
    atomic_json(root / "tables/reevaluation_summary.json", summary)
    gaps = [r["relative_global_gap"] for r in diagnostics]
    gap_text = (f"Mean unresolved relative gap: {np.mean(gaps):.6g}; maximum: {max(gaps):.6g}." if gaps else "No achieved references are available yet.")
    validation_text = "; ".join(f"{r['path']}: {r['status']}" for r in errors) or "No reference-check JSON artifacts are available yet."
    protocol_files = [p.name for p in (root / "protocol.json", root / "oracle_config.json") if p.exists()]
    lines = ["# Ellipsoidal-reference reevaluation of ten interactions", "",
             f"**{complete}/420 histories reevaluated**, {len(diagnostics):,}/53,760 pre-answer rounds. "
             f"Source histories: `{source}`. The displayed queries, actual answers, theta draws and original risk values are reused unchanged. "
             "This reevaluation generates no new preference answers and no new robot rollouts.", "",
             "## What the new ratio means", "",
             "The denominator is the largest **achieved numerical integrated EIG** among verified feasible grid questions, "
             "the unchanged displayed question when it lies inside the ellipsoid, and candidates from full-dimensional optimization. "
             "The search varies every candidate coordinate. It uses each method's own frozen posterior. "
             "The new curve is an **approximate ellipsoidal-reference ratio**, not a certified exact global-oracle ratio. "
             "Local or multistart optimization supplies an achieved value; it does not prove global optimality or a ratio limit of one.", "",
             "The common shape parameters are α=0.5 and ρE=16, with W=αI+(1−α)Σ/λmax(Σ). "
             "For K=[−1,1]^4, D=4 and rmax=2. The strongest shared radius condition occurs at M=16: "
             "D/2 + (1+√31)rmax = 15.1355287257 ≤ 16. "
             "The reference ellipsoid lies in the radius-16 ball around the posterior mean and therefore fits the original radius-32 feasible query domain. "
             "It is a fixed physical reference region, not a posterior credible region.", "",
             "The numerator is reevaluated at the original, unaltered displayed query with the same integration convention as the achieved denominator. "
             f"Displayed questions outside the ellipsoid: {summary['actual_outside_rounds']:,} among evaluated rounds. "
             "Outside questions are not inserted into the feasible denominator candidate pool; their numerators are still retained and ratios above one are not clipped.", "",
             "## Optimization gap and numerical uncertainty", "",
             "The analytic reference upper bound is min(log M, 2β²ρE² tr Σ). It equals log M throughout these saved histories. "
             "Tables retain the achieved denominator, this upper value, their absolute and relative gaps, and numerator/upper versus numerator/achieved. "
             "For exact posterior integrals these form an oracle-ratio bracket. Here the reported plug-in bracket uses numerical EIG; "
             "it is not a certified bracket for the exact Bayesian posterior. " + gap_text, "",
             "Three error sources remain separate: (1) the original posterior approximation, including finite 4096-particle SMC error; "
             "(2) numerical integration of EIG, including joint integration for product posteriors and any sampling variability; "
             "(3) global optimization error, represented by the unresolved achieved-to-upper gap. "
             "Increasing optimizer iterations does not repair posterior error, and increasing integration sample count does not certify a global solve. "
             "The original-versus-reevaluated numerator discrepancies are diagnostic differences, not themselves error bounds. "
             "Arbitrary reference geometries are integrated over the joint parameter law; the one-coordinate likelihood shortcut is inapplicable.", "",
             "Reference checks: " + validation_text, "",
             "Optimization and integration budgets: " + (", ".join(f"`{p}`" for p in protocol_files) if protocol_files else "not yet saved") + ". "
             f"Independent geometry/value-inclusion diagnostic violations: {violations}. See `tables/oracle_diagnostics_per_round.csv` and `tables/oracle_coverage_and_gaps.csv`.", "",
             "## Figures and tables", "",
             "The four primary figures are the M/beta sweeps of the new approximate ratio and unchanged full posterior risk. "
             "Supplementary figures show unresolved relative optimization gaps and the original auxiliary grid ratios, correctly labeled η_grid. "
             "Original figures are not overwritten and original grid-normalized values are never relabeled as ellipsoidal values.", "",
             "Thin lines show all ten individual interactions; thick lines are arithmetic means of the per-interaction ratios. "
             "A mean is dashed when fewer than ten finite values are available. Effective counts are saved per method, setting and round. "
             "No curve is smoothed, clipped or extrapolated. Endpoint tables use risk/error after 32/64/128 answers and information quantities before answers 32/64/128, at t=31/63/127. "
             "Any table SE describes between-interaction variation only and excludes posterior/integration/optimization uncertainty. "
             "No bootstrap band or significance claim is added.", "",
             "## Retained interpretation limits", "",
             "All settings remain d=4 with T=128 and ten interactions. The shared M=4, β=4 anchor is counted once. "
             "This differs from the manuscript's eight-interaction, T=4096 suite and its d=8 beta sweep. "
             "The oracle responds to original q coordinates; Meta-World clips do not alter the statistical outcomes. "
             "This is not RL training, a task-return comparison or evidence of perceptual identifiability or physical safety. "
             "SG stays at stage 0, scale 1 and its fixed stage center for every observed query. "
             "Ten finite interactions do not establish an asymptotic rate or prove the theoretical hypotheses on every history. "
             "The correctly specified beta sweep does not test beta misspecification.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary
