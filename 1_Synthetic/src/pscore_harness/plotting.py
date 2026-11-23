from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spec_suite import ALL_METHODS as SPEC_METHOD_ORDER, METHOD_LABELS as SPEC_LABELS


DEFAULT_METHOD_ORDER = list(SPEC_METHOD_ORDER) + [
    "hard",
    "soft",
    "hard_topb",
    "hard_greedy",
    "e_opt",
    "a_opt",
    "d_opt",
    "g_opt",
    "lingape",
    "ucb",
    "thompson",
    "epsilon_greedy",
    "random",
]


METHOD_LABELS = {
    "hard": "Hard-PScore",
    "soft": "Soft-PScore",
    "hard_topb": "Top-$m$ Hard",
    "hard_greedy": "Greedy Hard",
    "a_opt": "A-opt OED",
    "d_opt": "D-opt OED",
    "e_opt": "E-opt OED",
    "g_opt": "G-opt OED",
    "ucb": "UCB / GLM-UCB",
    "thompson": "Thompson sampling",
    "lingape": "LinGapE",
    "epsilon_greedy": "$\\epsilon$-greedy",
    "random": "Uniform random",
    "soft_tau_half": "Soft: $0.5\\tau_\\star$",
    "soft_tau_one": "Soft: $\\tau_\\star$",
    "soft_tau_two": "Soft: $2\\tau_\\star$",
    "ucb_beta_half": "UCB: $\\beta=0.5$",
    "ucb_beta_one": "UCB: $\\beta=1$",
    "ucb_beta_two": "UCB: $\\beta=2$",
    "ts_half": "TS: $c=0.5$",
    "ts_one": "TS: $c=1$",
    "ts_two": "TS: $c=2$",
}


# Keep the main-paper figures focused on the proposed methods and established
# comparators. Top-m Hard and Greedy Hard are diagnostic batch controls, so
# they remain in the automatically generated Appendix figures and tables.
MAIN_METHODS = {
    "hard",
    "soft",
    "e_opt",
    "a_opt",
    "d_opt",
    "lingape",
    "ucb",
    "random",
}
MAIN_METHOD_ORDER = ["hard", "soft", "a_opt", "d_opt", "e_opt", "lingape", "ucb", "random"]
PROPOSED_METHODS = {"hard", "soft"}
DISPLAY_EXCLUDED_METHODS = {"hard_topb", "hard_greedy"}


# Color-blind-friendly, fixed method identities. The proposed methods use the
# most saturated colors; comparators also differ by line style for grayscale
# and reduced-vision readability.
METHOD_STYLES: dict[str, dict[str, object]] = {
    "hard": {"color": "#0057B8", "linestyle": "-"},
    "soft": {"color": "#D7191C", "linestyle": "-"},
    "hard_topb": {"color": "#80B1D3", "linestyle": "--"},
    "hard_greedy": {"color": "#FB8072", "linestyle": "-."},
    "a_opt": {"color": "#009E73", "linestyle": "-."},
    "d_opt": {"color": "#E6AB02", "linestyle": ":"},
    "e_opt": {"color": "#6A3D9A", "linestyle": "--"},
    "g_opt": {"color": "#A6761D", "linestyle": (0, (5, 2))},
    "ucb": {"color": "#00A6D6", "linestyle": (0, (3, 1, 1, 1))},
    "thompson": {"color": "#E7298A", "linestyle": (0, (5, 2))},
    "lingape": {"color": "#8C564B", "linestyle": (0, (5, 2, 1, 2))},
    "epsilon_greedy": {"color": "#66A61E", "linestyle": (0, (3, 2))},
    "random": {"color": "#4D4D4D", "linestyle": (0, (7, 3))},
    "soft_tau_half": {"color": "#FDAE61", "linestyle": "--"},
    "soft_tau_one": {"color": "#D7191C", "linestyle": "-"},
    "soft_tau_two": {"color": "#ABDDA4", "linestyle": ":"},
    "ucb_beta_half": {"color": "#A6CEE3", "linestyle": "--"},
    "ucb_beta_one": {"color": "#00A6D6", "linestyle": "-"},
    "ucb_beta_two": {"color": "#1F78B4", "linestyle": ":"},
    "ts_half": {"color": "#F1B6DA", "linestyle": "--"},
    "ts_one": {"color": "#E7298A", "linestyle": "-"},
    "ts_two": {"color": "#AE017E", "linestyle": ":"},
}

# Persistent figure convention for the new 15-method suite.
METHOD_LABELS.update(SPEC_LABELS)
METHOD_LABELS["ours"] = "Ours"
METHOD_STYLES.update({
    "fb_opt2": {"color": "#0057B8", "linestyle": "-"},
    "ours": {"color": "#0057B8", "linestyle": "-"},
    "e_greedy": {"color": "#6A3D9A", "linestyle": "--"},
    "e_sdp": {"color": "#D55E00", "linestyle": "-."},
    "bait_glm": {"color": "#CC79A7", "linestyle": (0, (5, 2))},
    "ucb_glm": {"color": "#00A6D6", "linestyle": (0, (3, 1, 1, 1))},
    "ts_glm": {"color": "#E7298A", "linestyle": (0, (5, 2))},
    "b_emcm": {"color": "#8C564B", "linestyle": (0, (5, 2, 1, 2))},
    "lcmd": {"color": "#66A61E", "linestyle": (0, (3, 2))},
    "regvol": {"color": "#A6761D", "linestyle": ":"},
    "entropy": {"color": "#66A61E", "linestyle": (0, (3, 2))},
    "badge": {"color": "#8C564B", "linestyle": (0, (5, 2, 1, 2))},
    "firal": {"color": "#A6761D", "linestyle": ":"},
})
MAIN_METHODS.update(SPEC_METHOD_ORDER)
MAIN_METHOD_ORDER = list(SPEC_METHOD_ORDER) + MAIN_METHOD_ORDER
PROPOSED_METHODS.update({"fb_opt2", "ours"})
DEFAULT_METHOD_ORDER = list(dict.fromkeys(DEFAULT_METHOD_ORDER))
MAIN_METHOD_ORDER = list(dict.fromkeys(MAIN_METHOD_ORDER))


def aggregate_curves(frame: pd.DataFrame, aggregate_datasets: bool) -> pd.DataFrame:
    """Aggregate paired trajectories.

    For a single synthetic dataset, uncertainty is computed across seeds. For
    OpenML macro-averages, we first average seeds within each dataset and then
    compute the mean/SEM across datasets. This gives every dataset equal weight
    and avoids treating many response seeds from one matrix as independent
    datasets.
    """
    if frame.empty:
        return pd.DataFrame()

    base_keys = [
        "experiment",
        "source",
        "model",
        "protocol",
        "batch_size",
        "method",
        "acquired",
    ]
    if aggregate_datasets:
        dataset_keys = ["dataset", *base_keys]
        dataset_means = (
            frame.groupby(dataset_keys, dropna=False)[["lambda_min", "l2_error"]]
            .mean()
            .reset_index()
        )
        grouped = dataset_means.groupby(base_keys, dropna=False)
    else:
        group_keys = ["experiment", "source", "dataset", "model", "protocol", "batch_size", "method", "acquired"]
        grouped = frame.groupby(group_keys, dropna=False)
        base_keys = group_keys

    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(base_keys, keys, strict=True))
        for metric in ["lambda_min", "l2_error"]:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
            row[f"{metric}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _method_order(methods: Iterable[str]) -> list[str]:
    methods = [
        method for method in dict.fromkeys(methods)
        if method not in DISPLAY_EXCLUDED_METHODS
    ]
    known = [method for method in DEFAULT_METHOD_ORDER if method in methods]
    unknown = sorted(set(methods) - set(known))
    return known + unknown


def _plot_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str | None,
    fixed_method_order: list[str] | None = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No completed runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(r"Time step $t$ (acquired responses)")
        if y_label:
            ax.set_ylabel(y_label)
        return

    methods = fixed_method_order or _method_order(frame["method"].unique())
    # Draw baselines first and proposed methods last. Explicit z-orders are
    # retained as a second guard against the highlighted lines being obscured.
    plot_order = [method for method in methods if method not in PROPOSED_METHODS]
    plot_order.extend(method for method in methods if method in PROPOSED_METHODS)
    for method in plot_order:
        subset = frame[frame["method"] == method].sort_values("acquired")
        if subset.empty:
            continue
        x = subset["acquired"].to_numpy(dtype=float)
        mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
        sem = subset[f"{metric}_sem"].to_numpy(dtype=float)
        style = METHOD_STYLES.get(method, {"color": "#6B7280", "linestyle": "-"})
        proposed = method in PROPOSED_METHODS
        line = ax.plot(
            x,
            mean,
            label=METHOD_LABELS.get(method, method),
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=3.35 if proposed else 1.65,
            alpha=1.0 if proposed else 0.88,
            zorder=20 if proposed else 5,
        )[0]
        if proposed:
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=5.2, foreground="white", alpha=0.92),
                    path_effects.Normal(),
                ]
            )
        ax.fill_between(
            x,
            mean - 1.96 * sem,
            mean + 1.96 * sem,
            alpha=0.16 if proposed else 0.07,
            color=str(style["color"]),
            linewidth=0,
            zorder=2 if proposed else 1,
        )
    ax.set_title(title, fontweight="semibold")
    ax.set_xlabel(r"Time step $t$ (acquired responses)")
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def _combined_legend(
    axes: np.ndarray, method_order: Iterable[str] = DEFAULT_METHOD_ORDER
) -> tuple[list[object], list[str]]:
    by_label: dict[str, object] = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            by_label.setdefault(label, handle)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    ordered_labels = []
    for method in method_order:
        label = METHOD_LABELS.get(method, method)
        if label in by_label:
            ordered_labels.append(label)
    ordered_labels.extend(sorted(set(by_label) - set(ordered_labels)))
    return [by_label[label] for label in ordered_labels], ordered_labels


def _add_figure_legend(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    method_order: Iterable[str],
    ncol: int,
) -> None:
    handles, labels = _combined_legend(axes, method_order=method_order)
    legend = fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)
    proposed_labels = {METHOD_LABELS[method] for method in PROPOSED_METHODS}
    for text in legend.get_texts():
        if text.get_text() in proposed_labels:
            text.set_fontweight("bold")




def _select_context(frame: pd.DataFrame, **conditions: object) -> pd.DataFrame:
    if frame.empty or any(column not in frame.columns for column in conditions):
        return frame.iloc[0:0].copy()
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].to_numpy() == value
    return frame.loc[mask]

def make_main_results_figure(frame: pd.DataFrame, output_path: Path) -> None:
    """Legacy combined overview, using the focused main-paper method set."""
    synthetic = frame[
        (frame["experiment"] == "synthetic_main")
        & (frame["dataset"] == "anisotropic_ball")
        & (frame["method"].isin(MAIN_METHODS))
    ]
    openml = frame[
        (frame["experiment"] == "openml_main")
        & (frame["model"] == "logistic")
        & (frame["method"].isin(MAIN_METHODS))
    ]
    synth_agg = aggregate_curves(synthetic, aggregate_datasets=False)
    openml_agg = aggregate_curves(openml, aggregate_datasets=True)

    contexts = [
        _select_context(synth_agg, model="gaussian", protocol="sequential"),
        _select_context(synth_agg, model="logistic", protocol="batch"),
        _select_context(openml_agg, protocol="sequential"),
        _select_context(openml_agg, protocol="batch"),
    ]
    titles = [
        "Synthetic: sequential Gaussian",
        "Synthetic: batched logistic",
        "OpenML: sequential logistic",
        "OpenML: batched logistic",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.2), constrained_layout=True)
    for column, (context, title) in enumerate(zip(contexts, titles, strict=True)):
        _plot_metric(
            axes[0, column],
            context,
            "lambda_min",
            title,
            r"$\lambda_{\min}(V_t^\star)$" if column == 0 else None,
            fixed_method_order=MAIN_METHOD_ORDER,
        )
        _plot_metric(
            axes[1, column],
            context,
            "l2_error",
            "",
            r"$\|\widehat\theta_t-\theta^\star\|_2$" if column == 0 else None,
            fixed_method_order=MAIN_METHOD_ORDER,
        )

    _add_figure_legend(fig, axes, method_order=MAIN_METHOD_ORDER, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_protocol_results_figure(frame: pd.DataFrame, protocol: str, output_path: Path) -> None:
    """Main-paper figure for one acquisition protocol.

    Sequential and batch results are deliberately separated so each proposed
    method is the visual anchor of its own figure.
    """
    if protocol not in {"sequential", "batch"}:
        raise ValueError(f"Unsupported main-figure protocol: {protocol}")

    synthetic_model = "gaussian" if protocol == "sequential" else "logistic"
    synthetic = frame[
        (frame["experiment"] == "synthetic_main")
        & (frame["dataset"] == "anisotropic_ball")
        & (frame["model"] == synthetic_model)
        & (frame["protocol"] == protocol)
        & (frame["method"].isin(MAIN_METHODS))
    ]
    openml = frame[
        (frame["experiment"] == "openml_main")
        & (frame["model"] == "logistic")
        & (frame["protocol"] == protocol)
        & (frame["method"].isin(MAIN_METHODS))
    ]
    contexts = [
        aggregate_curves(synthetic, aggregate_datasets=False),
        aggregate_curves(openml, aggregate_datasets=True),
    ]
    protocol_title = "Sequential" if protocol == "sequential" else "Batched"
    titles = [
        f"Synthetic: {protocol_title.lower()} {synthetic_model}",
        f"OpenML: {protocol_title.lower()} logistic",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), constrained_layout=True)
    fig.suptitle(f"{protocol_title} acquisition", fontsize=16, fontweight="bold")
    for column, (context, title) in enumerate(zip(contexts, titles, strict=True)):
        _plot_metric(
            axes[0, column],
            context,
            "lambda_min",
            title,
            r"$\lambda_{\min}(V_t^\star)$" if column == 0 else r"Macro-mean $\lambda_{\min}(V_t^\star)$",
            fixed_method_order=MAIN_METHOD_ORDER,
        )
        _plot_metric(
            axes[1, column],
            context,
            "l2_error",
            "",
            r"$\|\widehat\theta_t-\theta^\star\|_2$"
            if column == 0
            else r"Macro-mean $\|\widehat\theta_t-\theta^\star\|_2$",
            fixed_method_order=MAIN_METHOD_ORDER,
        )

    _add_figure_legend(fig, axes, method_order=MAIN_METHOD_ORDER, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_synthetic_main_figure(frame: pd.DataFrame, output_path: Path) -> None:
    """Detailed synthetic-only figure retained for Appendix/debugging."""
    selected = frame[(frame["experiment"] == "synthetic_main") & (frame["dataset"] == "anisotropic_ball")]
    aggregated = aggregate_curves(selected, aggregate_datasets=False)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    seq = aggregated[(aggregated["model"] == "gaussian") & (aggregated["protocol"] == "sequential")]
    batch = aggregated[(aggregated["model"] == "logistic") & (aggregated["protocol"] == "batch")]
    _plot_metric(axes[0, 0], seq, "lambda_min", "Sequential Gaussian", r"$\lambda_{\min}(V_t^\star)$")
    _plot_metric(axes[0, 1], seq, "l2_error", "Sequential Gaussian", r"$\|\widehat\theta_t-\theta^\star\|_2$")
    _plot_metric(axes[1, 0], batch, "lambda_min", "Batched logistic", r"$\lambda_{\min}(V_t^\star)$")
    _plot_metric(axes[1, 1], batch, "l2_error", "Batched logistic", r"$\|\widehat\theta_t-\theta^\star\|_2$")
    _add_figure_legend(fig, axes, method_order=DEFAULT_METHOD_ORDER, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_openml_main_figure(frame: pd.DataFrame, output_path: Path) -> None:
    """Detailed OpenML-only figure retained for Appendix/debugging."""
    selected = frame[(frame["experiment"] == "openml_main") & (frame["model"] == "logistic")]
    aggregated = aggregate_curves(selected, aggregate_datasets=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    seq = aggregated[aggregated["protocol"] == "sequential"]
    batch = aggregated[aggregated["protocol"] == "batch"]
    _plot_metric(axes[0, 0], seq, "lambda_min", "OpenML: sequential", r"Macro-mean $\lambda_{\min}(V_t^\star)$")
    _plot_metric(axes[0, 1], seq, "l2_error", "OpenML: sequential", r"Macro-mean $\|\widehat\theta_t-\theta^\star\|_2$")
    _plot_metric(axes[1, 0], batch, "lambda_min", "OpenML: batch", r"Macro-mean $\lambda_{\min}(V_t^\star)$")
    _plot_metric(axes[1, 1], batch, "l2_error", "OpenML: batch", r"Macro-mean $\|\widehat\theta_t-\theta^\star\|_2$")
    _add_figure_legend(fig, axes, method_order=DEFAULT_METHOD_ORDER, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_group_figure(frame: pd.DataFrame, title: str, output_path: Path) -> None:
    aggregated = aggregate_curves(frame, aggregate_datasets=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)
    _plot_metric(axes[0], aggregated, "lambda_min", title, r"$\lambda_{\min}(V_t^\star)$")
    _plot_metric(axes[1], aggregated, "l2_error", title, r"$\|\widehat\theta_t-\theta^\star\|_2$")
    _add_figure_legend(fig, axes, method_order=DEFAULT_METHOD_ORDER, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_openml_family_comparison_figures(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    theta_regime: str = "weak",
    max_t: int = 200,
) -> list[Path]:
    """Create exactly four 3x2 family-comparison PDFs per OpenML dataset.

    Rows are Bernoulli, Gaussian, and Poisson. Columns are oracle information
    growth and parameter L2 error. Each PDF fixes one acquisition batch size,
    making all curves within a panel directly comparable through time step t.
    """
    if max_t <= 0:
        raise ValueError("max_t must be positive")

    family_experiments = [
        ("Bernoulli", "openml_bernoulli", "logistic"),
        ("Gaussian", "openml_gaussian", "gaussian"),
        ("Poisson", "openml_poisson", "poisson"),
    ]
    # LinGapE is defined only for Gaussian in this harness. Keep the displayed
    # acquisition-policy set identical across all three family rows.
    comparison_method_order = [method for method in DEFAULT_METHOD_ORDER if method != "lingape"]
    batch_specs = [(1, "sequential"), (5, "batch"), (10, "batch"), (20, "batch")]
    selected = frame[
        frame["experiment"].isin([item[1] for item in family_experiments])
        & (frame["theta_regime"] == theta_regime)
        & (frame["acquired"] <= max_t)
    ].copy()
    datasets = sorted(selected["dataset"].dropna().unique())
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for dataset in datasets:
        for batch_size, protocol in batch_specs:
            fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.2), constrained_layout=True)
            fig.suptitle(
                f"OpenML {dataset} — batch size {batch_size}, $t \\leq {max_t}$",
                fontsize=15,
                fontweight="bold",
            )
            for row, (family_label, experiment, model) in enumerate(family_experiments):
                context = selected[
                    (selected["dataset"] == dataset)
                    & (selected["experiment"] == experiment)
                    & (selected["model"] == model)
                    & (selected["protocol"] == protocol)
                    & (selected["batch_size"] == batch_size)
                    & (selected["method"].isin(comparison_method_order))
                ]
                if context.empty:
                    raise RuntimeError(
                        f"Missing comparison results: dataset={dataset}, family={family_label}, "
                        f"batch_size={batch_size}, theta_regime={theta_regime}"
                    )
                aggregated = aggregate_curves(context, aggregate_datasets=False)
                _plot_metric(
                    axes[row, 0],
                    aggregated,
                    "lambda_min",
                    family_label,
                    r"$\lambda_{\min}(V_t^\star)$",
                    fixed_method_order=comparison_method_order,
                )
                _plot_metric(
                    axes[row, 1],
                    aggregated,
                    "l2_error",
                    family_label,
                    r"$\|\widehat\theta_t-\theta^\star\|_2$",
                    fixed_method_order=comparison_method_order,
                )
                axes[row, 0].set_xlim(0, max_t)
                axes[row, 1].set_xlim(0, max_t)

            axes[0, 0].set_title(r"Bernoulli — $\lambda_{\min}(V_t^\star)$", fontweight="semibold")
            axes[0, 1].set_title(r"Bernoulli — $\|\widehat\theta_t-\theta^\star\|_2$", fontweight="semibold")
            _add_figure_legend(fig, axes, method_order=comparison_method_order, ncol=5)
            safe_dataset = str(dataset).replace("/", "-").replace(" ", "-")
            output_path = output_dir / f"{safe_dataset}__batch-{batch_size}.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            outputs.append(output_path)

    expected = len(datasets) * len(batch_specs)
    if len(outputs) != expected:
        raise RuntimeError(f"Expected {expected} comparison figures, generated {len(outputs)}")
    return outputs
