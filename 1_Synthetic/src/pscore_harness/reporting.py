from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import (
    MAIN_METHODS,
    METHOD_LABELS,
    make_group_figure,
    make_main_results_figure,
    make_openml_main_figure,
    make_protocol_results_figure,
    make_synthetic_main_figure,
)
from .utils import atomic_write_text


PROPOSED_BY_PROTOCOL = {"sequential": "hard", "batch": "soft"}


def _latex_escape(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "dataset", "model", "protocol", "method", "seed"]
    idx = frame.groupby(keys, dropna=False)["acquired"].idxmax()
    return frame.loc[idx].copy()


def _mean_final(frame: pd.DataFrame, aggregate_datasets: bool) -> pd.DataFrame:
    columns = [
        "experiment",
        "dataset",
        "model",
        "protocol",
        "method",
        "lambda_mean",
        "lambda_sem",
        "lambda_count",
        "error_mean",
        "error_sem",
        "error_count",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    final = _final_rows(frame)
    keys = ["experiment", "model", "protocol", "method"]
    if aggregate_datasets:
        # OpenML macro-results use datasets, not seed-dataset pairs, as the
        # independent units. First average paired response seeds within each
        # feature matrix, then compute the macro mean and SEM across matrices.
        units = (
            final.groupby(["dataset", *keys], dropna=False)[["lambda_min", "l2_error"]]
            .mean()
            .reset_index()
        )
    else:
        keys.insert(1, "dataset")
        units = final

    rows = []
    for key_values, block in units.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values, strict=True))
        lam = block["lambda_min"].astype(float).to_numpy()
        err = block["l2_error"].astype(float).to_numpy()
        row.update(
            {
                "lambda_mean": float(np.mean(lam)),
                "lambda_sem": float(np.std(lam, ddof=1) / np.sqrt(len(lam))) if len(lam) > 1 else 0.0,
                "lambda_count": int(len(lam)),
                "error_mean": float(np.mean(err)),
                "error_sem": float(np.std(err, ddof=1) / np.sqrt(len(err))) if len(err) > 1 else 0.0,
                "error_count": int(len(err)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

def _metric_value(table: pd.DataFrame, method: str, metric: str) -> float | None:
    subset = table[table["method"] == method]
    if subset.empty:
        return None
    column = "lambda_mean" if metric == "lambda_min" else "error_mean"
    return float(subset.iloc[0][column])


def _finding_sentence(table: pd.DataFrame, protocol: str, context: str) -> str:
    proposed = PROPOSED_BY_PROTOCOL[protocol]
    if table.empty or proposed not in set(table["method"]):
        return f"No completed {METHOD_LABELS.get(proposed, proposed)} run was found for {context}."
    prop_lambda = _metric_value(table, proposed, "lambda_min")
    prop_error = _metric_value(table, proposed, "l2_error")
    displayed_baselines = set(MAIN_METHODS) - {"hard", "soft"}
    baselines = table[table["method"].isin(displayed_baselines)]
    if baselines.empty or prop_lambda is None or prop_error is None:
        return f"Completed runs for {context} were insufficient for a baseline comparison."
    best_lambda_row = baselines.loc[baselines["lambda_mean"].idxmax()]
    best_error_row = baselines.loc[baselines["error_mean"].idxmin()]
    best_lambda_method = str(best_lambda_row["method"])
    best_error_method = str(best_error_row["method"])
    best_lambda = float(best_lambda_row["lambda_mean"])
    best_error = float(best_error_row["error_mean"])
    lambda_change = 100.0 * (prop_lambda - best_lambda) / max(abs(best_lambda), 1e-12)
    error_change = 100.0 * (best_error - prop_error) / max(abs(best_error), 1e-12)
    lambda_phrase = (
        f"{abs(lambda_change):.1f}\\% larger" if lambda_change >= 0 else f"{abs(lambda_change):.1f}\\% smaller"
    )
    error_phrase = (
        f"{abs(error_change):.1f}\\% lower" if error_change >= 0 else f"{abs(error_change):.1f}\\% higher"
    )
    return (
        f"In {context}, {METHOD_LABELS[proposed]} reached final $\\lambda_{{\\min}}={prop_lambda:.3f}$, "
        f"which was {lambda_phrase} than the strongest displayed comparator "
        f"({METHOD_LABELS.get(best_lambda_method, best_lambda_method)}). Its final $\\ell_2$ error was "
        f"${prop_error:.3f}$, {error_phrase} than the best displayed error comparator "
        f"({METHOD_LABELS.get(best_error_method, best_error_method)})."
    )


def _openml_win_sentence(final: pd.DataFrame, protocol: str) -> str:
    proposed = PROPOSED_BY_PROTOCOL[protocol]
    subset = final[(final["experiment"] == "openml_main") & (final["protocol"] == protocol)]
    if subset.empty:
        return f"No completed OpenML {protocol} runs were found."
    means = subset.groupby(["dataset", "method"])[["lambda_min", "l2_error"]].mean().reset_index()
    datasets = sorted(means["dataset"].unique())
    lambda_wins = 0
    error_wins = 0
    eligible = 0
    for dataset in datasets:
        block = means[means["dataset"] == dataset]
        if proposed not in set(block["method"]):
            continue
        eligible += 1
        lambda_wins += int(str(block.loc[block["lambda_min"].idxmax(), "method"]) == proposed)
        error_wins += int(str(block.loc[block["l2_error"].idxmin(), "method"]) == proposed)
    return (
        f"Across the {eligible} completed OpenML datasets in the {protocol} protocol and all prespecified methods, "
        f"{METHOD_LABELS[proposed]} had the largest final $\\lambda_{{\\min}}$ on {lambda_wins} datasets "
        f"and the smallest final $\\ell_2$ error on {error_wins} datasets."
    )


def _make_final_table(final: pd.DataFrame) -> str:
    subset = final[final["experiment"] == "openml_main"].copy()
    if subset.empty:
        return "% No OpenML results available.\n"
    rows = []
    for (dataset, protocol, method), block in subset.groupby(["dataset", "protocol", "method"]):
        lam = block["lambda_min"].astype(float).to_numpy()
        err = block["l2_error"].astype(float).to_numpy()
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "method": method,
                "lam_mean": float(np.mean(lam)),
                "lam_sem": float(np.std(lam, ddof=1) / np.sqrt(len(lam))) if len(lam) > 1 else 0.0,
                "err_mean": float(np.mean(err)),
                "err_sem": float(np.std(err, ddof=1) / np.sqrt(len(err))) if len(err) > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["dataset", "protocol", "method"])
    summary["lam_best"] = summary.groupby(["dataset", "protocol"])["lam_mean"].transform("max")
    summary["err_best"] = summary.groupby(["dataset", "protocol"])["err_mean"].transform("min")
    header = (
        r"Dataset & Protocol & Method & $\lambda_{\min}(V_T^\star)$ & "
        r"$\|\widehat\theta_T-\theta^\star\|_2$ \\"
    )
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength\LTleft{0pt plus 1fill}",
        r"\setlength\LTright{0pt plus 1fill}",
        r"\begin{longtable}{@{}lllcc@{}}",
        r"\caption{Final OpenML semi-synthetic results. Only the two prespecified metrics are reported; the best mean within each dataset and protocol is bold.}",
        r"\label{tab:openml-final-results}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\tablename\ \thetable\ -- continued from previous page}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in summary.iterrows():
        dataset = _latex_escape(str(row["dataset"]))
        protocol = _latex_escape(str(row["protocol"]))
        method = _latex_escape(METHOD_LABELS.get(str(row["method"]), str(row["method"])))
        lam_mean = f"{row['lam_mean']:.3f}"
        err_mean = f"{row['err_mean']:.3f}"
        if np.isclose(row["lam_mean"], row["lam_best"], rtol=1e-10, atol=1e-12):
            lam_mean = rf"\mathbf{{{lam_mean}}}"
        if np.isclose(row["err_mean"], row["err_best"], rtol=1e-10, atol=1e-12):
            err_mean = rf"\mathbf{{{err_mean}}}"
        lines.append(
            f"{dataset} & {protocol} & {method} & "
            f"${lam_mean}\\pm{row['lam_sem']:.3f}$ & "
            f"${err_mean}\\pm{row['err_sem']:.3f}$ "
            + r"\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"

def _make_synthetic_final_table(final: pd.DataFrame) -> str:
    subset = final[final["experiment"] == "synthetic_main"].copy()
    if subset.empty:
        return "% No synthetic main results available.\n"
    rows = []
    for (model, protocol, method), block in subset.groupby(["model", "protocol", "method"]):
        lam = block["lambda_min"].astype(float).to_numpy()
        err = block["l2_error"].astype(float).to_numpy()
        rows.append(
            {
                "model": model,
                "protocol": protocol,
                "method": method,
                "lam_mean": float(np.mean(lam)),
                "lam_sem": float(np.std(lam, ddof=1) / np.sqrt(len(lam))) if len(lam) > 1 else 0.0,
                "err_mean": float(np.mean(err)),
                "err_sem": float(np.std(err, ddof=1) / np.sqrt(len(err))) if len(err) > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["model", "protocol", "method"])
    summary["lam_best"] = summary.groupby(["model", "protocol"])["lam_mean"].transform("max")
    summary["err_best"] = summary.groupby(["model", "protocol"])["err_mean"].transform("min")
    header = (
        r"GLM & Protocol & Method & $\lambda_{\min}(V_T^\star)$ & "
        r"$\|\widehat\theta_T-\theta^\star\|_2$ \\"
    )
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength\LTleft{0pt plus 1fill}",
        r"\setlength\LTright{0pt plus 1fill}",
        r"\begin{longtable}{@{}lllcc@{}}",
        r"\caption{Final values in the main synthetic experiment. Means and standard errors are over paired synthetic seeds; the best mean within each GLM and protocol is bold.}",
        r"\label{tab:synthetic-final-results}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\tablename\ \thetable\ -- continued from previous page}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in summary.iterrows():
        model = _latex_escape(str(row["model"]))
        protocol = _latex_escape(str(row["protocol"]))
        method = _latex_escape(METHOD_LABELS.get(str(row["method"]), str(row["method"])))
        lam_mean = f"{row['lam_mean']:.3f}"
        err_mean = f"{row['err_mean']:.3f}"
        if np.isclose(row["lam_mean"], row["lam_best"], rtol=1e-10, atol=1e-12):
            lam_mean = rf"\mathbf{{{lam_mean}}}"
        if np.isclose(row["err_mean"], row["err_best"], rtol=1e-10, atol=1e-12):
            err_mean = rf"\mathbf{{{err_mean}}}"
        lines.append(
            f"{model} & {protocol} & {method} & "
            f"${lam_mean}\\pm{row['lam_sem']:.3f}$ & "
            f"${err_mean}\\pm{row['err_sem']:.3f}$ "
            + r"\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"

def _safe_main_figures(frame: pd.DataFrame, figures: Path) -> None:
    synthetic = frame[frame["experiment"] == "synthetic_main"]
    openml = frame[frame["experiment"] == "openml_main"]
    if not synthetic.empty or not openml.empty:
        make_main_results_figure(frame, figures / "main_results.pdf")
        make_protocol_results_figure(frame, "sequential", figures / "sequential_results.pdf")
        make_protocol_results_figure(frame, "batch", figures / "batched_results.pdf")
    # Detailed source-specific figures are retained for debugging and Appendix.
    if not synthetic.empty:
        make_synthetic_main_figure(frame, figures / "synthetic_main_detailed.pdf")
    if not openml.empty:
        make_openml_main_figure(frame, figures / "openml_main_detailed.pdf")


def generate_report(frame: pd.DataFrame, paper_dir: Path) -> dict[str, Path]:
    if "algorithm_suite" in frame and frame["algorithm_suite"].notna().any():
        from .spec_suite.reporting import generate_suite_report
        return generate_suite_report(frame, paper_dir)
    if "experiment" in frame.columns:
        frame = frame[~frame["experiment"].astype(str).str.startswith(("smoke", "pytest"))].copy()
    if frame.empty:
        raise ValueError("No non-smoke completed trajectories were available for reporting")
    generated = paper_dir / "generated"
    figures = generated / "figures"
    appendix_figures = generated / "appendix_figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    appendix_figures.mkdir(parents=True, exist_ok=True)

    _safe_main_figures(frame, figures)
    final = _final_rows(frame)
    synthetic_seq = _mean_final(
        frame[(frame["experiment"] == "synthetic_main") & (frame["model"] == "gaussian") & (frame["protocol"] == "sequential")],
        aggregate_datasets=False,
    )
    synthetic_batch = _mean_final(
        frame[(frame["experiment"] == "synthetic_main") & (frame["model"] == "logistic") & (frame["protocol"] == "batch")],
        aggregate_datasets=False,
    )
    openml_seq = _mean_final(
        frame[(frame["experiment"] == "openml_main") & (frame["model"] == "logistic") & (frame["protocol"] == "sequential")],
        aggregate_datasets=True,
    )
    openml_batch = _mean_final(
        frame[(frame["experiment"] == "openml_main") & (frame["model"] == "logistic") & (frame["protocol"] == "batch")],
        aggregate_datasets=True,
    )

    synthetic_seq_sentence = _finding_sentence(
        synthetic_seq, "sequential", "the representative synthetic sequential Gaussian setting"
    )
    synthetic_batch_sentence = _finding_sentence(
        synthetic_batch, "batch", "the representative synthetic batched logistic setting"
    )
    openml_seq_sentence = _finding_sentence(
        openml_seq, "sequential", "the OpenML sequential macro-average"
    )
    openml_batch_sentence = _finding_sentence(
        openml_batch, "batch", "the OpenML batched macro-average"
    )
    openml_seq_wins = _openml_win_sentence(final, "sequential")
    openml_batch_wins = _openml_win_sentence(final, "batch")
    result_blocks = [
        "\\paragraph{Synthetic sequential.} " + synthetic_seq_sentence,
        "\\paragraph{Synthetic batch.} " + synthetic_batch_sentence,
        "\\paragraph{OpenML aggregate.} " + openml_seq_sentence + " " + openml_batch_sentence,
        "\\paragraph{Dataset-level consistency.} " + openml_seq_wins + " " + openml_batch_wins,
    ]
    atomic_write_text(generated / "results_summary.tex", "\n\n".join(result_blocks) + "\n")
    conclusion_tex = (
        "The completed experiments use only the two prespecified outcomes, "
        "$\\lambda_{\\min}(V_t^\\star)$ and $\\|\\widehat\\theta_t-\\theta^\\star\\|_2$. "
        + synthetic_seq_sentence
        + " "
        + synthetic_batch_sentence
        + " "
        + openml_seq_wins
        + " "
        + openml_batch_wins
        + " These results should be interpreted as evidence about global parameter recovery, not reward or classification performance.\n"
    )
    atomic_write_text(generated / "conclusion_results.tex", conclusion_tex)
    atomic_write_text(generated / "synthetic_final_table.tex", _make_synthetic_final_table(final))
    atomic_write_text(generated / "openml_final_table.tex", _make_final_table(final))

    appendix_blocks: list[str] = []
    group_columns = [
        "experiment",
        "dataset",
        "model",
        "protocol",
        "batch_size",
        "setting_label",
    ]
    available_columns = [column for column in group_columns if column in frame.columns]
    for keys, block in frame.groupby(available_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        descriptor = dict(zip(available_columns, keys, strict=True))
        slug = "__".join(re.sub(r"[^A-Za-z0-9.-]+", "-", str(descriptor[k])) for k in available_columns)
        title = ", ".join(f"{k}={descriptor[k]}" for k in available_columns[1:])
        output = appendix_figures / f"{slug}.pdf"
        make_group_figure(block, title, output)
        relative = output.relative_to(paper_dir.parent).as_posix()
        appendix_blocks.extend(
            [
                r"\begin{figure*}[t]",
                r"\centering",
                rf"\includegraphics[width=0.98\textwidth]{{{relative}}}",
                rf"\caption{{Additional trajectories for {_latex_escape(title)}. Shading denotes 95\% confidence intervals across paired seeds.}}",
                r"\end{figure*}",
                "",
            ]
        )
    atomic_write_text(generated / "appendix_figures.tex", "\n".join(appendix_blocks))

    aggregate = (
        frame.groupby(["experiment", "dataset", "model", "protocol", "method", "acquired"], dropna=False)[
            ["lambda_min", "l2_error"]
        ]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    aggregate.to_csv(generated / "aggregate_results.csv", index=False)
    return {
        "main_results": figures / "main_results.pdf",
        "sequential_results": figures / "sequential_results.pdf",
        "batched_results": figures / "batched_results.pdf",
        "synthetic_detailed": figures / "synthetic_main_detailed.pdf",
        "openml_detailed": figures / "openml_main_detailed.pdf",
        "results_summary": generated / "results_summary.tex",
        "conclusion_results": generated / "conclusion_results.tex",
        "appendix_figures": generated / "appendix_figures.tex",
        "synthetic_table": generated / "synthetic_final_table.tex",
        "openml_table": generated / "openml_final_table.tex",
    }
