"""Four-panel reporting of known-truth synthetic parameter estimation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..utils import atomic_write_json
from . import METHOD_LABELS, methods_for_family, selector_seed
from .runner import validate_completed_run


def generate_suite_report(frame: pd.DataFrame, paper_dir: Path) -> dict[str, Path]:
    from ..plotting import METHOD_STYLES

    frame = frame[frame["algorithm_suite"].notna()].copy()
    if frame.empty:
        raise ValueError("No completed spec-suite trajectories")
    output = paper_dir / "generated" / "algorithm_suite"
    output.mkdir(parents=True, exist_ok=True)
    for run in frame["run_dir"].unique():
        validate_completed_run(Path(run))
    keys = ["experiment", "dataset", "dataset_variant", "dimension", "model", "protocol", "batch_size", "method", "acquired"]
    rows, diagnostics = [], []
    for values, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, values, strict=True))
        if group["seed"].duplicated().any():
            raise ValueError("Duplicate seed trajectory in one method/setting")
        errors = []
        for item in group.itertuples():
            with np.load(Path(item.run_dir) / "checkpoints" / f"round_{int(item.batch_index):06d}.npz", allow_pickle=False) as saved:
                errors.append(saved["signed_error"].copy())
        errors = np.asarray(errors)
        Q = errors.T @ errors / len(errors)
        row.update(seed_count=len(errors), R2=float(np.sqrt(np.trace(Q))),
                   Rdir=float(np.sqrt(max(0., np.linalg.eigvalsh(Q)[-1]))))
        for metric, column in (("mean_lambda_min_true", "lambda_min"),
                               ("mean_lambda_min_plugin", "plugin_lambda_min")):
            v = group[column].to_numpy(dtype=float)
            mean = float(v.mean())
            half = float(student_t.ppf(.975, len(v)-1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.
            row.update({metric: mean, metric + "_ci_low": mean-half, metric + "_ci_high": mean+half})
        for i, seed in enumerate(group["seed"]):
            if len(errors) > 1:
                keep = np.delete(errors, i, axis=0)
                leave_Q = keep.T @ keep / len(keep)
                diagnostics.append({**dict(zip(keys, values, strict=True)), "omitted_seed": int(seed),
                    "R2": float(np.sqrt(np.trace(leave_Q))),
                    "Rdir": float(np.sqrt(max(0., np.linalg.eigvalsh(leave_Q)[-1])))})
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    conditions = [k for k in keys if k not in ("method", "acquired")]
    method_keys = conditions + ["method"]
    endpoints = aggregate.loc[aggregate.groupby(method_keys, dropna=False)["acquired"].idxmax()].copy()
    for index, endpoint in endpoints.iterrows():
        mask = np.ones(len(frame), dtype=bool)
        for k in keys:
            mask &= frame[k].to_numpy() == endpoint[k]
        group = frame.loc[mask]
        errors = []
        for item in group.itertuples():
            with np.load(Path(item.run_dir) / "checkpoints" / f"round_{int(item.batch_index):06d}.npz", allow_pickle=False) as saved:
                errors.append(saved["signed_error"].copy())
        errors = np.asarray(errors)
        rng = np.random.default_rng(selector_seed("endpoint_bootstrap_" + str(endpoint["dataset"]),
                    int(endpoint["dimension"]), int(endpoint["batch_size"]), str(endpoint["method"])))
        draws = rng.integers(len(errors), size=(1000, len(errors)))
        sampled = errors[draws]
        Q = np.einsum("bsi,bsj->bij", sampled, sampled) / len(errors)
        risks = {"R2": np.sqrt(np.trace(Q, axis1=1, axis2=2)),
                 "Rdir": np.sqrt(np.maximum(0., np.linalg.eigvalsh(Q)[:, -1]))}
        for metric, v in risks.items():
            lo, hi = np.quantile(v, [.025, .975])
            endpoints.loc[index, metric + "_bootstrap_ci_low"] = lo
            endpoints.loc[index, metric + "_bootstrap_ci_high"] = hi
    aggregate.to_csv(output / "aggregate_curves.csv", index=False)
    endpoints.to_csv(output / "endpoints.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "leave_one_seed_out.csv", index=False)
    plots = []
    metrics = [("mean_lambda_min_true", r"$\mathrm{mean}\ \lambda_{\min}(V_T^\star)$"),
               ("mean_lambda_min_plugin", r"$\mathrm{mean}\ \lambda_{\min}(\widehat V_T)$"),
               ("R2", r"$R_2=\sqrt{\mathrm{tr}(Q_T)}$"),
               ("Rdir", r"$R_{\mathrm{dir}}=\sqrt{\lambda_{\max}(Q_T)}$")]
    for condition, block in aggregate.groupby(conditions, dropna=False):
        setting = dict(zip(conditions, condition, strict=True))
        order = [name for name in methods_for_family(str(setting["model"])) if name in set(block["method"])]
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
        for ax, (metric, label) in zip(axes.flat, metrics):
            for method in order:
                curve = block[block["method"] == method].sort_values("acquired")
                ax.plot(curve["acquired"], curve[metric], label=METHOD_LABELS[method],
                    linewidth=2.6 if method == "fb_opt2" else 1.45,
                    zorder=10 if method == "fb_opt2" else 2, **METHOD_STYLES[method])
            ax.set_xlabel(r"Additional responses $mT$")
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        top_values = np.r_[block[metrics[0][0]], block[metrics[1][0]]]
        pad = max(float(np.ptp(top_values)) * .05, .01)
        for ax in axes[0]:
            ax.set_ylim(max(0., float(top_values.min())-pad), float(top_values.max())+pad)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
        fig.suptitle(f"{setting['model']} · d={setting['dimension']} · m={setting['batch_size']}", y=.91)
        fig.subplots_adjust(top=.84, bottom=.10, left=.10, right=.98, hspace=.38, wspace=.28)
        stem = "__".join(str(setting[k]).replace("/", "-") for k in conditions)
        for extension in ("pdf", "png", "svg"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=180)
            plots.append(path.name)
        plt.close(fig)
    atomic_write_json(output / "manifest.json", {"figures": plots,
        "ours_style": {"color": "#0057B8", "linestyle": "-"},
        "truth": "Known synthetic theta_star, not empirical full-pool reference",
        "Q": "Uncentered mean error outer product, divided by seed count",
        "plot": "All family-eligible methods; linear axes; unsmoothed checkpoints; no bands",
        "uncertainty": "Descriptive Student-t intervals for information means, 1000 whole-seed endpoint bootstrap draws for root risks; leave-one-seed-out diagnostics"})
    return {"suite_curves": output / "aggregate_curves.csv", "suite_endpoints": output / "endpoints.csv",
            "suite_manifest": output / "manifest.json"}
