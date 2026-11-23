from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import RunSpec, expand_config, load_yaml
from .datasets import fetch_and_preprocess_openml
from .experiment import load_run_trajectories, run_spec
from .plotting import make_openml_family_comparison_figures
from .reporting import generate_report
from .utils import atomic_write_json, environment_manifest, jsonable


def _run_worker(spec: RunSpec, project_root: str, overwrite: bool) -> str:
    return str(run_spec(spec, Path(project_root), overwrite=overwrite))


def command_plan(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    specs = expand_config(config)
    payload = [dict(index=i, run_id=spec.run_id, spec=jsonable(spec.__dict__)) for i, spec in enumerate(specs)]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".jsonl":
            output.write_text("\n".join(json.dumps(item, sort_keys=True) for item in payload) + "\n", encoding="utf-8")
        else:
            atomic_write_json(output, payload)
    print(f"Planned {len(specs)} runs from {args.config}")
    for item in payload[: min(args.show, len(payload))]:
        print(f"[{item['index']:05d}] {item['run_id']}")
    if len(payload) > args.show:
        print(f"... {len(payload) - args.show} additional runs")
    return 0


def command_run_job(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    specs = expand_config(config)
    if args.index < 0 or args.index >= len(specs):
        raise IndexError(f"Run index {args.index} outside [0, {len(specs) - 1}]")
    path = run_spec(specs[args.index], Path(args.project_root).resolve(), overwrite=args.overwrite)
    print(path)
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    specs = expand_config(config)
    project_root = Path(args.project_root).resolve()
    workers = max(1, int(args.workers))
    print(f"Running {len(specs)} jobs with {workers} worker(s)")
    failures: list[tuple[str, str]] = []
    if workers == 1:
        for spec in tqdm(specs, desc="runs"):
            try:
                run_spec(spec, project_root, overwrite=args.overwrite)
            except Exception as exc:
                failures.append((spec.run_id, repr(exc)))
                if args.fail_fast:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_worker, spec, str(project_root), args.overwrite): spec
                for spec in specs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="runs"):
                spec = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((spec.run_id, repr(exc)))
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        raise
    if failures:
        print(f"{len(failures)} run(s) failed:", file=sys.stderr)
        for run_id, error in failures[:20]:
            print(f"  {run_id}: {error}", file=sys.stderr)
        return 1
    print("All requested runs completed or were resumed from checkpoints.")
    return 0


def command_prefetch(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    if config.get("source") != "openml":
        print("No OpenML datasets in this configuration.")
        return 0
    cache_root = Path(args.cache_root or config.get("cache_root", "data/openml_cache"))
    if not cache_root.is_absolute():
        cache_root = Path(args.project_root).resolve() / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    # Expand the YAML so preprocessing grids are honored, then deduplicate the
    # processed matrix cache keys. The preprocessing function itself ignores
    # theta/noise fields in its cache signature.
    unique: dict[str, dict[str, Any]] = {}
    for spec in expand_config(config):
        dataset = dict(spec.dataset)
        signature = json.dumps(
            {
                key: dataset.get(key)
                for key in [
                    "data_id",
                    "preprocess_seed",
                    "dimension",
                    "max_rows",
                    "radius",
                    "min_category_frequency",
                    "max_categories_per_feature",
                ]
            },
            sort_keys=True,
        )
        unique.setdefault(signature, dataset)

    for dataset in unique.values():
        data_id = int(dataset["data_id"])
        print(
            f"Fetching and preprocessing OpenML data_id={data_id}, "
            f"d={dataset.get('dimension', 20)}, preprocess_seed={dataset.get('preprocess_seed', 0)}"
        )
        fetch_and_preprocess_openml(
            dataset,
            cache_root=cache_root,
            seed=int(dataset.get("preprocess_seed", 0)),
        )
    print(f"Cached {len(unique)} processed matrix specification(s) under {cache_root}")
    return 0


def _validate_config_completion(config_path: str, project_root: Path) -> tuple[int, list[str], list[str]]:
    config = load_yaml(config_path)
    specs = expand_config(config)
    missing: list[str] = []
    failed: list[str] = []
    for spec in specs:
        output_root = Path(spec.output_root)
        if not output_root.is_absolute():
            output_root = project_root / output_root
        run_dir = output_root / spec.experiment_name / spec.run_id
        if (run_dir / "failed.json").exists():
            failed.append(spec.run_id)
        elif not (run_dir / "complete.json").exists() or not (run_dir / "trajectory.csv").exists():
            missing.append(spec.run_id)
        elif spec.protocol.get("algorithm_suite") == "spec_v1":
            from .spec_suite.runner import validate_completed_run
            try:
                validate_completed_run(run_dir)
            except (ValueError, KeyError, OSError, RuntimeError):
                failed.append(spec.run_id)
    return len(specs), missing, failed


def command_validate(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    any_problem = False
    for config_path in args.config:
        total, missing, failed = _validate_config_completion(config_path, project_root)
        complete = total - len(missing) - len(failed)
        print(f"{config_path}: complete={complete}/{total}, missing={len(missing)}, failed={len(failed)}")
        for run_id in failed[:10]:
            print(f"  FAILED  {run_id}")
        for run_id in missing[:10]:
            print(f"  MISSING {run_id}")
        any_problem = any_problem or bool(missing or failed)
    return 1 if any_problem else 0


def command_report(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    for config_path in args.require_config or []:
        total, missing, failed = _validate_config_completion(config_path, project_root)
        if missing or failed:
            raise RuntimeError(
                f"Refusing to report incomplete grid {config_path}: "
                f"complete={total-len(missing)-len(failed)}/{total}, "
                f"missing={len(missing)}, failed={len(failed)}"
            )
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    paper_dir = Path(args.paper_dir)
    if not paper_dir.is_absolute():
        paper_dir = project_root / paper_dir
    frame = load_run_trajectories(results_root)
    outputs = generate_report(frame, paper_dir)
    print("Generated paper artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def command_appendix_new(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    frame = load_run_trajectories(results_root)
    outputs = make_openml_family_comparison_figures(
        frame,
        output_dir,
        theta_regime=args.theta_regime,
        max_t=args.max_t,
    )
    print(f"Generated {len(outputs)} family-comparison PDFs under {output_dir}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    payload: dict[str, Any] = {
        "project_root": str(project_root),
        "environment": environment_manifest(),
        "cpu_count": os.cpu_count(),
        "writable": os.access(project_root, os.W_OK),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Projected-score experiment harness")
    parser.add_argument("--project-root", default=".", help="Harness repository root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Expand a YAML grid without running it")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output")
    plan.add_argument("--show", type=int, default=20)
    plan.set_defaults(func=command_plan)

    run = subparsers.add_parser("run", help="Run or resume all jobs in a YAML grid")
    run.add_argument("--config", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(func=command_run)

    run_job = subparsers.add_parser("run-job", help="Run one expanded-grid job by index")
    run_job.add_argument("--config", required=True)
    run_job.add_argument("--index", type=int, required=True)
    run_job.add_argument("--overwrite", action="store_true")
    run_job.set_defaults(func=command_run_job)

    prefetch = subparsers.add_parser("prefetch-openml", help="Download raw OpenML datasets once")
    prefetch.add_argument("--config", required=True)
    prefetch.add_argument("--cache-root")
    prefetch.set_defaults(func=command_prefetch)

    validate = subparsers.add_parser("validate", help="Check that every planned run completed")
    validate.add_argument("--config", action="append", required=True, help="May be supplied multiple times")
    validate.set_defaults(func=command_validate)

    report = subparsers.add_parser("report", help="Aggregate runs, make figures, and emit LaTeX")
    report.add_argument("--results-root", default="results")
    report.add_argument("--paper-dir", default="paper")
    report.add_argument(
        "--require-config",
        action="append",
        default=[],
        help="Refuse to report unless this YAML grid is complete; may be repeated",
    )
    report.set_defaults(func=command_report)

    appendix_new = subparsers.add_parser(
        "appendix-new", help="Generate per-dataset Bernoulli/Gaussian/Poisson 3x2 comparison PDFs"
    )
    appendix_new.add_argument("--results-root", default="results")
    appendix_new.add_argument("--output-dir", default="paper/generated/appendix_new_figures")
    appendix_new.add_argument("--theta-regime", default="weak")
    appendix_new.add_argument("--max-t", type=int, default=200)
    appendix_new.set_defaults(func=command_appendix_new)

    doctor = subparsers.add_parser("doctor", help="Print environment information")
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
