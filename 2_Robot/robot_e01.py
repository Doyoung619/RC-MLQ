"""Compact, paired E01 follow-up using the existing Robot experiment engines."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from metaworld_fast.line_template import SCALE_GRID, make_template
from metaworld_fast.metrics import information_ratio
from metaworld_fast.model import probabilities, stable_seed
from metaworld_fast.posterior_marginal import MarginalPosterior
from metaworld_fast.selectors import SelectorState, select_query
from metaworld_fast.statistical_runner import _atomic_npz, run_key
from metaworld_fast.util import atomic_json, digest, source_manifest, utcnow


SETTING = "m04_b04"
FIXED_SCALES = {"Fixed0p125": 0.125, "Fixed0p5": 0.5, "Fixed2p0": 2.0}
METHODS = ("RC", "DirectEIG", "PQ", *FIXED_SCALES)


def protocol(interactions, answers, root_seed, ellipsoid):
    return {
        "setting_id": SETTING,
        "M": 4,
        "dimension": 4,
        "beta": 4.0,
        "nodes": 257,
        "answers": answers,
        "root_seed": root_seed,
        "interaction_ids": list(range(interactions)),
        "methods": list(METHODS),
        "scale_grid": SCALE_GRID.tolist(),
        "fixed_scales": FIXED_SCALES,
        "selected_candidate_replays": interactions * answers * len(METHODS) * 4,
        "ellipsoid": (
            {
                "preanswer_rounds": sorted({0, answers // 2, answers - 1}),
                "alpha": 0.5,
                "radius": 16.0,
                "starts": 8,
                "steps": 160,
                "marginal_samples": 4096,
                "learning_rate": 0.002,
                "root_seed": 20261005,
                "device": "cpu",
            }
            if ellipsoid else None
        ),
    }


def jobs_for(config):
    shared = {key: config[key] for key in (
        "setting_id", "M", "dimension", "beta", "answers", "nodes", "root_seed",
    )}
    return [
        dict(shared, interaction_id=interaction, method=method)
        for interaction in config["interaction_ids"]
        for method in config["methods"]
    ]


def choose_query(method, posterior, step, rng, state, template, grid_eigs):
    if method == "DirectEIG":
        index = int(np.argmax(grid_eigs))
    elif method in FIXED_SCALES:
        index = int(np.flatnonzero(SCALE_GRID == FIXED_SCALES[method])[0])
    elif method == "RC":
        index = template.rc_index
    else:
        _, question = select_query(method, posterior, len(template.b), step, rng, state, template)
        return question, posterior.eig(question), np.nan
    return template.queries[index].copy(), float(grid_eigs[index]), float(SCALE_GRID[index])


def infer_one(job, root):
    """An interaction is the restart unit. Completed histories are atomic."""
    started = time.perf_counter()
    method, interaction, steps = job["method"], job["interaction_id"], job["answers"]
    seeds = {
        purpose: stable_seed(
            job["root_seed"], job["setting_id"], interaction,
            None if purpose == "theta" else method, purpose,
        )
        for purpose in ("theta", "response", "selector")
    }
    theta_rng, response_rng, selector_rng = (np.random.default_rng(seed) for seed in seeds.values())
    theta = theta_rng.uniform(-1, 1, job["dimension"])
    posterior = MarginalPosterior(job["dimension"], job["beta"], job["nodes"])
    selector_state = SelectorState()
    means, covariances = [], []
    series = {
        key: [] for key in (
            "q", "y", "eig", "reference_eig", "eta_grid", "reference_queries",
            "reference_eigs", "fisher_scores", "selected_scale", "ratio_status",
        )
    }
    for step in range(steps):
        template = make_template(posterior, job["M"])
        grid_eigs = np.array([posterior.eig(q) for q in template.queries])
        question, eig, scale = choose_query(
            method, posterior, step, selector_rng, selector_state, template, grid_eigs,
        )
        reference = float(np.max(grid_eigs))
        ratio, ratio_status = information_ratio(eig, reference)
        if method == "DirectEIG" and ratio_status == "resolved" and ratio != 1.0:
            raise FloatingPointError("Direct EIG must attain its own grid reference")
        answer = int(response_rng.choice(job["M"], p=probabilities(theta, question, job["beta"])))
        for key, value in (
            ("q", question), ("y", answer), ("eig", eig), ("reference_eig", reference),
            ("eta_grid", ratio), ("reference_queries", template.queries),
            ("reference_eigs", grid_eigs), ("fisher_scores", template.fisher_scores),
            ("selected_scale", scale), ("ratio_status", ratio_status),
        ):
            series[key].append(value)
        means.append(template.mean)
        covariances.append(template.covariance)
        posterior.update(question, answer)
    mean, covariance = posterior.moments()
    mu, cov = np.array([*means, mean]), np.array([*covariances, covariance])
    arrays = {name: np.asarray(values) for name, values in series.items()}
    arrays.update(mu=mu, cov=cov, theta=theta, risk=np.trace(cov, axis1=1, axis2=2),
                  error=np.sum((mu - theta)**2, axis=1))
    path = root / "statistical" / f"{run_key(job)}.npz"
    _atomic_npz(path, arrays)
    atomic_json(path.with_suffix(".json"), dict(
        job, seeds=seeds, answers_completed=steps, elapsed_seconds=time.perf_counter() - started,
        reference_scope="same_posterior_seven_scale_grid",
    ))
    return path


def replay_one(pool, path, root, job):
    with np.load(path, allow_pickle=False) as archive:
        questions = archive["q"]
    interaction = job["interaction_id"]
    requests = [
        dict(request_id=f"{path.stem}__t{step:03d}", context_id=interaction,
             q=question, interaction_id=interaction, t=step, method=job["method"], kind="selected")
        for step, question in enumerate(questions)
    ]
    prefix = root / "rollouts" / path.stem
    result = pool.execute_queries(requests, prefix)
    report = json.loads(prefix.with_suffix(".json").read_text())
    result["command_reconstruction_failures"] = sum(
        not clip.get("command_reconstruction_ok", False) for clip in report["per_candidate"]
    )
    atomic_json(root / "rollouts" / f"{path.stem}.done.json", result)
    return result


def audit_one(path, root, job, config):
    from metaworld_fast.ellipsoid_posterior import (
        MarginalInverseCDF, PosteriorSnapshot, joint_product_points,
    )
    from metaworld_fast.ellipsoid_recompute import optimize_batch
    from metaworld_fast.statistical_runner import replay_history

    started = time.perf_counter()
    rounds = config["preanswer_rounds"]
    with np.load(path, allow_pickle=False) as archive:
        histories = replay_history(
            archive["q"][:max(rounds)], archive["y"][:max(rounds)],
            job["beta"], "marginal", nodes=job["nodes"], checkpoints=rounds,
        )
        integration_count = config["marginal_samples"]
        integration_seed = stable_seed(
            config["root_seed"], job["setting_id"], job["interaction_id"], job["method"],
            "e01_ellipsoid_product_integration",
        )
        snapshots = []
        for step, posterior in histories.items():
            mean, covariance = posterior.moments()
            np.testing.assert_allclose(mean, archive["mu"][step], rtol=2e-10, atol=2e-12)
            np.testing.assert_allclose(covariance, archive["cov"][step], rtol=2e-10, atol=2e-12)
            laws = tuple(MarginalInverseCDF.from_marginal(marginal) for marginal in posterior.marginals)
            points = joint_product_points(laws, integration_count, integration_seed)
            snapshots.append(PosteriorSnapshot(
                step, points, np.full(integration_count, 1 / integration_count), mean, covariance,
                archive["q"][step], archive["reference_queries"][step], job["beta"],
                job["method"], path.stem, integration_seed, float(archive["eig"][step]),
                archive["reference_eigs"][step],
            ))
    result = optimize_batch(snapshots, config, config["device"], path.stem)
    records = []
    for index, snapshot in enumerate(snapshots):
        actual = float(result["actual_eig"][index])
        achieved = float(result["achieved_eig"][index])
        ratio, status = information_ratio(actual, achieved)
        records.append({
            "t": snapshot.t, "actual_eig": actual, "reference_eig": achieved,
            "approximate_ellipsoid_ratio": ratio if np.isfinite(ratio) else None,
            "ratio_status": status, "actual_inside": bool(result["actual_inside"][index]),
            "achieved_query": result["best_q"][index],
            "grid_eigs": result["grid_eigs"][index],
        })
    outcome = {
        "records": records, "elapsed_seconds": time.perf_counter() - started,
        "global_optimum_certified": False,
    }
    atomic_json(root / "ellipsoid" / path.with_suffix(".json").name, outcome)
    return outcome


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Active run already owns {root}") from exc
        yield


def summarize(paths, jobs, replay_results, audits, phase_seconds):
    endpoints = {method: {} for method in METHODS}
    inference_worker_seconds = 0.
    for job in jobs:
        path = paths[run_key(job)]
        with np.load(path, allow_pickle=False) as history:
            endpoints[job["method"]][job["interaction_id"]] = float(history["risk"][-1]), float(history["error"][-1])
        inference_worker_seconds += json.loads(path.with_suffix(".json").read_text())["elapsed_seconds"]
    reference = endpoints["RC"]
    interactions = sorted(reference)
    bootstrap_count = 2000 if len(interactions) >= 2 else 0
    bootstrap_indices = np.random.default_rng(20261004).integers(
        len(interactions), size=(bootstrap_count, len(interactions)),
    )
    method_summary = {}
    for method, values in endpoints.items():
        risk, error = np.mean(list(values.values()), axis=0)
        differences = np.array([np.subtract(values[index], reference[index]) for index in interactions])
        paired = differences.mean(axis=0)
        intervals = [None, None]
        if bootstrap_count:
            means = differences[bootstrap_indices].mean(axis=1)
            intervals = np.quantile(means, (0.025, 0.975), axis=0).T.tolist()
        method_summary[method] = {
            "interactions": len(values), "mean_final_posterior_mse": float(risk),
            "mean_final_squared_error": float(error),
            "mean_paired_posterior_mse_minus_rc": float(paired[0]),
            "mean_paired_squared_error_minus_rc": float(paired[1]),
            "paired_posterior_mse_minus_rc_ci95": intervals[0],
            "paired_squared_error_minus_rc_ci95": intervals[1],
        }
    counts = {
        key: sum(result[key] for result in replay_results)
        for key in ("logical_requests", "unique_executions", "cache_hits", "failed",
                    "simulation_valid", "command_reconstruction_failures")
    }
    return {
        "methods": method_summary, "rollout_counts": counts, "phase_wall_seconds": phase_seconds,
        "paired_bootstrap_replicates": bootstrap_count,
        "inference_worker_seconds": inference_worker_seconds,
        "rollout_shard_seconds": sum(result["elapsed_seconds"] for result in replay_results),
        "ellipsoid_histories": sum(len(audit["records"]) for audit in audits),
        "ellipsoid_worker_seconds": sum(audit["elapsed_seconds"] for audit in audits),
    }


def run(config, root, workers, rollout_workers):
    from metaworld_fast.contexts import prepare_contexts
    from metaworld_fast.rollout_pool import RolloutPool

    with locked(root):
        manifest = {
            "protocol": config,
            "source_sha256": digest(source_manifest() | {Path(__file__).name: Path(__file__).read_text()}),
            "versions": {name: version(name) for name in ("numpy", "scipy", "metaworld", "mujoco", "torch")},
            "response_source": "coordinate_model",
            "trajectory_source": "existing_reach_v3_realization_adapter",
        }
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen run differs; use a new --run-dir")
        atomic_json(manifest_path, manifest)
        (root / "statistical").mkdir(exist_ok=True)
        jobs = jobs_for(config)
        paths = {run_key(job): root / "statistical" / f"{run_key(job)}.npz" for job in jobs}
        completed = {"statistical": 0, "rollouts": 0, "ellipsoid": 0}
        totals = {"statistical": len(jobs), "rollouts": len(jobs),
                  "ellipsoid": len(jobs) if config["ellipsoid"] else 0}
        phase_seconds = {}

        def status(stage, error=None):
            state = {"stage": stage, "completed": completed, "totals": totals,
                     "pid": os.getpid(), "updated_at": utcnow(), "error": error}
            atomic_json(root / "status.json", state)
            print(json.dumps(state), flush=True)

        context = mp.get_context("spawn")
        try:
            status("inference")
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = []
                for job in jobs:
                    path = paths[run_key(job)]
                    if path.is_file() and path.with_suffix(".json").is_file():
                        completed["statistical"] += 1
                    else:
                        futures.append(pool.submit(infer_one, job, root))
                for future in as_completed(futures):
                    future.result()
                    completed["statistical"] += 1
                    status("inference")
            phase_seconds["inference"] = time.perf_counter() - started

            status("replay")
            started = time.perf_counter()
            context_dir = root / "contexts"
            prepare_contexts(context_dir, count=len(config["interaction_ids"]), root_seed=config["root_seed"])
            replay_results = []
            with RolloutPool(context_dir, workers=rollout_workers) as pool:
                for job in jobs:
                    path = paths[run_key(job)]
                    prefix = root / "rollouts" / path.stem
                    marker = prefix.with_name(prefix.name + ".done.json")
                    output_files = (Path(str(prefix) + ".npz"), Path(str(prefix) + ".json"))
                    if marker.is_file() and all(file.is_file() for file in output_files):
                        replay_results.append(json.loads(marker.read_text()))
                    else:
                        replay_results.append(replay_one(pool, path, root, job))
                    completed["rollouts"] += 1
                    status("replay")
            phase_seconds["replay"] = time.perf_counter() - started

            audits = []
            if config["ellipsoid"]:
                status("ellipsoid")
                started = time.perf_counter()
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = []
                    for job in jobs:
                        path = paths[run_key(job)]
                        result = root / "ellipsoid" / path.with_suffix(".json").name
                        if result.is_file():
                            audits.append(json.loads(result.read_text()))
                            completed["ellipsoid"] += 1
                        else:
                            futures.append(pool.submit(audit_one, path, root, job, config["ellipsoid"]))
                    for future in as_completed(futures):
                        audits.append(future.result())
                        completed["ellipsoid"] += 1
                        status("ellipsoid")
                phase_seconds["ellipsoid"] = time.perf_counter() - started

            summary = summarize(paths, jobs, replay_results, audits, phase_seconds)
            atomic_json(root / "summary.json", summary)
            counts = summary["rollout_counts"]
            failures = counts["failed"] + counts["command_reconstruction_failures"]
            status("complete_with_audit_failures" if failures else "complete")
            return int(bool(failures))
        except Exception as exc:
            status("error", str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--interactions", type=int, default=10)
    parser.add_argument("--answers", type=int, default=128)
    parser.add_argument("--root-seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--ellipsoid", action="store_true", help="audit three pre-answer checkpoints")
    args = parser.parse_args()
    if args.command != "plan" and args.run_dir is None:
        parser.error("--run-dir is required")
    if args.command == "status":
        print((args.run_dir / "status.json").read_text(), end="")
        return 0
    if not (1 <= args.interactions <= 10 and 1 <= args.answers <= 128):
        parser.error("interactions must be in 1..10 and answers in 1..128")
    if min(args.workers, args.rollout_workers) < 1:
        parser.error("worker counts must be positive")
    config = protocol(args.interactions, args.answers, args.root_seed, args.ellipsoid)
    if args.command == "plan":
        print(json.dumps(config | {"runs": len(jobs_for(config))}, indent=2))
        return 0
    return run(config, args.run_dir.resolve(), args.workers, args.rollout_workers)


if __name__ == "__main__":
    raise SystemExit(main())
