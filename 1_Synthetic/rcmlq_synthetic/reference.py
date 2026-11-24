"""Offline achieved EIG ellipsoid search at replayed posterior checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import product
import fcntl
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .backend import BACKEND_PATH
from .runner import atomic_json, atomic_write, initial_posterior
from metaworld_fast.ellipsoid_optimizer import ellipsoid_geometry, normalized_candidate_norms, optimize_ellipsoid
from metaworld_fast.ellipsoid_posterior import MarginalInverseCDF, joint_product_points
from metaworld_fast.line_template import make_template
from metaworld_fast.metrics import information_ratio, integrated_eig
from metaworld_fast.model import stable_seed

ALPHA, RADIUS = 0.5, 32.0


def _cube(mean, choices, inverse):
    rank = (choices - 1).bit_length()
    if 2**rank != choices or rank > len(mean):
        return None
    corners = np.zeros((choices, len(mean)))
    corners[:, :rank] = list(product((-1., 1.), repeat=rank))
    common_radius = RADIUS / np.linalg.norm(corners @ inverse, axis=1).max()
    return mean + common_radius * corners


def _evaluate(posterior, actual, template, t, job, protocol):
    mean, cov = template.mean, template.covariance
    seed = stable_seed(job["seed"], job["group"], job["interaction"], job["method"], f"ellipsoid-{t}")
    if posterior.factorized:
        marginals = [MarginalInverseCDF.from_marginal(marginal) for marginal in posterior.marginals]
        points = joint_product_points(marginals, protocol["points"], seed)
        refine = joint_product_points(marginals, protocol["refinement_points"], (seed + 1) % 2**64)
        weights = np.full(len(points), 1 / len(points))
        refine_weights = np.full(len(refine), 1 / len(refine))
    else:
        points = refine = posterior.particles
        weights = refine_weights = posterior.weights

    result = optimize_ellipsoid(
        points, weights, mean, cov, job["beta"], actual, template.queries,
        device=protocol["device"], alpha=ALPHA, radius=RADIUS, starts=protocol["starts"],
        steps=protocol["steps"], seed=seed,
    )
    _, inverse, _ = ellipsoid_geometry(mean, cov, ALPHA, RADIUS)
    eligible = [*template.queries, result["best_q"]]
    if result["actual_inside"]:
        eligible.append(actual)
    if t == 0:
        cube = _cube(mean, job["M"], inverse)
        if cube is not None:
            eligible.append(cube)

    def refine_eig(question):
        return integrated_eig(refine, refine_weights, question, job["beta"])

    values = np.array([refine_eig(q) for q in eligible])
    index = int(values.argmax())
    achieved = float(values[index])
    numerator = refine_eig(actual)
    approx_ratio, status = information_ratio(numerator, achieved)
    return {
        "t": t, "selected_eig": numerator, "achieved_eig": achieved, "ratio": approx_ratio,
        "ratio_status": status, "actual_feasible": result["actual_inside"],
        "reference_q": eligible[index], "search_seconds": result["seconds"],
        "exploratory_selected_eig": result["actual_eig"], "exploratory_achieved_eig": result["achieved_eig"],
        "searched_refined_eig": values[len(template.queries)],
        "reference_max_normalized_radius": np.max(normalized_candidate_norms(eligible[index], mean, inverse, RADIUS)),
        "integration_count": len(refine),
    }


def _evaluate_interaction(job, run_dir, destination, protocol):
    import torch

    torch.set_num_threads(1)
    start = perf_counter()
    key = job["key"]
    output = destination / f"{key}.npz"
    artifact = run_dir / "statistical" / f"{key}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as data:
            return {"key": key, "rounds": len(data["t"]), "reused": True}
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    if metadata["job"] != job or metadata["code_hash"] != protocol["campaign_code_hash"]:
        raise ValueError(f"Unexpected closed-loop metadata for {key}")
    with np.load(artifact, allow_pickle=False) as data:
        history = {name: data[name] for name in ("q", "y", "mu", "cov", "eig", "diag_t", "diag_eig")}
    if len(history["y"]) != job["answers"] or len(history["q"]) != job["answers"] + 1:
        raise ValueError(f"Incomplete trajectory for {key}")

    rounds = protocol["rounds"]
    if rounds == "all":
        rounds = range(job["answers"] + 1)
    elif rounds == "early_middle_terminal":
        rounds = (0, job["answers"] // 2, job["answers"])
    rounds = set(rounds)
    posterior = initial_posterior(job, np.random.default_rng(metadata["seeds"]["smc"]))
    records = []
    with threadpool_limits(limits=1):
        for t in range(max(rounds) + 1):
            if t in rounds:
                template = make_template(posterior, job["M"])
                np.testing.assert_allclose(template.mean, history["mu"][t], rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(template.covariance, history["cov"][t], rtol=2e-10, atol=2e-12)
                original_eig = posterior.eig(history["q"][t])
                np.testing.assert_allclose(original_eig, history["eig"][t], rtol=2e-10, atol=2e-12)
                record = _evaluate(posterior, history["q"][t], template, t, job, protocol)
                record["online_selected_eig"] = original_eig
                diagnostic = np.flatnonzero(history["diag_t"] == t)
                if len(diagnostic) and record["ratio_status"] == "resolved":
                    draws = history["diag_eig"][diagnostic[0]]
                    record["conditional_ratio"] = np.mean(draws) / record["achieved_eig"]
                    record["conditional_ratio_se"] = np.std(draws, ddof=1) / np.sqrt(len(draws)) / record["achieved_eig"]
                else:
                    record["conditional_ratio"] = record["conditional_ratio_se"] = np.nan
                records.append(record)
            if t < max(rounds):
                posterior.update(history["q"][t], int(history["y"][t]))

    with atomic_write(output) as stream:
        np.savez_compressed(stream, **{name: np.array([record[name] for record in records]) for name in records[0]})
    return {"key": key, "rounds": len(records), "elapsed_seconds": perf_counter() - start, "reused": False}


def evaluate_campaign(campaign, run_dir, *, rounds, all_rounds, device, starts, steps, points,
                      refinement_points, workers):
    import torch

    if starts < 2 or any(n < 2 or n & (n - 1) for n in (points, refinement_points)):
        raise ValueError("Use at least two starts and power-of-two integration counts")
    if refinement_points < points:
        raise ValueError("Refinement count must be at least the exploratory count")
    shortest = min(job["answers"] for job in campaign["jobs"])
    if rounds and any(not 0 <= t <= shortest for t in rounds):
        raise ValueError(f"Reference rounds must lie between 0 and {shortest}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if any(manifest[key] != value for key, value in campaign.items()):
        raise ValueError("Reference campaign does not match the completed run")
    digest = sha256(Path(__file__).read_bytes())
    for name in ("ellipsoid_optimizer.py", "ellipsoid_posterior.py"):
        digest.update((BACKEND_PATH / "metaworld_fast" / name).read_bytes())
    protocol = {
        "campaign_code_hash": manifest["code_hash"], "reference_code_hash": digest.hexdigest(),
        "rounds": sorted(set(rounds)) if rounds else "all" if all_rounds else "early_middle_terminal",
        "device": device, "torch": str(torch.__version__), "starts": starts, "steps": steps,
        "points": points, "refinement_points": refinement_points, "alpha": ALPHA, "radius": RADIUS,
        "reference_kind": "numerical_achieved_ellipsoid_eig",
    }
    label = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    destination = run_dir / "reference" / label
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "reference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(destination / "protocol.json", protocol)
        jobs = campaign["jobs"]
        if workers > 1:
            if device.startswith("cuda"):
                raise ValueError("Use a single process per CUDA device for reference evaluation")
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(_evaluate_interaction, job, run_dir, destination, protocol) for job in jobs]
                try:
                    for future in as_completed(futures):
                        print(json.dumps(future.result()), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        else:
            for job in jobs:
                print(json.dumps(_evaluate_interaction(job, run_dir, destination, protocol)), flush=True)
    print(destination)
