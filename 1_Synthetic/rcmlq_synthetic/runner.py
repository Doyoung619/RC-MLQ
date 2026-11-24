"""Closed-loop trajectories with resumable per-interaction checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from importlib import metadata
import json
from multiprocessing import get_context
from pathlib import Path
import os
import pickle
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .config import JOINT
from .backend import BACKEND_PATH
from .policies import RADIUS, choose
from metaworld_fast.line_template import SCALE_GRID, make_template
from metaworld_fast.metrics import information_ratio
from metaworld_fast.model import probabilities, stable_seed
from metaworld_fast.posterior_marginal import MarginalPosterior
from metaworld_fast.posterior_smc import SMCPosterior
from metaworld_fast.selectors import SelectorState

RANDOMIZED = ("PS", "GC", "SG", "SM")
PURPOSES = ("theta", "response", "selector", "smc", "diagnostics", "terminal")
SERIES = ("q", "q_raw", "y", "mu", "cov", "risk", "error", "eig", "grid_eig",
          "eta", "grid_eigs", "fisher_scores", "selected_scale", "ratio_status",
          "diag_t", "diag_eig", "ess", "resamplings", "acceptance", "remesh_count")


@contextmanager
def atomic_write(path, mode="wb"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode=mode, dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    with atomic_write(path, "w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")



def code_hash():
    numerical_files = ("line_template.py", "metrics.py", "model.py", "posterior_marginal.py",
                       "posterior_smc.py", "selectors.py")
    package_files = ("backend.py", "config.py", "policies.py", "runner.py")
    digest = hashlib.sha256()
    for base, names in ((BACKEND_PATH / "metaworld_fast", numerical_files),
                        (Path(__file__).parent, package_files)):
        for name in names:
            digest.update(name.encode())
            digest.update((base / name).read_bytes())
    return digest.hexdigest()


def initial_seeds(job):
    return {
        purpose: stable_seed(job["seed"], job["group"], job["interaction"],
                             None if purpose == "theta" else job["method"], purpose)
        for purpose in PURPOSES
    }


def initial_posterior(job, smc_rng):
    if job["method"] in JOINT:
        return SMCPosterior(job["d"], job["beta"], job["particles"], smc_rng)
    return MarginalPosterior(job["d"], job["beta"], job["nodes"])


def _record_moments(state):
    mean, covariance = state["posterior"].moments()
    values = (mean, covariance, np.trace(covariance), np.sum((mean - state["theta"])**2))
    for name, value in zip(("mu", "cov", "risk", "error"), values):
        state["data"][name].append(value)


def _new_state(job):
    seeds = initial_seeds(job)
    rng = {purpose: np.random.default_rng(seed) for purpose, seed in seeds.items()}
    posterior = initial_posterior(job, rng["smc"])
    state = {
        "seeds": seeds, "rng": rng, "posterior": posterior,
        "theta": rng["theta"].uniform(-1, 1, job["d"]), "policy": SelectorState(),
        "data": {name: [] for name in SERIES},
        "timings": {"policy": 0.0, "acquisition_grid_eig": 0.0, "evaluation_grid_eig": 0.0,
                    "update": 0.0, "diagnostics": 0.0, "checkpoint": 0.0},
        "elapsed": 0.0,
    }
    _record_moments(state)
    return state


def _next_query(job, state, t, terminal):
    posterior, data = state["posterior"], state["data"]
    grid = np.asarray(job["grid"])
    started = perf_counter()
    template = make_template(posterior, job["M"], grid)
    state["timings"]["policy"] += perf_counter() - started

    started = perf_counter()
    reference = template if np.array_equal(grid, SCALE_GRID) else make_template(posterior, job["M"])
    grid_values = np.array([posterior.eig(q) for q in reference.queries])
    scoring = grid_values if job["method"] == "RC-EIG" else None
    timing = "acquisition_grid_eig" if scoring is not None else "evaluation_grid_eig"
    state["timings"][timing] += perf_counter() - started

    started = perf_counter()
    policy = deepcopy(state["policy"]) if terminal else state["policy"]
    rng = state["rng"]["terminal" if terminal else "selector"]
    raw, q, scale = choose(job, posterior, template, t, rng, policy, scoring)
    if not np.isfinite(q).all() or np.any(np.linalg.norm(q - template.mean, axis=1) > RADIUS + 1e-10):
        raise FloatingPointError("Displayed question violates the radius-32 domain")
    state["timings"]["policy"] += perf_counter() - started

    reference_value = float(grid_values.max())
    started = perf_counter()
    eig = grid_values[grid.tolist().index(scale)] if scoring is not None else posterior.eig(q)
    eta, status = information_ratio(eig, reference_value)
    state["timings"]["evaluation_grid_eig"] += perf_counter() - started
    names = ("q", "q_raw", "eig", "grid_eig", "eta", "grid_eigs",
             "selected_scale", "fisher_scores", "ratio_status")
    values = (q, raw, eig, reference_value, eta, grid_values, np.nan if scale is None else scale,
              template.fisher_scores, status)
    for name, value in zip(names, values):
        data[name].append(value)

    if job["method"] in RANDOMIZED and t in job["diagnostic_rounds"]:
        started = perf_counter()
        diagnostic_policy = deepcopy(policy)
        diagnostic_rng = state["rng"]["diagnostics"]
        values = []
        for _ in range(64):
            _, diagnostic_q, _ = choose(job, posterior, template, t, diagnostic_rng, diagnostic_policy)
            values.append(posterior.eig(diagnostic_q))
        data["diag_t"].append(t)
        data["diag_eig"].append(values)
        state["timings"]["diagnostics"] += perf_counter() - started
    return q


def _receive_answer(job, state, q):
    started = perf_counter()
    data, posterior = state["data"], state["posterior"]
    answer = int(state["rng"]["response"].choice(job["M"], p=probabilities(state["theta"], q, job["beta"])))
    data["y"].append(answer)
    posterior.update(q, answer)
    diagnostics = posterior.last_diagnostics
    sweeps = diagnostics.get("sweeps", [])
    data["ess"].append(diagnostics.get("ess_after", np.nan))
    data["resamplings"].append(diagnostics.get("resamplings", 0))
    data["acceptance"].append(np.mean([sweep["acceptance"] for sweep in sweeps]) if sweeps else np.nan)
    data["remesh_count"].append(diagnostics.get("remesh_count", 0))
    _record_moments(state)
    state["timings"]["update"] += perf_counter() - started


def _save_checkpoint(path, job, digest, state):
    with atomic_write(path) as stream:
        pickle.dump({"job": job, "code_hash": digest, "state": state}, stream,
                    protocol=pickle.HIGHEST_PROTOCOL)


def run_interaction(job, run_dir, digest, *, max_answers=None):
    """Run one interaction, resuming only a matching locally created checkpoint."""
    key = job["key"]
    artifact = run_dir / "statistical" / f"{key}.npz"
    meta_path = artifact.with_suffix(".json")
    checkpoint = run_dir / "checkpoints" / f"{key}.pkl"
    if meta_path.exists() and artifact.exists():
        existing = json.loads(meta_path.read_text())
        if existing["job"] != job or existing["code_hash"] != digest:
            raise ValueError(f"Completed result belongs to a different job: {key}")
        with np.load(artifact, allow_pickle=False) as result:
            if len(result["y"]) != job["answers"] or len(result["q"]) != job["answers"] + 1:
                raise ValueError(f"Unexpected result length for {key}")
        checkpoint.unlink(missing_ok=True)
        return {"key": key, "elapsed_seconds": existing["elapsed_seconds"], "complete": True, "reused": True}
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["job"] != job or saved["code_hash"] != digest:
            raise ValueError(f"Checkpoint belongs to a different job: {key}")
        state = saved["state"]
    else:
        state = _new_state(job)

    target = job["answers"] if max_answers is None else min(job["answers"], max_answers)
    session_start = perf_counter()
    elapsed = state["elapsed"]
    with threadpool_limits(limits=1):
        for t in range(len(state["data"]["y"]), target):
            q = _next_query(job, state, t, terminal=False)
            _receive_answer(job, state, q)
            completed = t + 1
            if completed % job["checkpoint_every"] == 0 or completed == target:
                state["elapsed"] = elapsed + perf_counter() - session_start
                started = perf_counter()
                _save_checkpoint(checkpoint, job, digest, state)
                state["timings"]["checkpoint"] += perf_counter() - started
        complete = target == job["answers"]
        if complete:
            _next_query(job, state, target, terminal=True)
            arrays = {name: np.asarray(values) for name, values in state["data"].items()}
            with atomic_write(artifact) as stream:
                np.savez_compressed(stream, theta=state["theta"], **arrays)
            duration = elapsed + perf_counter() - session_start
            atomic_json(meta_path, {
                "job": job, "answers_completed": target, "code_hash": digest,
                "seeds": state["seeds"], "elapsed_seconds": duration,
                "time_breakdown_seconds": state["timings"],
                "posterior": "marginal_quadrature" if state["posterior"].factorized else "joint_smc",
                "reference_scope": "own_posterior_original_seven_scale_grid",
                "terminal_next_query_unanswered": True,
            })
            checkpoint.unlink(missing_ok=True)
        else:
            duration = state["elapsed"]
    return {"key": key, "elapsed_seconds": duration, "complete": complete, "reused": False}


def run_campaign(campaign, run_dir, workers):
    """Hold one campaign lock and distribute whole interactions to workers."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Campaign is already running: {run_dir}") from exc
        digest = code_hash()
        manifest = {
            **campaign, "code_hash": digest,
            "environment": {"python": sys.version.split()[0],
                            **{name: metadata.version(name) for name in ("numpy", "scipy", "threadpoolctl")}},
        }
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run directory has a different campaign, code version, or environment")
        atomic_json(manifest_path, manifest)
        jobs = campaign["jobs"]
        status_path = run_dir / "status.json"
        started = perf_counter()
        finished = 0

        def record(result):
            nonlocal finished
            finished += 1
            print(json.dumps({"finished": finished, "planned": len(jobs), **result}), flush=True)
            atomic_json(status_path, {"finished": finished, "planned": len(jobs),
                                     "last": result, "elapsed_seconds": perf_counter() - started})

        atomic_json(status_path, {"finished": 0, "planned": len(jobs)})
        if workers == 1:
            for job in jobs:
                record(run_interaction(job, run_dir, digest))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(run_interaction, job, run_dir, digest) for job in jobs]
                try:
                    for future in as_completed(futures):
                        record(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
