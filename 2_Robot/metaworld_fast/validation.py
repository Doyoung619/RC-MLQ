"""Independent numerical integration checks on immutable observed histories.

References re-evaluate the saved displayed questions; they never regenerate a
selector's adaptive interaction. Particle uncertainty is reported separately.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .reporting import atomic_json, finite_float, load_runs, read_json, write_csv

FROZEN_SETTINGS = ("m04_b04", "m16_b04", "m04_b16")
CHECK_COUNTS = (0, 32, 64, 128)
VALIDATION_VERSION = 2


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_hash():
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in
                                  ("validation.py", "posterior_marginal.py", "posterior_smc.py", "metrics.py", "model.py"))).hexdigest()


def _seed(*parts):
    payload = "|".join(map(str, (20261004, "numerical_reference") + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def evaluate_snapshot(posterior, data, t):
    mu, cov = posterior.moments()
    mu, cov = _numpy(mu), _numpy(cov)
    result = dict(mu=mu, cov=cov, risk=float(np.trace(cov)))
    if t < len(data["q"]):
        eig = float(_numpy(posterior.eig(data["q"][t])))
        # The reference family is frozen with the original interaction too.
        references = [float(_numpy(posterior.eig(q))) for q in data["reference_queries"][t]]
        reference_eig = max(references)
        result.update(eig=eig, reference_eig=reference_eig,
                      eta=eig / reference_eig if reference_eig > 0 else np.nan)
    return result


def recorded_snapshot(data, t):
    result = {key: np.asarray(data[key][t]) for key in ("mu", "cov", "risk")}
    if t < len(data["q"]):
        result.update({key: np.asarray(data[key][t]) for key in ("eig", "reference_eig", "eta")})
    return result


def compare_snapshots(base, references, relative_tolerance=.005, eta_tolerance=.005):
    """No unexplained error floors. Zero targets receive an absolute tolerance."""
    rows = []
    metrics = [("mean_l2", "mu", False), ("covariance_frobenius", "cov", False),
               ("risk", "risk", True)]
    metrics += [(key, key, key != "eta") for key in ("eig", "reference_eig", "eta") if key in base]
    for name, key, relative in metrics:
        values = np.stack([np.asarray(r[key], dtype=float) for r in references])
        mean = values.mean(axis=0)
        reference_se = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
        diff = np.asarray(base[key], dtype=float) - mean
        if name in ("mean_l2", "covariance_frobenius"):
            absolute_error = float(np.linalg.norm(diff))
            scale = float(np.linalg.norm(mean))
            uncertainty = float(np.linalg.norm(reference_se))
            # Mean and covariance matrix errors are descriptive; diagonal variance is assessed below.
            tolerance = None
            passed = None
        else:
            absolute_error, scale, uncertainty = abs(float(diff)), abs(float(mean)), float(reference_se)
            tolerance = relative_tolerance * scale if relative else eta_tolerance
            passed = bool(np.isfinite(absolute_error) and absolute_error <= tolerance + 3 * uncertainty)
        rows.append(dict(metric=name, original_value=np.asarray(base[key]).tolist(),
                         reference_mean=mean.tolist(), reference_se=reference_se.tolist(),
                         reference_replays=len(values), absolute_error=finite_float(absolute_error),
                         relative_error=finite_float(absolute_error / scale) if scale else None,
                         tolerance=tolerance, reference_uncertainty_norm=uncertainty,
                         decision="descriptive" if passed is None else "within_reference_uncertainty" if passed else "unresolved"))
    variances = np.stack([np.diag(np.asarray(r["cov"])) for r in references])
    mean = variances.mean(axis=0)
    se = variances.std(axis=0, ddof=1) / np.sqrt(len(variances)) if len(variances) > 1 else np.zeros_like(mean)
    original = np.diag(np.asarray(base["cov"]))
    for coordinate in range(len(mean)):
        error = abs(float(original[coordinate] - mean[coordinate]))
        tolerance = relative_tolerance * abs(float(mean[coordinate]))
        passed = np.isfinite(error) and error <= tolerance + 3 * float(se[coordinate])
        rows.append(dict(metric=f"variance_{coordinate}", original_value=float(original[coordinate]),
                         reference_mean=float(mean[coordinate]),
                         reference_se=float(se[coordinate]), reference_replays=len(references),
                         absolute_error=finite_float(error),
                         relative_error=finite_float(error / abs(mean[coordinate])) if mean[coordinate] else None,
                         tolerance=tolerance, reference_uncertainty_norm=float(se[coordinate]),
                         decision="within_reference_uncertainty" if passed else "unresolved"))
    return rows


def _replay(data, beta, kind, nodes, particles, seed, device, checks):
    # Keep this straightforward independent replay, independent of scheduler/RNG/checkpoint machinery.
    if kind == "marginal":
        from .posterior_marginal import MarginalPosterior
        posterior = MarginalPosterior(d=data["q"].shape[-1], beta=beta, nodes=nodes)
    else:
        from .posterior_smc import SMCPosterior
        posterior = SMCPosterior(d=data["q"].shape[-1], beta=beta, particles=particles,
                                 rng=np.random.default_rng(seed), device=device)
    result = {}
    if 0 in checks:
        result[0] = evaluate_snapshot(posterior, data, 0)
    for t, (q, y) in enumerate(zip(data["q"], data["y"]), start=1):
        posterior.update(q, int(y))
        if t in checks:
            result[t] = evaluate_snapshot(posterior, data, t)
    return result


def validate_history(run, run_dir, device="cpu"):
    start = time.monotonic()
    identity = (run["setting_id"], run["method"], run["interaction_id"])
    data = run["data"]
    checks = [t for t in CHECK_COUNTS if t <= len(data["y"])]
    beta = float(run["meta"].get("beta", int(run["setting_id"].split("_b")[1])))
    kind = "marginal" if run["method"] in ("RC", "PQ", "QI") else "joint"
    signature = _digest(run["path"])
    code_hash = _code_hash()
    output = Path(run_dir) / "validation" / ("_".join(map(str, identity)) + ".json")
    previous = read_json(output)
    if previous.get("source_sha256") == signature and previous.get("validation_version") == VALIDATION_VERSION and previous.get("validation_code_hash") == code_hash:
        return previous
    if "reference_queries" not in data:
        result = dict(identity=identity, status="unresolved", reason="Saved frozen reference questions absent",
                      source_sha256=signature, validation_version=VALIDATION_VERSION)
        atomic_json(output, result)
        return result
    repetitions = 1 if kind == "marginal" else 2
    replays = [_replay(data, beta, kind, nodes=513, particles=8192,
                       seed=_seed(*identity, 8192, repeat), device=device, checks=checks)
               for repeat in range(repetitions)]
    rows = []
    for t in checks:
        for row in compare_snapshots(recorded_snapshot(data, t), [r[t] for r in replays]):
            rows.append(dict(answer_count=t, **row))
    unresolved = any(r["decision"] == "unresolved" for r in rows)
    refinement_rows = []
    refinement = None
    if unresolved:
        refinement = _replay(data, beta, kind, nodes=1025, particles=16384,
                              seed=_seed(*identity, 16384, 0), device=device, checks=checks)
        for t in checks:
            # Assess reference stability separately from the original 4096-particle approximation.
            center = {key: np.mean([r[t][key] for r in replays], axis=0) for key in replays[0][t]}
            for row in compare_snapshots(center, [refinement[t]]):
                refinement_rows.append(dict(answer_count=t, comparison="reference_mean_vs_refinement", **row))
        if kind == "marginal":
            # A higher deterministic integration reference can replace the lower one; never rewrite original results.
            rows = [dict(answer_count=t, **row) for t in checks
                    for row in compare_snapshots(recorded_snapshot(data, t), [refinement[t]])]
            unresolved = any(r["decision"] == "unresolved" for r in rows + refinement_rows)
    result = dict(identity=identity, status="unresolved" if unresolved else "within_reference_uncertainty",
                  source=str(run["path"]), source_sha256=signature,
                  validation_version=VALIDATION_VERSION, validation_code_hash=code_hash, kind=kind, checks=checks,
                  reference_nodes=513 if kind == "marginal" else None,
                  reference_particles=8192 if kind == "joint" else None,
                  independent_reference_replays=repetitions,
                  refinement_attempted=refinement is not None,
                  refinement_nodes=1025 if refinement is not None and kind == "marginal" else None,
                  refinement_particles=16384 if refinement is not None and kind == "joint" else None,
                  comparisons=rows, reference_refinement_comparisons=refinement_rows,
                  reference_uncertainty_note="SE across two independent reference replays is itself imprecise; 3 SE is a descriptive numerical check, not a confidence claim. Original particle approximation uncertainty is not estimated by this reference SE." if kind == "joint" else "Deterministic resolution comparison; no Monte Carlo SE.",
                  elapsed_seconds=time.monotonic() - start)
    atomic_json(output, result)
    return result


def _dense_snapshot(q, y, beta, nodes, probe):
    """Independent tensor Gauss-Legendre integration, only used for d=2 checks."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    logweights = np.log(np.outer(w, w).ravel()) - np.log(4.0)
    for question, answer in zip(q, y):
        logits = 2 * beta * points @ question.T - beta * np.sum(question**2, axis=-1)
        logweights += logits[:, int(answer)] - logsumexp(logits, axis=-1)
    normalized_logweights = logweights - logsumexp(logweights)
    weights = np.exp(normalized_logweights)
    mu = weights @ points
    centered = points - mu
    cov = (centered * weights[:, None]).T @ centered
    logits = 2 * beta * points @ probe.T - beta * np.sum(probe**2, axis=-1)
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    pbar_log = logsumexp(normalized_logweights[:, None] + logp, axis=0)
    a = logp - pbar_log
    p, pbar = np.exp(logp), np.exp(pbar_log)
    contribution = p * a - p + pbar
    small = np.abs(a) < 1e-3
    x = a[small]
    contribution[small] = np.broadcast_to(pbar, a.shape)[small] * x*x * (1/2 + x*(1/3 + x*(1/8 + x*(1/30 + x/144))))
    eig = np.sum(weights[:, None] * contribution)
    return dict(mu=mu, cov=cov, risk=float(np.trace(cov)), eig=float(eig))


def validate_small_cases(run_dir, device="cpu"):
    """Three preregistered 2-D interactions, fixed joint random queries, T=32."""
    from .posterior_smc import SMCPosterior
    output = Path(run_dir) / "validation/small_d2_t32.json"
    previous = read_json(output)
    if previous.get("validation_version") == VALIDATION_VERSION and previous.get("validation_code_hash") == _code_hash():
        return previous
    cases = []
    for choices in (2, 3, 4):
        started = time.monotonic()
        rng = np.random.default_rng(_seed("small_case", choices))
        theta = rng.uniform(-1, 1, 2)
        questions = rng.normal(0, .8, size=(32, choices, 2))
        answers = []
        for question in questions:
            logits = -4 * np.sum((question - theta)**2, axis=-1)
            answers.append(rng.choice(choices, p=np.exp(logits - logsumexp(logits))))
        coarse = _dense_snapshot(questions, answers, 4., 81, questions[-1])
        dense = _dense_snapshot(questions, answers, 4., 161, questions[-1])
        particle = []
        for repeat in range(2):
            posterior = SMCPosterior(d=2, beta=4., particles=8192,
                                     rng=np.random.default_rng(_seed("small_smc", choices, repeat)), device=device)
            for question, answer in zip(questions, answers):
                posterior.update(question, int(answer))
            mu, cov = posterior.moments()
            cov = _numpy(cov)
            particle.append(dict(mu=_numpy(mu), cov=cov, risk=float(np.trace(cov)),
                                 eig=float(_numpy(posterior.eig(questions[-1])))))
        dense_rows = compare_snapshots(coarse, [dense])
        # Base=dense, references=independent particles measures particle integration error including MC spread.
        smc_rows = compare_snapshots(dense, particle)
        case = dict(choices=choices, dimension=2, answers=32, beta=4.,
                    quadrature_nodes_per_axis=[81, 161], reference_particles=8192,
                    independent_particle_replays=2, quadrature_resolution_comparisons=dense_rows,
                    smc_vs_dense_comparisons=smc_rows,
                    status="unresolved" if any(r["decision"] == "unresolved" for r in dense_rows + smc_rows) else "within_reference_uncertainty",
                    elapsed_seconds=time.monotonic() - started,
                    probe_note="Last saved question evaluated at final posterior for numerical integration comparison; no extra actual answer and not a pre-answer eta point.")
        cases.append(case)
        history_path = Path(run_dir) / f"validation/small_d2_m{choices}_history.npz"
        np.savez(history_path, q=questions, y=answers, theta=theta)
    result = dict(validation_version=VALIDATION_VERSION, validation_code_hash=_code_hash(), cases=cases,
                  status="unresolved" if any(c["status"] == "unresolved" for c in cases) else "within_reference_uncertainty")
    atomic_json(output, result)
    return result


def validate_run(run_dir, config=None, device="cpu", include_small=True):
    start = time.monotonic()
    root = Path(run_dir)
    (root / "validation").mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    selected = [r for r in runs if r["setting_id"] in FROZEN_SETTINGS and r["interaction_id"] == 0]
    results, errors = [], []
    for run in selected:
        identity = (run["setting_id"], run["method"], run["interaction_id"])
        print(f"VALIDATION start {identity}", flush=True)
        try:
            result = validate_history(run, root, device=device)
            results.append(result)
            print(f"VALIDATION {identity}: {result['status']}", flush=True)
        except Exception as exc:
            errors.append(dict(identity=identity, error=f"{type(exc).__name__}: {exc}"))
            print(f"VALIDATION error {identity}: {exc}", flush=True)
        atomic_json(root / "validation/progress.json", dict(completed=len(results), errors=errors,
                                                            expected_histories=18, elapsed_seconds=time.monotonic()-start))
    small = None
    if include_small:
        try:
            small = validate_small_cases(root, device=device)
        except Exception as exc:
            errors.append(dict(identity="small_d2_t32", error=f"{type(exc).__name__}: {exc}"))
    discrepancies = []
    for result in results:
        for row in result.get("comparisons", []) + result.get("reference_refinement_comparisons", []):
            item = dict(setting_id=result["identity"][0], method=result["identity"][1], interaction_id=0, **row)
            for key, value in list(item.items()):
                if isinstance(value, (list, dict)):
                    item[key] = json.dumps(value)
            discrepancies.append(item)
    write_csv(root / "tables/numerical_discrepancies.csv", discrepancies)
    unresolved = [r["identity"] for r in results if r["status"] == "unresolved"]
    missing = 18 - len(results)
    status = "unresolved" if unresolved or errors or missing or small and small["status"] == "unresolved" else "within_reference_uncertainty"
    result = dict(status=status, validation_version=VALIDATION_VERSION,
                  selection_rule="interaction 0, all six methods, settings m04_b04, m16_b04, m04_b16; fixed before method comparisons",
                  checkpoints=list(CHECK_COUNTS), selected_histories=len(selected), completed_histories=len(results),
                  expected_histories=18, missing_histories=missing, unresolved_histories=unresolved, errors=errors,
                  small_case_status=small["status"] if small else "not_run",
                  relative_variance_eig_tolerance=.005, absolute_eta_tolerance=.005,
                  refinement_budget="At most one additional 1025-node or 16384-particle replay per unresolved history",
                  uncertainty_interpretation="Independent 8192-particle reference SE is separated from original approximation error. Two replicates give an imprecise descriptive SE, not an exact truth or confidence guarantee.",
                  elapsed_seconds=time.monotonic() - start)
    atomic_json(root / "validation/summary.json", result)
    return result
