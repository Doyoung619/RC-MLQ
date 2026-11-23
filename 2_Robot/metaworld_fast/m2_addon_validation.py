"""Immutable-history audits and inexpensive ellipsoid bounds for the M=2 addon."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .ellipsoid_optimizer import ellipsoid_geometry, normalized_candidate_norms
from .m2_addon_reporting import METHODS, NEW_METHODS, load_runs, report_context
from .reporting import atomic_json, read_json, write_csv
from .validation import _replay, _seed, compare_snapshots, recorded_snapshot

VERSION = "m2-addon-frozen-history-v1"
CHECK_COUNTS = (0, 32, 64, 128)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle_bounds(mu, cov, eig, reference_eig, actual_q, grid_q, *, beta=4., alpha=.5,
                  radius=4+2*np.sqrt(3)):
    """Bounds for one frozen numerical posterior, without additional optimization.

    A numerator outside E may exceed the supremum over E, so its ratio is never
    clipped to one. Numerical failure remains unresolved; no denominator floor.
    """
    mu = np.asarray(mu, dtype=float); cov = np.asarray(cov, dtype=float)
    actual_q = np.asarray(actual_q, dtype=float); grid_q = np.asarray(grid_q, dtype=float)
    if mu.shape != (4,) or cov.shape != (4,4) or actual_q.shape != (2,4) or grid_q.shape != (7,2,4):
        raise ValueError("M=2/d=4 bounds require mu[4], cov[4,4], Q[2,4], seven grid Q[7,2,4]")
    if not 0 < radius <= 32:
        raise ValueError("Ellipsoid must fit inside the inherited radius-32 feasible domain")
    _, inverse, _ = ellipsoid_geometry(mu, cov, alpha=alpha, radius=radius)
    actual_norm = normalized_candidate_norms(actual_q, mu, inverse, radius)
    grid_norm = normalized_candidate_norms(grid_q, mu, inverse, radius)
    inside = bool(np.max(actual_norm) <= 1 + 1e-10)
    grid_inside = bool(np.max(grid_norm) <= 1 + 1e-10)
    numerator = float(eig); reference = float(reference_eig)
    upper = float(min(np.log(2), 2*beta**2*radius**2*np.trace(cov)))
    lower = max(reference, numerator) if inside else reference
    # A stored grid maximum cannot be used as a feasible witness if even one
    # saved template is outside; the argmax's identity may not be available.
    finite = all(np.isfinite(v) for v in (numerator, reference, lower, upper))
    resolved = bool(grid_inside and finite and numerator >= 0 and reference > 0
                    and upper > 0 and lower <= upper + 1e-10)
    reason = ("resolved" if resolved else "grid_outside_ellipsoid" if not grid_inside
              else "nonfinite_or_nonpositive_bound" if not finite or reference <= 0 or upper <= 0
              else "lower_exceeds_analytic_upper")
    ratio_lower = numerator / upper if resolved else None
    ratio_upper = numerator / lower if resolved else None
    if resolved and inside:
        ratio_upper = min(1., ratio_upper)
    return dict(eig_actual=numerator if np.isfinite(numerator) else None,
                eig_grid=reference if np.isfinite(reference) else None,
                eig_ellipsoid_lower=lower if grid_inside and np.isfinite(lower) else None,
                eig_ellipsoid_upper=upper if np.isfinite(upper) else None,
                eta_ellipsoid_lower=ratio_lower, eta_ellipsoid_upper=ratio_upper,
                oracle_gap=(upper-lower)/upper if resolved else None,
                actual_inside_ellipsoid=inside, grid_inside_ellipsoid=grid_inside,
                actual_max_normalized_radius=float(actual_norm.max()),
                grid_max_normalized_radius=float(grid_norm.max()),
                alpha_E=float(alpha), rho_E=float(radius), resolved=resolved,
                solver_status="no_optimization_grid_and_feasible_actual_witnesses",
                resolution_status=reason, global_optimum_certified=False,
                posterior_numerical_target="saved_preanswer_posterior_approximation")


def write_oracle_sidecar(run, run_dir, alpha=.5, radius=4+2*np.sqrt(3)):
    data = run["data"]
    output = Path(run_dir) / "oracle_sidecars" / f"{run['path'].stem}.json"
    required = {"mu", "cov", "eig", "reference_eig", "q", "reference_queries"}
    if not required.issubset(data):
        result = dict(method=run["method"], interaction_id=run["interaction_id"],
                      status="unresolved", missing_fields=sorted(required-set(data)), rounds=[])
        atomic_json(output, result); return result
    rows = []
    for t in range(len(data["y"])):
        rows.append(dict(t=t, **oracle_bounds(data["mu"][t], data["cov"][t], data["eig"][t],
                    data["reference_eig"][t], data["q"][t], data["reference_queries"][t],
                    alpha=alpha, radius=radius)))
    result = dict(method=run["method"], interaction_id=run["interaction_id"],
                  status="resolved" if all(r["resolved"] for r in rows) else "unresolved",
                  source=str(run["path"]), source_sha256=_sha256(run["path"]),
                  validation_version=VERSION, rounds=rows,
                  interpretation="Feasible witnesses and analytic upper for the saved numerical law; no global optimum is claimed.",
                  actual_history_modified=False)
    atomic_json(output, result)
    return result


def _history_kind(run):
    """Infer only the mathematical factorization allowed by displayed queries."""
    q = run["data"]["q"]
    coordinate_geometry = bool(np.all(np.sum(np.ptp(q, axis=1) > 1e-12, axis=1) <= 1))
    engine = str(run["meta"].get("numerical_engine", ""))
    declared = str(run["meta"].get("posterior_kind", run["meta"].get("engine", "")))
    expected_marginal = run["method"] in ("RC", "PQ", "QI", *NEW_METHODS[:6])
    use_marginal = "marginal" in declared or "positive_continuous" in engine or "sinh_simpson" in engine or expected_marginal
    if use_marginal and not coordinate_geometry:
        raise ValueError("Saved displayed candidates do not admit coordinate-marginal inference")
    return "marginal" if use_marginal else "joint"


def artifact_checks(run, context):
    data = run["data"]; n = len(data["y"]); errors = []
    expected_shapes = dict(q=(n,2,4), y=(n,), mu=(n+1,4), cov=(n+1,4,4),
                           risk=(n+1,), error=(n+1,), eta=(n,), eig=(n,), reference_eig=(n,),
                           reference_queries=(n,7,2,4))
    for field, shape in expected_shapes.items():
        if field not in data or data[field].shape != shape:
            errors.append(f"{field}: expected {shape}, got {data.get(field, np.array([])).shape}")
    if errors:
        return dict(method=run["method"], interaction_id=run["interaction_id"], status="failed", errors=errors)
    if n != 128:
        errors.append(f"Expected 128 actual answers; observed {n}")
    for field in ("q", "mu", "cov", "risk", "error", "eig", "reference_eig", "reference_queries"):
        if not np.isfinite(data[field]).all():
            errors.append(f"Nonfinite values in {field}")
    if not np.isin(data["y"], [0,1]).all():
        errors.append("Actual answers not binary")
    if not np.allclose(data["risk"], np.trace(data["cov"], axis1=1, axis2=2), rtol=1e-12, atol=1e-12):
        errors.append("Risk differs from full covariance trace")
    if "theta" not in data or data["theta"].shape != (4,):
        errors.append("Missing four-coordinate theta_star")
    else:
        if not np.allclose(data["error"], np.sum((data["mu"]-data["theta"])**2, axis=1), rtol=1e-12, atol=1e-12):
            errors.append("Actual squared error differs from posterior-mean error")
        source = context["source_run"] / "statistical" / f"m02_b04__RC__i{run['interaction_id']}.npz"
        if source.exists():
            with np.load(source, allow_pickle=False) as archive:
                if not np.array_equal(data["theta"], archive["theta"]):
                    errors.append("theta_star differs from the frozen interaction")
        else:
            errors.append("Frozen RC theta mapping unavailable")
    finite_ratio = np.isfinite(data["eta"])
    positive = data["reference_eig"] > 0
    checked = finite_ratio & positive
    if not np.allclose(data["eta"][checked], data["eig"][checked]/data["reference_eig"][checked], rtol=1e-11, atol=1e-12):
        errors.append("eta_grid does not equal its saved integrated EIG ratio")
    if np.any(finite_ratio & ~positive):
        errors.append("Resolved ratio has a nonpositive denominator")
    if run["method"] == "RC" and np.any(data["eta"][finite_ratio] > 1+1e-10):
        errors.append("RC eta_grid exceeds its own grid maximum")
    if "reference_eigs" in data and not np.allclose(data["reference_eig"], data["reference_eigs"].max(axis=1), rtol=1e-11, atol=1e-12):
        errors.append("Grid denominator differs from saved grid EIG maximum")
    try:
        kind = _history_kind(run)
    except ValueError as exc:
        errors.append(str(exc)); kind = "invalid_geometry"
    max_radius = float(np.linalg.norm(data["q"]-data["mu"][:-1,None,:], axis=2).max()) if n else None
    if max_radius is not None and max_radius > 32+1e-9:
        errors.append("Displayed query exceeds inherited radius-32 feasible domain")
    initial_analytic = bool(np.allclose(data["mu"][0], np.zeros(4), rtol=0, atol=1e-13)
                            and np.allclose(data["cov"][0], np.eye(4)/3, rtol=0, atol=1e-13))
    if run["meta"].get("initial_moments") == "analytic_uniform_prior" and not initial_analytic:
        errors.append("Declared analytic initial moments do not match the uniform prior")
    if run["method"] in ("PS","GC","sm_continuous","gausscloud_paper_L1","gausscloud_paper_L3","gausscloud_paper_L5"):
        diagnostic_times = np.array([t for t in (0,31,63,95,127) if t<n])
        if not np.array_equal(data.get("diag_t",np.array([])),diagnostic_times):
            errors.append("Missing or unexpected precommitted randomized-diagnostic rounds")
        if data.get("diag_eig",np.array([])).shape != (len(diagnostic_times),64):
            errors.append("Randomized diagnostics require exactly 64 EIG draws per checkpoint")
    if run.get("disposition") == "new":
        frozen = next((r for r in context["lock"].get("interactions",[]) if r["interaction_id"]==run["interaction_id"]),None)
        if frozen is not None and run["meta"].get("context_id") != frozen.get("context_id"):
            errors.append("New method context ID differs from the frozen interaction")
    if run["method"].startswith("gausscloud_paper_L"):
        stages = int(run["method"].split("_L")[-1]); lengths = [128//stages]*stages
        lengths[-1] += 128-sum(lengths); starts = np.cumsum([0]+lengths[:-1])
        expected_stage = np.searchsorted(starts, np.arange(n), side="right")-1
        for field in ("selector_stage", "selector_center", "selector_std", "q_raw"):
            if field not in data:
                errors.append(f"Missing schedule audit field {field}")
        if all(field in data for field in ("selector_stage", "selector_center", "selector_std", "q_raw")):
            if not np.array_equal(data["selector_stage"], expected_stage):
                errors.append("GCQ stage allocation differs from pinned author-code schedule")
            if not np.allclose(data["selector_std"], 1/np.sqrt(3)*2.**(-expected_stage), rtol=1e-13, atol=0):
                errors.append("GCQ standard deviation differs from frozen schedule")
            for j, start in enumerate(starts):
                if start < n:
                    stop = min(n, start+lengths[j])
                    if not np.allclose(data["selector_center"][start:stop], data["mu"][start], rtol=0, atol=1e-13):
                        errors.append(f"GCQ center not fixed at stage-entry mean: stage {j}")
    return dict(method=run["method"], interaction_id=run["interaction_id"],
                status="passed" if not errors else "failed", errors=errors, answers_completed=n,
                posterior_kind=kind, eta_grid_unresolved_rounds=int(np.sum(~finite_ratio)),
                eta_grid_above_one_rounds=int(np.sum(data["eta"][finite_ratio] > 1)),
                max_displayed_radius=max_radius, analytic_prior_trace=4/3,
                observed_initial_trace=float(data["risk"][0]), initial_moments_exact_analytic=initial_analytic,
                initial_numerical_covariance_error=float(np.linalg.norm(data["cov"][0]-np.eye(4)/3)))


def validate_history(run, run_dir, device="cpu"):
    start = time.monotonic(); data = run["data"]
    identity = run["setting_id"], run["method"], run["interaction_id"]
    kind = _history_kind(run)
    output = Path(run_dir) / "validation" / f"{run['path'].stem}.numerical.json"
    source_hash = _sha256(run["path"])
    own_hash = hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in
                             ("m2_addon_validation.py", "validation.py", "posterior_marginal.py", "posterior_smc.py"))).hexdigest()
    previous = read_json(output)
    if (previous.get("source_sha256") == source_hash and previous.get("validation_code_hash") == own_hash
            and previous.get("validation_version") == VERSION):
        return previous
    checks = [t for t in CHECK_COUNTS if t <= len(data["y"])]
    if "reference_queries" not in data:
        result = dict(identity=identity, status="unresolved", reason="Frozen reference queries absent", comparisons=[])
        atomic_json(output,result); return result
    repeats = 1 if kind == "marginal" else 2
    references = [_replay(data, 4., kind, 513, 8192, _seed(*identity, 8192, r), device, checks) for r in range(repeats)]
    # New GCQ queries use the specified analytic initial moments. Independent
    # numerical particle EIG is still compared; no fake exact-posterior claim.
    if run["meta"].get("initial_moments") == "analytic_uniform_prior" and 0 in checks:
        for reference in references:
            reference[0].update(mu=np.zeros(4), cov=np.eye(4)/3, risk=4/3)
    rows = [dict(answer_count=t, **row) for t in checks
            for row in compare_snapshots(recorded_snapshot(data,t), [r[t] for r in references])]
    unresolved = any(r["decision"] == "unresolved" for r in rows)
    refinement_rows = []
    if unresolved:
        refinement = _replay(data, 4., kind, 1025, 16384, _seed(*identity,16384,0), device, checks)
        if run["meta"].get("initial_moments") == "analytic_uniform_prior" and 0 in checks:
            refinement[0].update(mu=np.zeros(4),cov=np.eye(4)/3,risk=4/3)
        for t in checks:
            mean = {k: np.mean([r[t][k] for r in references],axis=0) for k in references[0][t]}
            refinement_rows.extend(dict(answer_count=t, **row) for row in compare_snapshots(mean,[refinement[t]]))
        if kind == "marginal":
            rows = [dict(answer_count=t, **row) for t in checks for row in compare_snapshots(recorded_snapshot(data,t),[refinement[t]])]
            unresolved = any(r["decision"] == "unresolved" for r in rows+refinement_rows)
    result = dict(identity=identity, status="unresolved" if unresolved else "within_reference_uncertainty",
                  source=str(run["path"]), source_sha256=source_hash, validation_version=VERSION,
                  validation_code_hash=own_hash, posterior_kind=kind, checkpoints=checks,
                  reference_nodes=513 if kind == "marginal" else None,
                  reference_particles=8192 if kind == "joint" else None, independent_replays=repeats,
                  comparisons=rows, reference_refinement_comparisons=refinement_rows,
                  refinement_attempted=bool(refinement_rows), actual_history_modified=False,
                  exact_eig_formula_used=True, exact_posterior_computed=False,
                  numerical_uncertainty="Deterministic quadrature refinement" if kind == "marginal" else
                      "Two independent 8192-particle references; 3 descriptive SE allowance is not a confidence guarantee or estimate of original particle uncertainty.",
                  elapsed_seconds=time.monotonic()-start)
    atomic_json(output,result)
    return result


def cpu_gpu_fixed_history_check(run, device="cpu"):
    """Compare arithmetic on one immutable bank; this is not a posterior reference."""
    if not str(device).startswith("cuda"):
        return dict(status="not_run_cpu_only", reason="No CUDA computation requested")
    from scipy.special import logsumexp
    from .posterior_smc import SMCPosterior
    from .ellipsoid_optimizer import eig_value_gradient
    from .metrics import integrated_eig

    data = run["data"]
    posterior = SMCPosterior(4,4.,4096,np.random.default_rng(_seed(run["path"].stem,"fixed_bank")),"cpu")
    points = posterior.particles.copy()
    comparisons = []
    for t in CHECK_COUNTS:
        if t > len(data["y"]):
            continue
        posterior.history = [(q,int(y)) for q,y in zip(data["q"][:t],data["y"][:t])]
        posterior._refresh_coefficients()
        posterior.device = "cpu"; cpu = posterior.target_logdensity(points)
        posterior.device = str(device); gpu = posterior.target_logdensity(points)
        weights = np.exp(cpu-logsumexp(cpu))
        query = data["q"][min(t,len(data["q"])-1)]
        cpu_eig = float(integrated_eig(points,weights,query,4.))
        gpu_eig = float(eig_value_gradient(points,weights,query,4.,device=device,gradient=False)[0])
        target_match = bool(np.allclose(cpu,gpu,rtol=1e-12,atol=1e-9))
        eig_match = bool(np.isclose(cpu_eig,gpu_eig,rtol=1e-9,atol=1e-12))
        comparisons.append(dict(answer_count=t, target_max_abs_difference=float(np.max(np.abs(cpu-gpu))),
                                eig_cpu=cpu_eig,eig_gpu=gpu_eig,target_match=target_match,eig_match=eig_match))
    return dict(status="passed" if all(r["target_match"] and r["eig_match"] for r in comparisons) else "unresolved",
                source=str(run["path"]), device=str(device), comparisons=comparisons,
                interpretation="Arithmetic parity on a fixed prior bank weighted by each saved full history; not an independent posterior accuracy reference or plotted EIG.",
                actual_history_modified=False)


def validate_devices(run_dir, devices=("cuda:0", "cuda:1", "cuda:2"), config=None):
    """Admission-time arithmetic checks; caller controls physical-device isolation."""
    import os
    context = report_context(run_dir, config)
    first = sorted(context["interaction_ids"])[0]
    run = next(r for r in load_runs(run_dir, config) if r["method"] == "GC" and r["interaction_id"] == first)
    checks = []
    for device in devices:
        try:
            checks.append(cpu_gpu_fixed_history_check(run, device))
        except Exception as exc:
            checks.append(dict(status="unresolved", device=str(device), error=f"{type(exc).__name__}: {exc}"))
    result = dict(status="passed" if checks and all(c["status"]=="passed" for c in checks) else "unresolved",
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=list(devices),
                  checks=checks, actual_histories_modified=False, validation_version=VERSION)
    atomic_json(Path(run_dir)/"validation/device_equivalence.json",result)
    return result


def validate_run(run_dir, config=None, device="cpu", numerical=True):
    context = report_context(run_dir,config); root = context["root"]
    runs = load_runs(root,config); start = time.monotonic()
    checks, sidecars, references, errors = [], [], [], []
    alpha = context["oracle"].get("alpha_E", .5)
    radius = context["oracle"].get("rho_E", 4+2*np.sqrt(3))
    expected = len(METHODS)*len(context["interaction_ids"])
    for run in runs:
        try:
            check = artifact_checks(run,context); checks.append(check)
            if check["status"] == "passed":
                sidecars.append(write_oracle_sidecar(run,root,alpha,radius))
        except Exception as exc:
            errors.append(dict(run_key=run["path"].stem, phase="artifact", error=f"{type(exc).__name__}: {exc}"))
    first = sorted(context["interaction_ids"])[0]
    selected = [r for r in runs if r["interaction_id"] == first]
    if numerical:
        for run in selected:
            print(f"M2 NUMERICAL VALIDATION {run['path'].stem}",flush=True)
            try:
                references.append(validate_history(run,root,device))
            except Exception as exc:
                errors.append(dict(run_key=run["path"].stem, phase="numerical", error=f"{type(exc).__name__}: {exc}"))
            atomic_json(root/"validation/progress.json",dict(completed_histories=len(references), expected_histories=len(METHODS),
                          errors=errors, elapsed_seconds=time.monotonic()-start))
    parity = dict(status="not_run")
    if numerical and selected:
        try:
            parity = cpu_gpu_fixed_history_check(next((r for r in selected if r["method"]=="GC"), selected[0]),device)
        except Exception as exc:
            parity = dict(status="unresolved",error=f"{type(exc).__name__}: {exc}")
            errors.append(dict(phase="cpu_gpu_arithmetic",error=parity["error"]))
        atomic_json(root/"validation/cpu_gpu_fixed_history.json",parity)
    discrepancy_rows = []
    for reference in references:
        for phase, rows in (("main_vs_reference",reference.get("comparisons",[])),
                            ("reference_vs_refinement",reference.get("reference_refinement_comparisons",[]))):
            discrepancy_rows.extend(dict(method=reference["identity"][1],interaction_id=first,comparison=phase,
                              **{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}) for row in rows)
    write_csv(root/"tables/integration_accuracy.csv",discrepancy_rows)
    atomic_json(root/"validation/artifact_checks.json",checks)
    artifact_status = "passed" if len(checks)==expected and all(c["status"]=="passed" for c in checks) else "incomplete_or_failed"
    numerical_status = "not_run" if not numerical else "within_reference_uncertainty" if (
        len(references)==len(METHODS) and all(r["status"]=="within_reference_uncertainty" for r in references)) else "unresolved"
    bounds_status = "resolved" if len(sidecars)==expected and all(s["status"]=="resolved" for s in sidecars) else "unresolved"
    status = "passed" if artifact_status=="passed" and numerical_status=="within_reference_uncertainty" and bounds_status=="resolved" and parity["status"] != "unresolved" and not errors else "unresolved"
    result = dict(status=status, artifact_status=artifact_status, numerical_status=numerical_status,
                  ellipsoid_bounds_status=bounds_status, expected_runs=expected, observed_runs=len(runs),
                  source_histories_checked=len(checks), validation_version=VERSION,
                  representative_selection_rule="first interaction in sorted frozen IDs, all fourteen configurations; fixed before outcome inspection",
                  representative_interaction_id=first, expected_reference_histories=len(METHODS),
                  completed_reference_histories=len(references), checkpoints=list(CHECK_COUNTS),
                  unresolved_histories=[r["identity"] for r in references if r["status"]=="unresolved"],
                  cpu_gpu_arithmetic_status=parity["status"],
                  relative_variance_eig_tolerance=.005, absolute_eta_grid_tolerance=.005,
                  errors=errors, actual_histories_modified=False, global_optimum_certified=False,
                  elapsed_seconds=time.monotonic()-start)
    atomic_json(root/"validation/summary.json",result)
    return result
