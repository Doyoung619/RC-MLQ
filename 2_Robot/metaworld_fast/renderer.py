"""Four-coordinate command adapter and real (non-rendered) motion audit.

This module's historical interface name is renderer, but rendering is opt-in
and never occurs in execute/rollout. No torch/CUDA imports belong here.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import time
import warnings
from pathlib import Path
import numpy as np
from .contexts import make_env, restore_state, source_hash, hash_value, atomic_json


def motion_basis(H=50):
    u = np.arange(H, dtype=np.float64)/(H-1)
    basis = np.zeros((H,3,4), dtype=np.float64)
    f, g = np.sin(np.pi*u), np.sin(2*np.pi*u)
    basis[:,0,0] = f; basis[:,1,1] = f; basis[:,2,2] = f; basis[:,0,3] = g
    return basis


def adapter_parameters(H=50):
    basis = motion_basis(H); flat = basis.reshape(-1,4)
    row_l1 = float(np.abs(basis).sum(axis=2).max())
    return {"version": "four-cartesian-temporal-bases-v1", "H": H,
            "basis": ["sin_pi_x","sin_pi_y","sin_pi_z","sin_2pi_x"],
            "basis_hash": hash_value(basis), "basis_rank": int(np.linalg.matrix_rank(flat)),
            "basis_condition": float(np.linalg.cond(flat)), "max_basis_row_l1": row_l1,
            "alpha": .35/(33*row_l1), "query_coordinate_bound": 33.,
            "baseline_scale": .6, "fixed_gripper": 0., "submitted_action_dtype": "float32",
            "command_reconstruction_tolerance": 1e-4,
            "position_units": "meters", "position_sensor": "official get_endeff_pos hand body center"}


def cache_key(q, context, adapter_hash=None):
    q = np.asarray(q)
    if q.shape != (4,): raise ValueError("Expected one four-dimensional candidate")
    return hash_value({"exact_q": q, "q_dtype": q.dtype.str,
                       "context": context["context_hash"], "H": context["H"],
                       "adapter": adapter_parameters(context["H"]),
                       "nominal_actions": context["nominal_actions"],
                       "simulator": context["versions"],
                       "adapter_code_hash": adapter_hash or source_hash()})


class RobotAdapter:
    def __init__(self, contexts, render_mode=None):
        if isinstance(contexts, (str, Path)):
            from .contexts import load_contexts
            contexts = load_contexts(contexts)
        self.contexts = {int(c["context_id"]): c for c in contexts}
        self.H = next(iter(self.contexts.values()))["H"]
        self.basis = motion_basis(self.H)
        self.pinv = np.linalg.pinv(self.basis.reshape(-1,4))
        self.params = adapter_parameters(self.H); self.alpha = self.params["alpha"]
        self.code_hash = source_hash()
        self.env = make_env(next(iter(self.contexts.values())), render_mode)
        self.render_mode = render_mode
        if render_mode is not None:
            # Qualitative-only free camera: installed corner2 XML is upside down.
            # These viewer settings never modify the physical MuJoCo model.
            self.env.mujoco_renderer.camera_id = -1
            self.env.mujoco_renderer.default_cam_config = {
                "distance": 1.4, "lookat": np.array([0., .65, .12]),
                "azimuth": 135., "elevation": -25.}
        self._key_prefix = {i: hash_value({"context":c["context_hash"],"adapter":self.params,"code":self.code_hash,
                             "nominal_actions":c["nominal_actions"],"versions":c["versions"]}) for i,c in self.contexts.items()}

    def key(self, q, context_id):
        # Full fixed prefix covers context/model/H/basis/gain/nominal/version/code.
        return hash_value({"prefix": self._key_prefix[int(context_id)], "q":np.asarray(q)})

    def commands(self, q, context_id):
        q = np.asarray(q)
        if q.shape != (4,) or not np.isfinite(q).all(): raise ValueError("q must be a finite 4-vector")
        if np.max(np.abs(q)) > 33 + 1e-12: raise ValueError(f"q outside fixed conservative envelope: {q}")
        context = self.contexts[int(context_id)]
        base = .6 * context["nominal_actions"][:,:3].astype(np.float64)
        commands = np.empty((self.H,4), dtype=np.float32)
        commands[:,:3] = base + self.alpha * np.einsum("hak,k->ha", self.basis, q)
        commands[:,3] = 0.
        if np.any(commands < context["action_low"]) or np.any(commands > context["action_high"]):
            raise RuntimeError("Adapter action bounds violated; silent clipping is forbidden")
        q_hat = self.pinv @ ((commands[:,:3].astype(np.float64)-base)/self.alpha).ravel()
        return commands, q_hat

    def rollout(self, q, context_id, collect_frames=False):
        started = time.perf_counter()
        context_id = int(context_id); context = self.contexts[context_id]
        q = np.asarray(q).copy(); key = self.key(q, context_id)
        positions = np.full((self.H+1,3), np.nan, dtype=np.float64)
        submitted = np.full((self.H,4), np.nan, dtype=np.float32)
        gripper = np.full(self.H+1, np.nan, dtype=np.float64)
        mocap = np.full((self.H+1,3), np.nan, dtype=np.float64)
        frames = []
        diag = {"context_id":context_id, "context_hash":context["context_hash"],
                "query_hash":hash_value(q), "adapter_hash":self.code_hash,
                "simulator_versions":context["versions"], "cache_key":key,
                "status":"failed", "simulation_valid":False, "action_bounds_ok":False,
                "adapter_clipping":False, "internal_action_clipping":False,
                "actual_steps":0, "terminated":False, "truncated":False,
                "exception":None, "warnings":[], "task_success":False,
                "workspace_saturation_steps":0, "workspace_saturation_components":0,
                "workspace_saturation_rate":0., "final_goal_distance":None,
                "command_reconstruction_error":None, "renderer_created":False}
        q_hat = np.full(4,np.nan)
        try:
            commands, q_hat = self.commands(q,context_id)
            error = float(np.linalg.norm(q_hat-q))
            diag["command_reconstruction_error"] = error
            diag["action_bounds_ok"] = True
            diag["command_reconstruction_ok"] = bool(error <= self.params["command_reconstruction_tolerance"])
            restore_start = time.perf_counter(); restore_state(self.env, context["state"])
            diag["context_restore_seconds"] = time.perf_counter()-restore_start
            positions[0] = self.env.get_endeff_pos().copy()
            mocap[0] = self.env.data.mocap_pos[0].copy()
            gripper[0] = np.linalg.norm(self.env.data.body("rightclaw").xpos-self.env.data.body("leftclaw").xpos)
            before_warning = np.asarray(self.env.data.warning.number).copy()
            if collect_frames: frames.append(self.env.render())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for h, action in enumerate(commands):
                    desired = self.env.data.mocap_pos[0] + action[:3]*self.env.action_scale
                    saturated = (desired < self.env.mocap_low) | (desired > self.env.mocap_high)
                    diag["workspace_saturation_steps"] += int(np.any(saturated))
                    diag["workspace_saturation_components"] += int(np.sum(saturated))
                    obs, _, terminated, truncated, info = self.env.step(action)
                    # Only commands actually submitted are stored as execution.
                    submitted[h] = action.copy(); diag["actual_steps"] = h+1
                    positions[h+1] = self.env.get_endeff_pos().copy()
                    mocap[h+1] = self.env.data.mocap_pos[0].copy()
                    gripper[h+1] = np.linalg.norm(self.env.data.body("rightclaw").xpos-self.env.data.body("leftclaw").xpos)
                    diag["task_success"] = bool(diag["task_success"] or info.get("success",False))
                    diag["terminated"],diag["truncated"] = bool(terminated),bool(truncated)
                    if collect_frames: frames.append(self.env.render())
                    if not np.isfinite(obs).all() or not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
                        raise RuntimeError("Non-finite simulator state")
                    if getattr(self.env,"_did_see_sim_exception",False): raise RuntimeError("MetaWorld simulator exception flag")
                    if terminated or truncated: break
                diag["warnings"] = [str(w.message) for w in caught]
            warning_delta = np.asarray(self.env.data.warning.number)-before_warning
            diag["mujoco_warning_counts"] = warning_delta.tolist()
            actual_steps = diag["actual_steps"]
            base = .6 * context["nominal_actions"][:actual_steps,:3].astype(np.float64)
            actual_basis = self.basis[:actual_steps].reshape(-1,4)
            q_hat = np.linalg.pinv(actual_basis) @ ((submitted[:actual_steps,:3].astype(np.float64)-base)/self.alpha).ravel()
            diag["command_reconstruction_error"] = float(np.linalg.norm(q_hat-q))
            diag["submitted_command_basis_rank"] = int(np.linalg.matrix_rank(actual_basis))
            diag["command_reconstruction_ok"] = bool(diag["command_reconstruction_error"] <= self.params["command_reconstruction_tolerance"] and diag["submitted_command_basis_rank"]==4)
            diag["workspace_saturation_rate"] = diag["workspace_saturation_steps"]/max(1,diag["actual_steps"])
            diag["final_goal_distance"] = float(np.linalg.norm(self.env.tcp_center-context["goal"]))
            diag["unexpected_short_clip"] = diag["actual_steps"] != self.H
            diag["simulation_valid"] = bool(diag["actual_steps"] == self.H and not np.any(warning_delta > 0) and not diag["warnings"])
            diag["status"] = "executed" if diag["simulation_valid"] else "failed"
            mjrenderer = getattr(self.env,"mujoco_renderer",None)
            diag["renderer_created"] = bool(getattr(mjrenderer,"_viewers",{}))
        except Exception as exc:
            diag["exception"] = f"{type(exc).__name__}: {exc}"
        diag["elapsed_seconds"] = time.perf_counter()-started
        result = {"q":q,"q_hat":q_hat,"positions":positions,"submitted_actions":submitted,
                  "gripper_distance":gripper,"mocap_positions":mocap,"diagnostics":diag}
        if collect_frames: result["frames"] = frames
        return result

    def close(self): self.env.close()


def pairwise_motion(q, positions, steps=None):
    q = np.asarray(q); positions = np.asarray(positions)
    pairs, q_distance, motion = [], [], []
    for i,j in itertools.combinations(range(len(q)),2):
        pairs.append([i,j]); q_distance.append(float(np.linalg.norm(q[i]-q[j])))
        count = min(int(steps[i]),int(steps[j])) if steps is not None else positions.shape[1]-1
        # Exclude the identical restored initial state from the 50-action RMS.
        delta = positions[i,1:count+1]-positions[j,1:count+1]
        motion.append(float(np.sqrt(np.mean(np.sum(delta*delta,axis=-1)))) if count and np.isfinite(delta).all() else None)
    return {"pair_slots":pairs,"coordinate_pair_distance":q_distance,"motion_pairwise_rms":motion,
            "rms_definition":"sqrt(mean_h ||hand_i(h)-hand_j(h)||_2^2), h=1..executed actions"}


def calibrate(contexts, output_path=None):
    """Frozen engineering points, independent of all preference outcomes."""
    adapter = RobotAdapter(contexts)
    records, sensitivity = [], []
    fixed = [np.zeros(4)] + [s*33*np.eye(4)[k] for k in range(4) for s in (-1,1)]
    fixed += [33*np.asarray(signs) for signs in itertools.product((-1,1),repeat=4)]
    try:
        for context_id in adapter.contexts:
            for q in fixed:
                result = adapter.rollout(q,context_id)
                records.append({"q":q.tolist(),**result["diagnostics"]})
            for center in (np.zeros(4),np.array([1.,-1.,.5,-.5])):
                for delta in (.01,.1,1.):
                    jac = []
                    for axis in range(4):
                        plus = adapter.rollout(center+delta*np.eye(4)[axis],context_id)
                        minus = adapter.rollout(center-delta*np.eye(4)[axis],context_id)
                        jac.append(((plus["positions"][1:]-minus["positions"][1:])/(2*delta)).ravel())
                    matrix = np.stack(jac,axis=1)
                    s = np.linalg.svd(matrix,compute_uv=False)
                    # Diagnostic rank threshold has physical units; save values
                    # so a human can distinguish tiny motion from full rank.
                    threshold = max(1e-7,float(s[0])*1e-6)
                    sensitivity.append({"context_id":context_id,"center":center.tolist(),"delta":delta,
                                        "jacobian_singular_values":s.tolist(),"rank_threshold":threshold,
                                        "numerical_rank":int(np.sum(s>threshold)),
                                        "coordinate_rms_sensitivity_meters_per_q":(np.linalg.norm(matrix,axis=0)/np.sqrt(adapter.H)).tolist()})
        checks = []
        for context_id in adapter.contexts:
            qa=np.array([.2,-.3,.4,-.5]); qb=-qa
            a1=adapter.rollout(qa,context_id); adapter.rollout(qb,context_id); a2=adapter.rollout(qa,context_id)
            difference=float(np.max(np.abs(a1["positions"]-a2["positions"])))
            checks.append({"context_id":context_id,"max_position_restore_order_error":difference,
                           "bitwise_equal":bool(np.array_equal(a1["positions"],a2["positions"]))})
        valid = all(r["simulation_valid"] and r["command_reconstruction_ok"] for r in records)
        rank_ok = all(r["numerical_rank"]==4 for r in sensitivity)
        result={"adapter":adapter.params,"fixed_points_per_context":len(fixed),"fixed_command_audit":records,
                "motion_sensitivity":sensitivity,"restore_order_checks":checks,
                "physical_interpretation":"finite_audit_passed" if valid and rank_ok and all(c["bitwise_equal"] for c in checks) else "unresolved",
                "limitations":"Full-rank command and finite-difference motion do not establish human perceptual identifiability, universal feasibility, collision-free motion, or safety."}
        if output_path: atomic_json(output_path,result)
        return result
    finally: adapter.close()


def render_video(contexts, q, context_id, output_path, fps=50):
    """Replay only the precommitted video subset after quantitative execution."""
    import imageio.v2 as imageio
    adapter=RobotAdapter(contexts,render_mode="rgb_array")
    try:
        result=adapter.rollout(q,context_id,collect_frames=True)
        if not result["diagnostics"]["simulation_valid"]: raise RuntimeError(result["diagnostics"])
        output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(output_path,result["frames"],fps=fps)
        atomic_json(output_path.with_suffix(".json"),result["diagnostics"])
        return result["diagnostics"]
    finally: adapter.close()


def _render_qualitative_videos_inprocess(run_dir):
    import imageio.v2 as imageio
    from .contexts import load_contexts
    root = Path(run_dir); output = root / "videos"; output.mkdir(parents=True, exist_ok=True)
    adapter = RobotAdapter(load_contexts(root/"contexts"), render_mode="rgb_array")
    records = []
    try:
        for method in ("RC", "PQ", "SG"):
            path = root/"statistical"/f"m04_b04__{method}__i0.npz"
            if not path.exists():
                records.append({"method":method,"status":"not_requested","reason":"statistical history unavailable"});continue
            with np.load(path,allow_pickle=False) as data:q=data["q"].copy()
            for t in (0,63,127):
                if len(q)<=t:continue
                for slot,candidate in enumerate(q[t]):
                    video=output/f"m04_b04__{method}__i0__t{t:03d}__c{slot}.mp4"
                    record={"method":method,"setting_id":"m04_b04","interaction_id":0,"t":t,"candidate_slot":slot,"path":str(video)}
                    try:
                        result=adapter.rollout(candidate,0,collect_frames=True)
                        if not result["diagnostics"]["simulation_valid"]:raise RuntimeError(result["diagnostics"])
                        with imageio.get_writer(video,fps=50,codec="libx264",quality=7,ffmpeg_params=["-threads","1"]) as writer:
                            for frame in result["frames"]:writer.append_data(frame)
                        atomic_json(video.with_suffix(".json"),result["diagnostics"])
                        record.update(status="executed",frames=len(result["frames"]))
                    except Exception as exc:record.update(status="failed",exception=f"{type(exc).__name__}: {exc}")
                    records.append(record)
                    atomic_json(output/"manifest.json",{"precommitted_subset":"m04_b04 interaction0 RC/PQ/SG t0,63,127 all4candidates", "clips":records})
        return {"planned_clips":36,"executed_clips":sum(r["status"]=="executed" for r in records),"failed_clips":sum(r["status"]=="failed" for r in records),"manifest":str(output/"manifest.json")}
    finally:adapter.close()


def render_qualitative_videos(run_dir):
    """A fresh rendering subprocess selects EGL before importing MuJoCo."""
    import subprocess,sys,os
    root=Path(run_dir).resolve();(root/"logs").mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl")
    log=root/"logs"/"videos.log"
    with log.open("a") as stream:
        result=subprocess.run([sys.executable,"-m","metaworld_fast.renderer","--videos-root",str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT)
    manifest=root/"videos"/"manifest.json"
    if result.returncode!=0:return {"status":"failed","returncode":result.returncode,"log":str(log)}
    records=json.loads(manifest.read_text())["clips"] if manifest.exists() else []
    return {"status":"completed" if sum(r["status"]=="executed" for r in records)==36 else "incomplete",
            "executed_clips":sum(r["status"]=="executed" for r in records),"manifest":str(manifest),"log":str(log)}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--videos-root",required=True);args=parser.parse_args()
    print(json.dumps(_render_qualitative_videos_inprocess(args.videos_root)),flush=True)
