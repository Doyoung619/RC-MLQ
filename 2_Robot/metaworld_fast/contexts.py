"""Pinned official reach-v3 contexts; complete deterministic simulator restore.

There are no Gym wrappers: MT1's public train class and task API are used,
so the only episode counter is SawyerXYZEnv.curr_path_length. Physics always
runs through the installed environment's step, never a low-level substitute.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import numpy as np

PINNED_METAWORLD = "3.0.0"
PINNED_MUJOCO = "3.14.0"
CONTEXT_VERSION = "reach-v3-full-integration-v1"
# Source inspection of official reset_model/step: these are the model arrays
# mutable through the task/reset API. Complete compiled model is hashed too.
MODEL_FIELDS = ("body_pos", "body_quat", "site_pos", "site_quat", "geom_pos", "geom_quat", "eq_data")


def stable_seed(root: int, *parts) -> int:
    b = json.dumps([int(root), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "little")


def _hash_update(h, value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        h.update(b"array" + a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            _hash_update(h, str(key)); _hash_update(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(type(value).__name__.encode())
        for item in value: _hash_update(h, item)
    elif isinstance(value, bytes): h.update(b"bytes" + value)
    else: h.update(repr(value).encode())


def hash_value(value):
    h = hashlib.sha256(); _hash_update(h, value); return h.hexdigest()


def source_hash():
    paths = [Path(__file__), Path(__file__).with_name("renderer.py")]
    return hash_value({p.name: p.read_bytes() for p in paths if p.exists()})


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _simple(value):
    if value is None or isinstance(value, (bool, int, float, str, bytes, np.ndarray, np.generic)): return True
    if isinstance(value, (tuple, list)): return all(_simple(v) for v in value)
    if isinstance(value, dict): return all(isinstance(k, (str, int)) and _simple(v) for k, v in value.items())
    return False


def capture_state(env):
    import mujoco
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(env.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, state_spec)
    attrs = {k: copy.deepcopy(v) for k, v in env.__dict__.items() if _simple(v)}
    generators = {k: copy.deepcopy(v.bit_generator.state) for k, v in env.__dict__.items() if isinstance(v, np.random.Generator)}
    spaces = {}
    from gymnasium.spaces import Space
    for k, v in env.__dict__.items():
        if isinstance(v, Space): spaces[k] = copy.deepcopy(v.np_random.bit_generator.state)
    return {"state_spec": state_spec, "integration": state,
            "model_fields": {k: getattr(env.model, k).copy() for k in MODEL_FIELDS},
            "attributes": attrs, "generators": generators, "space_rng": spaces,
            "numpy_global_rng": np.random.get_state()}


def restore_state(env, snapshot):
    import mujoco
    for key, val in snapshot["model_fields"].items(): getattr(env.model, key)[:] = val
    for key, val in snapshot["attributes"].items():
        if key != "render_mode": setattr(env, key, copy.deepcopy(val))
    for key, val in snapshot["generators"].items():
        generator = getattr(env, key, None)
        if not isinstance(generator, np.random.Generator):
            generator = np.random.default_rng(); setattr(env, key, generator)
        generator.bit_generator.state = copy.deepcopy(val)
    for key, val in snapshot["space_rng"].items(): getattr(env, key).np_random.bit_generator.state = copy.deepcopy(val)
    np.random.set_state(snapshot["numpy_global_rng"])
    mujoco.mj_setState(env.model, env.data, snapshot["integration"], snapshot["state_spec"])
    mujoco.mj_forward(env.model, env.data)
    # mj_forward can modify solver warm-start; restore integration exactly after
    # deriving Cartesian locations. Next official step will perform simulation.
    mujoco.mj_setState(env.model, env.data, snapshot["integration"], snapshot["state_spec"])


def compiled_model_hash(model):
    import mujoco
    with tempfile.TemporaryDirectory() as tmp:
        filename = Path(tmp) / "model.mjb"
        mujoco.mj_saveModel(model, str(filename), None)
        return hashlib.sha256(filename.read_bytes()).hexdigest()


def simulator_versions():
    return {name: importlib.metadata.version(name) for name in ("metaworld", "mujoco", "gymnasium", "numpy")}


def make_env(context=None, render_mode=None):
    from metaworld.envs.sawyer_reach_v3 import SawyerReachEnvV3
    from metaworld.types import Task
    env = SawyerReachEnvV3(render_mode=render_mode, camera_name="corner2", width=480, height=480)
    if context is not None:
        env.set_task(Task("reach-v3", context["task_data"]))
        env.seed(context["seed"])
        env.reset()
        restore_state(env, context["state"])
    return env


def prepare_contexts(output_dir, count=5, H=50, root_seed=20261004):
    """Create immutable contexts with official bounded expert commands once."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "contexts.pkl"
    if existing.exists():
        contexts = load_contexts(output_dir)
        if len(contexts) != count or any(c["H"] != H for c in contexts): raise ValueError("Existing context configuration differs")
        return contexts
    import metaworld
    from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
    versions = simulator_versions()
    if versions["metaworld"] != PINNED_METAWORLD or versions["mujoco"] != PINNED_MUJOCO:
        raise RuntimeError(f"Pinned simulator mismatch: {versions}")
    started = time.perf_counter()
    contexts = []
    # Independent context seeds, each with its own official MT1 task generator.
    for context_id in range(count):
        seed = stable_seed(root_seed, "physical_context", context_id)
        benchmark = metaworld.MT1("reach-v3", seed=seed)
        task = benchmark.train_tasks[0]
        env = benchmark.train_classes["reach-v3"](render_mode=None, camera_name="corner2")
        env.set_task(task); env.seed(seed); obs, _ = env.reset()
        snapshot = capture_state(env)
        model_hash = compiled_model_hash(env.model)
        policy = SawyerReachV3Policy()
        nominal, expert_raw, positions = [], [], [env.get_endeff_pos().copy()]
        clipped = 0
        for h in range(H):
            raw = np.asarray(policy.get_action(obs), dtype=np.float64)
            bounded = np.clip(raw, env.action_space.low, env.action_space.high).astype(np.float32)
            clipped += int(np.any(raw != bounded.astype(np.float64)))
            obs, _, terminated, truncated, _ = env.step(bounded)
            nominal.append(bounded.copy()); expert_raw.append(raw.copy()); positions.append(env.get_endeff_pos().copy())
            if terminated or truncated:
                raise RuntimeError(f"Nominal context {context_id} ended at {h+1}, expected {H}")
        context = {"context_id": context_id, "seed": seed, "H": H,
                   "task_data": task.data, "state": snapshot,
                   "nominal_actions": np.asarray(nominal, dtype=np.float32),
                   "expert_raw_actions": np.asarray(expert_raw),
                   "nominal_positions": np.asarray(positions), "nominal_bounded_or_float32_changed_steps": clipped,
                   "goal": snapshot["attributes"]["_target_pos"].copy(),
                   "versions": versions, "model_hash": model_hash,
                   "frame_skip": int(env.frame_skip), "physics_timestep": float(env.model.opt.timestep),
                   "action_low": env.action_space.low.copy(), "action_high": env.action_space.high.copy(),
                   "context_version": CONTEXT_VERSION}
        context["context_hash"] = hash_value(context)
        contexts.append(context); env.close()
    tmp = existing.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f: pickle.dump(contexts, f, protocol=5)
    os.replace(tmp, existing)
    from .renderer import motion_basis, adapter_parameters
    manifest = {"version": CONTEXT_VERSION, "context_creation_seconds": time.perf_counter()-started,
                "simulator_versions": versions, "adapter_code_hash": source_hash(),
                "adapter": adapter_parameters(H), "state_capture": "mjSTATE_INTEGRATION + simple env attributes + all RNGs + mutable model fields",
                "wrappers": [], "terminate_on_success": False,
                "model_mutation_inspection": "official reach-v3 reset_model updates goal site position; official step changes no model arrays",
                "storage": "numeric positions and submitted float32 actions; no hot-path images/full observations",
                "contexts": [{k: (v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()
                              if k in ("context_id","seed","H","goal","context_hash","model_hash","frame_skip","physics_timestep","action_low","action_high","nominal_bounded_or_float32_changed_steps")} for c in contexts]}
    atomic_json(output_dir / "manifest.json", manifest)
    return contexts


def load_contexts(directory):
    path = Path(directory)
    if path.is_dir(): path = path / "contexts.pkl"
    with path.open("rb") as f: contexts = pickle.load(f)
    installed = simulator_versions()
    for context in contexts:
        for name in ("metaworld", "mujoco", "gymnasium"):
            if installed[name] != context["versions"][name]: raise RuntimeError(f"Context simulator mismatch for {name}")
        candidate = dict(context); claimed = candidate.pop("context_hash")
        if hash_value(candidate) != claimed: raise RuntimeError("Context integrity check failed")
    return contexts
