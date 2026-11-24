"""Create a deterministic list of closed-loop interactions."""
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from . import backend
from metaworld_fast.line_template import SCALE_GRID

CORE = ("RC", "PQ", "QI", "PS", "GC")
BINARY = ("SG", "SM", "MEM", "GM", "2M", "IT", "IB")
ABLATION = ("RC-EIG", *(f"RC-Fixed-{scale:g}" for scale in SCALE_GRID))
JOINT = ("PS", "GC", "SG")


def _grid(values):
    grid = np.asarray(values, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or not np.all(grid > 0) or not np.all(np.diff(grid) > 0)):
        raise ValueError("A scale grid must be finite, positive and strictly increasing")
    return grid.tolist()


def load_campaign(path, *, answers=None, interactions=None, methods=None):
    config = yaml.safe_load(Path(path).read_text())
    if answers is not None:
        config["answers"] = answers
    if interactions is not None:
        config["interactions"] = interactions

    for name in ("answers", "interactions", "nodes", "particles", "checkpoint_every", "bootstrap"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["nodes"] < 5 or config["nodes"] % 2 == 0 or config["particles"] < 32:
        raise ValueError("Use odd quadrature nodes >=5 and at least 32 particles")
    if config["particles"] < 2 * max(config["choices"]):
        raise ValueError("Use at least twice as many particles as displayed choices")

    settings = {
        (d, choices, float(config["beta"]))
        for d, choices in product(config["dimensions"], config["choices"])
    }
    settings.update((item["d"], item["M"], float(item["beta"])) for item in config.get("extra_settings", []))
    grid_variants = {}
    for label, nodes in config.get("scale_variants", {}).items():
        if not label.replace("_", "").isalnum() or label in CORE + BINARY + ABLATION:
            raise ValueError(f"Invalid grid variant name: {label}")
        if isinstance(nodes, dict):
            nodes = np.geomspace(float(nodes["min"]), float(nodes["max"]), int(nodes["count"]))
        grid_variants[label] = _grid(nodes)
    all_methods = CORE + BINARY + ABLATION + tuple(grid_variants)
    methods_to_run = list(methods if methods is not None else config["methods"])
    binary_methods = [] if methods is not None else config.get("binary_methods", [])
    for method in methods_to_run + binary_methods:
        if method not in all_methods:
            raise ValueError(f"Unknown method: {method}")

    plan = []
    for d, choices, beta in sorted(settings):
        if type(d) is not int or d < 1 or type(choices) is not int or choices < 2 or beta <= 0 or not np.isfinite(beta):
            raise ValueError(f"Invalid experiment setting: d={d}, M={choices}, beta={beta}")
        group = f"d{d:02d}_m{choices:02d}_b{beta:g}"
        selected = dict.fromkeys(methods_to_run + (binary_methods if choices == 2 else []))
        for method in selected:
            if method in BINARY and choices != 2:
                if methods is not None and any(item[1] == 2 for item in settings):
                    continue
                raise ValueError(f"{method} requires binary choices")
            for interaction in range(config["interactions"]):
                plan.append({
                    "key": f"{group}__{method}__i{interaction:03d}", "group": group,
                    "method": method, "seed": config["root_seed"], "interaction": interaction,
                    "d": d, "M": choices, "beta": beta, "answers": config["answers"],
                    "nodes": config["nodes"], "particles": config["particles"],
                    "checkpoint_every": config["checkpoint_every"],
                    "grid": grid_variants.get(method, SCALE_GRID.tolist()),
                    "fisher_grid_variant": method in grid_variants,
                    "diagnostic_rounds": config.get("diagnostic_rounds", []),
                })
    if not plan:
        raise ValueError("Campaign has no runnable interactions")
    return {"schema": 1, "jobs": plan, "bootstrap": config["bootstrap"]}
