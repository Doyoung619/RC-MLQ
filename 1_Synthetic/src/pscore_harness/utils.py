from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy and return the run-local generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def stable_hash(payload: Any, length: int = 12) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_manifest() -> dict[str, Any]:
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import yaml

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
    }


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_finite(name: str, value: np.ndarray | float) -> None:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
