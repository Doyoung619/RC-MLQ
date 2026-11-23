from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .utils import stable_hash


RUN_SCHEMA_VERSION = "2026-10-05.spec-suite.1"


@dataclass(frozen=True)
class RunSpec:
    experiment_name: str
    source: str
    dataset: dict[str, Any]
    model: dict[str, Any]
    protocol: dict[str, Any]
    method: dict[str, Any]
    estimator: dict[str, Any]
    metrics: dict[str, Any]
    seed: int
    output_root: str
    cache_root: str

    @property
    def run_id(self) -> str:
        descriptive = [
            self.experiment_name,
            self.source,
            str(self.dataset.get("name", self.dataset.get("data_id", "data"))),
            str(self.model.get("family")),
            str(self.protocol.get("name")),
            str(self.method.get("name")),
            f"seed{self.seed}",
        ]
        clean = "__".join(x.replace("/", "-").replace(" ", "-") for x in descriptive)
        return f"{clean}__{stable_hash({'schema': RUN_SCHEMA_VERSION, 'spec': asdict(self)})}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return raw


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _expand_dict_grid(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand only values wrapped as {'grid': [...]} to avoid accidental list expansion."""
    keys: list[str] = []
    values: list[list[Any]] = []
    fixed: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict) and set(value) == {"grid"}:
            keys.append(key)
            values.append(_as_list(value["grid"]))
        else:
            fixed[key] = value
    if not keys:
        return [copy.deepcopy(base)]
    expanded = []
    for combo in itertools.product(*values):
        item = copy.deepcopy(fixed)
        item.update(dict(zip(keys, combo, strict=True)))
        expanded.append(item)
    return expanded


def expand_config(config: dict[str, Any]) -> list[RunSpec]:
    config = copy.deepcopy(config)
    if config.get("algorithm_suite") == "spec_v1":
        from .spec_suite import canonical_method, default_methods, methods_for_family
        config.setdefault("methods", default_methods())
        for protocol in config.get("protocols", []):
            protocol["algorithm_suite"] = "spec_v1"
        for method in config["methods"]:
            method["name"] = canonical_method(method["name"])
    required = ["experiment_name", "source", "datasets", "models", "protocols", "methods"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")

    seeds = [int(s) for s in _as_list(config.get("seeds", [0]))]
    estimator = copy.deepcopy(config.get("estimator", {}))
    metrics = copy.deepcopy(config.get("metrics", {}))
    output_root = str(config.get("output_root", "results"))
    cache_root = str(config.get("cache_root", "data/openml_cache"))

    datasets = [x for d in config["datasets"] for x in _expand_dict_grid(d)]
    models = [x for m in config["models"] for x in _expand_dict_grid(m)]
    protocols = [x for p in config["protocols"] for x in _expand_dict_grid(p)]
    methods = [x for m in config["methods"] for x in _expand_dict_grid(m)]

    specs: list[RunSpec] = []
    for dataset, model, protocol, method, seed in itertools.product(
        datasets, models, protocols, methods, seeds
    ):
        if protocol.get("algorithm_suite") == "spec_v1":
            from .spec_suite import methods_for_family
            if method["name"] not in methods_for_family(model["family"]):
                continue
        allow_models = method.get("allow_models")
        if allow_models and model.get("family") not in allow_models:
            continue
        allow_protocols = method.get("allow_protocols")
        if allow_protocols and protocol.get("name") not in allow_protocols:
            continue
        deny_models = method.get("deny_models", [])
        if model.get("family") in deny_models:
            continue
        specs.append(
            RunSpec(
                experiment_name=str(config["experiment_name"]),
                source=str(config["source"]),
                dataset=copy.deepcopy(dataset),
                model=copy.deepcopy(model),
                protocol=copy.deepcopy(protocol),
                method=copy.deepcopy(method),
                estimator=copy.deepcopy(estimator),
                metrics=copy.deepcopy(metrics),
                seed=seed,
                output_root=output_root,
                cache_root=cache_root,
            )
        )
    return specs


def iter_openml_dataset_ids(config: dict[str, Any]) -> Iterable[int]:
    if config.get("source") != "openml":
        return []
    return sorted({int(d["data_id"]) for d in config["datasets"]})
