"""Run two RC preference queries and replay every displayed candidate."""

import argparse
import json
from pathlib import Path

import numpy as np

from metaworld_fast.contexts import prepare_contexts
from metaworld_fast.renderer import RobotAdapter
from metaworld_fast.statistical_runner import run_statistical
from metaworld_fast.util import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    destination = args.run_dir.resolve()
    job = {
        "setting_id": "smoke_m04_b04",
        "M": 4,
        "beta": 4.0,
        "method": "RC",
        "interaction_id": 0,
        "answers": 2,
        "root_seed": args.seed,
    }
    result = run_statistical(job, destination)
    with np.load(result["artifact"], allow_pickle=False) as data:
        questions = data["q"]
    contexts = prepare_contexts(destination / "contexts", count=1, root_seed=args.seed)
    adapter = RobotAdapter(contexts)
    diagnostics = []
    try:
        for round_index, question in enumerate(questions):
            for candidate_index, query in enumerate(question):
                record = adapter.rollout(query, context_id=0)["diagnostics"]
                if not record["simulation_valid"] or not record["command_reconstruction_ok"]:
                    raise RuntimeError(f"Motion replay failed at round {round_index}, candidate {candidate_index}: {record}")
                diagnostics.append({"round": round_index, "candidate": candidate_index, **record})
    finally:
        adapter.close()

    summary = {
        "seed": args.seed,
        "answers": result["answers_completed"],
        "rollouts": len(diagnostics),
        "total_action_steps": sum(record["actual_steps"] for record in diagnostics),
        "statistical_artifact": result["artifact"],
    }
    atomic_json(destination / "smoke.json", {**summary, "motion_diagnostics": diagnostics})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
