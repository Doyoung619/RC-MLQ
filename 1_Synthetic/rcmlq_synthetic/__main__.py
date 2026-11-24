"""Command-line entry point."""
import argparse
import json
from pathlib import Path

from .config import load_campaign
from .runner import run_campaign


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "report", "reference"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--run-dir", type=Path, required=name != "plan")
        command.add_argument("--answers", type=positive_int)
        command.add_argument("--interactions", type=positive_int)
        command.add_argument("--methods", nargs="+")
        if name == "run":
            command.add_argument("--workers", type=positive_int, default=1)
        if name == "reference":
            command.add_argument("--rounds", nargs="+", type=int)
            command.add_argument("--all-rounds", action="store_true")
            command.add_argument("--workers", type=positive_int, default=1)
            command.add_argument("--device", default="cpu")
            command.add_argument("--starts", type=positive_int, default=3)
            command.add_argument("--steps", type=positive_int, default=40)
            command.add_argument("--integration-points", type=positive_int, default=512)
            command.add_argument("--refinement-points", type=positive_int, default=4096)
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config, answers=args.answers, interactions=args.interactions,
                             methods=args.methods)
    if args.command == "plan":
        print(json.dumps({"interactions": len(campaign["jobs"]),
                          "answers": sum(job["answers"] for job in campaign["jobs"]), **campaign}, indent=2))
    elif args.command == "run":
        run_campaign(campaign, args.run_dir, args.workers)
    elif args.command == "report":
        from .report import report_campaign
        report_campaign(campaign, args.run_dir)
    else:
        from .reference import evaluate_campaign
        if args.all_rounds and args.rounds:
            parser.error("Choose either --rounds or --all-rounds")
        evaluate_campaign(campaign, args.run_dir, rounds=args.rounds, all_rounds=args.all_rounds,
                          device=args.device, starts=args.starts, steps=args.steps,
                          points=args.integration_points, refinement_points=args.refinement_points, workers=args.workers)


if __name__ == "__main__":
    main()
