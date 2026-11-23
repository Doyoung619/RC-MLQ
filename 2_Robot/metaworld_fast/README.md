# RC-MLQ × Meta-World, d=4 / T=128 / five interactions

For a fresh checkout, use the [installation and execution guide](../README.md).
The paths below describe the original experiment host and archived runs.

This package implements the supplied Korean execution specification in an isolated `metaworld_fast/` directory. It preserves the coordinate-based synthetic preference model and replays selected questions as actual Meta-World reach-v3 motions. The seven settings and six methods require 210 statistical runs, 26,880 actual answers, and 161,280 logical selected-candidate clips before exact caching.

Use the project-local Python at `/data1/doyoung/Ideal_Control/.venv/bin/python` and set `PYTHONNOUSERSITE=1`. The isolated environment reuses selected pre-existing numerical libraries through a read-only package-path configuration; Meta-World and MuJoCo are installed in the environment. Exact paths and pinned versions are recorded in each run's `ENVIRONMENT.txt` and manifest. The environment is not a portable lockfile independent of the local library paths.

The CLI and detached launcher are maintained by the scheduler. Run from `/data1/doyoung/Ideal_Control`. `python -m metaworld_fast.cli --help` lists the actual installed interface. See the current run's `STATUS.md`, `ETA.md` and `logs/` for verified invocation, progress, measured throughput and completion estimates. An ETA based on profiling is an estimate and changes with history length, reference refinement and simulator throughput.

The launcher uses a detached session with unbuffered logs and records its PID. Do not start a second launcher for the same run directory. Resume the same frozen configuration and existing checkpoints; semantic statistical changes require a new run ID. Never terminate unrelated jobs.

## Artifacts and indexing

- `config_frozen.yaml`, `manifest.json`, `HARDWARE.json`, `ENVIRONMENT.txt`, `PROFILE.json`: frozen inputs, seeds, hashes, dependency versions and actual resource measurements.
- `statistical/*.npz` and matching JSON: displayed/raw questions, actual responses, fixed theta, all posterior moments, risk, realized error, pre-answer EIG/ratios and saved reference questions. Risk/error have indices 0 through 128; EIG/eta have indices 0 through 127. Checkpoints preserve actual histories and RNG state.
- `rollout_requests/`, `rollouts/`, `diagnostics/`, `videos/`: logical requests, actual submitted actions/positions, compact command and physical-motion checks, calibration, reference-grid audit and preset qualitative clips. Pending commands are not counted as executions.
- `validation/`: same-history independent numerical replays and small d=2 checks, including unresolved cases and reference uncertainty. The selected histories are fixed in `validation.py` before rankings are inspected.
- `figures/`: four raw-curve figures, each in vector PDF and PNG. Thin curves are individual interactions; thick curves are arithmetic means. Dashed means mark incomplete interaction counts.
- `tables/`: endpoint values at answer counts 32/64/128, corresponding eta at pre-answer t=31/63/127, effective sample counts, physical coverage/separation, numerical discrepancies and completion summary.
- `REPORT.md`: artifact-based results, status and required limitations. It can be regenerated from existing artifacts and does not equate missing data with success.

`reporting.generate_report(run_dir)` regenerates tables, figures and the report; `validation.validate_run(run_dir, device=...)` replays the prespecified numerical checks. These functions require saved statistical artifacts. Larger reference replays do not modify the original interaction or its result curves.

Read `PROVENANCE.md` and `LIMITATIONS.md` before interpreting results. The unavailable manuscript has not been independently audited. The beta sweep here is d=4, the oracle is defined on q rather than trajectories, and SG stays at stage 0 throughout the measured queries. No result is evidence of RL improvement or an asymptotic rate proof.

## Ten-interaction extension

The user subsequently requested exactly five additional interaction IDs, `5,6,7,8,9`.
`extension.py` preserves the original `0..4` artifacts and computes only the additional
210 interactions. The combined output has 420 runs and ten curves per method/setting.
Plot colors, layouts, axis definitions and mean calculation are unchanged. Statistical
engines and the robot adapter must match the original source hashes before reuse.

The verified detached launch/resume command is:

```bash
cd /data1/doyoung/Ideal_Control
metaworld_fast/scripts/extend_five_detached.sh \
  metaworld_fast/outputs/rcmlq_t128_s5_20261004T1933Z \
  metaworld_fast/outputs/rcmlq_t128_s10_20261004T2008Z
```

The original output remains intact. `CURRENT_RUN.txt` identifies the combined output;
its `STATUS.md`, `ETA.md`, `seed_manifest.json`, `EXTENSION.md` and `reused_statistical_sha256.json`
record progress, seed assignment and reuse. Original numerical unresolved cases remain
visible, with the same checks also applied to the first added interaction, ID5.
