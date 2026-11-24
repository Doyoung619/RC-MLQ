# Synthetic RC-MLQ experiments

This experiment simulates multiway choices around a latent ideal point drawn
uniformly from `[-1, 1]^d`. Each method proposes its own slate, receives one
squared-distance multinomial-logit response, and updates its own posterior.
Methods share a latent ideal point within each matched interaction and use
separate answer streams. RC, PQ, and QI use marginal quadrature. PS and GC use
tempered joint SMC.

The code reconstructs the manuscript protocol and reuses the numerical model
and posterior engines in `../2_Robot/metaworld_fast`. Run it from a full checkout
of this repository. Synthetic runs use CPU inference and require no simulator.

## Setup and smoke run

From `1_Synthetic` with Python 3.11 on Linux:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m rcmlq_synthetic plan --config configs/smoke.yaml
.venv/bin/python -m rcmlq_synthetic run --config configs/smoke.yaml --run-dir outputs/smoke --workers 1
.venv/bin/python -m rcmlq_synthetic report --config configs/smoke.yaml --run-dir outputs/smoke
.venv/bin/python -m pytest -q
```

## Experiment settings

| Configuration | Purpose |
| --- | --- |
| `configs/paper_main.yaml` | Main four-way experiment with \(d=M=4\), \(\beta=4\), eight interactions and 4,096 answers per method |
| `configs/paper_sweeps.yaml` | Dimension and choice-count grid, response-sensitivity sweep, and binary baselines |
| `configs/e01.yaml` | Matched comparison of Fisher selection, direct grid EIG, fixed scales and original controls |
| `configs/e02.yaml` | One-factor sensitivity to scale-grid resolution and range |

Inspect the resolved job list with `plan` before starting a campaign.
Use a new run directory for a different configuration or seed. On Linux,
the detached launcher uses the project Python and writes `launcher.log` and
`launcher.exit` to the run directory. Reuse the same settings and run directory
to resume.

```sh
./scripts/run_detached.sh configs/paper_main.yaml outputs/paper-main 4
```

## Interpreting the outputs

Each interaction stores its simulated questions and responses, posterior
moments, selected EIG and grid-reference EIG with its seeds and configuration.
Risk is the posterior covariance trace. Realized squared error uses the
simulated latent preference. The final query is evaluated after the last
response and receives no answer. The report writes `curves.csv`,
`paired_vs_rc.csv`, and `missing_jobs.csv`. Interval estimates resample whole
interactions and share indices across methods and rounds.

The optional ellipsoid stage replays those saved histories and records a
separate approximate ratio using a finite search over candidate coordinates.
Install CPU PyTorch for this stage. By default it checks early, middle, and
terminal states. Pass `--all-rounds` for a full reference curve or `--rounds`
to preselect checkpoints.

```sh
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
.venv/bin/python -m rcmlq_synthetic reference --config configs/smoke.yaml --run-dir outputs/smoke
```

The reconstruction has its own seeds and numerical provenance. It generates
new trajectories and preserves the original seven-scale grid for the
grid-reference diagnostic across the E02 sensitivity groups.
