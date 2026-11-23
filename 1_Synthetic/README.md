# Synthetic GLM-Score experiments

Generate a seeded anisotropic candidate pool and Gaussian or Bernoulli responses, then compare batch acquisition methods for GLM parameter estimation. Ours is `fb_opt2` (Fisher Balancing). A batch is selected using the current fit, then its responses are revealed and the ridge GLM is refit.

## Quick start

Run from this directory with Python 3.10 or newer. Python 3.11 is a good starting point for a new environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python -m pscore_harness plan --config configs/smoke.yaml
python -m pscore_harness run --config configs/smoke.yaml --workers 1 --fail-fast
python -m pscore_harness validate --config configs/smoke.yaml
python -m pytest -q
```

In Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. If the checkout path is already long, set a short absolute `output_root` in the YAML.

The smoke config runs two batches for all 12 eligible methods in each family (24 jobs). A run writes `trajectory.csv`, `theta_star.npy`, the selected row IDs, and per-batch checkpoints under `results_spec/<experiment_name>/<run_id>/`. The trajectory includes parameter error and the smallest eigenvalues of the true and plug-in Fisher matrices.

## Run with new data

Copy one of the YAML settings and change `seeds`. A new seed generates a new pool, target and response randomness. Methods within the same condition and seed share these inputs. The default `theta_regime: weak` places the target along the weakest direction of the generated pool. Set `theta_regime: random` to draw its direction randomly.

- [Smoke](configs/smoke.yaml) runs 24 jobs.
- [Pilot](configs/synthetic_pilot.yaml) runs 72 jobs.
- [Main](configs/synthetic_main.yaml) runs 2,160 jobs across dimensions, batch sizes, families and seeds.

Each run keeps one candidate pool. At each batch it selects distinct previously unobserved rows from that pool. The settings here describe a fixed-pool synthetic experiment. For a custom comparison, add a top-level `methods: [{name: fb_opt2}, {name: random}]` to the copied YAML.

Use the same YAML and output directory to resume a run. Use a distinct `experiment_name` to keep a custom campaign's results and reporting separate.

## Generate figures and tables

After completing the smoke config, run the following from this directory.

```bash
python -m pscore_harness report --results-root results_spec/smoke_algorithm_suite --paper-dir reports/smoke --require-config configs/smoke.yaml
```

The report writes four-panel figures, aggregate and endpoint tables, and seed diagnostics to `reports/smoke/generated/algorithm_suite/`. For other campaigns, update the results directory and config path together. See the [algorithm guide](docs/ALGORITHM_SUITE_KO.md) for acquisition, response generation and checkpoint details.
