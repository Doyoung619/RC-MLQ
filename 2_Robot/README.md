# Robot preference-learning experiment

RC-MLQ learns a synthetic ideal point from preferences over candidate coordinates. The Meta-World `reach-v3` simulation replays those candidates as MuJoCo motions and records whether they execute as specified. The standard experiment uses four dimensions, 128 answers per interaction, five interactions per method and setting, and six methods (RC, PQ, QI, PS, GC, SG).

## Install on Linux

Run from this directory with Python 3.11 on Linux. The experiment scheduler uses Linux process and file-lock APIs. For a minimal Debian or Ubuntu installation, install `python3-venv`, `libgl1`, `libegl1`, `libegl-mesa0`, `util-linux` and `tzdata` through the system package manager.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install -r requirements.txt
```

The PyTorch command installs the CPU build. For a CUDA-enabled run, install a compatible CUDA build of PyTorch 2.5.1 instead. The full pipeline uses EGL for the video stage. The [recorded environment](metaworld_fast/requirements-frozen.txt) is available for provenance.

## Try a small experiment

```bash
python smoke.py --run-dir metaworld_fast/outputs/smoke --seed 20261004
```

This runs one RC interaction with two answers and replays all eight displayed candidates. It saves inference arrays in the run directory's `statistical/` folder and motion diagnostics in `smoke.json`. Change the seed and use a new run directory for a fresh interaction.

## Run the compact E01 scale ablation

The follow-up holds the main setting fixed at four candidates, four preference dimensions and `beta=4`. Each method uses the same ten latent-preference and robot context seeds for 128 answers. RC-Fisher and Direct EIG select from the original seven scales. The comparison includes PQ and three prespecified fixed scales (0.125, 0.5 and 2.0).

From this directory, with the environment installed:

```bash
.venv/bin/python robot_e01.py plan --ellipsoid
bash run_e01_detached.sh metaworld_fast/outputs/robot_e01 --ellipsoid
.venv/bin/python robot_e01.py status --run-dir metaworld_fast/outputs/robot_e01
```

This schedules 60 statistical interactions and 30,720 selected candidate slots for the reach-v3 motion audit. Choices depend on the saved preference coordinates. The existing open-loop adapter maps those coordinates to MuJoCo motions, as it does in the [Robot implementation](metaworld_fast/PROVENANCE.md). Omitting `--ellipsoid` produces the same inference and motion audit with the seven-scale information ratio. With `--ellipsoid`, the separate numerical ellipsoid search evaluates the saved pre-answer histories at rounds 0, 64 and 127 using the existing search budget and geometry. Its achieved reference is approximate.

The run directory contains `summary.json` with paired endpoint differences and measured stage times, per-round histories in `statistical/`, simulator diagnostics in `rollouts/`, and the optional checkpoint search in `ellipsoid/`. The distinct metrics are posterior MSE (covariance trace) and squared error against the simulated latent preference. With at least two interactions, each paired difference includes a 95% percentile interval from 2,000 shared resamples of the interaction IDs. Treat intervals from the ten-interaction panel as exploratory. Stage wall times cover the latest invocation, and worker times cover all retained shards. The detached launcher writes to `logs/supervisor.log` and records the exit status in `exit_code`. Rerun with the same directory and arguments to resume at the next unfinished interaction or replay shard. Use a fresh directory with `--interactions 1 --answers 2` for a smoke run.

## Run the five-interaction benchmark

```bash
python -m metaworld_fast.cli inspect
python -m pytest -q metaworld_fast/tests
python -m metaworld_fast.cli run \
  --config metaworld_fast/config/final.yaml \
  --run-dir metaworld_fast/outputs/new_run
```

The full benchmark profiles the machine, then runs 210 interactions and their simulator replays. It writes checkpoints, trajectories, reports and figures under the given run directory. Use the same directory and config to resume. To generate a different set of interactions, change `experiment.root_seed` in a copy of the YAML and use a new run directory.

## Regenerate existing figures

The [figure package](results/README.md) includes the compressed curve CSV and a plotting script. This workflow needs NumPy and Matplotlib.

```bash
python -m pip install -r results/figure_editing/requirements.txt
python results/figure_editing/plot_figures.py --list
python results/figure_editing/plot_figures.py --only m2_addon_risk_all --formats svg pdf
```

## Extensions and provenance

- The ten-interaction extension takes a completed five-interaction run. Use `python -m metaworld_fast.extension --source <five-run-dir> --run-dir <ten-run-dir>`.
- The ellipsoid reference reevaluation takes a completed ten-interaction run and its protocol source text. See `python -m metaworld_fast.ellipsoid_recompute --help`.
- The historical M=2 add-on uses the ten-interaction and ellipsoid artifacts and its original server's GPU queue setup. Its source and specification are included for inspection. The compact figure package is sufficient to redraw its published curves.

The experiment source was imported from [Ideal-Robot-Control at aa4841b](https://github.com/Doyoung619/Ideal-Robot-Control/tree/aa4841bf7dcc90686568d45161c323aef459a2ec). See the [original experiment specification](CODEX_RCMLQ_METAWORLD_FAST_T128_S5_KO.md), [M=2 specification](metaworld_m2_addon_codex_prompt.md), [implementation provenance](metaworld_fast/PROVENANCE.md), and [numerical limitations](metaworld_fast/LIMITATIONS.md) for details.
