# RC-MLQ experiments

This repository contains two experiment entry points for Resolution-Calibrated
Multiway Line Queries (RC-MLQ).

| Directory | Experiment | First run |
| --- | --- | --- |
| [1_Synthetic](1_Synthetic/) | Synthetic ideal-point preference learning with simulated multiway choices | Run the CPU smoke configuration |
| [2_Robot](2_Robot/) | Preference learning with a Meta-World reach-v3 motion audit | Run one interaction and replay its candidates in MuJoCo |

Clone the full repository so the Synthetic entry point can use the numerical
model and posterior implementations in `2_Robot/metaworld_fast`. Each guide
describes setup, experiment settings, and generated artifacts. The Synthetic
implementation was reconstructed from the manuscript's protocol. Its new runs
have their own provenance.
