# RC-MLQ experiments

This repository contains two independent experiments. Clone it and follow the setup in the experiment directory you want to run.

```bash
git clone https://github.com/Doyoung619/RC-MLQ.git
cd RC-MLQ
```

| Directory | Experiment | First run |
| --- | --- | --- |
| [1_Synthetic](1_Synthetic/) | GLM-Score with generated Gaussian and Bernoulli responses | Run the two-batch smoke configuration on a CPU |
| [2_Robot](2_Robot/) | RC-MLQ preference learning with a Meta-World reach-v3 motion audit | Run one interaction and replay its candidates in MuJoCo on Linux |

Each guide includes installation, run commands, and the location of generated results. You can change the seed to generate a new experiment.
