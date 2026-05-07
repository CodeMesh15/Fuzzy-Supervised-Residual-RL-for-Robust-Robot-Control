# Fuzzy-Supervised Residual TD3 Experiments

## Research Question

Does a Mamdani fuzzy supervisor improve safety, sample efficiency, and disturbance rejection when TD3 is trained as a residual torque compensator for robotic trajectory tracking?

## What Is Tested

- `PID`: gravity-compensated PID baseline.
- `Fuzzy`: Mamdani fuzzy supervisor with nominal gravity compensation.
- `TD3 direct`: TD3 controls the full continuous torque.
- `Fuzzy-TD3 residual`: TD3 controls only a bounded residual torque added to the fuzzy supervisor.
- `SAC direct`: optional extra RL baseline.

## Disturbance Scenarios

- `nominal`: nominal payload and friction.
- `payload_2x`: payload doubles after 2.5 s with increased friction and sinusoidal torque disturbance.
- `severe_shift`: payload triples after 2.5 s with stronger friction and disturbance.

## Run

```powershell
python ims_fuzzy_td3_experiments/run_experiments.py --timesteps 12000 --seeds 0 1 2 --eval-episodes 4
```

For a heavier comparison:

```powershell
python ims_fuzzy_td3_experiments/run_experiments.py --timesteps 30000 --seeds 0 1 2 3 4 --eval-episodes 6 --include-sac --force-train
```

Outputs are written to `ims_fuzzy_td3_experiments/results/`:

- `metrics_raw.csv`: every rollout.
- `metrics_summary.csv`: mean, standard deviation, count, standard error, and 95% confidence interval.
- `figures/`: RMSE, safety, smoothness, learning-curve, and representative rollout plots.
- `experiment_manifest.json`: scenario and run configuration for reproducibility.

## Residual-Bound Ablation

```powershell
python ims_fuzzy_td3_experiments/run_residual_bound_ablation.py --bounds 6 10 14 18 --seeds 0 1 --timesteps 6000 --eval-episodes 3 --force-train
```

This writes:

- `results/ablations/residual_bound_raw.csv`
- `results/ablations/residual_bound_summary.csv`
- `results/figures/ablation_residual_bound_*.png`

## Revised Manuscript

The journal-style rewrite is in:

- `manuscript/revised_fuzzy_td3_manuscript.tex`
- `manuscript/references.bib`
- `manuscript/revised_fuzzy_td3_manuscript.pdf`

## Optional 6-DOF/URDF Scaffold

`pybullet_6dof_validation.py` is ready for a KUKA-style PyBullet check, but the current machine cannot build/install `pybullet` because Microsoft C++ Build Tools are missing.

```powershell
python ims_fuzzy_td3_experiments/pybullet_6dof_validation.py
```

## Manuscript-Framing Caution

This is a rigorous first benchmark, not yet a final industrial claim. For journal submission, the same protocol should be repeated on a standard 6-DOF robot model or hardware-in-the-loop setup, but the 2-DOF benchmark is useful because it makes the safety and residual-learning effects measurable and repeatable.
