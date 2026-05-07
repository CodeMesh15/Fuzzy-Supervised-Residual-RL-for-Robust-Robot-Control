# Research Findings: Fuzzy-Supervised Residual TD3

Final replicated run:

```powershell
python ims_fuzzy_td3_experiments/run_experiments.py --timesteps 12000 --seeds 0 1 2 --eval-episodes 4 --force-train
python ims_fuzzy_td3_experiments/run_experiments.py --timesteps 12000 --seeds 0 1 2 --eval-episodes 4 --skip-train
```

This produced 12 evaluation rollouts per controller per scenario.

## Core Result

The strongest defensible claim is not that Fuzzy-TD3 is universally superior. The stronger and more publishable claim is:

> A fuzzy-supervised residual TD3 controller improves tracking robustness under payload/friction distribution shift while preserving full-horizon completion and substantially reducing torque roughness relative to a tuned PID controller.

## Final Summary Table

| Scenario | Controller | RMSE | Failure-Penalized RMSE | Safety Violation Rate | Torque Smoothness RMS | Completion |
|---|---:|---:|---:|---:|---:|---:|
| nominal | Fuzzy | 0.0286 | 0.0286 | 0.0000 | 34.72 | 1.00 |
| nominal | PID | 0.0391 | 0.0391 | 0.0000 | 1134.74 | 1.00 |
| nominal | TD3 direct | 0.6909 | 0.9787 | 0.2893 | 318.90 | 0.67 |
| nominal | Fuzzy-TD3 residual | 0.1074 | 0.1074 | 0.0000 | 70.73 | 1.00 |
| payload_2x | Fuzzy | 0.5193 | 0.5193 | 0.2581 | 29.96 | 1.00 |
| payload_2x | PID | 0.4190 | 0.4190 | 0.1867 | 253.41 | 1.00 |
| payload_2x | TD3 direct | 0.7070 | 1.2478 | 0.2829 | 176.00 | 0.33 |
| payload_2x | Fuzzy-TD3 residual | 0.3837 | 0.3837 | 0.1747 | 27.80 | 1.00 |
| severe_shift | Fuzzy | 0.8893 | 0.8893 | 0.3300 | 35.44 | 1.00 |
| severe_shift | PID | 0.6736 | 0.6736 | 0.2731 | 253.18 | 1.00 |
| severe_shift | TD3 direct | 0.6724 | 1.5010 | 0.1615 | 35.42 | 0.00 |
| severe_shift | Fuzzy-TD3 residual | 0.6889 | 0.6889 | 0.2675 | 52.09 | 1.00 |

## Effect Sizes

Using failure-penalized RMSE:

- Under `payload_2x`, Fuzzy-TD3 improves over fuzzy-only by 26.1%, PID by 8.4%, and direct TD3 by 69.3%.
- Under `severe_shift`, Fuzzy-TD3 improves over fuzzy-only by 22.5% and direct TD3 by 54.1%; it is 2.3% worse than PID on RMSE but has much smoother torque.
- Under `nominal`, fuzzy-only and PID remain better than Fuzzy-TD3. This is important: the final paper should not claim universal dominance.

## Manuscript Reframing

Replace the current broad claim:

> Hybrid Fuzzy-TD3 is superior for trajectory tracking.

with:

> Fuzzy-TD3 is most valuable as a safety-preserving residual adaptation layer under plant uncertainty. It improves robustness under payload/friction shifts and avoids the early-training instability of direct torque TD3, while nominal-regime performance motivates future residual gating or uncertainty-triggered activation.

## Next Research Step

The natural journal extension is an uncertainty-gated residual controller:

```text
tau = tau_fuzzy + g(e, de, disturbance_score) * tau_TD3
```

where `g` tends to zero in nominal tracking and increases under persistent error, payload shift, or friction mismatch. That would preserve the excellent nominal fuzzy/PID behavior while keeping the adaptive advantage under distribution shift.

## Residual-Bound Ablation

The residual-bound ablation used bounds of 6, 10, 14, and 18 Nm, two training seeds, 6000 timesteps per bound/seed, and three evaluation episodes.

| Scenario | Fuzzy-only 0 Nm | 6 Nm | 10 Nm | 14 Nm | 18 Nm |
|---|---:|---:|---:|---:|---:|
| nominal | 0.0283 | 0.0911 | 0.0881 | 0.1109 | 0.0980 |
| payload_2x | 0.5192 | 0.4782 | 0.4217 | 0.4098 | 0.4028 |
| severe_shift | 0.8892 | 0.8054 | 0.7543 | 0.7259 | 0.6793 |

Interpretation: larger residual authority helps shifted-plant tracking but hurts nominal tracking relative to the fuzzy supervisor. This directly supports the future need for uncertainty-gated residual activation.

## 6-DOF Validation Attempt

`pybullet`, `pybullet_envs`, `mujoco`, and `gymnasium_robotics` were not installed. A `python -m pip install pybullet` attempt timed out after five minutes, and `pybullet` remained unavailable. The current manuscript therefore reports the completed 2-DOF benchmark and states 6-DOF URDF/hardware validation as a limitation and required next step.

## Expanded Journal-Readiness Update

An expanded run was completed with five training seeds, 12000 timesteps per learned controller, and 20 evaluation rollouts per controller per scenario:

```powershell
python ims_fuzzy_td3_experiments/run_experiments.py --timesteps 12000 --seeds 0 1 2 3 4 --eval-episodes 4 --include-gated --include-sac --include-ddpg --include-residual-sac
```

Added baselines:

- Robust PID / sliding-mode term.
- Computed torque and heuristic adaptive computed torque.
- Tuned fuzzy controller.
- Direct SAC and DDPG.
- Residual SAC.
- Trained gated Fuzzy-TD3.
- Evaluation-time quadratic-gated Fuzzy-TD3.

Key updated result:

- Residual SAC is the strongest learned shifted-plant controller: 0.3839 failure-penalized RMSE on `payload_2x` and 0.6804 on `severe_shift`.
- Fuzzy-TD3 residual remains strong: 0.3936 on `payload_2x` and 0.7065 on `severe_shift`.
- Evaluation-gated Fuzzy-TD3 nearly fixes nominal degradation: 0.0299 nominal RMSE versus 0.0284 for fuzzy-only and 0.1184 for always-active Fuzzy-TD3.
- Direct TD3/SAC/DDPG fail frequently, reinforcing the value of fuzzy-supervised residual learning.
- Under severe shift, residual SAC is statistically indistinguishable from PID on failure-penalized RMSE but uses much smoother torque.

The equal-budget residual-bound ablation was rerun with five seeds and 12000 timesteps per setting:

| Scenario | Fuzzy-only 0 Nm | 6 Nm | 10 Nm | 14 Nm | 18 Nm |
|---|---:|---:|---:|---:|---:|
| nominal | 0.0292 | 0.1561 | 0.1335 | 0.1308 | 0.1502 |
| payload_2x | 0.5192 | 0.4403 | 0.4136 | 0.3982 | 0.3933 |
| severe_shift | 0.8891 | 0.7887 | 0.7348 | 0.7081 | 0.6769 |

The stronger paper claim is now:

> Fuzzy-supervised residual RL is a robust adaptation layer. Residual SAC and TD3 improve shifted-plant tracking and avoid direct-RL failures, while uncertainty gating suppresses residual action in nominal operation.
