# Speaker Notes: Fuzzy-Supervised Residual RL

Suggested timing: 10 to 12 minutes.

## Slide 1: Title

Introduce the work as an upgraded version of the IMS term-paper idea. Emphasize that the project moved from a concept-only hybrid controller to a reproducible benchmark with baselines, ablations, and honest limitations.

## Slide 2: Why This Problem Matters

Start with the practical robotics motivation: industrial arms face payload changes, friction drift, and disturbances. Classical controllers are strong near their design model, but direct RL is risky because exploration in torque space can be unstable.

## Slide 3: Positioning the Idea

Explain the division of labor. The fuzzy controller gives immediate interpretable behavior; residual RL only learns the correction. This is the central design philosophy.

## Slide 4: Controller Architecture

Walk through the control law:

```text
tau = tau_fuzzy + g(e, de, xi) tau_RL
```

The most important point is the gate. When the baseline is doing well, the residual should be quiet. When persistent error appears, the residual should activate.

## Slide 5: What Changed From the Original Claim

Be very clear here. The old manuscript overclaimed universal superiority. The new claim is stronger scientifically because it says exactly where the method helps and where it does not.

## Slide 6: Experimental Protocol

Describe the two-link manipulator benchmark, the disturbance timing, and the three scenarios. Mention that this is reproducible and intentionally controlled, but not yet final industrial validation.

## Slide 7: Baselines and Metrics

Stress that the paper now compares against serious baselines: PID, robust PID, computed torque, adaptive computed torque, fuzzy-only, direct RL, and residual RL. Explain failure-penalized RMSE because it prevents failed RL rollouts from hiding behind short trajectories.

## Slide 8: Main Result

Use the plot to tell the core story. Nominal tracking is best with model-based or fuzzy control. Under shifts, residual RL becomes useful. Direct RL often fails, so the fuzzy supervisor matters.

## Slide 9: Numerical Takeaways

Highlight three numbers:

- Eval-gated Fuzzy-TD3 nearly matches fuzzy-only nominal tracking.
- Residual SAC is best under payload-2x.
- Residual SAC is close to PID under severe shift but much smoother.

## Slide 10: Safety

Explain that direct RL violates thresholds more often. The residual approach is safer because the learned component is bounded and supervised by a baseline controller.

## Slide 11: Torque Smoothness

Make the deployment argument: RMSE is not everything. A controller with rough torque can damage actuators or be undesirable in real systems. Residual RL can be smoother than PID-like baselines under shift.

## Slide 12: Gating Fixes the Main Weakness

This is the strongest conceptual slide. Always-active residual TD3 hurts nominal performance. The gate suppresses residual action in nominal operation and almost recovers fuzzy-only performance.

## Slide 13: Residual Authority Ablation

Explain the trade-off: increasing residual bound improves robustness under shifted dynamics but hurts nominal tracking. This is experimental evidence that gating is necessary.

## Slide 14: Ablation Table

Use the table if someone asks for exact values. Point out that severe-shift RMSE improves monotonically with residual authority, while nominal gets worse for all nonzero bounds.

## Slide 15: Representative Rollout

Use this to connect metrics to actual behavior over time. Mention that this helps reviewers see the transient response, not only summary statistics.

## Slide 16: Safety and Stability Framing

Do not overclaim. Say that bounded residual torque supports practical boundedness, but a full theorem would need CBF, Lyapunov-constrained RL, or passivity-constrained policy optimization.

## Slide 17: Limitations

This slide builds credibility. The work is not yet journal-ready for a top venue because 6-DOF validation is missing. Say that clearly before reviewers do.

## Slide 18: Next Experiments

Frame the next work as a direct path to a stronger paper: UR5/Panda/KUKA validation, learned gate, formal safety filter, more seeds, and latency tests.

## Slide 19: Final Takeaway

End with the clean final claim:

> Fuzzy-supervised residual RL is an uncertainty-triggered adaptation layer, not a universal controller replacement.

That sentence is the strongest and most defensible positioning for the research idea.
