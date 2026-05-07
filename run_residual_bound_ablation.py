from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / ".matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from controllers import FuzzySupervisor, make_fuzzy_env
from run_experiments import FIGURES, MODELS, RESULTS, ensure_dirs, evaluate_policy_or_controller
from two_link_env import SCENARIOS, TwoLinkManipulatorEnv


def train_bound_model(bound: float, seed: int, timesteps: int, force: bool) -> Path:
    ensure_dirs()
    ablation_dir = MODELS / "ablation_residual_bound"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    model_path = ablation_dir / f"fuzzy_td3_bound{bound:g}_seed{seed}.zip"
    if model_path.exists() and not force:
        return model_path

    env = DummyVecEnv(
        [
            lambda: make_fuzzy_env(
                "payload_2x",
                randomize_training=True,
                seed=seed,
                residual_limit=bound,
            )
        ]
    )
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.18 * np.ones(2))
    model = TD3(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=7e-4,
        buffer_size=80_000,
        learning_starts=1_000,
        batch_size=128,
        tau=0.01,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.12,
        target_noise_clip=0.35,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(model_path)
    return model_path


def evaluate_bounds(bounds: list[float], seeds: list[int], eval_episodes: int) -> pd.DataFrame:
    rows = []
    for bound in bounds:
        for seed in seeds:
            model_path = MODELS / "ablation_residual_bound" / f"fuzzy_td3_bound{bound:g}_seed{seed}.zip"
            model = TD3.load(model_path)
            for scenario in SCENARIOS:
                for episode in range(eval_episodes):
                    eval_seed = 80_000 + int(bound * 100) + seed * 20 + episode
                    env = make_fuzzy_env(scenario, seed=eval_seed, residual_limit=bound)
                    metrics, _ = evaluate_policy_or_controller(model, env, eval_seed)
                    metrics.update(
                        {
                            "scenario": scenario,
                            "controller": "Fuzzy-TD3 residual",
                            "residual_bound_nm": bound,
                            "model_seed": seed,
                            "eval_episode": episode,
                        }
                    )
                    rows.append(metrics)

    # Include fuzzy-only reference in the same table.
    fuzzy = FuzzySupervisor()
    for scenario in SCENARIOS:
        for seed in seeds:
            for episode in range(eval_episodes):
                eval_seed = 90_000 + seed * 20 + episode
                env = TwoLinkManipulatorEnv(scenario=scenario, supervisor=fuzzy)
                metrics, _ = evaluate_policy_or_controller(fuzzy, env, eval_seed)
                metrics.update(
                    {
                        "scenario": scenario,
                        "controller": "Fuzzy only",
                        "residual_bound_nm": 0.0,
                        "model_seed": seed,
                        "eval_episode": episode,
                    }
                )
                rows.append(metrics)

    df = pd.DataFrame(rows)
    out_dir = RESULTS / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "residual_bound_raw.csv", index=False)

    metric_cols = [
        "failure_penalized_rmse_rad",
        "rmse_rad",
        "safety_violation_rate",
        "torque_smoothness",
        "completed",
    ]
    summary = df.groupby(["scenario", "controller", "residual_bound_nm"], as_index=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    for metric in metric_cols:
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
    summary.to_csv(out_dir / "residual_bound_summary.csv", index=False)
    make_ablation_figures(df)
    return df


def make_ablation_figures(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"nominal": "#5E81AC", "payload_2x": "#D08770", "severe_shift": "#BF616A"}
    residual = df[df["controller"] == "Fuzzy-TD3 residual"].copy()

    for metric, ylabel, filename in [
        ("failure_penalized_rmse_rad", "Failure-penalized RMSE (rad)", "ablation_residual_bound_rmse.png"),
        ("safety_violation_rate", "Safety violation rate", "ablation_residual_bound_safety.png"),
        ("torque_smoothness", "Torque smoothness RMS", "ablation_residual_bound_smoothness.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.3, 4.4))
        for scenario, group in residual.groupby("scenario"):
            grouped = group.groupby("residual_bound_nm")[metric]
            mean = grouped.mean()
            sem = grouped.sem().fillna(0.0)
            ax.plot(mean.index, mean.values, marker="o", label=scenario, color=colors.get(scenario))
            ax.fill_between(mean.index, mean.values - 1.96 * sem.values, mean.values + 1.96 * sem.values, alpha=0.18)
        ax.set_xlabel("Residual torque bound (Nm)")
        ax.set_ylabel(ylabel)
        ax.set_title("Residual-bound sensitivity")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounds", type=float, nargs="+", default=[6.0, 10.0, 14.0, 18.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--timesteps", type=int, default=6000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not args.skip_train:
        for bound in args.bounds:
            for seed in args.seeds:
                train_bound_model(bound, seed, args.timesteps, args.force_train)
    evaluate_bounds(args.bounds, args.seeds, args.eval_episodes)
    print((RESULTS / "ablations" / "residual_bound_summary.csv").resolve())


if __name__ == "__main__":
    main()
