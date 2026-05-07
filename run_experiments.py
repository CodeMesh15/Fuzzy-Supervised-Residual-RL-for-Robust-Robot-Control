from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
(ROOT / ".matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from controllers import (
    AdaptiveComputedTorqueController,
    ComputedTorqueController,
    FuzzySupervisor,
    PIDController,
    RobustPIDController,
    TunedFuzzySupervisor,
    make_direct_env,
    make_fuzzy_env,
)
from two_link_env import SCENARIOS, TwoLinkManipulatorEnv

RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
FIGURES = RESULTS / "figures"
RAW = RESULTS / "raw"


class LearningCurveCallback(BaseCallback):
    def __init__(self, eval_env_factory, controller_name: str, eval_freq: int, out_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_factory = eval_env_factory
        self.controller_name = controller_name
        self.eval_freq = int(eval_freq)
        self.out_path = out_path
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        metrics = []
        for ep_seed in range(3):
            env = self.eval_env_factory()
            row, _ = evaluate_policy_or_controller(self.model, env, seed=10_000 + ep_seed)
            metrics.append(row)
        mean_rmse = float(np.mean([m["rmse_rad"] for m in metrics]))
        mean_reward = float(np.mean([m["return"] for m in metrics]))
        self.rows.append(
            {
                "controller": self.controller_name,
                "timesteps": self.num_timesteps,
                "eval_rmse_rad": mean_rmse,
                "eval_return": mean_reward,
            }
        )
        pd.DataFrame(self.rows).to_csv(self.out_path, index=False)
        return True


def ensure_dirs() -> None:
    for path in (RESULTS, MODELS, FIGURES, RAW):
        path.mkdir(parents=True, exist_ok=True)


def train_model(kind: str, seed: int, timesteps: int, force: bool) -> Path:
    ensure_dirs()
    model_path = MODELS / f"{kind}_seed{seed}.zip"
    curve_path = RAW / f"learning_{kind}_seed{seed}.csv"
    if model_path.exists() and not force:
        return model_path

    if kind == "td3_direct":
        env_factory = lambda: make_direct_env("payload_2x", randomize_training=True, seed=seed)
        model_cls = TD3
        kwargs = {}
    elif kind == "fuzzy_td3":
        env_factory = lambda: make_fuzzy_env("payload_2x", randomize_training=True, seed=seed)
        model_cls = TD3
        kwargs = {}
    elif kind == "gated_fuzzy_td3":
        env_factory = lambda: make_fuzzy_env(
            "payload_2x",
            randomize_training=True,
            seed=seed,
            residual_gate="uncertainty",
            passivity_filter=True,
        )
        model_cls = TD3
        kwargs = {}
    elif kind == "sac_direct":
        env_factory = lambda: make_direct_env("payload_2x", randomize_training=True, seed=seed)
        model_cls = SAC
        kwargs = {"ent_coef": "auto"}
    elif kind == "fuzzy_sac":
        env_factory = lambda: make_fuzzy_env("payload_2x", randomize_training=True, seed=seed)
        model_cls = SAC
        kwargs = {"ent_coef": "auto"}
    elif kind == "ddpg_direct":
        env_factory = lambda: make_direct_env("payload_2x", randomize_training=True, seed=seed)
        model_cls = DDPG
        kwargs = {}
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    env = DummyVecEnv([env_factory])
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.18 * np.ones(2))
    common_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=7e-4,
        buffer_size=120_000,
        learning_starts=1_000,
        batch_size=128,
        tau=0.01,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )
    if model_cls in (TD3, DDPG):
        common_kwargs["action_noise"] = action_noise
    if model_cls is TD3:
        common_kwargs["policy_delay"] = 2
        common_kwargs["target_policy_noise"] = 0.12
        common_kwargs["target_noise_clip"] = 0.35

    model = model_cls(**common_kwargs, **kwargs)
    eval_factory = lambda: env_factory()
    callback = LearningCurveCallback(eval_factory, kind, max(1_000, timesteps // 10), curve_path)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(model_path)
    return model_path


def rollout_classical(controller, env: TwoLinkManipulatorEnv, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    rows = []
    total_reward = 0.0
    for _ in range(env.max_steps):
        q = env.state[:2].copy()
        dq = env.state[2:].copy()
        qd, dqd, _ = env.desired_trajectory(env.step_count * env.dt)
        tau = controller(q, dq, qd, dqd, env.step_count * env.dt)
        if env.supervisor is None:
            action = tau / env.full_torque_limit
        else:
            action = np.zeros(2, dtype=float)
        obs, reward, terminated, truncated, info = env.step(action)
        info["command_tau"] = tau.copy()
        rows.append(flatten_info(info))
        total_reward += reward
        if terminated or truncated:
            break
    df = pd.DataFrame(rows)
    return summarize_rollout(df, total_reward, horizon=env.horizon), df


def evaluate_policy_or_controller(model_or_controller, env: TwoLinkManipulatorEnv, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    if not hasattr(model_or_controller, "predict"):
        return rollout_classical(model_or_controller, env, seed)

    obs, _ = env.reset(seed=seed)
    rows = []
    total_reward = 0.0
    for _ in range(env.max_steps):
        action, _ = model_or_controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(flatten_info(info))
        total_reward += reward
        if terminated or truncated:
            break
    df = pd.DataFrame(rows)
    return summarize_rollout(df, total_reward, horizon=env.horizon), df


def flatten_info(info: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {
        "t": float(info["t"]),
        "error_norm": float(info["error_norm"]),
        "torque_rms": float(info["torque_rms"]),
        "torque_smoothness": float(info["torque_smoothness"]),
        "safety_violation": float(info["safety_violation"]),
        "residual_gate": float(info.get("residual_gate", 0.0)),
    }
    for prefix in ("q", "qd", "error", "tau", "supervisor_tau", "residual_tau"):
        values = np.asarray(info[prefix], dtype=float)
        for idx, value in enumerate(values, start=1):
            row[f"{prefix}{idx}"] = float(value)
    return row


def summarize_rollout(df: pd.DataFrame, total_reward: float, horizon: float = 6.0) -> dict[str, Any]:
    if df.empty:
        return {
            "rmse_rad": np.nan,
            "failure_penalized_rmse_rad": np.nan,
            "itae": np.nan,
            "peak_error_rad": np.nan,
            "torque_rms": np.nan,
            "torque_smoothness": np.nan,
            "safety_violation_rate": np.nan,
            "residual_gate_mean": np.nan,
            "completed": 0.0,
            "duration_s": 0.0,
            "return": total_reward,
        }
    err_cols = [c for c in df.columns if c.startswith("error") and c[-1].isdigit()]
    err_sq = np.sum(df[err_cols].to_numpy() ** 2, axis=1)
    dt = float(df["t"].diff().median()) if len(df) > 1 else 0.02
    duration = float(df["t"].max())
    missing_steps = max(0, int(round((horizon - duration) / dt)))
    safety_error_sq = 2.4**2
    penalized_err_sq = np.concatenate([err_sq, np.full(missing_steps, safety_error_sq)])
    return {
        "rmse_rad": float(np.sqrt(np.mean(err_sq))),
        "failure_penalized_rmse_rad": float(np.sqrt(np.mean(penalized_err_sq))),
        "itae": float(np.sum(df["t"].to_numpy() * np.sqrt(err_sq) * dt)),
        "peak_error_rad": float(np.max(np.sqrt(err_sq))),
        "torque_rms": float(df["torque_rms"].mean()),
        "torque_smoothness": float(df["torque_smoothness"].mean()),
        "safety_violation_rate": float(df["safety_violation"].mean()),
        "residual_gate_mean": float(df["residual_gate"].mean()) if "residual_gate" in df else 0.0,
        "completed": float(duration >= 0.98 * horizon),
        "duration_s": duration,
        "return": float(total_reward),
    }


def evaluate_all(seeds: list[int], eval_episodes: int) -> pd.DataFrame:
    rows = []
    controller_specs: list[tuple[str, Any, int]] = [
        ("PID", PIDController(), -1),
        ("Robust PID / SMC", RobustPIDController(), -1),
        ("Computed torque", ComputedTorqueController(), -1),
        ("Adaptive computed torque", AdaptiveComputedTorqueController(), -1),
        ("Fuzzy", FuzzySupervisor(), -1),
        ("Tuned fuzzy", TunedFuzzySupervisor(), -1),
    ]
    for model_path in MODELS.glob("*.zip"):
        model_seed = model_seed_from_stem(model_path.stem)
        if model_seed not in seeds:
            continue
        if model_path.name.startswith("td3_direct"):
            controller_specs.append((model_path.stem, TD3.load(model_path), model_seed))
        elif model_path.name.startswith("gated_fuzzy_td3"):
            controller_specs.append((model_path.stem, TD3.load(model_path), model_seed))
        elif model_path.name.startswith("fuzzy_td3"):
            model = TD3.load(model_path)
            controller_specs.append((model_path.stem, model, model_seed))
            controller_specs.append((f"eval_gated_{model_path.stem}", model, model_seed))
        elif model_path.name.startswith("sac_direct"):
            controller_specs.append((model_path.stem, SAC.load(model_path), model_seed))
        elif model_path.name.startswith("fuzzy_sac"):
            controller_specs.append((model_path.stem, SAC.load(model_path), model_seed))
        elif model_path.name.startswith("ddpg_direct"):
            controller_specs.append((model_path.stem, DDPG.load(model_path), model_seed))

    for scenario in SCENARIOS:
        for name, controller, model_seed in sorted(controller_specs, key=lambda item: item[0]):
            replicate_seeds = seeds if model_seed < 0 else [model_seed]
            for seed in replicate_seeds:
                for episode in range(eval_episodes):
                    eval_seed = 50_000 + seed * 100 + episode
                    if getattr(controller, "as_supervisor", False):
                        env = TwoLinkManipulatorEnv(scenario=scenario, supervisor=controller)
                    elif name.startswith("eval_gated_fuzzy_td3"):
                        env = make_fuzzy_env(
                            scenario=scenario,
                            seed=eval_seed,
                            residual_gate="quadratic",
                            gate_threshold=0.18,
                            passivity_filter=False,
                        )
                    elif name.startswith("gated_fuzzy_td3"):
                        env = make_fuzzy_env(
                            scenario=scenario,
                            seed=eval_seed,
                            residual_gate="uncertainty",
                            passivity_filter=True,
                        )
                    elif name.startswith("fuzzy_td3"):
                        env = make_fuzzy_env(scenario=scenario, seed=eval_seed)
                    elif name.startswith("fuzzy_sac"):
                        env = make_fuzzy_env(scenario=scenario, seed=eval_seed)
                    else:
                        env = make_direct_env(scenario=scenario, seed=eval_seed)
                    metrics, trace = evaluate_policy_or_controller(controller, env, eval_seed)
                    metrics.update(
                        {
                            "scenario": scenario,
                            "controller": canonical_controller_name(name),
                            "model_seed": model_seed,
                            "replicate_seed": seed,
                            "eval_episode": episode,
                        }
                    )
                    rows.append(metrics)
                    if scenario == "severe_shift" and episode == 0 and seed == seeds[0]:
                        trace_name = safe_name(canonical_controller_name(name))
                        trace.to_csv(RAW / f"trace_{trace_name}_seed{seed}_{scenario}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics_raw.csv", index=False)
    summary = summarize_metrics(df)
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    return df


def canonical_controller_name(name: str) -> str:
    if name.startswith("td3_direct"):
        return "TD3 direct"
    if name.startswith("gated_fuzzy_td3"):
        return "Gated Fuzzy-TD3"
    if name.startswith("eval_gated_fuzzy_td3"):
        return "Eval-gated Fuzzy-TD3"
    if name.startswith("fuzzy_td3"):
        return "Fuzzy-TD3 residual"
    if name.startswith("sac_direct"):
        return "SAC direct"
    if name.startswith("fuzzy_sac"):
        return "Fuzzy-SAC residual"
    if name.startswith("ddpg_direct"):
        return "DDPG direct"
    return name


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def model_seed_from_stem(stem: str) -> int:
    if "_seed" not in stem:
        return -1
    try:
        return int(stem.rsplit("_seed", 1)[1])
    except ValueError:
        return -1


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_rad",
        "failure_penalized_rmse_rad",
        "itae",
        "peak_error_rad",
        "torque_rms",
        "torque_smoothness",
        "safety_violation_rate",
        "residual_gate_mean",
        "completed",
        "duration_s",
        "return",
    ]
    grouped = df.groupby(["scenario", "controller"], as_index=False)[metric_cols]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    for metric in metric_cols:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_sem"]
    return summary


def make_figures(metrics: pd.DataFrame, seeds: list[int]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    order = [
        "PID",
        "Robust PID / SMC",
        "Computed torque",
        "Adaptive computed torque",
        "Fuzzy",
        "Tuned fuzzy",
        "DDPG direct",
        "SAC direct",
        "TD3 direct",
        "Fuzzy-SAC residual",
        "Fuzzy-TD3 residual",
        "Eval-gated Fuzzy-TD3",
        "Gated Fuzzy-TD3",
    ]
    colors = {
        "PID": "#4C566A",
        "Robust PID / SMC": "#2E3440",
        "Computed torque": "#8FBCBB",
        "Adaptive computed torque": "#88C0D0",
        "Fuzzy": "#5E81AC",
        "Tuned fuzzy": "#81A1C1",
        "DDPG direct": "#B48EAD",
        "SAC direct": "#A3BE8C",
        "TD3 direct": "#D08770",
        "Fuzzy-SAC residual": "#EBCB8B",
        "Fuzzy-TD3 residual": "#BF616A",
        "Eval-gated Fuzzy-TD3": "#006D77",
        "Gated Fuzzy-TD3": "#2E7D32",
    }

    for metric, ylabel, filename in [
        ("rmse_rad", "Joint RMSE (rad)", "rmse_by_scenario.png"),
        ("failure_penalized_rmse_rad", "Failure-penalized RMSE (rad)", "penalized_rmse_by_scenario.png"),
        ("safety_violation_rate", "Safety violation rate", "safety_by_scenario.png"),
        ("torque_smoothness", "Torque smoothness RMS", "torque_smoothness_by_scenario.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11.0, 5.4))
        scenarios = list(SCENARIOS)
        width = min(0.75 / max(1, len([c for c in order if c in set(metrics["controller"])])), 0.10)
        x = np.arange(len(scenarios))
        present = [c for c in order if c in set(metrics["controller"])]
        for i, controller in enumerate(present):
            means = []
            cis = []
            for scenario in scenarios:
                values = metrics[(metrics["scenario"] == scenario) & (metrics["controller"] == controller)][metric]
                means.append(values.mean())
                cis.append(1.96 * values.sem() if len(values) > 1 else 0.0)
            offset = (i - (len(present) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=cis, capsize=3, label=controller, color=colors.get(controller))
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, fontsize=8)
        ax.set_title(ylabel + " under payload/friction shifts")
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=220)
        plt.close(fig)

    learning_files = [
        path for path in sorted(RAW.glob("learning_*.csv")) if model_seed_from_stem(path.stem) in seeds
    ]
    if learning_files:
        curves = pd.concat([pd.read_csv(path) for path in learning_files], ignore_index=True)
        curves["controller"] = curves["controller"].map(canonical_controller_name)
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for controller, group in curves.groupby("controller"):
            grouped = group.groupby("timesteps")["eval_rmse_rad"]
            mean = grouped.mean()
            sem = grouped.sem().fillna(0.0)
            ax.plot(mean.index, mean.values, label=controller, color=colors.get(controller))
            ax.fill_between(mean.index, mean.values - 1.96 * sem.values, mean.values + 1.96 * sem.values, alpha=0.18)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Evaluation RMSE (rad)")
        ax.set_title("Sample-efficiency curve")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(FIGURES / "learning_curve.png", dpi=220)
        plt.close(fig)

    trace_files = [
        path
        for path in sorted(RAW.glob("trace_*_seed*_severe_shift.csv"))
        if any(f"seed{seed}_" in path.name for seed in seeds)
    ]
    if trace_files:
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)
        for path in trace_files:
            name = path.stem.replace("trace_", "").replace("_severe_shift", "").replace("_", " ")
            trace = pd.read_csv(path)
            axes[0].plot(trace["t"], trace["error_norm"], label=name)
            axes[1].plot(trace["t"], trace["torque_rms"], label=name)
        axes[0].set_ylabel("Error norm (rad)")
        axes[1].set_ylabel("Torque RMS (Nm)")
        axes[1].set_xlabel("Time (s)")
        axes[0].set_title("Representative severe-shift rollout")
        axes[0].legend(frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "severe_shift_rollout.png", dpi=220)
        plt.close(fig)


def write_significance_tests(metrics: pd.DataFrame) -> None:
    try:
        from scipy import stats
    except Exception:
        return

    targets = ["Eval-gated Fuzzy-TD3", "Gated Fuzzy-TD3", "Fuzzy-TD3 residual", "Fuzzy-SAC residual"]
    baselines = [name for name in sorted(metrics["controller"].unique()) if name not in targets]
    rows = []
    for scenario in sorted(metrics["scenario"].unique()):
        scenario_df = metrics[metrics["scenario"] == scenario]
        for target in targets:
            target_values = scenario_df[scenario_df["controller"] == target]["failure_penalized_rmse_rad"].dropna()
            if len(target_values) < 2:
                continue
            for baseline in baselines:
                baseline_values = scenario_df[scenario_df["controller"] == baseline]["failure_penalized_rmse_rad"].dropna()
                if len(baseline_values) < 2:
                    continue
                t_stat, p_value = stats.ttest_ind(target_values, baseline_values, equal_var=False)
                rows.append(
                    {
                        "scenario": scenario,
                        "target": target,
                        "baseline": baseline,
                        "target_mean": float(target_values.mean()),
                        "baseline_mean": float(baseline_values.mean()),
                        "delta_percent": float((baseline_values.mean() - target_values.mean()) / baseline_values.mean() * 100.0),
                        "welch_t": float(t_stat),
                        "p_value": float(p_value),
                    }
                )
    out_dir = RESULTS / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "pairwise_welch_tests.csv", index=False)


def write_experiment_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "description": "Two-link manipulator benchmark for safe residual Fuzzy-TD3 control.",
        "arguments": vars(args),
        "scenarios": {name: asdict(config) for name, config in SCENARIOS.items()},
        "metrics": [
            "rmse_rad",
            "failure_penalized_rmse_rad",
            "itae",
            "peak_error_rad",
            "torque_rms",
            "torque_smoothness",
            "safety_violation_rate",
            "residual_gate_mean",
            "completed",
            "duration_s",
            "return",
        ],
    }
    (RESULTS / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--include-sac", action="store_true")
    parser.add_argument("--include-ddpg", action="store_true")
    parser.add_argument("--include-residual-sac", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_experiment_manifest(args)
    if not args.skip_train:
        for seed in args.seeds:
            train_model("td3_direct", seed, args.timesteps, args.force_train)
            train_model("fuzzy_td3", seed, args.timesteps, args.force_train)
            if args.include_gated:
                train_model("gated_fuzzy_td3", seed, args.timesteps, args.force_train)
            if args.include_sac:
                train_model("sac_direct", seed, args.timesteps, args.force_train)
            if args.include_residual_sac:
                train_model("fuzzy_sac", seed, args.timesteps, args.force_train)
            if args.include_ddpg:
                train_model("ddpg_direct", seed, args.timesteps, args.force_train)
    metrics = evaluate_all(args.seeds, args.eval_episodes)
    write_significance_tests(metrics)
    make_figures(metrics, args.seeds)
    print((RESULTS / "metrics_summary.csv").resolve())


if __name__ == "__main__":
    main()
