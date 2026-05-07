from __future__ import annotations

"""Optional 6/7-DOF PyBullet validation scaffold.

This script is intentionally dependency-gated: the current local environment
cannot install pybullet because Microsoft C++ Build Tools are missing. Once
pybullet is available, this provides a starting point for KUKA/Panda-style
joint-space validation using the same metrics as the two-link benchmark.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require_pybullet():
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise SystemExit(
            "pybullet is not installed. Install Microsoft C++ Build Tools or use "
            "a Python version with a prebuilt pybullet wheel, then run this script again."
        ) from exc
    return p, pybullet_data


def desired(t: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1, n + 1, dtype=float)
    qd = 0.35 * np.sin(0.45 * idx * t) + 0.10 * np.cos(0.20 * idx * t)
    dqd = 0.35 * 0.45 * idx * np.cos(0.45 * idx * t) - 0.10 * 0.20 * idx * np.sin(0.20 * idx * t)
    return qd, dqd


def fuzzy_pd_torque(q: np.ndarray, dq: np.ndarray, qd: np.ndarray, dqd: np.ndarray, torque_limit: float) -> np.ndarray:
    e = qd - q
    de = dqd - dq
    kp = np.linspace(55.0, 30.0, len(q))
    kd = np.linspace(7.0, 4.0, len(q))
    # Saturated fuzzy-like linguistic action: low residual authority near zero,
    # high correction under persistent large error.
    linguistic = np.tanh(e / 0.18) + 0.35 * np.tanh(de / 0.9)
    tau = kp * e + kd * de + 4.0 * linguistic
    return np.clip(tau, -torque_limit, torque_limit)


def run_kuka_validation(steps: int, dt: float, torque_limit: float, seed: int) -> pd.DataFrame:
    p, pybullet_data = require_pybullet()
    rng = np.random.default_rng(seed)
    client = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        p.loadURDF("plane.urdf")
        robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        joints = [
            j for j in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        ]
        n = len(joints)
        for j in joints:
            p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0.0)
            p.resetJointState(robot, j, targetValue=float(rng.normal(0.0, 0.02)), targetVelocity=0.0)

        rows = []
        for step in range(steps):
            t = step * dt
            states = p.getJointStates(robot, joints)
            q = np.array([s[0] for s in states], dtype=float)
            dq = np.array([s[1] for s in states], dtype=float)
            qd, dqd = desired(t, n)
            tau = fuzzy_pd_torque(q, dq, qd, dqd, torque_limit)
            if t > 2.5:
                tau += 0.8 * np.sin(7.0 * t + np.arange(n))
            for joint, command in zip(joints, tau):
                p.setJointMotorControl2(robot, joint, p.TORQUE_CONTROL, force=float(command))
            p.stepSimulation()
            error_norm = float(np.linalg.norm(qd - q))
            rows.append(
                {
                    "t": t,
                    "error_norm": error_norm,
                    "torque_rms": float(np.sqrt(np.mean(tau**2))),
                    "safety_violation": float(error_norm > 0.60),
                    "n_joints": n,
                }
            )
        return pd.DataFrame(rows)
    finally:
        p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--torque-limit", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("ims_fuzzy_td3_experiments/results/pybullet_kuka_validation.csv"))
    args = parser.parse_args()
    df = run_kuka_validation(args.steps, args.dt, args.torque_limit, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    rmse = float(np.sqrt(np.mean(df["error_norm"].to_numpy() ** 2)))
    print(f"Wrote {args.out} | RMSE={rmse:.4f} | safety={df['safety_violation'].mean():.4f}")


if __name__ == "__main__":
    main()

