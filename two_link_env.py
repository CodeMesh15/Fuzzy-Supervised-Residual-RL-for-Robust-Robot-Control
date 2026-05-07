from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Array = np.ndarray


@dataclass(frozen=True)
class ManipulatorParams:
    """Nominal physical parameters for a two-link planar manipulator."""

    l1: float = 1.0
    l2: float = 0.8
    lc1: float = 0.5
    lc2: float = 0.4
    m1: float = 1.8
    m2: float = 1.2
    i1: float = 0.18
    i2: float = 0.10
    gravity: float = 9.81
    viscous: tuple[float, float] = (0.08, 0.06)
    coulomb: tuple[float, float] = (0.18, 0.12)


@dataclass(frozen=True)
class DisturbanceConfig:
    """Disturbance scenario used for train/evaluation domain shifts."""

    name: str = "nominal"
    payload_multiplier: float = 1.0
    friction_multiplier: float = 1.0
    disturbance_amplitude: float = 0.0
    starts_at: float = 2.5


SCENARIOS: dict[str, DisturbanceConfig] = {
    "nominal": DisturbanceConfig("nominal", 1.0, 1.0, 0.0),
    "payload_2x": DisturbanceConfig("payload_2x", 2.0, 1.25, 0.40),
    "severe_shift": DisturbanceConfig("severe_shift", 3.0, 1.75, 0.80),
}


class TwoLinkManipulatorEnv(gym.Env):
    """Continuous-torque trajectory tracking environment.

    The environment implements standard two-link manipulator dynamics. For
    direct RL, the action is the full torque. For residual RL, pass a
    supervisor callback; the action is then a bounded residual torque added to
    the supervisor torque.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: DisturbanceConfig | str = "nominal",
        supervisor: Callable[[Array, Array, Array, Array, float], Array] | None = None,
        residual_limit: float = 14.0,
        residual_gate: str | None = None,
        gate_threshold: float = 0.22,
        gate_slope: float = 12.0,
        passivity_filter: bool = False,
        full_torque_limit: float = 35.0,
        dt: float = 0.02,
        horizon: float = 6.0,
        seed: int | None = None,
        randomize_training: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(scenario, str):
            scenario = SCENARIOS[scenario]
        self.params = ManipulatorParams()
        self.scenario = scenario
        self.supervisor = supervisor
        self.residual_limit = float(residual_limit)
        self.residual_gate = residual_gate
        self.gate_threshold = float(gate_threshold)
        self.gate_slope = float(gate_slope)
        self.passivity_filter = bool(passivity_filter)
        self.full_torque_limit = float(full_torque_limit)
        self.dt = float(dt)
        self.horizon = float(horizon)
        self.max_steps = int(round(horizon / dt))
        self.randomize_training = bool(randomize_training)
        self.rng = np.random.default_rng(seed)

        high = np.array(
            [
                np.pi,
                np.pi,
                8.0,
                8.0,
                np.pi,
                np.pi,
                8.0,
                8.0,
                np.pi,
                np.pi,
                8.0,
                8.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.state = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.prev_tau = np.zeros(2, dtype=np.float64)
        self.error_ema = 0.0
        self.payload_multiplier = scenario.payload_multiplier
        self.friction_multiplier = scenario.friction_multiplier
        self.disturbance_phase = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.prev_tau = np.zeros(2, dtype=np.float64)
        self.error_ema = 0.0
        self.disturbance_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        if self.randomize_training:
            self.payload_multiplier = float(self.rng.uniform(1.0, 2.6))
            self.friction_multiplier = float(self.rng.uniform(0.9, 1.8))
        else:
            self.payload_multiplier = self.scenario.payload_multiplier
            self.friction_multiplier = self.scenario.friction_multiplier

        qd, dqd, _ = self.desired_trajectory(0.0)
        q = qd + self.rng.normal(0.0, 0.04, size=2)
        dq = dqd + self.rng.normal(0.0, 0.03, size=2)
        self.state = np.concatenate([q, dq]).astype(np.float64)
        return self._obs(), {}

    def step(self, action: Array):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        t = self.step_count * self.dt
        q = self.state[:2]
        dq = self.state[2:]
        qd, dqd, _ = self.desired_trajectory(t)

        if self.supervisor is None:
            tau = action * self.full_torque_limit
            residual_tau = tau.copy()
            supervisor_tau = np.zeros(2, dtype=np.float64)
            gate = 1.0
        else:
            supervisor_tau = self.supervisor(q, dq, qd, dqd, t)
            pre_gate_residual_tau = action * self.residual_limit
            gate = self.compute_residual_gate(qd - q, dqd - dq)
            residual_tau = gate * pre_gate_residual_tau
            if self.passivity_filter:
                residual_tau = self.apply_passivity_filter(residual_tau, dqd - dq)
            tau = supervisor_tau + residual_tau
        tau = np.clip(tau, -self.full_torque_limit, self.full_torque_limit)

        qdd = self.forward_dynamics(q, dq, tau, t)
        dq_next = np.clip(dq + qdd * self.dt, -8.0, 8.0)
        q_next = self.wrap_angle(q + dq_next * self.dt)
        self.state = np.concatenate([q_next, dq_next])

        qd_next, dqd_next, _ = self.desired_trajectory(t + self.dt)
        e = self.wrap_angle(qd_next - q_next)
        de = dqd_next - dq_next
        err_norm = float(np.linalg.norm(e))
        self.error_ema = 0.98 * self.error_ema + 0.02 * err_norm
        torque_rms = float(np.sqrt(np.mean(tau**2)))
        smooth = float(np.sqrt(np.mean(((tau - self.prev_tau) / self.dt) ** 2)))
        self.prev_tau = tau.copy()

        effort_tau = residual_tau if self.supervisor is not None else tau
        reward = (
            -10.0 * float(np.dot(e, e))
            -0.30 * float(np.dot(de, de))
            -0.003 * float(np.dot(effort_tau, effort_tau))
            -0.00003 * smooth
        )
        if err_norm < 0.05:
            reward += 0.50
        if err_norm > 0.75:
            reward -= 8.0

        self.step_count += 1
        terminated = bool(err_norm > 2.4)
        truncated = self.step_count >= self.max_steps
        info = {
            "t": t + self.dt,
            "q": q_next.copy(),
            "dq": dq_next.copy(),
            "qd": qd_next.copy(),
            "dqd": dqd_next.copy(),
            "error": e.copy(),
            "tau": tau.copy(),
            "supervisor_tau": supervisor_tau.copy(),
            "residual_tau": residual_tau.copy(),
            "residual_gate": float(gate),
            "error_norm": err_norm,
            "torque_rms": torque_rms,
            "torque_smoothness": smooth,
            "safety_violation": float(err_norm > 0.60),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> Array:
        t = self.step_count * self.dt
        q = self.state[:2]
        dq = self.state[2:]
        qd, dqd, _ = self.desired_trajectory(t)
        e = self.wrap_angle(qd - q)
        de = dqd - dq
        obs = np.concatenate(
            [
                q,
                dq,
                qd,
                dqd,
                e,
                de,
                [np.sin(2.0 * np.pi * t / self.horizon), np.cos(2.0 * np.pi * t / self.horizon)],
            ]
        )
        return obs.astype(np.float32)

    def desired_trajectory(self, t: float) -> tuple[Array, Array, Array]:
        qd = np.array(
            [
                0.55 * np.sin(0.85 * t) + 0.18 * np.sin(1.65 * t),
                0.45 * np.cos(0.65 * t) - 0.22 * np.sin(1.20 * t),
            ],
            dtype=np.float64,
        )
        dqd = np.array(
            [
                0.55 * 0.85 * np.cos(0.85 * t) + 0.18 * 1.65 * np.cos(1.65 * t),
                -0.45 * 0.65 * np.sin(0.65 * t) - 0.22 * 1.20 * np.cos(1.20 * t),
            ],
            dtype=np.float64,
        )
        ddqd = np.array(
            [
                -0.55 * 0.85**2 * np.sin(0.85 * t) - 0.18 * 1.65**2 * np.sin(1.65 * t),
                -0.45 * 0.65**2 * np.cos(0.65 * t) + 0.22 * 1.20**2 * np.sin(1.20 * t),
            ],
            dtype=np.float64,
        )
        return qd, dqd, ddqd

    def forward_dynamics(self, q: Array, dq: Array, tau: Array, t: float) -> Array:
        m2_mult = self.payload_multiplier if t >= self.scenario.starts_at else 1.0
        friction_mult = self.friction_multiplier if t >= self.scenario.starts_at else 1.0
        m = self.mass_matrix(q, payload_multiplier=m2_mult)
        c = self.coriolis_vector(q, dq, payload_multiplier=m2_mult)
        g = self.gravity_vector(q, payload_multiplier=m2_mult)
        f = self.friction_vector(dq, friction_multiplier=friction_mult)
        d = self.external_disturbance(t)
        return np.linalg.solve(m, tau + d - c - g - f)

    def mass_matrix(self, q: Array, payload_multiplier: float = 1.0) -> Array:
        p = self.params
        m2 = p.m2 * payload_multiplier
        i2 = p.i2 * payload_multiplier
        q2 = q[1]
        m11 = (
            p.i1
            + i2
            + p.m1 * p.lc1**2
            + m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * np.cos(q2))
        )
        m12 = i2 + m2 * (p.lc2**2 + p.l1 * p.lc2 * np.cos(q2))
        m22 = i2 + m2 * p.lc2**2
        return np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    def coriolis_vector(self, q: Array, dq: Array, payload_multiplier: float = 1.0) -> Array:
        p = self.params
        m2 = p.m2 * payload_multiplier
        h = -m2 * p.l1 * p.lc2 * np.sin(q[1])
        return np.array(
            [
                h * (2.0 * dq[0] * dq[1] + dq[1] ** 2),
                -h * dq[0] ** 2,
            ],
            dtype=np.float64,
        )

    def gravity_vector(self, q: Array, payload_multiplier: float = 1.0) -> Array:
        p = self.params
        m2 = p.m2 * payload_multiplier
        return np.array(
            [
                (p.m1 * p.lc1 + m2 * p.l1) * p.gravity * np.cos(q[0])
                + m2 * p.lc2 * p.gravity * np.cos(q[0] + q[1]),
                m2 * p.lc2 * p.gravity * np.cos(q[0] + q[1]),
            ],
            dtype=np.float64,
        )

    def friction_vector(self, dq: Array, friction_multiplier: float = 1.0) -> Array:
        p = self.params
        viscous = friction_multiplier * np.array(p.viscous, dtype=np.float64)
        coulomb = friction_multiplier * np.array(p.coulomb, dtype=np.float64)
        return viscous * dq + coulomb * np.tanh(dq / 0.05)

    def external_disturbance(self, t: float) -> Array:
        if t < self.scenario.starts_at:
            return np.zeros(2, dtype=np.float64)
        amp = self.scenario.disturbance_amplitude
        return amp * np.array(
            [
                np.sin(8.0 * t + self.disturbance_phase),
                0.65 * np.cos(6.0 * t + 0.5 * self.disturbance_phase),
            ],
            dtype=np.float64,
        )

    def compute_residual_gate(self, e: Array, de: Array) -> float:
        if self.residual_gate is None:
            return 1.0
        e = self.wrap_angle(e)
        err_norm = float(np.linalg.norm(e))
        rate_norm = float(np.linalg.norm(de))
        if self.residual_gate == "uncertainty":
            score = 0.70 * err_norm + 0.18 * min(rate_norm / 2.0, 1.0) + 0.12 * self.error_ema
            z = np.clip(self.gate_slope * (score - self.gate_threshold), -40.0, 40.0)
            return float(1.0 / (1.0 + np.exp(-z)))
        if self.residual_gate == "hard":
            return float(err_norm > self.gate_threshold)
        if self.residual_gate == "quadratic":
            score = (err_norm / max(self.gate_threshold, 1e-6)) ** 2
            score += 0.20 * (self.error_ema / max(self.gate_threshold, 1e-6)) ** 2
            score += 0.05 * min(rate_norm / 2.0, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        raise ValueError(f"Unknown residual gate: {self.residual_gate}")

    @staticmethod
    def apply_passivity_filter(residual_tau: Array, de: Array, damping_margin: float = 0.05) -> Array:
        """Project residual torque so it cannot inject excessive tracking-error power."""
        de = np.asarray(de, dtype=np.float64)
        residual_tau = np.asarray(residual_tau, dtype=np.float64)
        denom = float(np.dot(de, de))
        if denom < 1e-9:
            return residual_tau
        power = float(np.dot(de, residual_tau))
        allowed = damping_margin * denom
        if power <= allowed:
            return residual_tau
        return residual_tau - ((power - allowed) / denom) * de

    @staticmethod
    def wrap_angle(x: Array) -> Array:
        return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi
