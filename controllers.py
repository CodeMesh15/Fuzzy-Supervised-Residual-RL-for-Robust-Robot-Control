from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from two_link_env import ManipulatorParams, TwoLinkManipulatorEnv


Array = np.ndarray


def _triangular(x: float, center: float, width: float = 0.5) -> float:
    return max(0.0, 1.0 - abs(x - center) / width)


class FuzzySupervisor:
    """Mamdani-style fuzzy PD supervisor with nominal gravity compensation."""

    as_supervisor = True
    labels = ("NB", "NS", "Z", "PS", "PB")
    centers = {"NB": -1.0, "NS": -0.5, "Z": 0.0, "PS": 0.5, "PB": 1.0}
    output_values = {"NB": -1.0, "NS": -0.5, "Z": 0.0, "PS": 0.5, "PB": 1.0}
    rule_table = [
        ["NB", "NB", "NB", "NS", "Z"],
        ["NB", "NB", "NS", "Z", "PS"],
        ["NB", "NS", "Z", "PS", "PB"],
        ["NS", "Z", "PS", "PB", "PB"],
        ["Z", "PS", "PB", "PB", "PB"],
    ]

    def __init__(
        self,
        error_scale: Array | None = None,
        derror_scale: Array | None = None,
        feedback_limit: Array | None = None,
        full_torque_limit: float = 35.0,
    ) -> None:
        self.error_scale = np.asarray(error_scale if error_scale is not None else [0.55, 0.55], dtype=float)
        self.derror_scale = np.asarray(derror_scale if derror_scale is not None else [2.0, 2.0], dtype=float)
        self.feedback_limit = np.asarray(feedback_limit if feedback_limit is not None else [18.0, 12.0], dtype=float)
        self.full_torque_limit = float(full_torque_limit)
        self.nominal_env = TwoLinkManipulatorEnv()

    def __call__(self, q: Array, dq: Array, qd: Array, dqd: Array, t: float = 0.0) -> Array:
        e = TwoLinkManipulatorEnv.wrap_angle(qd - q)
        de = dqd - dq
        tau_fb = np.array([self._joint_output(e[i], de[i], i) for i in range(2)], dtype=float)
        tau = self.nominal_env.gravity_vector(q, payload_multiplier=1.0) + tau_fb
        return np.clip(tau, -self.full_torque_limit, self.full_torque_limit)

    def _joint_output(self, e: float, de: float, joint: int) -> float:
        en = float(np.clip(e / self.error_scale[joint], -1.0, 1.0))
        den = float(np.clip(de / self.derror_scale[joint], -1.0, 1.0))
        e_members = [_triangular(en, self.centers[label]) for label in self.labels]
        de_members = [_triangular(den, self.centers[label]) for label in self.labels]

        numerator = 0.0
        denominator = 0.0
        for i, e_mu in enumerate(e_members):
            for j, de_mu in enumerate(de_members):
                strength = min(e_mu, de_mu)
                if strength <= 0.0:
                    continue
                out_label = self.rule_table[i][j]
                numerator += strength * self.output_values[out_label]
                denominator += strength
        crisp = numerator / denominator if denominator else 0.0
        return crisp * self.feedback_limit[joint]


class TunedFuzzySupervisor(FuzzySupervisor):
    """Higher-authority fuzzy baseline used as a tuned fuzzy-only ablation."""

    def __init__(self) -> None:
        super().__init__(
            error_scale=np.array([0.45, 0.45], dtype=float),
            derror_scale=np.array([1.7, 1.7], dtype=float),
            feedback_limit=np.array([24.0, 17.0], dtype=float),
        )


@dataclass
class PIDController:
    """Gravity-compensated PID baseline."""

    kp: Array = field(default_factory=lambda: np.array([78.0, 52.0], dtype=float))
    kd: Array = field(default_factory=lambda: np.array([13.0, 9.0], dtype=float))
    ki: Array = field(default_factory=lambda: np.array([3.5, 2.0], dtype=float))
    full_torque_limit: float = 35.0
    integral_limit: float = 0.75

    def __post_init__(self) -> None:
        self.integral = np.zeros(2, dtype=float)
        self.nominal_env = TwoLinkManipulatorEnv()

    def reset(self) -> None:
        self.integral = np.zeros(2, dtype=float)

    def __call__(self, q: Array, dq: Array, qd: Array, dqd: Array, t: float = 0.0) -> Array:
        e = TwoLinkManipulatorEnv.wrap_angle(qd - q)
        de = dqd - dq
        self.integral = np.clip(self.integral + e * self.nominal_env.dt, -self.integral_limit, self.integral_limit)
        tau = self.nominal_env.gravity_vector(q, payload_multiplier=1.0)
        tau = tau + self.kp * e + self.kd * de + self.ki * self.integral
        return np.clip(tau, -self.full_torque_limit, self.full_torque_limit)


@dataclass
class RobustPIDController(PIDController):
    """PID plus saturated sliding-mode term for disturbance rejection."""

    lam: Array = field(default_factory=lambda: np.array([3.0, 2.5], dtype=float))
    ks: Array = field(default_factory=lambda: np.array([7.0, 5.0], dtype=float))
    boundary_layer: float = 0.12

    def __call__(self, q: Array, dq: Array, qd: Array, dqd: Array, t: float = 0.0) -> Array:
        e = TwoLinkManipulatorEnv.wrap_angle(qd - q)
        de = dqd - dq
        self.integral = np.clip(self.integral + e * self.nominal_env.dt, -self.integral_limit, self.integral_limit)
        sliding = de + self.lam * e
        robust = self.ks * np.tanh(sliding / self.boundary_layer)
        tau = self.nominal_env.gravity_vector(q, payload_multiplier=1.0)
        tau = tau + self.kp * e + self.kd * de + self.ki * self.integral + robust
        return np.clip(tau, -self.full_torque_limit, self.full_torque_limit)


@dataclass
class ComputedTorqueController:
    """Nominal inverse-dynamics baseline."""

    kp: Array = field(default_factory=lambda: np.array([45.0, 35.0], dtype=float))
    kd: Array = field(default_factory=lambda: np.array([10.0, 8.0], dtype=float))
    full_torque_limit: float = 35.0

    def __post_init__(self) -> None:
        self.nominal_env = TwoLinkManipulatorEnv()

    def reset(self) -> None:
        pass

    def __call__(self, q: Array, dq: Array, qd: Array, dqd: Array, t: float = 0.0) -> Array:
        _, _, ddqd = self.nominal_env.desired_trajectory(t)
        e = TwoLinkManipulatorEnv.wrap_angle(qd - q)
        de = dqd - dq
        v = ddqd + self.kd * de + self.kp * e
        tau = self.nominal_env.mass_matrix(q, payload_multiplier=1.0) @ v
        tau = tau + self.nominal_env.coriolis_vector(q, dq, payload_multiplier=1.0)
        tau = tau + self.nominal_env.gravity_vector(q, payload_multiplier=1.0)
        tau = tau + self.nominal_env.friction_vector(dq, friction_multiplier=1.0)
        return np.clip(tau, -self.full_torque_limit, self.full_torque_limit)


@dataclass
class AdaptiveComputedTorqueController(ComputedTorqueController):
    """Heuristic adaptive inverse-dynamics controller with online payload estimate."""

    adaptation_rate: float = 0.18
    leakage: float = 0.015
    payload_hat: float = 1.0

    def reset(self) -> None:
        self.payload_hat = 1.0

    def __call__(self, q: Array, dq: Array, qd: Array, dqd: Array, t: float = 0.0) -> Array:
        _, _, ddqd = self.nominal_env.desired_trajectory(t)
        e = TwoLinkManipulatorEnv.wrap_angle(qd - q)
        de = dqd - dq
        sliding = de + 2.0 * e
        excitation = float(np.clip(np.linalg.norm(sliding), 0.0, 1.5))
        if t > 2.0:
            self.payload_hat += self.nominal_env.dt * (self.adaptation_rate * excitation - self.leakage * (self.payload_hat - 1.0))
            self.payload_hat = float(np.clip(self.payload_hat, 1.0, 3.0))
        v = ddqd + self.kd * de + self.kp * e
        tau = self.nominal_env.mass_matrix(q, payload_multiplier=self.payload_hat) @ v
        tau = tau + self.nominal_env.coriolis_vector(q, dq, payload_multiplier=self.payload_hat)
        tau = tau + self.nominal_env.gravity_vector(q, payload_multiplier=self.payload_hat)
        tau = tau + self.nominal_env.friction_vector(dq, friction_multiplier=1.0)
        return np.clip(tau, -self.full_torque_limit, self.full_torque_limit)


def make_fuzzy_env(
    scenario: str,
    randomize_training: bool = False,
    seed: int | None = None,
    residual_limit: float = 14.0,
    residual_gate: str | None = None,
    gate_threshold: float = 0.22,
    passivity_filter: bool = False,
) -> TwoLinkManipulatorEnv:
    return TwoLinkManipulatorEnv(
        scenario=scenario,
        supervisor=FuzzySupervisor(),
        residual_limit=residual_limit,
        residual_gate=residual_gate,
        gate_threshold=gate_threshold,
        passivity_filter=passivity_filter,
        randomize_training=randomize_training,
        seed=seed,
    )


def make_direct_env(scenario: str, randomize_training: bool = False, seed: int | None = None) -> TwoLinkManipulatorEnv:
    return TwoLinkManipulatorEnv(
        scenario=scenario,
        supervisor=None,
        randomize_training=randomize_training,
        seed=seed,
    )
