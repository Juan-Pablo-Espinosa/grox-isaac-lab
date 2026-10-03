# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Electrical power model of NOVA's 16 RobStride motors (energy cost of walking).

Per motor m:  P_m = max(tau_m * omega_m + P_cu(tau_m), 0)   [W]   (no regeneration credit, not across motors either)
              P_cu = 3 * I_rms^2 * R_phase = 1.5 * R_terminal * (tau_m / Kt)^2
  * Kt [N·m/A_rms] is the datasheet torque constant at the OUTPUT shaft (checked: rated phase current
    I_pk / sqrt(2) * Kt reproduces the rated torque for every model), so tau_m is the module's output torque and
    I = tau_m / Kt is the RMS phase current. Using A_rms (not A_pk) means there is no extra sqrt(2) factor.
  * R_terminal is the line-to-line (terminal) resistance; for a wye winding R_phase = R_terminal / 2, which gives
    the 3 * 1/2 = 1.5 factor.
  * Gearbox friction, no-load current and driver losses are NOT modelled.

Motor assignment (sim joint -> motor):
  * Hip_Pitch_*, Lowerleg_Pitch_*: RobStride 04; Hip_Roll_*: RobStride 03; Upperleg_Yaw_*: RobStride 06 -- tau = the
    joint's applied torque, omega = joint velocity.
  * Ankle: 2x RobStride 02 per side in a parallel linkage with ratios (N_p, N_r):
    tau_m1,2 = (tau_pitch / N_p +/- tau_roll / N_r) / 2, omega_m1,2 = N_p omega_pitch +/- N_r omega_roll
    (power-consistent: sum tau_m * omega_m = tau_pitch * omega_pitch + tau_roll * omega_roll). The env cfg attribute
    ``ankle_transmission = (N_p, N_r)`` selects it; without it (walking task, runs 4/5) the "direct" model (1, 1) is
    used, i.e. tau_m1,2 = (tau_pitch +/- tau_roll) / 2 -- unchanged from run 5.
  * Prismatics: RobStride 00 driving a T12x8 leadscrew (8 mm lead) directly. F = applied joint force [N], screw
    speed v = commanded target rate of the prismatic action term (PhysX reports phantom prismatic joint velocity).
    Driving (F * v > 0): tau_m = |F| * lead / (2 pi eta_drive), mechanical power +|F v| / eta_drive.
    Holding / backdriven (F * v <= 0): tau_m = |F| * lead * eta_back / (2 pi), mechanical power -|F v| * eta_back.
    omega_m = 2 pi |v| / lead. eta_drive = 0.58, eta_back = 0.32 (trapezoidal screw with bronze nut).

This module must stay Kit-free at import time (walking_env.py imports it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass(frozen=True)
class MotorSpec:
    """Datasheet constants of one RobStride module (output-shaft quantities)."""

    kt: float
    """Torque constant at the output shaft [N·m/A_rms]."""
    r_terminal: float
    """Terminal (line-to-line) winding resistance [Ohm]."""
    rated_torque: float
    """Rated (continuous) output torque [N·m]."""
    peak_torque: float
    """Peak output torque [N·m]."""
    rated_speed_rpm: float
    """Output speed at rated torque [rpm]."""
    no_load_speed_rpm: float = float("nan")
    """Output no-load speed at 48 V [rpm] (the module's maximum speed)."""

    @property
    def no_load_speed(self) -> float:
        """Output no-load speed [rad/s]."""
        return self.no_load_speed_rpm * 2.0 * math.pi / 60.0

    @property
    def rated_power(self) -> float:
        """Rated mechanical output power = rated torque x rated speed [W]."""
        return self.rated_torque * self.rated_speed_rpm * 2.0 * math.pi / 60.0


# Sources: RobStride / AIFITLAB product spec tables and user manuals (see the task report). R for RS06 and RS00 is not
# published: estimated from the rated copper loss per module mass of the three models that publish R (RS02 53, RS03
# 48, RS04 62 W/kg -> mean 54 W/kg): R = 54 * mass / (1.5 * I_rated_rms^2).
ROBSTRIDE = {
    "RS00": MotorSpec(1.48, 1.0, 5.0, 14.0, rated_speed_rpm=100.0, no_load_speed_rpm=315.0),  # R estimate
    "RS02": MotorSpec(1.22, 0.58, 6.0, 17.0, rated_speed_rpm=360.0, no_load_speed_rpm=410.0),
    "RS03": MotorSpec(2.36, 0.39, 20.0, 60.0, rated_speed_rpm=180.0, no_load_speed_rpm=195.0),
    "RS04": MotorSpec(2.1, 0.16, 40.0, 120.0, rated_speed_rpm=167.0, no_load_speed_rpm=200.0),
    "RS06": MotorSpec(1.09, 0.22, 11.0, 36.0, rated_speed_rpm=100.0, no_load_speed_rpm=480.0),  # R estimate
}
LEADSCREW_LEAD = 0.008
"""T12x8 lead [m/rev]."""
LEADSCREW_ETA_DRIVE = 0.58
"""Leadscrew efficiency when the motor drives the load."""
LEADSCREW_ETA_BACK = 0.32
"""Leadscrew efficiency when the load backdrives the screw (holding / lowering)."""

# (motor label, model, kind, joint name(s)); kind: "rev" direct, "ankle+" / "ankle-" linkage, "screw" leadscrew
_MOTORS = [
    *[(f"hip_pitch_{s}", "RS04", "rev", f"Hip_Pitch_{s}_Joint") for s in ("Left", "Right")],
    *[(f"knee_{s}", "RS04", "rev", f"Lowerleg_Pitch_{s}_Joint") for s in ("Left", "Right")],
    *[(f"hip_roll_{s}", "RS03", "rev", f"Hip_Roll_{s}_Joint") for s in ("Left", "Right")],
    *[(f"hip_yaw_{s}", "RS06", "rev", f"Upperleg_Yaw_{s}_Joint") for s in ("Left", "Right")],
    *[
        (f"ankle_{s}_{k}", "RS02", f"ankle{sgn}", (f"Feet_Pitch_{s}_Joint", f"Feet_Roll_{s}_Joint"))
        for s in ("Left", "Right")
        for k, sgn in (("m1", "+"), ("m2", "-"))
    ],
    *[
        (f"{seg}_prismatic_{s}", "RS00", "screw", f"{seg.capitalize()}leg_Prismatic_{s}_Joint")
        for seg in ("upper", "lower")
        for s in ("Left", "Right")
    ],
]
MOTOR_GROUPS = {
    "hip_pitch": "RS04",
    "knee": "RS04",
    "hip_roll": "RS03",
    "hip_yaw": "RS06",
    "ankle": "RS02",
    "prismatic": "RS00",
}
"""Group label prefix -> motor model (for breakdowns)."""

DASHBOARD_GROUPS = ("hip_knee", "roll", "yaw", "ankle", "leadscrew")
"""Coarse motor groups for dashboards / recordings / metrics."""


def dashboard_group(label: str) -> str:
    """Coarse group of a motor label (see :data:`DASHBOARD_GROUPS`)."""
    if label.startswith(("hip_pitch", "knee")):
        return "hip_knee"
    if label.startswith("hip_roll"):
        return "roll"
    if label.startswith("hip_yaw"):
        return "yaw"
    if label.startswith("ankle"):
        return "ankle"
    return "leadscrew"


PRISMATIC_DRIVER = "prismatic_driver"
"""Pass as ``action_term_name`` to take the leadscrew rate from the env's external prismatic driver."""


def _rate_source(env: ManagerBasedRLEnv, name: str):
    """Object exposing ``joint_ids`` and ``target_rate`` of the prismatics: an action term or the external driver."""
    return env.prismatic_driver if name == PRISMATIC_DRIVER else env.action_manager.get_term(name)


class ElectricalPowerModel:
    """Vectorized per-motor power of the 16 motors; built once per env (joint indices resolved by name)."""

    def __init__(
        self, env: ManagerBasedRLEnv, action_term_name: str, ankle_transmission: tuple[float, float] = (1.0, 1.0)
    ):
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        self.action_term_name = action_term_name
        self.labels = [m[0] for m in _MOTORS]
        self.models = [m[1] for m in _MOTORS]
        dev = env.device
        term = _rate_source(env, action_term_name)
        # prismatic joints in action-term order -> column of target_rate
        term_cols = {robot.joint_names[j]: c for c, j in enumerate(term.joint_ids)}
        self._direct = [i for i, m in enumerate(_MOTORS) if m[2] == "rev"]
        self._direct_j = torch.tensor([names.index(_MOTORS[i][3]) for i in self._direct], device=dev)
        self._ankle = [i for i, m in enumerate(_MOTORS) if m[2].startswith("ankle")]
        self._ankle_pitch_j = torch.tensor([names.index(_MOTORS[i][3][0]) for i in self._ankle], device=dev)
        self._ankle_roll_j = torch.tensor([names.index(_MOTORS[i][3][1]) for i in self._ankle], device=dev)
        self._ankle_sign = torch.tensor([1.0 if _MOTORS[i][2] == "ankle+" else -1.0 for i in self._ankle], device=dev)
        self.ankle_transmission = (float(ankle_transmission[0]), float(ankle_transmission[1]))
        self._screw = [i for i, m in enumerate(_MOTORS) if m[2] == "screw"]
        self._screw_j = torch.tensor([names.index(_MOTORS[i][3]) for i in self._screw], device=dev)
        self._screw_col = torch.tensor([term_cols[_MOTORS[i][3]] for i in self._screw], device=dev)
        # per-motor copper coefficient 1.5 * R / Kt^2 [W / (N·m)^2], in motor order
        self._cu = torch.tensor([1.5 * ROBSTRIDE[m].r_terminal / ROBSTRIDE[m].kt ** 2 for m in self.models], device=dev)
        self._order = torch.tensor(self._direct + self._ankle + self._screw, device=dev)
        self._inv = torch.argsort(self._order)
        self._cache_step = -1
        self._cache = None

    def compute(self, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-motor (P, mechanical part P - P_cu, P_cu, |tau_m|) [W, W, W, N·m], each of shape (num_envs, 16).

        Cached per env step (the reward and the step metrics share one evaluation).
        """
        if self._cache_step == env.common_step_counter and self._cache is not None:
            return self._cache
        robot = env.scene["robot"]
        tau_all = robot.data.applied_torque.torch
        qd_all = robot.data.joint_vel.torch
        # direct revolute motors
        tau_d, w_d = tau_all[:, self._direct_j], qd_all[:, self._direct_j]
        mech_d = tau_d * w_d
        # ankle linkage (approximation, see module docstring)
        tp, tr = tau_all[:, self._ankle_pitch_j], tau_all[:, self._ankle_roll_j]
        wp, wr = qd_all[:, self._ankle_pitch_j], qd_all[:, self._ankle_roll_j]
        n_p, n_r = self.ankle_transmission
        tau_a = 0.5 * (tp / n_p + self._ankle_sign * tr / n_r)
        mech_a = tau_a * (n_p * wp + self._ankle_sign * n_r * wr)
        # leadscrews
        force = tau_all[:, self._screw_j]
        v = _rate_source(env, self.action_term_name).target_rate[:, self._screw_col]
        fv = force * v
        driving = fv > 0.0
        k = LEADSCREW_LEAD / (2.0 * math.pi)
        tau_s = torch.where(driving, force.abs() * k / LEADSCREW_ETA_DRIVE, force.abs() * k * LEADSCREW_ETA_BACK)
        mech_s = torch.where(driving, fv.abs() / LEADSCREW_ETA_DRIVE, -fv.abs() * LEADSCREW_ETA_BACK)
        # back to motor order
        tau_m = torch.cat([tau_d, tau_a, tau_s], dim=1)[:, self._inv]
        mech = torch.cat([mech_d, mech_a, mech_s], dim=1)[:, self._inv]
        cu = self._cu * tau_m**2
        p = (mech + cu).clamp(min=0.0)
        self._cache = (p, p - cu, cu, tau_m.abs())
        self._cache_step = env.common_step_counter
        return self._cache


def power_model(env: ManagerBasedRLEnv, action_term_name: str = "prismatic") -> ElectricalPowerModel:
    """The env's (cached) :class:`ElectricalPowerModel`."""
    model = getattr(env, "_nova_power_model", None)
    if model is None or model.action_term_name != action_term_name:
        transmission = getattr(env.cfg, "ankle_transmission", None) or (1.0, 1.0)
        model = ElectricalPowerModel(env, action_term_name, transmission)
        env._nova_power_model = model
    return model


def electrical_power(env: ManagerBasedRLEnv, action_term_name: str = "prismatic") -> torch.Tensor:
    """Total electrical power P_elec = sum over the 16 motors of max(tau*omega + P_cu, 0) [W], shape (num_envs,).

    Returned positive (use a negative weight).
    """
    return power_model(env, action_term_name).compute(env)[0].sum(dim=1)
