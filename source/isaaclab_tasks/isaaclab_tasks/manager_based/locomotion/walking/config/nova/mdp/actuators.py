# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit actuator models matching NOVA's RobStride hardware (configs and derivations in mdp/hardware.py).

Both extend the installed :class:`isaaclab.actuators.IdealPDActuator`: the PD torque
kp (q* - q) + kd (qd* - qd) + tau_ff is computed every physics step (``Articulation.write_data_to_sim`` runs once per
decimation substep and the joint state is refreshed every substep), then saturated by the hardware model and applied
as a joint effort (PhysX drive stiffness / damping are zeroed for explicit groups).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import IdealPDActuator

if TYPE_CHECKING:
    from isaaclab.utils.types import ArticulationActions

    from .hardware import AnkleLinkagePDActuatorCfg, TorqueSpeedPDActuatorCfg


class _Envelope:
    """Piecewise-linear torque limit vs |speed| (motoring quadrant)."""

    def __init__(self, points: Sequence[tuple[float, float]], device: str):
        pts = sorted(points)
        self.speed = torch.tensor([p[0] for p in pts], device=device)
        self.torque = torch.tensor([p[1] for p in pts], device=device)

    def __call__(self, speed_abs: torch.Tensor) -> torch.Tensor:
        s = speed_abs.clamp(max=self.speed[-1].item())
        i = torch.bucketize(s, self.speed, right=True).clamp(1, len(self.speed) - 1)
        s0, s1 = self.speed[i - 1], self.speed[i]
        t0, t1 = self.torque[i - 1], self.torque[i]
        return t0 + (t1 - t0) * (s - s0) / (s1 - s0)


class TorqueSpeedPDActuator(IdealPDActuator):
    """PD with the motor's torque-speed envelope.

    Motoring quadrant (torque and speed in the same direction): |tau| <= min(T(|qd|), effort_limit) with T the
    tabulated 48 V T-N curve, falling to 0 at the no-load speed. Braking quadrant (torque opposing the motion):
    |tau| <= effort_limit (the drive can brake at full current).
    """

    cfg: TorqueSpeedPDActuatorCfg

    def __init__(self, cfg: TorqueSpeedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._envelope = _Envelope(cfg.tn_curve, self._device)
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        motoring = effort * self._joint_vel > 0.0
        limit = torch.where(
            motoring, torch.minimum(self._envelope(self._joint_vel.abs()), self.effort_limit), self.effort_limit
        )
        return torch.maximum(torch.minimum(effort, limit), -limit)


class AnkleLinkagePDActuator(IdealPDActuator):
    """PD on ankle pitch / roll; the torques are realised by two RS02 per side through the linkage.

    Per side: tau_1,2 = (tau_p / N_p +/- tau_r / N_r) / 2 and w_1,2 = N_p w_p +/- N_r w_r. Each motor's limit is
    ``motor_torque_limit`` (both quadrants) and, while motoring, also its T-N envelope at |w_i|. If a motor would
    exceed its limit, BOTH motors of that side are scaled by the same factor s = min(1, min_i limit_i / |tau_i|), so
    the commanded pitch : roll torque direction is kept (independent per-motor clipping would turn a saturated
    pitch + roll demand into pure roll). Mapped back, tau_p = N_p (tau_1 + tau_2), tau_r = N_r (tau_1 - tau_2), which
    satisfies the coupled limit |tau_p| / N_p + |tau_r| / N_r <= 2 x motor_torque_limit.
    """

    cfg: AnkleLinkagePDActuatorCfg

    def __init__(self, cfg: AnkleLinkagePDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = list(self.joint_names)
        self._pitch = [names.index(f"Feet_Pitch_{s}_Joint") for s in ("Left", "Right")]
        self._roll = [names.index(f"Feet_Roll_{s}_Joint") for s in ("Left", "Right")]
        self._envelope = _Envelope(cfg.motor_tn_curve, self._device)
        self.motor_torque = torch.zeros(self._num_envs, 4, device=self._device)
        """Motor output torques after clipping [N·m], (num_envs, 4): L m1, L m2, R m1, R m2."""
        self.motor_speed = torch.zeros_like(self.motor_torque)
        """Motor output speeds [rad/s], same layout."""

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        n_p, n_r = self.cfg.n_pitch, self.cfg.n_roll
        tp, tr = self.computed_effort[:, self._pitch], self.computed_effort[:, self._roll]  # (N, 2): L, R
        wp, wr = joint_vel[:, self._pitch], joint_vel[:, self._roll]
        tau = torch.stack([0.5 * (tp / n_p + tr / n_r), 0.5 * (tp / n_p - tr / n_r)], dim=-1)  # (N, side, motor)
        w = torch.stack([n_p * wp + n_r * wr, n_p * wp - n_r * wr], dim=-1)
        limit = torch.full_like(tau, self.cfg.motor_torque_limit)
        limit = torch.where(tau * w > 0.0, torch.minimum(limit, self._envelope(w.abs())), limit)
        scale = (limit / tau.abs().clamp(min=1e-9)).amin(dim=-1, keepdim=True).clamp(max=1.0)  # per side
        tau = tau * scale
        applied = self.computed_effort.clone()
        applied[:, self._pitch] = n_p * (tau[..., 0] + tau[..., 1])
        applied[:, self._roll] = n_r * (tau[..., 0] - tau[..., 1])
        self.applied_effort = applied
        self.motor_torque[:] = tau.reshape(-1, 4)
        self.motor_speed[:] = w.reshape(-1, 4)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
