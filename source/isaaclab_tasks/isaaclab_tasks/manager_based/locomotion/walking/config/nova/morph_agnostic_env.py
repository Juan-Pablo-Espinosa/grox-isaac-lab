# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class of the morphology-agnostic NOVA walking task: external prismatic driver + metrics.

The driver (mdp/prismatic_driver.py) is advanced and written to the robot at the start of every env step, before the
physics substeps, and re-anchored after resets (after the reset event has sampled the new joint state, before the
observation is computed). It is created lazily because the observation manager evaluates ``prismatic_targets``
while the managers are being built.

Metrics (``extras["log"]``, see walking_env.py for the logging mechanics; every key is emitted every step):
  * run-4 gait metrics (Metrics/prismatic_*, effort_sum_ratio_sq_mean, feet_separation_mean, air_time_asymmetry,
    single_stance_frac, flight_frac, max_swing_time_mean, leg_self_contact_frac, command_planar_speed_mean)
  * electrical power (mdp/power.py, METRIC ONLY): Metrics/p_elec_mean, p_mech_mean, p_copper_mean
  * Metrics/tau_frac/<group>: mean |tau| / effort limit over the joints of hip_knee, roll, yaw, ankle
  * Metrics/ankle_rs02_rated_frac_{mean,rms}: ankle motor torque (tau_pitch +/- tau_roll)/2 over the RS02's 6 N·m
  * Metrics/morph2d/e<i>_v<j>_{cot,p_elec,track_err,fall_rate}: running averages per cell of total extension
    q_U + q_L bins e1 [0, 0.06) e2 [0.06, 0.12) e3 [0.12, 0.19] m x commanded forward speed bins v1 [0, 0.25)
    v2 [0.25, 0.75) v3 [0.75, 1.25) v4 [1.25, 1.75) v5 [1.75, 2.25) v6 [2.25, 2.6] m/s (backward commands are not
    binned). Cells are assigned from the state BEFORE the step (so a fall is attributed to the cell it happened in).
    cot = P_elec / (m g max(|v_xy|, 0.1)); track_err = |v_xy - cmd_xy| (base frame) [m/s]; fall_rate = falls per
    env-second (terminations that are not time-outs).
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_apply_inverse

from .mdp.power import PRISMATIC_DRIVER, ROBSTRIDE, power_model
from .mdp.prismatic_driver import PrismaticDriver
from .mdp.rewards import effort_sum_ratio_sq

EXT_BIN_EDGES = (0.06, 0.12)
"""Inner edges of the total-extension bins [m] (outer 0 and 0.19)."""
SPEED_BIN_EDGES = (0.25, 0.75, 1.25, 1.75, 2.25)
"""Inner edges of the commanded forward speed bins [m/s] (outer 0 and 2.6)."""
EMA_DECAY = 0.99
"""Per-step decay of the binned running averages (~100 steps = ~4 iterations)."""
_CELL_FIELDS = ("cot", "p_elec", "track_err", "fall_rate")
TORQUE_GROUPS = {
    "hip_knee": ("Hip_Pitch_", "Lowerleg_Pitch_"),
    "roll": ("Hip_Roll_",),
    "yaw": ("Upperleg_Yaw_",),
    "ankle": ("Feet_Roll_", "Feet_Pitch_"),
}
"""Joint-name prefixes of the torque groups (|tau| / effort limit)."""


class NovaMorphAgnosticEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with an external prismatic driver and morphology / energy metrics."""

    @property
    def prismatic_driver(self) -> PrismaticDriver:
        """The external leg-length driver (created on first use)."""
        driver = getattr(self, "_prismatic_driver", None)
        if driver is None:
            driver = PrismaticDriver(self.cfg.prismatic_driver, self)
            self._prismatic_driver = driver
        return driver

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.prismatic_driver.reset(env_ids)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        driver = self.prismatic_driver
        driver.step()
        robot = self.scene["robot"]
        # pre-step state for the 2-D table (cell of the env when the step starts)
        q_pre = robot.data.joint_pos.torch[:, driver.joint_ids]
        ext_pre = q_pre[:, 0:2].mean(dim=1) + q_pre[:, 2:4].mean(dim=1)
        cmd_pre = self.command_manager.get_command("base_velocity").clone()

        obs, rew, terminated, truncated, extras = super().step(action)

        log = dict(extras.get("log", {}))
        log.update(self._gait_metrics(robot, driver))
        p_motor, p_mech, p_cu, tau_m = power_model(self, PRISMATIC_DRIVER).compute(self)
        p_elec = p_motor.sum(dim=1)
        log["Metrics/p_elec_mean"] = p_elec.mean()
        log["Metrics/p_mech_mean"] = p_mech.sum(dim=1).mean()
        log["Metrics/p_copper_mean"] = p_cu.sum(dim=1).mean()
        log.update(self._torque_metrics(robot, tau_m))
        log.update(self._cell_metrics(robot, ext_pre, cmd_pre, p_elec, terminated & ~truncated))
        extras["log"] = log
        self.extras = extras
        return obs, rew, terminated, truncated, extras

    def _gait_metrics(self, robot, driver: PrismaticDriver) -> dict[str, torch.Tensor]:
        """The run-4 metric set (walking_env.py at 9852d3b807), with the driver in place of the prismatic action."""
        if not hasattr(self, "_foot_ids"):
            names = robot.body_names
            self._hip_base_id = names.index("Hip_Base")
            self._foot_ids = [names.index("Feet_Pitch_Left"), names.index("Feet_Pitch_Right")]
            contact = self.scene["contact_forces"]
            self._sensor_foot_ids = [contact.body_names.index(n) for n in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
        q = robot.data.joint_pos.torch[:, driver.joint_ids]  # U_L, U_R, L_L, L_R
        cmd = self.command_manager.get_command("base_velocity")
        log = {
            "Metrics/prismatic_upper_q_mean": q[:, 0:2].mean(),
            "Metrics/prismatic_lower_q_mean": q[:, 2:4].mean(),
            "Metrics/prismatic_upper_q_std": q[:, 0:2].mean(dim=1).std(),
            "Metrics/prismatic_lower_q_std": q[:, 2:4].mean(dim=1).std(),
        }
        effort_cfg = self.reward_manager.get_term_cfg("effort_reward").params["asset_cfg"]
        log["Metrics/effort_sum_ratio_sq_mean"] = effort_sum_ratio_sq(self, effort_cfg).mean()
        pos, quat = robot.data.body_link_pos_w.torch, robot.data.body_link_quat_w.torch
        d = quat_apply_inverse(quat[:, self._hip_base_id], pos[:, self._foot_ids[0]] - pos[:, self._foot_ids[1]])
        log["Metrics/feet_separation_mean"] = torch.linalg.norm(d[:, :2], dim=1).mean()
        contact = self.scene["contact_forces"]
        last_air = contact.data.last_air_time.torch[:, self._sensor_foot_ids].mean(dim=0)
        log["Metrics/air_time_asymmetry"] = (last_air[0] - last_air[1]).abs() / (last_air.sum() + 1e-6)
        in_contact = contact.data.current_contact_time.torch[:, self._sensor_foot_ids] > 0.0
        log["Metrics/single_stance_frac"] = (in_contact.sum(dim=1) == 1).float().mean()
        at_clamp = (driver.target <= 0.0055) | (driver.target >= 0.0945)
        log["Metrics/prismatic_upper_at_clamp_frac"] = at_clamp[:, 0:2].float().mean()
        log["Metrics/prismatic_lower_at_clamp_frac"] = at_clamp[:, 2:4].float().mean()
        log["Metrics/flight_frac"] = (~in_contact).all(dim=1).float().mean()
        log["Metrics/max_swing_time_mean"] = (
            contact.data.current_air_time.torch[:, self._sensor_foot_ids].amax(dim=1).mean()
        )
        leg = self.scene["leg_contact"].data.force_matrix_w.torch
        log["Metrics/leg_self_contact_frac"] = (
            (torch.linalg.norm(leg, dim=-1) > 1.0).any(dim=2).any(dim=1).float().mean()
        )
        log["Metrics/prismatic_target_rate_abs_mean"] = driver.target_rate.abs().mean()
        log["Metrics/command_planar_speed_mean"] = torch.linalg.norm(cmd[:, :2], dim=1).mean()
        return log

    def _torque_metrics(self, robot, tau_m: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-group mean |tau| / effort limit and the ankle RS02 torque over its rated torque."""
        if not hasattr(self, "_tau_group_ids"):
            names = robot.joint_names
            self._tau_group_ids = {
                g: [i for i, n in enumerate(names) if n.startswith(prefixes)] for g, prefixes in TORQUE_GROUPS.items()
            }
            labels = power_model(self, PRISMATIC_DRIVER).labels
            self._ankle_motor_ids = [i for i, lab in enumerate(labels) if lab.startswith("ankle")]
        tau = robot.data.applied_torque.torch.abs()
        lim = robot.data.joint_effort_limits.torch
        log = {f"Metrics/tau_frac/{g}": (tau[:, ids] / lim[:, ids]).mean() for g, ids in self._tau_group_ids.items()}
        ankle = tau_m[:, self._ankle_motor_ids] / ROBSTRIDE["RS02"].rated_torque
        log["Metrics/ankle_rs02_rated_frac_mean"] = ankle.mean()
        log["Metrics/ankle_rs02_rated_frac_rms"] = ankle.pow(2).mean().sqrt()
        return log

    def _cell_metrics(
        self, robot, ext: torch.Tensor, cmd: torch.Tensor, p_elec: torch.Tensor, fell: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Running averages per (total extension, commanded forward speed) cell."""
        n_e, n_v = len(EXT_BIN_EDGES) + 1, len(SPEED_BIN_EDGES) + 1
        if not hasattr(self, "_cell_sum"):
            self._ext_edges = torch.tensor(EXT_BIN_EDGES, device=self.device)
            self._speed_edges = torch.tensor(SPEED_BIN_EDGES, device=self.device)
            self._cell_sum = torch.zeros(n_e * n_v, len(_CELL_FIELDS), device=self.device)
            self._cell_weight = torch.zeros(n_e * n_v, device=self.device)
            # per-env total mass (base-mass randomization is a startup event)
            self._total_mass = robot.data.body_mass.torch.sum(dim=1).to(self.device)
        v_b = robot.data.root_lin_vel_b.torch[:, :2]
        speed = torch.linalg.norm(robot.data.root_lin_vel_w.torch[:, :2], dim=1)
        cot = p_elec / (self._total_mass * 9.81 * speed.clamp(min=0.1))
        err = torch.linalg.norm(v_b - cmd[:, :2], dim=1)
        fall_rate = fell.float() / self.step_dt
        vals = torch.stack([cot, p_elec, err, fall_rate], dim=1)
        cell = torch.bucketize(ext, self._ext_edges, right=True) * n_v + torch.bucketize(
            cmd[:, 0], self._speed_edges, right=True
        )
        valid = (cmd[:, 0] >= 0.0).float()  # backward commands are not binned
        member = torch.nn.functional.one_hot(cell, n_e * n_v).float() * valid.unsqueeze(1)
        self._cell_sum.mul_(EMA_DECAY).add_(member.T @ vals)
        self._cell_weight.mul_(EMA_DECAY).add_(member.sum(dim=0))
        mean = self._cell_sum / self._cell_weight.unsqueeze(1)  # NaN until a cell has been visited
        log = {}
        for e in range(n_e):
            for v in range(n_v):
                for f, name in enumerate(_CELL_FIELDS):
                    log[f"Metrics/morph2d/e{e + 1}_v{v + 1}_{name}"] = mean[e * n_v + v, f]
        return log
