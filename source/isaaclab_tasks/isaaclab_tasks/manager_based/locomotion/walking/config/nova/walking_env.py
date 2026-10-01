# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class for the NOVA walking task: ManagerBasedRLEnv + per-step morphology/command metrics.

The metrics are written into ``extras["log"]``, which the rsl_rl logger (rsl_rl.utils.logger.Logger
.process_env_step) collects every step and averages per iteration; keys containing "/" are written to
TensorBoard under their own name. ManagerBasedRLEnv only *replaces* ``extras["log"]`` inside ``_reset_idx``,
so on steps without resets the same dict object would be appended again -- mutating it in place would alias
every stored entry to the latest values. A fresh dict is therefore built each step. The logger takes its key set from
the first step of an iteration, so every key is emitted on every step.

Leg-length statistics are computed over FREE envs only (LOCKED envs hold their reset sample, see
mdp/actions.py:PrismaticLengthAction). Speed-binned morphology metrics (Metrics/morph/b<k>_*) use FREE envs binned by
commanded planar speed, with an exponential running average over steps (sparse bins stay smooth):
    b1 [0, 0.25)  b2 [0.25, 0.75)  b3 [0.75, 1.25)  b4 [1.25, 1.75)  b5 [1.75, 2.1] m/s
per bin: upper_q / lower_q / total_q (L/R-averaged joint positions [m]), count (FREE envs in the bin this step),
p_elec [W], cot = P_elec / (m g max(|v_xy|, 0.1)) with m the per-env total robot mass.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_apply_inverse

from .mdp.power import power_model
from .mdp.rewards import effort_sum_ratio_sq

MORPH_BIN_EDGES = (0.25, 0.75, 1.25, 1.75)
"""Inner edges of the commanded planar speed bins [m/s] (outer: 0 and 2.1, the max |cmd_xy| is 2.09)."""
MORPH_EMA_DECAY = 0.99
"""Per-step decay of the binned running averages (time constant ~100 steps = ~4 iterations of 24 steps)."""
_MORPH_FIELDS = ("upper_q", "lower_q", "total_q", "p_elec", "cot")


class NovaWalkingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that logs per-step leg-length, power and command statistics."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        robot = self.scene["robot"]
        if not hasattr(self, "_foot_ids"):
            names = robot.body_names
            self._hip_base_id = names.index("Hip_Base")
            self._foot_ids = [names.index("Feet_Pitch_Left"), names.index("Feet_Pitch_Right")]
        prismatic_term = self.action_manager.get_term("prismatic")
        locked = getattr(prismatic_term, "locked", None)
        free = ~locked if locked is not None else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if not free.any():  # all-LOCKED evaluation: fall back to all envs for the leg-length statistics
            free = torch.ones_like(free)
        q = robot.data.joint_pos.torch[:, prismatic_term.joint_ids]  # order: Upper L, Upper R, Lower L, Lower R
        q_up, q_low = q[:, 0:2].mean(dim=1), q[:, 2:4].mean(dim=1)
        cmd = self.command_manager.get_command("base_velocity")
        log = dict(extras.get("log", {}))
        log["Metrics/morph_locked_frac"] = (~free).float().mean()
        log["Metrics/prismatic_upper_q_mean"] = q_up[free].mean()
        log["Metrics/prismatic_lower_q_mean"] = q_low[free].mean()
        # spread of the (L/R-averaged) leg lengths across FREE envs
        log["Metrics/prismatic_upper_q_std"] = q_up[free].std()
        log["Metrics/prismatic_lower_q_std"] = q_low[free].std()
        # raw effort, visible even when effort_reward saturates at 0 or 1
        effort_cfg = self.reward_manager.get_term_cfg("effort_reward").params["asset_cfg"]
        log["Metrics/effort_sum_ratio_sq_mean"] = effort_sum_ratio_sq(self, effort_cfg).mean()
        # horizontal distance between the feet in the base frame (standing ~0.21 m; run 1's split ~1.30 m)
        pos, quat = robot.data.body_link_pos_w.torch, robot.data.body_link_quat_w.torch
        d = quat_apply_inverse(quat[:, self._hip_base_id], pos[:, self._foot_ids[0]] - pos[:, self._foot_ids[1]])
        log["Metrics/feet_separation_mean"] = torch.linalg.norm(d[:, :2], dim=1).mean()
        # hopping detector: last completed swing duration per foot, averaged over envs (~0 alternating, ~1 one-leg hop)
        contact = self.scene["contact_forces"]
        if not hasattr(self, "_sensor_foot_ids"):
            self._sensor_foot_ids = [contact.body_names.index(n) for n in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
        last_air = contact.data.last_air_time.torch[:, self._sensor_foot_ids].mean(dim=0)
        log["Metrics/air_time_asymmetry"] = (last_air[0] - last_air[1]).abs() / (last_air.sum() + 1e-6)
        in_contact = contact.data.current_contact_time.torch[:, self._sensor_foot_ids] > 0.0
        log["Metrics/single_stance_frac"] = (in_contact.sum(dim=1) == 1).float().mean()
        # gait / morphology diagnostics
        target = prismatic_term.processed_actions  # U_L, U_R, L_L, L_R
        at_clamp = (target <= 0.0055) | (target >= 0.0945)
        log["Metrics/prismatic_upper_at_clamp_frac"] = at_clamp[free][:, 0:2].float().mean()
        log["Metrics/prismatic_lower_at_clamp_frac"] = at_clamp[free][:, 2:4].float().mean()
        log["Metrics/flight_frac"] = (~in_contact).all(dim=1).float().mean()
        log["Metrics/max_swing_time_mean"] = (
            contact.data.current_air_time.torch[:, self._sensor_foot_ids].amax(dim=1).mean()
        )
        leg = self.scene["leg_contact"].data.force_matrix_w.torch
        log["Metrics/leg_self_contact_frac"] = (
            (torch.linalg.norm(leg, dim=-1) > 1.0).any(dim=2).any(dim=1).float().mean()
        )
        log["Metrics/prismatic_target_rate_abs_mean"] = prismatic_term.target_rate.abs().mean()
        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
        log["Metrics/command_planar_speed_mean"] = cmd_speed.mean()
        # electrical power (same cached evaluation as the p_elec reward); mech = P - P_cu per motor (negative when a
        # motor brakes, since P = max(mech + cu, 0) per motor)
        p_motor, p_mech, p_cu, _ = power_model(self, "prismatic").compute(self)
        p_elec = p_motor.sum(dim=1)
        log["Metrics/p_elec_mean"] = p_elec.mean()
        log["Metrics/p_mech_mean"] = p_mech.sum(dim=1).mean()
        log["Metrics/p_copper_mean"] = p_cu.sum(dim=1).mean()
        log.update(self._morph_bin_metrics(free, cmd_speed, q_up, q_low, p_elec))
        extras["log"] = log
        self.extras = extras
        return obs, rew, terminated, truncated, extras

    def _morph_bin_metrics(
        self, free: torch.Tensor, cmd_speed: torch.Tensor, q_up: torch.Tensor, q_low: torch.Tensor, p_elec: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Speed-binned FREE-env leg lengths, power and cost of transport (running averages, see module docstring)."""
        robot = self.scene["robot"]
        if not hasattr(self, "_morph_sum"):
            n_bins = len(MORPH_BIN_EDGES) + 1
            self._morph_edges = torch.tensor(MORPH_BIN_EDGES, device=self.device)
            self._morph_sum = torch.zeros(n_bins, len(_MORPH_FIELDS), device=self.device)
            self._morph_weight = torch.zeros(n_bins, device=self.device)
            # per-env total mass (base-mass randomization is a startup event, so this is constant)
            self._total_mass = robot.data.body_mass.torch.sum(dim=1).to(self.device)
        speed = torch.linalg.norm(robot.data.root_lin_vel_w.torch[:, :2], dim=1)
        cot = p_elec / (self._total_mass * 9.81 * speed.clamp(min=0.1))
        vals = torch.stack([q_up, q_low, q_up + q_low, p_elec, cot], dim=1)  # (N, F)
        bins = torch.bucketize(cmd_speed, self._morph_edges, right=True)
        member = torch.nn.functional.one_hot(bins, self._morph_weight.numel()).float() * free.float().unsqueeze(1)
        count = member.sum(dim=0)
        self._morph_sum.mul_(MORPH_EMA_DECAY).add_(member.T @ vals)
        self._morph_weight.mul_(MORPH_EMA_DECAY).add_(count)
        mean = self._morph_sum / self._morph_weight.unsqueeze(1)  # NaN only until a bin has been visited once
        log = {}
        for b in range(self._morph_weight.numel()):
            for f, name in enumerate(_MORPH_FIELDS):
                log[f"Metrics/morph/b{b + 1}_{name}"] = mean[b, f]
            log[f"Metrics/morph/b{b + 1}_count"] = count[b]
        return log
