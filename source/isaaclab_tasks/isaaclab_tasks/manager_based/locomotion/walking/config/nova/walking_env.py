# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class for the NOVA walking task: ManagerBasedRLEnv + per-step morphology/command metrics.

The metrics are written into ``extras["log"]``, which the rsl_rl logger (rsl_rl.utils.logger.Logger
.process_env_step) collects every step and averages per iteration; keys containing "/" are written to
TensorBoard under their own name. ManagerBasedRLEnv only *replaces* ``extras["log"]`` inside ``_reset_idx``,
so on steps without resets the same dict object would be appended again -- mutating it in place would alias
every stored entry to the latest values. A fresh dict is therefore built each step.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_apply_inverse

from .mdp.rewards import effort_sum_ratio_sq


class NovaWalkingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that logs per-step leg-length and command statistics."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        robot = self.scene["robot"]
        if not hasattr(self, "_foot_ids"):
            names = robot.body_names
            self._hip_base_id = names.index("Hip_Base")
            self._foot_ids = [names.index("Feet_Pitch_Left"), names.index("Feet_Pitch_Right")]
        prismatic_term = self.action_manager.get_term("prismatic_vel")
        q = robot.data.joint_pos.torch[:, prismatic_term.joint_ids]  # order: Upper L, Upper R, Lower L, Lower R
        cmd = self.command_manager.get_command("base_velocity")
        log = dict(extras.get("log", {}))
        log["Metrics/prismatic_upper_q_mean"] = q[:, 0:2].mean()
        log["Metrics/prismatic_lower_q_mean"] = q[:, 2:4].mean()
        # spread of the (L/R-averaged) leg lengths across envs
        log["Metrics/prismatic_upper_q_std"] = q[:, 0:2].mean(dim=1).std()
        log["Metrics/prismatic_lower_q_std"] = q[:, 2:4].mean(dim=1).std()
        # raw effort, visible even when effort_reward saturates at 0 or 1
        effort_cfg = self.reward_manager.get_term_cfg("effort_reward").params["asset_cfg"]
        log["Metrics/effort_sum_ratio_sq_mean"] = effort_sum_ratio_sq(self, effort_cfg).mean()
        # horizontal distance between the feet in the base frame (standing ~0.21 m; run 1's split ~1.30 m)
        pos, quat = robot.data.body_link_pos_w.torch, robot.data.body_link_quat_w.torch
        d = quat_apply_inverse(quat[:, self._hip_base_id], pos[:, self._foot_ids[0]] - pos[:, self._foot_ids[1]])
        log["Metrics/feet_separation_mean"] = torch.linalg.norm(d[:, :2], dim=1).mean()
        log["Metrics/prismatic_target_rate_abs_mean"] = prismatic_term.target_rate.abs().mean()
        log["Metrics/command_planar_speed_mean"] = torch.linalg.norm(cmd[:, :2], dim=1).mean()
        extras["log"] = log
        self.extras = extras
        return obs, rew, terminated, truncated, extras
