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


class NovaWalkingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that logs per-step leg-length and command statistics."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        robot = self.scene["robot"]
        prismatic_term = self.action_manager.get_term("prismatic_vel")
        q = robot.data.joint_pos.torch[:, prismatic_term.joint_ids]  # order: Upper L, Upper R, Lower L, Lower R
        cmd = self.command_manager.get_command("base_velocity")
        log = dict(extras.get("log", {}))
        log["Metrics/prismatic_upper_q_mean"] = q[:, 0:2].mean()
        log["Metrics/prismatic_lower_q_mean"] = q[:, 2:4].mean()
        log["Metrics/prismatic_qdot_cmd_abs_mean"] = prismatic_term.commanded_velocity.abs().mean()
        log["Metrics/command_planar_speed_mean"] = torch.linalg.norm(cmd[:, :2], dim=1).mean()
        extras["log"] = log
        self.extras = extras
        return obs, rew, terminated, truncated, extras
