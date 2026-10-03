# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External leg-length driver for the morphology-agnostic task (the policy does not control the prismatics).

Each env holds a goal (q_U, q_L) [m] with left == right. With the random schedule on, every U(interval_range_s)
seconds (independent timer per env) a new goal is drawn uniformly in [q_min, q_max] for the upper and the lower pair
independently. The four position targets move toward the goal at most ``max_velocity`` (the leadscrew rate limit):

    q*_t = q*_{t-1} + clip(goal - q*_{t-1}, -max_velocity * dt, max_velocity * dt)

and are written with ``Articulation.set_joint_position_target_index`` on the four prismatic joints once per env step
(the buffer persists through the decimation substeps; ``JointPositionAction`` only writes the revolute targets). On
reset the targets and the goal snap to the joint positions just sampled by the reset event, and a new timer starts.
Teleop / evaluation scripts turn the schedule off and set goals with :meth:`PrismaticDriver.set_goal`.

Kit-free at import time (the task cfg imports the cfg class before the simulator starts).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class PrismaticDriverCfg:
    """Configuration of :class:`PrismaticDriver`."""

    joint_names: list[str] = MISSING
    """The four prismatic joints, ordered upper L, upper R, lower L, lower R."""
    max_velocity: float = 0.035
    """Leadscrew rate limit of the targets [m/s]."""
    q_min: float = 0.005
    """Lower bound of goals and targets [m]."""
    q_max: float = 0.095
    """Upper bound of goals and targets [m]."""
    interval_range_s: tuple[float, float] = (2.0, 8.0)
    """Time between goal draws [s], sampled uniformly per env."""
    schedule_enabled: bool = True
    """Whether the random goal schedule runs (teleop starts with it off)."""


class PrismaticDriver:
    """Rate-limited, externally scheduled position targets for the four prismatic joints."""

    def __init__(self, cfg: PrismaticDriverCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self._robot = env.scene["robot"]
        self.joint_ids, self.joint_names = self._robot.find_joints(cfg.joint_names, preserve_order=True)
        self.num_envs, self.device, self._dt = env.num_envs, env.device, env.step_dt
        self._dq_max = cfg.max_velocity * self._dt
        default = self._robot.data.default_joint_pos.torch[:, self.joint_ids].clone()
        self.target = default.clamp(cfg.q_min, cfg.q_max)
        """Current position targets [m], shape (num_envs, 4): U_L, U_R, L_L, L_R."""
        self.goal = self.target[:, [0, 2]].clone()
        """Goal lengths [m], shape (num_envs, 2): upper, lower (applied to both sides)."""
        self.target_rate = torch.zeros_like(self.target)
        """Rate of change of the targets this step [m/s], shape (num_envs, 4)."""
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.schedule_enabled = cfg.schedule_enabled
        self._sample_timer(torch.arange(self.num_envs, device=self.device))

    def _sample_timer(self, env_ids: torch.Tensor):
        lo, hi = self.cfg.interval_range_s
        self.timer[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def set_goal(
        self,
        upper: float | torch.Tensor | None = None,
        lower: float | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ):
        """Set the goal lengths [m] (clamped to [q_min, q_max]); ``None`` keeps that pair's goal."""
        ids = self._env_ids(env_ids)
        for col, value in ((0, upper), (1, lower)):
            if value is not None:
                self.goal[ids, col] = torch.as_tensor(value, device=self.device, dtype=torch.float32).clamp(
                    self.cfg.q_min, self.cfg.q_max
                )

    def step(self):
        """Advance the schedule by one env step, move the targets toward the goal and write them to the robot."""
        if self.schedule_enabled:
            self.timer -= self._dt
            due = (self.timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
            if due.numel():
                span = self.cfg.q_max - self.cfg.q_min
                self.goal[due] = self.cfg.q_min + span * torch.rand(len(due), 2, device=self.device)
                self._sample_timer(due)
        prev = self.target.clone()
        step = (self.goal[:, [0, 0, 1, 1]] - prev).clamp(-self._dq_max, self._dq_max)
        self.target[:] = (prev + step).clamp(self.cfg.q_min, self.cfg.q_max)
        self.target_rate[:] = (self.target - prev) / self._dt
        self.apply()

    def apply(self, env_ids: torch.Tensor | None = None):
        """Write the targets into the articulation's joint position target buffer."""
        if env_ids is None:
            self._robot.set_joint_position_target_index(target=self.target, joint_ids=self.joint_ids)
        else:
            self._robot.set_joint_position_target_index(
                target=self.target[env_ids], joint_ids=self.joint_ids, env_ids=env_ids
            )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Snap targets and goal to the freshly reset joint positions and restart the timers."""
        ids = self._env_ids(env_ids)
        q = self._robot.data.joint_pos.torch[ids][:, self.joint_ids].clamp(self.cfg.q_min, self.cfg.q_max)
        self.target[ids] = q
        self.goal[ids] = q[:, [0, 2]]
        self.target_rate[ids] = 0.0
        self._sample_timer(ids)
        self.apply(ids)
