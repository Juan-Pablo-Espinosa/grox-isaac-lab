# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-command action term for NOVA's leadscrew-driven prismatic (leg-length) joints.

The real leadscrews (T12x8 4-start trapezoidal, bronze nut, driven directly by a RobStride 00) are
non-backdrivable and speed-limited (~0.035-0.041 m/s at walking loads), so the policy does not command a
leg length directly. Instead each action is a normalized *velocity* command that
integrates into a persistent position target:

    a        = clip(raw, -1, 1)
    q_target = clamp(q_target + a * max_velocity * step_dt, q_min, q_max)

The target is written with ``Articulation.set_joint_position_target_index`` every physics substep,
so a zero action holds the current target (like a stopped leadscrew).

API implemented against the installed ``isaaclab.managers.action_manager.ActionTerm``:
abstract ``action_dim`` / ``raw_actions`` / ``processed_actions`` properties and
``process_actions(actions)`` (called once per env step) / ``apply_actions()`` (called every physics
substep), plus ``reset(env_ids)`` from ``ManagerTermBase``. ``ManagerBasedRLEnv._reset_idx`` runs
reset-mode events *before* ``action_manager.reset``, so ``reset`` sees the freshly sampled joint positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

# Deferred to TYPE_CHECKING, same reason as the standing task's mdp modules: importing
# isaaclab.assets / isaaclab.envs eagerly forces pxr resolution before Kit boots.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class PrismaticVelocityAction(ActionTerm):
    """Integrates normalized velocity commands into clamped position targets for prismatic joints."""

    cfg: PrismaticVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: PrismaticVelocityActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # max target change per env step [m]
        self._dq_max = self.cfg.max_velocity * env.step_dt
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # persistent position target, starts at the asset defaults
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._q_target = default.clamp(self.cfg.q_min, self.cfg.q_max)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The current integrated position targets [m]."""
        return self._q_target

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def joint_ids(self) -> list[int]:
        """Articulation joint indices of the controlled prismatic joints, in action order."""
        return self._joint_ids

    @property
    def commanded_velocity(self) -> torch.Tensor:
        """Commanded screw velocity clip(raw, -1, 1) * max_velocity [m/s], shape (num_envs, action_dim).

        Note: this is the *commanded* rate, so it stays non-zero when the target is pinned at a clamp bound.
        """
        return self._raw_actions.clamp(-1.0, 1.0) * self.cfg.max_velocity

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.clamp(-1.0, 1.0)
        self._q_target = (self._q_target + a * self._dq_max).clamp(self.cfg.q_min, self.cfg.q_max)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._q_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        # re-anchor the target to wherever the reset event just placed the joints
        q = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
        self._q_target[env_ids] = q.clamp(self.cfg.q_min, self.cfg.q_max)


@configclass
class PrismaticVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`PrismaticVelocityAction`."""

    class_type: type[ActionTerm] = PrismaticVelocityAction

    joint_names: list[str] = MISSING
    """Joint names or regex expressions of the prismatic joints."""
    preserve_order: bool = True
    """Keep the action ordering identical to :attr:`joint_names`."""
    max_velocity: float = 0.035
    """Leadscrew speed limit [m/s]; a unit action moves the target by ``max_velocity * step_dt``.

    Default 0.035 m/s: RobStride 00 (10:1, 315 rpm no-load) on an 8 mm-lead screw at walking loads."""
    q_min: float = 0.005
    """Lower clamp of the position target [m] (kept off the 0.0 hard stop)."""
    q_max: float = 0.095
    """Upper clamp of the position target [m] (kept off the 0.1 hard stop)."""
