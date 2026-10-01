# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for NOVA's leadscrew-driven prismatic (leg-length) joints.

:class:`PrismaticLengthAction` (current task): absolute desired length, rate-limited, with a morphology lock -- see
its docstring. :class:`PrismaticVelocityAction` (runs 1-4, kept for loading their checkpoints) is described below.

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
        self._dt = env.step_dt
        self._dq_max = self.cfg.max_velocity * self._dt
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # persistent position target, starts at the asset defaults
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._q_target = default.clamp(self.cfg.q_min, self.cfg.q_max)
        # actual rate of change of the integrated target this step [m/s]; 0 when pinned at a clamp bound
        self._target_rate = torch.zeros_like(self._q_target)

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
        Use :attr:`target_rate` for the rate the screw is actually driven at.
        """
        return self._raw_actions.clamp(-1.0, 1.0) * self.cfg.max_velocity

    @property
    def target_rate(self) -> torch.Tensor:
        """Actual rate of change of the clamped position target this step [m/s], shape (num_envs, action_dim).

        Equals :attr:`commanded_velocity` except when the target is pinned at ``q_min``/``q_max``, where it is
        exactly 0 (the motor does no work there). Zeroed on reset.
        """
        return self._target_rate

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.clamp(-1.0, 1.0)
        q_prev = self._q_target.clone()
        # in place, so the buffers stay normal tensors when stepped under torch.inference_mode
        self._q_target[:] = (q_prev + a * self._dq_max).clamp(self.cfg.q_min, self.cfg.q_max)
        self._target_rate[:] = (self._q_target - q_prev) / self._dt

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._q_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._target_rate[env_ids] = 0.0
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


class PrismaticLengthAction(PrismaticVelocityAction):
    """Absolute leg-length command with the leadscrew rate limit, plus a per-episode morphology lock.

    Each action is a normalized *desired length*; the position target moves toward it at most ``max_velocity``:

        l_des = c + h * clip(a, -1, 1),   c = (q_min + q_max) / 2,  h = (q_max - q_min) / 2   (0.05 +/- 0.045 m)
        q*_t  = q*_{t-1} + clip(l_des - q*_{t-1}, -max_velocity * dt, max_velocity * dt)

    Unlike the velocity mode, a zero-mean action does not integrate into a drift toward a hard stop: noise around a
    holds the target near c + h * a.

    Morphology lock: at every reset an env is LOCKED with probability ``lock_fraction`` (or per ``lock_per_env``).
    A locked env keeps its position targets at the reset sample for the whole episode (the 4 prismatic actions are
    ignored, :attr:`target_rate` is 0); FREE envs follow their actions. :attr:`locked` is exposed for the observation
    and the metrics.
    """

    cfg: PrismaticLengthActionCfg

    def __init__(self, cfg: PrismaticLengthActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._center = 0.5 * (self.cfg.q_min + self.cfg.q_max)
        self._half_range = 0.5 * (self.cfg.q_max - self.cfg.q_min)
        self._locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lock_per_env = None
        if self.cfg.lock_per_env is not None:
            if len(self.cfg.lock_per_env) != self.num_envs:
                raise ValueError(f"lock_per_env has {len(self.cfg.lock_per_env)} entries for {self.num_envs} envs")
            self._lock_per_env = torch.tensor(self.cfg.lock_per_env, dtype=torch.bool, device=self.device)

    @property
    def locked(self) -> torch.Tensor:
        """Per-env morphology-lock flag, shape (num_envs,), bool."""
        return self._locked

    @property
    def desired_length(self) -> torch.Tensor:
        """Desired length decoded from the last action c + h * clip(a, -1, 1) [m], shape (num_envs, action_dim)."""
        return self._center + self._half_range * self._raw_actions.clamp(-1.0, 1.0)

    @property
    def commanded_velocity(self) -> torch.Tensor:
        """Rate the target is driven at this step [m/s] (same as :attr:`target_rate` in length mode)."""
        return self._target_rate

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        q_prev = self._q_target.clone()
        step = (self.desired_length - q_prev).clamp(-self._dq_max, self._dq_max)
        step = torch.where(self._locked.unsqueeze(1), torch.zeros_like(step), step)
        self._q_target[:] = (q_prev + step).clamp(self.cfg.q_min, self.cfg.q_max)
        self._target_rate[:] = (self._q_target - q_prev) / self._dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # re-anchors the target to the freshly reset joints (the lock keeps exactly this target)
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if self._lock_per_env is not None:
            self._locked[env_ids] = self._lock_per_env[env_ids]
        else:
            self._locked[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.lock_fraction


@configclass
class PrismaticLengthActionCfg(PrismaticVelocityActionCfg):
    """Configuration for :class:`PrismaticLengthAction`."""

    class_type: type[ActionTerm] = PrismaticLengthAction

    lock_fraction: float = 0.0
    """Probability that an env is LOCKED for an episode (sampled at every reset)."""
    lock_per_env: list[bool] | None = None
    """Fixed lock flag per env index (overrides :attr:`lock_fraction`); length must equal the number of envs."""
