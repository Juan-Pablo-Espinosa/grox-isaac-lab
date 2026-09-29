# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walking-specific reward terms for NOVA_LOWERBODY_V2.

upright_reward / acceleration_reward / symmetry_reward are reused unchanged from the standing task
(imported in walking_env_cfg.py). Command tracking uses the stock
``isaaclab.envs.mdp.track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp``, which already read
``env.command_manager.get_command(command_name)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

# Deferred to TYPE_CHECKING -- see standing mdp/rewards.py (eager import crashes before Kit boots).
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def effort_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-k * sum_j (tau_j / tau_limit_j)^2) over the joints in ``asset_cfg``.

    Same %-of-own-limit formula and k=5.1782 as the standing task's effort_reward, but the limits are
    read at runtime from ``robot.data.joint_effort_limits`` (i.e. the walking actuator groups'
    effort_limit_sim: 120/120/60/36/32 N·m) instead of a hard-coded list, so they cannot drift from the
    actuator config. Pass the 12 revolute joints only -- prismatics are deliberately excluded.

    TODO: k was fitted on standing telemetry (p90 of pct_sq_sum=0.13386 -> reward 0.5); recalibrate on walking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    pct_sq_sum = torch.sum((torques / limits) ** 2, dim=1)

    k = 5.1782
    return torch.exp(-k * pct_sq_sum)


class PrismaticPowerReward(ManagerTermBase):
    """Mechanical power of the leadscrew joints: sum_j |F_j * qdot_j| [W] (no regeneration credit).

    F is the actuator's applied force (``robot.data.applied_torque``, N for prismatic joints). qdot is the
    FINITE DIFFERENCE of joint position over one env step, not ``robot.data.joint_vel``: measured while
    standing still, PhysX reports a biased prismatic joint velocity (mean |qd| ~0.0115 m/s, lower joints
    ~+0.02 m/s signed) although the positions change by only ~1e-5 m/s -- which would charge ~4 W of
    phantom power at rest instead of ~0.004 W. Returned as a raw positive quantity; the (negative) weight
    is set after telemetry. Holding position costs ~0 W, matching the non-backdrivable leadscrew.

    Note: RewardManager skips terms with weight 0.0 entirely, so this term is not evaluated (and its
    previous-position buffer not advanced) until it gets a non-zero weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_ids = asset_cfg.joint_ids
        self._prev_q = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_q[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        qdot = (q - self._prev_q) / env.step_dt
        self._prev_q = q.clone()
        force = self._asset.data.applied_torque.torch[:, self._joint_ids]
        return torch.sum(torch.abs(force * qdot), dim=1)
