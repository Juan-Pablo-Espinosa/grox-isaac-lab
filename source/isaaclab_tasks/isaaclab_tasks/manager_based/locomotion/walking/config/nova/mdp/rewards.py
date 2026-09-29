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

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

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


def prismatic_power_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_term_name: str = "prismatic_vel",
) -> torch.Tensor:
    """Commanded mechanical power of the leadscrews: sum_j |F_j * qdot_cmd_j| [W] (no regeneration credit).

    F is the actuator's applied force (``robot.data.applied_torque``, N). qdot_cmd is the COMMANDED screw
    velocity clip(a, -1, 1) * v_max exposed by :class:`PrismaticVelocityAction` -- i.e. what the RobStride 00
    is asked to spin at -- rather than the simulated joint velocity (PhysX reports a biased prismatic joint
    velocity at standstill, and impacts move the stiff drive without the motor doing work). Holding (a=0)
    costs 0 W, matching the non-backdrivable leadscrew. ``asset_cfg`` must list the prismatic joints in the
    same order as the action term (both use NOVA_PRISMATIC_JOINTS with preserve_order=True).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    force = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    qdot_cmd = env.action_manager.get_term(action_term_name).commanded_velocity
    return torch.sum(torch.abs(force * qdot_cmd), dim=1)
