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


def effort_sum_ratio_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum_j (tau_j / tau_limit_j)^2 over the joints in ``asset_cfg`` (limits read from the actuators)."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    return torch.sum((torques / limits) ** 2, dim=1)


def effort_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, k: float) -> torch.Tensor:
    """exp(-k * sum_j (tau_j / tau_limit_j)^2) over the joints in ``asset_cfg``.

    Same %-of-own-limit formula as the standing task's effort_reward, but the limits are read at runtime from
    ``robot.data.joint_effort_limits`` (the walking actuator groups' effort_limit_sim: 120/60/36/32 N·m) instead
    of a hard-coded list. Pass the 12 revolute joints only -- prismatics are deliberately excluded. ``k`` is set
    in the task cfg (see the calibration note there).
    """
    return torch.exp(-k * effort_sum_ratio_sq(env, asset_cfg))


def prismatic_power_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_term_name: str = "prismatic_vel",
) -> torch.Tensor:
    """Mechanical power of the leadscrews: sum_j |F_j * target_rate_j| [W] (no regeneration credit).

    F is the actuator's applied force (``robot.data.applied_torque``, N). target_rate is the actual rate of
    change of the clamped position target exposed by :class:`PrismaticVelocityAction` -- i.e. what the
    RobStride 00 really drives the screw at. Not the raw command clip(a, -1, 1) * v_max: that stays non-zero
    when the target is pinned at the software clamp (0.005 / 0.095 m) where the motor does no work, which
    would falsely penalize extreme leg lengths. Not the simulated joint velocity either: PhysX reports a
    biased prismatic joint velocity at standstill, and impacts move the stiff drive without the motor doing
    work. Holding (a=0) or pushing against a clamp costs 0 W. ``asset_cfg`` must list the prismatic joints in the
    same order as the action term (both use NOVA_PRISMATIC_JOINTS with preserve_order=True).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    force = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    target_rate = env.action_manager.get_term(action_term_name).target_rate
    return torch.sum(torch.abs(force * target_rate), dim=1)


def height_tracking_reward(
    env: ManagerBasedRLEnv,
    height_model: tuple[float, float, float],
    upper_prismatic_cfg: SceneEntityCfg,
    lower_prismatic_cfg: SceneEntityCfg,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(z - H(q))^2 / sigma^2): Hip_Base height vs the leg-length-dependent standing height.

    H(q) = H0 + k_up * q_up + k_low * q_low with ``height_model = (H0, k_up, k_low)`` (FK fit at the default revolute
    pose), q_up / q_low the L/R-averaged upper / lower prismatic positions [m]; z is the root height above the env
    origin [m]; ``sigma`` [m].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos.torch
    q_up = q[:, upper_prismatic_cfg.joint_ids].mean(dim=1)
    q_low = q[:, lower_prismatic_cfg.joint_ids].mean(dim=1)
    h0, k_up, k_low = height_model
    target = h0 + k_up * q_up + k_low * q_low
    z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((z - target) ** 2) / sigma**2)


def foot_flat_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Stance-only foot flatness: sum over feet in contact of exp(-tilt^2 / sigma^2), in [0, number of feet].

    tilt [rad] is the angle between the foot sole normal and world Z. The sole normal is the foot body's local +Z
    axis (at the all-zero joint pose the soles are exactly flat with the foot frames world-aligned). Feet in the air
    contribute 0. ``asset_cfg.body_ids`` and ``sensor_cfg.body_ids`` must list the same feet in the same order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]  # (N, F, 4) x,y,z,w
    x, y = quat[..., 0], quat[..., 1]
    n_z = 1.0 - 2.0 * (x**2 + y**2)  # world-Z component of the body +Z axis
    tilt = torch.acos(n_z.clamp(-1.0, 1.0))
    in_contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    return torch.sum(in_contact.float() * torch.exp(-(tilt**2) / sigma**2), dim=1)


def _feet_times(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    data = env.scene.sensors[sensor_cfg.name].data
    ids = sensor_cfg.body_ids
    return data, ids


def contact_balance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_diff: float = 1.0) -> torch.Tensor:
    """|last_air_L - last_air_R| + |last_contact_L - last_contact_R| [s], each difference clamped to ``max_diff``.

    Uses the sensor's last completed swing / stance durations of the two feet listed in ``sensor_cfg`` (L, R): ~0 for an
    alternating gait, large when one leg does all the stance work (one-leg hopping). Returned positive (use a
    negative weight).
    """
    data, ids = _feet_times(env, sensor_cfg)
    air = data.last_air_time.torch[:, ids]
    con = data.last_contact_time.torch[:, ids]
    d_air = (air[:, 0] - air[:, 1]).abs().clamp(max=max_diff)
    d_con = (con[:, 0] - con[:, 1]).abs().clamp(max=max_diff)
    return d_air + d_con


def flight_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 when every foot in ``sensor_cfg`` is out of contact (flight phase / bunny hop), else 0."""
    data, ids = _feet_times(env, sensor_cfg)
    in_contact = data.current_contact_time.torch[:, ids] > 0.0
    return (~in_contact).all(dim=1).float()


def max_swing_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_swing: float = 0.6, cap: float = 1.0
) -> torch.Tensor:
    """sum over feet of clamp(current_air_time - max_swing, 0, cap) [s]: a foot held in the air too long."""
    data, ids = _feet_times(env, sensor_cfg)
    return (data.current_air_time.torch[:, ids] - max_swing).clamp(min=0.0, max=cap).sum(dim=1)


def leg_self_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Number of (left-leg body, right-leg body) pairs pressing on each other with more than ``threshold`` [N].

    ``sensor_cfg`` names a contact sensor on the left-leg bodies filtered against the right-leg bodies, so its
    ``force_matrix_w`` holds one force per L-R body pair. Returned positive (use a negative weight).
    """
    f = env.scene.sensors[sensor_cfg.name].data.force_matrix_w.torch  # (N, B_left, F_right, 3)
    return (torch.linalg.norm(f, dim=-1) > threshold).sum(dim=(1, 2)).float()
