# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event for the NOVA walking task: joints AND root in one function.

Joints and root are reset in a single event on purpose: the root height depends on the prismatic
positions just sampled, and doing both here makes that ordering explicit instead of relying on the
order in which the EventManager applies separate reset terms.

Replaces the inherited ``reset_joints_by_scale`` (which multiplies the defaults, so NOVA's old all-zero
defaults were never randomized, and then clamps to soft limits) and ``reset_root_state_uniform``
(which used a fixed spawn height regardless of leg length).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# body-frame corners of each foot's collision-mesh AABB, computed once from USD (keyed by body name)
_FOOT_CORNERS: dict[str, torch.Tensor] = {}


def _foot_corners_local(asset: Articulation, foot_body_names: list[str]) -> dict[str, torch.Tensor]:
    """8 AABB corners of each foot's collision mesh, in the foot body frame (read from USD once)."""
    if all(n in _FOOT_CORNERS for n in foot_body_names):
        return _FOOT_CORNERS
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.SimulationContext.instance().stage
    root = stage.GetPrimAtPath(asset.cfg.prim_path.replace("env_.*", "env_0"))
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for body in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if body.GetName() not in foot_body_names or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        inv = xc.GetLocalToWorldTransform(body).GetInverse()
        pts = []
        # the feet are leaf bodies, so every collider under them belongs to them
        for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI) and UsdGeom.Mesh(p):
                rel = xc.GetLocalToWorldTransform(p) * inv
                pts += [list(rel.Transform(Gf.Vec3d(v))) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()]
        pts = torch.tensor(pts, dtype=torch.float32)
        lo, hi = pts.min(0).values, pts.max(0).values
        _FOOT_CORNERS[body.GetName()] = torch.tensor(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], device=asset.device
        )
    missing = [n for n in foot_body_names if n not in _FOOT_CORNERS]
    if missing:
        raise RuntimeError(f"Foot bodies not found for sole correction: {missing}")
    return _FOOT_CORNERS


def reset_nova_walking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    revolute_offset_range: tuple[float, float],
    prismatic_range: tuple[float, float],
    upper_prismatic_names: tuple[str, str],
    lower_prismatic_names: tuple[str, str],
    height_model: tuple[float, float, float],
    sole_clearance: float,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    foot_body_names: list[str] | None = None,
    lower_prismatic_range: tuple[float, float] | None = None,
    prismatic_per_env: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints and root.

    Joints:
      * revolute: default + U(revolute_offset_range), clamped to soft limits
      * prismatic: U(prismatic_range), one sample per L/R pair (upper pair, lower pair); the lower pair uses
        ``lower_prismatic_range`` if given. ``prismatic_per_env`` (one (q_up, q_low) [m] per env index) replaces the
        sampling entirely (evaluation sweeps / teleop).
      * all joint velocities zero
    Root:
      * x/y/yaw from ``pose_range``, roll/pitch zero, velocities from ``velocity_range``
      * z = H0 + k_up*q_up + k_low*q_low + sole_clearance, with ``height_model = (H0, k_up, k_low)``
        from the FK fit at the default revolute pose
      * if ``foot_body_names`` is given: kinematic correction -- ``sim.forward()`` (no physics step),
        find the lowest foot-collision AABB corner and shift the root so it sits exactly
        ``sole_clearance`` above the env origin. Needed because the revolute offsets tilt the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = asset.device

    # ---------------- joints ----------------
    up_ids, _ = asset.find_joints(list(upper_prismatic_names), preserve_order=True)
    low_ids, _ = asset.find_joints(list(lower_prismatic_names), preserve_order=True)
    prismatic_ids = set(up_ids) | set(low_ids)
    rev_ids = [i for i in range(asset.num_joints) if i not in prismatic_ids]

    q = asset.data.default_joint_pos.torch[env_ids].clone()
    q[:, rev_ids] += math_utils.sample_uniform(*revolute_offset_range, (n, len(rev_ids)), dev)
    soft = asset.data.soft_joint_pos_limits.torch[env_ids]
    q[:, rev_ids] = q[:, rev_ids].clamp(soft[:, rev_ids, 0], soft[:, rev_ids, 1])
    if prismatic_per_env is not None:
        fixed = torch.tensor(prismatic_per_env, dtype=torch.float32, device=dev)[env_ids]
        q_up, q_low = fixed[:, 0:1], fixed[:, 1:2]
    else:
        q_up = math_utils.sample_uniform(*prismatic_range, (n, 1), dev)
        q_low = math_utils.sample_uniform(*(lower_prismatic_range or prismatic_range), (n, 1), dev)
    q[:, up_ids] = q_up.expand(n, len(up_ids))
    q[:, low_ids] = q_low.expand(n, len(low_ids))
    asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)

    # ---------------- root ----------------
    default_pose = asset.data.default_root_pose.torch[env_ids].clone()
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    rng = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in keys], device=dev)
    s = math_utils.sample_uniform(rng[:, 0], rng[:, 1], (n, 6), dev)
    h0, k_up, k_low = height_model
    pos = default_pose[:, 0:3] + env.scene.env_origins[env_ids]
    pos[:, 0:2] += s[:, 0:2]
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + h0 + k_up * q_up[:, 0] + k_low * q_low[:, 0] + sole_clearance
    quat = math_utils.quat_mul(default_pose[:, 3:7], math_utils.quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]))
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    rng = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys], device=dev)
    vel = asset.data.default_root_vel.torch[env_ids] + math_utils.sample_uniform(rng[:, 0], rng[:, 1], (n, 6), dev)
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    # ---------------- kinematic sole correction ----------------
    if foot_body_names:
        corners = _foot_corners_local(asset, foot_body_names)
        env.sim.forward()  # PhysX update_articulations_kinematic: link poses from the joint state just written
        lt = wp.to_torch(asset.root_view.get_link_transforms()).reshape(asset.num_instances, asset.num_bodies, 7)
        lt = lt[env_ids.to(lt.device) if isinstance(env_ids, torch.Tensor) else env_ids]
        lowest = torch.full((n,), float("inf"), device=dev)
        for name in foot_body_names:
            b = asset.body_names.index(name)
            p, qb = lt[:, b, 0:3], lt[:, b, 3:7]
            c = corners[name].unsqueeze(0).expand(n, -1, -1)
            w = math_utils.quat_apply(qb.unsqueeze(1).expand(-1, c.shape[1], -1), c) + p.unsqueeze(1)
            lowest = torch.minimum(lowest, w[..., 2].min(dim=1).values)
        pos[:, 2] -= lowest - (env.scene.env_origins[env_ids, 2] + sole_clearance)
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
