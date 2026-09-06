# Termination terms for NOVA_LOWERBODY_V2 standing task — Grade 4
#
# OR-based logic: tilt failure OR height failure ends the episode.
# Chosen deliberately over AND so that a symmetric buckle-down (crouching low
# without much tilt) is caught independently of a sideways/backward topple.
# Both thresholds verified against the real PD-hold fall trajectory from
# test_nova_standing.py (both fire together at t=1.00s in that trajectory,
# neither firing prematurely during the stable t=0-0.75s window).

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def bad_tilt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    up_z_threshold: float = 0.7071,  # 45 deg from vertical
) -> torch.Tensor:
    """True if the robot has tilted past 45 degrees from vertical.

    up_z = 1 - 2*(x^2 + y^2), quaternion order (x,y,z,w) verified against
    Isaac Lab's internal convention this session.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    x, y = quat[:, 0], quat[:, 1]
    up_z = 1.0 - 2.0 * (x**2 + y**2)
    return up_z < up_z_threshold


def bad_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.596,  # 50% of hip-to-knee span (0.8195 -> 0.3732)
) -> torch.Tensor:
    """True if the robot's root height has dropped below the crouch limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return height < height_threshold
