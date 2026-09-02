from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy_effort_bounded(
    env: ManagerBasedRLEnv,
    k: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward term: bounded energy-effort penalty, 1/(1 + sum(torque^2)/k)."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_sum_sq = torch.sum(torque**2, dim=1) 
    reward_term = 1 / (1 + torque_sum_sq/k)
    return reward_term

def symmetry_penalty(
    env: ManagerBasedRLEnv,
    k: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded left-right torque asymmetry penalty, x/(x+k)."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque
    names = asset.data.joint_names

    left_idx = [names.index(n) for n in names if n.startswith("L")]
    right_idx = [names.index(n) for n in names if n.startswith("R")]

    left_torque = torque[:, left_idx]
    right_torque = torque[:, right_idx]

    raw = torch.sum((left_torque - right_torque) ** 2, dim=1)
    return raw / (raw + k)
