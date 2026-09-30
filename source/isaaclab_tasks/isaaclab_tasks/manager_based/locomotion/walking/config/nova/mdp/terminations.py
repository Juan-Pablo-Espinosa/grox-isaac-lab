# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walking-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_ground_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when any selected body touches the GROUND with more than ``threshold`` [N] (history max).

    Same as the stock ``isaaclab.envs.mdp.illegal_contact`` but reads the sensor's filtered ``force_matrix_w_history``
    (contacts with the filter prims only, here the ground collider) instead of the net contact force, so leg-on-leg
    self-collisions can never terminate an episode.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.force_matrix_w_history.torch[:, :, sensor_cfg.body_ids]  # (N, T, B, F, 3)
    return (torch.linalg.norm(f, dim=-1).sum(dim=-1).amax(dim=1) > threshold).any(dim=1)
