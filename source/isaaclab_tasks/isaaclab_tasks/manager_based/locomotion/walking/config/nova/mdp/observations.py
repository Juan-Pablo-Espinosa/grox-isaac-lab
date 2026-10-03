# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walking-specific observation terms for NOVA_LOWERBODY_V2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def morph_locked(env: ManagerBasedEnv, action_term_name: str = "prismatic") -> torch.Tensor:
    """Morphology-lock flag of the prismatic action term (1 = leg lengths fixed this episode), shape (num_envs, 1)."""
    return env.action_manager.get_term(action_term_name).locked.float().unsqueeze(1)


def prismatic_targets(env: ManagerBasedEnv) -> torch.Tensor:
    """External prismatic driver targets relative to the default length [m], shape (num_envs, 4): U_L, U_R, L_L, L_R.

    Tells a morphology-agnostic policy where the legs are heading (the joint positions say where they are).
    Requires an env with a ``prismatic_driver`` (see mdp/prismatic_driver.py).
    """
    driver = env.prismatic_driver
    default = env.scene["robot"].data.default_joint_pos.torch[:, driver.joint_ids]
    return driver.target - default
