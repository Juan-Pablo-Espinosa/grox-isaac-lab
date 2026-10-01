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
