# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Squat variant of the morphology-agnostic NOVA task: height_tracking_reward target lowered to H(q) - squat_offset.

Everything else is identical to :class:`~.morph_agnostic_env_cfg.NovaMorphAgnosticEnvCfg`. The offset is applied to
H0 of the linear height model, i.e. target = (H0 - squat_offset) + K_UP * q_up + K_LOW * q_low.
Termination check: base_height floor 0.47 m vs the squat target at the shortest driven leg (q = 0.005 m):
0.78822 + (0.99671 + 0.97613) * 0.005 - 0.10 = 0.698 m -> 0.228 m margin (0.218 m at q = 0).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .morph_agnostic_env_cfg import NovaMorphAgnosticEnvCfg

SQUAT_OFFSET = 0.10


@configclass
class NovaMorphAgnosticSquatEnvCfg(NovaMorphAgnosticEnvCfg):
    """Morph-agnostic NOVA walking with the base-height target lowered by ``squat_offset``."""

    squat_offset: float = SQUAT_OFFSET
    """Constant offset [m] subtracted from the height_tracking_reward target H(q)."""

    def __post_init__(self):
        super().__post_init__()
        h0, k_up, k_low = self.rewards.height_tracking_reward.params["height_model"]
        self.rewards.height_tracking_reward.params["height_model"] = (h0 - self.squat_offset, k_up, k_low)


@configclass
class NovaMorphAgnosticSquatEnvCfg_PLAY(NovaMorphAgnosticSquatEnvCfg):
    """Smaller, non-randomized variant for play/evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
