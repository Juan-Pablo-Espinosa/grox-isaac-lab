# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Morphology-agnostic NOVA walking: leg length is driven externally, the policy only walks.

Subclass of the walking task (same robot, actuators incl. the leadscrew model, default pose, contact sensors,
self-collisions, terminations, reset event, symmetry augmentation). Differences:
  * rewards = run 4 (commit 9852d3b807): effort_reward w=3 (k=0.3466), no p_elec. The run-4 prismatic_power_reward
    is dropped because the policy no longer moves the prismatics (see the task report for the full diff).
  * 12-D action (revolute JointPositionAction only); the four prismatic targets come from an external
    :class:`~.mdp.prismatic_driver.PrismaticDriver` (random schedule: new uniform upper / lower goal every U(2, 8) s,
    rate-limited to 0.035 m/s).
  * observation: run-4 terms (last action now 12-D) + the 4 driver targets -> 66-D.
  * commands: lin_vel_x (-0.8, 2.5) m/s.
height_tracking_reward keeps using the ACTUAL prismatic joint positions for H(q).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.standing.config.nova.standing_env_cfg import (
    NOVA_REVOLUTE_JOINTS,
)
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.standing_env_cfg import (
    ObservationsCfg as StandingObservationsCfg,
)

from .mdp import observations as walking_observations
from .mdp.prismatic_driver import PrismaticDriverCfg
from .walking_env_cfg import NOVA_PRISMATIC_JOINTS, NOVA_PRISMATIC_MAX_SPEED, NovaWalkingEnvCfg


@configclass
class MorphAgnosticActionsCfg:
    """12-D action: revolute position targets only (run 4's joint_pos term)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=NOVA_REVOLUTE_JOINTS, scale=0.25, use_default_offset=True
    )


@configclass
class MorphAgnosticObservationsCfg(StandingObservationsCfg):
    """Run-4 policy observation (last action 12-D) + the 4 external prismatic targets (relative to 0.05 m)."""

    @configclass
    class PolicyCfg(StandingObservationsCfg.PolicyCfg):
        prismatic_targets = ObsTerm(func=walking_observations.prismatic_targets)

    policy: PolicyCfg = PolicyCfg()


@configclass
class NovaMorphAgnosticEnvCfg(NovaWalkingEnvCfg):
    """NOVA walking with externally driven leg length (12-D action)."""

    actions: MorphAgnosticActionsCfg = MorphAgnosticActionsCfg()
    observations: MorphAgnosticObservationsCfg = MorphAgnosticObservationsCfg()
    prismatic_driver: PrismaticDriverCfg = PrismaticDriverCfg(
        joint_names=NOVA_PRISMATIC_JOINTS,
        max_velocity=NOVA_PRISMATIC_MAX_SPEED,
        q_min=0.005,
        q_max=0.095,
        interval_range_s=(2.0, 8.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # rewards back to run 4 (9852d3b807)
        self.rewards.effort_reward.weight = 3.0
        self.rewards.p_elec = None
        # run 4 had prismatic_power_reward (w=-0.0706) on its prismatic action; there is no such action here
        self.rewards.prismatic_power_reward = None
        # commands: forward range up to 2.5 m/s (vy / wz symmetric, as the mirror augmentation requires)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 2.5)


@configclass
class NovaMorphAgnosticEnvCfg_PLAY(NovaMorphAgnosticEnvCfg):
    """Smaller, non-randomized variant for play/evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
