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
  * actuators matched to the hardware (mdp/hardware.py, mdp/actuators.py): explicit PD at the physics rate with the
    RobStride 48 V torque-speed envelopes (hips / knees RS04, roll RS03, yaw RS06), the ankle as 2x RS02 through the
    pitch / roll linkage with a 6 N·m per-motor (coupled) limit, and rotor inertia as joint armature. Same PD gains as
    the walking task. The leadscrew drive is unchanged (implicit, k = 3e5).
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
from .mdp.hardware import (
    ANKLE_N_PITCH,
    ANKLE_N_ROLL,
    ARMATURE,
    RS02_CONTINUOUS_TORQUE,
    AnkleLinkagePDActuatorCfg,
    TorqueSpeedPDActuatorCfg,
    tn_curve,
)
from .mdp.power import ROBSTRIDE
from .mdp.prismatic_driver import PrismaticDriverCfg
from .walking_env_cfg import (
    NOVA_PRISMATIC_JOINTS,
    NOVA_PRISMATIC_MAX_SPEED,
    NOVA_REVOLUTE_GAINS,
    NovaWalkingEnvCfg,
    nova_walking_actuators,
)

# Velocity caps (PhysX max joint velocity) stay below the module no-load speeds: hips / knees 20 < 20.94 (RS04),
# roll 20 < 20.42 (RS03), yaw 20 (RS06 no-load 50.27), ankle pitch 42.94 / N_p, ankle roll min(42.94 / N_r, 15) rad/s.
ANKLE_PITCH_VELOCITY_LIMIT = ROBSTRIDE["RS02"].no_load_speed / ANKLE_N_PITCH
ANKLE_ROLL_VELOCITY_LIMIT = min(ROBSTRIDE["RS02"].no_load_speed / ANKLE_N_ROLL, 15.0)


def _armature(*patterns: str) -> dict[str, float]:
    return {p: ARMATURE[p] for p in patterns}


def nova_hardware_actuators() -> dict:
    """Explicit hardware-matched revolute actuators + the walking task's implicit leadscrew drive."""
    g = NOVA_REVOLUTE_GAINS

    def motor(joints, model, kp, kd, effort, velocity_sim):
        return TorqueSpeedPDActuatorCfg(
            joint_names_expr=list(joints),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            effort_limit_sim=effort,  # also what joint_effort_limits reports (effort reward, metrics)
            velocity_limit=ROBSTRIDE[model].no_load_speed,
            velocity_limit_sim=velocity_sim,
            armature=_armature(*joints),
            tn_curve=tn_curve(model),
        )

    ankle_limits = {
        "Feet_Pitch_.*": 2.0 * RS02_CONTINUOUS_TORQUE * ANKLE_N_PITCH,
        "Feet_Roll_.*": 2.0 * RS02_CONTINUOUS_TORQUE * ANKLE_N_ROLL,
    }
    return {
        "hip_pitch_knee": motor(
            ("Hip_Pitch_.*", "Lowerleg_Pitch_.*"), "RS04", g["hip_pitch_knee"][0], g["hip_pitch_knee"][1], 120.0, 20.0
        ),
        "hip_roll": motor(("Hip_Roll_.*",), "RS03", g["hip_roll"][0], g["hip_roll"][1], 60.0, 20.0),
        "upperleg_yaw": motor(("Upperleg_Yaw_.*",), "RS06", g["upperleg_yaw"][0], g["upperleg_yaw"][1], 36.0, 20.0),
        "feet": AnkleLinkagePDActuatorCfg(
            joint_names_expr=["Feet_Roll_.*", "Feet_Pitch_.*"],
            stiffness=g["feet"][0],
            damping=g["feet"][1],
            effort_limit=ankle_limits,
            effort_limit_sim=ankle_limits,
            velocity_limit={"Feet_Pitch_.*": ANKLE_PITCH_VELOCITY_LIMIT, "Feet_Roll_.*": ANKLE_ROLL_VELOCITY_LIMIT},
            velocity_limit_sim={"Feet_Pitch_.*": ANKLE_PITCH_VELOCITY_LIMIT, "Feet_Roll_.*": ANKLE_ROLL_VELOCITY_LIMIT},
            armature=_armature("Feet_Pitch_.*", "Feet_Roll_.*"),
            n_pitch=ANKLE_N_PITCH,
            n_roll=ANKLE_N_ROLL,
            motor_torque_limit=RS02_CONTINUOUS_TORQUE,
            motor_tn_curve=tn_curve("RS02"),
        ),
        "leg_length": nova_walking_actuators()["leg_length"],
    }


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
    ankle_transmission: tuple[float, float] = (ANKLE_N_PITCH, ANKLE_N_ROLL)
    """Ankle linkage ratios (N_p, N_r) used by the electrical power metric (mdp/power.py)."""
    prismatic_driver: PrismaticDriverCfg = PrismaticDriverCfg(
        joint_names=NOVA_PRISMATIC_JOINTS,
        max_velocity=NOVA_PRISMATIC_MAX_SPEED,
        q_min=0.005,
        q_max=0.095,
        interval_range_s=(2.0, 8.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # hardware-matched actuators (replaces the walking task's implicit revolute drives)
        self.scene.robot = self.scene.robot.replace(actuators=nova_hardware_actuators())
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
