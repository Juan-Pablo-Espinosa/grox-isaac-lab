# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NOVA_LOWERBODY_V2 command-conditioned walking with all 16 joints in the action space.

Built as a SUBCLASS of the standing task's NovaStandingEnvCfg (not a copy): standing already carries
several verified, robot-specific fixes that walking needs unchanged -- flat-plane PhysX preset, IMU
observation, ground rest_offset (foot-penetration fix), real asymmetric knee limits, and the
"base" -> "Hip_Base" body retargeting of the inherited base events. Subclassing keeps a single source
of truth for those. Everything walking changes is overridden here (actuators, default pose, spawn,
contact sensor, actions, commands, rewards, terminations, reset events); walking defines its OWN
ActionsCfg / RewardsCfg / TerminationsCfg classes, so no standing reward or termination is inherited.
Nothing in the standing task, its mdp modules, or nova.py is modified.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import rewards as standing_rewards
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import terminations as standing_terminations
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.standing_env_cfg import (
    NOVA_REVOLUTE_JOINTS,
    NovaStandingEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.standing_env_cfg import (
    EventsCfg as StandingEventsCfg,
)

from .mdp import events as walking_events
from .mdp import rewards as walking_rewards
from .mdp.actions import PrismaticVelocityActionCfg
from .mdp.contact_sensor_cfg import NestedBodyContactSensorCfg

NOVA_PRISMATIC_JOINTS = [
    "Upperleg_Prismatic_Left_Joint",
    "Upperleg_Prismatic_Right_Joint",
    "Lowerleg_Prismatic_Left_Joint",
    "Lowerleg_Prismatic_Right_Joint",
]
NOVA_FOOT_BODIES = ["Feet_Pitch_Left", "Feet_Pitch_Right"]

##
# Default pose (solved by pure-kinematics FK, see the task report / scripts in the commit message)
##
# Prismatics at mid-travel (0.05 m of the 0..0.1 m hard range), knees bent 0.30 rad (flexion is L+/R-,
# matching the startup knee limits L [0, 1.85], R [-1.85, 0]). Hip_Pitch / Feet_Pitch were solved per leg
# so the sole normal is parallel to world Z (achieved tilt 0.07 deg L / 0.01 deg R) and Hip_Base is over the
# midpoint of the two AREA-weighted sole centroids (dx 0.02 mm; dy 0.72 mm, the URDF's knee-origin asymmetry,
# which pitch joints cannot remove). The first solve used a vertex-weighted sole centroid, biased 23 mm toward
# the heel by the STL tessellation, which left Hip_Base 23 mm behind the sole centre. Signs were measured,
# not assumed: Hip_Pitch is OPPOSITE-sign L/R, Feet_Pitch is SAME-sign L/R (matches symmetry_reward).
NOVA_WALKING_DEFAULT_JOINT_POS = {
    "Upperleg_Prismatic_.*": 0.05,
    "Lowerleg_Prismatic_.*": 0.05,
    "Lowerleg_Pitch_Left_Joint": 0.30,
    "Lowerleg_Pitch_Right_Joint": -0.30,
    "Hip_Pitch_Left_Joint": 0.08811,
    "Hip_Pitch_Right_Joint": -0.08842,
    "Feet_Pitch_Left_Joint": 0.21891,
    "Feet_Pitch_Right_Joint": 0.21863,
    "Hip_Roll_.*": 0.0,
    "Upperleg_Yaw_.*": 0.0,
    "Feet_Roll_.*": 0.0,
}
# Standing height of Hip_Base above the lowest sole point at the default revolute pose, as a function of
# the prismatic positions: H = H0 + K_UP*q_up + K_LOW*q_low (exactly linear, fit residual < 0.001 mm).
# K_UP/K_LOW < 1 are the cosines of the thigh / shank tilt at this pose.
NOVA_H0 = 0.78822
NOVA_K_UP = 0.99671
NOVA_K_LOW = 0.97613
NOVA_SPAWN_CLEARANCE = 0.01
NOVA_DEFAULT_HEIGHT = NOVA_H0 + (NOVA_K_UP + NOVA_K_LOW) * 0.05  # 0.88686 m

##
# Actuators
##
# Prismatic leadscrew drive. Stiffness from a standing sweep at the default pose (16 envs, 2 s, revolute gains
# raised so the pose holds): steady-state sag mean/max 1e5: 0.81/1.17 mm, 3e5: 0.28/0.40 mm, 1e6: 0.09/0.12 mm,
# all stable. 3e5 is the lowest k with sag < 0.5 mm. Damping keeps the previous grade's damping RATIO:
# d = 100 * sqrt(k / 1e4) (k=1e4 -> d=100).
NOVA_PRISMATIC_STIFFNESS = 3.0e5
# Leadscrew hardware: T12x8 4-start trapezoidal screw (8 mm lead, bronze nut) driven directly by a RobStride 00
# (10:1, 5 N·m rated / 14 N·m peak, 315 rpm no-load): ~0.035-0.041 m/s at walking loads, ~2270 N continuous.
NOVA_PRISMATIC_MAX_SPEED = 0.035
NOVA_PRISMATIC_EFFORT_LIMIT = 2250.0
# PhysX joint velocity cap for the prismatics. NOT the leadscrew speed limit -- that is enforced on the
# position target by PrismaticVelocityAction. Measured: with velocity_limit_sim=0.015 the PhysX maxJointVelocity
# clamp lets body weight backdrive the lower prismatics continuously at exactly -0.015 m/s (q 0.048 -> 0.029 m in
# <1 s, sag grew WITH stiffness to 11/32/42 mm) -- the opposite of a non-backdrivable leadscrew. With the cap at
# 0.1 or 1.0 m/s the drive holds (sag = load/k) and applied_torque matches the PhysX joint force. 1.0 never binds
# in normal operation (peak transient |qdot| ~0.25 m/s at landing) while keeping a finite safety bound.
NOVA_PRISMATIC_SIM_VELOCITY_LIMIT = 1.0


def prismatic_damping(stiffness: float) -> float:
    """Damping that preserves the previous (k=1e4, d=100) damping ratio: d = 100*sqrt(k/1e4)."""
    return 100.0 * math.sqrt(stiffness / 1.0e4)


# Revolute gains are SIM-ONLY (hardware gains unknown). Lowest per-group kp for which 64 envs at the exact default
# pose, zero action, zero root reset velocity all stay up for 5 s (0 bad_tilt / illegal_contact). Hip/knee, roll and
# yaw come from a per-group descent from kp=2000 plus a combined check (minima 50/25/25 together with feet 150
# failed 56/64 at the first default pose; one ladder step up passed 64/64 twice). Feet were re-swept after the pose
# was re-solved on the area-weighted sole: kp 200 and 150 pass 64/64 (peak |tau| 52% / 69% of the 32 N*m limit),
# 100 fails (34/64). kd = 2*sqrt(kp*I_eff) (damping ratio 1), I_eff = diagonal of the floating-base joint-space mass
# matrix at the default pose [kg*m^2]: Hip_Pitch 1.085, Lowerleg_Pitch 0.166, Hip_Roll 1.246, Upperleg_Yaw 0.0335,
# Feet_Roll 0.00086, Feet_Pitch 0.00334.
NOVA_REVOLUTE_GAINS = {
    "hip_pitch_knee": (75.0, {"Hip_Pitch_.*": 18.04, "Lowerleg_Pitch_.*": 7.05}),
    "hip_roll": (35.0, 13.21),
    "upperleg_yaw": (35.0, 2.17),
    "feet": (150.0, {"Feet_Roll_.*": 0.72, "Feet_Pitch_.*": 1.42}),
}


def nova_walking_actuators(prismatic_stiffness: float = NOVA_PRISMATIC_STIFFNESS) -> dict[str, ImplicitActuatorCfg]:
    """Actuator groups split by effort limit [N·m / N], with the sim-only revolute gains above."""
    g = NOVA_REVOLUTE_GAINS
    return {
        "hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=["Hip_Pitch_.*", "Lowerleg_Pitch_.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.0,
            stiffness=g["hip_pitch_knee"][0],
            damping=g["hip_pitch_knee"][1],
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["Hip_Roll_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.0,
            stiffness=g["hip_roll"][0],
            damping=g["hip_roll"][1],
        ),
        "upperleg_yaw": ImplicitActuatorCfg(
            joint_names_expr=["Upperleg_Yaw_.*"],
            effort_limit_sim=36.0,
            velocity_limit_sim=20.0,
            stiffness=g["upperleg_yaw"][0],
            damping=g["upperleg_yaw"][1],
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["Feet_Roll_.*", "Feet_Pitch_.*"],
            effort_limit_sim=32.0,
            velocity_limit_sim=15.0,
            stiffness=g["feet"][0],
            damping=g["feet"][1],
        ),
        "leg_length": ImplicitActuatorCfg(
            joint_names_expr=["Upperleg_Prismatic_.*", "Lowerleg_Prismatic_.*"],
            effort_limit_sim=NOVA_PRISMATIC_EFFORT_LIMIT,
            velocity_limit_sim=NOVA_PRISMATIC_SIM_VELOCITY_LIMIT,
            stiffness=prismatic_stiffness,
            damping=prismatic_damping(prismatic_stiffness),
        ),
    }


##
# MDP settings
##


@configclass
class ActionsCfg:
    """16-D action: 12 revolute position targets, then 4 prismatic velocity commands.

    ActionManager concatenates terms in declaration order. ``joint_pos`` uses preserve_order=False, so its 12
    entries follow articulation joint order; ``prismatic_vel`` preserves NOVA_PRISMATIC_JOINTS order.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=NOVA_REVOLUTE_JOINTS, scale=0.25, use_default_offset=True
    )
    prismatic_vel = PrismaticVelocityActionCfg(
        asset_name="robot",
        joint_names=NOVA_PRISMATIC_JOINTS,
        max_velocity=NOVA_PRISMATIC_MAX_SPEED,
        q_min=0.005,
        q_max=0.095,
    )


_REVOLUTE = SceneEntityCfg("robot", joint_names=NOVA_REVOLUTE_JOINTS, preserve_order=True)


@configclass
class RewardsCfg:
    """Walking rewards. Weight-0 terms are wired and logged but not yet shaping (calibrate from telemetry)."""

    upright_reward = RewTerm(func=standing_rewards.upright_reward, weight=15.0)
    # Same functions as standing (they already track env.command_manager.get_command("base_velocity")); std
    # tightened from standing's drift-penalty values (2.0222 m/s, 1.1555 rad/s) to 0.5 for actual tracking.
    velocity_xy_reward = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=7.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    velocity_yaw_reward = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # 12 revolute joints only, limits read from the walking actuator groups at runtime.
    effort_reward = RewTerm(func=walking_rewards.effort_reward, weight=3.0, params={"asset_cfg": _REVOLUTE})
    # TODO: k=0.0000189 was fitted on standing telemetry; recalibrate the anchor on walking telemetry.
    acceleration_reward = RewTerm(
        func=standing_rewards.acceleration_reward, weight=2.0, params={"asset_cfg": _REVOLUTE}
    )
    # Kept wired but disabled: a walking gait is not L/R position-symmetric at every instant.
    symmetry_reward = RewTerm(func=standing_rewards.symmetry_reward, weight=0.0, params={"asset_cfg": _REVOLUTE})
    # Log-only term: RewardManager skips weight-0 terms entirely, so a tiny weight keeps it computed and logged
    # (Episode_Reward/<term> = weight * mean episode sum; divide by the weight for the raw value).
    # Leadscrew power sum|F*target_rate| [W] (0 when holding or pinned at a clamp); negative sign so scaling the
    # weight up later penalizes.
    prismatic_power_reward = RewTerm(
        func=walking_rewards.prismatic_power_reward,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=NOVA_PRISMATIC_JOINTS, preserve_order=True),
            "action_term_name": "prismatic_vel",
        },
    )
    # Stock biped term (single-stance time capped at threshold, zero when ||cmd_xy|| <= 0.1). Weight = 0.25 x
    # velocity_xy (7), the stock H1/G1 ROUGH ratio (0.25 / 1.0); the stock FLAT variants use 0.75-1.0x.
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=1.75,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES),
        },
    )


@configclass
class TerminationsCfg:
    """time_out, bad_tilt (unchanged), and illegal contact of non-foot bodies with anything."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_tilt = DoneTerm(func=standing_terminations.bad_tilt)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Hip_Base", "Upperleg_Yaw_.*", "Upperleg_Prismatic_.*", "Lowerleg_Pitch_.*"],
            ),
            "threshold": 5.0,
        },
    )


@configclass
class EventsCfg(StandingEventsCfg):
    """Standing's startup events (ground rest_offset, knee limits, base mass/com) + one combined reset."""

    push_robot = None
    reset_robot_joints = None
    reset_base = None
    reset_nova = EventTerm(
        func=walking_events.reset_nova_walking,
        mode="reset",
        params={
            "revolute_offset_range": (-0.1, 0.1),
            "prismatic_range": (0.005, 0.095),
            "upper_prismatic_names": ("Upperleg_Prismatic_Left_Joint", "Upperleg_Prismatic_Right_Joint"),
            "lower_prismatic_names": ("Lowerleg_Prismatic_Left_Joint", "Lowerleg_Prismatic_Right_Joint"),
            "height_model": (NOVA_H0, NOVA_K_UP, NOVA_K_LOW),
            "sole_clearance": NOVA_SPAWN_CLEARANCE,
            "foot_body_names": NOVA_FOOT_BODIES,
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


##
# Environment configuration
##


@configclass
class NovaWalkingEnvCfg(NovaStandingEnvCfg):
    """NOVA_LOWERBODY_V2 command-conditioned walking (16-DOF action)."""

    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # standing's post-init: robot, flat terrain, base-event retargeting, zeroed commands, contact sensor off
        super().__post_init__()

        # -- robot: walking actuators, default pose, and a spawn func that enables contact reporting on
        # every nested rigid body (see mdp/spawn.py). All overrides are on this task's copy only.
        self.scene.robot = self.scene.robot.replace(
            # string reference: mdp/spawn.py pulls in pxr-level spawner modules, and task configs are imported
            # before Kit is launched by the training scripts
            spawn=self.scene.robot.spawn.replace(
                func=f"{__package__}.mdp.spawn:spawn_usd_with_nested_contact_reporting"
            ),
            actuators=nova_walking_actuators(),
        )
        self.scene.robot.init_state.joint_pos = dict(NOVA_WALKING_DEFAULT_JOINT_POS)
        self.scene.robot.init_state.pos = (0.0, 0.0, NOVA_DEFAULT_HEIGHT + NOVA_SPAWN_CLEARANCE)

        # -- contact sensor back on: same settings as the stock locomotion preset (all bodies, history 3, air
        # time), but the nested-body-aware PhysX sensor (see mdp/contact_sensor.py). PhysX backend only.
        self.scene.contact_forces = NestedBodyContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
        )

        # -- commands (undo standing's zero pinning)
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1
        cmd.resampling_time_range = (12.0, 15.0)
        cmd.ranges.lin_vel_x = (-0.5, 1.0)
        cmd.ranges.lin_vel_y = (-0.4, 0.4)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = None  # unused without heading control (avoids the command term's warning)

        self.episode_length_s = 20.0


@configclass
class NovaWalkingEnvCfg_PLAY(NovaWalkingEnvCfg):
    """Smaller, non-randomized variant for play/evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
