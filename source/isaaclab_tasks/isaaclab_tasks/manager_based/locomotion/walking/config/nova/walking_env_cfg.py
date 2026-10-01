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
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.standing_env_cfg import (
    ObservationsCfg as StandingObservationsCfg,
)

from .mdp import events as walking_events
from .mdp import observations as walking_observations
from .mdp import power as walking_power
from .mdp import rewards as walking_rewards
from .mdp import terminations as walking_terminations
from .mdp.actions import PrismaticLengthActionCfg, PrismaticVelocityActionCfg
from .mdp.contact_sensor_cfg import NestedBodyContactSensorCfg

NOVA_PRISMATIC_JOINTS = [
    "Upperleg_Prismatic_Left_Joint",
    "Upperleg_Prismatic_Right_Joint",
    "Lowerleg_Prismatic_Left_Joint",
    "Lowerleg_Prismatic_Right_Joint",
]
NOVA_FOOT_BODIES = ["Feet_Pitch_Left", "Feet_Pitch_Right"]
# Static ground-plane collider of the flat terrain (TerrainImporterCfg terrain_type="plane").
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"

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
NOVA_EFFORT_K = 0.3466  # ln(2) / 2.0, see RewardsCfg.effort_reward
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


# Fraction of envs whose leg lengths are LOCKED at the reset sample for a whole episode (prismatic actions ignored).
# The other half chooses its leg lengths freely; the locked half forces the policy to walk well at every morphology,
# so the free half's choice is an actual preference rather than the only gait it ever learned.
NOVA_MORPH_LOCK_FRACTION = 0.5


@configclass
class ActionsCfg:
    """16-D action: 12 revolute position targets, then 4 absolute prismatic lengths (rate-limited).

    ActionManager concatenates terms in declaration order. ``joint_pos`` uses preserve_order=False, so its 12
    entries follow articulation joint order; ``prismatic`` preserves NOVA_PRISMATIC_JOINTS order.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=NOVA_REVOLUTE_JOINTS, scale=0.25, use_default_offset=True
    )
    # Runs 1-4 used PrismaticVelocityActionCfg (a velocity command integrated into the target): zero-mean exploration
    # noise random-walks the target into a clamp, and run 4 sat at the 0.005 m stop ~93% of the time.
    prismatic = PrismaticLengthActionCfg(
        asset_name="robot",
        joint_names=NOVA_PRISMATIC_JOINTS,
        max_velocity=NOVA_PRISMATIC_MAX_SPEED,
        q_min=0.005,
        q_max=0.095,
        lock_fraction=NOVA_MORPH_LOCK_FRACTION,
    )


@configclass
class ObservationsCfg(StandingObservationsCfg):
    """Standing's 66-D policy observation + the morphology-lock flag (index 66)."""

    @configclass
    class PolicyCfg(StandingObservationsCfg.PolicyCfg):
        morph_locked = ObsTerm(func=walking_observations.morph_locked, params={"action_term_name": "prismatic"})

    policy: PolicyCfg = PolicyCfg()


_REVOLUTE = SceneEntityCfg("robot", joint_names=NOVA_REVOLUTE_JOINTS, preserve_order=True)


@configclass
class RewardsCfg:
    """Walking rewards. Weight-0 terms are wired and logged but not yet shaping (calibrate from telemetry)."""

    upright_reward = RewTerm(func=standing_rewards.upright_reward, weight=15.0)
    # Same functions as standing (they already track env.command_manager.get_command("base_velocity")); std
    # tightened from standing's drift-penalty values (2.0222 m/s, 1.1555 rad/s) to 0.5 for actual tracking.
    # Weights 7/3 -> 14/6: run 2 converged to standing still (error_vel_xy ~= commanded speed), because walking
    # beat standing by only ~+0.75/step against the static-friendly terms (upright, effort, height, acceleration).
    velocity_xy_reward = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=14.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    velocity_yaw_reward = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=6.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # 12 revolute joints only, limits read from the walking actuator groups at runtime.
    # k = ln(2)/2.0: sum (tau/tau_lim)^2 = 2.0 earns 0.5 (stand p90 0.29 -> 0.904; run 1's saturated split at
    # 10.0 -> 0.031). History: 5.1782 (standing telemetry) was dead on walking torques (3e-23 at run 1's median);
    # 0.0693 (anchored on run 1's saturated median 10.0) barely separated moderate from low effort.
    # OFF for run 5 (replaced by p_elec; kept wired so effort_sum_ratio_sq_mean is still logged).
    effort_reward = RewTerm(
        func=walking_rewards.effort_reward, weight=0.0, params={"asset_cfg": _REVOLUTE, "k": NOVA_EFFORT_K}
    )
    # Stock joint_deviation_hip (joint_deviation_l1 on hip yaw + roll; H1 -0.2, G1 -0.1 vs a lin-vel tracking weight
    # of 1.0). -1.4 = H1's -0.2 x 7 (= G1's -0.1 x our 14x tracking weight). Targets splay exploits.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Hip_Roll_.*", "Upperleg_Yaw_.*"])},
    )
    # TODO: k=0.0000189 was fitted on standing telemetry; recalibrate the anchor on walking telemetry.
    acceleration_reward = RewTerm(
        func=standing_rewards.acceleration_reward, weight=2.0, params={"asset_cfg": _REVOLUTE}
    )
    # Kept wired but disabled: a walking gait is not L/R position-symmetric at every instant.
    symmetry_reward = RewTerm(func=standing_rewards.symmetry_reward, weight=0.0, params={"asset_cfg": _REVOLUTE})
    # OFF for run 5 (the leadscrews are part of p_elec). Was -0.0706 = -0.25 / 3.543 W (run 3's stochastic jitter).
    prismatic_power_reward = RewTerm(
        func=walking_rewards.prismatic_power_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=NOVA_PRISMATIC_JOINTS, preserve_order=True),
            "action_term_name": "prismatic",
        },
    )
    # Electrical power of all 16 motors [W] (mdp/power.py: RobStride datasheet copper loss + mechanical power, no
    # regeneration). The only energy term, so leg length is priced by what it costs per metre at the motors.
    # Weight = -2.0 / 161.2 W, the median P_elec of run 4 (model_14999, deterministic, 64 envs, 20 s) at 0.5 m/s
    # forward (stand 38.5 W, 1.0 m/s 209.2 W): a typical 0.5 m/s step costs ~2.0.
    p_elec = RewTerm(func=walking_power.electrical_power, weight=-0.0124, params={"action_term_name": "prismatic"})
    # Stock biped term (single-stance time capped at threshold, zero when ||cmd_xy|| <= 0.1). Weight 5.25 = 0.375 x
    # velocity_xy (14); stock ratios: H1/G1 rough 0.25, G1 flat 0.75, H1 flat 1.0. Raised from 1.75 (0.25 x 7)
    # after run 2 learned to stand still (air time ~0.0002).
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=5.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES),
        },
    )
    # Gait-shape terms (contact sensor air/contact times on Feet_Pitch_*). Run 3 learned a population-wide one-leg hop:
    # the left foot stayed airborne for up to 19 s, which feet_air_time_positive_biped pays at its 0.4 s cap every
    # single-stance step (min(air_L, contact_R) = 0.4) without any stepping. These counter one-leg hops (balance),
    # bunny hops (flight) and a foot parked in the air (max swing).
    contact_balance = RewTerm(
        func=walking_rewards.contact_balance_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES, preserve_order=True)},
    )
    flight = RewTerm(
        func=walking_rewards.flight_penalty,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES, preserve_order=True)},
    )
    max_swing = RewTerm(
        func=walking_rewards.max_swing_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES, preserve_order=True),
            "max_swing": 0.6,
            "cap": 1.0,
        },
    )
    # Left-leg body touching a right-leg body (self-collisions are on): number of L-R body pairs with > 1 N.
    leg_self_contact = RewTerm(
        func=walking_rewards.leg_self_contact_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("leg_contact"), "threshold": 1.0},
    )
    # Hip_Base height vs the leg-length-dependent standing height H(q); sigma 0.08 m: |z-H| 0.03 -> 0.869,
    # 0.10 -> 0.210, 0.30 -> 8e-7 (run 1's split sat ~0.3 m low).
    height_tracking_reward = RewTerm(
        func=walking_rewards.height_tracking_reward,
        weight=5.0,
        params={
            "height_model": (NOVA_H0, NOVA_K_UP, NOVA_K_LOW),
            "upper_prismatic_cfg": SceneEntityCfg("robot", joint_names=["Upperleg_Prismatic_.*"]),
            "lower_prismatic_cfg": SceneEntityCfg("robot", joint_names=["Lowerleg_Prismatic_.*"]),
            "sigma": 0.08,
        },
    )
    # Built but OFF (weight 0 = not evaluated by the RewardManager).
    # TODO: enable if the policy exploits tilted / rolled feet.
    foot_flat_reward = RewTerm(
        func=walking_rewards.foot_flat_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NOVA_FOOT_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=NOVA_FOOT_BODIES, preserve_order=True),
            "sigma": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    """time_out, bad_tilt (unchanged), a fixed height floor, and illegal contact of any non-foot body."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_tilt = DoneTerm(func=standing_terminations.bad_tilt)
    # Fixed floor (not leg-length dependent): 0.47 m ~= 0.6 * H0 (H0 = 0.788 m, shortest-leg standing height).
    # World-frame root z, valid on the flat plane.
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.47})
    # All 13 non-foot bodies (everything except Feet_Roll_* / Feet_Pitch_*) touching the GROUND with > 5 N. Ground-only
    # (filtered sensor) since self-collisions are on: leg-on-leg contact is penalized, never terminated.
    base_contact = DoneTerm(
        func=walking_terminations.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "ground_contact",
                body_names=["Hip_Base", "Hip_Pitch_.*", "Hip_Roll_.*", "Upperleg_.*", "Lowerleg_.*"],
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
    observations: ObservationsCfg = ObservationsCfg()
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
        # filtered sensors: every body vs the ground collider (illegal-contact termination), and left-leg vs right-leg
        # bodies (leg_self_contact penalty)
        self.scene.ground_contact = NestedBodyContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, filter_prim_paths_expr=[GROUND_COLLIDER]
        )
        self.scene.leg_contact = NestedBodyContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_Left", filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*_Right"]
        )
        # self-collisions on (walking only; the standing task and nova.py keep them off). Legs could pass through each
        # other before (run 3's L-R leg clouds were within 1 cm in 82-90% of samples).
        self.scene.robot.spawn.articulation_props = self.scene.robot.spawn.articulation_props.replace(
            enabled_self_collisions=True
        )

        # -- commands (undo standing's zero pinning)
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1
        cmd.resampling_time_range = (8.0, 12.0)
        # vy / wz ranges stay symmetric: required by the left-right mirror augmentation (mdp/symmetry.py)
        # run 5: forward range 1.5 -> 2.0 m/s (no command curriculum; the installed source has none for velocity)
        cmd.ranges.lin_vel_x = (-0.8, 2.0)
        cmd.ranges.lin_vel_y = (-0.6, 0.6)
        cmd.ranges.ang_vel_z = (-1.5, 1.5)
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


def make_run4_compatible(env_cfg: NovaWalkingEnvCfg) -> NovaWalkingEnvCfg:
    """Turn a current walking cfg into the observation/action interface of runs 1-4 (66-D obs, velocity action).

    For loading old checkpoints (teleop / evaluation only): drops the morph_locked observation, swaps the prismatic
    term back to :class:`PrismaticVelocityActionCfg` (no morphology lock) and restores run 4's forward command range
    (-0.8, 1.5) m/s. Rewards stay as configured. Modifies ``env_cfg`` in place and returns it.
    """
    old = env_cfg.actions.prismatic
    env_cfg.actions.prismatic = PrismaticVelocityActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        max_velocity=old.max_velocity,
        q_min=old.q_min,
        q_max=old.q_max,
    )
    env_cfg.observations.policy.morph_locked = None
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.5)
    return env_cfg
