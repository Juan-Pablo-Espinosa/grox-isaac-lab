# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots import NOVA_LOWERBODY_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventsCfg as BaseEventsCfg,
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg as BaseObservationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import events as mdp_events
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import rewards as mdp_rewards
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import terminations as mdp_terminations
from isaaclab_tasks.utils import PresetCfg

# The 12 revolute joints — RL-controlled. The 4 prismatic (leg-length morphology)
# joints are deliberately excluded: verified this session that with them out of the
# action term's joint_names, their PD-target buffer is never written by anything
# (JointPositionAction only ever calls set_joint_position_target_index on its own
# joint_ids; reset_joints_by_scale writes physical joint *state*, not the target
# buffer, and does so multiplicatively against a default of 0.0, which stays 0.0
# regardless of the random scale). Combined with NOVA_LOWERBODY_CFG's very stiff
# leg-length-locked actuator (stiffness=10000, damping=100) and init_state default
# of 0.0 for both prismatic groups, this holds them at nominal extension for the
# entire episode with no explicit event term needed.
NOVA_REVOLUTE_JOINTS = [
    "Hip_Pitch_Left_Joint",
    "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint",
    "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint",
    "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint",
    "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint",
    "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint",
    "Feet_Pitch_Right_Joint",
]

# NOVA's root/base body is "Hip_Base" (confirmed via the Newton shadow-model
# body_label dump this session) — NOT "base" (Anymal's convention, baked into the
# inherited EventsCfg's body_names filters).
NOVA_ROOT_BODY = "Hip_Base"


##
# Physics preset
##


@configclass
class NovaStandingPhysicsCfg(PresetCfg):
    """Flat-terrain physics preset, mirroring AnymalCFlatEnvCfg's PhysicsCfg exactly.

    The inherited LocomotionVelocityRoughEnvCfg.sim uses RoughPhysicsCfg, tuned for
    rough-terrain triangle-mesh contact (Newton branch: njmax=200, nconmax=100,
    cone="pyramidal", impratio=1.0, plus a 1cm shape margin explicitly noted upstream
    as "the single most important Newton setting for rough terrain"). Standing is a
    flat-plane task with none of that complexity -- this mirrors the same swap
    AnymalCFlatEnvCfg makes for its own flat variant.

    Note: the actually-running backend for this task is PhysX (verified via startup
    logs), and PhysxCfg here is byte-identical to RoughPhysicsCfg's PhysX branch
    (gpu_max_rigid_patch_count=10*2**15 either way) -- so this swap changes nothing
    for the current PhysX-backed training and does NOT by itself explain the measured
    foot-penetration (see ground_rest_offset event below for the actual verified fix).
    It's still the architecturally correct thing to do: it matches this repo's own
    flat/rough convention, and matters the moment this task is ever run on the
    newton_mjwarp backend.
    """

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=120, nconmax=15, cone="elliptic", impratio=100, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


##
# Scene definition
##


@configclass
class NovaStandingSceneCfg(MySceneCfg):
    """Standing-task scene: adds an IMU on the robot's root/hip base body.

    Not required for this grade's reward/termination logic, but wired into
    observations now (see ObservationsCfg below) since later grades will need it.
    """

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/Hip_Base")


##
# MDP settings
##


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Observations: base terms (incl. all 16 joints, prismatic included) + IMU."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")})
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")})

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP: the 12 revolute joints only."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NOVA_REVOLUTE_JOINTS,
        # 0.25 rad, not Anymal's 0.5 — verified against NOVA_LOWERBODY_CFG's actual
        # joint limits rather than copied blindly. NOVA's tightest revolute joint is
        # Feet_Roll at +/-0.44 rad; scale=0.5 there would let a single unit-magnitude
        # action span ~114% of its entire range. scale=0.25 keeps a unit action
        # within Feet_Roll's half-range with margin (57%), while still giving the
        # wider hip/knee pitch joints (+/-2.0-2.09 rad) plenty of room to move.
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the standing task — see mdp/rewards.py for derivations.

    Ceiling is 42 (15 upright + 10 height + 3 effort + 2 acceleration + 7
    velocity_xy + 3 velocity_yaw + 2 symmetry).
    """

    upright_reward = RewTerm(func=mdp_rewards.upright_reward, weight=15.0)
    height_reward = RewTerm(func=mdp_rewards.height_reward, weight=10.0)

    # asset_cfg passed explicitly via params for effort_reward/acceleration_reward/
    # symmetry_reward (below) -- REQUIRED, not optional. The manager's own
    # SceneEntityCfg-resolution pass only walks RewTermCfg.params, never a
    # function's own default parameter value (verified this session: relying on
    # the default left asset_cfg.joint_ids unresolved at slice(None), silently
    # summing over all 16 joints -- including the 4 stiff prismatic ones -- instead
    # of the intended 12 revolute joints, undetected until symmetry_reward's
    # element-wise indexing raised a hard shape error against it; see
    # mdp/rewards.py's module-level comment for the full writeup).
    effort_reward = RewTerm(
        func=mdp_rewards.effort_reward,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NOVA_REVOLUTE_JOINTS, preserve_order=True)},
    )
    acceleration_reward = RewTerm(
        func=mdp_rewards.acceleration_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NOVA_REVOLUTE_JOINTS, preserve_order=True)},
    )

    # Closes the reward gap that let the policy satisfy upright_reward/height_reward
    # via controlled backward toppling + horizontal sliding rather than genuinely
    # standing in place (JP caught this live: error_vel_xy ~3.3-3.8 m/s and
    # error_vel_yaw ~2.0 rad/s, sustained and completely unpunished, for the entire
    # first 1500-iteration run). mdp.track_lin_vel_xy_exp / track_ang_vel_z_exp
    # (isaaclab.envs.mdp.rewards, re-exported here) already compute exactly this
    # error against the base_velocity command -- pinned to all-zero ranges (see
    # __post_init__ below), verified to genuinely stay zero for every env despite
    # the inherited heading_command=True (its heading-error-driven override is
    # torch.clip'd to ranges.ang_vel_z, which is (0.0, 0.0)) -- so these terms were
    # simply missing from RewardsCfg, not broken. exp(-error^2/std^2) form, verified
    # against isaaclab/envs/mdp/rewards.py source directly. std values are JP's
    # conversion from k=1/std^2, anchored to the actual observed drift magnitude
    # (xy~3.5 m/s, yaw~2.0 rad/s) targeting reward=0.05 there: std_xy=2.0222 (from
    # k=0.24455), std_yaw=1.1555 (from k=0.748933) -- verified both land reward
    # ~0.0500 at their anchors and ~0.95-0.98 for small natural sway (<0.25 units).
    velocity_xy_reward = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=7.0, params={"command_name": "base_velocity", "std": 2.0222}
    )
    velocity_yaw_reward = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 1.1555}
    )

    # Rewards L/R torque symmetry using the EMPIRICALLY VERIFIED per-pair sign
    # convention (mdp/rewards.py's symmetry_reward docstring has the full
    # derivation) -- 4 of 6 pairs are opposite-sign (Hip_Pitch, Hip_Roll,
    # Upperleg_Yaw, Lowerleg_Pitch/the knee), only 2 are same-sign (Feet_Roll,
    # Feet_Pitch). k=0.002346 anchored to the real symmetry_error p90 (295.44
    # N*m^2, using this same corrected convention) from the converged
    # model_1550.pt checkpoint, targeting reward=0.5 there.
    symmetry_reward = RewTerm(
        func=mdp_rewards.symmetry_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NOVA_REVOLUTE_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the standing task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_tilt = DoneTerm(func=mdp_terminations.bad_tilt)
    bad_height = DoneTerm(func=mdp_terminations.bad_height)


@configclass
class EventsCfg(BaseEventsCfg):
    """Adds startup events: ground rest_offset (foot-penetration fix) and the
    real-hardware asymmetric knee joint limits (Issue 3).

    Task-level, robot-config-untouched fixes -- see mdp/events.py for the full
    root-cause writeups and scripts/tools/measure_foot_penetration.py for the
    penetration measurement methodology.
    """

    ground_rest_offset = EventTerm(
        func=mdp_events.set_ground_rest_offset,
        mode="startup",
        params={"prim_path": "/World/ground", "rest_offset": 0.003, "contact_offset": 0.005},
    )

    # Real hardware data: knees only bend one direction; left/right are mirrored.
    # The prior [-2.090, 2.090] symmetric range was a known placeholder (flagged
    # since Grade 3). Applied via write_joint_position_limit_to_sim_index at
    # startup rather than touching the USD/URDF -- see mdp/events.py.
    knee_joint_limits = EventTerm(
        func=mdp_events.set_asymmetric_joint_pos_limits,
        mode="startup",
        params={
            "limits": {
                "Lowerleg_Pitch_Left_Joint": (0.0, 1.85),
                "Lowerleg_Pitch_Right_Joint": (-1.85, 0.0),
            }
        },
    )


##
# Environment configuration
##


@configclass
class NovaStandingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """NOVA_LOWERBODY_V2 standing/balance task.

    Architecturally "walking with commanded velocity pinned to zero": built on the
    shared LocomotionVelocityRoughEnvCfg base (flat-terrain override, same pattern
    as AnymalCFlatEnvCfg) so domain randomization, contact-sensor scaffolding, and
    a future command-conditioned walking grade are inherited/natural extensions
    rather than a rebuild.
    """

    sim: SimulationCfg = SimulationCfg(physics=NovaStandingPhysicsCfg())
    scene: NovaStandingSceneCfg = NovaStandingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -- robot
        self.scene.robot = NOVA_LOWERBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Verified spawn height 0.82m (matches test_nova_standing.py and the
        # height_reward / bad_height target_height default) — NOT the raw
        # NOVA_LOWERBODY_CFG default of 0.85m. ArticulationCfg.replace() deep-copies
        # nested fields (verified via configclass's _custom_post_init), so this does
        # not mutate the shared NOVA_LOWERBODY_CFG module-level singleton.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.82)

        # -- flat terrain (pure standing task, no rough-terrain locomotion yet) —
        # mirrors AnymalCFlatEnvCfg's override pattern exactly.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # -- contact sensor: disabled for this grade. NOVA's USD authors each rigid
        # body as a USD-prim CHILD of its kinematic parent (Hip_Base -> Hip_Pitch_Left
        # -> Hip_Roll_Left -> ...), not as flat siblings. isaaclab.sim.schemas.
        # activate_contact_sensors() (schemas.py:687-693) explicitly assumes rigid
        # bodies are never nested inside one another ("nested rigid bodies are not
        # allowed by SDK") and stops descending the tree at the first RigidBodyAPI
        # match per branch -- so for NOVA it only ever tags the root body ("Hip_Base")
        # with PhysxContactReportAPI, leaving the other 16 untagged. ContactSensor's
        # own body-discovery walk is correctly recursive (verified: contact_sensor.py
        # uses get_all_matching_child_prims, which does NOT stop at a match) -- it
        # just has nothing to find beyond Hip_Base. This activation runs during scene
        # spawn, before sim.reset(), which is before ANY EventTermCfg (even
        # mode="startup") can run -- so there is no clean per-task workaround; a real
        # fix means patching either core Isaac Lab (affects every robot) or the
        # shared NOVA USD asset (affects every future grade). Deferred rather than
        # rushed here since Grade 4's reward/termination design never references
        # contact_forces. Revisit when a future grade actually needs foot contact.
        self.scene.contact_forces = None

        # -- retarget the inherited EventsCfg's body_names="base" filters to NOVA's
        # actual root body name. base_com is wrapped in a PresetCfg (physx vs.
        # newton_mjwarp); the physx/default branch holds the real EventTerm on
        # .default, newton_mjwarp is already None so nothing to retarget there.
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=NOVA_ROOT_BODY)
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=NOVA_ROOT_BODY
        )
        if self.events.base_com is not None and self.events.base_com.default is not None:
            self.events.base_com.default.params["asset_cfg"] = SceneEntityCfg("robot", body_names=NOVA_ROOT_BODY)

        # -- pure standing: pin all commanded velocities to zero rather than
        # removing the command manager, so a future command-conditioned walking
        # grade is a natural extension of this same cfg.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # -- episode length: untrained PD-hold topples by t=1.25s (verified this
        # session); 10s gives ample margin once the policy is actually balancing.
        self.episode_length_s = 10.0


class NovaStandingEnvCfg_PLAY(NovaStandingEnvCfg):
    """Smaller, non-randomized variant for play/evaluation — mirrors AnymalCFlatEnvCfg_PLAY."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
