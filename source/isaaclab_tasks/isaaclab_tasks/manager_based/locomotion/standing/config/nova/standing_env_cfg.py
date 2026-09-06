# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots import NOVA_LOWERBODY_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg as BaseObservationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import rewards as mdp_rewards
from isaaclab_tasks.manager_based.locomotion.standing.config.nova.mdp import terminations as mdp_terminations

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
    """Reward terms for the standing task — see mdp/rewards.py for derivations."""

    upright_reward = RewTerm(func=mdp_rewards.upright_reward, weight=15.0)
    height_reward = RewTerm(func=mdp_rewards.height_reward, weight=10.0)
    effort_reward = RewTerm(func=mdp_rewards.effort_reward, weight=3.0)
    acceleration_reward = RewTerm(func=mdp_rewards.acceleration_reward, weight=2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the standing task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_tilt = DoneTerm(func=mdp_terminations.bad_tilt)
    bad_height = DoneTerm(func=mdp_terminations.bad_height)


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

    scene: NovaStandingSceneCfg = NovaStandingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
