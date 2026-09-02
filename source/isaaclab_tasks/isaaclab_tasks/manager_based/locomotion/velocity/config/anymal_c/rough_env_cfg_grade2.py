from isaaclab.utils.configclass import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import math

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.mdp import rewards as mdp_grade2

@configclass
class RewardsCfg:
    """From scratch reward terms - grade 2"""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    energy_effort = RewTerm(
        func=mdp_grade2.energy_effort_bounded,
        weight=1.5,
        params={"k": 5069.0},
    )

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-3,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )

    symmetry = RewTerm(
        func=mdp_grade2.symmetry_penalty,
        weight=-1.0,
        params={"k": 100.0},
    )


@configclass
class AnymalCRoughEnvCfg_Grade2(LocomotionVelocityRoughEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

