"""Configuration for the NOVA_LOWERBODY_V2 prismatic-limbed humanoid (Daedamorph Robotics)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

NOVA_HIP_KNEE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["Hip_Pitch_.*", "Lowerleg_Pitch_.*"],
    effort_limit_sim=120.0,
    velocity_limit_sim=20.0,
    stiffness=25.0,
    damping=0.5,
)

NOVA_HIP_ROLL_YAW_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["Hip_Roll_.*", "Upperleg_Yaw_.*"],
    effort_limit_sim=60.0,
    velocity_limit_sim=20.0,
    stiffness=25.0,
    damping=0.5,
)

NOVA_FEET_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["Feet_Roll_.*", "Feet_Pitch_.*"],
    effort_limit_sim=34.0,
    velocity_limit_sim=15.0,
    stiffness=25.0,
    damping=0.5,
)

NOVA_LEG_LENGTH_LOCKED_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["Upperleg_Prismatic_.*", "Lowerleg_Prismatic_.*"],
    effort_limit_sim=500.0,
    velocity_limit_sim=0.015,
    stiffness=10000.0,
    damping=100.0,
)

##
# Configuration - Articulation.
##

NOVA_LOWERBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "/home/jpech/IsaacLab/source/isaaclab_assets/data/Robots/Daedamorph/"
            "nova_lowerbody_v2/NOVA_LOWERBODY_V2/NOVA_LOWERBODY_V2.usda"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
           disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=0.1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "Upperleg_Prismatic_.*": 0.0,
            "Lowerleg_Prismatic_.*": 0.0,
        },
    ),
    actuators={
        "hip_knee": NOVA_HIP_KNEE_ACTUATOR_CFG,
        "hip_roll_yaw": NOVA_HIP_ROLL_YAW_ACTUATOR_CFG,
        "feet": NOVA_FEET_ACTUATOR_CFG,
        "leg_length_locked": NOVA_LEG_LENGTH_LOCKED_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of the NOVA lower-body humanoid. Leg-length (prismatic) joints are locked at 0.0
(shortest configuration) via a stiff actuator group - not intended to be driven by an RL policy this grade."""