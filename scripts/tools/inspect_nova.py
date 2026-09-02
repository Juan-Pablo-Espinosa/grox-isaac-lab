"""Load NOVA_LOWERBODY_V2 and inspect the real articulation at runtime."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

USD_PATH = (
    "/home/jpech/IsaacLab/source/isaaclab_assets/data/Robots/Daedamorph/"
    "nova_lowerbody_v2/NOVA_LOWERBODY_V2/NOVA_LOWERBODY_V2.usda"
)

sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.5])

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
cfg = sim_utils.DomeLightCfg(intensity=3000.0)
cfg.func("/World/Light", cfg)

robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    # placeholder actuator group covering every joint - real gains come later
    actuators={"all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
)
robot = Articulation(cfg=robot_cfg)

sim.reset()

print("=" * 80)
print("JOINT NAMES:", robot.data.joint_names)
print("NUM JOINTS:", robot.num_joints)
print("BODY NAMES:", robot.data.body_names)
print("NUM BODIES:", robot.num_bodies)
print("IS FIXED BASE:", robot.is_fixed_base)
print("DEFAULT JOINT POS:", robot.data.default_joint_pos)
print("JOINT LIMITS (lower/upper), shape:", robot.data.joint_limits.shape)
print(robot.data.joint_limits)
print("=" * 80)

for _ in range(30):
    sim.step()

simulation_app.close()