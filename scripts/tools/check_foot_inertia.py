"""Check foot/ankle body mass relative to other bodies, to test whether the accel
spikes are explainable by low inertia under normal actuator torque."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots import NOVA_LOWERBODY_CFG

sim_cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args_cli.device)
sim = SimulationContext(sim_cfg)
robot_cfg = NOVA_LOWERBODY_CFG.replace(prim_path="/World/envs/env_0/Robot")
robot = Articulation(cfg=robot_cfg)
sim.reset()

masses = robot.root_physx_view.get_masses()
masses = masses.numpy() if hasattr(masses, "numpy") else masses
for name, mass in zip(robot.data.body_names, masses[0]):
    print(f"{name:30s} mass={mass:.5f} kg")

simulation_app.close()
