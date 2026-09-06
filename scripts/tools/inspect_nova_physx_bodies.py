"""Print NOVA's actual PhysX body names and prim paths (ground truth, not the Newton shadow model)."""
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

print("=" * 80)
print("body_names (labels):")
for n in robot.data.body_names:
    print(f"  {n}")

print("root_physx_view body prim paths (ground truth for PhysX prim_path patterns):")
try:
    for p in robot.root_physx_view.prim_paths[: len(robot.data.body_names)]:
        print(f"  {p}")
except Exception as e:
    print(f"  (could not read root_physx_view.prim_paths: {e})")

print("=" * 80)
simulation_app.close()
