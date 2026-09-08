"""Print the actual default_joint_pos for all 16 joints (the pose test_nova_standing.py
holds via PD at t=0), to check whether it's genuinely non-zero for the 12 revolute
joints (needed for a meaningful same-sign vs opposite-sign sanity check)."""
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
for name, pos in zip(robot.data.joint_names, robot.data.default_joint_pos[0].tolist()):
    print(f"{name:32s} default_joint_pos={pos:.6f}")
print("=" * 80)

simulation_app.close()
