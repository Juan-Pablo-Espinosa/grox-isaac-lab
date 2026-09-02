"""Passive free-fall test - no actuation, just confirming gravity/contact work cleanly."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots import NOVA_LOWERBODY_CFG

sim_cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args_cli.device)  # finer timestep for stability
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 2.0, 1.2], [0.0, 0.0, 0.5])

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
cfg = sim_utils.DomeLightCfg(intensity=3000.0)
cfg.func("/World/Light", cfg)

robot_cfg = NOVA_LOWERBODY_CFG.replace(prim_path="/World/Robot")
robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
robot = Articulation(cfg=robot_cfg)

sim.reset()

print("=" * 80)
print("Passive free-fall - NO actuation commands sent at all. Watching 5s.")
print("=" * 80)

for step in range(1200):  # 5s at 240Hz
    # no set_joint_position_target call - joints are fully passive
    sim.step()
    robot.update(sim_cfg.dt)

    if step % 60 == 0:
        pos = robot.data.root_pos_w[0]
        lin_vel = robot.data.root_lin_vel_w[0]
        print(
            f"t={step * sim_cfg.dt:5.2f}s  "
            f"height={pos[2].item():6.3f}m  "
            f"lin_vel_mag={torch.norm(lin_vel).item():6.3f}"
        )

print("=" * 80)
print("Free-fall test complete.")
print("=" * 80)

simulation_app.close()
