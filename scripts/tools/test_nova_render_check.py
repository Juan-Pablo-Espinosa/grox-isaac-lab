"""Pure rendering sanity check: gravity off, robot suspended, one joint swings gently.
No falling, no ground contact, no possibility of solver explosion. If you don't see
motion here, it is definitively a rendering issue, not a physics issue."""
import argparse
import math

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

sim_cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 2.0, 1.2], [0.0, 0.0, 1.0])

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
cfg = sim_utils.DomeLightCfg(intensity=3000.0)
cfg.func("/World/Light", cfg)

robot_cfg = NOVA_LOWERBODY_CFG.replace(prim_path="/World/envs/env_0/Robot")
robot_cfg.init_state.pos = (0.0, 0.0, 1.5)  # well clear of the ground, no gravity so it stays put
robot = Articulation(cfg=robot_cfg)

sim.reset()

default_joint_pos = robot.data.default_joint_pos.clone()
joint_names = robot.data.joint_names
sweep_joint_name = "Hip_Pitch_Left_Joint"
sweep_idx = joint_names.index(sweep_joint_name)

print("=" * 80)
print(f"Gravity OFF. Robot suspended at z=1.5m. Sweeping '{sweep_joint_name}' slowly.")
print("Watching 8s.")
print("=" * 80)

amplitude = 1.2
freq_hz = 0.25  # slow, ~4s per full cycle, easy to see with your eyes

for step in range(1920):  # 8s at 240Hz
    t = step * sim_cfg.dt
    target = default_joint_pos.clone()
    target[0, sweep_idx] = amplitude * math.sin(2 * math.pi * freq_hz * t)
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_cfg.dt)
    if step % 60 == 0:
        pos = robot.data.root_pos_w[0]
        joint_pos = robot.data.joint_pos[0, sweep_idx].item()
        print(f"t={t:5.2f}s  root_height={pos[2].item():6.3f}m  {sweep_joint_name}={joint_pos:6.3f}rad")

print("=" * 80)
print("Render check complete.")
print("=" * 80)
simulation_app.close()
