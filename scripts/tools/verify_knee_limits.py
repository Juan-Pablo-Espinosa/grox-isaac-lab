"""Verify the live joint_pos_limits reflect the asymmetric knee override after reset,
and sample a rollout to confirm the knee never exceeds its new asymmetric range."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Standing-Nova-v0", device=args_cli.device, num_envs=4)
env = gym.make("Isaac-Standing-Nova-v0", cfg=cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
knee_ids, knee_names = robot.find_joints(["Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint"], preserve_order=True)
print("knee names/ids:", list(zip(knee_names, knee_ids)))
print("live joint_pos_limits:", robot.data.joint_pos_limits[0, knee_ids].tolist())
print("live soft_joint_pos_limits:", robot.data.soft_joint_pos_limits[0, knee_ids].tolist())

min_pos = torch.full((2,), float("inf"))
max_pos = torch.full((2,), float("-inf"))
for step in range(300):
    action = torch.randn(env.unwrapped.action_space.shape) * 0.5
    env.step(action)
    pos = robot.data.joint_pos[0, knee_ids]
    min_pos = torch.minimum(min_pos, pos)
    max_pos = torch.maximum(max_pos, pos)

print(f"observed knee pos range over 300 random-action steps: min={min_pos.tolist()} max={max_pos.tolist()}")

env.close()
simulation_app.close()
