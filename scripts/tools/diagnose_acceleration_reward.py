"""Diagnose why Episode_Reward/acceleration_reward logs exactly 0.0000.

Checks: (1) is asset.data.joint_acc actually populated/non-zero, (2) does the
reward's exp(-k * accel_sq_sum) underflow to exact 0.0 in float32 given real
accel magnitudes.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

REVOLUTE = [
    "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
]

cfg = parse_env_cfg("Isaac-Standing-Nova-v0", device=args_cli.device, num_envs=4)
env = gym.make("Isaac-Standing-Nova-v0", cfg=cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
joint_ids, _ = robot.find_joints(REVOLUTE)

k = 0.000866
for step in range(30):
    action = torch.zeros(env.unwrapped.action_space.shape)
    env.step(action)
    acc = robot.data.joint_acc[:, joint_ids]
    accel_sq_sum = torch.sum(acc**2, dim=1)
    exponent = -k * accel_sq_sum
    reward = torch.exp(exponent)
    print(
        f"step={step:3d} "
        f"joint_acc[0] sample={acc[0, :3].tolist()} "
        f"accel_sq_sum={accel_sq_sum.tolist()} "
        f"exponent={exponent.tolist()} "
        f"reward={reward.tolist()}"
    )

env.close()
simulation_app.close()
