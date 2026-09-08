"""Smoke test: gym.make('Isaac-Standing-Nova-v0') actually builds and steps once."""
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
print("cfg parsed OK:", type(cfg).__name__)

env = gym.make("Isaac-Standing-Nova-v0", cfg=cfg)
print("SUCCESS: env created:", type(env.unwrapped))

obs, _ = env.reset()
print("obs keys:", list(obs.keys()))
print("policy obs shape:", obs["policy"].shape)

action = torch.zeros(env.unwrapped.action_space.shape)
obs, rew, terminated, truncated, info = env.step(action)
print("step OK. reward shape:", rew.shape, "reward sample:", rew[:4])
print("terminated:", terminated[:4], "truncated:", truncated[:4])

env.close()
simulation_app.close()
print("SMOKE TEST PASSED")
