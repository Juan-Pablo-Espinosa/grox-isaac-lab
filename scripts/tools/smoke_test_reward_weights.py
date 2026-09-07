"""Smoke test: confirm symmetry_reward's weight change (2.0 -> 6.0) is applied and
all 7 reward terms sum to the new 46 ceiling. Does not train -- just inspects the
built RewardManager's term configs and does one rollout step as a sanity check."""
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

rm = env.unwrapped.reward_manager
weights = {}
for name in rm.active_terms:
    term_cfg = rm.get_term_cfg(name)
    weights[name] = term_cfg.weight

print("=" * 80)
print("REWARD TERM WEIGHTS")
for name, w in weights.items():
    print(f"  {name:24s} weight={w}")
total = sum(weights.values())
print(f"  {'TOTAL':24s} weight={total}")
print("=" * 80)

assert len(weights) == 7, f"Expected 7 reward terms, found {len(weights)}: {list(weights.keys())}"
assert "symmetry_reward" in weights, "symmetry_reward term missing"
assert weights["symmetry_reward"] == 6.0, f"symmetry_reward weight is {weights['symmetry_reward']}, expected 6.0"
assert total == 46.0, f"Total weight ceiling is {total}, expected 46.0"

obs, _ = env.reset()
action = torch.zeros(env.unwrapped.action_space.shape)
obs, rew, terminated, truncated, info = env.step(action)
print("rollout step OK. reward sample:", rew[:4])

env.close()
simulation_app.close()
print("SMOKE TEST PASSED: symmetry_reward weight=6.0 applied, 7 terms sum to 46.")
