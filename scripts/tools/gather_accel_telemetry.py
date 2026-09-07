"""Gather real accel_sq_sum telemetry from the trained Isaac-Standing-Nova-v0 policy,
for recalibrating acceleration_reward's k (Issue 1 follow-up).

Uses the actual trained checkpoint (not zero/random actions) so the data reflects
what the reward term will really see. Splits samples into:
  - "early" (settling): first N steps after a reset (robot still converging to
    standing pose after falling/randomized reset -- push events don't actually
    fire within an episode given episode_length_s=10.0 and push interval_range_s
    =(10.0, 15.0), so this is the observable "actively correcting" regime here)
  - "stable" (confirmed near-nominal standing): up_z > 0.97 and height within
    0.03m of target, and NOT in the early window
  - everything else ("transitional")
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--early_window_steps", type=int, default=50, help="~1s at decimation=4, dt=0.005")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import importlib.metadata as metadata

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

REVOLUTE = [
    "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
]

env_cfg = parse_env_cfg("Isaac-Standing-Nova-v0", device=args_cli.device, num_envs=args_cli.num_envs)
agent_cfg = load_cfg_from_registry("Isaac-Standing-Nova-v0", "rsl_rl_cfg_entry_point")
installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

import gymnasium as gym

env = gym.make("Isaac-Standing-Nova-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

robot = env.unwrapped.scene["robot"]
joint_ids, _ = robot.find_joints(REVOLUTE, preserve_order=True)
target_height = 0.82

obs = env.get_observations()
steps_since_reset = torch.zeros(env.unwrapped.num_envs, dtype=torch.long)

all_accel_sq: list[float] = []
early_accel_sq: list[float] = []
stable_accel_sq: list[float] = []
transitional_accel_sq: list[float] = []

with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)

        joint_acc = robot.data.joint_acc[:, joint_ids]
        accel_sq_sum = torch.sum(joint_acc**2, dim=1)

        quat = robot.data.root_quat_w
        x, y = quat[:, 0], quat[:, 1]
        up_z = 1.0 - 2.0 * (x**2 + y**2)
        height = robot.data.root_pos_w[:, 2]

        is_early = steps_since_reset < args_cli.early_window_steps
        is_stable = (~is_early) & (up_z > 0.97) & ((height - target_height).abs() < 0.03)
        is_transitional = (~is_early) & (~is_stable)

        vals = accel_sq_sum.cpu().numpy()
        all_accel_sq.extend(vals.tolist())
        early_accel_sq.extend(vals[is_early.cpu().numpy()].tolist())
        stable_accel_sq.extend(vals[is_stable.cpu().numpy()].tolist())
        transitional_accel_sq.extend(vals[is_transitional.cpu().numpy()].tolist())

        steps_since_reset += 1
        steps_since_reset[dones.bool().cpu()] = 0


def report(name: str, data: list[float]) -> None:
    if not data:
        print(f"{name}: NO SAMPLES")
        return
    arr = np.array(data)
    print(f"\n{name}  (n={len(arr)})")
    print(f"  mean   = {arr.mean():.4f}")
    print(f"  median = {np.percentile(arr, 50):.4f}")
    for p in [10, 50, 90, 99]:
        print(f"  p{p:<3d}  = {np.percentile(arr, p):.4f}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}")


print("=" * 80)
print(f"Total steps: {args_cli.steps}, num_envs: {args_cli.num_envs}")
report("ALL SAMPLES", all_accel_sq)
report("EARLY (settling, first %d steps post-reset)" % args_cli.early_window_steps, early_accel_sq)
report("STABLE (up_z>0.97, |height-target|<0.03, not early)", stable_accel_sq)
report("TRANSITIONAL (neither early nor confirmed-stable)", transitional_accel_sq)
print("=" * 80)

env.close()
simulation_app.close()
