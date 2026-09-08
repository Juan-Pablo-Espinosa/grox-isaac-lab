"""Investigate whether the extreme accel_sq_sum tail is a physically real (if brief)
event or a numerical/contact-discontinuity artifact.

For each control step, records per-env: full per-joint joint_acc and joint_vel (12
revolute joints), root height, up_z, steps_since_reset. Afterward, finds the
highest-accel_sq_sum (env, step) events and prints the joint_vel trace for a small
window of steps before/after each, plus which specific joints dominate the spike,
plus foot height at that moment (contact_forces sensor is disabled for this task --
see standing_env_cfg.py -- so foot height is the best available contact proxy).
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--top_k", type=int, default=8)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

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
foot_ids, foot_names = robot.find_joints(["Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint"], preserve_order=True)
foot_body_ids = [robot.data.body_names.index(n) for n in ["Feet_Pitch_Left", "Feet_Pitch_Right"]]

obs = env.get_observations()
steps_since_reset = torch.zeros(env.unwrapped.num_envs, dtype=torch.long)

# History buffers: [step][env] -> tensor
joint_acc_hist: list[torch.Tensor] = []
joint_vel_hist: list[torch.Tensor] = []
foot_z_hist: list[torch.Tensor] = []
foot_vz_hist: list[torch.Tensor] = []
steps_since_reset_hist: list[torch.Tensor] = []

with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)

        joint_acc_hist.append(robot.data.joint_acc[:, joint_ids].clone())
        joint_vel_hist.append(robot.data.joint_vel[:, joint_ids].clone())
        foot_z_hist.append(robot.data.body_pos_w[:, foot_body_ids, 2].clone())
        foot_vz_hist.append(robot.data.body_link_lin_vel_w[:, foot_body_ids, 2].clone())
        steps_since_reset_hist.append(steps_since_reset.clone())

        steps_since_reset += 1
        steps_since_reset[dones.bool().cpu()] = 0

# Stack: (steps, num_envs, ...)
joint_acc_all = torch.stack(joint_acc_hist)  # (S, E, 12)
joint_vel_all = torch.stack(joint_vel_hist)
foot_z_all = torch.stack(foot_z_hist)  # (S, E, 2)
foot_vz_all = torch.stack(foot_vz_hist)
steps_since_reset_all = torch.stack(steps_since_reset_hist)

accel_sq_sum_all = (joint_acc_all**2).sum(dim=-1)  # (S, E)
flat = accel_sq_sum_all.flatten()
top_vals, top_idx = torch.topk(flat, args_cli.top_k)
S, E = accel_sq_sum_all.shape

print("=" * 90)
print(f"Top {args_cli.top_k} accel_sq_sum events across {S} steps x {E} envs:")
print("=" * 90)

for rank in range(args_cli.top_k):
    flat_i = top_idx[rank].item()
    s, e = flat_i // E, flat_i % E
    val = top_vals[rank].item()
    print(f"\n--- Rank {rank+1}: step={s} env={e} accel_sq_sum={val:.2f} steps_since_reset={steps_since_reset_all[s, e].item()} ---")
    per_joint_acc = joint_acc_all[s, e]
    dominant = torch.argsort(per_joint_acc.abs(), descending=True)[:3]
    print("  per-joint joint_acc (dominant 3):")
    for j in dominant:
        print(f"    {REVOLUTE[j]:32s} acc={per_joint_acc[j].item():14.2f} rad/s^2")
    print("  joint_vel trace (dominant joint, steps s-2..s+2):")
    dom_j = dominant[0].item()
    for ds in range(-2, 3):
        ss = s + ds
        if 0 <= ss < S:
            v = joint_vel_all[ss, e, dom_j].item()
            a = joint_acc_all[ss, e, dom_j].item() if ss < S else float("nan")
            marker = " <== SPIKE" if ds == 0 else ""
            print(f"    step={ss:4d}  vel={v:10.4f} rad/s  acc={a:14.2f} rad/s^2{marker}")
    print("  foot body z / vz (both feet) at step s-1, s, s+1:")
    for ds in range(-1, 2):
        ss = s + ds
        if 0 <= ss < S:
            fz = foot_z_all[ss, e].tolist()
            fvz = foot_vz_all[ss, e].tolist()
            print(f"    step={ss:4d}  foot_z={[round(x,4) for x in fz]}  foot_vz={[round(x,3) for x in fvz]}")

print("\n" + "=" * 90)
print("Overall: fraction of samples with accel_sq_sum > 1e6:", (flat > 1e6).float().mean().item())
print("Overall: fraction of samples with steps_since_reset < 5 among top-1000:", end=" ")
top1000_vals, top1000_idx = torch.topk(flat, min(1000, flat.numel()))
ssr_flat = steps_since_reset_all.flatten()
print((ssr_flat[top1000_idx] < 5).float().mean().item())

env.close()
simulation_app.close()
