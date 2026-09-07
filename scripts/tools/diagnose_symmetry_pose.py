"""Diagnose JP's observed "both legs bending the same way" pose using the
converged checkpoint from run 2026-09-07_12-28-15 (weight=6.0 symmetry_reward
run). Logs ACTUAL joint POSITIONS (not torque) for every L/R pair once the
policy's standing pose has settled, and checks each pair against the
FK-verified sign convention:
  opposite-sign (left == -right): Hip_Pitch, Hip_Roll, Upperleg_Yaw, Lowerleg_Pitch
  same-sign     (left == right):  Feet_Roll, Feet_Pitch
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle_steps", type=int, default=250)
parser.add_argument("--avg_window", type=int, default=50)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import numpy as np
import torch

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

PAIRS = [
    ("Hip_Pitch", "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint", "opposite"),
    ("Hip_Roll", "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint", "opposite"),
    ("Upperleg_Yaw", "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint", "opposite"),
    ("Lowerleg_Pitch", "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint", "opposite"),
    ("Feet_Roll", "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint", "same"),
    ("Feet_Pitch", "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint", "same"),
]
ALL_JOINT_NAMES = [n for _, l, r, _ in PAIRS for n in (l, r)]

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
joint_ids, resolved_names = robot.find_joints(ALL_JOINT_NAMES, preserve_order=True)
name_to_col = {n: i for i, n in enumerate(resolved_names)}
print("Resolved joint order:", resolved_names)

obs = env.get_observations()

pos_hist, torque_hist = [], []
with torch.inference_mode():
    for step in range(args_cli.settle_steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        pos_hist.append(robot.data.joint_pos[:, joint_ids].clone())
        torque_hist.append(robot.data.applied_torque[:, joint_ids].clone())

pos_all = torch.stack(pos_hist)  # (S, E, len(ALL_JOINT_NAMES))
torque_all = torch.stack(torque_hist)  # (S, E, len(ALL_JOINT_NAMES))
# average over the final avg_window steps to get the settled pose
settled = pos_all[-args_cli.avg_window :].mean(dim=0).cpu().numpy()  # (E, len(ALL_JOINT_NAMES))
settled_torque = torque_all[-args_cli.avg_window :].mean(dim=0).cpu().numpy()  # (E, len(ALL_JOINT_NAMES))
settled_last = pos_all[-1].cpu().numpy()  # (E,) last-step snapshot too

E = settled.shape[0]
TOL = 0.05  # rad

print("=" * 100)
print(f"SETTLED POSE JOINT ANGLES (mean over last {args_cli.avg_window} of {args_cli.settle_steps} steps)")
print("=" * 100)
for pair_name, lname, rname, convention in PAIRS:
    lcol, rcol = name_to_col[lname], name_to_col[rname]
    print(f"\n--- {pair_name} ({convention}-sign expected) ---")
    for e in range(E):
        l_val = settled[e, lcol]
        r_val = settled[e, rcol]
        if convention == "opposite":
            expected_r = -l_val
            residual = r_val - expected_r  # should be ~0 if opposite-sign holds
        else:
            expected_r = l_val
            residual = r_val - expected_r  # should be ~0 if same-sign holds
        ok = abs(residual) < TOL
        print(
            f"  env{e}: left={l_val:+.4f}  right={r_val:+.4f}  "
            f"expected_right={expected_r:+.4f}  residual={residual:+.4f}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )

print("\n" + "=" * 100)
print("SAME-SIDE-BENDING CHECK: for opposite-sign pairs, do left and right have the")
print("SAME sign (both positive or both negative), which would visually look like")
print("'both legs pointing the same way' regardless of whether |left|==|right|?")
print("=" * 100)
for pair_name, lname, rname, convention in PAIRS:
    if convention != "opposite":
        continue
    lcol, rcol = name_to_col[lname], name_to_col[rname]
    for e in range(E):
        l_val = settled[e, lcol]
        r_val = settled[e, rcol]
        same_sign = (l_val > 0) == (r_val > 0)
        print(f"  {pair_name} env{e}: left={l_val:+.4f} right={r_val:+.4f} same_sign={same_sign}")

print("\n" + "=" * 100)
print("TORQUE SYMMETRY CHECK (what symmetry_reward ACTUALLY measures, per rewards.py)")
print("=" * 100)
for pair_name, lname, rname, convention in PAIRS:
    lcol, rcol = name_to_col[lname], name_to_col[rname]
    print(f"\n--- {pair_name} ({convention}-sign expected, torque) ---")
    for e in range(E):
        l_t = settled_torque[e, lcol]
        r_t = settled_torque[e, rcol]
        if convention == "opposite":
            err = l_t + r_t
        else:
            err = l_t - r_t
        print(f"  env{e}: torque_left={l_t:+.3f}  torque_right={r_t:+.3f}  pair_error={err:+.3f}")

np.savez(
    "/tmp/claude-1000/-home-jpech/5ba19bdf-c729-48af-94da-4563a4e7d18b/scratchpad/symmetry_pose_raw.npz",
    pos_all=pos_all.cpu().numpy(),
    torque_all=torque_all.cpu().numpy(),
    resolved_names=np.array(resolved_names),
)
print("\nSaved raw joint position history to symmetry_pose_raw.npz")

env.close()
simulation_app.close()
