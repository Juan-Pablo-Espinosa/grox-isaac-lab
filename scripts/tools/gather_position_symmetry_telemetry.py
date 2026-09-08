"""Position-based symmetry_error telemetry from the CURRENT DEGENERATE checkpoint
(model_2999.pt, run 2026-09-07_12-28-15) -- the run exhibiting the "diva pose"
(Upperleg_Yaw_Left pinned at its +0.79 rad hard limit while Right sits elsewhere,
confirmed via direct FK mirror-image comparison this session).

This checkpoint's bad, asymmetric stance is USEFUL data here: it tells us what
magnitude of position-based symmetry_error corresponds to the exact bad behavior
we're trying to eliminate, the same way the original 3.3-3.8 m/s drift told us
where to anchor velocity_xy_reward's k.

Gathers, per control step per env, joint_pos for all 12 revolute joints and the
resulting symmetry_error using the EMPIRICALLY VERIFIED per-pair sign convention
(same as the old torque-based version, just applied to position now):
  opposite-sign (L+R)^2: Hip_Pitch, Hip_Roll, Upperleg_Yaw, Lowerleg_Pitch
  same-sign     (L-R)^2: Feet_Roll, Feet_Pitch

Reports mean/median/p10/p90/p99 for the aggregate symmetry_error, plus per-pair
breakdowns so it's clear which pair(s) dominate the error (expected: Upperleg_Yaw).
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
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

JOINT_NAMES = [
    "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
]
# (pair_name, left_idx, right_idx, same_sign)
PAIRS = [
    ("Hip_Pitch", 0, 1, False),
    ("Hip_Roll", 2, 3, False),
    ("Upperleg_Yaw", 4, 5, False),
    ("Lowerleg_Pitch", 6, 7, False),
    ("Feet_Roll", 8, 9, True),
    ("Feet_Pitch", 10, 11, True),
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
joint_ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)

obs = env.get_observations()

pos_hist: list[torch.Tensor] = []  # (steps, envs, 12)

with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        pos_hist.append(robot.data.joint_pos[:, joint_ids].clone())

pos_all = torch.stack(pos_hist)  # (S, E, 12)

sym_err = torch.zeros(pos_all.shape[:2], device=pos_all.device)  # (S, E)
per_pair_err = {}
for pair_name, l_idx, r_idx, same_sign in PAIRS:
    if same_sign:
        term = (pos_all[..., l_idx] - pos_all[..., r_idx]) ** 2
    else:
        term = (pos_all[..., l_idx] + pos_all[..., r_idx]) ** 2
    per_pair_err[pair_name] = term
    sym_err += term

print("=" * 100)
print(f"Checkpoint: {args_cli.checkpoint}")
print(f"Rollout: {args_cli.steps} steps x {args_cli.num_envs} envs = {args_cli.steps * args_cli.num_envs} samples")
print("=" * 100)

print(f"\n{'joint pair':16s} {'convention':>10s} {'mean':>10s} {'median':>10s} {'p90':>10s} {'p99':>10s}")
for pair_name, l_idx, r_idx, same_sign in PAIRS:
    e = per_pair_err[pair_name].cpu().numpy().flatten()
    conv = "same" if same_sign else "opposite"
    print(
        f"{pair_name:16s} {conv:>10s} {e.mean():10.5f} {np.percentile(e,50):10.5f} "
        f"{np.percentile(e,90):10.5f} {np.percentile(e,99):10.5f}"
    )

print("\n" + "=" * 100)
print("AGGREGATE POSITION-BASED symmetry_error (sum of 6 per-pair terms above, rad^2):")
se = sym_err.cpu().numpy().flatten()
print(
    f"  mean={se.mean():.5f}  median={np.percentile(se,50):.5f}  p10={np.percentile(se,10):.5f}  "
    f"p90={np.percentile(se,90):.5f}  p99={np.percentile(se,99):.5f}  max={se.max():.5f}"
)
print("=" * 100)

env.close()
simulation_app.close()
