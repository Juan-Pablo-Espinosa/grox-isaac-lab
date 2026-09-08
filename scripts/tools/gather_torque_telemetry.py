"""Per-joint torque / percentage-of-effort-limit telemetry from the trained
Isaac-Standing-Nova-v0 policy (checkpoint model_1550.pt, converged run
2026-09-07_10-09-35, iteration 1550).

Gathers, per control step per env, each of the 12 revolute joints' applied_torque
and torque-as-percentage-of-its-own-effort_limit_sim, plus two derived aggregates:

  - percentage_torque_sq_sum: sum over the 12 joints of (torque_i / limit_i)^2.
    This is this script's own interpretation (not verified against an earlier
    spec this session doesn't have in context) -- a direct percentage-normalized
    analog of effort_reward's raw torque_sq_sum, flagged as such in the report.
  - symmetry_error: sum over the 6 L/R joint pairs of the per-pair symmetric
    error term, using the EMPIRICALLY VERIFIED sign convention per pair (see
    scripts/tools/verify_pair_sign_convention.py): same-sign pairs
    (Feet_Roll, Feet_Pitch) use (torque_L - torque_R)^2; opposite-sign pairs
    (Hip_Pitch, Hip_Roll, Upperleg_Yaw, Lowerleg_Pitch) use (torque_L +
    torque_R)^2. Both the corrected and the original naive (all same-sign,
    mirroring rough_env_cfg_grade2.py's mdp.symmetry_penalty precedent)
    versions are reported for direct comparison.

Produces a PNG plot (mean, across envs, of each joint's %-of-limit over the
rollout) and per-joint mean/median/p90 summary stats.
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--out_png", type=str, default="scripts/tools/output/torque_pct_limit.png")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

# (joint_name, effort_limit_sim [N*m]) -- verified directly from
# source/isaaclab_assets/isaaclab_assets/robots/nova.py's actuator groups.
REVOLUTE = [
    ("Hip_Pitch_Left_Joint", 120.0), ("Hip_Pitch_Right_Joint", 120.0),
    ("Hip_Roll_Left_Joint", 60.0), ("Hip_Roll_Right_Joint", 60.0),
    ("Upperleg_Yaw_Left_Joint", 60.0), ("Upperleg_Yaw_Right_Joint", 60.0),
    ("Lowerleg_Pitch_Left_Joint", 120.0), ("Lowerleg_Pitch_Right_Joint", 120.0),
    ("Feet_Roll_Left_Joint", 34.0), ("Feet_Roll_Right_Joint", 34.0),
    ("Feet_Pitch_Left_Joint", 34.0), ("Feet_Pitch_Right_Joint", 34.0),
]
JOINT_NAMES = [n for n, _ in REVOLUTE]
LIMITS = torch.tensor([lim for _, lim in REVOLUTE])
# (Left idx, Right idx, same_sign) -- same_sign verified empirically via forward
# kinematics (scripts/tools/verify_pair_sign_convention.py): perturb one joint at
# a time, check whether Right(+theta) or Right(-theta) produces the true
# Y-mirror-image body position of Left(+theta). Only Feet_Roll/Feet_Pitch are
# same-sign; Hip_Pitch, Hip_Roll, Upperleg_Yaw, and Lowerleg_Pitch (the knee) are
# all opposite-sign -- notably this is NOT just "axis Y-component flips": Hip_Roll
# has an IDENTICAL (unmirrored) axis vector on both sides yet is still
# opposite-sign, since mirroring is an orientation-reversing transform that can
# flip the effective rotation sign even for an unmirrored raw axis vector.
PAIRS = [
    (0, 1, False),  # Hip_Pitch: opposite-sign
    (2, 3, False),  # Hip_Roll: opposite-sign
    (4, 5, False),  # Upperleg_Yaw: opposite-sign
    (6, 7, False),  # Lowerleg_Pitch (knee): opposite-sign
    (8, 9, True),   # Feet_Roll: same-sign
    (10, 11, True),  # Feet_Pitch: same-sign
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
limits_dev = LIMITS.to(env.unwrapped.device)

obs = env.get_observations()

torque_hist: list[torch.Tensor] = []  # (steps, envs, 12)

with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        torque_hist.append(robot.data.applied_torque[:, joint_ids].clone())

torque_all = torch.stack(torque_hist)  # (S, E, 12)
pct_all = (torque_all.abs() / limits_dev) * 100.0  # (S, E, 12) percent of limit

pct_sq_sum = ((torque_all / limits_dev) ** 2).sum(dim=-1)  # (S, E) percentage_torque_sq_sum

# CORRECTED symmetry_error: same-sign pairs use (L-R)^2, opposite-sign pairs use
# (L+R)^2, per the empirically-verified convention above. Also compute the naive
# (all same-sign) version for direct, honest comparison against the last report.
sym_err_corrected = torch.zeros_like(pct_sq_sum)
sym_err_naive = torch.zeros_like(pct_sq_sum)
for l_idx, r_idx, same_sign in PAIRS:
    sym_err_naive += (torque_all[..., l_idx] - torque_all[..., r_idx]) ** 2
    if same_sign:
        sym_err_corrected += (torque_all[..., l_idx] - torque_all[..., r_idx]) ** 2
    else:
        sym_err_corrected += (torque_all[..., l_idx] + torque_all[..., r_idx]) ** 2

# --- per-joint summary stats ---
print("=" * 100)
print(f"Checkpoint: {args_cli.checkpoint}")
print(f"Rollout: {args_cli.steps} steps x {args_cli.num_envs} envs = {args_cli.steps * args_cli.num_envs} samples/joint")
print("=" * 100)
print(f"{'joint':32s} {'limit':>7s} {'mean_Nm':>9s} {'mean_%':>8s} {'median_%':>10s} {'p90_%':>8s}")
pct_np = pct_all.numpy()
torque_np = torque_all.numpy()
for j, (name, limit) in enumerate(REVOLUTE):
    t = torque_np[:, :, j].flatten()
    p = pct_np[:, :, j].flatten()
    print(
        f"{name:32s} {limit:7.1f} {np.abs(t).mean():9.3f} {p.mean():8.3f} "
        f"{np.percentile(p, 50):10.3f} {np.percentile(p, 90):8.3f}"
    )

print("\n" + "=" * 100)
print("Aggregate percentage_torque_sq_sum (sum of (torque_i/limit_i)^2 over 12 joints):")
psq = pct_sq_sum.numpy().flatten()
print(f"  mean={psq.mean():.5f}  median={np.percentile(psq,50):.5f}  p90={np.percentile(psq,90):.5f}  p99={np.percentile(psq,99):.5f}")

print("\nAggregate symmetry_error -- NAIVE (all pairs (torque_L - torque_R)^2, N*m^2):")
se_n = sym_err_naive.numpy().flatten()
print(f"  mean={se_n.mean():.5f}  median={np.percentile(se_n,50):.5f}  p90={np.percentile(se_n,90):.5f}  p99={np.percentile(se_n,99):.5f}")

print(
    "\nAggregate symmetry_error -- CORRECTED (Feet_Roll/Feet_Pitch (L-R)^2, "
    "Hip_Pitch/Hip_Roll/Upperleg_Yaw/Lowerleg_Pitch (L+R)^2, N*m^2):"
)
se_c = sym_err_corrected.numpy().flatten()
print(f"  mean={se_c.mean():.5f}  median={np.percentile(se_c,50):.5f}  p90={np.percentile(se_c,90):.5f}  p99={np.percentile(se_c,99):.5f}")
print("=" * 100)

# --- plot: mean-across-envs %-of-limit per joint, over rollout timesteps ---
os.makedirs(os.path.dirname(args_cli.out_png), exist_ok=True)
mean_pct_over_envs = pct_np.mean(axis=1)  # (S, 12)

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.tab20(np.linspace(0, 1, 12))
for j, (name, limit) in enumerate(REVOLUTE):
    ax.plot(mean_pct_over_envs[:, j], label=f"{name} ({limit:.0f} N*m)", color=colors[j], linewidth=1.2)
ax.set_xlabel("control step")
ax.set_ylabel("torque as % of joint's own effort_limit_sim")
ax.set_title(f"Per-joint torque %-of-limit, mean across {args_cli.num_envs} envs\ncheckpoint: model_1550.pt (iter 1550)")
ax.legend(loc="upper right", fontsize=7, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args_cli.out_png, dpi=150)
print(f"\nPlot saved to: {os.path.abspath(args_cli.out_png)}")

env.close()
simulation_app.close()
