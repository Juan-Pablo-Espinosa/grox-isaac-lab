"""Dynamic consistency diagnostic for JP's observed "unrealistic sliding" during
early iterations of the current fresh (no-checkpoint) run.

Uses model_0.pt from run 2026-09-07_11-23-15 (the only run in the logs starting
at iteration 0, confirming it's genuinely fresh/unresumed) -- the closest thing
to "the earliest available checkpoint" showing early-training behavior.

Logs, per control step: per-joint torque (checked against each joint's own
effort_limit_sim -- a hard, verifiable invariant), per-joint joint_acc, root
(Hip_Base) linear velocity and linear acceleration (via body_com_lin_acc_w, a
PhysX-native pull-on-demand value, not a manual finite-difference -- and
Hip_Base is the HEAVIEST body at 5.42kg, so it doesn't have the low-inertia
noise issue found earlier for Feet_Roll), and both feet's height (contact
proxy, since contact_forces is disabled for this task).

Also empirically reads the live ground and foot friction material values, to
verify against cfg expectations and confirm the rest_offset/contact_offset fix
(a completely different USD schema -- physxCollision:* -- from friction
material properties) didn't touch friction at all.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
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
foot_body_ids = [robot.data.body_names.index(n) for n in ["Feet_Pitch_Left", "Feet_Pitch_Right"]]
root_id = robot.data.body_names.index("Hip_Base")

# --- live friction material check ---
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics

stage = sim_utils.SimulationContext.instance().stage


def read_material_friction(prim_path_hint: str, root_search: str):
    """Find a PhysicsMaterialAPI-carrying prim under root_search matching prim_path_hint
    in its name, and read its static/dynamic friction attributes."""
    root = stage.GetPrimAtPath(root_search)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim_path_hint in prim.GetName() and UsdPhysics.MaterialAPI(prim):
            sf = prim.GetAttribute("physics:staticFriction").Get()
            df = prim.GetAttribute("physics:dynamicFriction").Get()
            return str(prim.GetPath()), sf, df
    return None


ground_mat = read_material_friction("", "/World/ground")

# Robot-side material is applied by randomize_rigid_body_material via the PhysX
# TENSOR API directly (asset.root_view.set_material_properties), not by binding a
# USD material prim -- so a USD-stage prim search will never find it (this is why
# the earlier prim-tree search legitimately returned None). Query the same tensor
# API back to read the actually-applied values.
import warp as wp

robot_mat_props = wp.to_torch(robot.root_view.get_material_properties())  # (num_envs, num_shapes, 3): static, dynamic, restitution
print("=" * 100)
print("LIVE FRICTION MATERIAL CHECK")
print(f"  ground material (USD prim): {ground_mat}")
print(f"  robot material (PhysX tensor API, env0 first 3 shapes [static, dynamic, restitution]):")
print(f"    {robot_mat_props[0, :3].cpu().numpy()}")
print(f"  robot material -- static friction range across all envs/shapes: "
      f"[{robot_mat_props[..., 0].min().item():.4f}, {robot_mat_props[..., 0].max().item():.4f}]")
print(f"  robot material -- dynamic friction range across all envs/shapes: "
      f"[{robot_mat_props[..., 1].min().item():.4f}, {robot_mat_props[..., 1].max().item():.4f}]")
print("=" * 100)
foot_mat_env0 = ("robot_root_view", robot_mat_props[..., 0].mean().item(), robot_mat_props[..., 1].mean().item())

obs = env.get_observations()

torque_hist, acc_hist, root_vel_hist, root_acc_hist, foot_z_hist, foot_vel_hist = [], [], [], [], [], []

with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        torque_hist.append(robot.data.applied_torque[:, joint_ids].clone())
        acc_hist.append(robot.data.joint_acc[:, joint_ids].clone())
        root_vel_hist.append(robot.data.root_lin_vel_w.clone())
        root_acc_hist.append(robot.data.body_com_lin_acc_w[:, root_id, :].clone())
        foot_z_hist.append(robot.data.body_pos_w[:, foot_body_ids, 2].clone())
        foot_vel_hist.append(robot.data.body_lin_vel_w[:, foot_body_ids, :].clone())

torque_all = torch.stack(torque_hist)  # (S, E, 12)
root_vel_all = torch.stack(root_vel_hist)  # (S, E, 3)
root_acc_all = torch.stack(root_acc_hist)  # (S, E, 3)
foot_z_all = torch.stack(foot_z_hist)  # (S, E, 2)
foot_vel_all = torch.stack(foot_vel_hist)  # (S, E, 2, 3)

# --- invariant 1: torque never exceeds its own effort_limit_sim ---
over_limit = torque_all.abs() > limits_dev * 1.001  # small tolerance for float rounding
n_violations = over_limit.sum().item()
print(f"\nTORQUE-LIMIT INVARIANT: {n_violations} / {over_limit.numel()} samples exceed effort_limit_sim")
if n_violations > 0:
    idx = over_limit.nonzero()[0]
    s, e, j = idx.tolist()
    print(f"  first violation: step={s} env={e} joint={JOINT_NAMES[j]} torque={torque_all[s,e,j].item():.3f} limit={LIMITS[j].item()}")

# --- invariant 2: horizontal root acceleration vs friction-availability bound ---
# mu*g bound uses the LOWER of the two combined static frictions as a conservative
# estimate (actual PhysX combine is "multiply" per terrain material, so the true
# bound is smaller still -- this is a deliberately generous upper bound).
g = 9.81
mu_ground = ground_mat[1] if ground_mat else 1.0
mu_robot = foot_mat_env0[1] if foot_mat_env0 else 0.8
mu_conservative = max(mu_ground, mu_robot)  # generous upper bound, not the true combined value
a_max_friction = mu_conservative * g

root_horiz_acc = torch.norm(root_acc_all[..., :2], dim=-1)  # (S, E)
foot_min_z = foot_z_all.min(dim=-1).values  # (S, E) -- lower of the two feet
# "in contact" proxy: local sole offset is -0.031m (verified earlier this session);
# treat foot body z < 0.035 as plausibly near/at ground contact.
in_contact = foot_min_z < 0.035

contact_horiz_acc = root_horiz_acc[in_contact]
airborne_horiz_acc = root_horiz_acc[~in_contact]

print(f"\nFRICTION-BOUND CHECK (conservative mu={mu_conservative}, a_max={a_max_friction:.2f} m/s^2)")
print(f"  samples classified 'near-contact' (foot_z<0.035): {in_contact.sum().item()} / {in_contact.numel()}")
if contact_horiz_acc.numel() > 0:
    ca = contact_horiz_acc.cpu().numpy()
    print(f"  near-contact root horiz accel: mean={ca.mean():.3f} p90={np.percentile(ca,90):.3f} p99={np.percentile(ca,99):.3f} max={ca.max():.3f}")
    print(f"  fraction EXCEEDING friction bound while near-contact: {(ca > a_max_friction).mean():.4f}")
if airborne_horiz_acc.numel() > 0:
    aa = airborne_horiz_acc.cpu().numpy()
    print(f"  airborne/falling root horiz accel (bound does NOT apply): mean={aa.mean():.3f} p90={np.percentile(aa,90):.3f} max={aa.max():.3f}")

# --- invariant 3: direct foot-slip signal ---
# The mu*g bound on Hip_Base COM acceleration (invariant 2) is confounded by
# legitimate toppling dynamics -- internal joint torques can produce whatever
# COM acceleration is kinematically consistent with a rotating multi-body chain
# even with ZERO foot slip, since the friction constraint only bounds the
# contact patch, not the whole-body COM. The direct, unconfounded signal for
# "sliding" is the foot body's OWN horizontal velocity while it is resting on
# the ground (low height AND low vertical velocity, i.e. not just passing
# through that height mid-fall).
foot_vz = foot_vel_all[..., 2]  # (S, E, 2)
foot_horiz_vel = torch.norm(foot_vel_all[..., :2], dim=-1)  # (S, E, 2)
resting = (foot_z_all < 0.035) & (foot_vz.abs() < 0.05)  # (S, E, 2)

print(f"\nDIRECT FOOT-SLIP CHECK (foot resting: z<0.035 AND |vz|<0.05 m/s)")
print(f"  samples classified 'resting': {resting.sum().item()} / {resting.numel()}")
if resting.sum().item() > 0:
    resting_slip = foot_horiz_vel[resting].cpu().numpy()
    print(f"  resting-foot horizontal speed: mean={resting_slip.mean():.4f} p50={np.percentile(resting_slip,50):.4f}"
          f" p90={np.percentile(resting_slip,90):.4f} p99={np.percentile(resting_slip,99):.4f} max={resting_slip.max():.4f} m/s")
    print(f"  fraction of resting samples with horizontal speed > 0.1 m/s (visible slip): {(resting_slip > 0.1).mean():.4f}")
    print(f"  fraction of resting samples with horizontal speed > 0.3 m/s (fast slip): {(resting_slip > 0.3).mean():.4f}")

# --- invariant 4: PERSISTENCE-filtered foot-slip (stronger evidence than invariant 3) ---
# A single-frame coincidence of low foot z + low foot vz during a fast flailing
# swing (plausible for an untrained/near-random early policy) is weak evidence of
# true ground contact. Require the "resting" condition to hold for >=3 CONSECUTIVE
# control steps (the foot staying near ground height with near-zero vertical
# velocity for 3+ steps in a row is a much stronger signature of an actual
# ground-contact drag than an isolated flail-through frame).
resting_np = resting.cpu().numpy()  # (S, E, 2)
S, E, F = resting_np.shape
persistent_mask = np.zeros_like(resting_np)
for e in range(E):
    for f in range(F):
        col = resting_np[:, e, f]
        run_len = 0
        for s in range(S):
            if col[s]:
                run_len += 1
            else:
                run_len = 0
            if run_len >= 3:
                persistent_mask[s, e, f] = True

persistent_mask_t = torch.from_numpy(persistent_mask).to(foot_horiz_vel.device)
print(f"\nPERSISTENCE-FILTERED FOOT-SLIP CHECK (resting for >=3 consecutive steps)")
print(f"  samples classified 'persistently resting': {persistent_mask.sum()} / {resting_np.size}")
if persistent_mask.sum() > 0:
    persist_slip = foot_horiz_vel[persistent_mask_t].cpu().numpy()
    print(f"  persistently-resting foot horizontal speed: mean={persist_slip.mean():.4f} p50={np.percentile(persist_slip,50):.4f}"
          f" p90={np.percentile(persist_slip,90):.4f} max={persist_slip.max():.4f} m/s")
    print(f"  fraction with horizontal speed > 0.1 m/s: {(persist_slip > 0.1).mean():.4f}")
else:
    print("  NO samples had the foot resting for 3+ consecutive steps -- all near-ground/low-vz")
    print("  events in this early-checkpoint rollout were single-frame flail-throughs, not sustained contact.")

print(f"\nRoot vertical velocity range: min={root_vel_all[...,2].min().item():.3f} max={root_vel_all[...,2].max().item():.3f} m/s")
print(f"Root horizontal velocity range: min={torch.norm(root_vel_all[...,:2],dim=-1).min().item():.3f} max={torch.norm(root_vel_all[...,:2],dim=-1).max().item():.3f} m/s")
print("=" * 100)

np.savez(
    "/tmp/claude-1000/-home-jpech/5ba19bdf-c729-48af-94da-4563a4e7d18b/scratchpad/diag_raw_tensors.npz",
    torque_all=torque_all.cpu().numpy(),
    root_vel_all=root_vel_all.cpu().numpy(),
    root_acc_all=root_acc_all.cpu().numpy(),
    foot_z_all=foot_z_all.cpu().numpy(),
    foot_vel_all=foot_vel_all.cpu().numpy(),
)
print("Saved raw tensors to diag_raw_tensors.npz for further offline analysis.")

env.close()
simulation_app.close()
