# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record scripted showcase runs of a morphology-agnostic NOVA policy for offline rendering (Blender).

Launch (headless)::

    ./isaaclab.sh -p scripts/nova/showcase_record.py --headless --scene hero \\
        --checkpoint logs/rsl_rl/nova_morph_agnostic/<run>/model_N.pt --out ~/nova_showcase/hero.npz
    ./isaaclab.sh -p scripts/nova/showcase_record.py --headless --scene lineup --checkpoint ... --out .../lineup.npz

Alignment snapshot (Newton viewer, needs a display): ``--visualizer newton --snapshot_step K`` stops after control
step K and saves the viewer frame (<out>.png) and its camera (<out>.json) next to the npz.

Deterministic policy (mean action), fixed seed, no observation noise, no pushes, no base mass / COM randomization, no
reset offsets (default pose, yaw 0, zero velocity), random leg-length schedule OFF. Leg-length goals follow the
scripted timeline and are rate-limited by the task's PrismaticDriver (<= 0.035 m/s). Commands are smoothstep-
interpolated between keyframes (no step changes).

npz contents (T = steps + 1: frame 0 is the reset state, frame k is after control step k, dt = 0.02 s; R robots):
    body_pos (T,R,B,3) [m] world, body_quat (T,R,B,4) world, xyzw; body_names (B,)
    root_lin_vel_w / root_lin_vel_b / root_ang_vel_b (T,R,3); command (T,R,3) = (vx, vy, wz) base frame
    prism_target / prism_pos (T,R,4) [m] U_L, U_R, L_L, L_R; foot_contact (T,R,2) bool (left, right)
    joint_pos (T,R,J); joint_names (J,); env_origins (R,3); done (T,R) bool; dt; scene
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Record NOVA showcase runs to npz.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-MorphAgnostic-Play-v0")
parser.add_argument("--scene", choices=("hero", "lineup"), default="hero")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--snapshot_step", type=int, default=None, help="Save a Newton viewer frame after this step.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import json  # noqa: E402
import math  # noqa: E402

import nova_common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Command keyframes (t [s], vx, vy, wz); smoothstep between consecutive keyframes.
HERO_COMMAND = [
    (0.0, 0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, 0.0),
    (4.5, 0.6, 0.0, 0.0),  # 3-10 walk 0.6
    (10.0, 0.6, 0.0, 0.0),
    (11.0, 0.8, 0.0, 0.0),  # 10-16 walk 0.8 (growing)
    (16.0, 0.8, 0.0, 0.0),
    (17.0, 1.0, 0.0, 0.0),  # 16-22 walk 1.0 (shrinking fast)
    (21.3, 1.0, 0.0, 0.0),
    (22.3, 0.0, 0.0, 0.0),
    (23.1, 0.0, 0.0, 1.0),  # 22-28 turn in place +1, then -1 (antisymmetric about t = 25: net yaw 0)
    (24.6, 0.0, 0.0, 1.0),
    (25.4, 0.0, 0.0, -1.0),
    (26.9, 0.0, 0.0, -1.0),
    (27.7, 0.0, 0.0, 0.0),
    (28.0, 0.0, 0.0, 0.0),
    (29.0, 0.0, 0.4, 0.0),  # 28-34 sideways 0.4 (upper grows)
    (33.2, 0.0, 0.4, 0.0),
    (34.0, 0.0, 0.0, 0.0),
    (35.0, -0.5, 0.0, 0.0),  # 34-40 backward 0.5
    (39.2, -0.5, 0.0, 0.0),
    (40.0, 0.0, 0.0, 0.0),
    (48.0, 2.0, 0.0, 0.0),  # 40-50 accelerate to 2.0 (long legs)
    (50.0, 2.0, 0.0, 0.0),
    (54.0, 0.5, 0.0, 0.0),  # 50-56 decelerate to 0.5 (shrinking)
    (56.0, 0.5, 0.0, 0.0),
    (57.5, 0.0, 0.0, 0.0),  # 56-60 stop, stand
    (60.0, 0.0, 0.0, 0.0),
]
# Leg-length GOAL keyframes (t [s], upper mm, lower mm); linear between keyframes, the driver rate-limits the
# targets to 35 mm/s (a jump in the goal = fastest possible change).
HERO_LENGTH = [
    (0.0, 50, 50),
    (11.0, 50, 50),
    (14.0, 95, 95),  # ~15 mm/s growth
    (16.0, 95, 95),
    (16.02, 5, 5),  # fast shrink (35 mm/s)
    (28.0, 5, 5),
    (28.02, 95, 5),  # upper only
    (34.0, 95, 5),
    (34.02, 50, 50),
    (40.0, 50, 50),
    (40.02, 95, 95),
    (50.0, 95, 95),
    (50.02, 5, 5),
    (56.0, 5, 5),
    (56.02, 50, 50),
    (60.0, 50, 50),
]
LINEUP_MORPHS = [(5, 5), (50, 50), (95, 95), (5, 95), (95, 5)]
LINEUP_SPACING = 1.6
LINEUP_COMMAND = [
    (0.0, 0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0, 0.0),
    (3.0, 1.0, 0.0, 0.0),
    (12.0, 1.0, 0.0, 0.0),
    (12.8, 0.0, 0.0, 0.8),
    (15.0, 0.0, 0.0, 0.8),
]


def smoothstep_keys(keys, t: float) -> np.ndarray:
    """Value of the keyframe track at ``t`` with smoothstep easing between keyframes."""
    if t <= keys[0][0]:
        return np.array(keys[0][1:], dtype=float)
    for (t0, *a), (t1, *b) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            s = (t - t0) / max(t1 - t0, 1e-9)
            s = s * s * (3.0 - 2.0 * s)
            return (1.0 - s) * np.array(a, dtype=float) + s * np.array(b, dtype=float)
    return np.array(keys[-1][1:], dtype=float)


def linear_keys(keys, t: float) -> np.ndarray:
    ts = [k[0] for k in keys]
    return np.array([np.interp(t, ts, [k[i] for k in keys]) for i in range(1, len(keys[0]))])


def main():
    hero = args_cli.scene == "hero"
    n = 1 if hero else len(LINEUP_MORPHS)
    duration = (HERO_COMMAND if hero else LINEUP_COMMAND)[-1][0]
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    if nova_common.checkpoint_action_dim(resume_path) != 12:
        raise SystemExit(f"[showcase] {resume_path} is not a 12-D morphology-agnostic policy.")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    env_cfg.seed = args_cli.seed
    nova_common.pin_commands(env_cfg)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.prismatic_driver.schedule_enabled = False
    reset = env_cfg.events.reset_nova.params
    reset["revolute_offset_range"] = (0.0, 0.0)
    reset["pose_range"] = {}
    reset["velocity_range"] = {}
    morphs = [HERO_LENGTH[0][1:]] if hero else LINEUP_MORPHS
    reset["prismatic_per_env"] = [(up * 1e-3, lo * 1e-3) for up, lo in morphs]
    torch.manual_seed(args_cli.seed)
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)
    u = env.unwrapped
    dev = u.device
    robot, contact = u.scene["robot"], u.scene["contact_forces"]
    driver = u.prismatic_driver
    foot_ids = [contact.body_names.index(b) for b in ("Feet_Pitch_Left", "Feet_Pitch_Right")]

    # clean line along y (all facing +x), centred on y = 0
    origins = torch.zeros(n, 3, device=dev)
    origins[:, 1] = (torch.arange(n, device=dev, dtype=torch.float32) - (n - 1) / 2) * LINEUP_SPACING
    u.scene.terrain.env_origins[:] = origins

    cmd_term = u.command_manager.get_term("base_velocity")
    cmd_vec = torch.zeros(n, 3, device=dev)
    nova_common.route_resample(u, cmd_vec)

    viz = None
    for v in getattr(u.sim, "_visualizers", []):
        if type(v).__name__ == "NewtonVisualizer" and getattr(v, "_viewer", None) is not None:
            viz = v
    if args_cli.snapshot_step is not None and viz is None:
        raise SystemExit("[showcase] --snapshot_step needs --visualizer newton (with a display).")

    dt = u.step_dt
    steps = int(round(duration / dt)) if args_cli.snapshot_step is None else args_cli.snapshot_step
    rec = {k: [] for k in ("body_pos", "body_quat", "root_lin_vel_w", "root_lin_vel_b", "root_ang_vel_b")}
    rec.update({k: [] for k in ("command", "prism_target", "prism_pos", "foot_contact", "joint_pos", "done")})

    def sample(done):
        d = robot.data
        rec["body_pos"].append(d.body_link_pos_w.torch.cpu().numpy().copy())
        rec["body_quat"].append(d.body_link_quat_w.torch.cpu().numpy().copy())
        rec["root_lin_vel_w"].append(d.root_lin_vel_w.torch.cpu().numpy().copy())
        rec["root_lin_vel_b"].append(d.root_lin_vel_b.torch.cpu().numpy().copy())
        rec["root_ang_vel_b"].append(d.root_ang_vel_b.torch.cpu().numpy().copy())
        rec["command"].append(cmd_term.vel_command_b.cpu().numpy().copy())
        rec["prism_target"].append(driver.target.cpu().numpy().copy())
        rec["prism_pos"].append(d.joint_pos.torch[:, driver.joint_ids].cpu().numpy().copy())
        rec["foot_contact"].append((contact.data.current_contact_time.torch[:, foot_ids] > 0.0).cpu().numpy())
        rec["joint_pos"].append(d.joint_pos.torch.cpu().numpy().copy())
        rec["done"].append(done.cpu().numpy().astype(bool))

    def set_inputs(t: float):
        c = smoothstep_keys(HERO_COMMAND if hero else LINEUP_COMMAND, t)
        cmd_vec[:] = torch.tensor(c, device=dev, dtype=torch.float32)
        cmd_term.vel_command_b[:] = cmd_vec
        if hero:
            up, lo = linear_keys(HERO_LENGTH, t) * 1e-3
            driver.set_goal(float(up), float(lo))

    falls = []
    cam = None
    with torch.inference_mode():
        set_inputs(0.0)
        obs, _ = env.reset()
        q0 = robot.data.root_quat_w.torch[0].cpu().numpy()
        print(f"[showcase] root quat after reset (expect identity, xyzw = 0 0 0 1): {np.round(q0, 4)}")
        sample(torch.zeros(n, dtype=torch.bool))
        for k in range(1, steps + 1):
            # inputs for step k are the timeline at the step's start time; the command is also written after the
            # step's own command-manager update so it is what the policy sees next
            set_inputs((k - 1) * dt)
            if viz is not None and k == args_cli.snapshot_step:
                root = robot.data.root_pos_w.torch[0].cpu().numpy()
                eye = (float(root[0] + 1.6), float(root[1] - 1.9), 0.75)
                target = (float(root[0]), float(root[1]), 0.45)
                viz._apply_camera_pose((eye, target))
            obs, _, dones, _ = env.step(policy(obs))
            if dones.any():
                falls += [(k * dt, int(i)) for i in dones.nonzero().flatten().tolist()]
            cmd_term.vel_command_b[:] = cmd_vec
            sample(dones.bool())
            if k % 500 == 0:
                print(f"[showcase] t = {k * dt:5.1f} s  root = {np.round(rec['body_pos'][-1][:, 0], 2).tolist()}")
        if viz is not None and args_cli.snapshot_step is not None:
            v = viz._viewer
            img = v.get_frame().numpy()
            c = v.camera
            front = np.array(c.get_front(), dtype=float)
            cam = {
                "pos": [float(x) for x in c.pos],
                "front": front.tolist(),
                "fov_vertical_deg": float(c.fov),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "step": args_cli.snapshot_step,
            }

    out = os.path.expanduser(args_cli.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    arrays = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(
        out,
        **arrays,
        body_names=np.array(robot.body_names),
        joint_names=np.array(robot.joint_names),
        env_origins=origins.cpu().numpy(),
        morphs_mm=np.array(morphs, dtype=float),
        dt=dt,
        scene=args_cli.scene,
    )
    if cam is not None:
        from PIL import Image

        Image.fromarray(img).save(os.path.splitext(out)[0] + ".png")
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump(cam, f, indent=2)
    # summary
    vel = arrays["root_lin_vel_b"]
    cmd = arrays["command"]
    err = np.linalg.norm(vel[1:, :, :2] - cmd[1:, :, :2], axis=-1)
    tgt, pos = arrays["prism_target"], arrays["prism_pos"]
    rate = np.abs(np.diff(tgt, axis=0)).max() / dt
    print(f"[showcase] saved {out}: {arrays['body_pos'].shape[0]} frames, {n} robot(s)")
    print(f"[showcase] min root height {arrays['body_pos'][:, :, 0, 2].min():.3f} m")
    print(f"[showcase] mean |v_xy - cmd_xy| = {err.mean():.3f} m/s, max target rate {rate * 1e3:.1f} mm/s")
    print(f"[showcase] max |prismatic target - actual| = {np.abs(tgt - pos).max() * 1e3:.1f} mm")
    print(f"[showcase] FALLS: {falls if falls else 'none'}")
    if math.isfinite(rate) and falls:
        print("[showcase] WARNING: the robot fell -- adjust the timeline and re-record.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
