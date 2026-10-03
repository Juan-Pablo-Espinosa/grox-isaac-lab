# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware speed check: p95 motor speed per joint group in fast walking vs the RobStride no-load speed.

Launch (headless)::

    ./isaaclab.sh -p scripts/nova/speed_check.py --checkpoint logs/rsl_rl/nova_morph_agnostic/<run>/model_N.pt \\
        --headless   # [--speeds 2.0,2.5] [--lengths 0.005,0.095] [--envs 64] [--settle 3] [--duration 10]
    # walking task run 5+: add --task Isaac-Walking-Nova-Play-v0 (leg lengths are LOCKED)

One simulation, ``--envs`` envs per (forward speed, leg length) cell; the leg length is held (upper = lower = the
cell's value). After ``--settle`` s, ``--duration`` s are measured on envs that have not fallen. Motor speed:
  * Hip_Pitch, Lowerleg_Pitch (RS04), Hip_Roll (RS03), Upperleg_Yaw (RS06): |joint velocity|
  * ankle RS02 motors: |omega_pitch +/- omega_roll| (same linkage approximation as the power model)
  * leadscrew RS00: 2 pi |target rate| / lead (8 mm)
Each group's p95 is compared with the module's no-load output speed at 48 V (datasheet), and flagged above 80%.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="p95 motor speed vs RobStride no-load speed.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-MorphAgnostic-Play-v0")
parser.add_argument("--speeds", type=str, default="2.0,2.5")
parser.add_argument("--lengths", type=str, default="0.005,0.095")
parser.add_argument("--envs", type=int, default=64, help="Envs per (speed, length) cell.")
parser.add_argument("--settle", type=float, default=3.0)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--flag", type=float, default=0.8, help="Flag groups whose p95 exceeds this fraction of no-load.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import math  # noqa: E402

import nova_common  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.power import (  # noqa: E402
    LEADSCREW_LEAD,
    ROBSTRIDE,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# group -> (motor model, joint name prefixes); ankle and leadscrew are handled specially
GROUPS = {
    "hip_pitch": ("RS04", ("Hip_Pitch_",)),
    "knee": ("RS04", ("Lowerleg_Pitch_",)),
    "hip_roll": ("RS03", ("Hip_Roll_",)),
    "hip_yaw": ("RS06", ("Upperleg_Yaw_",)),
    "ankle": ("RS02", ()),
    "leadscrew": ("RS00", ()),
}


def main():
    speeds = [float(x) for x in args_cli.speeds.split(",")]
    lengths = [float(x) for x in args_cli.lengths.split(",")]
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    cells = [(v, q) for v in speeds for q in lengths]
    n = len(cells) * args_cli.envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    agnostic = hasattr(env_cfg, "prismatic_driver")
    act_dim = nova_common.checkpoint_action_dim(resume_path)
    if act_dim != (12 if agnostic else 16) or (
        not agnostic and nova_common.checkpoint_obs_dim(resume_path) == nova_common.LEGACY_OBS_DIM
    ):
        raise SystemExit(f"[speed] {resume_path} ({act_dim}-D action) does not match {args_cli.task}.")
    nova_common.pin_commands(env_cfg)
    env_cfg.events.base_external_force_torque = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_nova.params["prismatic_per_env"] = [(q, q) for _, q in cells for _ in range(args_cli.envs)]
    if agnostic:
        env_cfg.prismatic_driver.schedule_enabled = False
    else:
        env_cfg.actions.prismatic.lock_per_env = [True] * n
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)
    u = env.unwrapped
    dev = u.device
    robot = u.scene["robot"]
    leg_src = u.prismatic_driver if agnostic else u.action_manager.get_term("prismatic")
    names = list(robot.joint_names)
    ids = {g: [i for i, nm in enumerate(names) if nm.startswith(p)] for g, (_, p) in GROUPS.items() if p}
    fp = [names.index(f"Feet_Pitch_{s}_Joint") for s in ("Left", "Right")]
    fr = [names.index(f"Feet_Roll_{s}_Joint") for s in ("Left", "Right")]
    cmd_vec = torch.zeros(n, 3, device=dev)
    cmd_vec[:, 0] = torch.tensor([v for v, _ in cells for _ in range(args_cli.envs)], device=dev)
    nova_common.route_resample(u, cmd_vec)
    u.command_manager.get_term("base_velocity").vel_command_b[:] = cmd_vec
    cell_of = torch.arange(n, device=dev) // args_cli.envs
    dt = u.step_dt
    settle, total = int(args_cli.settle / dt), int((args_cli.settle + args_cli.duration) / dt)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    samples = {g: [] for g in GROUPS}
    vx_sum, vx_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(total):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
            if step < settle:
                continue
            qd = robot.data.joint_vel.torch
            nan = torch.tensor(float("nan"), device=dev)
            mask = alive.unsqueeze(1)
            for g, jid in ids.items():
                samples[g].append(torch.where(mask, qd[:, jid].abs(), nan))
            ankle = torch.cat([(qd[:, fp] + qd[:, fr]).abs(), (qd[:, fp] - qd[:, fr]).abs()], dim=1)
            samples["ankle"].append(torch.where(mask, ankle, nan))
            screw = leg_src.target_rate.abs() * 2.0 * math.pi / LEADSCREW_LEAD
            samples["leadscrew"].append(torch.where(mask, screw, nan))
            vx_sum += alive.float() * robot.data.root_lin_vel_b.torch[:, 0]
            vx_n += alive.float()

    stacked = {g: torch.stack(s) for g, s in samples.items()}  # (T, n, motors)
    print(f"\n=== HARDWARE SPEED CHECK: {resume_path}")
    print(
        "  no-load output speed at 48 V (datasheet): "
        + ", ".join(
            f"{m} {ROBSTRIDE[m].no_load_speed_rpm:.0f} rpm = {ROBSTRIDE[m].no_load_speed:.2f} rad/s" for m in ROBSTRIDE
        )
    )
    print(
        f"  (leadscrew: RS00 no-load {ROBSTRIDE['RS00'].no_load_speed:.2f} rad/s = "
        f"{ROBSTRIDE['RS00'].no_load_speed_rpm / 60 * LEADSCREW_LEAD * 1000:.1f} mm/s of screw travel)"
    )
    header = f"  {'cell':22s} {'alive':>6s} {'vx':>6s} | " + " | ".join(f"{g:>16s}" for g in GROUPS)
    print(header + "\n  " + " " * 37 + " | ".join(f"{'p95 rad/s (%)':>16s}" for _ in GROUPS))
    flagged = []
    for c, (v, q) in enumerate(cells):
        sel = cell_of == c
        n_alive = int(alive[sel].sum())
        vx = (vx_sum[sel].sum() / vx_n[sel].sum().clamp(min=1)).item()
        parts = []
        for g, (model, _) in GROUPS.items():
            x = stacked[g][:, sel].flatten()
            p95 = (
                torch.nanquantile(x[~torch.isnan(x)][:16_000_000], 0.95).item()
                if (~torch.isnan(x)).any()
                else float("nan")
            )
            frac = p95 / ROBSTRIDE[model].no_load_speed
            parts.append(f"{p95:7.2f} ({frac * 100:4.0f}%)" + ("!" if frac > args_cli.flag else " "))
            if frac > args_cli.flag:
                flagged.append(
                    f"{g} at vx {v} legs {q * 1000:.0f} mm: p95 {p95:.2f} rad/s = {frac * 100:.0f}% of {model} no-load"
                )
        print(
            f"  vx {v:.1f} legs {q * 1000:3.0f} mm     {n_alive:3d}/{int(sel.sum()):<3d} {vx:6.2f} | "
            + " | ".join(parts)
        )
    print(f"  FLAGGED (> {args_cli.flag * 100:.0f}% of no-load): " + ("; ".join(flagged) if flagged else "none"))
    sys.stdout.flush()  # Kit's shutdown exits without flushing a redirected stdout
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
