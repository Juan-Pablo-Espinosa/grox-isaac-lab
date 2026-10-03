# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg-length sweep of a trained NOVA walking policy: cost of transport per (upper, lower) length and speed.

Launch (headless)::

    # morphology-agnostic task (default): every cell's leg length is held by the external prismatic driver
    ./isaaclab.sh -p scripts/nova/morph_sweep.py --checkpoint logs/rsl_rl/nova_morph_agnostic/<run>/model_N.pt \
        --headless   # speeds {0, 0.5, 1, 1.5, 2, 2.5}
    # walking task run 5+ (LOCKED grid + FREE choice, see below)
    ./isaaclab.sh -p scripts/nova/morph_sweep.py --task Isaac-Walking-Nova-Play-v0 --checkpoint <run 5 model> --headless
    #   [--envs_per_cell 32] [--free_envs_per_speed 64] [--settle 3] [--duration 15] [--speeds ...] [--out DIR]

Morphology-agnostic task: grid upper q x lower q x forward speed, ``--envs_per_cell`` envs per cell, the random
schedule off and each env's driver goal = its cell's lengths (falls respawn at the same lengths). No FREE block.

Walking task, all cells run in ONE simulation, side by side:
  * LOCKED block: grid upper q x lower q in {0.005, 0.035, 0.065, 0.095} m x forward speed {0, 0.5, 1.0, 1.5, 2.0}
    m/s, ``--envs_per_cell`` envs per cell; leg lengths held at the cell values (falls respawn at the same values).
  * FREE block: the same speeds, ``--free_envs_per_speed`` envs each, the policy chooses its leg lengths (respawn
    lengths are random per env, fixed across respawns).
Each env runs ``--settle`` s, then ``--duration`` s of measurement. An env that falls is counted and excluded from
the statistics from then on. Deterministic policy, no observation noise, no pushes, commands fixed (vy = wz = 0).

Per cell (samples pooled over the cell's surviving env-steps): actual speed (planar |v_xy| and forward v_x in the
base frame), tracking error |v_xy - cmd_xy|, falls, P_elec (mdp/power.py), CoT = mean P_elec / (m g max(mean |v|,
0.1)) with m the mean robot mass of the cell, single-stance and flight fractions. FREE rows also give the chosen q_U,
q_L (mean / std over env-steps of the last 5 s). Every row also has step_freq (touchdowns of both feet per second
of surviving env time) [Hz] and stride_length (distance per two touchdowns) [m].

Outputs in ``--out`` (default: <checkpoint dir>/morph_sweep_<checkpoint name>/): sweep.csv, cot_heatmaps.png (one
CoT heatmap per speed, the FREE choice marked), optimal_length.png (lowest-CoT locked total length vs speed, with the
FREE total).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Leg-length x speed sweep of a NOVA walking policy.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-MorphAgnostic-Play-v0")
parser.add_argument("--envs_per_cell", type=int, default=32)
parser.add_argument("--free_envs_per_speed", type=int, default=64)
parser.add_argument("--settle", type=float, default=3.0, help="Seconds before measuring [s].")
parser.add_argument("--duration", type=float, default=15.0, help="Measurement time [s].")
parser.add_argument("--free_window", type=float, default=5.0, help="FREE leg-length window at the end [s].")
parser.add_argument("--lengths", type=str, default="0.005,0.035,0.065,0.095")
parser.add_argument("--speeds", type=str, default=None, help="Default 0..2.5 (agnostic) / 0..2.0 (walking).")
parser.add_argument("--out", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import csv  # noqa: E402
import math  # noqa: E402

import nova_common  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.power import (  # noqa: E402
    PRISMATIC_DRIVER,
    power_model,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

G = 9.81


def build_layout(lengths: list[float], speeds: list[float], e_cell: int, e_free: int, mode: str = "locked"):
    """Per-env cell assignment: lists of (mode, speed, q_up, q_low) per cell and the per-env cell index."""
    cells, env_cell = [], []
    for v in speeds:
        for qu in lengths:
            for ql in lengths:
                cells.append((mode, v, qu, ql))
                env_cell += [len(cells) - 1] * e_cell
    for v in speeds if e_free else []:
        cells.append(("free", v, float("nan"), float("nan")))
        env_cell += [len(cells) - 1] * e_free
    return cells, env_cell


def main():
    lengths = [float(x) for x in args_cli.lengths.split(",")]
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    agnostic = hasattr(env_cfg, "prismatic_driver")
    act_dim = nova_common.checkpoint_action_dim(resume_path)
    if agnostic and act_dim != 12:
        raise SystemExit(
            f"[sweep] {args_cli.task} needs a 12-D (morphology-agnostic) policy; {resume_path} is {act_dim}-D."
        )
    if not agnostic and (act_dim != 16 or nova_common.checkpoint_obs_dim(resume_path) == nova_common.LEGACY_OBS_DIM):
        raise SystemExit(
            f"[sweep] {resume_path} is not a walking run-5+ policy (67-D obs, 16-D action): not supported."
        )
    speeds = [
        float(x) for x in (args_cli.speeds or ("0,0.5,1.0,1.5,2.0,2.5" if agnostic else "0,0.5,1.0,1.5,2.0")).split(",")
    ]
    e_free = 0 if agnostic else args_cli.free_envs_per_speed
    cells, env_cell = build_layout(lengths, speeds, args_cli.envs_per_cell, e_free, "driven" if agnostic else "locked")
    n = len(env_cell)
    gen = torch.Generator().manual_seed(0)
    free_len = (0.005 + 0.09 * torch.rand(n, 2, generator=gen)).tolist()  # FREE respawn lengths (fixed per env)
    per_env_len = [
        (cells[c][2], cells[c][3]) if cells[c][0] != "free" else tuple(free_len[i]) for i, c in enumerate(env_cell)
    ]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    nova_common.pin_commands(env_cfg)
    env_cfg.events.base_external_force_torque = None
    env_cfg.observations.policy.enable_corruption = False
    if agnostic:
        env_cfg.prismatic_driver.schedule_enabled = False  # driver goal = respawn length = the cell's length
    else:
        env_cfg.actions.prismatic.lock_per_env = [cells[c][0] == "locked" for c in env_cell]
    env_cfg.events.reset_nova.params["prismatic_per_env"] = per_env_len
    print(
        f"[sweep] {'agnostic' if agnostic else 'walking'}: {len(cells)} cells, {n} envs ({args_cli.envs_per_cell}/cell"
        f"{'' if agnostic else f', {e_free}/free speed'})"
    )
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)
    u = env.unwrapped
    dev = u.device
    robot, contact = u.scene["robot"], u.scene["contact_forces"]
    prismatic = u.prismatic_driver if agnostic else u.action_manager.get_term("prismatic")
    rate_source = PRISMATIC_DRIVER if agnostic else "prismatic"
    foot_ids = [contact.body_names.index(b) for b in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
    cell_of = torch.tensor(env_cell, device=dev)
    cmd_vec = torch.zeros(n, 3, device=dev)
    cmd_vec[:, 0] = torch.tensor([cells[c][1] for c in env_cell], device=dev)
    nova_common.route_resample(u, cmd_vec)
    cmd_term = u.command_manager.get_term("base_velocity")
    mass = robot.data.body_mass.torch.sum(dim=1).to(dev)

    dt = u.step_dt
    settle, total = int(args_cli.settle / dt), int((args_cli.settle + args_cli.duration) / dt)
    window = total - int(args_cli.free_window / dt)
    keys = ("n", "speed", "vx", "err", "p", "ss", "flight", "touch", "n_w", "qu", "qu2", "ql", "ql2")
    acc = {k: torch.zeros(n, device=dev) for k in keys}
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fall_events = torch.zeros(n, device=dev)
    prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        cmd_term.vel_command_b[:] = cmd_vec
        obs, _ = env.reset()
        for step in range(total):
            obs, _, dones, extras = env.step(policy(obs))
            fell = dones.bool()
            fall_events += fell.float()
            alive &= ~fell
            if step < settle:
                continue
            m = alive.float()
            v_b = robot.data.root_lin_vel_b.torch[:, :2]
            p_elec = power_model(u, rate_source).compute(u)[0].sum(dim=1)
            in_contact = contact.data.current_contact_time.torch[:, foot_ids] > 0.0
            touchdowns = (in_contact & ~prev_contact).sum(dim=1).float()
            prev_contact = in_contact
            if step == settle:  # first measured step: only initialise the contact state
                touchdowns = torch.zeros_like(touchdowns)
            acc["touch"] += m * touchdowns
            acc["n"] += m
            acc["speed"] += m * torch.linalg.norm(v_b, dim=1)
            acc["vx"] += m * v_b[:, 0]
            acc["err"] += m * torch.linalg.norm(v_b - cmd_vec[:, :2], dim=1)
            acc["p"] += m * p_elec
            acc["ss"] += m * (in_contact.sum(dim=1) == 1).float()
            acc["flight"] += m * (~in_contact).all(dim=1).float()
            if step >= window:
                q = robot.data.joint_pos.torch[:, prismatic.joint_ids]
                qu, ql = q[:, 0:2].mean(dim=1), q[:, 2:4].mean(dim=1)
                acc["n_w"] += m
                acc["qu"] += m * qu
                acc["qu2"] += m * qu**2
                acc["ql"] += m * ql
                acc["ql2"] += m * ql**2

    # ---- aggregate per cell
    rows = []
    for c, (mode, v, qu, ql) in enumerate(cells):
        sel = cell_of == c
        s = {k: acc[k][sel].sum().item() for k in keys}
        nn = max(s["n"], 1.0)
        speed = s["speed"] / nn if s["n"] else float("nan")
        p_mean = s["p"] / nn if s["n"] else float("nan")
        row = {
            "mode": mode,
            "speed_cmd": v,
            "q_upper": qu,
            "q_lower": ql,
            "total_q": qu + ql,
            "n_envs": int(sel.sum()),
            "n_fell": int((~alive[sel]).sum()),
            "fall_events": int(fall_events[sel].sum()),
            "speed_actual": speed,
            "vx_actual": s["vx"] / nn if s["n"] else float("nan"),
            "track_err": s["err"] / nn if s["n"] else float("nan"),
            "p_elec": p_mean,
            "cot": p_mean / (mass[sel].mean().item() * G * max(speed, 0.1)) if s["n"] else float("nan"),
            "single_stance": s["ss"] / nn if s["n"] else float("nan"),
            "flight": s["flight"] / nn if s["n"] else float("nan"),
            "step_freq": s["touch"] / (nn * dt) if s["n"] else float("nan"),
            "stride_length": s["speed"] * dt / (s["touch"] / 2.0) if s["touch"] >= 2 else float("nan"),
            "q_upper_std": float("nan"),
            "q_lower_std": float("nan"),
        }
        if mode == "free" and s["n_w"]:
            nw = s["n_w"]
            row["q_upper"], row["q_lower"] = s["qu"] / nw, s["ql"] / nw
            row["total_q"] = row["q_upper"] + row["q_lower"]
            row["q_upper_std"] = math.sqrt(max(s["qu2"] / nw - row["q_upper"] ** 2, 0.0))
            row["q_lower_std"] = math.sqrt(max(s["ql2"] / nw - row["q_lower"] ** 2, 0.0))
        rows.append(row)
    env.close()

    out = args_cli.out or os.path.join(
        os.path.dirname(resume_path), "morph_sweep_" + os.path.splitext(os.path.basename(resume_path))[0]
    )
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, lengths, speeds, out)
    print_summary(rows, speeds)
    print(f"[sweep] wrote {out}/sweep.csv, cot_heatmaps.png, optimal_length.png")
    sys.stdout.flush()  # Kit's shutdown exits without flushing a redirected stdout


def best_locked(rows, v):
    """Lowest-CoT locked cell at speed v among cells where at most half the envs fell (None if none)."""
    ok = [
        r
        for r in rows
        if r["mode"] != "free" and r["speed_cmd"] == v and not math.isnan(r["cot"]) and r["n_fell"] <= 0.5 * r["n_envs"]
    ]
    return min(ok, key=lambda r: r["cot"]) if ok else None


def plot(rows, lengths, speeds, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k = len(lengths)
    fig, axes = plt.subplots(1, len(speeds), figsize=(4.8 * len(speeds), 4.6), squeeze=False)
    fig.subplots_adjust(wspace=0.38)
    finite = [r["cot"] for r in rows if r["mode"] != "free" and math.isfinite(r["cot"])]
    vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
    ext = [lengths[0] - 0.015, lengths[-1] + 0.015, lengths[0] - 0.015, lengths[-1] + 0.015]
    for ax, v in zip(axes[0], speeds):
        grid = np.full((k, k), np.nan)  # [lower, upper]
        for r in rows:
            if r["mode"] != "free" and r["speed_cmd"] == v:
                grid[lengths.index(r["q_lower"]), lengths.index(r["q_upper"])] = r["cot"]
                ax.text(
                    r["q_upper"],
                    r["q_lower"],
                    f"{r['cot']:.2f}\n{r['n_fell']}F" if r["n_fell"] else f"{r['cot']:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="w",
                )
        im = ax.imshow(grid, origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
        free = next((r for r in rows if r["mode"] == "free" and r["speed_cmd"] == v), None)
        if free and math.isfinite(free["q_upper"]):
            ax.errorbar(
                free["q_upper"],
                free["q_lower"],
                xerr=free["q_upper_std"],
                yerr=free["q_lower_std"],
                fmt="*",
                color="r",
                ms=14,
                mec="k",
                capsize=3,
            )
        best = best_locked(rows, v)
        if best:
            ax.plot(best["q_upper"], best["q_lower"], "s", mfc="none", mec="r", ms=26, mew=2)
        has_free = any(r["mode"] == "free" for r in rows)
        free_txt = ""
        if has_free:
            free_txt = f" | FREE * CoT {free['cot']:.2f}" if free and math.isfinite(free["cot"]) else " | FREE: n/a"
        ax.set_title(f"cmd vx {v:.1f} m/s{free_txt}\n(red square: lowest-CoT cell, nF = falls)", fontsize=9)
        ax.set_xlabel("upper prismatic q [m]")
        ax.set_ylabel("lower prismatic q [m]")
        ax.set_xticks(lengths)
        ax.set_yticks(lengths)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="CoT = P_elec / (m g v)")
    fig.savefig(os.path.join(out, "cot_heatmaps.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bv = [(v, best_locked(rows, v)) for v in speeds]
    bv = [(v, b) for v, b in bv if b]
    ax.plot([v for v, _ in bv], [b["total_q"] for _, b in bv], "s-", label="lowest-CoT cell (q_U + q_L)")
    fr = [r for r in rows if r["mode"] == "free" and math.isfinite(r["total_q"])]
    ax.errorbar(
        [r["speed_cmd"] for r in fr],
        [r["total_q"] for r in fr],
        yerr=[math.hypot(r["q_upper_std"], r["q_lower_std"]) for r in fr],
        fmt="*-",
        ms=10,
        capsize=3,
        label="FREE policy choice (mean +/- std)",
    )
    ax.set_xlabel("commanded forward speed [m/s]")
    ax.set_ylabel("total prismatic extension q_U + q_L [m]")
    ax.set_ylim(0.0, 0.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "optimal_length.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows, speeds):
    print("\n=== MORPH SWEEP (grid: lowest-CoT cell; free (walking task): chosen lengths)")
    for v in speeds:
        b = best_locked(rows, v)
        locked = [r for r in rows if r["mode"] != "free" and r["speed_cmd"] == v]
        fell = sum(r["n_fell"] for r in locked)
        n = sum(r["n_envs"] for r in locked)
        fr = next((r for r in rows if r["mode"] == "free" and r["speed_cmd"] == v), None)
        btxt = (
            f"best U {b['q_upper']:.3f} L {b['q_lower']:.3f} CoT {b['cot']:.3f} (v {b['speed_actual']:.2f}, "
            f"{b['step_freq']:.2f} Hz, stride {b['stride_length']:.3f} m)"
            if b
            else "no cell with <= 50% falls"
        )
        ftxt = (
            f" | FREE U {fr['q_upper']:.4f}+/-{fr['q_upper_std']:.4f} L {fr['q_lower']:.4f}+/-{fr['q_lower_std']:.4f} "
            f"CoT {fr['cot']:.3f} v {fr['speed_actual']:.2f} err {fr['track_err']:.2f} "
            f"falls {fr['n_fell']}/{fr['n_envs']}"
            if fr
            else ""
        )
        print(f"  v {v:.1f}: grid falls {fell}/{n}; {btxt}{ftxt}")


if __name__ == "__main__":
    main()
    simulation_app.close()
