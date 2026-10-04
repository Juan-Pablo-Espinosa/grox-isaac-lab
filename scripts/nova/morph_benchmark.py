# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible morphology benchmark of a morphology-agnostic NOVA walking policy.

Launch (headless)::

    ./isaaclab.sh -p scripts/nova/morph_benchmark.py --checkpoint logs/rsl_rl/nova_morph_agnostic/<run>/model_N.pt \\
        --headless   # [--envs_per_cell 64] [--settle 3] [--duration 10] [--seed 42] [--only mid:fwd1.0,...] [--out DIR]

All cells run side by side in ONE simulation: 5 morphologies (q_U, q_L) x 9 commands (forward vx 0..2.5, lateral vy
0.4, yaw in place wz 1.0, backward vx -0.5), ``--envs_per_cell`` envs each. Deterministic policy, no observation
noise, no pushes, the random leg-length schedule OFF and every env's leg lengths pinned to its cell (falls respawn at
the same lengths), commands fixed (no resampling), fixed seed. ``--settle`` s, then ``--duration`` s measured. An env
that falls at any time is counted in the fall rate and excluded from every other metric.

Sampling: every control step (50 Hz), from the actuator state of the last physics substep of that step (explicit
actuators: computed vs applied torque, ankle motor torques from the linkage actuator itself).

Output (default <checkpoint dir>/benchmark_<timestamp>/): summary.csv (one row per cell), report.md (self-contained
markdown, also printed), and PNG plots.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Morphology benchmark of a morph-agnostic NOVA policy.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-MorphAgnostic-Play-v0")
parser.add_argument("--envs_per_cell", type=int, default=64)
parser.add_argument("--settle", type=float, default=3.0)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--only", type=str, default=None, help="Comma list of cells 'morph:command', e.g. mid:fwd1.0.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=42, help="Env / torch seed (reset randomization).")
parser.add_argument(
    "--layout_note", type=str, default=None, help="Measured layout sensitivity, quoted in the report's caveats."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import csv  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import nova_common  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.power import (  # noqa: E402
    DASHBOARD_GROUPS,
    PRISMATIC_DRIVER,
    ROBSTRIDE,
    dashboard_group,
    power_model,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

G = 9.81
MORPHS = {
    "short": (0.005, 0.005),
    "mid": (0.050, 0.050),
    "long": (0.095, 0.095),
    "shortU-longL": (0.005, 0.095),
    "longU-shortL": (0.095, 0.005),
}
COMMANDS = {f"fwd{v:.1f}": (v, 0.0, 0.0) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)}
COMMANDS.update({"lat0.4": (0.0, 0.4, 0.0), "yaw1.0": (0.0, 0.0, 1.0), "back0.5": (-0.5, 0.0, 0.0)})
TORQUE_GROUPS = {
    "hip_knee": ("Hip_Pitch_", "Lowerleg_Pitch_"),
    "roll": ("Hip_Roll_",),
    "yaw": ("Upperleg_Yaw_",),
    "ankle": ("Feet_Roll_", "Feet_Pitch_"),
}
GROUP_MOTOR = {"hip_knee": "RS04", "roll": "RS03", "yaw": "RS06"}
ANKLE_MOTORS = ("L_m1", "L_m2", "R_m1", "R_m2")
CAP_FRAC = 0.99  # "at the cap": |tau_motor| >= 99% of the 6 N·m limit


def cells_layout(only: set[str] | None):
    cells = [(m, c) for m in MORPHS for c in COMMANDS if only is None or f"{m}:{c}" in only]
    if not cells:
        raise SystemExit(f"[benchmark] no cell matches --only {sorted(only)}")
    return cells


def main():
    t_start = time.time()
    only = set(args_cli.only.split(",")) if args_cli.only else None
    cells = cells_layout(only)
    E = args_cli.envs_per_cell
    n = len(cells) * E
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    if nova_common.checkpoint_action_dim(resume_path) != 12:
        raise SystemExit(f"[benchmark] {resume_path} is not a 12-D morphology-agnostic policy.")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    if not hasattr(env_cfg, "prismatic_driver"):
        raise SystemExit(f"[benchmark] {args_cli.task} is not the morphology-agnostic task.")
    env_cfg.seed = args_cli.seed
    nova_common.pin_commands(env_cfg)
    env_cfg.events.base_external_force_torque = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.prismatic_driver.schedule_enabled = False
    env_cfg.events.reset_nova.params["prismatic_per_env"] = [MORPHS[m] for m, _ in cells for _ in range(E)]
    torch.manual_seed(args_cli.seed)
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)
    u = env.unwrapped
    dev = u.device
    robot, contact = u.scene["robot"], u.scene["contact_forces"]
    names = list(robot.joint_names)
    feet = robot.actuators["feet"]
    n_p, n_r = feet.cfg.n_pitch, feet.cfg.n_roll
    cap = CAP_FRAC * feet.cfg.motor_torque_limit
    foot_ids = [contact.body_names.index(b) for b in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
    group_joints = {g: [i for i, nm in enumerate(names) if nm.startswith(p)] for g, p in TORQUE_GROUPS.items()}
    pitch_j = [names.index(f"Feet_Pitch_{s}_Joint") for s in ("Left", "Right")]
    roll_j = [names.index(f"Feet_Roll_{s}_Joint") for s in ("Left", "Right")]
    # explicit revolute actuators (hip_knee / roll / yaw): joint ids + per-joint effort limits for clip detection
    env_acts = [a for k, a in robot.actuators.items() if k != "feet" and not a.is_implicit_model]
    no_load = torch.zeros(len(names), device=dev)
    for g, motor in GROUP_MOTOR.items():
        no_load[group_joints[g]] = ROBSTRIDE[motor].no_load_speed
    model = power_model(u, PRISMATIC_DRIVER)
    gmat = torch.zeros(len(model.labels), len(DASHBOARD_GROUPS), device=dev)
    for i, lab in enumerate(model.labels):
        gmat[i, DASHBOARD_GROUPS.index(dashboard_group(lab))] = 1.0
    cmd_vec = torch.tensor([COMMANDS[c] for _, c in cells for _ in range(E)], device=dev)
    nova_common.route_resample(u, cmd_vec)
    cmd_term = u.command_manager.get_term("base_velocity")
    mass = robot.data.body_mass.torch.sum(dim=1).to(dev)

    dt = u.step_dt
    settle, total = int(args_cli.settle / dt), int((args_cli.settle + args_cli.duration) / dt)
    acc_keys = [
        "n",
        "vx",
        "vy",
        "wz",
        "speed",
        "err_xy",
        "err_yaw",
        "p",
        "mech",
        "cu",
        "touch",
        "contact",
        "ss",
        "flight",
    ]
    acc_keys += [f"p_{g}" for g in DASHBOARD_GROUPS] + [f"cu_{g}" for g in DASHBOARD_GROUPS]
    acc = {k: torch.zeros(n, device=dev) for k in acc_keys}
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    store = {"tau_frac": [], "clip": [], "speed_frac": [], "motor": [], "foot": [], "tp": [], "tr": []}
    with torch.inference_mode():
        cmd_term.vel_command_b[:] = cmd_vec
        obs, _ = env.reset()
        for step in range(total):
            obs, _, dones, _ = env.step(policy(obs))
            fell |= dones.bool()
            if step < settle:
                continue
            d = robot.data
            v = d.root_lin_vel_b.torch[:, :2]
            wz = d.root_ang_vel_b.torch[:, 2]
            p_m, mech_m, cu_m, _ = model.compute(u)
            in_contact = contact.data.current_contact_time.torch[:, foot_ids] > 0.0
            touch = (in_contact & ~prev_contact).sum(dim=1).float() if step > settle else torch.zeros(n, device=dev)
            prev_contact = in_contact
            acc["n"] += 1.0
            acc["vx"] += v[:, 0]
            acc["vy"] += v[:, 1]
            acc["wz"] += wz
            acc["speed"] += torch.linalg.norm(v, dim=1)
            acc["err_xy"] += torch.linalg.norm(v - cmd_vec[:, :2], dim=1)
            acc["err_yaw"] += (wz - cmd_vec[:, 2]).abs()
            acc["p"] += p_m.sum(dim=1)
            acc["mech"] += mech_m.sum(dim=1)
            acc["cu"] += cu_m.sum(dim=1)
            pg, cg = p_m @ gmat, cu_m @ gmat
            for i, g in enumerate(DASHBOARD_GROUPS):
                acc[f"p_{g}"] += pg[:, i]
                acc[f"cu_{g}"] += cg[:, i]
            acc["touch"] += touch
            acc["contact"] += in_contact.float().sum(dim=1)
            acc["ss"] += (in_contact.sum(dim=1) == 1).float()
            acc["flight"] += (~in_contact).all(dim=1).float()
            tau = d.applied_torque.torch
            store["tau_frac"].append((tau.abs() / d.joint_effort_limits.torch).half())
            clip = torch.zeros(n, len(names), dtype=torch.bool, device=dev)
            for act in env_acts:  # clipped by the torque-speed envelope (not by the static effort limit)
                ce, ae = act.computed_effort, act.applied_effort
                hit = (ce.abs() > ae.abs() + 1e-4) & (ae.abs() < act.effort_limit - 1e-3)
                clip[:, act.joint_indices] = hit
            store["clip"].append(clip)
            store["speed_frac"].append((d.joint_vel.torch.abs() / no_load.clamp(min=1e-9)).half())
            store["motor"].append(feet.motor_torque.clone())
            store["foot"].append(in_contact)
            store["tp"].append(tau[:, pitch_j].abs())
            store["tr"].append(tau[:, roll_j].abs())
    st = {k: torch.stack(v) for k, v in store.items()}  # (T, n, ...)
    T = st["motor"].shape[0]

    rows = []
    for ci, (morph, cname) in enumerate(cells):
        sel = torch.zeros(n, dtype=torch.bool, device=dev)
        sel[ci * E : (ci + 1) * E] = True
        ok = sel & ~fell
        cmd = COMMANDS[cname]
        moving = math.hypot(cmd[0], cmd[1]) > 0
        r = {
            "morph": morph,
            "q_U": MORPHS[morph][0],
            "q_L": MORPHS[morph][1],
            "command": cname,
            "cmd_vx": cmd[0],
            "cmd_vy": cmd[1],
            "cmd_wz": cmd[2],
            "n_envs": E,
            "n_fell": int((sel & fell).sum()),
        }
        r["fall_rate"] = r["n_fell"] / E
        k = ok.nonzero().flatten()
        if len(k) == 0:
            rows.append(r)
            continue
        a = {key: acc[key][k] / acc["n"][k] for key in acc_keys if key != "n"}
        dur = acc["n"][k] * dt
        speed = a["speed"]
        per_env = {
            "speed": speed,
            "vx": a["vx"],
            "vy": a["vy"],
            "wz": a["wz"],
            "track_err_xy": a["err_xy"],
            "track_err_yaw": a["err_yaw"],
            "p_elec": a["p"],
            "p_mech": a["mech"],
            "p_copper": a["cu"],
            "step_freq": acc["touch"][k] / dur,
            "duty_factor": a["contact"] / 2.0,
            "single_stance": a["ss"],
            "flight": a["flight"],
        }
        for g in DASHBOARD_GROUPS:
            per_env[f"p_{g}"] = a[f"p_{g}"]
            per_env[f"cu_{g}"] = a[f"cu_{g}"]
        stride = speed * dur / (acc["touch"][k] / 2.0).clamp(min=1e-9)
        per_env["stride_length"] = torch.where(acc["touch"][k] >= 2, stride, torch.full_like(stride, float("nan")))
        if moving:
            per_env["cot"] = a["p"] / (mass[k] * G * speed.clamp(min=0.1))
            per_env["energy_per_m"] = a["p"] / speed.clamp(min=0.1)
        for key, x in per_env.items():
            x = x[torch.isfinite(x)]
            r[key] = float(x.mean()) if len(x) else float("nan")
            r[key + "_std"] = float(x.std()) if len(x) > 1 else float("nan")
        if not moving:
            r["cot"] = r["cot_std"] = r["energy_per_m"] = r["energy_per_m_std"] = float("nan")
        # ---- per joint group torque / clipping / speed (pooled over the cell's surviving env-steps)
        tf = st["tau_frac"][:, k].float()
        for g, js in group_joints.items():
            x = tf[:, :, js].flatten()
            r[f"{g}_tau_mean_pct"] = float(x.mean() * 100)
            r[f"{g}_tau_p95_pct"] = float(torch.quantile(x[:16_000_000], 0.95) * 100)
            r[f"{g}_tau_max_pct"] = float(x.max() * 100)
            if g in GROUP_MOTOR:
                r[f"{g}_envelope_clip_frac"] = float(st["clip"][:, k][:, :, js].float().mean())
                sf = st["speed_frac"][:, k][:, :, js].float().flatten()
                r[f"{g}_speed_p95_pct_noload"] = float(torch.quantile(sf[:16_000_000], 0.95) * 100)
        # ---- ankle motors
        mt = st["motor"][:, k].abs()  # (T, envs, 4): L m1, L m2, R m1, R m2
        foot = st["foot"][:, k]  # (T, envs, 2): L, R in contact
        stance = foot[:, :, [0, 0, 1, 1]]
        at_cap = mt >= cap
        cont, peak = ROBSTRIDE["RS02"].rated_torque, ROBSTRIDE["RS02"].peak_torque
        x = mt.flatten()
        r["ankle_mean_pct_cont"] = float(x.mean() / cont * 100)
        r["ankle_rms_pct_cont"] = float(x.pow(2).mean().sqrt() / cont * 100)
        r["ankle_p95_pct_cont"] = float(torch.quantile(x[:16_000_000], 0.95) / cont * 100)
        r["ankle_max_pct_cont"] = float(x.max() / cont * 100)
        r["ankle_rms_pct_peak"] = float(x.pow(2).mean().sqrt() / peak * 100)
        r["ankle_p95_pct_peak"] = float(torch.quantile(x[:16_000_000], 0.95) / peak * 100)
        r["ankle_max_pct_peak"] = float(x.max() / peak * 100)
        r["ankle_cap_frac"] = float(at_cap.float().mean())
        r["ankle_any4_cap_frac"] = float(at_cap.any(dim=-1).float().mean())
        r["ankle_cap_frac_stance"] = float(at_cap[stance].float().mean()) if stance.any() else float("nan")
        r["ankle_cap_frac_swing"] = float(at_cap[~stance].float().mean()) if (~stance).any() else float("nan")
        r["ankle_cap_share_in_stance"] = float(stance[at_cap].float().mean()) if at_cap.any() else float("nan")
        r["ankle_stance_frac"] = float(stance.float().mean())
        tp, tr = st["tp"][:, k], st["tr"][:, k]  # (T, envs, 2) joint |tau|
        pitch_part = (tp / (2.0 * n_p))[:, :, [0, 0, 1, 1]]
        roll_part = (tr / (2.0 * n_r))[:, :, [0, 0, 1, 1]]
        tot = pitch_part + roll_part
        r["ankle_pitch_share"] = float(pitch_part.sum() / tot.sum().clamp(min=1e-9))
        r["ankle_pitch_share_at_cap"] = (
            float(pitch_part[at_cap].sum() / tot[at_cap].sum().clamp(min=1e-9)) if at_cap.any() else float("nan")
        )
        r["ankle_joint_pitch_mean_Nm"] = float(tp.mean())
        r["ankle_joint_roll_mean_Nm"] = float(tr.mean())
        for mi, mn in enumerate(ANKLE_MOTORS):
            xm = mt[:, :, mi].flatten()
            r[f"ankle_{mn}_mean_pct_cont"] = float(xm.mean() / cont * 100)
            r[f"ankle_{mn}_rms_pct_cont"] = float(xm.pow(2).mean().sqrt() / cont * 100)
            r[f"ankle_{mn}_p95_pct_cont"] = float(torch.quantile(xm, 0.95) / cont * 100)
            r[f"ankle_{mn}_max_pct_peak"] = float(xm.max() / peak * 100)
            r[f"ankle_{mn}_cap_frac"] = float(at_cap[:, :, mi].float().mean())
        rows.append(r)
    env.close()

    out = args_cli.out or os.path.join(os.path.dirname(resume_path), "benchmark_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    keys = []
    for r in rows:
        keys += [kk for kk in r if kk not in keys]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "envs_per_cell": E,
        "settle": args_cli.settle,
        "duration": args_cli.duration,
        "seed": args_cli.seed,
        "samples": T,
        "dt": dt,
        "n_p": n_p,
        "n_r": n_r,
        "runtime_s": time.time() - t_start,
        "layout_note": args_cli.layout_note,
    }
    report = make_report(rows, meta)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report)
    make_plots(rows, out)
    print("\n" + report)
    print(f"[benchmark] output: {out}  (runtime {meta['runtime_s']:.0f} s)")
    sys.stdout.flush()  # Kit's shutdown exits without flushing a redirected stdout


# ----------------------------------------------------------------------------------------------------------------------
# report


def fmt(x, digits=2, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}" + ("%" if pct else "")


def pm(r, key, digits=2):
    if key not in r or not math.isfinite(r.get(key, float("nan"))):
        return "n/a"
    s = r.get(key + "_std", float("nan"))
    return f"{r[key]:.{digits}f} ± {s:.{digits}f}" if math.isfinite(s) else f"{r[key]:.{digits}f}"


def cell_table(rows, yaw: bool = False):
    head = (
        ["morphology (q_U, q_L)", "speed m/s", "track err xy m/s"]
        + (["yaw err rad/s"] if yaw else [])
        + [
            "falls",
            "P_elec W",
            "CoT",
            "step Hz",
            "stride m",
            "ankle RMS %cont",
            "ankle cap-frac",
            "hip_knee p95 %",
            "roll p95 %",
        ]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        m = f"{r['morph']} ({r['q_U'] * 1000:.0f}, {r['q_L'] * 1000:.0f} mm)"
        vals = (
            [m, pm(r, "speed"), pm(r, "track_err_xy")]
            + ([pm(r, "track_err_yaw")] if yaw else [])
            + [
                f"{r['n_fell']}/{r['n_envs']}",
                pm(r, "p_elec", 0),
                pm(r, "cot"),
                pm(r, "step_freq"),
                pm(r, "stride_length"),
                fmt(r.get("ankle_rms_pct_cont"), 0, True),
                fmt(r.get("ankle_cap_frac"), 3),
                fmt(r.get("hip_knee_tau_p95_pct"), 0, True),
                fmt(r.get("roll_tau_p95_pct"), 0, True),
            ]
        )
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def ankle_table(rows):
    head = [
        "command",
        "morphology",
        "cap-frac (per motor)",
        "any-of-4 at cap",
        "cap-frac in stance",
        "cap-frac in swing",
        "share of cap time in stance",
        "pitch share (all)",
        "pitch share (at cap)",
        "mean |tau_pitch| N·m",
        "mean |tau_roll| N·m",
        "RMS %cont",
        "p95 %cont",
        "max %peak",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        if "ankle_cap_frac" not in r:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    r["command"],
                    r["morph"],
                    fmt(r["ankle_cap_frac"], 3),
                    fmt(r["ankle_any4_cap_frac"], 3),
                    fmt(r["ankle_cap_frac_stance"], 3),
                    fmt(r["ankle_cap_frac_swing"], 3),
                    fmt(r["ankle_cap_share_in_stance"], 2),
                    fmt(r["ankle_pitch_share"], 2),
                    fmt(r["ankle_pitch_share_at_cap"], 2),
                    fmt(r["ankle_joint_pitch_mean_Nm"], 1),
                    fmt(r["ankle_joint_roll_mean_Nm"], 1),
                    fmt(r["ankle_rms_pct_cont"], 0, True),
                    fmt(r["ankle_p95_pct_cont"], 0, True),
                    fmt(r["ankle_max_pct_peak"], 0, True),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def make_report(rows, meta) -> str:
    ok = [r for r in rows if "speed" in r]
    out = [
        f"# NOVA morphology benchmark — {os.path.basename(os.path.dirname(meta['checkpoint']))}/"
        f"{os.path.basename(meta['checkpoint'])}",
        "",
    ]
    out += [
        "## (a) Setup",
        "",
        f"- Policy: `{meta['checkpoint']}` (task `{meta['task']}`, 12-D action, leg lengths set externally), "
        f"code commit `{meta['commit']}`.",
        f"- Deterministic policy (mean action), no observation noise, no pushes, random leg-length schedule OFF, "
        f"leg lengths pinned per cell (falls respawn at the same lengths), commands fixed, seed {meta['seed']}.",
        f"- {meta['envs_per_cell']} envs per cell, all cells in one simulation; {meta['settle']:.0f} s settle, "
        f"{meta['duration']:.0f} s measured ({meta['samples']} control steps at {1 / meta['dt']:.0f} Hz). "
        f"Runtime {meta['runtime_s']:.0f} s.",
        "- Morphologies: prismatic extensions q_U (thigh) and q_L (shank) in mm; 5 = shortest, 95 = longest.",
        "- Actuators as trained: explicit PD with RobStride 48 V torque-speed envelopes and rotor armature; ankle = "
        f"2x RS02 per side through a linkage with N_pitch = {meta['n_p']:g}, N_roll = {meta['n_r']:g}; each RS02 "
        "limited to 6 N·m (its continuous rating), i.e. |tau_pitch|/3 + |tau_roll| <= 12 N·m.",
        "- Definitions: speed = mean planar base speed |v_xy|; track err = mean |v_xy - cmd_xy| (and |w_z - cmd|); "
        "values are mean ± std across envs; falls = envs that terminated at any time (excluded from all other "
        "metrics); P_elec = sum over 16 motors of max(tau w + 1.5 R (tau/Kt)^2, 0) (datasheet constants, no "
        "regeneration); CoT = P_elec / (m g max(speed, 0.1)) per env (n/a for commands with no planar motion); "
        "step Hz = touchdowns of both feet per second; stride = distance per two touchdowns; % values are |torque| "
        "over the joint's sim limit (hip/knee 120, roll 60, yaw 36, ankle pitch 36 / roll 12 N·m); ankle "
        "%cont / %peak = RS02 motor torque over 6 / 17 N·m; cap-frac = fraction of samples with motor torque "
        f">= {CAP_FRAC * 100:.0f}% of 6 N·m.",
        "- Caveats: ankle linkage ratios N_p = 3, N_r = 1 are NOT yet confirmed from CAD (they set the motor "
        "torques and the ankle copper loss); R for RS00/RS02/RS06 from the 2026-09-17 datasheet; torques are "
        "sampled once per control step from the last physics substep; simulation only (flat ground, PhysX).",
        "- The ankle motors cannot exceed the 6 N·m cap in this model, so max %peak is bounded at 6/17 = 35%; "
        "%peak is reported only to relate the cap to the RS02's 17 N·m peak rating.",
        "- Reproducibility: rerunning with the same cell layout and seed reproduces every number exactly. "
        + (
            meta["layout_note"]
            or "Changing the layout (number of cells / envs) changes the reset randomization "
            "and GPU contact ordering, so per-cell values move slightly between layouts."
        ),
        "",
    ]
    out += ["## (b) Forward walking (one table per commanded vx)", ""]
    for c in [c for c in COMMANDS if c.startswith("fwd")]:
        rs = [r for r in ok if r["command"] == c]
        if rs:
            out += [f"### vx = {COMMANDS[c][0]:.1f} m/s", "", cell_table(rs), ""]
    out += ["## (c) Lateral, yaw-in-place, backward", ""]
    for c, title in (
        ("lat0.4", "lateral vy = 0.4 m/s"),
        ("yaw1.0", "yaw in place w_z = 1.0 rad/s"),
        ("back0.5", "backward vx = -0.5 m/s"),
    ):
        rs = [r for r in ok if r["command"] == c]
        if rs:
            out += [f"### {title}", "", cell_table(rs, yaw=(c == "yaw1.0")), ""]
    out += ["## (d) Ankle", "", ankle_table(ok), ""]
    out += ankle_answer(ok)
    out += ["## (e) Observations", ""] + observations(rows) + [""]
    return "\n".join(out)


def ankle_answer(rows):
    if not rows:
        return []
    cap = [r["ankle_cap_frac"] for r in rows]
    any4 = [r["ankle_any4_cap_frac"] for r in rows]
    st = [r["ankle_cap_share_in_stance"] for r in rows if math.isfinite(r["ankle_cap_share_in_stance"])]
    ps = [r["ankle_pitch_share_at_cap"] for r in rows if math.isfinite(r["ankle_pitch_share_at_cap"])]
    rms = [r["ankle_rms_pct_cont"] for r in rows]
    hi = max(rows, key=lambda r: r["ankle_cap_frac"])
    lo = min(rows, key=lambda r: r["ankle_cap_frac"])
    return [
        '### Answer to "the ankle seems almost always at top torque"',
        "",
        f"- Per motor, the fraction of time at the 6 N·m cap ranges from {min(cap):.3f} ({lo['morph']}, "
        f"{lo['command']}) to {max(cap):.3f} ({hi['morph']}, {hi['command']}); median over cells "
        f"{sorted(cap)[len(cap) // 2]:.3f}.",
        f"- The fraction of time that AT LEAST ONE of the 4 ankle motors is at the cap ranges from {min(any4):.3f} to "
        f"{max(any4):.3f} (median {sorted(any4)[len(any4) // 2]:.3f}). The teleop dashboard's solid black line is "
        "exactly this worst-of-4 quantity (max over both feet), so it reads higher than any single motor.",
        f"- Of the time a motor is at the cap, the share during stance of its own foot is "
        f"{min(st):.2f}-{max(st):.2f} across cells; the pitch component's share of the motor torque at the cap is "
        f"{min(ps):.2f}-{max(ps):.2f}."
        if st and ps
        else "- No cap events.",
        f"- RMS ankle motor torque is {min(rms):.0f}-{max(rms):.0f}% of the 6 N·m continuous rating across cells.",
        "",
    ]


def observations(rows):
    ok = [r for r in rows if "speed" in r]
    obs = []
    for c in [c for c in COMMANDS if c.startswith("fwd") and COMMANDS[c][0] > 0]:
        rs = [r for r in ok if r["command"] == c and math.isfinite(r.get("cot", float("nan")))]
        if len(rs) < 2:
            continue
        lo, hi = min(rs, key=lambda r: r["cot"]), max(rs, key=lambda r: r["cot"])
        fs = min(rs, key=lambda r: r["step_freq"]), max(rs, key=lambda r: r["step_freq"])
        obs.append(
            f"- vx {COMMANDS[c][0]:.1f}: CoT lowest {lo['cot']:.2f} ({lo['morph']}), highest {hi['cot']:.2f} "
            f"({hi['morph']}), ratio {hi['cot'] / lo['cot']:.2f}; actual speed {min(r['speed'] for r in rs):.2f}-"
            f"{max(r['speed'] for r in rs):.2f} m/s; step frequency {fs[0]['step_freq']:.2f} Hz ({fs[0]['morph']})"
            f" to {fs[1]['step_freq']:.2f} Hz ({fs[1]['morph']}); ankle cap-frac "
            f"{min(r['ankle_cap_frac'] for r in rs):.3f}-{max(r['ankle_cap_frac'] for r in rs):.3f}."
        )
    falls = [r for r in rows if r["n_fell"] > 0]
    if falls:
        worst = sorted(falls, key=lambda r: -r["n_fell"])[:5]
        obs.append(
            "- Falls: "
            + ", ".join(f"{r['morph']} {r['command']} {r['n_fell']}/{r['n_envs']}" for r in worst)
            + f" ({len(falls)} of {len(rows)} cells had at least one fall)."
        )
    else:
        obs.append(f"- Falls: none in any of the {len(rows)} cells.")
    track = sorted(ok, key=lambda r: -r["track_err_xy"])[:3]
    obs.append(
        "- Largest xy tracking error: "
        + ", ".join(f"{r['morph']} {r['command']} {r['track_err_xy']:.2f} m/s" for r in track)
        + "."
    )
    for g in ("hip_knee", "roll"):
        hi = max(ok, key=lambda r: r[f"{g}_tau_p95_pct"])
        obs.append(
            f"- Highest {g} p95 torque: {hi[f'{g}_tau_p95_pct']:.0f}% of limit ({hi['morph']}, {hi['command']}); "
            f"highest envelope-clip fraction {max(r[f'{g}_envelope_clip_frac'] for r in ok):.3f}."
        )
    sp = max(ok, key=lambda r: r["hip_knee_speed_p95_pct_noload"])
    obs.append(
        f"- Highest hip/knee p95 joint speed: {sp['hip_knee_speed_p95_pct_noload']:.0f}% of RS04 no-load "
        f"({sp['morph']}, {sp['command']})."
    )
    return obs


# ----------------------------------------------------------------------------------------------------------------------
# plots


def make_plots(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if "speed" in r]
    fwd = [c for c in COMMANDS if c.startswith("fwd")]

    def series(key, morph):
        xs, ys = [], []
        for c in fwd:
            r = next((r for r in ok if r["morph"] == morph and r["command"] == c), None)
            if r is not None and math.isfinite(r.get(key, float("nan"))):
                xs.append(COMMANDS[c][0])
                ys.append(r[key])
        return xs, ys

    def line_plot(keys_titles, fname):
        fig, axes = plt.subplots(1, len(keys_titles), figsize=(5.5 * len(keys_titles), 4.2), squeeze=False)
        for ax, (key, title) in zip(axes[0], keys_titles):
            for m in MORPHS:
                xs, ys = series(key, m)
                ax.plot(xs, ys, "o-", label=m)
            ax.set_xlabel("commanded vx [m/s]")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.savefig(os.path.join(out, fname), dpi=110, bbox_inches="tight")
        plt.close(fig)

    line_plot([("cot", "CoT")], "cot_vs_speed.png")
    line_plot([("step_freq", "step frequency [Hz]"), ("stride_length", "stride length [m]")], "gait_vs_speed.png")
    line_plot(
        [
            ("ankle_cap_frac", "ankle motor fraction of time at 6 N·m cap"),
            ("ankle_any4_cap_frac", "fraction of time ANY of 4 ankle motors at cap"),
        ],
        "ankle_cap_vs_speed.png",
    )
    line_plot([("ankle_rms_pct_cont", "ankle motor RMS torque [% of 6 N·m]")], "ankle_rms_vs_speed.png")
    line_plot(
        [("fall_rate", "fall rate (fraction of envs)"), ("track_err_xy", "tracking error xy [m/s]")],
        "falls_tracking_vs_speed.png",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    groups = list(TORQUE_GROUPS)
    for ax, c in zip(axes, ("fwd1.0", "fwd2.0")):
        width = 0.8 / len(MORPHS)
        for i, m in enumerate(MORPHS):
            r = next((r for r in ok if r["morph"] == m and r["command"] == c), None)
            if r is None:
                continue
            ax.bar([j + i * width for j in range(len(groups))], [r[f"{g}_tau_p95_pct"] for g in groups], width, label=m)
            ax.scatter(
                [j + i * width for j in range(len(groups))],
                [r[f"{g}_tau_mean_pct"] for g in groups],
                marker="_",
                color="k",
                s=60,
            )
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(groups))], groups)
        ax.set_title(f"|tau| / joint limit at vx {COMMANDS[c][0]:.1f}: p95 (bars), mean (ticks) [%]", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, "torque_groups_1_2ms.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
    simulation_app.close()
