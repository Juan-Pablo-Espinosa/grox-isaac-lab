# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare NOVA teleop recordings (CSV files written by scripts/nova/teleop_walk.py, key V).

Usage (plain Python, no simulator)::

    ~/env_isaaclab/bin/python scripts/nova/compare_recordings.py rec_A.csv rec_B.csv [...] [--settle 3] [--out PNG]
        [--transmission {recorded,direct,linkage} [--n_pitch 3 --n_roll 1]]

--transmission re-derives the four ankle motors from the recorded ankle joint torques / velocities (so recordings made
with an older model can be re-analysed): direct = the run-5 model tau_1,2 = (tau_p +/- tau_r) / 2, w_1,2 = w_p +/- w_r;
linkage = tau_1,2 = (tau_p / N_p +/- tau_r / N_r) / 2, w_1,2 = N_p w_p +/- N_r w_r (defaults N_p = 3, N_r = 1).
Per motor P = max(tau w + 1.5 R (tau / Kt)^2, 0) with RS02 R = 0.58 (direct) / 0.55 (linkage, datasheet); the ankle
group, P_elec totals
and the ankle RS02 load are replaced accordingly. "recorded" (default) uses the CSV as written.

The first ``--settle`` seconds of every recording are discarded. Per recording:
  * mean actual speed |v_xy| and v_x (base frame) [m/s]; tracking error mean |v_xy - cmd_xy| [m/s]
  * mean P_elec [W]; energy [J]; distance walked (base path length, reset jumps removed) [m]
  * CoT = energy / (m g distance); energy per metre [J/m]
  * mean and p95 of |tau| / tau_max per joint group (hip_knee, roll, yaw, ankle)
  * ankle RS02 torque / 6 N·m rated (max motor per step): mean and p95 [%]
  * step frequency (touchdowns of both feet per second) [Hz] and stride length (distance per two steps) [m]
  * ankle group mean P and copper P [W]; mean q_U, q_L [mm]; falls (reset jumps) in the window
Prints a table (one column per recording) and saves overlay plots (default next to the first CSV).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

GROUPS = ("hip_knee", "roll", "yaw", "ankle")
_POWER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "source",
    "isaaclab_tasks",
    "isaaclab_tasks",
    "manager_based",
    "locomotion",
    "walking",
    "config",
    "nova",
    "mdp",
    "power.py",
)


def _power_module():
    """The task's power model module (torch-only at import), loaded by path so no Isaac Lab import is needed."""
    spec = importlib.util.spec_from_file_location("nova_power_model", _POWER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def retransmit(df: pd.DataFrame, n_p: float, n_r: float) -> pd.DataFrame:
    """Recompute the ankle motors (and every total that includes them) with transmission ratios (n_p, n_r)."""
    pm = _power_module()
    rs02 = pm.ROBSTRIDE["RS02"]
    # linkage = the morph-agnostic model (datasheet RS02 resistance); direct = the run-5 model and constants
    r = pm.DATASHEET_RESISTANCE["RS02"] if (n_p, n_r) != (1.0, 1.0) else rs02.r_terminal
    k_cu = 1.5 * r / rs02.kt**2
    df = df.copy()
    new_p = new_cu = new_mech = 0.0
    fracs = []
    for side in ("Left", "Right"):
        tp, tr = df[f"tau_Feet_Pitch_{side}_Joint"], df[f"tau_Feet_Roll_{side}_Joint"]
        wp, wr = df[f"qd_Feet_Pitch_{side}_Joint"], df[f"qd_Feet_Roll_{side}_Joint"]
        for m, sgn in (("m1", 1.0), ("m2", -1.0)):
            tau = 0.5 * (tp / n_p + sgn * tr / n_r)
            w = n_p * wp + sgn * n_r * wr
            cu = k_cu * tau**2
            p = np.maximum(tau * w + cu, 0.0)
            label = f"ankle_{side}_{m}"
            df[f"p_{label}"] = p
            df[f"ankle_rs02_frac_{label}"] = tau.abs() / rs02.rated_torque
            fracs.append(tau.abs())
            new_p, new_cu, new_mech = new_p + p, new_cu + cu, new_mech + (p - cu)
    for total, group, new in (
        ("p_elec", "p_ankle", new_p),
        ("p_copper", "cu_ankle", new_cu),
        ("p_mech", "mech_ankle", new_mech),
    ):
        df[total] = df[total] - df[group] + new
        df[group] = new
    peak = pd.concat(fracs, axis=1).max(axis=1)
    df["ankle_rs02_frac_max"] = peak / rs02.rated_torque
    if "ankle_rs02_peak_frac_max" in df:
        df["ankle_rs02_peak_frac_max"] = peak / rs02.peak_torque
    df["ankle_n_pitch"], df["ankle_n_roll"] = n_p, n_r
    return df


def label_of(path: str) -> str:
    """Short label from the recording file name (command and leg length)."""
    m = re.search(r"rec_\d{8}-\d{6}_(.*)\.csv$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def analyze(path: str, settle: float, transmission: tuple[float, float] | None = None) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    if transmission is not None:
        df = retransmit(df, *transmission)
    dt = float(np.median(np.diff(df["t"].to_numpy())))
    w = df[df["t"] >= df["t"].iloc[0] + settle].reset_index(drop=True)
    if len(w) < 2:
        raise SystemExit(f"{path}: shorter than the {settle} s settle window")
    step = np.hypot(np.diff(w["pos_x"]), np.diff(w["pos_y"]))
    jumps = step > 0.5  # a reset teleports the base
    dist = float(step[~jumps].sum())
    duration = len(w) * dt
    energy = float(w["p_elec"].sum() * dt)
    mass = float(w["mass"].mean())
    speed = np.hypot(w["vx"], w["vy"])
    err = np.hypot(w["vx"] - w["cmd_vx"], w["vy"] - w["cmd_vy"])
    touch = 0
    for foot in ("contact_L", "contact_R"):
        c = w[foot].to_numpy() > 0.5
        touch += int((c[1:] & ~c[:-1]).sum())
    r = {
        "duration_s": duration,
        "cmd_vx": float(w["cmd_vx"].mean()),
        "speed_mean": float(speed.mean()),
        "vx_mean": float(w["vx"].mean()),
        "track_err_mean": float(err.mean()),
        "p_elec_mean_W": float(w["p_elec"].mean()),
        "p_mech_mean_W": float(w["p_mech"].mean()),
        "p_copper_mean_W": float(w["p_copper"].mean()),
        "energy_J": energy,
        "distance_m": dist,
        "cot": energy / (mass * 9.81 * dist) if dist > 0.1 else float("nan"),
        "energy_per_m_J": energy / dist if dist > 0.1 else float("nan"),
    }
    for g in GROUPS:
        x = w[f"tau_frac_{g}"].to_numpy()
        r[f"tau_frac_{g}_mean"] = float(x.mean())
        r[f"tau_frac_{g}_p95"] = float(np.percentile(x, 95))
    a = w["ankle_rs02_frac_max"].to_numpy() * 100.0
    r["ankle_rs02_pct_rated_mean"] = float(a.mean())
    r["ankle_rs02_pct_rated_p95"] = float(np.percentile(a, 95))
    r["p_ankle_mean_W"] = float(w["p_ankle"].mean())
    r["cu_ankle_mean_W"] = float(w["cu_ankle"].mean())
    r["transmission"] = (
        f"{w['ankle_n_pitch'].iloc[0]:g}/{w['ankle_n_roll'].iloc[0]:g}" if "ankle_n_pitch" in w else "1/1 (old CSV)"
    )
    r["step_freq_hz"] = touch / duration
    r["stride_length_m"] = dist / (touch / 2.0) if touch >= 2 else float("nan")
    r["q_U_mm"] = float(w["q_U"].mean())
    r["q_L_mm"] = float(w["q_L"].mean())
    r["falls"] = int(jumps.sum())
    w = w.assign(
        t=w["t"] - w["t"].iloc[0],
        energy_w=np.cumsum(w["p_elec"]) * dt,
        dist_w=np.r_[0.0, np.cumsum(np.where(jumps, 0.0, step))],
    )
    return r, w


def plot(results, frames, labels, out: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.subplots_adjust(hspace=0.32, wspace=0.25)
    for w, lab in zip(frames, labels):
        n = max(1, int(round(0.5 / float(np.median(np.diff(w["t"]))))))  # 0.5 s rolling mean
        ax[0, 0].plot(w["t"], w["vx"].rolling(n, min_periods=1).mean(), label=lab)
        ax[0, 1].plot(w["t"], w["p_elec"].rolling(n, min_periods=1).mean(), label=lab)
        ax[0, 2].plot(w["dist_w"], w["energy_w"], label=lab)
        ax[1, 2].plot(w["t"], w["q_U"], label=f"{lab} q_U")
        ax[1, 2].plot(w["t"], w["q_L"], "--", label=f"{lab} q_L")
    ax[0, 0].plot(frames[0]["t"], frames[0]["cmd_vx"], "k:", label="cmd vx (first)")
    ax[0, 0].set_title("forward speed v_x [m/s] (0.5 s mean)")
    ax[0, 1].set_title("P_elec [W] (0.5 s mean)")
    ax[0, 2].set_title("cumulative energy [J] vs distance [m]")
    ax[1, 2].set_title("prismatic extension [mm]")
    x = np.arange(len(GROUPS) + 1)
    width = 0.8 / len(results)
    for i, (r, lab) in enumerate(zip(results, labels)):
        vals = [r[f"tau_frac_{g}_mean"] for g in GROUPS] + [r["ankle_rs02_pct_rated_mean"] / 100.0]
        p95 = [r[f"tau_frac_{g}_p95"] for g in GROUPS] + [r["ankle_rs02_pct_rated_p95"] / 100.0]
        ax[1, 0].bar(x + i * width, vals, width, label=lab)
        ax[1, 0].scatter(x + i * width, p95, marker="_", s=200, color="k")
    ax[1, 0].set_xticks(x + 0.4 - width / 2, list(GROUPS) + ["ankle RS02\n/ rated"])
    ax[1, 0].set_title("|tau| / tau_max: mean (bars), p95 (black ticks)")
    keys = [
        ("cot", "CoT"),
        ("energy_per_m_J", "J/m / 100"),
        ("step_freq_hz", "step Hz"),
        ("stride_length_m", "stride m"),
    ]
    for i, (r, lab) in enumerate(zip(results, labels)):
        vals = [r["cot"], r["energy_per_m_J"] / 100.0, r["step_freq_hz"], r["stride_length_m"]]
        ax[1, 1].bar(np.arange(len(keys)) + i * width, vals, width, label=lab)
    ax[1, 1].set_xticks(np.arange(len(keys)) + 0.4 - width / 2, [k[1] for k in keys])
    ax[1, 1].set_title("efficiency and gait")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.savefig(out, dpi=110, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="+")
    p.add_argument("--settle", type=float, default=3.0, help="Seconds discarded at the start of each recording.")
    p.add_argument("--out", type=str, default=None, help="Overlay plot PNG (default: next to the first CSV).")
    p.add_argument("--transmission", choices=["recorded", "direct", "linkage"], default="recorded")
    p.add_argument("--n_pitch", type=float, default=3.0, help="linkage: ankle pitch ratio N_p.")
    p.add_argument("--n_roll", type=float, default=1.0, help="linkage: ankle roll ratio N_r.")
    a = p.parse_args()
    labels = [label_of(c) for c in a.csv]
    tr = {"recorded": None, "direct": (1.0, 1.0), "linkage": (a.n_pitch, a.n_roll)}[a.transmission]
    results, frames = zip(*(analyze(c, a.settle, tr) for c in a.csv))
    table = pd.DataFrame(list(results), index=labels).T
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200, "display.max_columns", 20):
        print(
            f"\n=== {len(a.csv)} recordings, first {a.settle:.1f} s discarded, ankle transmission: {a.transmission}"
            + (f" (N_p {tr[0]:g}, N_r {tr[1]:g})" if tr else "")
        )
        print(table.to_string())
    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(a.csv[0])), "compare_" + "_vs_".join(labels)[:120] + ".png"
    )
    plot(results, frames, labels, out)
    print(f"\nplots: {out}")


if __name__ == "__main__":
    main()
