# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live teleop dashboard for NOVA (matplotlib, TkAgg) running in its OWN process.

The teleop script starts this file as a subprocess (:class:`DashboardClient`) and streams JSON lines over its stdin
from a background thread fed by a bounded queue: the simulation never waits on the plot (frames are dropped when the
queue is full). Messages: a header {"names": [...], "dt": s, "window": s}, then {"t": s, "row": [...]} samples
(~10 Hz), {"cmd": "reset"} (zero the energy / CoT integrators), {"cmd": "rec", "on": bool} and
{"cmd": "save", "path": png}. Without a display the window is not shown, but "save" still renders a PNG.

Panels (rolling window, default 20 s): commanded vs actual vx / vy / wz; P_elec stacked by motor group with total
mechanical vs copper; cumulative energy [J] and running CoT; |tau| / tau_max per joint group and ankle RS02 torque /
rated; prismatic q_U, q_L, targets and goals [mm]; foot contact timeline.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

GROUPS = ("hip_knee", "roll", "yaw", "ankle", "leadscrew")
TAU_GROUPS = ("hip_knee", "roll", "yaw", "ankle")


class DashboardClient:
    """Sim-side handle: starts the dashboard process and feeds it without ever blocking the caller."""

    def __init__(self, names: list[str], dt: float, window: float = 20.0, max_queue: int = 64):
        self.names = names
        self.dropped = 0
        self.sent = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)], stdin=subprocess.PIPE, text=True, bufsize=1 << 16
        )
        self.proc.stdin.write(json.dumps({"names": names, "dt": dt, "window": window}) + "\n")
        self.proc.stdin.flush()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self):
        while True:
            msg = self._q.get()
            if msg is None:
                return
            try:
                self.proc.stdin.write(msg + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                return  # window closed

    def _put(self, msg: dict):
        try:
            self._q.put_nowait(json.dumps(msg))
            self.sent += 1
        except queue.Full:
            self.dropped += 1

    def push(self, t: float, row: list[float]):
        self._put({"t": t, "row": row})

    def command(self, cmd: str, **kw):
        self._put({"cmd": cmd, **kw})

    def close(self, timeout: float = 5.0):
        try:
            self._q.put(None, timeout=1.0)
            self._thread.join(timeout=timeout)
            self.proc.stdin.close()
            self.proc.wait(timeout=timeout)
        except Exception:
            self.proc.kill()


def main():
    import collections
    import contextlib
    import math

    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    header = json.loads(sys.stdin.readline())
    names = header["names"]
    col = {n: i for i, n in enumerate(names)}
    window = header.get("window", 20.0)
    maxlen = int(window / max(header["dt"], 1e-3)) + 10
    buf = collections.deque(maxlen=maxlen)  # (t, row, energy, cot)
    inbox: collections.deque = collections.deque()
    state = {"energy": 0.0, "dist": 0.0, "last_t": None, "rec": False, "eof": False, "pending_save": []}

    def reader():
        for line in sys.stdin:
            inbox.append(json.loads(line))
        state["eof"] = True

    threading.Thread(target=reader, daemon=True).start()

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.22, left=0.06, right=0.94, top=0.94, bottom=0.06)
    with contextlib.suppress(Exception):  # not every backend has a window manager
        fig.canvas.manager.set_window_title("NOVA teleop dashboard")
    (ax_v, ax_p), (ax_e, ax_tau), (ax_q, ax_c) = axes
    ax_cot = ax_e.twinx()

    def get(rows, name):
        i = col[name]
        return [r[i] for r in rows]

    def drain():
        while inbox:
            m = inbox.popleft()
            if "cmd" in m:
                if m["cmd"] == "reset":
                    state["energy"], state["dist"] = 0.0, 0.0
                elif m["cmd"] == "rec":
                    state["rec"] = m["on"]
                    if m["on"]:
                        state["energy"], state["dist"] = 0.0, 0.0
                elif m["cmd"] == "save":
                    state["pending_save"].append(m["path"])
                continue
            t, row = m["t"], m["row"]
            if state["last_t"] is not None and t > state["last_t"]:
                h = t - state["last_t"]
                state["energy"] += row[col["p_elec"]] * h
                state["dist"] += math.hypot(row[col["vx"]], row[col["vy"]]) * h
            state["last_t"] = t
            cot = state["energy"] / (row[col["mass"]] * 9.81 * state["dist"]) if state["dist"] > 0.1 else float("nan")
            buf.append((t, row, state["energy"], cot))

    def draw(_frame=None):
        drain()
        if not buf:
            return
        ts = [b[0] for b in buf]
        rows = [b[1] for b in buf]
        for ax in (ax_v, ax_p, ax_e, ax_cot, ax_tau, ax_q, ax_c):
            ax.cla()
        for name, c in (("vx", "C0"), ("vy", "C1"), ("wz", "C2")):
            ax_v.plot(ts, get(rows, "cmd_" + name), "--", color=c, lw=1)
            ax_v.plot(ts, get(rows, name), color=c, lw=1.4, label=f"{name} (dashed: cmd)")
        ax_v.set_title("velocity: commanded (dashed) vs actual [m/s, rad/s]", fontsize=9)
        ax_v.legend(fontsize=7, loc="upper left")
        ax_p.stackplot(ts, *[get(rows, "p_" + g) for g in GROUPS], labels=GROUPS, alpha=0.75)
        ax_p.plot(ts, get(rows, "p_mech"), "k-", lw=1, label="mechanical")
        ax_p.plot(ts, get(rows, "p_copper"), "k:", lw=1.2, label="copper")
        ax_p.set_title(f"P_elec by group (now {rows[-1][col['p_elec']]:.0f} W) [W]", fontsize=9)
        ax_p.legend(fontsize=7, loc="upper left", ncol=4)
        ax_e.plot(ts, [b[2] for b in buf], "C3", lw=1.4)
        ax_e.set_ylabel("energy [J]", color="C3", fontsize=8)
        ax_cot.plot(ts, [b[3] for b in buf], "C4", lw=1.4)
        ax_cot.set_ylabel("running CoT", color="C4", fontsize=8)
        cot_now = buf[-1][3]
        ax_e.set_title(
            f"cumulative energy {buf[-1][2]:.0f} J | running CoT "
            f"{'n/a' if math.isnan(cot_now) else f'{cot_now:.2f}'}"
            f"{' | REC' if state['rec'] else ''}",
            fontsize=9,
        )
        for g in TAU_GROUPS:
            ax_tau.plot(ts, get(rows, "tau_frac_" + g), lw=1.2, label=g)
        ax_tau.plot(ts, get(rows, "ankle_rs02_frac_max"), "k-", lw=1.2, label="ankle RS02 / 6 N·m (max motor)")
        ax_tau.axhline(1.0, color="r", lw=0.8, ls=":")
        ax_tau.set_title("|tau| / tau_max per group, ankle RS02 torque / rated", fontsize=9)
        ax_tau.legend(fontsize=7, loc="upper left", ncol=3)
        for name, c in (("q_U", "C0"), ("q_L", "C1")):
            ax_q.plot(ts, get(rows, name), color=c, lw=1.6, label=name)
            ax_q.plot(ts, get(rows, name + "_target"), "--", color=c, lw=1, label=name + " target")
            ax_q.plot(ts, get(rows, name + "_goal"), ":", color=c, lw=1, label=name + " goal")
        ax_q.set_ylim(0, 100)
        ax_q.set_title("prismatic extension [mm]", fontsize=9)
        ax_q.legend(fontsize=7, loc="upper left", ncol=3)
        ax_c.fill_between(ts, 1.1, [1.1 + 0.8 * x for x in get(rows, "contact_L")], step="post", color="C0")
        ax_c.fill_between(ts, 0.1, [0.1 + 0.8 * x for x in get(rows, "contact_R")], step="post", color="C1")
        ax_c.set_yticks([0.5, 1.5], ["R", "L"])
        ax_c.set_ylim(0, 2)
        ax_c.set_title("foot contact", fontsize=9)
        for ax in (ax_q, ax_c):
            ax.set_xlabel("t [s]", fontsize=8)
        ax_v.set_xlim(max(ts[0], ts[-1] - window), ts[-1] + 1e-3)
        for ax in (ax_v, ax_p, ax_e, ax_tau, ax_q, ax_c):
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        while state["pending_save"]:
            fig.savefig(state["pending_save"].pop(0), dpi=110)

    if matplotlib.get_backend().lower() == "agg":
        import time

        while not (state["eof"] and not inbox):
            draw()
            time.sleep(0.2)
        return
    anim = FuncAnimation(fig, draw, interval=150, cache_frame_data=False)  # noqa: F841 (keep a reference)

    def poll_eof():
        if state["eof"] and not inbox:
            draw()
            plt.close(fig)
        else:
            fig.canvas.get_tk_widget().after(500, poll_eof)

    fig.canvas.get_tk_widget().after(500, poll_eof)
    plt.show()


if __name__ == "__main__":
    main()
