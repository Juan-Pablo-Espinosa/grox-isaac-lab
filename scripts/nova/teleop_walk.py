# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive teleop playback for the NOVA walking policy.

Drive a trained Isaac-Walking-Nova policy with the keyboard: set the velocity command (vx, vy, yaw rate) live and
watch it in the Newton viewer. Optionally take over the four prismatic (leg-length) outputs manually.

Launch (newest run, latest checkpoint by default)::

    ./isaaclab.sh -p scripts/nova/teleop_walk.py --task Isaac-Walking-Nova-Play-v0 --visualizer newton
    #   [--checkpoint /path/model_N.pt | --load_run <run_dir_name> [--checkpoint model_N.pt]] [--num_envs N]
    #   [--locked U | U,L | random]   spawn morphology-LOCKED envs (leg lengths fixed per episode) at U (upper and
    #                                 lower) or U,L [m] in [0.005, 0.095], or at the training reset sample

Checkpoints: run 5+ (67-D obs, absolute-length prismatic action) and runs 1-4 (66-D obs, velocity-mode action) are
both loaded -- the interface is detected from the checkpoint's actor input size and the env is rebuilt to match
(walking_env_cfg.make_run4_compatible). --locked needs a run 5+ checkpoint (older policies never saw the lock).

Input: keys pressed in the NEWTON VIEWER window (read through the viewer's own key events / key-down state, so
holding a key works) and, as a secondary channel, keys typed in the launching TERMINAL (raw tty, tap-to-increment).
No root, no global X11 hook. The viewer already uses W/A/S/D, arrows, Q/E (camera), H (UI), SPACE (pause),
. (step), F (frame), ESC (close), so in the viewer use X for "stop" and G for help; in the terminal SPACE and H also
work.

Keys:
    I / K   target vx +/- 0.1 m/s        J / L   target vy +/- 0.1 m/s     U / O   target yaw rate +/- 0.25 rad/s
    X       all commands to 0 (terminal: also SPACE)
    R       reset robot                  P       random lateral push (~0.5 m/s root velocity kick)
    M       toggle MANUAL leg length     [ / ]   manual target leg length -/+ 0.01 m (upper and lower together)
    C       toggle camera follow         G       help (terminal: also H)
Holding I/K/J/L/U/O in the viewer keeps changing the target at 1 m/s^2 (yaw 2.5 rad/s^2); the applied command always
ramps toward the target at that rate. Targets are clamped to the env's command ranges (read from the task cfg).
MANUAL leg length overrides the 4 prismatic actions of FREE envs; LOCKED envs keep their lengths.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# reuse the official rsl_rl CLI helpers (--checkpoint / --load_run / ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Keyboard teleop for the NOVA walking policy.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--selftest", action="store_true", help="Run the scripted key-injection test and exit.")
parser.add_argument(
    "--locked",
    type=str,
    default=None,
    help="Spawn morphology-LOCKED envs: 'U' or 'U,L' leg-length targets [m], or 'random' (training reset sample).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import atexit  # noqa: E402
import math  # noqa: E402
import select  # noqa: E402
import signal  # noqa: E402
import termios  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tty  # noqa: E402

import nova_common  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.actions import PrismaticLengthAction  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.walking_env_cfg import (  # noqa: E402
    make_run4_compatible,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# command clamps, filled from the env's command cfg ranges in main()
RANGES: dict[str, tuple[float, float]] = {}
STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.25}
RATE = {"vx": 1.0, "vy": 1.0, "wz": 2.5}  # ramp [unit/s]: 1 m/s^2 linear, 2.5 rad/s^2 yaw
AXIS_KEYS = {"i": ("vx", +1), "k": ("vx", -1), "j": ("vy", +1), "l": ("vy", -1), "u": ("wz", +1), "o": ("wz", -1)}
HOLD_DELAY = 0.35  # s before a held viewer key starts auto-ramping the target
LEG_MIN, LEG_MAX, LEG_STEP = 0.005, 0.095, 0.01


class Teleop:
    """Command / mode state machine fed by key events from the viewer and the terminal (and the self-test)."""

    def __init__(self):
        self.target = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.manual = False
        self.leg = 0.05
        self.follow = True
        self.request_reset = False
        self.request_push = False
        self.messages: list[str] = []
        self.lock = threading.Lock()

    def _set_target(self, axis: str, value: float):
        lo, hi = RANGES[axis]
        if value < lo - 1e-9 or value > hi + 1e-9:
            self.messages.append(f"[teleop] {axis} target {value:+.2f} outside training range [{lo}, {hi}] -> clamped")
        self.target[axis] = round(min(max(value, lo), hi), 4)

    def key(self, k: str):
        """Handle one key press (lower-case name)."""
        with self.lock:
            if k in AXIS_KEYS:
                axis, sgn = AXIS_KEYS[k]
                self._set_target(axis, self.target[axis] + sgn * STEP[axis])
            elif k in ("x", "space"):
                self.target = {a: 0.0 for a in self.target}
            elif k == "r":
                self.request_reset = True
            elif k == "p":
                self.request_push = True
            elif k == "m":
                self.manual = not self.manual
                self.messages.append(f"[teleop] leg length {'MANUAL' if self.manual else 'AUTO (policy)'}")
            elif k in ("[", "]"):
                if not self.manual:
                    self.messages.append("[teleop] [ / ] only act in MANUAL leg-length mode (press M)")
                else:
                    self.leg = round(min(max(self.leg + (LEG_STEP if k == "]" else -LEG_STEP), LEG_MIN), LEG_MAX), 4)
                    self.messages.append(f"[teleop] manual leg-length target {self.leg * 1000:.0f} mm")
            elif k == "c":
                self.follow = not self.follow
                self.messages.append(f"[teleop] camera follow {'ON' if self.follow else 'OFF'}")
            elif k in ("g", "h"):
                self.messages.append(__doc__.split("Keys:")[1].split("Holding")[0].rstrip())

    def hold(self, axis: str, sgn: int, dt: float):
        """Continuous target change while a viewer key is held."""
        with self.lock:
            self._set_target(axis, self.target[axis] + sgn * RATE[axis] * dt)

    def ramp(self, dt: float):
        with self.lock:
            for a in self.cmd:
                d = self.target[a] - self.cmd[a]
                self.cmd[a] += max(-RATE[a] * dt, min(RATE[a] * dt, d))


class TerminalKeys:
    """Background reader of raw key presses from the launching terminal (cbreak mode, restored on exit)."""

    def __init__(self, teleop: Teleop):
        self.teleop = teleop
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old = None
        self.stop = threading.Event()
        if self.fd is None:
            return
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        atexit.register(self.restore)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if not r:
                continue
            ch = os.read(self.fd, 1).decode(errors="ignore")
            if ch == " ":
                self.teleop.key("space")
            elif ch:
                self.teleop.key(ch.lower())

    def restore(self):
        self.stop.set()
        if self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            self.old = None


class ViewerKeys:
    """Key events + held-key state from the Newton viewer (newton.viewer.ViewerGL), if one is active."""

    def __init__(self, sim, teleop: Teleop):
        self.teleop = teleop
        self.viz, self.viewer = None, None
        for v in getattr(sim, "_visualizers", []):
            if type(v).__name__ == "NewtonVisualizer" and getattr(v, "_viewer", None) is not None:
                self.viz, self.viewer = v, v._viewer
        self.pressed_at: dict[str, float] = {}
        if self.viewer is None:
            return
        import pyglet

        self.k = pyglet.window.key
        self.names = {getattr(self.k, n.upper()): n for n in "ikjluoxrpmcg"}
        self.names[self.k.BRACKETLEFT] = "["
        self.names[self.k.BRACKETRIGHT] = "]"
        self.viewer.renderer.register_key_press(self._on_press)

    def _on_press(self, symbol, modifiers):
        name = self.names.get(symbol)
        if name is not None and not self.viewer._ui_is_capturing_keyboard():
            self.pressed_at[name] = time.monotonic()
            self.teleop.key(name)

    def poll_held(self, dt: float):
        if self.viewer is None:
            return
        now = time.monotonic()
        for name, (axis, sgn) in AXIS_KEYS.items():
            if self.viewer.is_key_down(name):
                if now - self.pressed_at.get(name, now) > HOLD_DELAY:
                    self.teleop.hold(axis, sgn, dt)
            else:
                self.pressed_at.pop(name, None)

    def follow(self, robot, env_origin):
        if self.viz is None:
            return
        p = (robot.data.root_pos_w.torch[0] - env_origin).tolist()
        target = (p[0], p[1], 0.6)
        eye = (p[0] - 2.2, p[1] - 2.2, 1.6)
        self.viz._apply_camera_pose((eye, target))


def parse_locked(text: str | None) -> tuple[float, float] | str | None:
    """--locked value -> (upper, lower) [m], 'random', or None."""
    if text is None or text == "random":
        return text
    vals = [float(x) for x in text.split(",")]
    if len(vals) not in (1, 2) or not all(LEG_MIN - 1e-9 <= x <= LEG_MAX + 1e-9 for x in vals):
        raise SystemExit(f"--locked expects U or U,L in [{LEG_MIN}, {LEG_MAX}] m, or 'random'; got {text!r}")
    return vals[0], vals[-1]


def main():
    locked = parse_locked(args_cli.locked)
    # ---- policy checkpoint (same resolution as scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py) and interface
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    legacy = nova_common.checkpoint_obs_dim(resume_path) == nova_common.LEGACY_OBS_DIM
    if legacy and locked is not None:
        raise SystemExit(
            "[teleop] --locked needs a run-5+ checkpoint (67-D observation with the morph_locked flag); "
            f"{resume_path} is a runs 1-4 policy (66-D, velocity-mode leg action) that never saw a locked morphology."
        )

    # ---- env cfg (teleop: no resampling, no standing envs, no time-out; falls still reset)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    nova_common.pin_commands(env_cfg)
    if legacy:
        make_run4_compatible(env_cfg)
        print("[teleop] runs 1-4 checkpoint (66-D obs): velocity-mode leg action, no morphology lock")
    else:
        env_cfg.actions.prismatic.lock_fraction = 1.0 if locked is not None else 0.0
        if isinstance(locked, tuple):
            env_cfg.events.reset_nova.params["prismatic_range"] = (locked[0], locked[0])
            env_cfg.events.reset_nova.params["lower_prismatic_range"] = (locked[1], locked[1])
        if locked is not None:
            print(
                f"[teleop] morphology LOCKED: {locked if isinstance(locked, str) else f'U {locked[0]} L {locked[1]} m'}"
            )
    r = env_cfg.commands.base_velocity.ranges
    RANGES.update(vx=tuple(r.lin_vel_x), vy=tuple(r.lin_vel_y), wz=tuple(r.ang_vel_z))
    print(f"[teleop] command clamps (env cfg): {RANGES}")
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)

    u = env.unwrapped
    dev, n = u.device, u.num_envs
    robot, contact = u.scene["robot"], u.scene["contact_forces"]
    cmd_term = u.command_manager.get_term("base_velocity")
    prismatic = u.action_manager.get_term("prismatic")
    length_mode = isinstance(prismatic, PrismaticLengthAction)
    lock_flags = getattr(prismatic, "locked", torch.zeros(n, dtype=torch.bool, device=dev))
    foot_ids = [contact.body_names.index(b) for b in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
    teleop = Teleop()
    cmd_vec = torch.zeros(n, 3, device=dev)

    def write_command():
        cmd_vec[:] = torch.tensor([teleop.cmd["vx"], teleop.cmd["vy"], teleop.cmd["wz"]], device=dev)
        cmd_term.vel_command_b[:] = cmd_vec

    # a fall-reset resamples the command; route that to the teleop command instead of a random one
    nova_common.route_resample(u, cmd_vec)

    viewer_keys = ViewerKeys(u.sim, teleop)
    term_keys = TerminalKeys(teleop) if not args_cli.selftest else None
    print(
        "[teleop] input:",
        "Newton viewer keys" if viewer_keys.viewer is not None else "no viewer",
        "+ terminal keys" if term_keys is not None and term_keys.fd is not None else "",
    )
    teleop.key("g")

    stop = {"flag": False}

    def _sigint(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    obs = env.get_observations()
    dt = u.step_dt
    falls, last_status, step = 0, 0.0, 0
    script = SelfTest(teleop, viewer_keys) if args_cli.selftest else None
    try:
        while simulation_app.is_running() and not stop["flag"]:
            t0 = time.monotonic()
            viewer_keys.poll_held(dt)
            if script is not None and script.update(step * dt, robot, prismatic, u):
                break
            teleop.ramp(dt)
            if teleop.request_reset:
                teleop.request_reset = False
                write_command()
                obs, _ = env.reset()
            if teleop.request_push:
                teleop.request_push = False
                yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)[2]
                side = torch.where(torch.rand(n, device=dev) < 0.5, -0.5, 0.5)
                vel = robot.data.root_vel_w.torch.clone()
                vel[:, 0] += -torch.sin(yaw) * side
                vel[:, 1] += torch.cos(yaw) * side
                robot.write_root_velocity_to_sim_index(root_velocity=vel)
                teleop.messages.append("[teleop] push!")
            write_command()
            with torch.inference_mode():
                actions = policy(obs)
                if teleop.manual:
                    pc = prismatic.cfg
                    if length_mode:
                        # absolute-length action: a = (l - c) / h; the term rate-limits the target to 0.035 m/s
                        c, h = 0.5 * (pc.q_min + pc.q_max), 0.5 * (pc.q_max - pc.q_min)
                        actions[:, -4:] = (teleop.leg - c) / h
                    else:
                        # velocity-mode action (runs 1-4): drive the integrated target toward the manual length
                        err = teleop.leg - prismatic.processed_actions
                        actions[:, -4:] = (err / (pc.max_velocity * dt)).clamp(-1.0, 1.0)
            obs, _, dones, extras = env.step(actions)
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~time_outs).sum())  # time_out is disabled, so every done is a fall
            if viewer_keys.viewer is not None and teleop.follow:
                viewer_keys.follow(robot, u.scene.env_origins[0])
            step += 1
            # ---- status line (~5 Hz) and messages
            for m in teleop.messages:
                sys.stdout.write("\r\033[K" + m + "\n")
            teleop.messages.clear()
            if time.monotonic() - last_status > 0.2:
                last_status = time.monotonic()
                v = robot.data.root_lin_vel_b.torch[0]
                w = robot.data.root_ang_vel_b.torch[0, 2]
                q = robot.data.joint_pos.torch[0, prismatic.joint_ids] * 1000
                inc = contact.data.current_contact_time.torch[0, foot_ids] > 0
                mode = "AUTO"
                if lock_flags[0]:
                    mode = "LOCKED"
                elif teleop.manual:
                    mode = f"MANUAL {teleop.leg * 1000:.0f}mm"
                sys.stdout.write(
                    f"\r\033[Kcmd ({teleop.cmd['vx']:+.2f},{teleop.cmd['vy']:+.2f},{teleop.cmd['wz']:+.2f}) "
                    f"tgt ({teleop.target['vx']:+.2f},{teleop.target['vy']:+.2f},{teleop.target['wz']:+.2f}) | "
                    f"body ({v[0]:+.2f},{v[1]:+.2f},{w:+.2f}) | "
                    f"leg mm U {q[0]:.0f}/{q[1]:.0f} L {q[2]:.0f}/{q[3]:.0f} | "
                    f"feet {'L' if inc[0] else '-'}{'R' if inc[1] else '-'} | "
                    f"{mode} | falls {falls}"
                )
                sys.stdout.flush()
            # real-time pacing (the viewer would otherwise run the robot faster than wall clock)
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0 and script is None:
                time.sleep(sleep)
    finally:
        if term_keys is not None:
            term_keys.restore()
        sys.stdout.write("\n")
        if script is not None:
            script.report()
        env.close()


class SelfTest:
    """Scripted key injection (same Teleop.key() path the keyboard uses) with measurements."""

    def __init__(self, teleop: Teleop, viewer_keys: ViewerKeys | None = None):
        self.t = teleop
        self.vk = viewer_keys
        self.events = [(0.5, "i")] * 5  # five taps -> vx target 0.5
        self.done_keys = 0
        self.log = {"fwd": [], "yaw": [], "stop": [], "leg": [], "leg_rate": [], "tgt_rate": [], "reset": None}
        self.prev_t = None
        self.yaw_total = 0.0
        self.phase = "fwd"
        self.prev_q = None
        self.yaw0 = None

    def update(self, t, robot, prismatic, u):
        d = robot.data
        v = d.root_lin_vel_b.torch[0]
        w = d.root_ang_vel_b.torch[0, 2].item()
        yaw = math_utils.euler_xyz_from_quat(d.root_quat_w.torch[:1])[2].item()
        q = d.joint_pos.torch[0, prismatic.joint_ids].clone()
        if t < 0.6:
            if self.done_keys < 5:
                if self.done_keys == 0 and self.vk is not None and self.vk.viewer is not None:
                    # first tap goes through the Newton viewer's registered key-press callback path
                    before = self.t.target["vx"]
                    self.vk._on_press(self.vk.k.I, 0)
                    print(f"\n[selftest] viewer key path: I -> vx target {before:+.2f} -> {self.t.target['vx']:+.2f}")
                else:
                    self.t.key("i")
                self.done_keys += 1
        elif t < 8.0:  # forward 0.5 for ~5 s after the ramp (ramp takes 0.5 s)
            if t > 2.5:
                self.log["fwd"].append((v[0].item(), v[1].item(), self.t.cmd["vx"]))
        elif t < 8.1:
            self.t.key("x")
            for _ in range(4):
                self.t.key("u")  # yaw target 1.0 rad/s, vx 0
            self.yaw0 = yaw
        elif t < 14.0:
            if t > 9.5:
                self.log["yaw"].append((w, self.t.cmd["wz"]))
            self.yaw_total += math_utils.wrap_to_pi(torch.tensor(yaw - self.yaw0)).item()  # unwrapped heading change
            self.yaw0 = yaw
        elif t < 14.1:
            self.t.key("x")
        elif t < 17.0:
            if t > 16.0:
                self.log["stop"].append((v[0].item(), v[1].item(), w))
        elif t < 17.1:
            self.t.key("m")
            self.leg_start = q.tolist()
            self.t.leg = 0.05
        elif t < 24.0:
            if self.prev_q is not None:
                self.log["leg_rate"].append(((q - self.prev_q).abs().max() / u.step_dt).item())
            # the term's own rate (zeroed on reset, so a fall-reset re-anchoring is not counted as motion)
            self.log["tgt_rate"].append(prismatic.target_rate[0].abs().max().item())
            self.log["leg"] = q.tolist()
        elif t < 24.1:
            self.t.key("r")
        elif t < 24.5:
            self.log["reset"] = (d.root_pos_w.torch[0, :2] - u.scene.env_origins[0, :2]).tolist(), q.tolist()
        else:
            return True
        self.prev_q = q
        return False

    def report(self):
        import statistics as st

        L = self.log
        fx = [a for a, _, _ in L["fwd"]]
        fy = [b for _, b, _ in L["fwd"]]
        print("\n=== SELFTEST")
        if fx:
            print(
                f"forward cmd 0.5: body vx mean {st.mean(fx):+.3f} (min {min(fx):+.3f}, max {max(fx):+.3f}), "
                f"|vx-0.5| mean {st.mean(abs(x - 0.5) for x in fx):.3f}, max {max(abs(x - 0.5) for x in fx):.3f}; "
                f"vy mean {st.mean(fy):+.3f}  ({len(fx)} samples over {len(fx) * 0.02:.1f} s)"
            )
        if L["yaw"]:
            ws = [a for a, _ in L["yaw"]]
            print(
                f"yaw cmd 1.0: body wz mean {st.mean(ws):+.3f} (min {min(ws):+.3f}); heading change "
                f"{math.degrees(self.yaw_total):+.0f} deg over ~6 s (unwrapped, incl. the 1 s ramp)"
            )
        if L["stop"]:
            print(
                f"stop: last 1 s body |vx| mean {st.mean(abs(a) for a, _, _ in L['stop']):.3f}, |vy| "
                f"{st.mean(abs(b) for _, b, _ in L['stop']):.3f}, |wz| {st.mean(abs(c) for _, _, c in L['stop']):.3f}"
            )
        if L["leg_rate"]:
            print(
                f"manual leg 0.05: start (U_L,U_R,L_L,L_R) {[round(x * 1000, 1) for x in self.leg_start]} mm -> "
                f"end {[round(x * 1000, 1) for x in L['leg']]} mm; TARGET rate max {max(L['tgt_rate']):.4f} m/s; "
                f"joint rate max {max(L['leg_rate']):.4f} / p50 {st.median(L['leg_rate']):.4f} m/s"
            )
            moving = [r for r in L["tgt_rate"] if r > 0.03]
            print(
                f"   steps with the target moving: {len(moving)} -> {len(moving) * 0.02:.2f} s "
                f"(45 mm at 35 mm/s needs ~1.29 s)"
            )
        if L["reset"]:
            print(
                f"reset: root xy offset from env origin {[round(x, 3) for x in L['reset'][0]]} m, "
                f"prismatics {[round(x * 1000, 1) for x in L['reset'][1]]} mm"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
