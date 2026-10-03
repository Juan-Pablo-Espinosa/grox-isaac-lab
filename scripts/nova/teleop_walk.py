# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive teleop playback for the NOVA walking policies, with a live dashboard and CSV recording.

Drive a trained policy with the keyboard: set the velocity command (vx, vy, yaw rate) live, change the leg length,
watch the robot in the Newton viewer and the signals in a separate dashboard window, and record runs for comparison
(scripts/nova/compare_recordings.py).

Launch (newest run, latest checkpoint by default)::

    # morphology-agnostic task (leg length driven by the keyboard / the training schedule, not by the policy)
    ./isaaclab.sh -p scripts/nova/teleop_walk.py --task Isaac-Walking-Nova-MorphAgnostic-Play-v0 --visualizer newton
    #   [--length U,L]  start leg length [m] (default 0.05,0.05)    [--schedule]  start with the random schedule on
    # walking task (runs 1-5: the policy controls leg length unless MANUAL)
    ./isaaclab.sh -p scripts/nova/teleop_walk.py --task Isaac-Walking-Nova-Play-v0 --visualizer newton
    #   [--locked U | U,L | random]   (run 5+) spawn morphology-LOCKED envs   [--length U,L]  start in MANUAL
    # common: [--checkpoint /path/model_N.pt | --load_run <run_dir> [--checkpoint model_N.pt]] [--num_envs N]
    #         [--no_dashboard] [--dashboard_snapshot PNG]

Checkpoints: the interface is detected from the checkpoint (actor input / output size) and must match the task:
morphology-agnostic = 66-D obs / 12-D action; walking run 5+ = 67 / 16; walking runs 1-4 = 66 / 16 (the env is
rebuilt with walking_env_cfg.make_run4_compatible). Mismatches are refused with a message.

Input: keys pressed in the NEWTON VIEWER window (the viewer's own key events / key-down state, so holding a key
works) and, as a secondary channel, keys typed in the launching TERMINAL (raw tty, tap-to-increment). No root, no
global X11 hook. The viewer already uses W/A/S/D, arrows, Q/E (camera), H (UI), SPACE (pause), . (step), F (frame),
ESC (close), so in the viewer use X for "stop" and G for help; in the terminal SPACE and H also work.

Keys:
    I / K   target vx +/- 0.1 m/s        J / L   target vy +/- 0.1 m/s     U / O   target yaw rate +/- 0.25 rad/s
    X       all commands to 0 (terminal: also SPACE)
    R       reset robot                  P       random lateral push (~0.5 m/s root velocity kick)
    [ / ]   leg length -/+ 10 mm, upper AND lower       - / =   upper only       { / }   lower only (Shift+[ / ])
    T       (agnostic) toggle the training random leg-length schedule (off at start unless --schedule)
    M       (walking task) toggle MANUAL leg length (the keys above only act in MANUAL)
    V       start / stop recording (CSV in <run dir>/recordings/, path printed when stopped)
    C       toggle camera follow         G       help (terminal: also H)
Holding I/K/J/L/U/O in the viewer keeps changing the target at 1 m/s^2 (yaw 2.5 rad/s^2); the applied command always
ramps toward the target at that rate. Targets are clamped to the env's command ranges (read from the task cfg).
Leg length always moves at <= 0.035 m/s (leadscrew rate limit). Status line:
    vx <cmd>><actual> vy wz | P_elec | U <actual>><goal> L <actual>><goal> mm | leg mode | REC | falls
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# reuse the official rsl_rl CLI helpers (--checkpoint / --load_run / ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Keyboard teleop for the NOVA walking policies.")
parser.add_argument("--task", type=str, default="Isaac-Walking-Nova-MorphAgnostic-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--selftest",
    nargs="?",
    const="basic",
    choices=["basic", "record"],
    default=None,
    help="Scripted key-injection test and exit: 'basic' (commands, legs, reset) or 'record' (two recordings).",
)
parser.add_argument("--record_seconds", type=float, default=8.0, help="selftest 'record': length of each recording.")
parser.add_argument("--record_lengths", type=str, default="0.005,0.065", help="selftest 'record': two leg lengths.")
parser.add_argument("--record_vx", type=float, default=0.5, help="selftest 'record': forward command [m/s].")
parser.add_argument(
    "--locked",
    type=str,
    default=None,
    help="(walking run 5+) spawn morphology-LOCKED envs: 'U' or 'U,L' [m], or 'random' (training reset sample).",
)
parser.add_argument("--length", type=str, default=None, help="Start leg length 'U,L' (or 'U') [m].")
parser.add_argument("--schedule", action="store_true", help="(agnostic) start with the random leg schedule on.")
parser.add_argument("--no_dashboard", action="store_true", help="Do not open the live dashboard window.")
parser.add_argument("--dashboard_snapshot", type=str, default=None, help="Save the dashboard as this PNG on exit.")
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
import nova_signals  # noqa: E402
import torch  # noqa: E402
from dashboard import DashboardClient  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.actions import PrismaticLengthAction  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.power import PRISMATIC_DRIVER  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.walking.config.nova.walking_env_cfg import (  # noqa: E402
    make_run4_compatible,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# command clamps, filled from the env's command cfg ranges in main()
RANGES: dict[str, tuple[float, float]] = {}
STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.25}
RATE = {"vx": 1.0, "vy": 1.0, "wz": 2.5}  # ramp [unit/s]: 1 m/s^2 linear, 2.5 rad/s^2 yaw
AXIS_KEYS = {"i": ("vx", +1), "k": ("vx", -1), "j": ("vy", +1), "l": ("vy", -1), "u": ("wz", +1), "o": ("wz", -1)}
LEG_KEYS = {
    "[": (-1, "both"),
    "]": (1, "both"),
    "-": (-1, "upper"),
    "=": (1, "upper"),
    "{": (-1, "lower"),
    "}": (1, "lower"),
}
HOLD_DELAY = 0.35  # s before a held viewer key starts auto-ramping the target
LEG_MIN, LEG_MAX, LEG_STEP = 0.005, 0.095, 0.01
DASHBOARD_HZ = 10.0


class Teleop:
    """Command / mode state machine fed by key events from the viewer and the terminal (and the self-tests)."""

    def __init__(self, agnostic: bool):
        self.agnostic = agnostic
        self.target = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.manual = False  # walking task: MANUAL leg length
        self.schedule = False  # agnostic task: random leg-length schedule
        self.leg_u, self.leg_l = 0.05, 0.05
        self.follow = True
        self.request_reset = False
        self.request_push = False
        self.request_record = False
        self.schedule_changed = False
        self.messages: list[str] = []
        self.lock = threading.Lock()

    def _set_target(self, axis: str, value: float):
        lo, hi = RANGES[axis]
        if value < lo - 1e-9 or value > hi + 1e-9:
            self.messages.append(f"[teleop] {axis} target {value:+.2f} outside training range [{lo}, {hi}] -> clamped")
        self.target[axis] = round(min(max(value, lo), hi), 4)

    def _leg_key(self, k: str):
        if not self.agnostic and not self.manual:
            self.messages.append("[teleop] leg keys only act in MANUAL leg-length mode (press M)")
            return
        sgn, which = LEG_KEYS[k]

        def clamp(x):
            return round(min(max(x + sgn * LEG_STEP, LEG_MIN), LEG_MAX), 4)

        if which in ("both", "upper"):
            self.leg_u = clamp(self.leg_u)
        if which in ("both", "lower"):
            self.leg_l = clamp(self.leg_l)
        note = " (schedule is ON: overridden at its next draw, press T)" if self.agnostic and self.schedule else ""
        self.messages.append(f"[teleop] leg goal U {self.leg_u * 1000:.0f} mm, L {self.leg_l * 1000:.0f} mm{note}")

    def key(self, k: str):
        """Handle one key press (lower-case name; '{' / '}' for Shift+[ / ])."""
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
                if self.agnostic:
                    self.messages.append("[teleop] leg length is always external in this task (T: random schedule)")
                else:
                    self.manual = not self.manual
                    self.messages.append(f"[teleop] leg length {'MANUAL' if self.manual else 'AUTO (policy)'}")
            elif k == "t":
                if not self.agnostic:
                    self.messages.append("[teleop] T (random leg schedule) only exists in the morph-agnostic task")
                else:
                    self.schedule = not self.schedule
                    self.schedule_changed = True
                    self.messages.append(f"[teleop] random leg-length schedule {'ON' if self.schedule else 'OFF'}")
            elif k in LEG_KEYS:
                self._leg_key(k)
            elif k == "v":
                self.request_record = True
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
        self.names = {getattr(self.k, n.upper()): n for n in "ikjluoxrpmcgtv"}
        self.names[self.k.MINUS] = "-"
        self.names[self.k.EQUAL] = "="
        self.names[self.k.BRACELEFT] = "{"
        self.names[self.k.BRACERIGHT] = "}"
        self.viewer.renderer.register_key_press(self._on_press)

    def _on_press(self, symbol, modifiers):
        shift = bool(modifiers & self.k.MOD_SHIFT)
        if symbol in (self.k.BRACKETLEFT, self.k.BRACKETRIGHT):
            name = ("{" if shift else "[") if symbol == self.k.BRACKETLEFT else ("}" if shift else "]")
        else:
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


_STALE = object()  # forces LegControl.pre_step to push its state


class LegControl:
    """Uniform access to the leg-length source: the external driver (agnostic) or the prismatic action term."""

    def __init__(self, u, teleop: Teleop):
        self.u, self.teleop = u, teleop
        self.agnostic = teleop.agnostic
        self.n, self.dev = u.num_envs, u.device
        if self.agnostic:
            self.driver = u.prismatic_driver
            self.rate_source = PRISMATIC_DRIVER
            self.reset_cfg = u.event_manager.get_term_cfg("reset_nova")
        else:
            self.term = u.action_manager.get_term("prismatic")
            self.rate_source = "prismatic"
            self.length_mode = isinstance(self.term, PrismaticLengthAction)
            self.locked = getattr(self.term, "locked", torch.zeros(self.n, dtype=torch.bool, device=self.dev))
        self._applied = _STALE

    def target(self) -> torch.Tensor:
        return self.driver.target if self.agnostic else self.term.processed_actions

    def rate(self) -> torch.Tensor:
        return self.driver.target_rate if self.agnostic else self.term.target_rate

    def goal(self) -> torch.Tensor:
        if self.agnostic:
            return self.driver.goal
        if self.teleop.manual:
            return torch.tensor([[self.teleop.leg_u, self.teleop.leg_l]], device=self.dev).expand(self.n, 2)
        return self.term.processed_actions[:, [0, 2]]

    def pre_step(self):
        """Agnostic: push the teleop goal / schedule state into the driver and the respawn lengths."""
        if not self.agnostic:
            return
        t = self.teleop
        if t.schedule_changed:
            t.schedule_changed = False
            self.driver.schedule_enabled = t.schedule
            if not t.schedule:  # continue from where the schedule left the legs
                t.leg_u, t.leg_l = (round(x, 4) for x in self.driver.goal[0].tolist())
            self._applied = _STALE
        want = None if t.schedule else (t.leg_u, t.leg_l)
        if want != self._applied:
            if want is not None:
                self.driver.set_goal(want[0], want[1])
            # respawn after a fall at the teleop length (schedule off) or at a random length (schedule on)
            self.reset_cfg.params["prismatic_per_env"] = None if want is None else [want] * self.n
            self._applied = want

    def override_actions(self, actions: torch.Tensor, dt: float):
        """Walking task MANUAL: drive the 4 prismatic action entries toward the teleop leg length."""
        if self.agnostic or not self.teleop.manual:
            return
        legs = torch.tensor([self.teleop.leg_u] * 2 + [self.teleop.leg_l] * 2, device=self.dev)
        pc = self.term.cfg
        if self.length_mode:
            c, h = 0.5 * (pc.q_min + pc.q_max), 0.5 * (pc.q_max - pc.q_min)
            actions[:, -4:] = (legs - c) / h
        else:
            actions[:, -4:] = ((legs - self.term.processed_actions) / (pc.max_velocity * dt)).clamp(-1.0, 1.0)

    def mode(self) -> str:
        if self.agnostic:
            return "sched ON" if self.teleop.schedule else "sched off"
        if bool(self.locked[0]):
            return "LOCKED"
        return "MANUAL" if self.teleop.manual else "AUTO"


def parse_lengths(text: str | None, flag: str) -> tuple[float, float] | None:
    """'U' or 'U,L' [m] -> (U, L)."""
    if text is None:
        return None
    vals = [float(x) for x in text.split(",")]
    if len(vals) not in (1, 2) or not all(LEG_MIN - 1e-9 <= x <= LEG_MAX + 1e-9 for x in vals):
        raise SystemExit(f"{flag} expects U or U,L in [{LEG_MIN}, {LEG_MAX}] m; got {text!r}")
    return vals[0], vals[-1]


def configure(env_cfg, resume_path: str):
    """Match the env cfg to the task + checkpoint interface; returns (agnostic, description)."""
    obs_dim = nova_common.checkpoint_obs_dim(resume_path)
    act_dim = nova_common.checkpoint_action_dim(resume_path)
    agnostic = hasattr(env_cfg, "prismatic_driver")
    if agnostic:
        if act_dim != 12:
            raise SystemExit(
                f"[teleop] {args_cli.task} needs a morphology-agnostic policy (66-D obs, 12-D action); {resume_path} "
                f"has {obs_dim}-D obs / {act_dim}-D action "
                "(a walking-task policy: use --task Isaac-Walking-Nova-Play-v0)."
            )
        if args_cli.locked is not None:
            raise SystemExit("[teleop] --locked is a walking-task option; here use --length U,L (and T / --schedule).")
        length = parse_lengths(args_cli.length, "--length") or (0.05, 0.05)
        env_cfg.prismatic_driver.schedule_enabled = args_cli.schedule
        if not args_cli.schedule:
            env_cfg.events.reset_nova.params["prismatic_per_env"] = [length] * env_cfg.scene.num_envs
        return True, length, "morphology-agnostic (66-D obs, 12-D action, external leg driver)"
    if act_dim != 16:
        raise SystemExit(
            f"[teleop] {args_cli.task} needs a walking policy (16-D action); {resume_path} has a {act_dim}-D action "
            "(a morphology-agnostic policy: use --task Isaac-Walking-Nova-MorphAgnostic-Play-v0)."
        )
    locked = args_cli.locked
    if obs_dim == nova_common.LEGACY_OBS_DIM:
        if locked is not None:
            raise SystemExit(
                "[teleop] --locked needs a run-5+ checkpoint (67-D observation with the morph_locked flag); "
                f"{resume_path} is a runs 1-4 policy (66-D, velocity-mode leg action) that never saw a locked "
                "morphology."
            )
        make_run4_compatible(env_cfg)
        desc = "walking runs 1-4 (66-D obs, velocity-mode leg action)"
    else:
        env_cfg.actions.prismatic.lock_fraction = 1.0 if locked is not None else 0.0
        if locked not in (None, "random"):
            u_l = parse_lengths(locked, "--locked")
            env_cfg.events.reset_nova.params["prismatic_range"] = (u_l[0], u_l[0])
            env_cfg.events.reset_nova.params["lower_prismatic_range"] = (u_l[1], u_l[1])
        desc = "walking run 5+ (67-D obs, absolute-length leg action)" + (f", LOCKED {locked}" if locked else "")
    return False, parse_lengths(args_cli.length, "--length"), desc


def main():
    # ---- policy checkpoint (same resolution as scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py) and interface
    agent_cfg, resume_path = nova_common.resolve_agent_and_checkpoint(args_cli.task, args_cli, cli_args)
    # ---- env cfg (teleop: no resampling, no standing envs, no time-out; falls still reset)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    nova_common.pin_commands(env_cfg)
    agnostic, length, desc = configure(env_cfg, resume_path)
    print(f"[teleop] {desc}")
    r = env_cfg.commands.base_velocity.ranges
    RANGES.update(vx=tuple(r.lin_vel_x), vy=tuple(r.lin_vel_y), wz=tuple(r.ang_vel_z))
    print(f"[teleop] command clamps (env cfg): {RANGES}")
    env, policy = nova_common.make_env_and_policy(args_cli.task, env_cfg, agent_cfg, resume_path)

    u = env.unwrapped
    dev, n = u.device, u.num_envs
    robot = u.scene["robot"]
    teleop = Teleop(agnostic)
    teleop.schedule = agnostic and args_cli.schedule
    if length is not None:
        teleop.leg_u, teleop.leg_l = length
        teleop.manual = not agnostic
    legs = LegControl(u, teleop)
    cmd_term = u.command_manager.get_term("base_velocity")
    cmd_vec = torch.zeros(n, 3, device=dev)

    def write_command():
        cmd_vec[:] = torch.tensor([teleop.cmd["vx"], teleop.cmd["vy"], teleop.cmd["wz"]], device=dev)
        cmd_term.vel_command_b[:] = cmd_vec

    # a fall-reset resamples the command; route that to the teleop command instead of a random one
    nova_common.route_resample(u, cmd_vec)

    viewer_keys = ViewerKeys(u.sim, teleop)
    term_keys = TerminalKeys(teleop) if args_cli.selftest is None else None
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

    dt = u.step_dt
    obs = env.get_observations()
    signals = nova_signals.SignalExtractor(u, legs.rate_source, legs.target, legs.goal)
    recorder = nova_signals.Recorder(signals.names, os.path.join(os.path.dirname(resume_path), "recordings"), dt)
    dash = None
    if not args_cli.no_dashboard:
        dash = DashboardClient([signals.names[i] for i in signals.dashboard_idx], dt * round(1 / (DASHBOARD_HZ * dt)))
        print(f"[teleop] dashboard: separate process (pid {dash.proc.pid}), matplotlib, ~{DASHBOARD_HZ:.0f} Hz")
    dash_idx = torch.tensor(signals.dashboard_idx, device=dev)
    dash_every = max(1, round(1.0 / (DASHBOARD_HZ * dt)))
    falls, last_status, step = 0, 0.0, 0
    recordings: list[str] = []
    script = None
    if args_cli.selftest == "basic":
        script = SelfTest(teleop, viewer_keys)
    elif args_cli.selftest == "record":
        script = RecordTest(teleop, args_cli)
    wall0 = time.monotonic()
    try:
        while simulation_app.is_running() and not stop["flag"]:
            t0 = time.monotonic()
            viewer_keys.poll_held(dt)
            if script is not None and script.update(step * dt, robot, legs, u):
                break
            teleop.ramp(dt)
            legs.pre_step()
            if teleop.request_reset:
                teleop.request_reset = False
                write_command()
                obs, _ = env.reset()
                if dash is not None:
                    dash.command("reset")
            if teleop.request_push:
                teleop.request_push = False
                yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)[2]
                side = torch.where(torch.rand(n, device=dev) < 0.5, -0.5, 0.5)
                vel = robot.data.root_vel_w.torch.clone()
                vel[:, 0] += -torch.sin(yaw) * side
                vel[:, 1] += torch.cos(yaw) * side
                robot.write_root_velocity_to_sim_index(root_velocity=vel)
                teleop.messages.append("[teleop] push!")
            if teleop.request_record:
                teleop.request_record = False
                if recorder.active:
                    path = recorder.stop()
                    recordings.append(path)
                    teleop.messages.append(f"[teleop] recording saved: {path}" if path else "[teleop] empty recording")
                else:
                    q = robot.data.joint_pos.torch[0, signals.pris_ids] * 1000
                    c = teleop.cmd
                    recorder.start(
                        f"vx{c['vx']:+.2f}_vy{c['vy']:+.2f}_wz{c['wz']:+.2f}"
                        f"_U{q[0:2].mean():03.0f}_L{q[2:4].mean():03.0f}"
                    )
                    teleop.messages.append("[teleop] recording ... (V to stop)")
                if dash is not None:
                    dash.command("rec", on=recorder.active)
            write_command()
            with torch.inference_mode():
                actions = policy(obs)
                legs.override_actions(actions, dt)
                obs, _, dones, extras = env.step(actions)
                time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
                fell = bool((dones.bool() & ~time_outs)[0])
                falls += int((dones.bool() & ~time_outs).sum())  # time_out is disabled, so every done is a fall
                if fell and dash is not None:
                    dash.command("reset")
                need_row = recorder.active or (dash is not None and step % dash_every == 0)
                if need_row:
                    row = signals.row(0)
                    recorder.add(row)
                    if dash is not None and step % dash_every == 0:
                        dash.push(round(step * dt, 4), row[dash_idx].tolist())
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
                q = robot.data.joint_pos.torch[0, signals.pris_ids] * 1000
                g = legs.goal()[0] * 1000
                p = nova_signals.power_model(u, legs.rate_source).compute(u)[0][0].sum()
                c = teleop.cmd
                rec = f" | REC {recorder.elapsed():.1f}s" if recorder.active else ""
                sys.stdout.write(
                    f"\r\033[Kvx {c['vx']:+.2f}>{v[0]:+.2f} vy {c['vy']:+.2f}>{v[1]:+.2f} wz {c['wz']:+.2f}>{w:+.2f}"
                    f" | {p:4.0f} W | U {q[0:2].mean():2.0f}>{g[0]:2.0f} L {q[2:4].mean():2.0f}>{g[1]:2.0f} mm"
                    f" | {legs.mode()}{rec} | falls {falls}"
                )
                sys.stdout.flush()
            # real-time pacing (the viewer would otherwise run the robot faster than wall clock)
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0 and script is None:
                time.sleep(sleep)
    finally:
        wall = time.monotonic() - wall0
        if recorder.active:
            recordings.append(recorder.stop())
        recordings = [r for r in recordings if r]
        if term_keys is not None:
            term_keys.restore()
        sys.stdout.write("\n")
        print(
            f"[teleop] {step} control steps in {wall:.1f} s wall = {step / max(wall, 1e-9):.1f} steps/s "
            f"({'with' if dash is not None else 'without'} dashboard)"
        )
        for path in recordings:
            print(f"[teleop] recording: {path}")
        if dash is not None:
            if args_cli.dashboard_snapshot:
                dash.command("save", path=os.path.abspath(args_cli.dashboard_snapshot))
            dash.close()
            print(f"[teleop] dashboard messages sent {dash.sent}, dropped {dash.dropped}")
            if args_cli.dashboard_snapshot:
                print(f"[teleop] dashboard snapshot: {os.path.abspath(args_cli.dashboard_snapshot)}")
        if script is not None:
            script.report()
        sys.stdout.flush()
        env.close()


class SelfTest:
    """Scripted key injection (same Teleop.key() path the keyboard uses) with measurements."""

    def __init__(self, teleop: Teleop, viewer_keys: ViewerKeys | None = None):
        self.t = teleop
        self.vk = viewer_keys
        self.done_keys = 0
        self.log = {"fwd": [], "yaw": [], "stop": [], "leg": [], "leg_rate": [], "tgt_rate": [], "reset": None}
        self.yaw_total = 0.0
        self.prev_q = None
        self.yaw0 = None

    def update(self, t, robot, legs: LegControl, u):
        d = robot.data
        v = d.root_lin_vel_b.torch[0]
        w = d.root_ang_vel_b.torch[0, 2].item()
        yaw = math_utils.euler_xyz_from_quat(d.root_quat_w.torch[:1])[2].item()
        q = d.joint_pos.torch[0, legs.driver.joint_ids if legs.agnostic else legs.term.joint_ids].clone()
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
            if hasattr(self, "leg_start"):  # this window spans several steps: act once
                self.prev_q = q
                return False
            if not self.t.manual and not legs.agnostic:
                self.t.key("m")
            self.leg_start = q.tolist()
            # drive every leg from wherever it is to 0.05 m via the leg keys: down to the stop, then up 4.5 steps
            for _ in range(10):
                self.t.key("[")
            for _ in range(4):
                self.t.key("]")
            self.t.leg_u = self.t.leg_l = 0.05  # (0.005 + 4 x 0.01 = 0.045 -> set the last 5 mm directly)
        elif t < 24.0:
            if self.prev_q is not None:
                self.log["leg_rate"].append(((q - self.prev_q).abs().max() / u.step_dt).item())
            # the source's own rate (zeroed on reset, so a fall-reset re-anchoring is not counted as motion)
            self.log["tgt_rate"].append(legs.rate()[0].abs().max().item())
            self.log["leg"] = q.tolist()
        elif t < 24.1:
            self.t.key("r")
        elif t < 24.5:
            self.log["reset"] = (d.root_pos_w.torch[0, :2] - u.scene.env_origins[0, :2]).tolist(), q.tolist()
        elif legs.agnostic and t < 36.0:  # random schedule ON for ~11.5 s (T), then OFF
            goal = [round(x * 1000) for x in legs.goal()[0].tolist()]
            if not self.t.schedule:
                self.t.key("t")
                self.log["sched"] = [goal]
            elif goal != self.log["sched"][-1]:
                self.log["sched"].append(goal)
            if t > 25.0 and "sched_state_on" not in self.log:
                self.log["sched_state_on"] = (legs.driver.schedule_enabled, legs.reset_cfg.params["prismatic_per_env"])
        elif legs.agnostic and t < 37.0:
            if self.t.schedule:
                self.t.key("t")
                self.log["sched_off_goal"] = [round(x * 1000) for x in legs.goal()[0].tolist()]
            self.log["sched_off_end"] = [round(x * 1000) for x in legs.goal()[0].tolist()]
            pe = legs.reset_cfg.params["prismatic_per_env"]
            self.log["sched_state_off"] = (legs.driver.schedule_enabled, pe[0] if pe else None)
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
                f"leg goal 0.05: start (U_L,U_R,L_L,L_R) {[round(x * 1000, 1) for x in self.leg_start]} mm -> "
                f"end {[round(x * 1000, 1) for x in L['leg']]} mm; TARGET rate max {max(L['tgt_rate']):.4f} m/s; "
                f"joint rate max {max(L['leg_rate']):.4f} / p50 {st.median(L['leg_rate']):.4f} m/s"
            )
        if L["reset"]:
            print(
                f"reset: root xy offset from env origin {[round(x, 3) for x in L['reset'][0]]} m, "
                f"prismatics {[round(x * 1000, 1) for x in L['reset'][1]]} mm"
            )
        if L.get("sched"):
            print(
                f"schedule ON (T) 11.5 s: env-0 goal sequence (U, L mm) {L['sched']} ({len(L['sched']) - 1} draws); "
                f"T again -> OFF, goal held {L['sched_off_goal']} -> {L['sched_off_end']} after 1 s"
            )
            on, off = L.get("sched_state_on"), L.get("sched_state_off")
            print(
                f"  driver.schedule_enabled ON={on[0]} (respawn {'random' if on[1] is None else on[1][0]}) / "
                f"OFF={off[0]} (respawn at {off[1]}); every fall restarts that env's U(2, 8) s timer"
            )


class RecordTest:
    """Scripted recording session via the keyboard path: walk at vx, record at leg length A, then at length B."""

    def __init__(self, teleop: Teleop, args):
        self.t = teleop
        self.vx = args.record_vx
        self.lengths = [float(x) for x in args.record_lengths.split(",")]
        self.rec = args.record_seconds
        self.plan = []  # (time, action)
        t = 0.5
        self.plan.append((t, "start"))
        for i, length in enumerate(self.lengths):
            self.plan.append((t, ("legs", length)))
            # worst-case leg travel (90 mm at 35 mm/s) + 3 s to settle (+ 2 s for the command ramp the first time)
            t += 0.09 / 0.035 + 3.0 + (2.0 if i == 0 else 0.0)
            self.plan.append((t, "v"))
            t += self.rec
            self.plan.append((t, "v"))
            t += 0.1
        self.end = t + 0.2
        self.i = 0

    def _legs(self, length: float):
        if not self.t.agnostic and not self.t.manual:
            self.t.key("m")
        steps = round((length - self.t.leg_u) / LEG_STEP)
        for _ in range(abs(steps)):
            self.t.key("]" if steps > 0 else "[")
        self.t.leg_u = self.t.leg_l = length  # exact value (key steps are 10 mm)

    def update(self, t, robot, legs, u):
        while self.i < len(self.plan) and t >= self.plan[self.i][0]:
            action = self.plan[self.i][1]
            if action == "start":
                for _ in range(round(self.vx / STEP["vx"])):
                    self.t.key("i")
            elif action == "v":
                self.t.key("v")
            else:
                self._legs(action[1])
            self.i += 1
        return t >= self.end

    def report(self):
        print(f"\n=== RECORD TEST: vx {self.vx}, lengths {self.lengths}, {self.rec} s each (paths above)")


if __name__ == "__main__":
    main()
    simulation_app.close()
