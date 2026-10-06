"""Diagnose JP's observed hip-joint shaking/oscillation during standing, using the
latest post-symmetry-fix checkpoint (model_2999.pt, run 2026-09-07_16-06-09).

Two phases:

PHASE 1 -- control-rate (50 Hz) logging of raw policy action, joint position, and
joint velocity for Hip_Pitch_Left/Right and Hip_Roll_Left/Right during normal
rollout, once the pose has settled. FFT gives an approximate oscillation
frequency/amplitude for each signal; comparing action-signal amplitude to
position/velocity-signal amplitude indicates whether the policy is itself
commanding jitter (case a) or the action is smooth while position/velocity still
rings (pointing at underdamped PD, case b). Frequency content is inherently
limited to below Nyquist = 25 Hz (half the 50 Hz control rate) at this
resolution -- a strong peak near that ceiling is itself suggestive of a
higher-frequency artifact aliasing down, flagged as such rather than resolved.

PHASE 2 -- FREEZE TEST, the clean discriminator. After the pose settles, read
back the actuator's OWN current joint_pos_target for all 12 revolute joints
(reflecting the last raw action actually applied) and hold it fixed -- bypassing
the policy and action manager entirely -- while manually stepping physics at
full substep resolution (every 0.005s sim.dt, not just every 0.02s control
step). If joint position/velocity keeps ringing around a target that is
PROVABLY no longer changing, that is direct, policy-independent evidence of
underdamped PD (case b). If it settles smoothly, PD is not the cause and any
oscillation seen in Phase 1 must trace back to the policy's own action signal
(case a).
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--settle_steps", type=int, default=300)
parser.add_argument("--log_steps", type=int, default=250)
parser.add_argument("--freeze_substeps", type=int, default=400)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

ALL_REVOLUTE = [
    "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
]
TEST_JOINTS = ["Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint", "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint"]

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
all_ids, all_names = robot.find_joints(ALL_REVOLUTE, preserve_order=True)
name_to_slice = {n: i for i, n in enumerate(all_names)}
test_slices = [name_to_slice[n] for n in TEST_JOINTS]

# action-space column for each test joint: ActionsCfg uses the same NOVA_REVOLUTE_JOINTS
# order (12 joints, 1:1 with the action tensor), so action_col == index into ALL_REVOLUTE.
action_cols = test_slices

obs = env.get_observations()

# --- settle into standing pose ---
with torch.inference_mode():
    for step in range(args_cli.settle_steps):
        actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)

# --- PHASE 1: control-rate logging ---
action_hist, pos_hist, vel_hist = [], [], []
with torch.inference_mode():
    for step in range(args_cli.log_steps):
        actions = policy(obs)
        action_hist.append(actions[:, action_cols].clone())
        obs, rew, dones, extras = env.step(actions)
        pos_hist.append(robot.data.joint_pos[:, all_ids][:, test_slices].clone())
        vel_hist.append(robot.data.joint_vel[:, all_ids][:, test_slices].clone())

action_all = torch.stack(action_hist).cpu().numpy()  # (S, E, 4)
pos_all = torch.stack(pos_hist).cpu().numpy()
vel_all = torch.stack(vel_hist).cpu().numpy()

control_dt = env.unwrapped.step_dt  # seconds per control step
control_hz = 1.0 / control_dt
print("=" * 100)
print(f"PHASE 1: control-rate logging ({args_cli.log_steps} steps @ {control_hz:.1f} Hz control rate, {args_cli.num_envs} envs)")
print(f"Nyquist limit at this sampling rate: {control_hz/2:.1f} Hz")
print("=" * 100)


def dominant_freq_amp(signal_1d: np.ndarray, dt: float):
    """FFT-based dominant frequency (excluding DC) and its amplitude."""
    n = len(signal_1d)
    detrended = signal_1d - signal_1d.mean()
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(spectrum) <= 1:
        return 0.0, 0.0
    peak_idx = 1 + np.argmax(spectrum[1:])  # skip DC bin
    amp = 2 * spectrum[peak_idx] / n
    return freqs[peak_idx], amp


near_nyquist_count = 0
total_count = 0
for j, name in enumerate(TEST_JOINTS):
    print(f"\n--- {name} ---")
    for e in range(min(args_cli.num_envs, 4)):  # report first 4 envs
        a_sig = action_all[:, e, j]
        p_sig = pos_all[:, e, j]
        v_sig = vel_all[:, e, j]
        a_freq, a_amp = dominant_freq_amp(a_sig, control_dt)
        p_freq, p_amp = dominant_freq_amp(p_sig, control_dt)
        v_freq, v_amp = dominant_freq_amp(v_sig, control_dt)
        print(
            f"  env{e}: action peak={a_freq:5.2f}Hz amp={a_amp:.5f}  |  "
            f"pos peak={p_freq:5.2f}Hz amp={p_amp:.6f}rad  |  "
            f"vel peak={v_freq:5.2f}Hz amp={v_amp:.5f}rad/s  |  "
            f"pos_std={p_sig.std():.6f}  vel_std={v_sig.std():.5f}"
        )
    # aggregate over ALL envs for this joint (not just the 4 printed above)
    for e in range(args_cli.num_envs):
        a_freq, _ = dominant_freq_amp(action_all[:, e, j], control_dt)
        total_count += 1
        if a_freq >= 20.0:
            near_nyquist_count += 1

print(f"\nAGGREGATE (all {args_cli.num_envs} envs x 4 joints = {total_count} combos):")
print(f"  action peak-frequency >= 20 Hz (near/at Nyquist={control_hz/2:.0f}Hz): {near_nyquist_count}/{total_count} "
      f"({100.0*near_nyquist_count/total_count:.1f}%)")

# --- PHASE 2: freeze test ---
print("\n" + "=" * 100)
print("PHASE 2: FREEZE TEST -- hold current actuator target fixed, step physics at full")
print("substep resolution (bypassing policy/action manager entirely)")
print("=" * 100)

frozen_target = robot.data.joint_pos_target[:, all_ids].clone()  # (E, 12), current actual targets
sim = env.unwrapped.sim
sim_dt = env.unwrapped.physics_dt

freeze_pos_hist, freeze_vel_hist = [], []
with torch.inference_mode():
    for sub in range(args_cli.freeze_substeps):
        robot.set_joint_position_target(frozen_target, joint_ids=all_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        freeze_pos_hist.append(robot.data.joint_pos[:, all_ids][:, test_slices].clone())
        freeze_vel_hist.append(robot.data.joint_vel[:, all_ids][:, test_slices].clone())

freeze_pos_all = torch.stack(freeze_pos_hist).cpu().numpy()  # (substeps, E, 4)
freeze_vel_all = torch.stack(freeze_vel_hist).cpu().numpy()
freeze_target_np = frozen_target[:, test_slices].cpu().numpy()

print(f"\nFrozen for {args_cli.freeze_substeps} substeps @ {1.0/sim_dt:.1f} Hz (sim.dt={sim_dt}s) = {args_cli.freeze_substeps*sim_dt:.2f}s wall time")
for j, name in enumerate(TEST_JOINTS):
    print(f"\n--- {name} (frozen target, per env) ---")
    for e in range(min(args_cli.num_envs, 4)):
        target = freeze_target_np[e, j]
        p_sig = freeze_pos_all[:, e, j]
        v_sig = freeze_vel_all[:, e, j]
        p_freq, p_amp = dominant_freq_amp(p_sig, sim_dt)
        # compare first-half vs second-half position std to check decay (damped) vs sustained (undamped/limit-cycle)
        half = len(p_sig) // 2
        std_first_half = p_sig[:half].std()
        std_second_half = p_sig[half:].std()
        print(
            f"  env{e}: frozen_target={target:+.5f}  final_pos={p_sig[-1]:+.5f}  "
            f"pos_range=[{p_sig.min():+.5f},{p_sig.max():+.5f}]  "
            f"pos_std_1sthalf={std_first_half:.6f} pos_std_2ndhalf={std_second_half:.6f}  "
            f"vel_final={v_sig[-1]:+.5f}  peak_freq={p_freq:.2f}Hz"
        )

env.close()
simulation_app.close()
