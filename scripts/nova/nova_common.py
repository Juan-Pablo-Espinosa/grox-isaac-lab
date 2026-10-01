# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the NOVA playback / evaluation scripts (import after the simulation app is launched).

Policy loading mirrors scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py (registry agent cfg -> cli overrides ->
deprecation handling -> checkpoint path -> RslRlVecEnvWrapper -> OnPolicyRunner.load -> inference policy).
"""

from __future__ import annotations

import importlib.metadata as metadata
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

LEGACY_OBS_DIM = 66
"""Policy observation size of runs 1-4 (no morph_locked flag, velocity-mode prismatic action)."""


def resolve_agent_and_checkpoint(task: str, args_cli, cli_args) -> tuple[object, str]:
    """Agent cfg (with CLI overrides) and the checkpoint path, exactly as the official play script resolves them."""
    train_task = task.split(":")[-1].replace("-Play", "")
    agent_cfg = load_cfg_from_registry(train_task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return agent_cfg, resume_path


def checkpoint_obs_dim(path: str) -> int:
    """Input size of the checkpoint's actor MLP (66 = runs 1-4, 67 = morphology-lock flag)."""
    state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    return int(state["mlp.0.weight"].shape[1])


def make_env_and_policy(task: str, env_cfg, agent_cfg, resume_path: str):
    """Build the wrapped env and load the deterministic inference policy (mean actions)."""
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    return env, runner.get_inference_policy(device=env.unwrapped.device)


def pin_commands(env_cfg) -> None:
    """Command cfg for scripted / teleop playback: no resampling, no standing envs, no heading, no time-out."""
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.heading_command = False
    env_cfg.terminations.time_out = None


def route_resample(unwrapped, cmd_vec: torch.Tensor) -> None:
    """Make fall resets re-apply ``cmd_vec`` (num_envs, 3) instead of sampling a random command."""
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    cmd_term._resample_command = lambda env_ids: cmd_term.vel_command_b.__setitem__(env_ids, cmd_vec[env_ids])
