# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right mirror symmetry of NOVA's observations and actions, for rsl_rl symmetry augmentation.

Mirror = reflection through the robot's sagittal (x-z) plane, y -> -y:
  * polar (linear) vectors  (x, y, z) -> ( x, -y,  z): base_lin_vel, projected_gravity, imu_lin_acc
  * axial (angular) vectors (x, y, z) -> (-x,  y, -z): base_ang_vel, imu_ang_vel
  * velocity command (vx, vy, wz) -> (vx, -vy, -wz)
  * scalar flags (morph_locked) are mirror-invariant
  * joints: Left <-> Right swap with a per-pair sign, R = s * L, measured by FK (pose of every right-leg body vs the
    mirrored left-leg body; wrong sign costs 16-1005 mm / 24-183 deg of error):
        Hip_Pitch -1, Hip_Roll -1, Upperleg_Yaw -1, Lowerleg_Pitch -1, Feet_Roll -1,
        Upperleg_Prismatic +1, Lowerleg_Prismatic +1, Feet_Pitch +1
    The default pose satisfies q_R = s * q_L (to 3e-4 rad), so joint_pos_rel, joint_vel and the actions (offsets /
    velocity commands relative to the defaults) all mirror with the same swap + sign.

The observation layout is not hard-coded: index maps are built once from the live observation / action managers
(term order and sizes) and the articulation joint order, and cached on the env.
This module must stay Kit-free (it is imported by the agent cfg before the simulator starts).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["MIRROR_SIGN", "compute_symmetric_states", "mirror_actions", "mirror_policy_obs"]

MIRROR_SIGN = {
    "Hip_Pitch": -1.0,
    "Hip_Roll": -1.0,
    "Upperleg_Yaw": -1.0,
    "Upperleg_Prismatic": 1.0,
    "Lowerleg_Pitch": -1.0,
    "Lowerleg_Prismatic": 1.0,
    "Feet_Roll": -1.0,
    "Feet_Pitch": 1.0,
}
_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
_VECTOR_TERMS = {
    "base_lin_vel": _POLAR,
    "projected_gravity": _POLAR,
    "imu_lin_acc": _POLAR,
    "base_ang_vel": _AXIAL,
    "imu_ang_vel": _AXIAL,
    "velocity_commands": (1.0, -1.0, -1.0),
}
_INVARIANT_TERMS = {"morph_locked"}


def _joint_map(names: list[str]) -> tuple[list[int], list[float]]:
    """Permutation (index of the mirrored joint) and sign for a list of joint names."""
    perm, sign = [], []
    for n in names:
        m = n.replace("_Left_", "_TMP_").replace("_Right_", "_Left_").replace("_TMP_", "_Right_")
        perm.append(names.index(m))
        sign.append(MIRROR_SIGN[n.replace("_Left_Joint", "").replace("_Right_Joint", "")])
    return perm, sign


def _maps(env) -> dict[str, torch.Tensor]:
    """Build (and cache) the flat permutation / sign vectors for the policy observation and the action."""
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    if getattr(u, "_nova_mirror_maps", None) is not None:
        return u._nova_mirror_maps
    joint_names = list(u.scene["robot"].joint_names)
    act_names = []
    for term_name in u.action_manager.active_terms:
        act_names += list(u.action_manager.get_term(term_name)._joint_names)
    om = u.observation_manager
    obs_perm, obs_sign, off = [], [], 0
    for name, dims in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
        size = int(torch.tensor(dims).prod())
        if name in _VECTOR_TERMS:
            assert size == 3, f"{name}: expected 3 dims, got {size}"
            obs_perm += [off, off + 1, off + 2]
            obs_sign += list(_VECTOR_TERMS[name])
        elif name in _INVARIANT_TERMS:
            obs_perm += list(range(off, off + size))
            obs_sign += [1.0] * size
        elif name in ("joint_pos", "joint_vel"):
            assert size == len(joint_names), f"{name}: {size} != {len(joint_names)} joints"
            p, s = _joint_map(joint_names)
            obs_perm += [off + i for i in p]
            obs_sign += s
        elif name == "actions":
            assert size == len(act_names), f"actions: {size} != {len(act_names)}"
            p, s = _joint_map(act_names)
            obs_perm += [off + i for i in p]
            obs_sign += s
        else:
            raise ValueError(f"No mirror rule for observation term '{name}'")
        off += size
    ap, asg = _joint_map(act_names)
    dev = u.device
    u._nova_mirror_maps = {
        "obs_perm": torch.tensor(obs_perm, device=dev),
        "obs_sign": torch.tensor(obs_sign, device=dev),
        "act_perm": torch.tensor(ap, device=dev),
        "act_sign": torch.tensor(asg, device=dev),
    }
    return u._nova_mirror_maps


def mirror_policy_obs(env, obs: torch.Tensor) -> torch.Tensor:
    """Mirror a (batch, 67) policy observation tensor."""
    m = _maps(env)
    return obs[:, m["obs_perm"]] * m["obs_sign"]


def mirror_actions(env, actions: torch.Tensor) -> torch.Tensor:
    """Mirror a (batch, 16) action tensor (12 revolute in action-term order, then U_L, U_R, L_L, L_R)."""
    m = _maps(env)
    return actions[:, m["act_perm"]] * m["act_sign"]


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl_rl ``data_augmentation_func``: returns [original; left-right mirror] stacked along the batch."""
    obs_aug = None
    if obs is not None:
        b = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            obs_aug[key][:b] = obs[key]
            obs_aug[key][b:] = mirror_policy_obs(env, obs[key])
    act_aug = None
    if actions is not None:
        b = actions.shape[0]
        act_aug = torch.cat([actions, mirror_actions(env, actions)], dim=0)
    return obs_aug, act_aug
