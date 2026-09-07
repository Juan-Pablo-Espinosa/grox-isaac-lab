# Custom startup events for NOVA_LOWERBODY_V2 standing task -- Grade 4, foot-penetration fix.
#
# Root cause (verified quantitatively via scripts/tools/measure_foot_penetration.py):
# the actual foot-sole world height (computed by combining the foot's fixed local
# rest-pose geometry offset with LIVE body_pos_w/body_quat_w, since USD BBoxCache
# reads the static authored transform and never reflects live PhysX state) dips to
# roughly -3mm to -7mm below ground during transient contact loading before
# recovering. This is bounded by NOVA_LOWERBODY_CFG's own
# RigidBodyPropertiesCfg.max_depenetration_velocity=0.1 m/s (nova.py) -- a
# deliberately conservative cap from a prior grade to avoid NaN explosions on
# contact -- which limits how fast the solver may correct any overlap once it
# occurs. That setting must not change (per explicit instruction).
#
# This is a task-level (not robot-level) mitigation: give the GROUND plane a small
# positive rest_offset, so PhysX resolves rest equilibrium with a deliberate small
# air gap rather than zero clearance, absorbing the residual transient overlap that
# the conservative depenetration cap would otherwise leave briefly visible. Verified
# empirically (see standing_env_cfg.py's __post_init__ comment) that this reduces
# measured negative sole-z excursions.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# isaaclab.sim (CollisionBaseCfg / modify_collision_properties) is, like
# isaaclab.assets / isaaclab.envs, a lazy_export package -- an eager module-level
# `from isaaclab.sim import CollisionBaseCfg` forces immediate resolution of
# isaaclab.sim.schemas, which imports raw pxr at ITS OWN module level. Since this
# mdp module gets imported while parsing the env cfg (via hydra_task_config),
# before SimulationApp/Kit has booted, that reproduced the exact same
# "free(): invalid pointer" native crash fixed earlier in mdp/rewards.py and
# mdp/terminations.py (same "Modules [...pxr...] were loaded before SimulationApp"
# warning, verified via a training-run regression). Unlike Articulation/
# ManagerBasedRLEnv in those files, CollisionBaseCfg and modify_collision_properties
# are actually CALLED here, not just used as annotations, so TYPE_CHECKING can't
# defer them -- imported locally inside each function instead, deferring
# resolution to call time (well after Kit has booted, since events only run
# during env construction/reset).


def set_ground_rest_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path: str,
    rest_offset: float,
    contact_offset: float,
) -> None:
    """Apply a small positive rest_offset (+ a larger contact_offset) to the ground
    plane's collider (startup-only).

    PhysX requires rest_offset < contact_offset. modify_collision_properties writes
    the dataclass's fields one at a time, so the plugin logs transient "contact
    offset must be positive and greater then restOffset" / "rest offset must be
    lesser then contact offset" validation warnings at the moment each individual
    field is written (briefly comparing the new value of one field against the
    stale value of the other) -- verified harmless by reading back the final
    authored USD attributes afterward, which land exactly on the requested
    (contact_offset, rest_offset) pair, self-consistent and correctly separated.

    env_ids is unused: the ground plane is a single global prim shared across all
    envs, not a per-env one, so this event's effect is inherently global regardless
    of which env_ids triggered it (mode="startup" invokes it once, unscoped).
    """
    from isaaclab.sim import CollisionBaseCfg
    from isaaclab.sim.schemas import modify_collision_properties

    modify_collision_properties(prim_path, CollisionBaseCfg(contact_offset=contact_offset, rest_offset=rest_offset))


def set_asymmetric_joint_pos_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    limits: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Override joint position limits at runtime (startup-only) with real hardware
    values, without touching the USD asset or re-running convert_urdf_direct.py.

    Verified mechanism: Articulation.write_joint_position_limit_to_sim_index()
    (isaaclab_physx/assets/articulation/articulation.py:1403) exists specifically
    for this -- a runtime override distinct from whatever is baked into the USD.
    ArticulationCfg itself has no direct hard-limit-override field (only
    soft_joint_pos_limit_factor, a global scale on whatever hard limits already
    exist) -- this is the correct, less-invasive mechanism for this Isaac Lab
    version, confirmed by reading the source rather than assumed.

    Args:
        limits: {joint_name: (lower, upper)} in radians (or meters for prismatic).
    """
    asset = env.scene[asset_cfg.name]
    joint_names = list(limits.keys())
    # preserve_order=True: default False may return ids in the articulation's own
    # index order rather than input order, which would silently mismatch against
    # limits_tensor below (built by zipping index i to joint_names[i]).
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    num_envs = asset.num_instances
    limits_tensor = torch.zeros((num_envs, len(joint_ids), 2), device=asset.device)
    for i, name in enumerate(joint_names):
        lower, upper = limits[name]
        limits_tensor[:, i, 0] = lower
        limits_tensor[:, i, 1] = upper
    asset.write_joint_position_limit_to_sim_index(limits=limits_tensor, joint_ids=joint_ids)
