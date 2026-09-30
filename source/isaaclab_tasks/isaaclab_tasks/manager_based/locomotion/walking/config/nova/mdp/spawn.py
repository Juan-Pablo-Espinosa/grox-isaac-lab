# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-level spawn function: contact reporting on ALL of NOVA's (nested) rigid bodies + self-collision filter pairs.

Root cause (documented in the standing task, standing_env_cfg.py): NOVA's USD authors every link as a
USD *child* of its kinematic parent. ``isaaclab.sim.schemas.activate_contact_sensors`` stops descending
at the first prim with ``RigidBodyAPI`` ("nested rigid bodies are not allowed by SDK"), so only
``Hip_Base`` ever receives ``PhysxContactReportAPI`` and ``ContactSensor`` has nothing else to read.

Fix without touching core Isaac Lab, nova.py or the USD asset: the walking task swaps the robot
spawner's ``func`` for this one. It performs the exact same spawn as the stock
``isaaclab.sim.spawners.from_files.spawn_from_usd`` (which is just ``@clone`` around
``_spawn_from_usd_file``) and then applies the same two schemas/attributes that
``activate_contact_sensors`` applies -- but to *every* rigid body in the subtree. It runs on the
prototype prim before ``@clone`` copies it to the other envs, so all envs inherit the schemas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils.prims import clone

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim.spawners.from_files import from_files_cfg


# Non-adjacent body pairs whose convex-hull colliders overlap at the default pose. PhysX already ignores collisions
# between directly jointed links, but these pairs are separated only by a small intermediate link (Hip_Roll /
# Feet_Roll). With articulation self-collisions enabled they were in permanent contact at rest (every env, every step),
# which toppled the robot in a zero-action stand (489 bad_tilt terminations in 5 s x 64 envs) and blocked hip roll.
# Only these pairs are filtered; every other self-collision (incl. all left-right leg contacts) stays active.
NOVA_SELF_COLLISION_FILTER_PAIRS = [
    (f"{a}_{side}", f"{b}_{side}")
    for side in ("Left", "Right")
    for a, b in (("Hip_Pitch", "Upperleg_Yaw"), ("Lowerleg_Prismatic", "Feet_Pitch"))
]


@clone
def spawn_usd_with_nested_contact_reporting(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Same as ``spawn_from_usd``, then tag every nested rigid body for contact reporting."""
    # pxr imported lazily (eager pxr imports before Kit boots crash the app -- see standing mdp/rewards.py)
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    bodies = {}
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[p.GetName()] = p
    # collision filtering for the resting-overlap pairs (see NOVA_SELF_COLLISION_FILTER_PAIRS)
    for a, b in NOVA_SELF_COLLISION_FILTER_PAIRS:
        UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
    for p in Usd.PrimRange(prim):
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        applied = p.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in applied:
            p.AddAppliedSchema("PhysxRigidBodyAPI")
        safe_set_attribute_on_usd_prim(p, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
        if "PhysxContactReportAPI" not in applied:
            p.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(p, "physxContactReport:threshold", 0.0, camel_case=False)
    return prim
