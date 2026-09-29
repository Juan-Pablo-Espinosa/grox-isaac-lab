# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-level spawn function that enables contact reporting on ALL of NOVA's (nested) rigid bodies.

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
