# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX contact sensor implementation that supports NOVA's NESTED rigid bodies (task-level, core Isaac Lab untouched).

Stock ``isaaclab_physx.sensors.ContactSensor._initialize_impl`` finds the contact-reporting bodies with a
recursive walk (so it does find all 17 NOVA bodies once mdp/spawn.py has tagged them), but then builds the
PhysX view from ``f"{body_parent}/({name1}|{name2}|...)"`` -- which only matches DIRECT children of the
robot prim. NOVA authors every link as a USD child of its parent (Robot/Geometry/Hip_Base/Hip_Pitch_Left/...),
so the view matches nothing and env construction fails (``body_physx_view`` is None).

Measured PhysX view semantics (probe, 3 envs):
  * a list of 17 per-body globs ``env_*/Robot/<nested path>`` -> 51 bodies but ordered BODY-major
    (env_0:Hip_Base, env_1:Hip_Base, ...). The sensor kernels index ``env * num_bodies + body`` (env-major),
    so this would silently mix up envs/bodies.
  * ``**`` globs / alternation with "/" do not match nested paths.
  * an explicit per-env list ``env_{e}/Robot/<nested path>`` in env-major order -> correct env-major view.

This subclass therefore replaces only the view construction with the explicit env-major list, verifies
the ordering at init, and otherwise reuses the stock buffers/kernels. Filtered contacts are not supported.

This module imports Kit-dependent packages (omni.physics via isaaclab_physx.physics); it is only loaded lazily
through the string ``class_type`` in :mod:`.contact_sensor_cfg`, because task configs are imported before Kit starts.
"""

from __future__ import annotations

import re

from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor

from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source


class NestedBodyContactSensor(ContactSensor):
    """``ContactSensor`` whose PhysX view is built from explicit, env-major paths of nested bodies."""

    def _initialize_impl(self):
        # base-class setup (sim handles, env count, buffers bookkeeping) -- skip ContactSensor's flat-glob views
        super(ContactSensor, self)._initialize_impl()
        if self.cfg.filter_prim_paths_expr or self.cfg.track_contact_points or self.cfg.track_friction_forces:
            raise ValueError("NestedBodyContactSensor supports net contact forces only (no filters / contact points).")
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            return bool(name_pattern.fullmatch(prim.GetName())) and "PhysxContactReportAPI" in prim.GetAppliedSchemas()

        matches = resolve_matching_prims_from_source(parent_expr)
        if not matches:
            raise RuntimeError(f"No prim found at '{parent_expr}'.")
        asset_prim, body_parent = matches[0]
        walk_root = asset_prim.GetPath().pathString
        prims = get_all_matching_child_prims(walk_root, predicate=has_contact_report, traverse_instance_prims=False)
        rel_paths = [p.GetPath().pathString[len(walk_root) + 1 :] for p in prims]
        if not rel_paths:
            raise RuntimeError(f"Sensor at '{self.cfg.prim_path}' found no bodies with PhysxContactReportAPI.")
        # body_parent comes back in glob form ("/World/envs/env_*/Robot"); accept the regex form too
        parent = body_parent.replace("env_.*", "env_*")
        if parent.count("env_*") != 1 or "*" in parent.replace("env_*", ""):
            raise RuntimeError(f"Unsupported sensor parent expression '{body_parent}' (expected exactly one 'env_*').")

        # explicit env-major list: env_0/<all bodies>, env_1/<all bodies>, ...
        patterns = [f"{parent.replace('env_*', f'env_{e}')}/{rel}" for e in range(self._num_envs) for rel in rel_paths]
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(patterns)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            patterns, max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(rel_paths) * self._num_envs
        )
        if self._body_physx_view is None or self._contact_view is None:
            raise RuntimeError("Failed to create PhysX views for the nested-body contact sensor.")
        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != len(rel_paths):
            raise RuntimeError(f"Contact view found {self._num_sensors} bodies per env, expected {len(rel_paths)}.")
        # verify env-major ordering (the stock kernels and body_names rely on it)
        got = self.body_physx_view.prim_paths
        for e in (0, self._num_envs - 1):
            block = got[e * self._num_sensors : (e + 1) * self._num_sensors]
            if [p.rsplit("/", 1)[-1] for p in block] != [r.rsplit("/", 1)[-1] for r in rel_paths] or any(
                f"/env_{e}/" not in p for p in block
            ):
                raise RuntimeError(f"Contact view is not env-major at env {e}: {block[:3]}...")
        self._create_buffers()
