# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a NOVA showcase recording (scripts/nova/showcase_record.py) in Blender with EEVEE.

Run with Blender's bundled Python (not Isaac Lab's)::

    blender --background --factory-startup --python scripts/nova/showcase_blender.py -- \\
        --npz ~/nova_showcase/data/hero.npz --mode hero --out ~/nova_showcase/frames/hero [--frames 800:1300]
    ... --mode lineup --npz .../lineup.npz --out .../frames/lineup
    ... --mode align --npz .../snap.npz --camera .../snap.json --frames 900:900 --out .../align

Every link's STL is imported with its URDF <visual><origin> applied and parented to one empty per body; each empty is
keyframed every frame from the recorded world poses (frame index = recording step, 50 fps). Meshes above 200k faces
are decimated. Studio: white cyclorama bowl, soft area key / fill + rim light on a rig that follows the robot(s).
Camera paths are computed here from the recording (zero-phase critically damped smoothing of the root position).
Besides the PNG frames, <out>/screen.json holds per frame the image-space position of every robot (for HUD labels).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

URDF = os.path.expanduser("~/Downloads/NOVA_LOWERBODY_V2.1/urdf/NOVA_LOWERBODY_V2.1.urdf")
MAX_FACES = 200_000
TARGET_FACES = 150_000
FPS = 50
HERO_DIST = 3.55
"""Hero camera distance [m]: 50 mm lens (vertical FOV covers 0.405 * d) -> a ~1.0 m robot fills ~70% of the frame."""
HERO_LOOK_Z = 0.50
"""Fixed hero look-at height [m] (robot spans ~0 .. 1.0 m)."""
HIP_SHORT, HIP_LONG = 0.80, 0.98
"""Standing hip height [m] at (5, 5) and (95, 95) mm (measured in the lineup recording: 0.802 / 0.980)."""

# material per link (prefix match): structure aluminium, motor blocks dark anodized, telescoping segments crimson
LINK_MATERIAL = {
    "Hip_Base": "aluminium",
    "Hip_Pitch": "anodized",
    "Hip_Roll": "anodized",
    "Upperleg_Yaw": "aluminium",
    "Upperleg_Prismatic": "crimson",
    "Lowerleg_Pitch": "aluminium",
    "Lowerleg_Prismatic": "crimson",
    "Feet_Roll": "anodized",
    "Feet_Pitch": "aluminium",
}


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_linear(h: str) -> tuple[float, float, float, float]:
    return tuple(srgb_to_linear(int(h[i : i + 2], 16) / 255.0) for i in (1, 3, 5)) + (1.0,)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--mode", choices=("hero", "lineup", "align"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", default=None, help="first:last (inclusive) recording frames to render")
    p.add_argument("--camera", default=None, help="Newton viewer camera json (align mode)")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--percent", type=int, default=100, help="resolution percentage (quick tests)")
    p.add_argument("--blend", default=None, help="also save the scene as .blend")
    p.add_argument(
        "--height_ref",
        choices=("ruler", "lines", "none"),
        default="lines",
        help="hero height reference: vertical hip-height scale beside the robot, or lines at the shortest / longest"
        " hip heights behind it",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------- scene building


def set_if(obj, attr, value):
    if hasattr(obj, attr):
        with contextlib.suppress(TypeError, ValueError):
            setattr(obj, attr, value)


def make_materials() -> dict[str, bpy.types.Material]:
    def principled(name, color, metallic, roughness, aniso=0.0, coat=0.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = color
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Roughness"].default_value = roughness
        if "Anisotropic" in b.inputs:
            b.inputs["Anisotropic"].default_value = aniso
        if "Coat Weight" in b.inputs:
            b.inputs["Coat Weight"].default_value = coat
            b.inputs["Coat Roughness"].default_value = 0.15
        return m

    return {
        "aluminium": principled("Aluminium", (0.78, 0.79, 0.81, 1.0), 1.0, 0.30, aniso=0.6),
        "anodized": principled("DarkAnodized", (0.035, 0.037, 0.042, 1.0), 0.85, 0.38),
        "crimson": principled("WPICrimson", hex_linear("#AC2B37"), 0.25, 0.32, coat=0.6),
    }


def import_link_meshes(materials) -> dict[str, tuple[bpy.types.Mesh, Matrix, str]]:
    """Mesh datablock, visual-origin matrix and material key per URDF link."""
    root = ET.parse(URDF).getroot()
    mesh_dir = os.path.join(os.path.dirname(os.path.dirname(URDF)), "meshes")
    out = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        name = link.get("name")
        o = vis.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        offset = Matrix.Translation(xyz) @ Euler(rpy, "XYZ").to_matrix().to_4x4()
        fname = os.path.basename(vis.find("geometry/mesh").get("filename"))
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.wm.stl_import(filepath=os.path.join(mesh_dir, fname))
        obj = bpy.context.selected_objects[0]
        key = next(v for k, v in LINK_MATERIAL.items() if name.startswith(k))
        obj.data.materials.clear()
        obj.data.materials.append(materials[key])
        if key == "crimson":
            obj.data.materials.append(materials["anodized"])
            n_motor = mark_motor_parts(obj.data, material_index=1)
            print(f"[blender] {name}: {n_motor} motor-like parts -> anodized")
        n_faces = len(obj.data.polygons)
        if n_faces > MAX_FACES:
            mod = obj.modifiers.new("decimate", "DECIMATE")
            mod.ratio = TARGET_FACES / n_faces
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            old = obj.data
            obj.modifiers.clear()
            obj.data = mesh
            bpy.data.meshes.remove(old)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35.0))
        obj.data.name = f"mesh_{name}"
        print(f"[blender] {name}: {n_faces} -> {len(obj.data.polygons)} faces, {key}")
        out[name] = (obj.data, offset, key)
        bpy.data.objects.remove(obj)
    return out


def mark_motor_parts(mesh: bpy.types.Mesh, material_index: int, min_size: float = 0.06, tol: float = 0.05) -> int:
    """Give motor-like loose parts (two bounding-box sides >= ``min_size`` [m] and equal within ``tol``, i.e. a round
    housing) ``material_index``. Returns the number of such parts."""
    nv, nf = len(mesh.vertices), len(mesh.polygons)
    co = np.empty(3 * nv)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    starts, totals = np.empty(nf, dtype=np.int64), np.empty(nf, dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    tri = loops[starts[:, None] + np.minimum(np.arange(3)[None, :], totals[:, None] - 1)]  # first 3 verts per face
    label = np.arange(nv)
    while True:  # min-label propagation over faces + pointer jumping = connected components
        m = label[tri].min(axis=1)
        new = label.copy()
        for j in range(3):
            np.minimum.at(new, tri[:, j], m)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    mat = np.zeros(nf, dtype=np.int64)
    mesh.polygons.foreach_get("material_index", mat)
    count = 0
    face_label = label[tri[:, 0]]
    for lab in np.unique(label):
        d = np.sort(np.ptp(co[label == lab], axis=0))
        if (d[2] >= min_size and d[1] >= min_size and d[2] - d[1] <= tol * d[2]) or (
            d[1] >= min_size and d[0] >= min_size and d[1] - d[0] <= tol * d[1]
        ):
            mat[face_label == lab] = material_index
            count += 1
    mesh.polygons.foreach_set("material_index", mat)
    return count


def fcurves_of(obj):
    """F-curves of the object's action (layered actions in Blender >= 4.4, legacy list before)."""
    action = obj.animation_data.action
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return action.fcurves
    from bpy_extras import anim_utils

    return anim_utils.action_get_channelbag_for_slot(action, obj.animation_data.action_slot).fcurves


def bake_channels(obj, frames: np.ndarray, channels: dict[str, np.ndarray]):
    """Linear keys for every frame: channels {data_path: (T, n)}."""
    for path, vals in channels.items():
        for i in range(vals.shape[1]):
            obj.keyframe_insert(data_path=path, index=i, frame=float(frames[0]))
    fcs = {(fc.data_path, fc.array_index): fc for fc in fcurves_of(obj)}
    for path, vals in channels.items():
        for i in range(vals.shape[1]):
            fc = fcs[(path, i)]
            fc.keyframe_points.clear()
            fc.keyframe_points.add(len(frames))
            co = np.empty(2 * len(frames), dtype=np.float32)
            co[0::2], co[1::2] = frames, vals[:, i]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * len(frames))  # LINEAR
            fc.update()


def build_robots(data, meshes, frames: np.ndarray):
    names = [str(n) for n in data["body_names"]]
    pos = data["body_pos"][frames]  # (F, R, B, 3)
    quat = data["body_quat"][frames]  # xyzw
    n_robots = pos.shape[1]
    for r in range(n_robots):
        for b, name in enumerate(names):
            empty = bpy.data.objects.new(f"R{r}_{name}", None)
            empty.rotation_mode = "QUATERNION"
            bpy.context.scene.collection.objects.link(empty)
            mesh, offset, _ = meshes[name]
            obj = bpy.data.objects.new(f"R{r}_{name}_mesh", mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = empty
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = offset
            q = quat[:, r, b][:, [3, 0, 1, 2]].astype(np.float64)  # wxyz
            # sign continuity (q and -q are the same rotation)
            for k in range(1, len(q)):
                if np.dot(q[k], q[k - 1]) < 0.0:
                    q[k] = -q[k]
            bake_channels(empty, frames, {"location": pos[:, r, b], "rotation_quaternion": q})


def build_studio(center_xy, radius: float):
    """White cyclorama bowl (flat floor + quarter-circle cove), white world."""
    flat, cove, rings, seg = radius, 12.0, 24, 192
    prof = [(0.0, 0.0)] + [(flat * (i / 8.0), 0.0) for i in range(1, 9)]
    prof += [(flat + cove * math.sin(a), cove * (1.0 - math.cos(a))) for a in np.linspace(0, math.pi / 2, rings)[1:]]
    prof += [(flat + cove, cove + 30.0)]
    verts, faces = [], []
    for j in range(seg):
        a = 2 * math.pi * j / seg
        for rr, z in prof:
            verts.append((center_xy[0] + rr * math.cos(a), center_xy[1] + rr * math.sin(a), z))
    n = len(prof)
    for j in range(seg):
        j2 = (j + 1) % seg
        for i in range(n - 1):
            faces.append((j * n + i, j2 * n + i, j2 * n + i + 1, j * n + i + 1))
    me = bpy.data.meshes.new("cyclorama")
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    m = bpy.data.materials.new("Backdrop")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.82, 0.82, 0.83, 1.0)
    b.inputs["Roughness"].default_value = 0.9
    # a little self-illumination keeps the far cove even (no hard terminator from the sun)
    b.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    b.inputs["Emission Strength"].default_value = 0.35
    me.materials.append(m)
    obj = bpy.data.objects.new("cyclorama", me)
    bpy.context.scene.collection.objects.link(obj)

    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.9, 0.9, 0.92, 1.0)
    bg.inputs["Strength"].default_value = 0.45
    bpy.context.scene.world = world


def build_light_rig(track: np.ndarray, frames: np.ndarray):
    """Key / fill / top / rim area lights parented to an empty that follows ``track`` (F, 3)."""
    rig = bpy.data.objects.new("light_rig", None)
    bpy.context.scene.collection.objects.link(rig)

    def area(name, energy, size, loc, color=(1.0, 1.0, 1.0)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = energy, size, color
        set_if(ld, "use_shadow", True)
        set_if(ld, "shadow_jitter", True)
        ob = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.parent = rig
        ob.location = loc
        d = Vector((0.0, 0.0, 0.6)) - Vector(loc)
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return ob

    area("key", 300.0, 2.5, (3.5, -3.0, 4.5), (1.0, 0.97, 0.93))
    area("fill", 220.0, 6.0, (-1.0, 4.5, 3.0), (0.93, 0.96, 1.0))
    area("top", 120.0, 8.0, (0.0, 0.0, 6.0))
    area("rim", 700.0, 2.5, (-4.0, 1.5, 2.2))
    bake_channels(rig, frames, {"location": track})
    # sun: the main shadow caster (soft, but defined contact shadows under the feet)
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy, sd.angle, sd.color = 4.5, math.radians(3.0), (1.0, 0.97, 0.93)
    set_if(sd, "use_shadow", True)
    sun = bpy.data.objects.new("sun", sd)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = Vector((-0.35, -0.6, -1.3)).to_track_quat("-Z", "Y").to_euler()
    return rig


def smooth_zero_phase(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Critically damped 2nd-order low-pass run forward then backward (no lag), along axis 0."""
    w = 1.0 / tau

    def run(sig):
        y, v = sig[0].copy(), np.zeros_like(sig[0])
        out = np.empty_like(sig)
        for k in range(len(sig)):
            a = w * w * (sig[k] - y) - 2.0 * w * v
            v = v + a * dt
            y = y + v * dt
            out[k] = y
        return out

    return run(run(x)[::-1])[::-1]


def keys_smooth(keys, t: np.ndarray) -> np.ndarray:
    """Smoothstep-interpolated keyframe track (t, value) evaluated at times ``t``."""
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        if tk <= keys[0][0]:
            out[k] = keys[0][1]
            continue
        out[k] = keys[-1][1]
        for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
            if tk <= t1:
                s = (tk - t0) / max(t1 - t0, 1e-9)
                out[k] = a + (b - a) * s * s * (3 - 2 * s)
                break
    return out


def camera_path(mode: str, data, frames: np.ndarray, dt: float):
    """Camera position and look-at target per frame (F, 3) each, and the light-rig track."""
    t = frames * dt
    root = data["body_pos"][:, :, 0]  # (T, R, 3)
    if mode == "hero":
        # horizontal follow only: camera and look-at heights are fixed (not tied to the hip), so the robot visibly
        # grows / shrinks on screen. HERO_DIST frames the (95, 95) mm robot (~1.0 m tall) at ~70% of frame height.
        p = smooth_zero_phase(root[:, 0, :2].astype(np.float64), dt, 0.45)[frames]
        az = np.radians(keys_smooth([(0, -38), (21.5, -38), (28.0, 28), (34.0, 28), (38.5, -38), (60, -38)], t))
        d0 = HERO_DIST
        dist = keys_smooth([(0, 4.4), (3.5, d0), (40.5, d0), (46.0, 6.0), (51.0, 6.0), (55.5, d0), (60, d0)], t)
        height = keys_smooth([(0, 0.85), (40.5, 0.85), (46.0, 1.5), (51.0, 1.5), (55.5, 0.85), (60, 0.85)], t)
        target = np.column_stack([p, np.full(len(p), HERO_LOOK_Z)])
        cam = np.column_stack([p[:, 0] + dist * np.cos(az), p[:, 1] + dist * np.sin(az), height])
        return cam, target, np.column_stack([p, np.zeros(len(p))]), az
    # lineup: static wide front-3/4, then a slow dolly along the line (y) while following the group forward
    gx = smooth_zero_phase(root[:, :, 0].mean(axis=1, keepdims=True).astype(np.float64), dt, 0.8)[frames, 0]
    ys = root[0, :, 1]
    y_lo, y_hi = float(ys.min()), float(ys.max())
    w = keys_smooth([(0, 0.0), (3.5, 0.0), (6.5, 1.0), (60, 1.0)], t)  # 0 static -> 1 tracking
    gx_eff = (1 - w) * gx[0] + w * gx
    cam_y = keys_smooth([(0, y_lo - 5.0), (3.5, y_lo - 5.0), (15.0, y_hi - 0.5)], t)
    cam_x = gx_eff + keys_smooth([(0, 7.0), (3.5, 7.0), (7.0, 3.6), (15.0, 3.6)], t)
    cam_z = keys_smooth([(0, 1.7), (3.5, 1.7), (7.0, 1.0), (15.0, 1.0)], t)
    tgt_y = keys_smooth([(0, 0.5 * (y_lo + y_hi)), (3.5, 0.5 * (y_lo + y_hi)), (7.0, y_lo + 1.0)], t)
    tgt_y = np.where(t > 7.0, np.minimum(cam_y + 2.6, y_hi), tgt_y)
    tgt_y = smooth_zero_phase(tgt_y[:, None], dt, 0.8)[:, 0]
    target = np.column_stack([gx_eff + 0.3, tgt_y, np.full_like(t, 0.45)])
    cam = np.column_stack([cam_x, cam_y, cam_z])
    return cam, target, np.column_stack([gx_eff, np.zeros_like(t), np.zeros_like(t)]), None


def build_height_ref(kind: str, track: np.ndarray, az: np.ndarray, frames: np.ndarray):
    """Faint height reference on a rig that follows the robot (horizontally) and faces the camera.

    ``ruler``: vertical scale at the robot's depth, beside it on the camera's left (ticks every 5 cm from 0.70 to
    1.05 m, labels at 0.8 / 0.9 / 1.0 m "hip height"). ``lines``: horizontal lines just behind the robot at the
    standing hip height of the shortest and longest configurations. Geometry is in true world heights (rig at z = 0),
    emissive (unaffected by the lights), casts no shadows, and is occluded by the robot.
    """
    rig = bpy.data.objects.new("height_ref", None)
    bpy.context.scene.collection.objects.link(rig)
    mat = bpy.data.materials.new("HeightRef")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.42, 0.43, 0.45, 1.0)
    em.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    font = bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None

    def finish(ob):
        ob.data.materials.append(mat)
        ob.parent = rig
        ob.visible_shadow = False
        bpy.context.scene.collection.objects.link(ob)

    def bar(name, x0, x1, z0, z1):
        """Quad in the rig's XZ plane (facing the camera)."""
        me = bpy.data.meshes.new(name)
        me.from_pydata([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)], [], [(0, 1, 2, 3)])
        finish(bpy.data.objects.new(name, me))

    def label(name, text, x, z, size, align="LEFT"):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body, cu.size, cu.align_x, cu.align_y = text, size, align, "CENTER"
        if font is not None:
            cu.font = font
        ob = bpy.data.objects.new(name, cu)
        ob.location = (x, 0.0, z)
        ob.rotation_euler = (math.pi / 2, 0.0, 0.0)  # stand up in XZ, facing -Y (the camera)
        finish(ob)

    if kind == "ruler":
        depth, x = 0.0, -0.55
        bar("ruler_spine", x - 0.002, x + 0.002, 0.70, 1.05)
        for i in range(8):
            z = 0.70 + 0.05 * i
            major = i in (2, 4, 6)
            bar(f"tick_{i}", x, x + (0.045 if major else 0.025), z - 0.0015, z + 0.0015)
            if major:
                label(f"lbl_{i}", f"{z:.1f} m", x - 0.015, z, 0.032, "RIGHT")
        label("ruler_title", "hip height", x, 1.085, 0.03, "CENTER")
    else:
        depth = 0.35
        for name, z, text in (
            ("short", HIP_SHORT, "hip height · shortest legs (5 / 5 mm)"),
            ("long", HIP_LONG, "hip height · longest legs (95 / 95 mm)"),
        ):
            bar(f"line_{name}", -1.05, 1.05, z - 0.0015, z + 0.0015)
            label(f"lbl_{name}", text, -1.05, z + 0.03, 0.042)
    # follow the robot horizontally; face the camera (local x = camera right); ``depth`` behind the robot
    fwd = np.column_stack([-np.cos(az), -np.sin(az)])
    loc = np.column_stack([track[:, :2] + depth * fwd, np.zeros(len(track))])
    rot = np.column_stack([np.zeros(len(az)), np.zeros(len(az)), az + math.pi / 2])
    bake_channels(rig, frames, {"location": loc, "rotation_euler": rot})


def build_camera(cam_pos: np.ndarray, target: np.ndarray, frames: np.ndarray, lens: float = 50.0):
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = lens, 36.0, 0.05, 500.0
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    quats = []
    for c, tg in zip(cam_pos, target):
        q = (Vector(tg) - Vector(c)).to_track_quat("-Z", "Y")
        quats.append([q.w, q.x, q.y, q.z])
    q = np.array(quats)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0.0:
            q[k] = -q[k]
    bake_channels(cam, frames, {"location": cam_pos, "rotation_quaternion": q})
    return cam


def setup_render(args, width=1920, height=1080):
    sc = bpy.context.scene
    r = sc.render
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    r.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    r.resolution_x, r.resolution_y, r.resolution_percentage = width, height, args.percent
    r.fps, r.fps_base = FPS, 1.0
    r.use_motion_blur = False
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.compression = 15
    ee = sc.eevee
    set_if(ee, "taa_render_samples", args.samples)
    set_if(ee, "use_shadows", True)
    set_if(ee, "shadow_ray_count", 2)
    set_if(ee, "shadow_step_count", 8)
    set_if(ee, "use_raytracing", True)
    set_if(ee, "use_fast_gi", True)
    set_if(ee, "fast_gi_method", "GLOBAL_ILLUMINATION")
    set_if(ee, "fast_gi_distance", 0.6)
    set_if(ee, "use_gtao", True)  # pre-4.2 EEVEE
    set_if(ee, "gtao_distance", 0.6)
    vs = sc.view_settings
    set_if(vs, "view_transform", "AgX")
    set_if(vs, "look", "AgX - Medium High Contrast")
    vs.exposure = -0.5
    print(f"[blender] engine {r.engine}, {r.resolution_x}x{r.resolution_y} @ {args.percent}%, {args.samples} spp")


def main():
    args = parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    data = np.load(os.path.expanduser(args.npz))
    dt = float(data["dt"])
    T = data["body_pos"].shape[0]
    first, last = (0, T - 1) if args.frames is None else (int(v) for v in args.frames.split(":"))
    last = min(last, T - 1)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    materials = make_materials()
    meshes = import_link_meshes(materials)
    # keys for the rendered range only (+ the whole recording is used for the smoothing of the camera)
    frames = np.arange(first, last + 1)
    build_robots(data, meshes, frames)

    root = data["body_pos"][:, :, 0]
    center = 0.5 * (root[:, :, :2].reshape(-1, 2).min(axis=0) + root[:, :, :2].reshape(-1, 2).max(axis=0))
    span = float(np.linalg.norm(root[:, :, :2].reshape(-1, 2).max(axis=0) - root[:, :, :2].reshape(-1, 2).min(axis=0)))
    build_studio(center, radius=0.5 * span + 25.0)

    if args.mode == "align":
        with open(os.path.expanduser(args.camera)) as fh:
            cfg = json.load(fh)
        setup_render(args, cfg["width"], cfg["height"])
        cd = bpy.data.cameras.new("cam")
        cd.sensor_fit = "VERTICAL"
        cd.angle_y = math.radians(cfg["fov_vertical_deg"])
        cd.clip_start = 0.01
        cam = bpy.data.objects.new("cam", cd)
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.location = cfg["pos"]
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = Vector(cfg["front"]).to_track_quat("-Z", "Y")
        track = np.column_stack([root[frames, 0, :2], np.zeros(len(frames))])
    else:
        setup_render(args)
        cam_pos, target, track, az = camera_path(args.mode, data, frames, dt)
        cam = build_camera(cam_pos, target, frames)
        if args.mode == "hero" and args.height_ref != "none":
            build_height_ref(args.height_ref, track, az, frames)
    build_light_rig(track, frames)

    sc.frame_start, sc.frame_end = first, last
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.expanduser(args.blend))

    # image-space robot anchors (root projected to the floor, and the hip) for HUD labels
    screen = {}
    n_robots = data["body_pos"].shape[1]
    for f in frames[:: 1 if args.mode == "lineup" else 25]:
        sc.frame_set(int(f))
        pts = []
        for r in range(n_robots):
            p = root[f, r]
            a = world_to_camera_view(sc, cam, Vector((p[0], p[1], 0.0)))
            h = world_to_camera_view(sc, cam, Vector((p[0], p[1], p[2])))
            pts.append([a.x, 1.0 - a.y, a.z, h.x, 1.0 - h.y])
        screen[int(f)] = pts
    with open(os.path.join(out, "screen.json"), "w") as fh:
        json.dump(screen, fh)

    sc.render.filepath = os.path.join(out, "#####")
    import time

    t0 = time.time()
    bpy.ops.render.render(animation=True)
    n = last - first + 1
    print(f"[blender] rendered {n} frames in {time.time() - t0:.1f} s ({(time.time() - t0) / n:.2f} s/frame)")


main()
