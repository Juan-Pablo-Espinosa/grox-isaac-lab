"""Measure actual foot COLLISION-MESH bottom-surface world-Z during a standing rollout
(ground plane at z=0).

USD BBoxCache reads the statically-authored USD transform, not the live PhysX state
(Fabric writes bypass what plain pxr.Usd sees) -- confirmed by observing a value frozen
bit-for-bit across 200 steps. Instead: get the foot's LOCAL rest-pose geometry offset
(a fixed property, safe to read once via BBoxCache) from the body's own origin, then
combine that fixed local offset with the LIVE world position/orientation from
robot.data.body_pos_w / body_quat_w (already verified reliable, sourced directly from
PhysX tensors) via proper quaternion rotation to get an accurate live world height.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab_tasks
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Standing-Nova-v0", device=args_cli.device, num_envs=4)
env = gym.make("Isaac-Standing-Nova-v0", cfg=cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
body_names = robot.data.body_names
foot_ids = [i for i, n in enumerate(body_names) if "Feet" in n]
print("foot body indices/names:", [(i, body_names[i]) for i in foot_ids])

stage = sim_utils.SimulationContext.instance().stage
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_], useExtentsHint=True)


def find_prim_by_name(root_path: str, name: str):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


# Fixed local min-Z offset (bottom-of-sole minus body origin), read once from the
# authored rest pose -- this is a static geometry property, safe to read via BBoxCache.
local_min_z_offset = {}
for i in foot_ids:
    name = body_names[i]
    prim = find_prim_by_name("/World/envs/env_0/Robot", name)
    bbox_cache.Clear()
    local_bbox = bbox_cache.ComputeUntransformedBound(prim)
    local_min_z_offset[i] = local_bbox.ComputeAlignedRange().GetMin()[2]
print("local sole-min-z offsets (rest pose, body-local frame):", local_min_z_offset)

for step in range(200):
    action = torch.zeros(env.unwrapped.action_space.shape)
    env.step(action)
    if step % 20 == 0:
        live_sole_z = []
        for i in foot_ids:
            pos = robot.data.body_pos_w[0, i]
            quat = robot.data.body_quat_w[0, i]  # (x, y, z, w)
            local_offset = torch.tensor([0.0, 0.0, local_min_z_offset[i]])
            world_offset = quat_apply(quat.unsqueeze(0), local_offset.unsqueeze(0))[0]
            live_sole_z.append((pos[2] + world_offset[2]).item())
        print(f"step={step:3d} live_sole_z={live_sole_z}")

env.close()
simulation_app.close()
