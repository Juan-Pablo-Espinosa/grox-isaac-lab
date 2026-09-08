"""Determine, empirically via forward kinematics, whether physically symmetric
standing means torque_L == torque_R (same-sign) or torque_L == -torque_R
(opposite-sign) for each of the 6 L/R joint pairs.

Method: for each pair, perturb ONLY that joint (all others held at 0) by +theta
on Left, +theta on Right, and -theta on Right (three separate parallel envs),
then read the resulting WORLD-frame position of that joint's own child body.
The nominal (theta=0) Left/Right body positions are themselves Y-mirror images
(confirmed from the URDF's mirrored origin offsets). If Right(+theta)'s position
is the Y-mirror image of Left(+theta)'s position, the pair is same-sign. If
Right(-theta) is the mirror instead, the pair is opposite-sign.

Uses a raw Articulation (not the env), so the knee's runtime limit override
(applied only via NovaStandingEnvCfg's startup event) doesn't constrain this
test -- this checks the geometric ground truth from the USD/URDF-derived
kinematics, independent of any later runtime limit fix.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots import NOVA_LOWERBODY_CFG

THETA = 0.3

PAIRS = [
    ("Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint", "Hip_Pitch_Left", "Hip_Pitch_Right"),
    ("Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint", "Hip_Roll_Left", "Hip_Roll_Right"),
    ("Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint", "Upperleg_Yaw_Left", "Upperleg_Yaw_Right"),
    ("Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint", "Lowerleg_Pitch_Left", "Lowerleg_Pitch_Right"),
    ("Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint", "Feet_Roll_Left", "Feet_Roll_Right"),
    ("Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint", "Feet_Pitch_Left", "Feet_Pitch_Right"),
]
# 3 test configs per pair: (which joint gets perturbed, sign)
# env layout: pair_idx*3 + {0: Left+theta, 1: Right+theta, 2: Right-theta}
NUM_ENVS = len(PAIRS) * 3

sim_cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
sim = SimulationContext(sim_cfg)

robot_cfg: ArticulationCfg = NOVA_LOWERBODY_CFG.replace(prim_path="/World/envs/env_.*/Robot")
robot_cfg.init_state.pos = (0.0, 0.0, 2.0)  # clear of ground, gravity off anyway

# Manually clone envs (raw Articulation script, no InteractiveScene) -- mirrors the
# established test_nova_standing.py pattern extended to N envs via a simple grid.
from pxr import UsdGeom

stage = sim_utils.SimulationContext.instance().stage
spacing = 3.0
for i in range(NUM_ENVS):
    xform = UsdGeom.Xform.Define(stage, f"/World/envs/env_{i}")
    xform.AddTranslateOp().Set((float(i % 6) * spacing, float(i // 6) * spacing, 0.0))

robot = Articulation(cfg=robot_cfg)
sim.reset()

joint_names = robot.data.joint_names
targets = torch.zeros(NUM_ENVS, len(joint_names))

for pair_idx, (l_joint, r_joint, l_body, r_body) in enumerate(PAIRS):
    l_id = joint_names.index(l_joint)
    r_id = joint_names.index(r_joint)
    e_lplus, e_rplus, e_rminus = pair_idx * 3, pair_idx * 3 + 1, pair_idx * 3 + 2
    targets[e_lplus, l_id] = THETA
    targets[e_rplus, r_id] = THETA
    targets[e_rminus, r_id] = -THETA

robot.set_joint_position_target(targets)
for _ in range(200):
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_cfg.dt)

body_names = robot.data.body_names
print("=" * 100)
for pair_idx, (l_joint, r_joint, l_body, r_body) in enumerate(PAIRS):
    e_lplus, e_rplus, e_rminus = pair_idx * 3, pair_idx * 3 + 1, pair_idx * 3 + 2
    l_body_id = body_names.index(l_body)
    r_body_id = body_names.index(r_body)

    # Measure body position relative to this env's own root (Hip_Base), which
    # cancels out the per-env grid placement offset used to keep envs apart.
    root_id = 0  # Hip_Base is body index 0

    pos_Lplus = (robot.data.body_pos_w[e_lplus, l_body_id] - robot.data.body_pos_w[e_lplus, root_id]).tolist()
    pos_Rplus = (robot.data.body_pos_w[e_rplus, r_body_id] - robot.data.body_pos_w[e_rplus, root_id]).tolist()
    pos_Rminus = (robot.data.body_pos_w[e_rminus, r_body_id] - robot.data.body_pos_w[e_rminus, root_id]).tolist()

    mirror_Lplus = [pos_Lplus[0], -pos_Lplus[1], pos_Lplus[2]]

    def dist(a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    d_same = dist(mirror_Lplus, pos_Rplus)
    d_opp = dist(mirror_Lplus, pos_Rminus)
    verdict = "SAME-SIGN" if d_same < d_opp else "OPPOSITE-SIGN"

    achieved_l = robot.data.joint_pos[e_lplus, joint_names.index(l_joint)].item()
    achieved_rplus = robot.data.joint_pos[e_rplus, joint_names.index(r_joint)].item()
    achieved_rminus = robot.data.joint_pos[e_rminus, joint_names.index(r_joint)].item()

    print(f"\nPair: {l_joint} / {r_joint}   (verdict: {verdict})")
    print(f"  achieved joint angles: L(+theta)={achieved_l:.4f}  R(+theta)={achieved_rplus:.4f}  R(-theta)={achieved_rminus:.4f}")
    print(f"  Left(+theta) body pos rel. root:      {[round(x,5) for x in pos_Lplus]}")
    print(f"  Y-mirror of that:                     {[round(x,5) for x in mirror_Lplus]}")
    print(f"  Right(+theta) body pos rel. root:      {[round(x,5) for x in pos_Rplus]}   dist_to_mirror={d_same:.6f}")
    print(f"  Right(-theta) body pos rel. root:      {[round(x,5) for x in pos_Rminus]}   dist_to_mirror={d_opp:.6f}")

print("=" * 100)
simulation_app.close()
