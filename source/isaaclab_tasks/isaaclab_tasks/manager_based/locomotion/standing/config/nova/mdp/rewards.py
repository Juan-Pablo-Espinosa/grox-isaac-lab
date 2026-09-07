# Reward terms for NOVA_LOWERBODY_V2 standing task — Grade 4
#
# All terms derived and numerically verified against real robot data this session:
#   - up_z formula verified against Isaac Lab's own quaternion convention (x,y,z,w),
#     confirmed via isaaclab.utils.math docstrings.
#   - k values for uprightness/height solved algebraically from a target reward at
#     the real termination boundary (see terminations.py), then cross-checked
#     against the actual PD-hold fall trajectory from test_nova_standing.py.
#   - effort k is PROVISIONAL: analytically anchored (single-leg ~33 N*m estimate,
#     doubled for two legs) rather than verified against real active-balancing
#     telemetry, since no trained policy exists yet to measure that from. Deliberately
#     halved for a conservative/lenient starting bias per JP's explicit instruction.
#     Flagged for recheck once first training run produces real effort data.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

# Deferred to TYPE_CHECKING (matching the working anymal_c/mdp/rewards.py precedent):
# isaaclab.assets/isaaclab.envs use lazy __getattr__-based exports, so an eager,
# module-level `from isaaclab.assets import Articulation` forces immediate resolution
# of Articulation's real (pxr-touching) implementation at THIS module's import time.
# Since this mdp module gets imported while parsing the env cfg -- before
# SimulationApp/Kit has booted -- that premature pxr import caused a native
# "free(): invalid pointer" crash during Kit startup (verified via py-spy crash
# traceback + a control test: identical crash for Isaac-Standing-Nova-v0, none for
# Isaac-Velocity-Flat-Anymal-C-v0, the only difference being this eager import).
# Both names are used only as type annotations here; `from __future__ import
# annotations` makes annotations lazy strings, so this changes zero runtime behavior.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Uprightness reward: exp(-k * (1 - up_z)^2).

    up_z = 1 - 2*(x^2 + y^2), derived from quaternion (x,y,z,w) — verified
    against Isaac Lab's internal convention this session.
    k=16.16 solved so reward=0.25 exactly at the tilt termination boundary
    (up_z=0.7071, 45 deg from vertical).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w  # (num_envs, 4), order (x, y, z, w)
    x, y = quat[:, 0], quat[:, 1]
    up_z = 1.0 - 2.0 * (x**2 + y**2)

    k = 16.16
    return torch.exp(-k * (1.0 - up_z) ** 2)


def height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.82,
) -> torch.Tensor:
    """Height reward: exp(-k * (target_height - height)^2).

    target_height=0.82 verified via USD BBoxCache query (Grade 4 interstitial).
    k=27.63 solved so reward=0.25 exactly at the height termination boundary
    (0.596m, 50% of hip-to-knee span).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]

    k = 27.63
    return torch.exp(-k * (target_height - height) ** 2)


def effort_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[
        "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
        "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
        "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
        "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
        "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
        "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
    ]),
) -> torch.Tensor:
    """Effort reward: exp(-k * sum(torque^2)) over the 12 revolute joints only.

    Prismatic joints excluded — they are morphology-control, not policy-controlled
    (established design decision, Grade 3/4).

    k=0.000159 -- PROVISIONAL. Analytically anchored to ~33 N*m single-leg support
    estimate (both legs -> torque_sq ~2178), targeting reward=0.5 there, then HALVED
    per JP's explicit request for a conservative/lenient starting bias. Not yet
    verified against real active-balancing telemetry (no trained policy exists yet).
    Revisit after first training run with real effort data.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_sq_sum = torch.sum(torques**2, dim=1)

    k = 0.000159  # solved for 0.5 reward at anchor (2178), then halved for conservative margin
    return torch.exp(-k * torque_sq_sum)


def acceleration_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[
        "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
        "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
        "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
        "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
        "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
        "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
    ]),
) -> torch.Tensor:
    """Acceleration reward: exp(-k * sum(joint_acc^2)) over the 12 revolute joints.

    k=0.0000189 -- REAL, data-derived value (was k=0.000866, an analytical guess
    anchored to a rough ~10 rad/s^2-per-joint estimate that turned out to be off by
    ~4 orders of magnitude, which is why the reward logged exactly 0.0000 every
    iteration of the first 1500-iteration run: exp(-k*accel_sq_sum) was underflowing
    to bit-exact 0.0 in float32 almost every step).

    Derived from real accel_sq_sum telemetry gathered from the trained
    model_1499.pt checkpoint (384,000 samples, 256 envs x 1500 steps, real policy
    actions -- see scripts/tools/gather_accel_telemetry.py). Anchored to the
    STABLE-regime (up_z>0.97, |height-target|<0.03, post-settling) median of
    11,819.69, targeting reward=0.8 there (verified: exp(-0.0000189*11819.69) =
    0.7998; k=0.0000189 supplied by JP, applied as given -- not re-derived here).
    Full STABLE-regime percentiles for reference: p10=2,391, median=11,820,
    p90=112,971, p99=1,292,975.

    The extreme tail (p99 and beyond, up to ~2e8 in the full dataset) was
    investigated separately (scripts/tools/investigate_accel_tail.py) and found to
    be dominated almost exclusively by Feet_Roll_{Left,Right} -- confirmed via
    per-body mass query (scripts/tools/check_foot_inertia.py) to be a near-massless
    link (0.048kg, ~100x lighter than every neighboring body in the chain). Not a
    ground-contact artifact (foot height stayed well clear of ground at every
    top-8 spike this session observed); joint_acc's finite-difference computation
    itself is correct (verified via isaaclab_physx articulation_data.py -- it's
    recomputed every physics substep at dt=0.005s, not a coarse control-step
    average), but the underlying joint_vel signal for this specific low-inertia
    body is itself noisy (violent substep-to-substep sign flips, one observed
    event exceeding the joint's own velocity_limit_sim=15 rad/s by ~70%) --
    genuinely tiny torques on a near-massless body produce huge but momentary
    angular accelerations. Flagged for awareness if joint_acc is ever used in an
    observation rather than just this reward penalty; does not affect the STABLE-
    regime median this k is anchored to.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    accel_sq_sum = torch.sum(joint_acc**2, dim=1)

    k = 0.0000189
    return torch.exp(-k * accel_sq_sum)
