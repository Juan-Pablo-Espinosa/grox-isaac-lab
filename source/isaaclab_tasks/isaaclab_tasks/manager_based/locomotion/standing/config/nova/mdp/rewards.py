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


# CRITICAL WIRING REQUIREMENT for effort_reward, acceleration_reward, and
# symmetry_reward (any reward term below that reads asset_cfg.joint_ids): the
# manager's own SceneEntityCfg-resolution pass ONLY walks RewTermCfg.params
# (isaaclab/managers/manager_base.py:394-395, `for key, value in
# term_cfg.params.items(): self._resolve_param_value(...)`) -- it does NOT
# introspect function-signature DEFAULT parameter values. A RewTerm wired as
# `RewTerm(func=effort_reward, weight=3.0)` with no explicit `params=` means
# asset_cfg.resolve() is NEVER called; asset_cfg.joint_ids silently stays at its
# unresolved slice(None) default (all 16 joints, prismatic included), while
# asset_cfg.joint_names still shows the correct 12-name list (set directly by
# the constructor, independent of resolve()) -- an easy-to-miss inconsistency.
# Verified this exact failure mode this session: it doesn't crash a pure-sum
# formula (sum over 16 vs 12 is still a valid op, just silently wrong, which is
# why effort_reward/acceleration_reward carried this bug undetected until
# symmetry_reward's element-wise per-pair indexing raised a hard shape error
# against it). standing_env_cfg.py's RewardsCfg MUST pass
# params={"asset_cfg": SceneEntityCfg(..., preserve_order=True)} explicitly for
# all three of these terms -- relying on the function default is not enough.
#
# Verified per-joint effort_limit_sim [N*m] (source/isaaclab_assets/isaaclab_assets/
# robots/nova.py's actuator groups), in the SAME order as the joint_names list
# below -- used by both effort_reward and symmetry_reward.
_REVOLUTE_JOINT_NAMES = [
    "Hip_Pitch_Left_Joint", "Hip_Pitch_Right_Joint",
    "Hip_Roll_Left_Joint", "Hip_Roll_Right_Joint",
    "Upperleg_Yaw_Left_Joint", "Upperleg_Yaw_Right_Joint",
    "Lowerleg_Pitch_Left_Joint", "Lowerleg_Pitch_Right_Joint",
    "Feet_Roll_Left_Joint", "Feet_Roll_Right_Joint",
    "Feet_Pitch_Left_Joint", "Feet_Pitch_Right_Joint",
]
_REVOLUTE_EFFORT_LIMITS = [120.0, 120.0, 60.0, 60.0, 60.0, 60.0, 120.0, 120.0, 34.0, 34.0, 34.0, 34.0]


def effort_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_REVOLUTE_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Effort reward: exp(-k * sum((torque_i/limit_i)^2)) over the 12 revolute
    joints, normalized by each joint's own effort_limit_sim (percentage-of-max,
    not raw N*m) -- so joints with very different torque budgets (Feet: 34 N*m
    vs Hip_Pitch/Lowerleg_Pitch: 120 N*m) are penalized on a comparable scale.

    preserve_order=True on asset_cfg: SceneEntityCfg.resolve() calls
    Articulation.find_joints(..., preserve_order=self.preserve_order), default
    False -- without this, joint_ids could come back reordered relative to
    _REVOLUTE_EFFORT_LIMITS's index-matched order, silently pairing the wrong
    joint with the wrong limit. Verified this exact risk this session while
    building the offline telemetry script (scripts/tools/gather_torque_telemetry.py).

    Prismatic joints excluded — they are morphology-control, not policy-controlled
    (established design decision, Grade 3/4).

    k=5.1782 -- REAL, data-derived value (was k=0.000159 on raw torque_sq_sum, an
    analytical guess). Solved so reward=0.5 at the real percentage_torque_sq_sum
    p90 telemetry value (0.13386) gathered from the converged model_1550.pt
    checkpoint (scripts/tools/gather_torque_telemetry.py). Verified:
    exp(-5.1782*0.13386) = 0.500.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limits = torch.tensor(_REVOLUTE_EFFORT_LIMITS, dtype=torques.dtype, device=torques.device)
    pct_sq_sum = torch.sum((torques / limits) ** 2, dim=1)

    k = 5.1782
    return torch.exp(-k * pct_sq_sum)


def acceleration_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_REVOLUTE_JOINT_NAMES),
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


def symmetry_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_REVOLUTE_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Symmetry reward: exp(-k * symmetry_error), where symmetry_error sums a
    per-pair error term over the 6 L/R joint pairs, using the sign convention
    each pair actually requires for CORRECT symmetric standing to read as zero
    error -- NOT a naive (pos_L - pos_R)^2 for every pair.

    Measures POSITION symmetry (asset.data.joint_pos), not torque. An earlier
    torque-based version of this term (torque_L ~= -torque_R per pair) let the
    policy find a degenerate solution: park Upperleg_Yaw_Left at its hard limit
    (+0.79 rad, needs near-zero holding torque there) while Right sat elsewhere
    entirely (+0.54 to +0.71, nowhere near the mirrored -0.79) -- torque
    symmetry was satisfied (reward 0.97-0.9995 across 8 independent envs) while
    the actual STANCE was grossly asymmetric (the "diva pose"). Confirmed via
    direct FK mirror-image comparison on the converged checkpoint: 4 of 6 pairs
    failed to match their own theoretical mirror image in real joint-position
    space despite near-maximal torque-based reward. Position is the quantity
    that actually needs to be symmetric for a symmetric-looking stance, so this
    is what's now measured.

    preserve_order=True on asset_cfg: same reasoning as effort_reward -- pairing
    positions[:, i] with a specific named joint below requires joint_ids to come
    back in _REVOLUTE_JOINT_NAMES's order, which find_joints does not guarantee
    unless explicitly requested.

    Sign convention was NOT guessed from joint names or from position-limit
    asymmetry alone (the knee's [0,1.85]/[-1.85,0] limits are suggestive but not
    proof by themselves). It was verified empirically via forward kinematics
    (scripts/tools/verify_pair_sign_convention.py): perturb one joint at a time
    (all others held at 0, gravity off), then check whether Right(+theta) or
    Right(-theta) reproduces the true Y-mirror-image of Left(+theta)'s resulting
    body position. Also confirmed the "nominal pose" sanity check the process
    started with is degenerate for this robot: default_joint_pos is exactly 0.0
    for all 16 joints (verified via scripts/tools/check_default_joint_pos.py), so
    pos_L-pos_R=0 and pos_L+pos_R=0 both hold trivially there regardless of which
    convention is correct -- it has zero discriminating power, which is why the
    FK-perturbation test was used instead.

    Verified result: 4 of 6 pairs are opposite-sign (physically symmetric
    standing means torque_L ~= -torque_R), not just the knee as originally
    suspected -- Hip_Pitch, Hip_Roll, Upperleg_Yaw, and Lowerleg_Pitch are all
    opposite-sign; only Feet_Roll and Feet_Pitch are same-sign. Notably this is
    NOT simply "does the raw axis vector's Y-component flip between sides":
    Hip_Roll's axis is (1,0,0), IDENTICAL and unmirrored on both sides in the
    URDF, yet it is still empirically opposite-sign -- mirroring a body is an
    orientation-reversing transform, which can flip the effective rotation sign
    even for an unmirrored raw axis vector. This is exactly why the convention
    was verified in simulation rather than read off the URDF by inspection.

    k=1.4255 -- REAL, data-derived value for the POSITION-based symmetry_error
    (rad^2). Solved so reward=0.05 at the real median degenerate-behavior
    telemetry value (2.10150 rad^2) gathered from checkpoint model_2999.pt --
    the exact "diva pose" run, dominated by Upperleg_Yaw_Left pinned at its
    +0.79 rad hard limit (scripts/tools/gather_position_symmetry_telemetry.py).
    Verified: exp(-1.4255*2.1015) = 0.0500.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    positions = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # indices within this 12-joint slice, matching _REVOLUTE_JOINT_NAMES's order
    hip_pitch_l, hip_pitch_r = positions[:, 0], positions[:, 1]
    hip_roll_l, hip_roll_r = positions[:, 2], positions[:, 3]
    upperleg_yaw_l, upperleg_yaw_r = positions[:, 4], positions[:, 5]
    lowerleg_pitch_l, lowerleg_pitch_r = positions[:, 6], positions[:, 7]
    feet_roll_l, feet_roll_r = positions[:, 8], positions[:, 9]
    feet_pitch_l, feet_pitch_r = positions[:, 10], positions[:, 11]

    symmetry_error = (
        (hip_pitch_l + hip_pitch_r) ** 2  # opposite-sign
        + (hip_roll_l + hip_roll_r) ** 2  # opposite-sign
        + (upperleg_yaw_l + upperleg_yaw_r) ** 2  # opposite-sign
        + (lowerleg_pitch_l + lowerleg_pitch_r) ** 2  # opposite-sign (the knee)
        + (feet_roll_l - feet_roll_r) ** 2  # same-sign
        + (feet_pitch_l - feet_pitch_r) ** 2  # same-sign
    )

    k = 1.4255
    return torch.exp(-k * symmetry_error)
