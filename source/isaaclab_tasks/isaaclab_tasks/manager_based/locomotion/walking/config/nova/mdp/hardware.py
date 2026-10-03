# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NOVA actuator hardware model: RobStride datasheet curves / inertias, ankle linkage, actuator configs.

Datasheet source: RobStride product specification "灵足时代RS系列产品规格介绍 (2026.09.17)"
(github.com/RobStride/Product_Information). Per module it gives the 48 V T-N table (load torque vs rotational speed
at the OUTPUT) and "低速端等效惯量 Moment Inertia", the rotor inertia reflected to the LOW-SPEED (output) side, i.e.
already multiplied by the gear ratio squared. Joint armature therefore uses that value directly
(= I_rotor * ratio^2).

Ankle: two RS02 per side sit on the shank and drive the foot through a 4-bar parallel linkage. With constant ratios
N_p (pitch, motors same direction) and N_r (roll, motors opposite) and motor output torques tau_1,2 / speeds w_1,2:
    tau_pitch = N_p (tau_1 + tau_2),   tau_roll = N_r (tau_1 - tau_2)
    tau_1,2   = (tau_pitch / N_p +/- tau_roll / N_r) / 2
    w_1,2     = N_p w_pitch +/- N_r w_roll
Virtual work: tau_1 w_1 + tau_2 w_2 = (tau_p/N_p)(N_p w_p) + (tau_r/N_r)(N_r w_r) = tau_p w_p + tau_r w_r.
Per-motor limit |tau_i| <= 6 N·m (RS02 continuous) is exactly the coupled joint limit
|tau_pitch| / N_p + |tau_roll| / N_r <= 12 N·m (pure pitch 36, pure roll 12 N·m at N_p = 3, N_r = 1).

Kit-free (imported by the task cfg before the simulator starts); the actuator classes live in mdp/actuators.py.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass

# ---------------------------------------------------------------------------------------------------------------------
# Ankle linkage ratios.
# TODO(JP): confirm N_p and N_r from CAD. Assumed constant over the joint range (a 4-bar's ratio varies with angle).
ANKLE_N_PITCH = 3.0
"""Ankle pitch transmission ratio (joint torque per summed motor torque; both motors same direction)."""
ANKLE_N_ROLL = 1.0
"""Ankle roll transmission ratio (joint torque per differential motor torque; motors opposite)."""
# ---------------------------------------------------------------------------------------------------------------------

RS02_CONTINUOUS_TORQUE = 6.0
"""RS02 rated (continuous) output torque [N·m]: the sim limit per ankle motor chosen by JP."""

RPM = 2.0 * math.pi / 60.0
_ACTUATORS = __name__.rsplit(".", 1)[0] + ".actuators"  # string class_type: mdp/actuators.py imports torch-level code

# 48 V T-N tables at the output, (speed [rpm], max load torque [N·m]), ascending speed, from the 2026.09.17 spec. The
# first point (0 rpm, peak) and the last (no-load speed, 0) are added: below the lowest tabulated speed the module
# delivers its peak torque, above the highest it falls linearly to 0 at the no-load speed.
TN_CURVES_RPM = {
    "RS00": [(0.0, 14.0), (100.0, 14.0), (150.0, 12.0), (225.0, 8.0), (270.0, 5.0), (315.0, 0.5), (315.5, 0.0)],
    "RS02": [(0.0, 17.0), (219.0, 17.0), (273.0, 14.0), (326.0, 10.0), (365.0, 7.0), (407.0, 0.5), (410.0, 0.0)],
    "RS03": [(0.0, 60.0), (120.0, 60.0), (145.0, 50.0), (178.0, 30.0), (187.0, 20.0), (188.0, 10.0), (195.0, 0.0)],
    "RS04": [(0.0, 120.0), (95.0, 120.0), (148.0, 80.0), (160.0, 60.0), (184.0, 40.0), (190.0, 10.0), (200.0, 0.0)],
    "RS06": [(0.0, 36.0), (280.0, 36.0), (320.0, 30.0), (390.0, 20.0), (426.0, 11.0), (430.0, 5.0), (480.0, 0.0)],
}
OUTPUT_INERTIA = {"RS00": 0.001, "RS02": 4.2e-3, "RS03": 0.02, "RS04": 0.04, "RS06": 0.012}
"""Rotor inertia reflected to the output (datasheet "low-speed-end equivalent inertia") [kg·m²]."""


def tn_curve(model: str) -> list[tuple[float, float]]:
    """T-N envelope of a module as (output speed [rad/s], max torque [N·m]) points."""
    return [(rpm * RPM, tau) for rpm, tau in TN_CURVES_RPM[model]]


# Joint armature [kg·m²]: sum over the driving motors of the output-side inertia x (joint-to-motor ratio)^2.
ARMATURE = {
    "Hip_Pitch_.*": OUTPUT_INERTIA["RS04"],
    "Lowerleg_Pitch_.*": OUTPUT_INERTIA["RS04"],
    "Hip_Roll_.*": OUTPUT_INERTIA["RS03"],
    "Upperleg_Yaw_.*": OUTPUT_INERTIA["RS06"],
    "Feet_Pitch_.*": 2.0 * OUTPUT_INERTIA["RS02"] * ANKLE_N_PITCH**2,
    "Feet_Roll_.*": 2.0 * OUTPUT_INERTIA["RS02"] * ANKLE_N_ROLL**2,
}
# The RS00 + 8 mm-lead screw would reflect OUTPUT_INERTIA["RS00"] * (2 pi / 0.008)^2 = 617 kg onto each prismatic
# joint (~300-450x the moving segment mass of 1.4-2.1 kg). NOT applied: with the leadscrew's stiff position drive it
# would make each leg segment a 617 kg mass on a k = 3e5 N/m spring (3.5 Hz, damping ratio 0.02) -- an oscillator
# the real screw does not have (it is driven at <= 35 mm/s and holds by friction, which the stiff drive models).


@configclass
class TorqueSpeedPDActuatorCfg(IdealPDActuatorCfg):
    """Explicit PD actuator with a tabulated motor torque-speed envelope (see mdp/actuators.py)."""

    class_type: type | str = (
        "isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.actuators:TorqueSpeedPDActuator"
    )
    tn_curve: list[tuple[float, float]] = MISSING
    """(output speed [rad/s], max torque [N·m]) points, ascending speed, ending at (no-load speed, 0)."""


@configclass
class AnkleLinkagePDActuatorCfg(IdealPDActuatorCfg):
    """Explicit PD on ankle pitch/roll with the 2-motor linkage and per-motor limits (see mdp/actuators.py)."""

    class_type: type | str = (
        "isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.actuators:AnkleLinkagePDActuator"
    )
    n_pitch: float = ANKLE_N_PITCH
    n_roll: float = ANKLE_N_ROLL
    motor_torque_limit: float = RS02_CONTINUOUS_TORQUE
    """Per-motor torque limit [N·m], both quadrants."""
    motor_tn_curve: list[tuple[float, float]] = MISSING
    """Per-motor T-N envelope (motoring quadrant), (motor output speed [rad/s], torque [N·m])."""
