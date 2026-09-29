# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the nested-body PhysX contact sensor (implementation in :mod:`.contact_sensor`).

Kept separate from the implementation because task configs are imported *before* Kit is launched by the
training scripts; ``class_type`` is a string so the Kit-dependent implementation loads only when the sensor
is instantiated (same pattern as the stock ``isaaclab_physx.sensors.ContactSensorCfg``).
"""

from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.utils.configclass import configclass


@configclass
class NestedBodyContactSensorCfg(PhysXContactSensorCfg):
    """PhysX contact sensor config whose view supports rigid bodies nested in the USD hierarchy."""

    class_type: type | str = "{DIR}.contact_sensor:NestedBodyContactSensor"
