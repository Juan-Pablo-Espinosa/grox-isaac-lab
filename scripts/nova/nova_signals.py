# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step signals of one NOVA env for the teleop dashboard and recordings (import after the app is launched).

One row per control step, built on the GPU (no host sync unless a consumer pulls it):
  cmd_vx/vy/wz, vx/vy/wz (base frame) [m/s, rad/s]; pos_x/pos_y (world, relative to the env origin) [m];
  p_elec / p_mech / p_copper [W] (mdp/power.py: P = max(mech + cu, 0) per motor, mech = P - cu);
  p_<group>, mech_<group>, cu_<group> for hip_knee, roll, yaw, ankle, leadscrew [W];
  tau_frac_<group> = mean |tau| / effort limit over the joints of hip_knee, roll, yaw, ankle;
  ankle_rs02_frac_max and ankle_rs02_frac_<motor> = |ankle motor torque| / 6 N·m (RS02 rated);
  ankle_rs02_peak_frac_max = max ankle motor |torque| / 17 N·m (RS02 peak);
  q_U, q_L (L/R mean joint position), q_U_target, q_L_target, q_U_goal, q_L_goal [mm];
  contact_L, contact_R (0/1); mass [kg]; ankle_n_pitch, ankle_n_roll (transmission used by the power model);
  tau_<joint> [N·m or N], qd_<joint> [rad/s or m/s], p_<motor> [W];
  taum_<motor> = |motor output torque| [N·m], pct_cont_<motor> / pct_peak_<motor> = taum / rated / peak torque [%].
The recorder adds t [s], energy_J (cumulative) and cot_running (energy / (m g path length)) when it writes the CSV.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from isaaclab_tasks.manager_based.locomotion.walking.config.nova.mdp.power import (
    _MOTORS,
    DASHBOARD_GROUPS,
    ROBSTRIDE,
    dashboard_group,
    power_model,
)

TORQUE_GROUPS = {
    "hip_knee": ("Hip_Pitch_", "Lowerleg_Pitch_"),
    "roll": ("Hip_Roll_",),
    "yaw": ("Upperleg_Yaw_",),
    "ankle": ("Feet_Roll_", "Feet_Pitch_"),
}


class SignalExtractor:
    """Builds the signal row of one env from the live env state."""

    def __init__(self, unwrapped, rate_source: str, leg_target, leg_goal):
        """Signal source for ``unwrapped``.

        ``leg_target()`` -> (num_envs, 4) prismatic targets [m]; ``leg_goal()`` -> (num_envs, 2) upper/lower goal [m].
        """
        self.u = unwrapped
        self.rate_source = rate_source
        self.leg_target, self.leg_goal = leg_target, leg_goal
        robot = unwrapped.scene["robot"]
        self.robot, self.contact = robot, unwrapped.scene["contact_forces"]
        dev = unwrapped.device
        self.foot_ids = [self.contact.body_names.index(b) for b in ("Feet_Pitch_Left", "Feet_Pitch_Right")]
        model = power_model(unwrapped, rate_source)
        self.labels = list(model.labels)
        groups = [dashboard_group(lab) for lab in self.labels]
        self.group_matrix = torch.zeros(len(self.labels), len(DASHBOARD_GROUPS), device=dev)
        for i, g in enumerate(groups):
            self.group_matrix[i, DASHBOARD_GROUPS.index(g)] = 1.0
        self.ankle_motor_ids = [i for i, lab in enumerate(self.labels) if lab.startswith("ankle")]
        jn = list(robot.joint_names)
        self.joint_names = jn
        self.tau_groups = {g: [i for i, n in enumerate(jn) if n.startswith(p)] for g, p in TORQUE_GROUPS.items()}
        prismatic_joints = [
            "Upperleg_Prismatic_Left_Joint",
            "Upperleg_Prismatic_Right_Joint",
            "Lowerleg_Prismatic_Left_Joint",
            "Lowerleg_Prismatic_Right_Joint",
        ]
        self.pris_ids = [jn.index(n) for n in prismatic_joints]
        self.mass = robot.data.body_mass.torch.sum(dim=1).to(dev)
        self.transmission = torch.tensor(model.ankle_transmission, device=dev)
        specs = [ROBSTRIDE[m[1]] for m in _MOTORS]
        self.inv_rated = torch.tensor([100.0 / sp.rated_torque for sp in specs], device=dev)
        self.inv_peak = torch.tensor([100.0 / sp.peak_torque for sp in specs], device=dev)
        self.names = (
            ["cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "pos_x", "pos_y", "p_elec", "p_mech", "p_copper"]
            + [f"{k}_{g}" for g in DASHBOARD_GROUPS for k in ("p", "mech", "cu")]
            + [f"tau_frac_{g}" for g in TORQUE_GROUPS]
            + ["ankle_rs02_frac_max"]
            + [f"ankle_rs02_frac_{self.labels[i]}" for i in self.ankle_motor_ids]
            + ["ankle_rs02_peak_frac_max"]
            + ["q_U", "q_L", "q_U_target", "q_L_target", "q_U_goal", "q_L_goal", "contact_L", "contact_R", "mass"]
            + ["ankle_n_pitch", "ankle_n_roll"]
            + [f"tau_{n}" for n in jn]
            + [f"qd_{n}" for n in jn]
            + [f"p_{lab}" for lab in self.labels]
            + [f"taum_{lab}" for lab in self.labels]
            + [f"pct_cont_{lab}" for lab in self.labels]
            + [f"pct_peak_{lab}" for lab in self.labels]
        )
        # per-joint / per-motor columns are recorded but not sent to the dashboard
        detail = {f"tau_{n}" for n in jn} | {f"qd_{n}" for n in jn} | {f"p_{lab}" for lab in self.labels}
        detail |= {f"{k}_{lab}" for lab in self.labels for k in ("taum", "pct_cont", "pct_peak")}
        self.dashboard_idx = [i for i, n in enumerate(self.names) if n not in detail]

    def row(self, env_id: int = 0) -> torch.Tensor:
        """Signal row of env ``env_id`` (1-D tensor on the env device, columns :attr:`names`)."""
        u, d, e = self.u, self.robot.data, env_id
        p, mech, cu, tau_m = power_model(u, self.rate_source).compute(u)
        p, mech, cu, tau_m = p[e], mech[e], cu[e], tau_m[e]
        cmd = u.command_manager.get_command("base_velocity")[e]
        v = d.root_lin_vel_b.torch[e]
        w = d.root_ang_vel_b.torch[e, 2:3]
        pos = d.root_pos_w.torch[e, :2] - u.scene.env_origins[e, :2]
        tau, qd, q = d.applied_torque.torch[e], d.joint_vel.torch[e], d.joint_pos.torch[e]
        lim = d.joint_effort_limits.torch[e]
        frac = torch.stack([(tau[ids].abs() / lim[ids]).mean() for ids in self.tau_groups.values()])
        ankle = tau_m[self.ankle_motor_ids] / ROBSTRIDE["RS02"].rated_torque
        qp = q[self.pris_ids]
        tgt = self.leg_target()[e]
        goal = self.leg_goal()[e]
        legs = torch.stack([qp[0:2].mean(), qp[2:4].mean(), tgt[0:2].mean(), tgt[2:4].mean(), goal[0], goal[1]]) * 1000
        contact = (self.contact.data.current_contact_time.torch[e, self.foot_ids] > 0.0).float()
        grp = torch.stack([p @ self.group_matrix, mech @ self.group_matrix, cu @ self.group_matrix], dim=1).flatten()
        return torch.cat(
            [
                cmd[:3],
                v[:2],
                w,
                pos,
                torch.stack([p.sum(), mech.sum(), cu.sum()]),
                grp,
                frac,
                ankle.max().unsqueeze(0),
                ankle,
                (tau_m[self.ankle_motor_ids].max() / ROBSTRIDE["RS02"].peak_torque).unsqueeze(0),
                legs,
                contact,
                self.mass[e : e + 1],
                self.transmission,
                tau,
                qd,
                p,
                tau_m,
                tau_m * self.inv_rated,
                tau_m * self.inv_peak,
            ]
        )


class Recorder:
    """Accumulates signal rows on the GPU while recording; writes one CSV per recording."""

    def __init__(self, names: list[str], out_dir: str, dt: float):
        self.names, self.out_dir, self.dt = names, out_dir, dt
        self.rows: list[torch.Tensor] | None = None
        self.label = ""

    @property
    def active(self) -> bool:
        return self.rows is not None

    def start(self, label: str):
        self.rows, self.label, self.t0 = [], label, time.strftime("%Y%m%d-%H%M%S")

    def add(self, row: torch.Tensor):
        if self.rows is not None:
            self.rows.append(row)

    def elapsed(self) -> float:
        return 0.0 if self.rows is None else len(self.rows) * self.dt

    def stop(self) -> str | None:
        """Write the CSV and return its path (None if nothing was recorded)."""
        rows, self.rows = self.rows, None
        if not rows:
            return None
        data = torch.stack(rows).cpu().numpy().astype(np.float64)
        col = {n: i for i, n in enumerate(self.names)}
        t = np.arange(len(data)) * self.dt
        energy = np.cumsum(data[:, col["p_elec"]]) * self.dt
        step = np.r_[0.0, np.hypot(np.diff(data[:, col["pos_x"]]), np.diff(data[:, col["pos_y"]]))]
        step[step > 0.5] = 0.0  # a reset teleports the base: not distance walked
        dist = np.cumsum(step)
        cot = np.where(dist > 0.1, energy / (data[:, col["mass"]] * 9.81 * np.maximum(dist, 1e-9)), np.nan)
        out = np.column_stack([t, data, energy, dist, cot])
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"rec_{self.t0}_{self.label}.csv")
        header = ",".join(["t"] + self.names + ["energy_J", "distance_m", "cot_running"])
        np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.6g")
        return path
