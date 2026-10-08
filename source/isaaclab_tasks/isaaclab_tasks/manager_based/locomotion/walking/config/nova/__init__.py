# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Walking-Nova-v0",
    entry_point=f"{__name__}.walking_env:NovaWalkingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:NovaWalkingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaWalkingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walking-Nova-Play-v0",
    entry_point=f"{__name__}.walking_env:NovaWalkingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:NovaWalkingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaWalkingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walking-Nova-MorphAgnostic-v0",
    entry_point=f"{__name__}.morph_agnostic_env:NovaMorphAgnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.morph_agnostic_env_cfg:NovaMorphAgnosticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaMorphAgnosticPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walking-Nova-MorphAgnostic-Play-v0",
    entry_point=f"{__name__}.morph_agnostic_env:NovaMorphAgnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.morph_agnostic_env_cfg:NovaMorphAgnosticEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaMorphAgnosticPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walking-Nova-MorphAgnostic-Squat-v0",
    entry_point=f"{__name__}.morph_agnostic_env:NovaMorphAgnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.morph_squat_env_cfg:NovaMorphAgnosticSquatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaMorphAgnosticSquatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walking-Nova-MorphAgnostic-Squat-Play-v0",
    entry_point=f"{__name__}.morph_agnostic_env:NovaMorphAgnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.morph_squat_env_cfg:NovaMorphAgnosticSquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NovaMorphAgnosticSquatPPORunnerCfg",
    },
)
