# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..mdp.symmetry import compute_symmetric_states


@configclass
class NovaWalkingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Long run: ~1.02 s/iter at 4096 envs headless (symmetry augmentation, self-collisions, filtered contact sensors)
    # -> ~4.2 h for 15000 iterations; slower while a viewer / remote-desktop session shares the GPU.
    max_iterations = 15000
    save_interval = 250
    experiment_name = "nova_walking"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # History: 0.005 -> 0.001 after run 1's std climbed 1.0 -> 8.25 (effort term dead); 0.001 -> 0.003 after run 2
        # collapsed to standing still at std 0.46 (too little exploration to find a gait); 0.003 -> 0.0025 for run 4.
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Left-right mirror augmentation (rsl_rl built-in): every PPO minibatch is doubled with its mirror image, so the
        # policy cannot prefer one stance leg (run 3 converged to a population-wide right-leg hop). Mirror loss off.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=compute_symmetric_states
        ),
    )
