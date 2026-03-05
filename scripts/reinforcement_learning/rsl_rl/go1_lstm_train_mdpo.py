"""
LSTM baseline control experiment for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO.

This uses the EXACT same pipeline, env config, rewards, and MDPO hyperparameters
as go1_cpg_train_mdpo.py — but replaces the CPG-Reflex policy with a standard
LSTM ActorCriticRecurrent from rsl_rl.

Purpose: verify the training pipeline works. If the LSTM walks but CPG doesn't,
the issue is CPG-specific, not pipeline-related.

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_lstm_train_mdpo.py
"""

# ==============================================================================
# Step 0: Launch Isaac Sim (must happen before any other Isaac imports)
# ==============================================================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ==============================================================================
# Imports
# ==============================================================================
import math
import os
import time
from collections import deque
from datetime import datetime
from tqdm import tqdm

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import MDPO
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.utils import unpad_trajectories

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# Utilities
# ==============================================================================
def _flatten_obs(obs: TensorDict) -> torch.Tensor:
    if "policy" in obs.keys():
        policy_obs = obs["policy"]
        if isinstance(policy_obs, torch.Tensor):
            return policy_obs
        tensors = [policy_obs[k] for k in sorted(policy_obs.keys())]
        return torch.cat(tensors, dim=-1)
    tensors = []
    for key in sorted(obs.keys()):
        t = obs[key]
        if isinstance(t, TensorDict):
            for sub_key in sorted(t.keys()):
                tensors.append(t[sub_key])
        else:
            tensors.append(t)
    return torch.cat(tensors, dim=-1)


# ==============================================================================
# Configuration — IDENTICAL to go1_cpg_train_mdpo.py
# ==============================================================================
env_cfg = UnitreeGo1FlatEnvCfg()
env_cfg.scene.num_envs = 4096 * 2
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"
env_cfg.actions.joint_pos.scale = 0.5
device = "cuda:0"

# --- Reward overrides (SAME as CPG experiment) ---
# env_cfg.rewards.action_rate_l2 = None
env_cfg.rewards.flat_orientation_l2.weight = -5.0
env_cfg.rewards.lin_vel_z_l2.weight = -0.5
env_cfg.rewards.track_lin_vel_xy_exp.weight = 6.0 # 3.0
env_cfg.rewards.track_ang_vel_z_exp.weight = 5.0 # 2.0
env_cfg.rewards.feet_air_time.weight = 0.5

max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# MDPO (SAME as CPG experiment)
mdpo_cfg = dict(
    num_learning_epochs=4,
    num_mini_batches=2,
    clip_param=0.2,
    value_clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    distill_coef=0.02,
    value_loss_coef=1.0,
    entropy_coef=0.001,
    learning_rate=3e-3,
    min_learning_rate=3e-5,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="exponential",
    use_muon=False,
)

# LSTM network config — roughly same capacity as CPG policy
lstm_hidden_size = 128
actor_hidden_dims = [128, 128]
critic_hidden_dims = [256, 128]
activation = "elu"
init_noise_std = 1.0


# ==============================================================================
# Step 1: Create environment
# ==============================================================================
env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

log_dir = os.path.join("logs", "rsl_rl", "go1_lstm_mdpo", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)


# ==============================================================================
# Step 2: Create LSTM networks and MDPO algorithm
# ==============================================================================
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs = obs_flat.shape[-1]  # 48
num_actions = env.num_actions  # 12

print(f"[LSTM Baseline] Obs dim: {num_obs}, Action dim: {num_actions}")

policy_kwargs = dict(
    num_actor_obs=num_obs,
    num_critic_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation=activation,
    rnn_type="lstm",
    rnn_hidden_size=lstm_hidden_size,
    rnn_num_layers=1,
    init_noise_std=init_noise_std,
)
actor_critic_1 = ActorCriticRecurrent(**policy_kwargs).to(device)
actor_critic_2 = ActorCriticRecurrent(**policy_kwargs).to(device)

mdpo = MDPO(actor_critic_1, actor_critic_2, device=device, **mdpo_cfg)

mdpo.init_storage(
    num_envs=env.num_envs,
    num_transitions_per_env=num_steps_per_env,
    actor_obs_shape=[num_obs],
    critic_obs_shape=[num_obs],
    action_shape=[num_actions],
)

total_params = sum(p.numel() for p in actor_critic_1.parameters()) + sum(p.numel() for p in actor_critic_2.parameters())
print(f"Policy: {actor_critic_1.__class__.__name__}")
print(f"  Actor MLP params: {sum(p.numel() for p in actor_critic_1.actor.parameters())}")
print(f"  Critic MLP params: {sum(p.numel() for p in actor_critic_1.critic.parameters())}")
print(f"  LSTM actor params: {sum(p.numel() for p in actor_critic_1.memory_a.parameters())}")
print(f"  LSTM critic params: {sum(p.numel() for p in actor_critic_1.memory_c.parameters())}")
print(f"Total parameters (both policies): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")


# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
wandb.init(
    project="isaaclab-go1-velocity",
    name=f"mdpo_lstm_baseline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": "LSTM-Baseline",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "num_obs": num_obs,
        "num_actions": num_actions,
        "lstm_hidden_size": lstm_hidden_size,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "activation": activation,
        **mdpo_cfg,
    },
)


# ==============================================================================
# Step 4: Training loop
# ==============================================================================
rewbuffer = deque(maxlen=100)
lenbuffer = deque(maxlen=100)
cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)

# Staggered env resets
env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
obs_td = env.get_observations().to(device)

print(f"\nStarting MDPO + LSTM baseline training for {max_iterations} iterations...")
print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
print(f"  LSTM: hidden_size={lstm_hidden_size}, Actor MLP: {actor_hidden_dims}, Critic: {critic_hidden_dims}")

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    mdpo.train_mode()
    # Record start positions to measure displacement
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()  # (B, 2)
    with torch.no_grad():
        for step in tqdm(range(num_steps_per_env), desc=f"[{iteration}] collect", leave=False):
            obs_flat = _flatten_obs(obs_td)  # (B, 48)
            actions = mdpo.act(obs_flat, obs_flat)

            obs_td, rewards, dones, extras = env.step(actions.detach().to(env.device))
            obs_td = obs_td.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            mdpo.process_env_step(rewards, dones, extras)

            # Episode tracking
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

    # Measure displacement over rollout
    end_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]  # (B, 2)
    displacement = (end_pos - start_pos).norm(dim=-1)  # (B,) meters
    mean_displacement = displacement.mean().item()

    collection_time = time.time() - iter_start

    # ==================================================================
    # Phase 2: Compute returns and update
    # ==================================================================
    learn_start = time.time()
    obs_flat = _flatten_obs(obs_td)
    mdpo.compute_returns(obs_flat)
    mdpo.update_dropout_masks()
    mean_value_loss, mean_surrogate_loss, mean_kl_div = mdpo.update(iteration, max_iterations)
    mdpo.reset_dropout_masks()
    learn_time = time.time() - learn_start
    total_time = time.time() - iter_start

    # ==================================================================
    # Logging
    # ==================================================================
    fps = int(env.num_envs * num_steps_per_env / total_time)
    mean_reward = np.mean(rewbuffer) if rewbuffer else 0.0
    mean_ep_len = np.mean(lenbuffer) if lenbuffer else 0.0

    # Extract per-term reward info from extras (populated on env resets)
    log_extras = extras.get("log", {}) if isinstance(extras, dict) else {}

    log_dict = {
        "Loss/value": mean_value_loss,
        "Loss/surrogate": mean_surrogate_loss,
        "Loss/distill_kl": mean_kl_div,
        "Loss/learning_rate": mdpo.learning_rate,
        "Perf/fps": fps,
        "Perf/collection_time": collection_time,
        "Perf/learn_time": learn_time,
        "Train/mean_reward": mean_reward,
        "Train/mean_episode_length": mean_ep_len,
        "Train/mean_displacement": mean_displacement,
    }

    # Log all individual reward terms from the environment
    for key, value in log_extras.items():
        if key.startswith("Episode_Reward/"):
            log_dict[key] = value.item() if isinstance(value, torch.Tensor) else value

    if iteration % 10 == 0:
        log_dict["LSTM/action_std_mean"] = actor_critic_1.action_std.mean().item()

    wandb.log(log_dict, step=iteration)

    if iteration % 10 == 0:
        # Extract specific reward terms for printing
        r_base_height = log_extras.get("Episode_Reward/base_height", 0.0)
        r_foot_clear = log_extras.get("Episode_Reward/foot_clearance", 0.0)
        r_flat_orient = log_extras.get("Episode_Reward/flat_orientation_l2", 0.0)
        if isinstance(r_base_height, torch.Tensor):
            r_base_height = r_base_height.item()
        if isinstance(r_foot_clear, torch.Tensor):
            r_foot_clear = r_foot_clear.item()
        if isinstance(r_flat_orient, torch.Tensor):
            r_flat_orient = r_flat_orient.item()

        print(
            f"[{iteration:4d}/{max_iterations}]  "
            f"reward={mean_reward:7.2f}  "
            f"ep_len={mean_ep_len:5.1f}  "
            f"v_loss={mean_value_loss:.4f}  "
            f"surr={mean_surrogate_loss:.4f}  "
            f"kl={mean_kl_div:.4f}  "
            f"lr={mdpo.learning_rate:.2e}  "
            f"fps={fps}  "
            f"disp={mean_displacement:.3f}m  "
            f"collect={collection_time:.2f}s  learn={learn_time:.2f}s"
        )
        print(
            f"  [LSTM] std_mean={actor_critic_1.action_std.mean().item():.4f}  "
            f"base_height={r_base_height:.4f}  foot_clearance={r_foot_clear:.4f}  "
            f"flat_orient={r_flat_orient:.4f}"
        )

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        ckpt_data = {
            "iter": iteration + 1,
            "architecture": "lstm_baseline",
            "model_1_state_dict": actor_critic_1.state_dict(),
            "model_2_state_dict": actor_critic_2.state_dict(),
            "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
            "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
        }
        torch.save(ckpt_data, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

# ==============================================================================
# Cleanup
# ==============================================================================
wandb.finish()
env.close()
simulation_app.close()
print("Training complete.")
