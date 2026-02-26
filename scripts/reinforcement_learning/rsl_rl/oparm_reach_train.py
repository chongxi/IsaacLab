"""
Custom training script for Isaac-Reach-OpenArm-Bi-v0 with RSL-RL (PPO).

The training loop is fully exposed so you can modify the loss function,
add custom logging, or change the update logic.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/custom_train.py
"""

# ==============================================================================
# Step 0: Launch Isaac Sim (must happen before any other Isaac imports)
# ==============================================================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ==============================================================================
# Now we can import everything else
# ==============================================================================
import os
import time
from collections import deque
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.agents.rsl_rl_ppo_cfg import (
    OpenArmReachPPORunnerCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ==============================================================================
# Step 1: Configure environment and agent
# ==============================================================================
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 4096
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

agent_cfg = OpenArmReachPPORunnerCfg()

# ==============================================================================
# Step 2: Create environment and runner
#
# We still use OnPolicyRunner to build the PPO algorithm, ActorCritic network,
# and RolloutStorage. We just won't call runner.learn() — instead we run
# the training loop ourselves.
# ==============================================================================
env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

# ==============================================================================
# Step 2.5: Initialize Weights & Biases
# ==============================================================================
wandb.init(
    project="isaaclab-openarm-reach",
    name=f"ppo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "seed": env_cfg.seed,
        "device": agent_cfg.device,
        "clip_actions": agent_cfg.clip_actions,
        **agent_cfg.to_dict(),
    },
    dir=log_dir,
    save_code=True,
)

# ==============================================================================
# Step 3: Extract the components we need from the runner
#
# After OnPolicyRunner.__init__(), the following are ready:
#   runner.alg          — PPO algorithm instance
#   runner.alg.policy   — ActorCritic neural network
#   runner.alg.optimizer — Adam optimizer for the policy
#   runner.alg.storage  — RolloutStorage (created on first learn call)
# ==============================================================================
alg = runner.alg            # PPO instance
policy = alg.policy         # ActorCritic(nn.Module)
optimizer = alg.optimizer   # Adam optimizer
device = agent_cfg.device

num_steps_per_env = agent_cfg.num_steps_per_env
max_iterations = agent_cfg.max_iterations


def save_checkpoint(path: str, iteration: int):
    """Save model checkpoint (same format as OnPolicyRunner.save)."""
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": iteration,
        "infos": None,
    }, path)

# ==============================================================================
# Step 4: Custom training loop
#
# This is the same logic as OnPolicyRunner.learn() + PPO.update(),
# but fully exposed so you can modify anything.
# ==============================================================================

# Randomize initial episode lengths (for exploration diversity)
env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

# Get initial observations
obs = env.get_observations().to(device)
policy.train()

# Logging buffers
rewbuffer = deque(maxlen=100)
lenbuffer = deque(maxlen=100)
cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)

start_time = time.time()

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts (no gradients needed)
    # ==================================================================
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            # --- Actor forward pass ---
            actions = alg.act(obs)
            # act() internally stores: transition.actions, .values, .actions_log_prob,
            #                          .action_mean, .action_sigma, .observations

            # --- Environment step ---
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)

            # --- Store transition ---
            alg.process_env_step(obs, rewards, dones, extras)
            # process_env_step() internally:
            #   1. Updates observation normalizers
            #   2. Bootstraps rewards on timeouts (value * gamma * timeout_mask)
            #   3. Adds transition to RolloutStorage
            #   4. Resets hidden states for done envs (if recurrent)

            # --- Logging ---
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        collection_time = time.time() - iter_start

        # --- Compute returns and advantages (GAE) ---
        alg.compute_returns(obs)
        # compute_returns() internally:
        #   1. Evaluates V(last_obs) for bootstrapping
        #   2. Computes GAE advantages and returns in RolloutStorage

    # ==================================================================
    # Phase 2: PPO update (with gradients)
    #
    # *** THIS IS WHERE YOU MODIFY THE LOSS ***
    #
    # The original PPO loss is:
    #   loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
    #
    # You can add custom terms, change coefficients, or replace losses entirely.
    # ==================================================================
    learn_start = time.time()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0

    # Mini-batch generator yields data from the rollout storage
    generator = alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)

    for (
        obs_batch,            # TensorDict of observations
        actions_batch,        # [batch, num_actions] — actions taken
        target_values_batch,  # [batch, 1] — V(s) at collection time
        advantages_batch,     # [batch, 1] — GAE advantages
        returns_batch,        # [batch, 1] — discounted returns
        old_actions_log_prob_batch,  # [batch, 1] — log π_old(a|s)
        old_mu_batch,         # [batch, num_actions] — old action mean
        old_sigma_batch,      # [batch, num_actions] — old action std
        hidden_states_batch,  # tuple (actor_hidden, critic_hidden) — for recurrent only
        masks_batch,          # [batch] — masks for recurrent
    ) in generator:

        # --- Forward pass with current policy ---
        policy.act(obs_batch)
        actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)  # log π_new(a|s)
        value_batch = policy.evaluate(obs_batch)                            # V_new(s)
        entropy_batch = policy.entropy                                      # H(π_new)
        mu_batch = policy.action_mean
        sigma_batch = policy.action_std

        # --- Adaptive learning rate (KL-based) ---
        if alg.desired_kl is not None and alg.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1e-5)
                    + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                    / (2.0 * sigma_batch.square())
                    - 0.5,
                    dim=-1,
                )
                kl_mean = kl.mean()

                if kl_mean > alg.desired_kl * 2.0:
                    alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
                elif kl_mean < alg.desired_kl / 2.0 and kl_mean > 0.0:
                    alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = alg.learning_rate

        # --- Surrogate (policy) loss ---
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
        surrogate = -advantages_batch.squeeze() * ratio
        surrogate_clipped = -advantages_batch.squeeze() * torch.clamp(
            ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # --- Value function loss ---
        if alg.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -alg.clip_param, alg.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # ==============================================================
        # *** TOTAL LOSS — MODIFY HERE ***
        #
        # Original PPO:
        #   loss = surrogate_loss + value_coef * value_loss - entropy_coef * entropy
        #
        # Examples of modifications:
        #   - Add L2 regularization:
        #       l2_reg = sum(p.pow(2).sum() for p in policy.parameters())
        #       loss += 1e-4 * l2_reg
        #   - Add action smoothness penalty:
        #       action_penalty = actions_batch.diff(dim=0).pow(2).mean()
        #       loss += 0.01 * action_penalty
        #   - Change value loss coefficient:
        #       loss = surrogate_loss + 0.5 * value_loss - 0.01 * entropy
        # ==============================================================
        loss = (
            surrogate_loss
            + alg.value_loss_coef * value_loss
            - alg.entropy_coef * entropy_batch.mean()
        )

        # --- Backward pass and optimizer step ---
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), alg.max_grad_norm)
        optimizer.step()

        # --- Accumulate metrics ---
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy_batch.mean().item()

    # Average over all mini-batch updates
    num_updates = alg.num_learning_epochs * alg.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates

    # Clear rollout storage for next iteration
    alg.storage.clear()

    learn_time = time.time() - learn_start
    iter_time = time.time() - iter_start

    # ==================================================================
    # Phase 3: Logging
    # ==================================================================
    total_steps = (iteration + 1) * num_steps_per_env * env.num_envs
    fps = num_steps_per_env * env.num_envs / (collection_time + learn_time)

    if len(rewbuffer) > 0:
        mean_reward = sum(rewbuffer) / len(rewbuffer)
        mean_ep_len = sum(lenbuffer) / len(lenbuffer)
    else:
        mean_reward = 0.0
        mean_ep_len = 0.0

    # Get current noise std for logging
    if hasattr(policy, "std"):
        noise_std = policy.std.mean().item()
    elif hasattr(policy, "log_std"):
        noise_std = policy.log_std.exp().mean().item()
    else:
        noise_std = 0.0

    # --- Console logging ---
    print(
        f"Iter {iteration:4d}/{max_iterations} | "
        f"Reward: {mean_reward:6.2f} | "
        f"EpLen: {mean_ep_len:6.0f} | "
        f"Value_Loss: {mean_value_loss:.4f} | "
        f"Policy_Loss: {mean_surrogate_loss:.4f} | "
        f"Entropy: {mean_entropy:.4f} | "
        f"NoiseStd: {noise_std:.3f} | "
        f"LR: {alg.learning_rate:.1e} | "
        f"FPS: {fps:,.0f}"
    )

    # --- Wandb logging ---
    wandb.log({
        "reward/mean": mean_reward,
        "reward/episode_length": mean_ep_len,
        "loss/value": mean_value_loss,
        "loss/surrogate": mean_surrogate_loss,
        "loss/entropy": mean_entropy,
        "policy/noise_std": noise_std,
        "policy/learning_rate": alg.learning_rate,
        "perf/fps": fps,
        "perf/collection_time": collection_time,
        "perf/learn_time": learn_time,
        "perf/iter_time": iter_time,
        "train/total_steps": total_steps,
    }, step=iteration)

    # ==================================================================
    # Phase 4: Save checkpoints
    # ==================================================================
    if iteration % agent_cfg.save_interval == 0:
        save_path = os.path.join(log_dir, f"model_{iteration}.pt")
        save_checkpoint(save_path, iteration)
        print(f"  -> Saved checkpoint: {save_path}")

# Save final model
final_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
save_checkpoint(final_path, max_iterations)
print(f"Final model saved: {final_path}")
print(f"Total training time: {time.time() - start_time:.1f}s")

# Log final model as wandb artifact
artifact = wandb.Artifact(
    name=f"model-{wandb.run.id}",
    type="model",
    description=f"PPO policy trained for {max_iterations} iterations",
)
artifact.add_file(final_path)
wandb.log_artifact(artifact)

# ==============================================================================
# Clean up
# ==============================================================================
env.close()
wandb.finish()
simulation_app.close()
