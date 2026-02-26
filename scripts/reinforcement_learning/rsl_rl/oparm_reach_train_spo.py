"""
Custom training script for Isaac-Reach-OpenArm-Bi-v0 with SPO (Simple Policy Optimization).

SPO replaces PPO's clipped surrogate loss with a quadratic penalty on the
probability ratio, as described in:
    L_p = - 1/N * sum{ ratio * A_hat - |A_hat| / (2*epsilon) * (ratio - 1)^2 }

Everything else (GAE, value loss, entropy bonus, adaptive LR) remains the same.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/oparm_reach_train_spo.py
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

# ==============================================================================
# Step 1: Configure environment and agent
# ==============================================================================
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 4096
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

agent_cfg = OpenArmReachPPORunnerCfg()
agent_cfg.max_iterations = 1500  # Train longer than default (550) for SPO

# LR schedule: linear decay from initial_lr → final_lr over lr_decay_iters,
# then hold at final_lr for the remaining iterations.
initial_lr = 1e-2   # same as PPO default
final_lr = 3e-4     # SPO paper default — safe for SPO's quadratic penalty
lr_decay_iters = 500  # match PPO's default training length for decay phase

# ==============================================================================
# Step 2: Create environment and runner
#
# We reuse OnPolicyRunner to build the ActorCritic network, optimizer,
# and RolloutStorage. We just replace the policy loss with SPO.
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
    name=f"spo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "SPO",
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
# ==============================================================================
alg = runner.alg            # PPO instance (we reuse everything except the loss)
policy = alg.policy         # ActorCritic(nn.Module)
optimizer = alg.optimizer   # Adam optimizer
device = agent_cfg.device

num_steps_per_env = agent_cfg.num_steps_per_env
max_iterations = agent_cfg.max_iterations

# SPO uses epsilon (same as PPO's clip_param) for the quadratic penalty
epsilon = 0.2 # alg.clip_param  # typically 0.2


def save_checkpoint(path: str, iteration: int):
    """Save model checkpoint (same format as OnPolicyRunner.save)."""
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": iteration,
        "infos": None,
    }, path)

# ==============================================================================
# Step 4: Custom training loop with SPO loss
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
    # Phase 1: Collect rollouts (identical to PPO)
    # ==================================================================
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
            alg.process_env_step(obs, rewards, dones, extras)

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
        # Unlike PPO, SPO uses per-mini-batch advantage normalization
        # (matching the official SPO repo). We call storage.compute_returns
        # directly with normalize_advantage=False, then normalize per
        # mini-batch inside the training loop below.
        last_values = policy.evaluate(obs).detach()
        alg.storage.compute_returns(
            last_values, alg.gamma, alg.lam, normalize_advantage=False
        )

    # ==================================================================
    # Phase 2: SPO update (with gradients)
    #
    # SPO policy loss (the ONLY difference from PPO):
    #
    #   L_p = -1/N * sum{ ratio * A_hat
    #                      - |A_hat| / (2 * epsilon) * (ratio - 1)^2 }
    #
    # where ratio = pi_new(a|s) / pi_old(a|s)
    #
    # Intuition: The first term is the standard policy gradient.
    # The second term is a quadratic penalty that discourages the ratio
    # from deviating from 1, weighted by |A_hat| — bigger advantages
    # get a bigger penalty, keeping updates conservative.
    #
    # NOTE: Unlike PPO (which uses global advantage normalization),
    # SPO uses per-mini-batch advantage normalization, matching the
    # official SPO repo (https://github.com/MyRepositories-hub/Simple-Policy-Optimization).
    # ==================================================================
    learn_start = time.time()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0

    generator = alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)

    for (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hidden_states_batch,
        masks_batch,
    ) in generator:

        # --- Forward pass with current policy ---
        policy.act(obs_batch)
        actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)
        value_batch = policy.evaluate(obs_batch)
        entropy_batch = policy.entropy

        # ==============================================================
        # *** SPO POLICY LOSS (replaces PPO clipped surrogate) ***
        #
        # PPO:  L = max(ratio * A, clip(ratio, 1-e, 1+e) * A)
        # SPO:  L = -(ratio * A - |A| / (2*e) * (ratio - 1)^2)
        #
        # The SPO loss is simpler — no clipping, just a quadratic
        # penalty on how far the ratio deviates from 1.
        # ==============================================================
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
        advantages = advantages_batch.squeeze()

        # Per-mini-batch advantage normalization (matching official SPO repo)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # SPO: ratio * A - |A| / (2*epsilon) * (ratio - 1)^2
        # We negate because we minimize the loss (maximize the objective)
        surrogate_loss = -(
            ratio * advantages
            - advantages.abs() / (2.0 * epsilon) * (ratio - 1.0).pow(2)
        ).mean()

        # --- Value function loss (same as PPO) ---
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
        # *** TOTAL LOSS ***
        #
        # Same structure as PPO:
        #   L = L_policy + c1 * L_value - c2 * L_entropy
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

    # --- LR schedule: linear decay initial_lr → final_lr, then hold ---
    if iteration < lr_decay_iters:
        frac = 1.0 - iteration / lr_decay_iters
        lr_now = final_lr + frac * (initial_lr - final_lr)
    else:
        lr_now = final_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now

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
        f"SPO_Loss: {mean_surrogate_loss:.4f} | "
        f"Entropy: {mean_entropy:.4f} | "
        f"NoiseStd: {noise_std:.3f} | "
        f"LR: {lr_now:.1e} | "
        f"FPS: {fps:,.0f}"
    )

    # --- Wandb logging ---
    wandb.log({
        "reward/mean": mean_reward,
        "reward/episode_length": mean_ep_len,
        "loss/value": mean_value_loss,
        "loss/spo_surrogate": mean_surrogate_loss,
        "loss/entropy": mean_entropy,
        "policy/noise_std": noise_std,
        "policy/learning_rate": lr_now,
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
    description=f"SPO policy trained for {max_iterations} iterations",
)
artifact.add_file(final_path)
wandb.log_artifact(artifact)

# ==============================================================================
# Clean up
# ==============================================================================
env.close()
wandb.finish()
simulation_app.close()
