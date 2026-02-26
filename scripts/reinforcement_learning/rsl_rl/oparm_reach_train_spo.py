"""
Custom training script for Isaac-Reach-OpenArm-Bi-v0 with SPO (Simple Policy Optimization).

This version uses a fully custom nn.Module (no RSL-RL ActorCritic dependency).
The network architecture is defined explicitly in PyTorch, making it easy to
swap in any actor/critic architecture you want.

SPO replaces PPO's clipped surrogate loss with a quadratic penalty:
    L_p = - 1/N * sum{ ratio * A - |A| / (2*epsilon) * (ratio - 1)^2 }

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
# Imports
# ==============================================================================
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from tensordict import TensorDict
from torch.distributions import Normal
from rsl_rl.storage import RolloutStorage

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# Custom Actor-Critic Network
#
# This is a plain PyTorch nn.Module. You can modify the architecture freely:
# - Change hidden dims, activation, number of layers
# - Add normalization layers, residual connections, etc.
# - Use different architectures for actor vs critic
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight initialization (matching CleanRL / official SPO repo)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Gaussian actor-critic with separate actor and critic networks."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [64, 64],
        critic_hidden_dims: list[int] = [64, 64],
        activation: str = "tanh",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        # Activation function
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
            "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # --- Actor network: obs → action mean ---
        # Orthogonal init: sqrt(2) for hidden layers, 0.01 for output (small initial actions)
        actor_layers = []
        in_dim = num_obs
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        actor_layers.append(nn.Tanh())  # Assuming actions are in [-1, 1]; remove if not needed
        self.actor = nn.Sequential(*actor_layers)

        # --- Critic network: obs → V(s) ---
        # Orthogonal init: sqrt(2) for hidden layers, 1.0 for output
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # --- Action noise: learnable log_std ---
        # Official SPO repo: nn.Parameter(torch.zeros(1, action_dim))
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))

        # Distribution (populated by forward_actor / act)
        self._distribution: Normal | None = None

        # Disable distribution validation for speed
        Normal.set_default_validate_args(False)

    @property
    def action_std(self) -> torch.Tensor:
        return self.log_std.exp()

    @property
    def action_mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    def forward_actor(self, obs: torch.Tensor) -> Normal:
        """Compute action distribution from observations."""
        mean = self.actor(obs)
        # Clamp log_std to prevent std explosion or collapse
        log_std = self.log_std.clamp(-20.0, 2.0)
        std = log_std.exp().expand_as(mean)
        self._distribution = Normal(mean, std)
        return self._distribution

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from observations."""
        return self.critic(obs)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample action from the current policy."""
        dist = self.forward_actor(obs)
        return dist.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) for evaluation."""
        return self.actor(obs)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute V(s)."""
        return self.forward_critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability of actions under the current distribution."""
        return self._distribution.log_prob(actions).sum(dim=-1)


def _flatten_obs(obs: TensorDict) -> torch.Tensor:
    """Extract the flat observation tensor from a TensorDict.

    RslRlVecEnvWrapper already concatenates all observation terms into a
    single tensor under the "policy" key. So obs["policy"] is already
    a [num_envs, obs_dim] tensor — no further flattening needed.
    """
    if "policy" in obs.keys():
        policy_obs = obs["policy"]
        # RslRlVecEnvWrapper already flattens → policy_obs is a Tensor
        if isinstance(policy_obs, torch.Tensor):
            return policy_obs
        # Fallback: if it's still a TensorDict, concatenate
        tensors = [policy_obs[k] for k in sorted(policy_obs.keys())]
        return torch.cat(tensors, dim=-1)
    # No "policy" key — concatenate all top-level tensors
    tensors = []
    for key in sorted(obs.keys()):
        t = obs[key]
        if isinstance(t, TensorDict):
            for sub_key in sorted(t.keys()):
                tensors.append(t[sub_key])
        else:
            tensors.append(t)
    return torch.cat(tensors, dim=-1)


def _row_is_finite(x: torch.Tensor) -> torch.Tensor:
    """Return per-env finite mask for tensor shaped [num_envs, ...]."""
    if not torch.is_floating_point(x):
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    return torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=-1)


def _sanitize_policy_obs(obs: TensorDict) -> tuple[TensorDict, torch.Tensor]:
    """Sanitize policy observations and return invalid env mask."""
    obs_flat = _flatten_obs(obs)
    invalid_env = ~_row_is_finite(obs_flat)

    if invalid_env.any() and "policy" in obs.keys() and isinstance(obs["policy"], torch.Tensor):
        policy_obs = torch.nan_to_num(obs["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        policy_obs[invalid_env] = 0.0
        obs["policy"] = policy_obs

    return obs, invalid_env


def _sanitize_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sanitize sampled actions and return invalid env mask."""
    invalid_env = ~_row_is_finite(actions)
    if invalid_env.any():
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions[invalid_env] = 0.0
    return actions, invalid_env


def _sanitize_rewards(rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sanitize rewards and return invalid env mask."""
    invalid_env = ~_row_is_finite(rewards)
    if invalid_env.any():
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        rewards[invalid_env] = 0.0
    return rewards, invalid_env


# ==============================================================================
# Configuration
# ==============================================================================
# Environment
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 4096
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

# Use relative joint position actions (delta command) for stability.
# With tanh actor head, raw action is approximately in [-1, 1], so the effective
# joint delta per step is scale * action.
env_cfg.actions.left_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_left_joint.*"],
    scale=0.3,
    use_zero_offset=True,
    alpha=0.3,
)
env_cfg.actions.right_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_right_joint.*"],
    scale=0.3,
    use_zero_offset=True,
    alpha=0.3,
)
device = "cuda:0"

# Training
max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# SPO hyperparameters (matching official repo defaults)
epsilon = 0.2             # quadratic penalty coefficient
gamma = 0.99              # discount factor
lam = 0.95                # GAE lambda
value_loss_coef = 1.0     # critic loss weight (official repo: c_1=0.5)
entropy_coef = 0.001      # entropy bonus (official MuJoCo repo: c_2=0.0)
max_grad_norm = 1.0       # gradient clipping (official repo: 0.5)
num_learning_epochs = 8   # SGD epochs per iteration (official repo: update_epochs=10)
num_mini_batches = 4      # mini-batches per epoch
use_clipped_value_loss = True
clip_param = 0.2          # value loss clip range

# LR schedule (official repo: lr=3e-4, linear decay to 0)
initial_lr = 1e-2
final_lr = 3e-4
lr_decay_iters = 500     # decay over full training (same as max_iterations)

# Network architecture — change these freely!
actor_hidden_dims = [128, 128]
critic_hidden_dims = [64, 64]
activation = "elu"       # official SPO repo uses Tanh
init_noise_std = 1.0

# ==============================================================================
# Step 1: Create environment
# ==============================================================================
env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

log_dir = os.path.join("logs", "rsl_rl", "openarm_bi_reach", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

# ==============================================================================
# Step 2: Create network, optimizer, storage
# ==============================================================================
# Figure out observation dimension by querying the environment
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs = obs_flat.shape[-1]
num_actions = env.num_actions

print(f"Observation dim: {num_obs}, Action dim: {num_actions}")

# Create our custom network
policy = ActorCritic(
    num_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation=activation,
    init_noise_std=init_noise_std,
).to(device)

print(f"Actor:  {policy.actor}")
print(f"Critic: {policy.critic}")
print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

# Optimizer
optimizer = torch.optim.AdamW(policy.parameters(), lr=initial_lr)

# Rollout storage (reuse RSL-RL's GAE computation)
storage = RolloutStorage(
    training_type="rl",
    num_envs=env.num_envs,
    num_transitions_per_env=num_steps_per_env,
    obs=obs_td,
    actions_shape=(num_actions,),
    device=device,
)
transition = RolloutStorage.Transition()

# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
wandb.init(
    project="isaaclab-openarm-reach",
    name=f"spo_custom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "SPO",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "epsilon": epsilon,
        "gamma": gamma,
        "lam": lam,
        "initial_lr": initial_lr,
        "final_lr": final_lr,
        "lr_decay_iters": lr_decay_iters,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "activation": activation,
        "init_noise_std": init_noise_std,
        "num_learning_epochs": num_learning_epochs,
        "num_mini_batches": num_mini_batches,
        "max_grad_norm": max_grad_norm,
        "value_loss_coef": value_loss_coef,
        "entropy_coef": entropy_coef,
    },
    dir=log_dir,
    save_code=True,
)


def save_checkpoint(path: str, iteration: int):
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": iteration,
    }, path)


# ==============================================================================
# Step 4: Training loop
# ==============================================================================
# Randomize initial episode lengths
env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

obs_td = env.get_observations().to(device)
policy.train()

rewbuffer = deque(maxlen=100)
lenbuffer = deque(maxlen=100)
cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)

start_time = time.time()

for iteration in range(max_iterations):
    iter_start = time.time()
    forced_done_count = 0

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            obs_td, invalid_obs_before = _sanitize_policy_obs(obs_td)
            obs_flat = _flatten_obs(obs_td)

            # Forward pass through our custom network
            actions = policy.act(obs_flat)
            actions, invalid_actions = _sanitize_actions(actions)
            values = policy.evaluate(obs_flat)
            log_probs = policy.get_actions_log_prob(actions)
            invalid_now = invalid_obs_before | invalid_actions

            # Store transition (RolloutStorage interface)
            transition.observations = obs_td
            transition.actions = actions.detach()
            transition.values = values.detach()
            transition.actions_log_prob = log_probs.detach()
            transition.action_mean = policy.action_mean.detach()
            transition.action_sigma = policy.action_std.expand(env.num_envs, -1).detach()

            # Step environment
            obs_td, rewards, dones, extras = env.step(actions.detach().to(env.device))
            obs_td = obs_td.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            obs_td, invalid_obs_after = _sanitize_policy_obs(obs_td)
            rewards, invalid_rewards = _sanitize_rewards(rewards)
            invalid_env = invalid_now | invalid_obs_after | invalid_rewards
            if invalid_env.any():
                forced_done_count += int(invalid_env.sum().item())
                if dones.ndim > 1:
                    dones[invalid_env] = 1
                else:
                    dones[invalid_env] = True
                rewards[invalid_env] = 0.0

            # Bootstrap on time-outs (so truncation != termination)
            transition.rewards = rewards.clone()
            transition.dones = dones
            if "time_outs" in extras:
                transition.rewards += gamma * torch.squeeze(
                    transition.values * extras["time_outs"].unsqueeze(1).to(device), 1
                )

            storage.add_transitions(transition)
            transition.clear()

            # --- Logging ---
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        collection_time = time.time() - iter_start

        # Compute returns (GAE) — normalize per mini-batch, not globally
        obs_flat = _flatten_obs(obs_td)
        last_values = policy.evaluate(obs_flat).detach()
        storage.compute_returns(last_values, gamma, lam, normalize_advantage=False)

    # ==================================================================
    # Phase 2: SPO update
    #
    # SPO loss (replaces PPO's clipped surrogate):
    #   L = -(ratio * A - |A| / (2*eps) * (ratio - 1)^2)
    #
    # Per-mini-batch advantage normalization (matching official SPO repo).
    # ==================================================================
    learn_start = time.time()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    dropped_minibatch_samples = 0

    generator = storage.mini_batch_generator(num_mini_batches, num_learning_epochs)

    for (
        obs_batch,           # TensorDict
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
        obs_flat_batch = _flatten_obs(obs_batch)

        valid_rows = (
            _row_is_finite(obs_flat_batch)
            & _row_is_finite(actions_batch)
            & _row_is_finite(target_values_batch)
            & _row_is_finite(advantages_batch)
            & _row_is_finite(returns_batch)
            & _row_is_finite(old_actions_log_prob_batch)
        )

        if not torch.any(valid_rows):
            dropped_minibatch_samples += int(valid_rows.numel())
            continue

        dropped_minibatch_samples += int((~valid_rows).sum().item())
        obs_flat_batch = obs_flat_batch[valid_rows]
        actions_batch = actions_batch[valid_rows]
        target_values_batch = target_values_batch[valid_rows]
        advantages_batch = advantages_batch[valid_rows]
        returns_batch = returns_batch[valid_rows]
        old_actions_log_prob_batch = old_actions_log_prob_batch[valid_rows]

        # Forward pass through our custom network
        policy.forward_actor(obs_flat_batch)
        actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)
        value_batch = policy.evaluate(obs_flat_batch)
        entropy_batch = policy.entropy

        if not (
            torch.isfinite(actions_log_prob_batch).all()
            and torch.isfinite(value_batch).all()
            and torch.isfinite(entropy_batch).all()
        ):
            continue

        # --- SPO policy loss ---
        log_ratio = actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1)
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        advantages = advantages_batch.squeeze(-1)

        # Per-mini-batch advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        surrogate_loss = -(
            ratio * advantages
            - advantages.abs() / (2.0 * epsilon) * (ratio - 1.0).pow(2)
        ).mean()

        # --- Value loss (clipped) ---
        if use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -clip_param, clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # --- Total loss ---
        loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()

        if not torch.isfinite(loss):
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        with torch.no_grad():
            policy.log_std.data.nan_to_num_(nan=0.0, posinf=2.0, neginf=-20.0)
            policy.log_std.data.clamp_(-20.0, 2.0)

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy_batch.mean().item()

    num_updates = num_learning_epochs * num_mini_batches
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

    storage.clear()
    learn_time = time.time() - learn_start
    iter_time = time.time() - iter_start

    # ==================================================================
    # Phase 3: Logging
    # ==================================================================
    total_steps = (iteration + 1) * num_steps_per_env * env.num_envs
    fps = num_steps_per_env * env.num_envs / (collection_time + learn_time)

    mean_reward = sum(rewbuffer) / len(rewbuffer) if rewbuffer else 0.0
    mean_ep_len = sum(lenbuffer) / len(lenbuffer) if lenbuffer else 0.0
    noise_std = policy.action_std.mean().item()

    print(
        f"Iter {iteration:4d}/{max_iterations} | "
        f"Reward: {mean_reward:6.2f} | "
        f"EpLen: {mean_ep_len:6.0f} | "
        f"Value_Loss: {mean_value_loss:.4f} | "
        f"SPO_Loss: {mean_surrogate_loss:.4f} | "
        f"Entropy: {mean_entropy:.4f} | "
        f"NoiseStd: {noise_std:.3f} | "
        f"LR: {lr_now:.1e} | "
        f"FPS: {fps:,.0f} | "
        f"ForcedDone: {forced_done_count} | "
        f"Dropped: {dropped_minibatch_samples}"
    )

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
        "debug/forced_done_count": forced_done_count,
        "debug/dropped_minibatch_samples": dropped_minibatch_samples,
        "train/total_steps": total_steps,
    }, step=iteration)

    # ==================================================================
    # Phase 4: Checkpoints
    # ==================================================================
    if iteration % save_interval == 0:
        save_path = os.path.join(log_dir, f"model_{iteration}.pt")
        save_checkpoint(save_path, iteration)
        print(f"  -> Saved checkpoint: {save_path}")

# Save final model
final_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
save_checkpoint(final_path, max_iterations)
print(f"Final model saved: {final_path}")
print(f"Total training time: {time.time() - start_time:.1f}s")

# Wandb artifact
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
