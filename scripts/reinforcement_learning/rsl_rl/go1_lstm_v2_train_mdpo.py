"""
LSTM baseline control experiment for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO.

This uses the EXACT same pipeline, env config, rewards, and MDPO hyperparameters
as go1_cpg_train_mdpo.py — but replaces the CPG-Reflex policy with a standard
LSTM ActorCriticRecurrent from rsl_rl.

Purpose: verify the training pipeline works. If the LSTM walks but CPG doesn't,
the issue is CPG-specific, not pipeline-related.

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_lstm_v2_train_mdpo.py
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


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    if act_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Invalid activation function: {act_name}")


class ActorCritic(nn.Module):
    """In-file copy of rsl_rl ActorCritic used by MDPO."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


class Memory(nn.Module):
    """In-file copy of rsl_rl recurrent memory wrapper."""

    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        rnn_type = type.lower()
        if rnn_type == "gru":
            rnn_cls = nn.GRU
        elif rnn_type == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError(f"RNN type {type} not supported in this script. Use 'lstm' or 'gru'.")

        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones, use_random_init=True):
        if self.hidden_states is not None:
            for hidden_state in self.hidden_states:
                if dones.sum() > 0:
                    if use_random_init:
                        batch_size = hidden_state.size(-2)
                        shuffle_indices = torch.randperm(batch_size, device=hidden_state.device)
                        shuffled_hidden = hidden_state[..., shuffle_indices, :]
                        hidden_state[..., dones == 1, :] = shuffled_hidden[..., dones == 1, :]
                    else:
                        hidden_state[..., dones == 1, :] = 0.0


class SimpleConsistentDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.scale_factor = 1.0 / (1.0 - p)

    def forward(self, x, dropout_masks=None):
        if self.training:
            if dropout_masks is None:
                dropout_masks = torch.empty_like(x).bernoulli_(1 - self.p)
            out = x * dropout_masks * self.scale_factor
            return out, dropout_masks
        return x, None


class LinearConstDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_p, activation):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = SimpleConsistentDropout(p=dropout_p)
        self.dropout_masks = None

    def forward(self, x, dropout_masks=None):
        x = self.linear(x)
        x = self.activation(x)
        if dropout_masks is None:
            x, self.dropout_masks = self.dropout(x, dropout_masks=self.dropout_masks)
        else:
            x, _ = self.dropout(x, dropout_masks)
        return x

    def get_dropout_mask(self):
        return self.dropout_masks

    def reset_dropout_mask(self):
        self.dropout_masks = None


class ActorCriticRecurrent(ActorCritic):
    """In-file copy of rsl_rl ActorCriticRecurrent (training path only)."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        dropout=0.2,
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=actor_hidden_dims[0],
            num_critic_obs=critic_hidden_dims[0],
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        act = get_activation(activation)
        self.linear_dropout_actor = LinearConstDropout(
            in_features=rnn_hidden_size,
            out_features=actor_hidden_dims[0],
            dropout_p=dropout,
            activation=act,
        )
        self.linear_dropout_critic = LinearConstDropout(
            in_features=rnn_hidden_size,
            out_features=critic_hidden_dims[0],
            dropout_p=dropout,
            activation=act,
        )

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        input_a = self.linear_dropout_actor(input_a.squeeze(0), dropout_masks=dropout_masks)
        return super().act(input_a)

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        input_a = self.linear_dropout_actor(input_a.squeeze(0), dropout_masks=dropout_masks)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        input_c = self.linear_dropout_critic(input_c.squeeze(0), dropout_masks=dropout_masks)
        return super().evaluate(input_c)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_dropout_masks(self):
        return self.linear_dropout_actor.get_dropout_mask(), self.linear_dropout_critic.get_dropout_mask()

    def reset_dropout_masks(self):
        self.linear_dropout_actor.reset_dropout_mask()
        self.linear_dropout_critic.reset_dropout_mask()

    def get_actor_parameters(self):
        return (
            list(self.actor.parameters())
            + list(self.memory_a.parameters())
            + list(self.linear_dropout_actor.parameters())
            + [self.log_std]
        )

    def get_critic_parameters(self):
        return (
            list(self.critic.parameters())
            + list(self.memory_c.parameters())
            + list(self.linear_dropout_critic.parameters())
        )


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
# env_cfg.rewards.flat_orientation_l2.weight = -0.5
# env_cfg.rewards.lin_vel_z_l2.weight = -0.5
# env_cfg.rewards.track_lin_vel_xy_exp.weight = 3.0
# env_cfg.rewards.track_ang_vel_z_exp.weight = 2.0
# env_cfg.rewards.feet_air_time.weight = 0.5

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

log_dir = os.path.join("logs", "rsl_rl", "go1_lstm_v2_mdpo", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
    name=f"mdpo_lstm_baseline_v2_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": "LSTM-Baseline-v2-inline",
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

    if iteration % 10 == 0:
        log_dict["LSTM/action_std_mean"] = actor_critic_1.action_std.mean().item()

    wandb.log(log_dict, step=iteration)

    if iteration % 10 == 0:
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
            f"  [LSTM] std_mean={actor_critic_1.action_std.mean().item():.4f}"
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
