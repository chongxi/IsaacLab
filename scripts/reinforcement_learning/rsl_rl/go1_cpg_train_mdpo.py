"""
Training script for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO + CPG clock.

Architecture: Autonomous MANC CPG provides rhythmic clock signal as input to MLP.
    cpg_rates = cpg_step(cpg_x, cpg_a)          # (B, 4, 2) from 4 MANC oscillators
    obs_aug = cat(obs, cpg_rates_flat)           # (B, 48 + 8 = 56)
    action = actor_mlp(obs_aug)                  # unconstrained MLP

CPG dynamics (W_rec, tau, etc.) are FROZEN buffers — oscillation guaranteed.
The MLP learns to use or ignore the rhythmic signal as needed.

CPG state maps to LSTM's (h, c) for recurrent storage:
    cpg_x: (B, 4, 3) → flatten → (1, B, 12) as "hidden state"
    cpg_a: (B, 4, 3) → flatten → (1, B, 12) as "cell state"

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_cpg_train_mdpo.py
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
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


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


# (BatchedQuadrupedDecoder removed — CPG now runs autonomously with fixed params)


# ==============================================================================
# CPG_Clock_ActorCritic: CPG provides rhythmic clock input to MLP
# ==============================================================================
class CPG_Clock_ActorCritic(nn.Module):
    """Actor-critic where autonomous MANC CPG provides clock signal to MLP.

    The CPG oscillates autonomously (frozen dynamics, fixed dn/sht).
    Its firing rates are concatenated with observations as input to the actor MLP.
    The MLP learns to use the rhythmic signal as it sees fit.

    CPG state (cpg_x, cpg_a) maps to LSTM (h, c) for recurrent storage.
    """

    is_recurrent = True

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [128, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        cpg_dt: float = 0.1,
        cpg_substeps: int = 1,
        bptt_length: int = 0,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions  # 12
        self.cpg_dt = cpg_dt
        self.cpg_substeps = cpg_substeps
        self.bptt_length = bptt_length
        self.n_legs = 4
        self.n_neurons = 3
        self.cpg_clock_dim = self.n_legs * 2  # 8: E1,E2 rates from each leg
        self._device = device

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # === CPG dynamics (FROZEN — guarantees oscillation) ===
        self.register_buffer("tau_x", torch.tensor([0.18, 0.24, 0.35]))
        self.register_buffer("tau_a", torch.tensor([0.80, 1.00, 1e6]))
        self.register_buffer("g_adapt", torch.tensor([1.8, 1.5, 0.0]))
        self.register_buffer("W_rec", torch.tensor([
            [0.00, 0.54, -6.30],
            [6.30, 0.00, -2.16],
            [0.36, 1.80,  0.00],
        ]))
        self.register_buffer("W_in", torch.tensor([2.25, 0.0, 0.0]))
        self.register_buffer("bias", torch.tensor([0.0, -0.2, -0.3]))

        self.register_buffer("init_cpg_x", torch.randn(4, 3) * 0.5)
        self.register_buffer("init_cpg_a", torch.zeros(4, 3))

        # Fixed CPG drive
        self.fixed_dn = 0.88   # sigmoid(2.0)
        self.fixed_sht = 1.0

        # === Actor MLP: obs(48) + cpg_clock(8) → actions(12) ===
        actor_layers = []
        in_dim = num_obs + self.cpg_clock_dim  # 48 + 8 = 56
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

        # === Critic MLP: obs(48) → value (no clock needed) ===
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # === Action noise ===
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # === CPG state ===
        self.cpg_x: torch.Tensor | None = None  # (B, 4, 3)
        self.cpg_a: torch.Tensor | None = None  # (B, 4, 3)

    def init_cpg_state(self, num_envs: int):
        self.cpg_x = self.init_cpg_x.unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.cpg_a = self.init_cpg_a.unsqueeze(0).expand(num_envs, -1, -1).clone()

    def _cpg_step(self, cpg_x, cpg_a):
        """Autonomous MANC CPG Euler step. No gradients flow through this."""
        ext_input = self.W_in * self.fixed_dn

        for _ in range(self.cpg_substeps):
            r = F.gelu(cpg_x)
            rec_input = self.fixed_dn * torch.einsum("bln,mn->blm", r, self.W_rec)
            dxdt = -cpg_x + rec_input + ext_input + self.bias - cpg_a
            dadt = -cpg_a + self.g_adapt * r
            cpg_x = cpg_x + self.cpg_dt * dxdt
            cpg_a = cpg_a + self.cpg_dt * dadt

        new_r = F.gelu(cpg_x)
        return new_r, cpg_x, cpg_a

    def _get_clock_signal(self, cpg_r):
        """Extract clock signal from CPG rates: E1, E2 from each leg → (B, 8)."""
        return cpg_r[:, :, :2].reshape(-1, self.cpg_clock_dim)  # (B, 8)

    def _compute_action_mean(self, obs, cpg_x, cpg_a):
        """Step CPG, concat clock with obs, pass through actor MLP."""
        # CPG step (no grad needed — frozen dynamics, clock is just input)
        with torch.no_grad():
            new_r, new_cpg_x, new_cpg_a = self._cpg_step(cpg_x, cpg_a)
            clock = self._get_clock_signal(new_r)  # (B, 8)

        # Actor: obs + clock → actions
        obs_aug = torch.cat([obs, clock], dim=-1)  # (B, 56)
        action_mean = self.actor(obs_aug)  # (B, 12)

        return action_mean, new_cpg_x, new_cpg_a

    # === rsl_rl ActorCritic interface ===

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        batch_mode = masks is not None

        if batch_mode:
            if isinstance(hidden_states, (list, tuple)):
                cpg_x_flat = hidden_states[0]
                cpg_a_flat = hidden_states[1]
            else:
                cpg_x_flat = hidden_states
                cpg_a_flat = torch.zeros_like(cpg_x_flat)

            cpg_x = cpg_x_flat.squeeze(0).reshape(-1, 4, 3)
            cpg_a = cpg_a_flat.squeeze(0).reshape(-1, 4, 3)

            L = observations.shape[0]
            means_list = []

            for t in range(L):
                if self.bptt_length > 0 and t % self.bptt_length == 0 and t > 0:
                    cpg_x = cpg_x.detach()
                    cpg_a = cpg_a.detach()
                obs_t = observations[t]
                mean_t, cpg_x, cpg_a = self._compute_action_mean(obs_t, cpg_x, cpg_a)
                means_list.append(mean_t)

            means_padded = torch.stack(means_list, dim=0)
            means = unpad_trajectories(means_padded, masks)

            std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(means)
            self.distribution = Normal(means, std)
            return self.distribution.sample()
        else:
            mean, self.cpg_x, self.cpg_a = self._compute_action_mean(
                observations, self.cpg_x, self.cpg_a
            )
            std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(mean)
            self.distribution = Normal(mean, std)
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            values_padded = self.critic(critic_observations)
            values = unpad_trajectories(values_padded, masks)
            return values
        else:
            return self.critic(critic_observations)

    def act_inference(self, observations):
        mean, self.cpg_x, self.cpg_a = self._compute_action_mean(
            observations, self.cpg_x, self.cpg_a
        )
        return mean

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_hidden_states(self):
        cpg_x_flat = self.cpg_x.reshape(1, -1, 12)
        cpg_a_flat = self.cpg_a.reshape(1, -1, 12)
        return (cpg_x_flat, cpg_a_flat), (cpg_x_flat, cpg_a_flat)

    def reset(self, dones=None):
        if dones is None or self.cpg_x is None:
            return
        done_mask = dones.bool()
        if done_mask.any():
            n_done = done_mask.sum().item()
            self.cpg_x[done_mask] = self.init_cpg_x.unsqueeze(0).expand(n_done, -1, -1)
            self.cpg_a[done_mask] = self.init_cpg_a.unsqueeze(0).expand(n_done, -1, -1)

    def get_dropout_masks(self):
        return (None, None)

    def reset_dropout_masks(self):
        pass

    def get_actor_parameters(self):
        return [*self.actor.parameters(), self.log_std]

    def get_critic_parameters(self):
        return list(self.critic.parameters())


# ==============================================================================
# LSTM_ActorCritic: control experiment — same interface as CPG_Reflex_ActorCritic
# ==============================================================================
class ConsistentDropout(nn.Module):
    """Dropout that reuses the same mask across a trajectory for consistent regularization."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p)
        self.mask: torch.Tensor | None = None

    def forward(self, x, mask=None):
        if not self.training:
            return x, None
        if mask is not None:
            return x * mask * self.scale, mask
        if self.mask is None or self.mask.shape != x.shape:
            self.mask = torch.empty_like(x).bernoulli_(1 - self.p)
        return x * self.mask * self.scale, self.mask

    def reset_mask(self):
        self.mask = None

    def get_mask(self):
        return self.mask


class LSTM_ActorCritic(nn.Module):
    """LSTM actor-critic with the same interface as CPG_Reflex_ActorCritic.

    Uses LSTM hidden state stored in the same (h, c) slots that CPG uses,
    so the rollout storage and MDPO pipeline work identically.
    """

    is_recurrent = True

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [128, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        lstm_hidden_size: int = 128,
        dropout: float = 0.2,
        device: str = "cuda:0",
        **kwargs,  # accept and ignore CPG-specific kwargs
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self._device = device

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # === LSTM memory ===
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=num_obs, hidden_size=lstm_hidden_size, num_layers=1, batch_first=False)
        self.h: torch.Tensor | None = None  # (1, B, H)
        self.c: torch.Tensor | None = None  # (1, B, H)

        # === Post-LSTM dropout + projection (matches rsl_rl ActorCriticRecurrent) ===
        self.post_lstm_linear = nn.Linear(lstm_hidden_size, actor_hidden_dims[0])
        self.post_lstm_act = act_fn()
        self.post_lstm_dropout = ConsistentDropout(p=dropout)

        # === Actor MLP: post-LSTM features → actions ===
        actor_layers = []
        in_dim = actor_hidden_dims[0]
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

        # === Critic MLP: obs → value (feedforward, no LSTM needed) ===
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # === Action noise ===
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def init_hidden(self, num_envs: int):
        """Initialize LSTM hidden state."""
        self.h = torch.zeros(1, num_envs, self.lstm_hidden_size, device=self._device)
        self.c = torch.zeros(1, num_envs, self.lstm_hidden_size, device=self._device)

    # Alias so the same init call works as CPG
    def init_cpg_state(self, num_envs: int):
        self.init_hidden(num_envs)

    def _post_lstm(self, lstm_out, dropout_masks=None):
        """LSTM output → dropout → activation → features."""
        x = self.post_lstm_linear(lstm_out)
        x = self.post_lstm_act(x)
        x, _ = self.post_lstm_dropout(x, mask=dropout_masks)
        return x

    def _forward_lstm_single(self, obs):
        """Single-step LSTM forward. obs: (B, D) → features: (B, H)."""
        out, (self.h, self.c) = self.lstm(obs.unsqueeze(0), (self.h, self.c))
        return self._post_lstm(out.squeeze(0))

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None, **kwargs):
        batch_mode = masks is not None

        if batch_mode:
            # Restore hidden state from storage
            if isinstance(hidden_states, (list, tuple)):
                h = hidden_states[0]  # (1, num_traj, H)
                c = hidden_states[1]  # (1, num_traj, H)
            else:
                h = hidden_states
                c = torch.zeros_like(h)

            # Run LSTM over full sequence
            out, _ = self.lstm(observations, (h, c))  # (L, num_traj, H)
            out = unpad_trajectories(out, masks)
            features = self._post_lstm(out, dropout_masks=dropout_masks)

            means = self.actor(features)
            std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(means)
            self.distribution = Normal(means, std)
            return self.distribution.sample()
        else:
            # Single-step mode during collection
            features = self._forward_lstm_single(observations)
            mean = self.actor(features)
            std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(mean)
            self.distribution = Normal(mean, std)
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            values_padded = self.critic(critic_observations)
            values = unpad_trajectories(values_padded, masks)
            return values
        else:
            return self.critic(critic_observations)

    def act_inference(self, observations):
        features = self._forward_lstm_single(observations)
        return self.actor(features)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_hidden_states(self):
        """Return LSTM state in same format as CPG: (actor_h_c, critic_h_c)."""
        return (self.h, self.c), (self.h, self.c)

    def reset(self, dones=None):
        if dones is None or self.h is None:
            return
        done_mask = dones.bool()
        if done_mask.any():
            self.h[:, done_mask] = 0.0
            self.c[:, done_mask] = 0.0

    def get_dropout_masks(self):
        return self.post_lstm_dropout.get_mask(), None

    def reset_dropout_masks(self):
        self.post_lstm_dropout.reset_mask()

    def get_actor_parameters(self):
        return [
            *self.lstm.parameters(),
            *[self.post_lstm_linear.weight, self.post_lstm_linear.bias],
            *self.actor.parameters(),
            self.log_std,
        ]

    def get_critic_parameters(self):
        return list(self.critic.parameters())


# ==============================================================================
# Configuration
# ==============================================================================
env_cfg = UnitreeGo1FlatEnvCfg()
env_cfg.scene.num_envs = 4096 * 2
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"
env_cfg.actions.joint_pos.scale = 0.5  # standard Go1 setting; unbounded networks learn to fill clip range
device = "cuda:0"

# --- Reward overrides (SAME as CPG experiment) ---
# env_cfg.rewards.action_rate_l2 = None
env_cfg.rewards.flat_orientation_l2.weight = -5.0
env_cfg.rewards.lin_vel_z_l2.weight = -0.5
env_cfg.rewards.track_lin_vel_xy_exp.weight = 6.0 # 3.0
env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(1.0)
env_cfg.rewards.track_ang_vel_z_exp.weight = 5.0 # 2.0
env_cfg.rewards.feet_air_time.weight = 0.5

max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# CPG (frozen clock — no gradients flow through CPG)
cpg_dt = 0.3
cpg_substeps = 1
bptt_length = 0

# MDPO
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

# Network
actor_hidden_dims = [128, 128]
critic_hidden_dims = [256, 128]
activation = "elu"
init_noise_std = 1.0


# ==============================================================================
# Step 1: Create environment
# ==============================================================================
env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

log_dir = os.path.join("logs", "rsl_rl", "go1_cpg_mdpo", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)


# ==============================================================================
# Step 2: Create networks and MDPO algorithm
# ==============================================================================
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs = obs_flat.shape[-1]  # 48
num_actions = env.num_actions  # 12

# -------------------------------------------------------
# Toggle: set USE_LSTM = True for LSTM baseline, False for CPG
# -------------------------------------------------------
USE_LSTM = False

print(f"Obs dim: {num_obs}, Action dim: {num_actions}")

if USE_LSTM:
    # Use rsl_rl's ActorCriticRecurrent directly (this works in go1_lstm_train_mdpo.py)
    policy_kwargs = dict(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        rnn_type="lstm",
        rnn_hidden_size=128,
        rnn_num_layers=1,
        init_noise_std=init_noise_std,
    )
    actor_critic_1 = ActorCriticRecurrent(**policy_kwargs).to(device)
    actor_critic_2 = ActorCriticRecurrent(**policy_kwargs).to(device)

    mdpo = MDPO(actor_critic_1, actor_critic_2, device=device, **mdpo_cfg)
    # No special LR groups needed for LSTM

else:
    policy_kwargs = dict(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
        cpg_dt=cpg_dt,
        cpg_substeps=cpg_substeps,
        bptt_length=bptt_length,
        device=device,
    )
    actor_critic_1 = CPG_Clock_ActorCritic(**policy_kwargs).to(device)
    actor_critic_2 = CPG_Clock_ActorCritic(**policy_kwargs).to(device)

    num_envs_1 = env.num_envs // 2
    num_envs_2 = env.num_envs - num_envs_1
    actor_critic_1.init_cpg_state(num_envs_1)
    actor_critic_2.init_cpg_state(num_envs_2)

    mdpo = MDPO(actor_critic_1, actor_critic_2, device=device, **mdpo_cfg)
    # No special LR groups — CPG is frozen, MLP gets normal gradients

mdpo.init_storage(
    num_envs=env.num_envs,
    num_transitions_per_env=num_steps_per_env,
    actor_obs_shape=[num_obs],
    critic_obs_shape=[num_obs],
    action_shape=[num_actions],
)

total_params = sum(p.numel() for p in actor_critic_1.parameters()) + sum(p.numel() for p in actor_critic_2.parameters())
print(f"Policy: {actor_critic_1.__class__.__name__}")
print(f"Total parameters (both policies): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")


# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
arch_name = "LSTM" if USE_LSTM else "CPG-Clock"
wandb.init(
    project="isaaclab-go1-velocity",
    name=f"mdpo_{arch_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": arch_name,
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "num_obs": num_obs,
        "num_actions": num_actions,
        "cpg_dt": cpg_dt,
        "cpg_substeps": cpg_substeps,
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

print(f"\nStarting MDPO + {arch_name} training for {max_iterations} iterations...")
print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
if not USE_LSTM:
    print(f"  CPG clock: dt={cpg_dt}, substeps={cpg_substeps} (frozen, 8-dim clock input)")
    print(f"  Actor MLP: {actor_hidden_dims} (input: obs(48) + clock(8) = 56), Critic: {critic_hidden_dims}")
else:
    print(f"  LSTM: hidden_size=128, Actor MLP: [128, 128], Critic: {critic_hidden_dims}")

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
    # Gait weight curriculum: multiply by 2 every 200 iterations, cap at 20
    # ==================================================================
    if iteration > 0 and iteration % 200 == 0:
        gait_cfg = env.unwrapped.reward_manager.get_term_cfg("gait")
        new_weight = min(gait_cfg.weight * 2, 20.0)
        if gait_cfg.weight < 20.0:
            gait_cfg.weight = new_weight
            env.unwrapped.reward_manager.set_term_cfg("gait", gait_cfg)
            print(f"  [Curriculum] Gait weight updated: {new_weight:.1f}")

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

    # Log gait weight curriculum
    gait_weight = env.unwrapped.reward_manager.get_term_cfg("gait").weight
    log_dict["Curriculum/gait_weight"] = gait_weight

    # Log architecture-specific diagnostics
    if iteration % 10 == 0:
        log_dict["action_std_mean"] = actor_critic_1.log_std.clamp(-5.0, 0.5).exp().mean().item()
        if not USE_LSTM:
            log_dict["CPG/cpg_x_rms"] = actor_critic_1.cpg_x.pow(2).mean().sqrt().item()

    wandb.log(log_dict, step=iteration)

    if iteration % 10 == 0:
        def _to_float(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        r_base_height = _to_float(log_extras.get("Episode_Reward/base_height", 0.0))
        r_foot_clear = _to_float(log_extras.get("Episode_Reward/foot_clearance", 0.0))
        r_flat_orient = _to_float(log_extras.get("Episode_Reward/flat_orientation_l2", 0.0))
        r_track_lin = _to_float(log_extras.get("Episode_Reward/track_lin_vel_xy_exp", 0.0))
        r_track_ang = _to_float(log_extras.get("Episode_Reward/track_ang_vel_z_exp", 0.0))
        r_feet_air = _to_float(log_extras.get("Episode_Reward/feet_air_time", 0.0))
        r_gait = _to_float(log_extras.get("Episode_Reward/gait", 0.0))
        r_feet_slide = _to_float(log_extras.get("Episode_Reward/feet_slide", 0.0))
        r_min_lift = _to_float(log_extras.get("Episode_Reward/min_foot_lift", 0.0))

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
            f"  std={actor_critic_1.log_std.clamp(-5.0, 0.5).exp().mean().item():.4f}  "
            f"base_height={r_base_height:.4f}  foot_clear={r_foot_clear:.4f}  "
            f"flat_orient={r_flat_orient:.4f}  "
            f"track_lin={r_track_lin:.4f}  track_ang={r_track_ang:.4f}  "
            f"feet_air={r_feet_air:.4f}  gait={r_gait:.4f}  "
            f"slide={r_feet_slide:.4f}  lift={r_min_lift:.4f}"
        )

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        ckpt_data = {
            "iter": iteration + 1,
            "architecture": "lstm" if USE_LSTM else "cpg_clock",
            "model_1_state_dict": actor_critic_1.state_dict(),
            "model_2_state_dict": actor_critic_2.state_dict(),
            "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
            "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
            "cpg_config": {
                "cpg_dt": cpg_dt,
                "cpg_substeps": cpg_substeps,
            },
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
