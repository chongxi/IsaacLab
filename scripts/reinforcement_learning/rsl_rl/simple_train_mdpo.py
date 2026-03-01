"""
Training script for Isaac-Reach-OpenArm-Bi-v0 with MDPO (Meta Distilled Policy Optimization).

MDPO uses two policies that learn from each other through mutual KL-divergence
distillation, combined with standard PPO updates. This improves exploration and
sample efficiency compared to single-policy PPO.

Usage:
    python scripts/reinforcement_learning/rsl_rl/simple_train_mdpo.py \
        --task=Isaac-Reach-OpenArm-Bi-v0 --headless
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

from rsl_rl.algorithms import MDPO

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
    OpenArmReachEnvCfgErrObs,
)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# Neural Network Definitions
#
# These are the same architectures from oparm_reach_play_spo.py, extended with
# the rsl_rl ActorCritic interface that MDPO expects.
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic_MLP(nn.Module):
    """Gaussian actor-critic with separate MLP actor and critic networks."""

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
        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # Actor
        actor_layers = []
        in_dim = num_obs
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        actor_layers.append(nn.Tanh())
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def get_hidden_states(self):
        return (None, None)

    def get_dropout_masks(self):
        return (None, None)

    def reset_dropout_masks(self):
        pass

    def get_actor_parameters(self):
        return list(self.actor.parameters()) + [self.log_std]

    def get_critic_parameters(self):
        return list(self.critic.parameters())


class NeuralJacobianPolicy(nn.Module):
    """Error-driven attention actor with standard MLP critic.

    The actor uses attention between joint states (q, dq) and task-space errors
    to compute actions. This architecture is specifically designed for bimanual
    manipulation with explicit error observations.
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [64, 64],
        critic_hidden_dims: list[int] = [64, 64],
        activation: str = "tanh",
        init_noise_std: float = 1.0,
        gate: bool = False,
    ):
        super().__init__()

        if num_actions % 2 != 0:
            raise ValueError(f"Expected even number of actions (bimanual), got {num_actions}.")

        self.dof_per_arm = num_actions // 2
        self.err_dim_per_arm = 6
        self.expected_obs_dim = 6 * self.dof_per_arm + 2 * self.err_dim_per_arm
        if num_obs != self.expected_obs_dim:
            raise ValueError(
                "NeuralJacobianPolicy expects explicit error observation layout "
                "[l_q, r_q, l_dq, r_dq, l_err6, r_err6, l_prev_a, r_prev_a]. "
                f"Got num_obs={num_obs}, expected={self.expected_obs_dim}."
            )

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        attn_dim = actor_hidden_dims[0] if len(actor_hidden_dims) > 0 else 64
        # Query input: (sin(q_i), cos(q_i), sin(dq_i), cos(dq_i)) — 4 values per joint
        self.joint_query = layer_init(nn.Linear(4, attn_dim), std=0.01)
        # Key input: (sin(q), cos(q), sin(dq), cos(dq)) — 4 * dof_per_arm values
        arm_key_input_dim = 4 * self.dof_per_arm
        self.arm_key = nn.Sequential(
            layer_init(nn.Linear(arm_key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.err_dim_per_arm * attn_dim), std=0.01),
        )
        self.joint_id_embed = nn.Parameter(torch.zeros(self.dof_per_arm, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)
        self.attn_scale = float(attn_dim) ** -0.5
        self.a_scale = nn.Parameter(torch.ones(self.dof_per_arm, self.err_dim_per_arm))

        # Error-magnitude gating: sigmoid(linear(||error||)) per arm
        self.use_gate = gate
        if gate:
            self.error_gate = layer_init(nn.Linear(1, 1), std=0.01)

        # Critic
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs: torch.Tensor):
        n = self.dof_per_arm
        left_q = obs[:, 0:n]
        right_q = obs[:, n:2 * n]
        left_dq = obs[:, 2 * n:3 * n]
        right_dq = obs[:, 3 * n:4 * n]
        left_err = obs[:, 4 * n:4 * n + self.err_dim_per_arm]
        right_err = obs[:, 4 * n + self.err_dim_per_arm:4 * n + 2 * self.err_dim_per_arm]
        return left_q, right_q, left_dq, right_dq, left_err, right_err

    def _actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        left_q, right_q, left_dq, right_dq, left_err, right_err = self._split_obs(obs)

        # Query: per-joint trig features (sin(q_i), cos(q_i), sin(dq_i), cos(dq_i))
        left_joint_tokens = torch.stack([torch.sin(left_q), torch.cos(left_q), torch.sin(left_dq), torch.cos(left_dq)], dim=-1)
        right_joint_tokens = torch.stack([torch.sin(right_q), torch.cos(right_q), torch.sin(right_dq), torch.cos(right_dq)], dim=-1)
        left_query = self.joint_query(left_joint_tokens)
        right_query = self.joint_query(right_joint_tokens)
        joint_bias = self.joint_id_embed.unsqueeze(0)
        left_query = left_query + joint_bias
        right_query = right_query + joint_bias

        # Key: global trig features (sin(q_all), cos(q_all), sin(dq_all), cos(dq_all))
        left_state = torch.cat([torch.sin(left_q), torch.cos(left_q), torch.sin(left_dq), torch.cos(left_dq)], dim=-1)
        right_state = torch.cat([torch.sin(right_q), torch.cos(right_q), torch.sin(right_dq), torch.cos(right_dq)], dim=-1)
        left_key = self.arm_key(left_state).view(-1, self.err_dim_per_arm, left_query.shape[-1])
        right_key = self.arm_key(right_state).view(-1, self.err_dim_per_arm, right_query.shape[-1])

        left_logits = torch.einsum("bnd,bmd->bnm", left_query, left_key) * self.attn_scale
        right_logits = torch.einsum("bnd,bmd->bnm", right_query, right_key) * self.attn_scale
        left_A = left_logits
        right_A = right_logits

        left_u = torch.bmm(left_A, left_err.unsqueeze(-1)).squeeze(-1)
        right_u = torch.bmm(right_A, right_err.unsqueeze(-1)).squeeze(-1)

        if self.use_gate:
            left_gate = torch.sigmoid(self.error_gate(left_err.norm(dim=-1, keepdim=True)))
            right_gate = torch.sigmoid(self.error_gate(right_err.norm(dim=-1, keepdim=True)))
            left_u = left_gate * left_u
            right_u = right_gate * right_u

        raw_u = torch.cat([left_u, right_u], dim=-1)
        return raw_u

    def update_distribution(self, observations: torch.Tensor):
        mean = self._actor_mean(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._actor_mean(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def get_hidden_states(self):
        return (None, None)

    def get_dropout_masks(self):
        return (None, None)

    def reset_dropout_masks(self):
        pass

    def get_actor_parameters(self):
        params = [
            *self.joint_query.parameters(),
            *self.arm_key.parameters(),
            self.joint_id_embed,
            self.a_scale,
            self.log_std,
        ]
        if self.use_gate:
            params.extend(self.error_gate.parameters())
        return params

    def get_critic_parameters(self):
        return list(self.critic.parameters())


class NeuralJacobianLocalPolicy(nn.Module):
    """Local-attention Jacobian actor: both query and key are per-token local.

    Query: per-joint (q_i, dq_i) + joint_id_embed  →  local
    Key:   per-error-dim (err_e) + error_id_embed   →  local
    A = Q @ K^T  →  (joints × error_dims) pseudo-Jacobian
    action = A @ error
    """

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

        if num_actions % 2 != 0:
            raise ValueError(f"Expected even number of actions (bimanual), got {num_actions}.")

        self.dof_per_arm = num_actions // 2
        self.err_dim_per_arm = 6
        self.expected_obs_dim = 6 * self.dof_per_arm + 2 * self.err_dim_per_arm
        if num_obs != self.expected_obs_dim:
            raise ValueError(
                "NeuralJacobianLocalPolicy expects explicit error observation layout "
                "[l_q, r_q, l_dq, r_dq, l_err6, r_err6, l_prev_a, r_prev_a]. "
                f"Got num_obs={num_obs}, expected={self.expected_obs_dim}."
            )

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        attn_dim = actor_hidden_dims[0] if len(actor_hidden_dims) > 0 else 64
        self.joint_query = layer_init(nn.Linear(2, attn_dim), std=0.01)
        self.joint_id_embed = nn.Parameter(torch.zeros(self.dof_per_arm, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)
        self.error_key = layer_init(nn.Linear(1, attn_dim), std=0.01)
        self.error_id_embed = nn.Parameter(torch.zeros(self.err_dim_per_arm, attn_dim))
        nn.init.normal_(self.error_id_embed, mean=0.0, std=0.02)

        self.attn_scale = float(attn_dim) ** -0.5

        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs: torch.Tensor):
        n = self.dof_per_arm
        left_q = obs[:, 0:n]
        right_q = obs[:, n:2 * n]
        left_dq = obs[:, 2 * n:3 * n]
        right_dq = obs[:, 3 * n:4 * n]
        left_err = obs[:, 4 * n:4 * n + self.err_dim_per_arm]
        right_err = obs[:, 4 * n + self.err_dim_per_arm:4 * n + 2 * self.err_dim_per_arm]
        return left_q, right_q, left_dq, right_dq, left_err, right_err

    def _actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        left_q, right_q, left_dq, right_dq, left_err, right_err = self._split_obs(obs)

        left_joint_tokens = torch.stack([left_q, left_dq], dim=-1)
        right_joint_tokens = torch.stack([right_q, right_dq], dim=-1)
        left_query = self.joint_query(left_joint_tokens) + self.joint_id_embed.unsqueeze(0)
        right_query = self.joint_query(right_joint_tokens) + self.joint_id_embed.unsqueeze(0)

        left_err_tokens = left_err.unsqueeze(-1)
        right_err_tokens = right_err.unsqueeze(-1)
        left_key = self.error_key(left_err_tokens) + self.error_id_embed.unsqueeze(0)
        right_key = self.error_key(right_err_tokens) + self.error_id_embed.unsqueeze(0)

        left_A = torch.einsum("bnd,bmd->bnm", left_query, left_key) * self.attn_scale
        right_A = torch.einsum("bnd,bmd->bnm", right_query, right_key) * self.attn_scale

        left_u = torch.bmm(left_A, left_err.unsqueeze(-1)).squeeze(-1)
        right_u = torch.bmm(right_A, right_err.unsqueeze(-1)).squeeze(-1)

        return torch.cat([left_u, right_u], dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        mean = self._actor_mean(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._actor_mean(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def get_hidden_states(self):
        return (None, None)

    def get_dropout_masks(self):
        return (None, None)

    def reset_dropout_masks(self):
        pass

    def get_actor_parameters(self):
        params = [
            *self.joint_query.parameters(),
            *self.error_key.parameters(),
            self.joint_id_embed,
            self.error_id_embed,
            self.log_std,
        ]
        return params

    def get_critic_parameters(self):
        return list(self.critic.parameters())


# ==============================================================================
# Utility
# ==============================================================================
def _flatten_obs(obs: TensorDict) -> torch.Tensor:
    """Extract flat observation tensor from a TensorDict."""
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
# Configuration
# ==============================================================================
# Environment
USE_ERROR_OBS_EXPERIMENT = True
env_cfg = OpenArmReachEnvCfgErrObs() if USE_ERROR_OBS_EXPERIMENT else OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 4096 * 2  # MDPO uses twice as many envs to train two policies
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

env_cfg.actions.left_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_left_joint.*"],
    scale=0.5,
    use_zero_offset=True,
    alpha=0.7,
)
env_cfg.actions.right_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_right_joint.*"],
    scale=0.5,
    use_zero_offset=True,
    alpha=0.7,
)
device = "cuda:0"

# Training
max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# MDPO hyperparameters
mdpo_cfg = dict(
    num_learning_epochs=8,
    num_mini_batches=4,
    clip_param=0.2,
    value_clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    distill_coef=0.02,
    value_loss_coef=1.0,
    entropy_coef=0.001,
    learning_rate=1e-2,
    min_learning_rate=3e-5,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="exponential",
    use_muon=False,  # Use Adam (Muon may not be available)
)

# Network architecture
actor_hidden_dims = [64, 64]
critic_hidden_dims = [64, 64]
activation = "elu"
init_noise_std = 1.0
policy_class_name = "njp"  # "mlp", "njp", or "njp_local"
use_error_gate = True  # error-magnitude gating (NJP only)

# ==============================================================================
# Step 1: Create environment
# ==============================================================================
env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

log_dir = os.path.join("logs", "rsl_rl", "openarm_bi_reach", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

# ==============================================================================
# Step 2: Create networks and MDPO algorithm
# ==============================================================================
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs = obs_flat.shape[-1]
num_actions = env.num_actions

print(f"Observation dim: {num_obs}, Action dim: {num_actions}")

policy_class_map = {
    "mlp": ActorCritic_MLP,
    "njp": NeuralJacobianPolicy,
    "njp_local": NeuralJacobianLocalPolicy,
}
policy_cls = policy_class_map[policy_class_name.lower()]

# MDPO uses two actor-critic networks
policy_kwargs = dict(
    num_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation=activation,
    init_noise_std=init_noise_std,
)
if policy_cls is NeuralJacobianPolicy:
    policy_kwargs["gate"] = use_error_gate

actor_critic_1 = policy_cls(**policy_kwargs)
actor_critic_2 = policy_cls(**policy_kwargs)

mdpo = MDPO(actor_critic_1, actor_critic_2, device=device, **mdpo_cfg)
mdpo.init_storage(
    num_envs=env.num_envs,
    num_transitions_per_env=num_steps_per_env,
    actor_obs_shape=[num_obs],
    critic_obs_shape=[num_obs],  # No privileged observations
    action_shape=[num_actions],
)

print(f"Policy 1: {actor_critic_1.__class__.__name__}")
print(f"Policy 2: {actor_critic_2.__class__.__name__}")
total_params = sum(p.numel() for p in actor_critic_1.parameters()) + sum(p.numel() for p in actor_critic_2.parameters())
print(f"Total parameters (both networks): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")

# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
wandb.init(
    project="isaaclab-openarm-reach",
    name=f"mdpo_custom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "policy_class": policy_class_name,
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

env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
obs_td = env.get_observations().to(device)

print(f"\nStarting MDPO training for {max_iterations} iterations...")
print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    mdpo.train_mode()
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            obs_flat = _flatten_obs(obs_td)

            # MDPO splits envs between the two policies internally
            actions = mdpo.act(obs_flat, obs_flat)

            # Step environment
            obs_td, rewards, dones, extras = env.step(actions.detach().to(env.device))
            obs_td = obs_td.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            # MDPO stores transitions and resets hidden states internally
            mdpo.process_env_step(rewards, dones, extras)

            # Episode tracking
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

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
    }
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
            f"fps={fps}"
        )

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        torch.save(
            {
                "iter": iteration + 1,
                "model_1_state_dict": actor_critic_1.state_dict(),
                "model_2_state_dict": actor_critic_2.state_dict(),
                "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
                "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint: {ckpt_path}")

# ==============================================================================
# Cleanup
# ==============================================================================
wandb.finish()
env.close()
simulation_app.close()
print("Training complete.")
