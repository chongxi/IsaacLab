"""
Training script for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO + CPG-NJP.

Architecture: CPG feedforward + dual Neural Jacobian reflexes
    action = f_cpg(q, dq, prev_a, phase, cmd_vel)  # gait generator
           + A_bal(q, dq) @ gravity_error            # balance reflex
           + A_vel(q, dq) @ velocity_error           # velocity correction

3 independent phase oscillators (one per command dimension):
    phase_vx:  freq = base_freq * |vx|     # forward/backward oscillation
    phase_vy:  freq = base_freq * |vy|     # lateral oscillation
    phase_wz:  freq = base_freq * |wz|     # turning oscillation
    phase_features = [sin(phase_vx), sin(phase_vy), sin(phase_wz),
                      cos(phase_vx), cos(phase_vy), cos(phase_wz)]  # 6 dims

The CPG MLP learns to combine these 3 oscillation patterns into joint-level gaits.
When a command dimension is zero, its phase freezes (no oscillation).
No RNN needed — phases are explicit clocks maintained in the training loop.

Obs layout (48 base + 6 phase = 54 augmented):
    [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), commands(3),
     joint_pos(12), joint_vel(12), prev_actions(12), phase_sin_cos(6)]

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_velocity_train_mdpo.py
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

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import MDPO

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# Utilities
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


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
# Network: CPG-NJP for Quadruped Locomotion
# ==============================================================================
class LocomotionCPG_NJP(nn.Module):
    """CPG feedforward + dual-NJP reflex for quadruped locomotion.

    Actor:
        action = f_cpg(q, dq, prev_a, phase, cmd)  # gait generator
               + A_bal(q, dq) @ gravity_error       # balance reflex
               + A_vel(q, dq) @ velocity_error      # velocity correction

    3 independent phase oscillators (vx, vy, wz) each produce sin/cos features.
    CPG MLP learns to map these oscillation patterns to joint-level gaits.

    Obs layout (54 = 48 base + 6 phase):
        [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), commands(3),
         joint_pos(12), joint_vel(12), prev_actions(12), phase_sin_cos(6)]
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [64, 64],
        critic_hidden_dims: list[int] = [64, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        gate: bool = True,
        init_base_freq: float = 2 * math.pi * 2.0,
    ):
        super().__init__()

        if num_actions % 4 != 0:
            raise ValueError(f"Expected actions divisible by 4 (quadruped), got {num_actions}.")

        self.num_joints = num_actions  # 12
        self.num_legs = 4
        self.joints_per_leg = num_actions // 4  # 3
        self.gravity_error_dim = 3
        self.velocity_error_dim = 3  # [vx_err, vy_err, wz_err]
        self.cmd_dim = 3
        self.num_phases = 3  # one per command dim: vx, vy, wz
        self.phase_feature_dim = 2 * self.num_phases  # 6: sin/cos per phase

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]
        attn_dim = actor_hidden_dims[0] if actor_hidden_dims else 64

        # === CPG feedforward pathway ===
        # Input: joint_pos(12) + joint_vel(12) + prev_actions(12) + phase(6) + commands(3) = 45
        cpg_input_dim = 3 * self.num_joints + self.phase_feature_dim + self.cmd_dim
        cpg_layers = []
        in_dim = cpg_input_dim
        for h_dim in actor_hidden_dims:
            cpg_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            cpg_layers.append(act_fn())
            in_dim = h_dim
        cpg_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.cpg = nn.Sequential(*cpg_layers)

        # === Balance NJP pathway ===
        # Query: per-joint trig features (sin(q_i), cos(q_i), sin(dq_i), cos(dq_i)) → 4 per joint
        self.joint_query = layer_init(nn.Linear(4, attn_dim), std=0.01)
        self.joint_id_embed = nn.Parameter(torch.zeros(self.num_joints, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)

        # Key: global joint trig features → 3 gravity-error-dim vectors
        key_input_dim = 4 * self.num_joints  # 48
        self.balance_key = nn.Sequential(
            layer_init(nn.Linear(key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.gravity_error_dim * attn_dim), std=0.01),
        )
        self.attn_scale = float(attn_dim) ** -0.5

        # === Velocity NJP pathway (shares joint_query/joint_id_embed with balance) ===
        # Key: global joint trig features → 3 velocity-error-dim vectors
        self.velocity_key = nn.Sequential(
            layer_init(nn.Linear(key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.velocity_error_dim * attn_dim), std=0.01),
        )

        # Error-magnitude gating: sigmoid(linear(||error||)) per pathway
        self.use_gate = gate
        if gate:
            self.balance_gate = layer_init(nn.Linear(1, 1), std=0.01)
            self.velocity_gate = layer_init(nn.Linear(1, 1), std=0.01)

        # === Critic (standard MLP on full augmented obs) ===
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

        # Learnable phase oscillator base frequencies (one per phase: vx, vy, wz)
        self.log_base_freq = nn.Parameter(torch.log(init_base_freq * torch.ones(self.num_phases)))

    def _parse_obs(self, obs: torch.Tensor):
        """Parse augmented obs: 48 base + 6 phase = 54 dims."""
        base_lin_vel = obs[:, 0:3]
        base_ang_vel = obs[:, 3:6]
        projected_gravity = obs[:, 6:9]
        commands = obs[:, 9:12]
        joint_pos = obs[:, 12:24]
        joint_vel = obs[:, 24:36]
        prev_actions = obs[:, 36:48]
        phase_features = obs[:, 48:54]
        return (base_lin_vel, base_ang_vel, projected_gravity, commands,
                joint_pos, joint_vel, prev_actions, phase_features)

    def _actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        (base_lin_vel, base_ang_vel, projected_gravity, commands,
         joint_pos, joint_vel, prev_actions, phase_features) = self._parse_obs(obs)

        # === CPG feedforward (receives prev_actions + commands for direction-dependent gaits) ===
        cpg_input = torch.cat([joint_pos, joint_vel, prev_actions, phase_features, commands], dim=-1)  # [B, 45]
        cpg_action = self.cpg(cpg_input)  # [B, 12]

        # === Shared query: per-joint trig features → 12 tokens ===
        joint_tokens = torch.stack([
            torch.sin(joint_pos), torch.cos(joint_pos),
            torch.sin(joint_vel), torch.cos(joint_vel),
        ], dim=-1)  # [B, 12, 4]
        Q = self.joint_query(joint_tokens) + self.joint_id_embed.unsqueeze(0)  # [B, 12, d]

        # Shared global trig features for both keys
        global_trig = torch.cat([
            torch.sin(joint_pos), torch.cos(joint_pos),
            torch.sin(joint_vel), torch.cos(joint_vel),
        ], dim=-1)  # [B, 48]

        # === Balance NJP: A_bal(q, dq) @ gravity_error ===
        gravity_error = projected_gravity - projected_gravity.new_tensor([0.0, 0.0, -1.0])  # [B, 3]
        K_bal = self.balance_key(global_trig).view(
            -1, self.gravity_error_dim, Q.shape[-1]
        )  # [B, 3, d]
        A_bal = torch.einsum("bnd,bmd->bnm", Q, K_bal) * self.attn_scale  # [B, 12, 3]
        bal_correction = torch.bmm(A_bal, gravity_error.unsqueeze(-1)).squeeze(-1)  # [B, 12]

        # === Velocity NJP: A_vel(q, dq) @ velocity_error ===
        velocity_error = torch.cat([
            commands[:, :2] - base_lin_vel[:, :2],   # vx, vy tracking error
            commands[:, 2:3] - base_ang_vel[:, 2:3],  # wz tracking error
        ], dim=-1)  # [B, 3]
        K_vel = self.velocity_key(global_trig).view(
            -1, self.velocity_error_dim, Q.shape[-1]
        )  # [B, 3, d]
        A_vel = torch.einsum("bnd,bmd->bnm", Q, K_vel) * self.attn_scale  # [B, 12, 3]
        vel_correction = torch.bmm(A_vel, velocity_error.unsqueeze(-1)).squeeze(-1)  # [B, 12]

        # === Error-magnitude gating ===
        if self.use_gate:
            g_bal = torch.sigmoid(self.balance_gate(gravity_error.norm(dim=-1, keepdim=True)))
            g_vel = torch.sigmoid(self.velocity_gate(velocity_error.norm(dim=-1, keepdim=True)))
            bal_correction = g_bal * bal_correction
            vel_correction = g_vel * vel_correction

        return cpg_action + bal_correction + vel_correction

    # === rsl_rl ActorCritic interface ===

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
            *self.cpg.parameters(),
            *self.joint_query.parameters(),
            *self.balance_key.parameters(),
            *self.velocity_key.parameters(),
            self.joint_id_embed,
            self.log_std,
            self.log_base_freq,
        ]
        if self.use_gate:
            params.extend(self.balance_gate.parameters())
            params.extend(self.velocity_gate.parameters())
        return params

    def get_critic_parameters(self):
        return list(self.critic.parameters())


# ==============================================================================
# Network: Standard MLP Baseline (matches official rsl_rl PPO config)
# ==============================================================================
class ActorCritic_MLP(nn.Module):
    """Standard MLP actor-critic baseline for comparison.

    Matches the official Go1 velocity training config:
        actor:  obs(48) → 128 → 128 → 128 → actions(12)
        critic: obs(48) → 128 → 128 → 128 → 1

    No phase augmentation needed — takes raw 48-dim obs directly.
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [128, 128, 128],
        critic_hidden_dims: list[int] = [128, 128, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # Actor MLP
        actor_layers = []
        in_dim = num_obs
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
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

    # === rsl_rl ActorCritic interface ===

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
        return [*self.actor.parameters(), self.log_std]

    def get_critic_parameters(self):
        return list(self.critic.parameters())


# ==============================================================================
# Configuration
# ==============================================================================
# Set to True to use the standard MLP baseline instead of CPG-NJP
USE_BASELINE_MLP = True
# Environment
env_cfg = UnitreeGo1FlatEnvCfg()
env_cfg.scene.num_envs = 4096 * 2  # MDPO splits envs between two policies
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"
device = "cuda:0"

# Training
max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# CPG phase oscillator (3 independent phases: vx, vy, wz)
init_base_freq = 2 * math.pi * 2.0  # 2 Hz initial frequency (learnable)
control_dt = 0.02                    # sim_dt(0.005) × decimation(4)

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
    entropy_coef=0.01,
    learning_rate=1e-2,
    min_learning_rate=3e-5,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="exponential",
    use_muon=False,
)

# Network architecture
actor_hidden_dims = [128, 128]
critic_hidden_dims = [256, 128]
activation = "elu"
init_noise_std = 1.0
use_error_gate = True


# ==============================================================================
# Step 1: Create environment
# ==============================================================================
env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

log_dir = os.path.join("logs", "rsl_rl", "go1_velocity_flat", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)


# ==============================================================================
# Step 2: Create networks and MDPO algorithm
# ==============================================================================
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs_raw = obs_flat.shape[-1]       # 48
num_phase_features = 2 * 3            # sin/cos per phase (vx, vy, wz) = 6
num_actions = env.num_actions          # 12

if USE_BASELINE_MLP:
    num_obs = num_obs_raw  # 48 (no phase augmentation)
    print(f"[BASELINE MLP] Obs dim: {num_obs}, Action dim: {num_actions}")
    mlp_kwargs = dict(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        init_noise_std=init_noise_std,
    )
    actor_critic_1 = ActorCritic_MLP(**mlp_kwargs)
    actor_critic_2 = ActorCritic_MLP(**mlp_kwargs)
else:
    num_obs = num_obs_raw + num_phase_features  # 54
    print(f"[CPG-NJP] Raw obs dim: {num_obs_raw}, Augmented obs dim: {num_obs} (48+6 phase), Action dim: {num_actions}")
    policy_kwargs = dict(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
        gate=use_error_gate,
        init_base_freq=init_base_freq,
    )
    actor_critic_1 = LocomotionCPG_NJP(**policy_kwargs)
    actor_critic_2 = LocomotionCPG_NJP(**policy_kwargs)

mdpo = MDPO(actor_critic_1, actor_critic_2, device=device, **mdpo_cfg)
mdpo.init_storage(
    num_envs=env.num_envs,
    num_transitions_per_env=num_steps_per_env,
    actor_obs_shape=[num_obs],
    critic_obs_shape=[num_obs],
    action_shape=[num_actions],
)

print(f"Policy: {actor_critic_1.__class__.__name__}")
total_params = sum(p.numel() for p in actor_critic_1.parameters()) + sum(p.numel() for p in actor_critic_2.parameters())
print(f"Total parameters (both policies): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")


# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
arch_name = "baseline_mlp" if USE_BASELINE_MLP else "cpg_njp"
wandb.init(
    project="isaaclab-go1-velocity",
    name=f"mdpo_{arch_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": "baseline-MLP-128x3" if USE_BASELINE_MLP else "CPG-NJP-3phase",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "num_obs": num_obs,
        "num_actions": num_actions,
        **({} if USE_BASELINE_MLP else {
            "cpg_init_base_freq_hz": init_base_freq / (2 * math.pi),
            "cpg_num_phases": 3,
            "control_dt": control_dt,
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
        }),
        "activation": activation,
        **mdpo_cfg,
    },
)


# ==============================================================================
# Phase oscillator helper
# ==============================================================================
def compute_phase_features(phase: torch.Tensor) -> torch.Tensor:
    """Compute sin/cos features for 3 independent phases (vx, vy, wz). Returns [B, 6]."""
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # [B, 6]


# ==============================================================================
# Step 4: Training loop
# ==============================================================================
rewbuffer = deque(maxlen=100)
lenbuffer = deque(maxlen=100)
cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)

# Staggered env resets (helps when rollout_length << episode_length)
env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
obs_td = env.get_observations().to(device)

# Phase oscillator state: 3 independent phases per environment [B, 3]
phase = torch.zeros(env.num_envs, 3, device=device)

if USE_BASELINE_MLP:
    print(f"\nStarting MDPO + Baseline MLP training for {max_iterations} iterations...")
    print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
    print(f"  MLP: actor=[128,128,128], critic=[128,128,128], obs={num_obs}")
else:
    print(f"\nStarting MDPO + CPG-NJP training for {max_iterations} iterations...")
    print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
    print(f"  CPG: {init_base_freq / (2 * math.pi):.1f} Hz init freq (learnable), 3 independent phases (vx, vy, wz)")

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    mdpo.train_mode()
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            obs_flat = _flatten_obs(obs_td)

            if USE_BASELINE_MLP:
                policy_obs = obs_flat  # [B, 48] raw obs
            else:
                # Compute phase features and augment obs
                phase_feat = compute_phase_features(phase)  # [B, 6]
                policy_obs = torch.cat([obs_flat, phase_feat], dim=-1)  # [B, 54]

            # MDPO splits envs between the two policies internally
            actions = mdpo.act(policy_obs, policy_obs)

            # Step environment
            obs_td, rewards, dones, extras = env.step(actions.detach().to(env.device))
            obs_td = obs_td.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            # Advance 3 independent phase oscillators (only for CPG-NJP)
            if not USE_BASELINE_MLP:
                cmd = obs_flat[:, 9:12]  # velocity commands [vx, vy, wz]
                base_freq = torch.zeros(env.num_envs, 3, device=device)
                base_freq[mdpo.indices_1] = actor_critic_1.log_base_freq.exp()
                base_freq[mdpo.indices_2] = actor_critic_2.log_base_freq.exp()
                freq = base_freq * cmd.abs()  # [B, 3]
                phase = (phase + freq * control_dt) % (2 * math.pi)
                phase[dones.bool()] = 0.0  # reset all 3 phases on episode resets

            # MDPO stores transitions and resets hidden states
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
    if USE_BASELINE_MLP:
        last_obs = obs_flat
    else:
        phase_feat = compute_phase_features(phase)
        last_obs = torch.cat([obs_flat, phase_feat], dim=-1)
    mdpo.compute_returns(last_obs)
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
            f"fps={fps}  "
            f"collect={collection_time:.2f}s  learn={learn_time:.2f}s"
        )

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        ckpt_data = {
            "iter": iteration + 1,
            "architecture": "baseline_mlp" if USE_BASELINE_MLP else "cpg_njp",
            "model_1_state_dict": actor_critic_1.state_dict(),
            "model_2_state_dict": actor_critic_2.state_dict(),
            "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
            "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
        }
        if not USE_BASELINE_MLP:
            ckpt_data["cpg_config"] = {
                "control_dt": control_dt,
                "num_phases": 3,
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
