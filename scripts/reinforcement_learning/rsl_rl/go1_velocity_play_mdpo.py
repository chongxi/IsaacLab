"""
Play a trained MDPO checkpoint for Isaac-Velocity-Flat-Unitree-Go1-v0.

Supports two architectures (auto-detected from checkpoint):
    - "baseline_mlp": standard MLP actor-critic [128,128,128]
    - "cpg_njp": CPG feedforward + dual-NJP reflex with 3 phase oscillators

MDPO trains TWO policies simultaneously. The checkpoint contains both:
    - "model_1_state_dict": policy 1
    - "model_2_state_dict": policy 2
    - "architecture": "baseline_mlp" or "cpg_njp"

By default this script loads policy 1. Set POLICY_INDEX = 2 to play policy 2.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/go1_velocity_play_mdpo.py
"""

# ==============================================================================
# Step 0: Launch Isaac Sim with GUI
# ==============================================================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ==============================================================================
# Imports
# ==============================================================================
import math

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
)


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


# ==============================================================================
# Network: LocomotionCPG_NJP (must match training script exactly)
# ==============================================================================
class LocomotionCPG_NJP(nn.Module):
    """CPG feedforward + dual-NJP reflex for quadruped locomotion.

    Actor:
        action = f_cpg(q, dq, prev_a, phase, cmd)  # gait generator
               + A_bal(q, dq) @ gravity_error       # balance reflex
               + A_vel(q, dq) @ velocity_error      # velocity correction
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
        self.velocity_error_dim = 3
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
        self.joint_query = layer_init(nn.Linear(4, attn_dim), std=0.01)
        self.joint_id_embed = nn.Parameter(torch.zeros(self.num_joints, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)

        key_input_dim = 4 * self.num_joints
        self.balance_key = nn.Sequential(
            layer_init(nn.Linear(key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.gravity_error_dim * attn_dim), std=0.01),
        )
        self.attn_scale = float(attn_dim) ** -0.5

        # === Velocity NJP pathway ===
        self.velocity_key = nn.Sequential(
            layer_init(nn.Linear(key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.velocity_error_dim * attn_dim), std=0.01),
        )

        # Error-magnitude gating
        self.use_gate = gate
        if gate:
            self.balance_gate = layer_init(nn.Linear(1, 1), std=0.01)
            self.velocity_gate = layer_init(nn.Linear(1, 1), std=0.01)

        # === Critic ===
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

        # CPG feedforward (with prev_actions for sequential behavior)
        cpg_input = torch.cat([joint_pos, joint_vel, prev_actions, phase_features, commands], dim=-1)
        cpg_action = self.cpg(cpg_input)

        # Shared query
        joint_tokens = torch.stack([
            torch.sin(joint_pos), torch.cos(joint_pos),
            torch.sin(joint_vel), torch.cos(joint_vel),
        ], dim=-1)
        Q = self.joint_query(joint_tokens) + self.joint_id_embed.unsqueeze(0)

        global_trig = torch.cat([
            torch.sin(joint_pos), torch.cos(joint_pos),
            torch.sin(joint_vel), torch.cos(joint_vel),
        ], dim=-1)

        # Balance NJP
        gravity_error = projected_gravity - projected_gravity.new_tensor([0.0, 0.0, -1.0])
        K_bal = self.balance_key(global_trig).view(-1, self.gravity_error_dim, Q.shape[-1])
        A_bal = torch.einsum("bnd,bmd->bnm", Q, K_bal) * self.attn_scale
        bal_correction = torch.bmm(A_bal, gravity_error.unsqueeze(-1)).squeeze(-1)

        # Velocity NJP
        velocity_error = torch.cat([
            commands[:, :2] - base_lin_vel[:, :2],
            commands[:, 2:3] - base_ang_vel[:, 2:3],
        ], dim=-1)
        K_vel = self.velocity_key(global_trig).view(-1, self.velocity_error_dim, Q.shape[-1])
        A_vel = torch.einsum("bnd,bmd->bnm", Q, K_vel) * self.attn_scale
        vel_correction = torch.bmm(A_vel, velocity_error.unsqueeze(-1)).squeeze(-1)

        # Gating
        if self.use_gate:
            g_bal = torch.sigmoid(self.balance_gate(gravity_error.norm(dim=-1, keepdim=True)))
            g_vel = torch.sigmoid(self.velocity_gate(velocity_error.norm(dim=-1, keepdim=True)))
            bal_correction = g_bal * bal_correction
            vel_correction = g_vel * vel_correction

        return cpg_action + bal_correction + vel_correction

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self._actor_mean(obs)


# ==============================================================================
# Network: Standard MLP Baseline (must match training script exactly)
# ==============================================================================
class ActorCritic_MLP(nn.Module):
    """Standard MLP actor-critic baseline.

    actor:  obs(48) → 128 → 128 → 128 → actions(12)
    critic: obs(48) → 128 → 128 → 128 → 1
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

        actor_layers = []
        in_dim = num_obs
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

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

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


# ==============================================================================
# Checkpoint — update this path to the checkpoint you want to play
# ==============================================================================
# CHECKPOINT_PATH = "logs/rsl_rl/go1_velocity_flat/REPLACE_WITH_TIMESTAMP/model_1500.pt"
# CHECKPOINT_PATH = "logs/rsl_rl/go1_velocity_flat/2026-03-01_23-17-19/model_400.pt"
CHECKPOINT_PATH = "logs/rsl_rl/go1_velocity_flat/2026-03-01_23-24-21/model_1500.pt"

# Which of the two MDPO policies to play (1 or 2)
POLICY_INDEX = 1

# ==============================================================================
# Network config — must match the training run
# ==============================================================================
ACTOR_HIDDEN_DIMS = [128, 128]
CRITIC_HIDDEN_DIMS = [256, 128]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0
USE_ERROR_GATE = True

# ==============================================================================
# Environment (play config: fewer envs, no noise)
# ==============================================================================
env_cfg = UnitreeGo1FlatEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"
env_cfg.observations.policy.enable_corruption = False

env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

# ==============================================================================
# Build policy and load checkpoint
# ==============================================================================
device = env.unwrapped.device
obs, info = env.reset()
obs_flat = _flatten_obs(obs)
num_obs_raw = obs_flat.shape[-1]  # 48
num_actions = env.num_actions       # 12

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
state_dict_key = f"model_{POLICY_INDEX}_state_dict"

# Auto-detect architecture from checkpoint (fallback to cpg_njp for old checkpoints)
arch = checkpoint.get("architecture", "cpg_njp")
is_baseline_mlp = (arch == "baseline_mlp")

if is_baseline_mlp:
    num_obs = num_obs_raw  # 48
    policy = ActorCritic_MLP(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        init_noise_std=INIT_NOISE_STD,
    ).to(device)
else:
    num_obs = num_obs_raw + 6  # 54
    policy = LocomotionCPG_NJP(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation=ACTIVATION,
        init_noise_std=INIT_NOISE_STD,
        gate=USE_ERROR_GATE,
    ).to(device)

policy.load_state_dict(checkpoint[state_dict_key])
policy.eval()

print(f"Loaded MDPO checkpoint: {CHECKPOINT_PATH}")
print(f"  Policy index: {POLICY_INDEX} of 2")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  Architecture: {arch}")
print(f"  Policy class: {policy.__class__.__name__}")

if is_baseline_mlp:
    print(f"  MLP: actor=[128,128,128], critic=[128,128,128], obs={num_obs}")
else:
    cpg_cfg = checkpoint.get("cpg_config", {})
    control_dt = cpg_cfg.get("control_dt", 0.02)
    learned_freq_hz = policy.log_base_freq.exp() / (2 * math.pi)
    print(f"  Network: actor={ACTOR_HIDDEN_DIMS}, critic={CRITIC_HIDDEN_DIMS}, act={ACTIVATION}")
    print(f"  Obs dim: {num_obs_raw} raw + 6 phase = {num_obs}, Action dim: {num_actions}")
    print(f"  Learned CPG freq (Hz): vx={learned_freq_hz[0]:.2f}, vy={learned_freq_hz[1]:.2f}, wz={learned_freq_hz[2]:.2f}")

# ==============================================================================
# Phase oscillator helper
# ==============================================================================
def compute_phase_features(phase: torch.Tensor) -> torch.Tensor:
    """Compute sin/cos features for 3 independent phases (vx, vy, wz). Returns [B, 6]."""
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


# ==============================================================================
# Play loop
# ==============================================================================
phase = torch.zeros(env.num_envs, 3, device=device)

while simulation_app.is_running():
    with torch.inference_mode():
        obs_flat = _flatten_obs(obs)

        if is_baseline_mlp:
            policy_obs = obs_flat  # [B, 48]
        else:
            phase_feat = compute_phase_features(phase)  # [B, 6]
            policy_obs = torch.cat([obs_flat, phase_feat], dim=-1)  # [B, 54]

        # Inference (deterministic, no noise)
        actions = policy.act_inference(policy_obs)

        # Step environment
        obs, _, dones, _ = env.step(actions)

        # Advance phase oscillators (only for CPG-NJP)
        if not is_baseline_mlp and isinstance(policy, LocomotionCPG_NJP):
            cmd = obs_flat[:, 9:12]  # [vx, vy, wz]
            freq = policy.log_base_freq.exp() * cmd.abs()  # [B, 3]
            phase = (phase + freq * control_dt) % (2 * math.pi)
            phase[dones.bool()] = 0.0
