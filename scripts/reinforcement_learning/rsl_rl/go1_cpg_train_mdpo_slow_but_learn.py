"""
Training script for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO + MANC CPG (RNN) + MLP Reflex.

Architecture: Differentiable CPG (top-down) + MLP reflex (bottom-up)
    action = cpg_offsets(cmd, cpg_state)  # rhythmic gait from 4 coupled MANC oscillators
           + reflex_mlp(obs)              # sensory corrections for balance/tracking

The MANC CPG is a 3-neuron oscillator with internal state (x, a). When kept differentiable
(no @torch.no_grad, no .detach()), gradients flow through Euler integration steps via BPTT.
This allows RL to learn the decoder: W_delta (readout), w_dn/b_dn (amplitude), w_sht/b_sht (frequency).

CPG state maps to LSTM's (h, c) for recurrent storage:
    cpg_x: (B, 4, 3) → flatten → (1, B, 12) as "hidden state"
    cpg_a: (B, 4, 3) → flatten → (1, B, 12) as "cell state"

Obs layout (48 dims):
    [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), commands(3),
     joint_pos(12), joint_vel(12), prev_actions(12)]

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_cpg_train_mdpo.py
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
import torch.nn.functional as F
import wandb
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import MDPO
from rsl_rl.utils import unpad_trajectories

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
# Stagger initialization: compute trot-phased initial CPG states
# ==============================================================================
def stagger_init_cpg(device, dt=0.01):
    """Run a single MANC CPG to its limit cycle, detect the period, and return
    initial (x, a) states for 4 legs at trot phase offsets.

    Returns:
        init_x: (4, 3) membrane potentials
        init_a: (4, 3) adaptation currents
    """
    # MANC constants
    tau_x = torch.tensor([0.18, 0.24, 0.35])
    tau_a = torch.tensor([0.80, 1.00, 1e6])
    g_adapt = torch.tensor([1.8, 1.5, 0.0])
    W_rec = torch.tensor([
        [0.00, 0.54, -6.30],
        [6.30, 0.00, -2.16],
        [0.36, 1.80,  0.00],
    ])
    W_in = torch.tensor([2.25, 0.0, 0.0])
    bias = torch.tensor([0.0, -0.2, -0.3])

    # Warm up to reach limit cycle
    x = torch.randn(3) * 0.01
    a = torch.zeros(3)
    for _ in range(500):
        r = torch.tanh(x)
        tau_x_eff = tau_x / (0.3 + 0.7 * 0.5)
        tau_a_eff = tau_a / (0.2 + 2.5 * 0.5)
        dxdt = (-x + W_rec @ r + W_in + bias - a) / tau_x_eff
        dadt = (-a + g_adapt * r) / tau_a_eff
        x = x + dt * dxdt
        a = a + dt * dadt

    # Record states along limit cycle
    states_x, states_a = [], []
    for _ in range(300):
        r = torch.tanh(x)
        tau_x_eff = tau_x / (0.3 + 0.7 * 0.5)
        tau_a_eff = tau_a / (0.2 + 2.5 * 0.5)
        dxdt = (-x + W_rec @ r + W_in + bias - a) / tau_x_eff
        dadt = (-a + g_adapt * r) / tau_a_eff
        x = x + dt * dxdt
        a = a + dt * dadt
        states_x.append(x.clone())
        states_a.append(a.clone())

    states_x = torch.stack(states_x)  # (300, 3)
    states_a = torch.stack(states_a)

    # Find period via E1 positive zero-crossings
    r_e1 = torch.tanh(states_x[:, 0])
    crossings = []
    for i in range(1, len(r_e1)):
        if r_e1[i - 1] < 0 and r_e1[i] >= 0:
            crossings.append(i)

    if len(crossings) >= 2:
        period = crossings[-1] - crossings[-2]
        cycle_start = crossings[-2]
    else:
        period = 100
        cycle_start = 0

    # Trot phase offsets: FR=0°, FL=180°, RR=180°, RL=0°
    phase_offsets = [0.0, 0.5, 0.5, 0.0]
    init_x = torch.zeros(4, 3)
    init_a = torch.zeros(4, 3)
    for leg in range(4):
        idx = cycle_start + int(phase_offsets[leg] * period)
        idx = min(idx, len(states_x) - 1)
        init_x[leg] = states_x[idx]
        init_a[leg] = states_a[idx]

    return init_x.to(device), init_a.to(device)


# ==============================================================================
# BatchedQuadrupedDecoder: cmd → (dn, sht, W_eff)
# ==============================================================================
class BatchedQuadrupedDecoder(nn.Module):
    """Maps batched [vx, vy, omega] → (dn, sht, W_eff) for 4-leg quadruped.

    Learnable parameters (80 total):
        w_dn(3) + b_dn(1) = 4
        w_sht(3) + b_sht(1) = 4
        W_delta(4, 6, 3) = 72
    """

    def __init__(self, cmd_dim=3):
        super().__init__()
        self.cmd_dim = cmd_dim
        self.n_legs = 4
        self.joints_per_leg = 3

        self.w_dn = nn.Parameter(torch.zeros(cmd_dim))
        self.b_dn = nn.Parameter(torch.tensor(2.0))
        self.w_sht = nn.Parameter(torch.zeros(cmd_dim))
        self.b_sht = nn.Parameter(torch.tensor(0.0))
        self.W_delta = nn.Parameter(torch.zeros(4, self.joints_per_leg * 2, cmd_dim))
        self.register_buffer("W_base", torch.zeros(4, self.joints_per_leg, 2))

    def forward(self, cmd):
        """
        Args:
            cmd: (B, 3) velocity commands
        Returns:
            dn:    (B,) DN gate in (0, 1)
            sht:   (B,) serotonin > 0.3
            W_eff: (B, 4, 3, 2) per-leg readout matrices
        """
        # (B, 3) @ (3,) → (B,)
        dn = torch.sigmoid(cmd @ self.w_dn + self.b_dn)
        sht = F.softplus(cmd @ self.w_sht + self.b_sht) + 0.3

        # (4, 6, 3) @ (B, 3, 1) via einsum → (B, 4, 6) → (B, 4, 3, 2)
        dW = torch.einsum("lfc,bc->blf", self.W_delta, cmd)  # (B, 4, 6)
        dW = dW.reshape(-1, 4, self.joints_per_leg, 2)  # (B, 4, 3, 2)
        W_eff = self.W_base.unsqueeze(0) + dW  # (B, 4, 3, 2)

        return dn, sht, W_eff


# ==============================================================================
# CPG_Reflex_ActorCritic: recurrent policy with differentiable CPG + MLP reflex
# ==============================================================================
class CPG_Reflex_ActorCritic(nn.Module):
    """Recurrent actor-critic with differentiable MANC CPG and MLP reflex.

    CPG state (cpg_x, cpg_a) maps to LSTM (h, c) for recurrent storage.
    """

    is_recurrent = True

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        reflex_hidden_dims: list[int] = [128, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        cpg_dt: float = 0.01,
        cpg_substeps: int = 2,
        coupling_strength: float = 1.5,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions  # 12
        self.cpg_dt = cpg_dt
        self.cpg_substeps = cpg_substeps
        self.n_legs = 4
        self.n_neurons = 3
        self.coupling_strength = coupling_strength
        self._device = device

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # === CPG constants (frozen buffers) ===
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

        # Trot coupling matrix (4x4)
        self.register_buffer("C", torch.tensor([
            [ 0., -1., -1.,  1.],
            [-1.,  0.,  1., -1.],
            [-1.,  1.,  0., -1.],
            [ 1., -1., -1.,  0.],
        ]))

        # Stagger initialization values (precomputed on CPU, moved to device)
        init_x, init_a = stagger_init_cpg(device)
        self.register_buffer("init_cpg_x", init_x)  # (4, 3)
        self.register_buffer("init_cpg_a", init_a)  # (4, 3)

        # === Learnable decoder: cmd → CPG control ===
        self.decoder = BatchedQuadrupedDecoder(cmd_dim=3)
        # Initialize W_base with reasonable defaults for Go1 leg motion
        W_base = torch.tensor([
            [0.00, 0.00],   # abduction
            [0.20, 0.10],   # hip
            [0.05, -0.20],  # knee
        ])
        self.decoder.W_base.copy_(W_base.unsqueeze(0).expand(4, -1, -1))

        # === Reflex MLP: obs → corrections ===
        reflex_layers = []
        in_dim = num_obs
        for h_dim in reflex_hidden_dims:
            reflex_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            reflex_layers.append(act_fn())
            in_dim = h_dim
        reflex_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.reflex_mlp = nn.Sequential(*reflex_layers)

        # === Critic MLP: obs → value ===
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

        # === CPG state (initialized later in init_cpg_state) ===
        self.cpg_x: torch.Tensor | None = None  # (B, 4, 3)
        self.cpg_a: torch.Tensor | None = None  # (B, 4, 3)

    def init_cpg_state(self, num_envs: int):
        """Initialize CPG state for all environments to staggered trot."""
        self.cpg_x = self.init_cpg_x.unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.cpg_a = self.init_cpg_a.unsqueeze(0).expand(num_envs, -1, -1).clone()

    def _cpg_euler_step(self, cpg_x, cpg_a, dn, sht):
        """Differentiable Euler step for 4 coupled MANC CPGs.

        Args:
            cpg_x: (B, 4, 3) membrane potentials
            cpg_a: (B, 4, 3) adaptation currents
            dn:    (B,) DN gate
            sht:   (B,) serotonin

        Returns:
            new_r:   (B, 4, 3) firing rates after step
            new_cpg_x: (B, 4, 3) updated membrane potentials
            new_cpg_a: (B, 4, 3) updated adaptation currents
        """
        for _ in range(self.cpg_substeps):
            r = torch.tanh(cpg_x)  # (B, 4, 3)

            # Effective time constants modulated by serotonin
            # sht: (B,) → (B, 1, 1) for broadcasting
            sht_3d = sht.unsqueeze(-1).unsqueeze(-1)
            tau_x_eff = self.tau_x / (0.3 + 0.7 * sht_3d)  # (B, 1, 3)
            tau_a_eff = self.tau_a / (0.2 + 2.5 * sht_3d)   # (B, 1, 3)

            # Inter-leg coupling on E1 neurons
            r_e1 = r[:, :, 0]  # (B, 4)
            coupling = self.coupling_strength * (r_e1 @ self.C.T)  # (B, 4)

            # Recurrent input: dn * (W_rec @ r) for each leg
            # r: (B, 4, 3), W_rec: (3, 3) → (B, 4, 3)
            dn_3d = dn.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            rec_input = dn_3d * torch.einsum("bln,mn->blm", r, self.W_rec)  # (B, 4, 3)

            # External input
            ext_input = self.W_in * dn.unsqueeze(-1)  # (B, 3)

            # Coupling only on E1 (neuron index 0)
            coupling_input = torch.zeros_like(cpg_x)  # (B, 4, 3)
            coupling_input[:, :, 0] = coupling

            # dx/dt
            dxdt = (
                -cpg_x + rec_input + ext_input.unsqueeze(1) + self.bias + coupling_input - cpg_a
            ) / tau_x_eff

            # da/dt
            dadt = (-cpg_a + self.g_adapt * r) / tau_a_eff

            cpg_x = cpg_x + self.cpg_dt * dxdt
            cpg_a = cpg_a + self.cpg_dt * dadt

        new_r = torch.tanh(cpg_x)
        return new_r, cpg_x, cpg_a

    def _compute_action_mean(self, obs, cpg_x, cpg_a):
        """Compute action mean from CPG + reflex for a single timestep.

        Args:
            obs:   (B, 48)
            cpg_x: (B, 4, 3)
            cpg_a: (B, 4, 3)

        Returns:
            action_mean: (B, 12)
            new_cpg_x:   (B, 4, 3)
            new_cpg_a:   (B, 4, 3)
        """
        # Extract commands for decoder
        cmd = obs[:, 9:12]  # (B, 3)
        dn, sht, W_eff = self.decoder(cmd)  # dn:(B,), sht:(B,), W_eff:(B,4,3,2)

        # Differentiable CPG step
        new_r, new_cpg_x, new_cpg_a = self._cpg_euler_step(cpg_x, cpg_a, dn, sht)

        # Per-leg readout: W_eff (B,4,3,2) @ r[:,:,:2] (B,4,2) → (B,4,3)
        cpg_offsets = torch.einsum("bljn,bln->blj", W_eff, new_r[:, :, :2])  # (B, 4, 3)
        cpg_offsets = cpg_offsets.reshape(-1, 12)  # (B, 12)

        # Reflex corrections
        corrections = self.reflex_mlp(obs)  # (B, 12)

        action_mean = cpg_offsets + corrections
        return action_mean, new_cpg_x, new_cpg_a

    # === rsl_rl ActorCritic interface ===

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        """Sample actions.

        Single-step mode (collection): observations is (B, 48)
        Batch mode (training): observations is (L, num_traj, 48) with masks
        """
        batch_mode = masks is not None

        if batch_mode:
            # Restore CPG state from saved hidden states
            # hidden_states is a tuple (cpg_x_flat, cpg_a_flat) each (1, num_traj, 12)
            if isinstance(hidden_states, (list, tuple)):
                cpg_x_flat = hidden_states[0]  # (1, num_traj, 12)
                cpg_a_flat = hidden_states[1]  # (1, num_traj, 12)
            else:
                cpg_x_flat = hidden_states
                cpg_a_flat = torch.zeros_like(cpg_x_flat)

            cpg_x = cpg_x_flat.squeeze(0).reshape(-1, 4, 3)  # (num_traj, 4, 3)
            cpg_a = cpg_a_flat.squeeze(0).reshape(-1, 4, 3)

            L = observations.shape[0]
            means_list = []

            # Loop over timesteps for BPTT
            for t in range(L):
                obs_t = observations[t]  # (num_traj, 48)
                mean_t, cpg_x, cpg_a = self._compute_action_mean(obs_t, cpg_x, cpg_a)
                means_list.append(mean_t)

            means_padded = torch.stack(means_list, dim=0)  # (L, num_traj, 12)
            means = unpad_trajectories(means_padded, masks)  # (num_steps, mini_batch_envs, 12)

            std = self.log_std.exp().expand_as(means)
            self.distribution = Normal(means, std)
            return self.distribution.sample()
        else:
            # Single-step mode during collection
            mean, self.cpg_x, self.cpg_a = self._compute_action_mean(
                observations, self.cpg_x, self.cpg_a
            )
            std = self.log_std.exp().expand_as(mean)
            self.distribution = Normal(mean, std)
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        """Evaluate state values. Critic is feedforward (no CPG state needed)."""
        if masks is not None:
            # Batch mode: unpad then pass through critic
            values_padded = self.critic(critic_observations)  # (L, num_traj, 1)
            values = unpad_trajectories(values_padded, masks)  # (num_steps, mini_batch_envs, 1)
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
        """Return CPG state in LSTM (h, c) format for storage."""
        cpg_x_flat = self.cpg_x.reshape(1, -1, 12)  # (1, B, 12)
        cpg_a_flat = self.cpg_a.reshape(1, -1, 12)   # (1, B, 12)
        # Both actor and critic slots get the same CPG state
        return (cpg_x_flat, cpg_a_flat), (cpg_x_flat, cpg_a_flat)

    def reset(self, dones=None):
        """Reset CPG state for done environments to staggered trot."""
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
        return [
            *self.decoder.parameters(),
            *self.reflex_mlp.parameters(),
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
device = "cuda:0"

max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# CPG
cpg_dt = 0.01
cpg_substeps = 2
coupling_strength = 1.5

# MDPO
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

# Network
reflex_hidden_dims = [128, 128]
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

print(f"[CPG-Reflex] Obs dim: {num_obs}, Action dim: {num_actions}")

policy_kwargs = dict(
    num_obs=num_obs,
    num_actions=num_actions,
    reflex_hidden_dims=reflex_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation=activation,
    init_noise_std=init_noise_std,
    cpg_dt=cpg_dt,
    cpg_substeps=cpg_substeps,
    coupling_strength=coupling_strength,
    device=device,
)
actor_critic_1 = CPG_Reflex_ActorCritic(**policy_kwargs).to(device)
actor_critic_2 = CPG_Reflex_ActorCritic(**policy_kwargs).to(device)

# Initialize CPG states — each policy handles half the envs
num_envs_1 = env.num_envs // 2
num_envs_2 = env.num_envs - num_envs_1
actor_critic_1.init_cpg_state(num_envs_1)
actor_critic_2.init_cpg_state(num_envs_2)

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
print(f"  Decoder params: {sum(p.numel() for p in actor_critic_1.decoder.parameters())}")
print(f"  Reflex MLP params: {sum(p.numel() for p in actor_critic_1.reflex_mlp.parameters())}")
print(f"  Critic params: {sum(p.numel() for p in actor_critic_1.critic.parameters())}")
print(f"Total parameters (both policies): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")


# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
wandb.init(
    project="isaaclab-go1-velocity",
    name=f"mdpo_cpg_reflex_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": "CPG-Reflex-RNN",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "num_obs": num_obs,
        "num_actions": num_actions,
        "cpg_dt": cpg_dt,
        "cpg_substeps": cpg_substeps,
        "coupling_strength": coupling_strength,
        "reflex_hidden_dims": reflex_hidden_dims,
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

print(f"\nStarting MDPO + CPG-Reflex training for {max_iterations} iterations...")
print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
print(f"  CPG: dt={cpg_dt}, substeps={cpg_substeps}, coupling={coupling_strength}")
print(f"  Reflex MLP: {reflex_hidden_dims}, Critic: {critic_hidden_dims}")

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    mdpo.train_mode()
    with torch.inference_mode():
        for step in range(num_steps_per_env):
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

    # Log CPG diagnostics
    if iteration % 10 == 0:
        with torch.no_grad():
            r1 = torch.tanh(actor_critic_1.cpg_x)  # (B1, 4, 3)
            e1_rates_1 = r1[:, :, 0].mean(dim=0)  # (4,) mean E1 per leg
            log_dict.update({
                "CPG/E1_FR": e1_rates_1[0].item(),
                "CPG/E1_FL": e1_rates_1[1].item(),
                "CPG/E1_RR": e1_rates_1[2].item(),
                "CPG/E1_RL": e1_rates_1[3].item(),
                "CPG/dn_mean": torch.sigmoid(
                    _flatten_obs(obs_td)[mdpo.indices_1][:, 9:12] @ actor_critic_1.decoder.w_dn
                    + actor_critic_1.decoder.b_dn
                ).mean().item(),
            })

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

    # Gradient check on first iteration
    if iteration == 0:
        grad_info = []
        for name, p in actor_critic_1.decoder.named_parameters():
            if p.grad is not None:
                grad_info.append(f"    {name}: grad_norm={p.grad.norm().item():.6f}")
            else:
                grad_info.append(f"    {name}: grad=None")
        print("  [Gradient check] Decoder parameters after first update:")
        for g in grad_info:
            print(g)

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        ckpt_data = {
            "iter": iteration + 1,
            "architecture": "cpg_reflex",
            "model_1_state_dict": actor_critic_1.state_dict(),
            "model_2_state_dict": actor_critic_2.state_dict(),
            "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
            "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
            "cpg_config": {
                "cpg_dt": cpg_dt,
                "cpg_substeps": cpg_substeps,
                "coupling_strength": coupling_strength,
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
