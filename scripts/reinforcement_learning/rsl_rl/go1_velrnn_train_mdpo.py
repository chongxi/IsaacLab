"""
Training script for Isaac-Velocity-Flat-Unitree-Go1-v0 with MDPO + Velocity-Modulated RNN CPG.

Architecture: Vanilla RNN where recurrent weights are velocity-command-modulated:
    W_rec(t) = vx·W_vx + vy·W_vy + omega·W_omega
    r_new = normalize(r + dt·(-r + W_rec @ r + bias))

    - When cmd=0: W_rec=0, r stays fixed (standing still)
    - When cmd≠0: W_rec has complex eigenvalues → r oscillates on unit sphere (walking)
    - Different cmd → different W_rec → different oscillation patterns (speed, direction, turning)
    - No hand-designed DN, no phase coupling — velocity IS the dynamics

    Motor MLP: [r_per_leg(k) + base_vel(3) + base_ang_vel(3) + gravity(3) + joint_pos(3) + joint_vel(3)] → 3 actions
    Per-leg shared MLP (same weights for all 4 legs)

Inspired by:
    - Pugliese et al. (2025): 3-neuron CPG (E1-E2-I1) per leg, DNg100 controls frequency
    - DNa01/02: separate steering neurons encode turning via L-R difference
    - Key insight: velocity command modulates synaptic weights (neuromodulation)

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_velrnn_train_mdpo.py
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
# VelRNN_ActorCritic: Velocity-Modulated RNN CPG
# ==============================================================================
class VelRNN_ActorCritic(nn.Module):
    """Three independent velocity-modulated RNN CPGs for locomotion.

    Each velocity component drives its own CPG:
        r_vx_new  = normalize(vx  * W_vx  @ r_vx)  @ C   — forward/backward oscillation
        r_vy_new  = normalize(vy  * W_vy  @ r_vy)  @ C   — lateral oscillation
        r_om_new  = normalize(om  * W_om  @ r_om)  @ C   — turning oscillation

    When a velocity component is zero, its CPG freezes (standing behavior for that DOF).
    When nonzero, the orthogonal W rotates the state → oscillation.

    Motor MLP (per-leg shared):
        [r_vx_leg(k), r_vy_leg(k), r_om_leg(k), base_vel(3), base_ang_vel(3),
         gravity(3), joint_pos(3), joint_vel(3)] → 3 actions
    """

    is_recurrent = True

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        n_neurons: int = 12,
        rnn_dt: float = 0.1,
        rnn_substeps: int = 1,
        cos_coupling_alpha: float = 0.5,
        bptt_length: int = 0,
        actor_hidden_dims: list[int] = [128, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions  # 12
        self.n_neurons = n_neurons
        self.rnn_dt = rnn_dt
        self.rnn_substeps = rnn_substeps
        self.bptt_length = bptt_length
        self._device = device

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        # === Three independent CPG recurrence matrices ===
        n = n_neurons
        self.W_vx = nn.Parameter(torch.empty(n, n))
        self.W_vy = nn.Parameter(torch.empty(n, n))
        self.W_omega = nn.Parameter(torch.empty(n, n))
        nn.init.orthogonal_(self.W_vx)
        nn.init.orthogonal_(self.W_vy)
        nn.init.orthogonal_(self.W_omega)

        # === Per-CPG proprioceptive feedback MLPs ===
        # Input: base_vel(3) + base_ang_vel(3) + gravity(3) + joint_pos(12) + joint_vel(12) = 33
        # Output: n_neurons (injected into RNN dynamics)
        fb_input_dim = 3 + 3 + 3 + 12 + 12  # 33
        fb_hidden = 64
        self.fb_vx = nn.Sequential(
            layer_init(nn.Linear(fb_input_dim, fb_hidden)), act_fn(),
            layer_init(nn.Linear(fb_hidden, n), std=0.01),
        )
        self.fb_vy = nn.Sequential(
            layer_init(nn.Linear(fb_input_dim, fb_hidden)), act_fn(),
            layer_init(nn.Linear(fb_hidden, n), std=0.01),
        )
        self.fb_omega = nn.Sequential(
            layer_init(nn.Linear(fb_input_dim, fb_hidden)), act_fn(),
            layer_init(nn.Linear(fb_hidden, n), std=0.01),
        )

        # === Soft cos(i-j) coupling matrix (shared across all 3 CPGs) ===
        idx = torch.arange(n, dtype=torch.float32)
        C_cos = torch.cos(2 * math.pi * (idx.unsqueeze(0) - idx.unsqueeze(1)) / n)
        C_cos = C_cos / (n / 2)  # spectral norm = 1
        C = cos_coupling_alpha * C_cos + (1 - cos_coupling_alpha) * torch.eye(n)
        self.register_buffer("cos_coupling", C)

        # === Motor MLP: all 3 CPG states + obs → 12 joint actions ===
        # Input: r_vx(n) + r_vy(n) + r_omega(n) + base_vel(3) + base_ang_vel(3) + gravity(3) + joint_pos(12) + joint_vel(12)
        motor_input_dim = 3 * n_neurons + 3 + 3 + 3 + 12 + 12
        motor_output_dim = num_actions  # 12
        motor_layers = []
        in_dim = motor_input_dim
        for h_dim in actor_hidden_dims:
            motor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            motor_layers.append(act_fn())
            in_dim = h_dim
        motor_layers.append(layer_init(nn.Linear(in_dim, motor_output_dim), std=0.01))
        self.motor_mlp = nn.Sequential(*motor_layers)

        # === Critic MLP: full obs(48) → value ===
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

        # === RNN states: 3 independent CPGs packed as (B, 3*n) ===
        self.r: torch.Tensor | None = None  # (B, 3*n_neurons)

    def init_cpg_state(self, num_envs: int):
        """Initialize all 3 CPG states at rest (zeros → standing)."""
        self.r = torch.zeros(num_envs, 3 * self.n_neurons, device=self._device)

    def _rnn_step_single(self, r, cmd_scalar, W, fb):
        """One CPG step: rate-RNN with proprioceptive feedback.

        dr/dt = -r + cmd * W @ r + fb
        r_new = tanh(r + dt * dr)

        - Leak (-r): decays r to zero when undriven (standing)
        - cmd * W @ r: velocity-driven rotation, magnitude scales amplitude
        - fb: proprioceptive feedback anchors r to actual body state
        - tanh: bounds r to [-1,1], prevents explosion while preserving amplitude
        - Larger cmd → larger |r| → motor MLP produces bigger movements
        """
        dt = self.rnn_dt
        dr = -r + cmd_scalar * (r @ W.T) + fb  # (B, n)
        r_new = torch.tanh(r + dt * dr)
        return r_new

    def _rnn_step(self, r_packed, cmd, proprio):
        """Step all 3 CPGs with proprioceptive feedback.

        Args:
            r_packed: (B, 3*n) packed state [r_vx, r_vy, r_omega]
            cmd: (B, 3) velocity command [vx, vy, omega]
            proprio: (B, 33) proprioception [base_vel, base_ang_vel, gravity, joint_pos, joint_vel]

        Returns:
            r_packed_new: (B, 3*n) updated packed state
        """
        n = self.n_neurons
        r_vx = r_packed[:, :n]
        r_vy = r_packed[:, n:2*n]
        r_om = r_packed[:, 2*n:]

        vx = cmd[:, 0:1]      # (B, 1)
        vy = cmd[:, 1:2]      # (B, 1)
        omega = cmd[:, 2:3]   # (B, 1)

        # Each CPG gets its own proprioceptive feedback
        fb_vx = self.fb_vx(proprio)       # (B, n)
        fb_vy = self.fb_vy(proprio)       # (B, n)
        fb_om = self.fb_omega(proprio)    # (B, n)

        r_vx_new = self._rnn_step_single(r_vx, vx, self.W_vx, fb_vx)
        r_vy_new = self._rnn_step_single(r_vy, vy, self.W_vy, fb_vy)
        r_om_new = self._rnn_step_single(r_om, omega, self.W_omega, fb_om)

        return torch.cat([r_vx_new, r_vy_new, r_om_new], dim=-1)

    def _compute_action_mean(self, obs, r_packed):
        """Step 3 CPGs with proprioceptive feedback + motor MLP → 12 joint actions.

        Motor MLP input:
          r_vx(n) + r_vy(n) + r_omega(n) + base_vel(3) + base_ang_vel(3) +
          gravity(3) + joint_pos(12) + joint_vel(12)
        """
        cmd = obs[:, 9:12]

        # Body state + proprioception (shared by feedback MLPs and motor MLP)
        proprio = torch.cat([
            obs[:, 0:3],     # base_vel
            obs[:, 3:6],     # base_ang_vel
            obs[:, 6:9],     # gravity
            obs[:, 12:24],   # joint_pos (12)
            obs[:, 24:36],   # joint_vel (12)
        ], dim=-1)  # (B, 33)

        new_r = self._rnn_step(r_packed, cmd, proprio)

        # Motor input: all 3 CPG states + proprio
        motor_input = torch.cat([new_r, proprio], dim=-1)  # (B, 3n+33)
        action_mean = self.motor_mlp(motor_input)  # (B, 12)

        return action_mean, new_r

    # === rsl_rl ActorCritic interface ===

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        batch_mode = masks is not None

        if batch_mode:
            if isinstance(hidden_states, (list, tuple)):
                r_flat = hidden_states[0]
            else:
                r_flat = hidden_states
            r = r_flat.squeeze(0)  # (num_traj, 3*n)

            L = observations.shape[0]
            means_list = []

            for t in range(L):
                if self.bptt_length > 0 and t % self.bptt_length == 0 and t > 0:
                    r = r.detach()
                obs_t = observations[t]
                mean_t, r = self._compute_action_mean(obs_t, r)
                means_list.append(mean_t)

            means_padded = torch.stack(means_list, dim=0)
            means = unpad_trajectories(means_padded, masks)

            std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(means)
            self.distribution = Normal(means, std)
            return self.distribution.sample()
        else:
            mean, self.r = self._compute_action_mean(observations, self.r)
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
        mean, self.r = self._compute_action_mean(observations, self.r)
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
        """Return RNN state in MDPO-compatible format: ((h, c), (h, c))."""
        r_flat = self.r.unsqueeze(0)  # (1, B, 3*n)
        dummy = torch.zeros_like(r_flat)
        return (r_flat, dummy), (r_flat, dummy)

    def reset(self, dones=None):
        if dones is None or self.r is None:
            return
        done_mask = dones.bool()
        if done_mask.any():
            self.r[done_mask] = 0.0  # reset to rest state

    def get_dropout_masks(self):
        return (None, None)

    def reset_dropout_masks(self):
        pass

    def compute_ortho_loss(self):
        """Orthogonality regularization: ||W^T W - I||_F^2 for each W matrix."""
        I = torch.eye(self.n_neurons, device=self.W_vx.device)
        loss = 0.0
        for W in [self.W_vx, self.W_vy, self.W_omega]:
            loss = loss + (W.T @ W - I).pow(2).sum()
        return loss

    def get_actor_parameters(self):
        return [
            *self.motor_mlp.parameters(),
            self.log_std,
            self.W_vx, self.W_vy, self.W_omega,
            *self.fb_vx.parameters(),
            *self.fb_vy.parameters(),
            *self.fb_omega.parameters(),
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
env_cfg.actions.joint_pos.scale = 0.5
device = "cuda:0"

# --- Reward overrides ---
env_cfg.rewards.flat_orientation_l2.weight = -5.0
env_cfg.rewards.lin_vel_z_l2.weight = -0.5
env_cfg.rewards.track_lin_vel_xy_exp.weight = 6.0
env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(1.0)
env_cfg.rewards.track_ang_vel_z_exp.weight = 5.0
env_cfg.rewards.feet_air_time.weight = 0.1

max_iterations = 1500
num_steps_per_env = 24
save_interval = 50

# VelRNN config
n_neurons = 32         # 4 legs × 8 neurons per leg
rnn_dt = 0.1
rnn_substeps = 1
cos_coupling_alpha = 0.5  # 0=identity (no coupling), 1=hard rank-2 projection, 0.5=soft blend
bptt_length = 0
ortho_coef = 0.1      # orthogonality regularization: ||W^T W - I||_F^2

# MDPO
mdpo_cfg = dict(
    num_learning_epochs=4,
    num_mini_batches=4,
    clip_param=0.2,
    value_clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    distill_coef=0.02,
    value_loss_coef=1.0,
    entropy_coef=0.001,
    learning_rate=1e-3,
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

log_dir = os.path.join("logs", "rsl_rl", "go1_velrnn_mdpo", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)


# ==============================================================================
# Step 2: Create networks and MDPO algorithm
# ==============================================================================
obs_td = env.get_observations().to(device)
obs_flat = _flatten_obs(obs_td)
num_obs = obs_flat.shape[-1]  # 48
num_actions = env.num_actions  # 12

print(f"Obs dim: {num_obs}, Action dim: {num_actions}")

policy_kwargs = dict(
    num_obs=num_obs,
    num_actions=num_actions,
    n_neurons=n_neurons,
    rnn_dt=rnn_dt,
    rnn_substeps=rnn_substeps,
    cos_coupling_alpha=cos_coupling_alpha,
    bptt_length=bptt_length,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation=activation,
    init_noise_std=init_noise_std,
    device=device,
)
actor_critic_1 = VelRNN_ActorCritic(**policy_kwargs).to(device)
actor_critic_2 = VelRNN_ActorCritic(**policy_kwargs).to(device)

# Compile the hot path
actor_critic_1._compute_action_mean = torch.compile(actor_critic_1._compute_action_mean)
actor_critic_2._compute_action_mean = torch.compile(actor_critic_2._compute_action_mean)

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

motor_input_dim = 3 * n_neurons + 3 + 3 + 3 + 12 + 12  # 3 CPGs + body(9) + jpos(12) + jvel(12)
rnn_params = 3 * n_neurons * n_neurons + 3 * n_neurons  # 3 × (W + init_r)
total_params = sum(p.numel() for p in actor_critic_1.parameters()) + sum(p.numel() for p in actor_critic_2.parameters())
print(f"Policy: {actor_critic_1.__class__.__name__}")
print(f"  3 CPGs: {n_neurons} neurons each, {rnn_params} RNN params (3 × {n_neurons}×{n_neurons} + 3 × init)")
print(f"  Motor MLP: [{motor_input_dim} → {actor_hidden_dims} → 12]")
print(f"  Critic MLP: [{num_obs} → {critic_hidden_dims} → 1]")
print(f"Total parameters (both policies): {total_params:,}")
print(f"Envs split: policy 1 gets {mdpo.indices_1.numel()}, policy 2 gets {mdpo.indices_2.numel()}")


# ==============================================================================
# Step 3: Initialize wandb
# ==============================================================================
wandb.init(
    project="isaaclab-go1-velocity",
    name=f"mdpo_VelRNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "algorithm": "MDPO",
        "architecture": "VelRNN-3CPG",
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": num_steps_per_env,
        "num_obs": num_obs,
        "num_actions": num_actions,
        "n_neurons": n_neurons,
        "neurons_per_cpg": n_neurons,
        "rnn_dt": rnn_dt,
        "rnn_substeps": rnn_substeps,
        "rnn_params": rnn_params,
        "cos_coupling_alpha": cos_coupling_alpha,
        "ortho_coef": ortho_coef,
        "motor_input_dim": motor_input_dim,
        "motor_output_dim": 12,
        "actor_type": "single MLP",
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

print(f"\nStarting MDPO + VelRNN-3CPG training for {max_iterations} iterations...")
print(f"  {env.num_envs} envs x {num_steps_per_env} steps = {env.num_envs * num_steps_per_env} samples/iter")
print(f"  3 independent CPGs: {n_neurons} neurons each (vx, vy, omega)")
print(f"  Each: r_new = normalize(cmd * W @ r) @ C_soft(alpha={cos_coupling_alpha})")
print(f"  Motor: MLP [{motor_input_dim} → {actor_hidden_dims} → 12], Critic: {critic_hidden_dims}")

for iteration in range(max_iterations):
    iter_start = time.time()

    # ==================================================================
    # Phase 1: Collect rollouts
    # ==================================================================
    mdpo.train_mode()
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
    with torch.no_grad():
        for step in tqdm(range(num_steps_per_env), desc=f"[{iteration}] collect", leave=False):
            obs_flat = _flatten_obs(obs_td)
            actions = mdpo.act(obs_flat, obs_flat)

            obs_td, rewards, dones, extras = env.step(actions.detach().to(env.device))
            obs_td = obs_td.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            mdpo.process_env_step(rewards, dones, extras)

            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

    end_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
    displacement = (end_pos - start_pos).norm(dim=-1)
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

    # Orthogonality regularization step for W_vx, W_vy, W_omega
    ortho_loss_1 = ortho_coef * actor_critic_1.compute_ortho_loss()
    ortho_loss_2 = ortho_coef * actor_critic_2.compute_ortho_loss()
    mdpo.optimizer_1.zero_grad()
    ortho_loss_1.backward()
    nn.utils.clip_grad_norm_(actor_critic_1.parameters(), max_norm=1.0)
    mdpo.optimizer_1.step()
    mdpo.optimizer_2.zero_grad()
    ortho_loss_2.backward()
    nn.utils.clip_grad_norm_(actor_critic_2.parameters(), max_norm=1.0)
    mdpo.optimizer_2.step()
    mean_ortho_loss = 0.5 * (ortho_loss_1.item() + ortho_loss_2.item())

    learn_time = time.time() - learn_start
    total_time = time.time() - iter_start

    # ==================================================================
    # Logging
    # ==================================================================
    fps = int(env.num_envs * num_steps_per_env / total_time)
    mean_reward = np.mean(rewbuffer) if rewbuffer else 0.0
    mean_ep_len = np.mean(lenbuffer) if lenbuffer else 0.0

    log_extras = extras.get("log", {}) if isinstance(extras, dict) else {}

    log_dict = {
        "Loss/value": mean_value_loss,
        "Loss/surrogate": mean_surrogate_loss,
        "Loss/distill_kl": mean_kl_div,
        "Loss/ortho": mean_ortho_loss,
        "Loss/learning_rate": mdpo.learning_rate,
        "Perf/fps": fps,
        "Perf/collection_time": collection_time,
        "Perf/learn_time": learn_time,
        "Train/mean_reward": mean_reward,
        "Train/mean_episode_length": mean_ep_len,
        "Train/mean_displacement": mean_displacement,
    }

    for key, value in log_extras.items():
        if key.startswith("Episode_Reward/"):
            log_dict[key] = value.item() if isinstance(value, torch.Tensor) else value

    if iteration % 10 == 0:
        n = n_neurons
        log_dict["action_std_mean"] = actor_critic_1.log_std.clamp(-5.0, 0.5).exp().mean().item()
        log_dict["RNN/r_vx_rms"] = actor_critic_1.r[:, :n].pow(2).mean().sqrt().item()
        log_dict["RNN/r_vy_rms"] = actor_critic_1.r[:, n:2*n].pow(2).mean().sqrt().item()
        log_dict["RNN/r_om_rms"] = actor_critic_1.r[:, 2*n:].pow(2).mean().sqrt().item()
        log_dict["RNN/W_vx_norm"] = actor_critic_1.W_vx.data.norm().item()
        log_dict["RNN/W_vy_norm"] = actor_critic_1.W_vy.data.norm().item()
        log_dict["RNN/W_omega_norm"] = actor_critic_1.W_omega.data.norm().item()

    wandb.log(log_dict, step=iteration)

    if iteration % 10 == 0:
        def _to_float(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        r_track_lin = _to_float(log_extras.get("Episode_Reward/track_lin_vel_xy_exp", 0.0))
        r_track_ang = _to_float(log_extras.get("Episode_Reward/track_ang_vel_z_exp", 0.0))
        r_flat_orient = _to_float(log_extras.get("Episode_Reward/flat_orientation_l2", 0.0))
        r_feet_air = _to_float(log_extras.get("Episode_Reward/feet_air_time", 0.0))

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
            f"W_vx={actor_critic_1.W_vx.data.norm().item():.3f}  "
            f"W_vy={actor_critic_1.W_vy.data.norm().item():.3f}  "
            f"W_om={actor_critic_1.W_omega.data.norm().item():.3f}  "
            f"track_lin={r_track_lin:.4f}  track_ang={r_track_ang:.4f}  "
            f"flat={r_flat_orient:.4f}  air={r_feet_air:.4f}"
        )

    # ==================================================================
    # Checkpointing
    # ==================================================================
    if (iteration + 1) % save_interval == 0 or iteration == max_iterations - 1:
        ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
        ckpt_data = {
            "iter": iteration + 1,
            "architecture": "velrnn",
            "model_1_state_dict": actor_critic_1.state_dict(),
            "model_2_state_dict": actor_critic_2.state_dict(),
            "optimizer_1_state_dict": mdpo.optimizer_1.state_dict(),
            "optimizer_2_state_dict": mdpo.optimizer_2.state_dict(),
            "rnn_config": {
                "n_neurons": n_neurons,
                "rnn_dt": rnn_dt,
                "rnn_substeps": rnn_substeps,
                "motor_type": "per-leg shared MLP",
                "motor_input_dim": motor_input_dim,
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
