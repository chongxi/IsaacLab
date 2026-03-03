"""
Play a trained MDPO + CPG-Reflex checkpoint for Isaac-Velocity-Flat-Unitree-Go1-v0.

The CPG-Reflex architecture is recurrent: internal CPG state (cpg_x, cpg_a) persists
across timesteps and must be reset on episode dones.

MDPO trains TWO policies. By default this loads policy 1. Set POLICY_INDEX = 2 for policy 2.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/go1_cpg_play_mdpo.py
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
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# BatchedQuadrupedDecoder (must match training)
# ==============================================================================
class BatchedQuadrupedDecoder(nn.Module):
    def __init__(self, cmd_dim=3, coupling_strength=1.5):
        super().__init__()
        self.cmd_dim = cmd_dim
        self.n_legs = 4
        self.joints_per_leg = 3

        # Per-leg dn and sht
        self.w_dn = nn.Parameter(torch.zeros(cmd_dim, 4))
        self.b_dn = nn.Parameter(torch.full((4,), 2.0))
        self.w_sht = nn.Parameter(torch.zeros(cmd_dim, 4))
        self.b_sht = nn.Parameter(torch.zeros(4))

        self.W_delta = nn.Parameter(torch.zeros(4, self.joints_per_leg * 2, cmd_dim))
        self.register_buffer("W_base", torch.zeros(4, self.joints_per_leg, 2))

    def forward(self, cmd):
        dn = torch.sigmoid(cmd @ self.w_dn + self.b_dn)   # (B, 4)
        sht = F.softplus(cmd @ self.w_sht + self.b_sht) + 0.3  # (B, 4)

        dW = torch.einsum("lfc,bc->blf", self.W_delta, cmd)
        dW = dW.reshape(-1, 4, self.joints_per_leg, 2)
        W_eff = self.W_base.unsqueeze(0) + dW

        return dn, sht, W_eff


# ==============================================================================
# CPG_Reflex_ActorCritic (must match training)
# ==============================================================================
class CPG_Reflex_ActorCritic(nn.Module):
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
        bptt_length: int = 8,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.cpg_dt = cpg_dt
        self.cpg_substeps = cpg_substeps
        self.bptt_length = bptt_length
        self.n_legs = 4
        self.n_neurons = 3
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

        self.register_buffer("init_cpg_x", torch.randn(4, 3) * 0.01)
        self.register_buffer("init_cpg_a", torch.zeros(4, 3))

        # === Learnable decoder ===
        self.decoder = BatchedQuadrupedDecoder(cmd_dim=3, coupling_strength=coupling_strength)
        W_base = torch.tensor([
            [0.02,  0.02],   # abduction
            [0.25,  0.20],   # hip
            [0.10, -0.40],   # knee
        ])
        self.decoder.W_base.copy_(W_base.unsqueeze(0).expand(4, -1, -1))

        # === Reflex MLP ===
        reflex_layers = []
        in_dim = num_obs
        for h_dim in reflex_hidden_dims:
            reflex_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            reflex_layers.append(act_fn())
            in_dim = h_dim
        reflex_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        self.reflex_mlp = nn.Sequential(*reflex_layers)

        # === Critic MLP ===
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
        self.cpg_x: torch.Tensor | None = None
        self.cpg_a: torch.Tensor | None = None

    def init_cpg_state(self, num_envs: int):
        self.cpg_x = self.init_cpg_x.unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.cpg_a = self.init_cpg_a.unsqueeze(0).expand(num_envs, -1, -1).clone()

    def _cpg_euler_step(self, cpg_x, cpg_a, dn, sht):
        sht_3d = sht.unsqueeze(-1)           # (B, 4, 1)
        tau_x_eff = self.tau_x / (0.3 + 0.7 * sht_3d)
        tau_a_eff = self.tau_a / (0.2 + 2.5 * sht_3d)
        dn_3d = dn.unsqueeze(-1)             # (B, 4, 1)
        ext_input = self.W_in * dn_3d        # (B, 4, 3)

        for _ in range(self.cpg_substeps):
            r = torch.tanh(cpg_x)
            rec_input = dn_3d * torch.einsum("bln,mn->blm", r, self.W_rec)
            dxdt = (-cpg_x + rec_input + ext_input + self.bias - cpg_a) / tau_x_eff
            dadt = (-cpg_a + self.g_adapt * r) / tau_a_eff
            cpg_x = cpg_x + self.cpg_dt * dxdt
            cpg_a = cpg_a + self.cpg_dt * dadt

        new_r = torch.tanh(cpg_x)
        return new_r, cpg_x, cpg_a

    def _compute_action_mean(self, obs, cpg_x, cpg_a):
        cmd = obs[:, 9:12]
        dn, sht, W_eff = self.decoder(cmd)
        new_r, new_cpg_x, new_cpg_a = self._cpg_euler_step(cpg_x, cpg_a, dn, sht)
        cpg_offsets = torch.einsum("bljn,bln->blj", W_eff, new_r[:, :, :2])
        cpg_offsets = cpg_offsets.reshape(-1, 12)
        corrections = torch.tanh(self.reflex_mlp(obs))
        action_mean = cpg_offsets + corrections
        return action_mean, new_cpg_x, new_cpg_a

    def act_inference(self, observations):
        mean, self.cpg_x, self.cpg_a = self._compute_action_mean(
            observations, self.cpg_x, self.cpg_a
        )
        return mean

    def reset(self, dones=None):
        if dones is None or self.cpg_x is None:
            return
        done_mask = dones.bool()
        if done_mask.any():
            n_done = done_mask.sum().item()
            self.cpg_x[done_mask] = self.init_cpg_x.unsqueeze(0).expand(n_done, -1, -1)
            self.cpg_a[done_mask] = self.init_cpg_a.unsqueeze(0).expand(n_done, -1, -1)


# ==============================================================================
# Checkpoint — update this path
# ==============================================================================
CHECKPOINT_PATH = "logs/rsl_rl/go1_cpg_mdpo/2026-03-03_01-58-28/model_1200.pt"

# Which MDPO policy to play (1 or 2)
POLICY_INDEX = 1

# ==============================================================================
# Environment (play config: fewer envs, no noise)
# ==============================================================================
env_cfg = UnitreeGo1FlatEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"
env_cfg.actions.joint_pos.scale = 0.5  # must match training
env_cfg.observations.policy.enable_corruption = False

env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

# ==============================================================================
# Build policy and load checkpoint
# ==============================================================================
device = env.unwrapped.device
obs, info = env.reset()
obs_flat = _flatten_obs(obs)
num_obs = obs_flat.shape[-1]  # 48
num_actions = env.num_actions  # 12

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
state_dict_key = f"model_{POLICY_INDEX}_state_dict"

cpg_cfg = checkpoint.get("cpg_config", {})

policy = CPG_Reflex_ActorCritic(
    num_obs=num_obs,
    num_actions=num_actions,
    reflex_hidden_dims=[128, 128],
    critic_hidden_dims=[256, 128],
    activation="elu",
    init_noise_std=0.3,
    cpg_dt=cpg_cfg.get("cpg_dt", 0.01),
    cpg_substeps=cpg_cfg.get("cpg_substeps", 2),
    coupling_strength=cpg_cfg.get("coupling_strength", 0.0),
    bptt_length=0,
    device=str(device),
).to(device)

policy.load_state_dict(checkpoint[state_dict_key])
policy.eval()
policy.init_cpg_state(env.num_envs)

print(f"Loaded MDPO + CPG-Reflex checkpoint: {CHECKPOINT_PATH}")
print(f"  Policy index: {POLICY_INDEX} of 2")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  CPG config: dt={cpg_cfg.get('cpg_dt', 0.01)}, substeps={cpg_cfg.get('cpg_substeps', 2)}")

with torch.no_grad():
    d = policy.decoder
    dn_vals = torch.sigmoid(d.b_dn).tolist()
    sht_vals = (F.softplus(d.b_sht) + 0.3).tolist()
    print(f"  Decoder: dn=[{dn_vals[0]:.3f},{dn_vals[1]:.3f},{dn_vals[2]:.3f},{dn_vals[3]:.3f}]  "
          f"sht=[{sht_vals[0]:.2f},{sht_vals[1]:.2f},{sht_vals[2]:.2f},{sht_vals[3]:.2f}]  "
          f"W_delta_norm={d.W_delta.norm().item():.4f}")

# ==============================================================================
# Play loop
# ==============================================================================
while simulation_app.is_running():
    with torch.inference_mode():
        obs_flat = _flatten_obs(obs)
        actions = policy.act_inference(obs_flat)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)
