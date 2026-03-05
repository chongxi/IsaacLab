"""
Play a trained MDPO + CPG-Reflex checkpoint for Isaac-Velocity-Flat-Unitree-Go1-v0.

The CPG-Reflex architecture is recurrent: internal CPG state (cpg_x, cpg_a) persists
across timesteps and must be reset on episode dones.

Architecture: Autonomous CPG (fixed dn/sht) + additive MLP reflex
    action = W_readout @ cpg_rates   # rhythmic gait from 4 MANC oscillators
           + 0.3 * tanh(reflex_mlp)  # balance corrections

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
# CPG_Reflex_ActorCritic (must match training architecture)
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
        cpg_dt: float = 0.1,
        cpg_substeps: int = 1,
        bptt_length: int = 0,
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

        # === Fixed CPG drive (autonomous — no decoder MLP) ===
        self.fixed_dn = 0.88   # sigmoid(2.0) — constant drive amplitude
        self.fixed_sht = 1.0   # ~1Hz oscillation frequency

        # === Learnable CPG readout: rates(2) → joint offsets(3) per leg, shared ===
        self.W_readout = nn.Parameter(torch.randn(3, 2) * 0.3)

        # === Reflex MLP: non-rhythmic obs(21) → joint corrections(12) ===
        self.num_reflex_obs = 21
        reflex_layers = []
        in_dim = self.num_reflex_obs
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

    def _cpg_euler_step(self, cpg_x, cpg_a):
        ext_input = self.W_in * self.fixed_dn  # (3,)

        for _ in range(self.cpg_substeps):
            r = F.gelu(cpg_x)
            rec_input = self.fixed_dn * torch.einsum("bln,mn->blm", r, self.W_rec)
            dxdt = (-cpg_x + rec_input + ext_input + self.bias - cpg_a)
            dadt = (-cpg_a + self.g_adapt * r)
            cpg_x = cpg_x + self.cpg_dt * dxdt
            cpg_a = cpg_a + self.cpg_dt * dadt

        new_r = F.gelu(cpg_x)
        return new_r, cpg_x, cpg_a

    def _compute_action_mean(self, obs, cpg_x, cpg_a):
        # Autonomous CPG step
        new_r, new_cpg_x, new_cpg_a = self._cpg_euler_step(cpg_x, cpg_a)

        # CPG readout: W_readout @ [E1, E2] → joint offsets, shared across legs
        cpg_offsets = torch.einsum("jn,bln->blj", self.W_readout, new_r[:, :, :2])
        cpg_offsets = cpg_offsets.reshape(-1, 12)

        # Additive MLP
        reflex_obs = torch.cat([
            obs[:, 3:6], obs[:, 6:9], obs[:, 9:12], obs[:, 24:36],
        ], dim=-1)
        corrections = 0.3 * torch.tanh(self.reflex_mlp(reflex_obs))

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
# CHECKPOINT_PATH = "logs/rsl_rl/go1_cpg_mdpo/2026-03-03_18-02-49/model_1500.pt"
CHECKPOINT_PATH = "logs/rsl_rl/go1_cpg_mdpo/2026-03-03_20-47-22/model_1500.pt"

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

# Reward overrides (must match training)
env_cfg.rewards.action_rate_l2 = None
env_cfg.rewards.flat_orientation_l2.weight = -0.5
env_cfg.rewards.lin_vel_z_l2.weight = -0.5
env_cfg.rewards.track_lin_vel_xy_exp.weight = 6.0
env_cfg.rewards.track_ang_vel_z_exp.weight = 5.0
env_cfg.rewards.feet_air_time.weight = 0.5

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
    cpg_dt=cpg_cfg.get("cpg_dt", 0.1),
    cpg_substeps=cpg_cfg.get("cpg_substeps", 1),
    bptt_length=0,
    device=str(device),
).to(device)

policy.load_state_dict(checkpoint[state_dict_key])
policy.eval()
policy.init_cpg_state(env.num_envs)

print(f"Loaded MDPO + CPG-Reflex checkpoint: {CHECKPOINT_PATH}")
print(f"  Policy index: {POLICY_INDEX} of 2")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  CPG config: dt={cpg_cfg.get('cpg_dt', 0.1)}, substeps={cpg_cfg.get('cpg_substeps', 1)}")
print(f"  W_readout_norm={policy.W_readout.norm().item():.4f}")

# ==============================================================================
# Play loop
# ==============================================================================
step_count = 0
while simulation_app.is_running():
    with torch.no_grad():
        obs_flat = _flatten_obs(obs)
        actions = policy.act_inference(obs_flat)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)

        step_count += 1
        if step_count % 100 == 0:
            cmd = obs_flat[:, 9:12]
            print(f"[step {step_count}]  action_rms={actions.pow(2).mean().sqrt():.4f}  "
                  f"cmd_norm={cmd.norm(dim=-1).mean():.3f}  "
                  f"cpg_x_rms={policy.cpg_x.pow(2).mean().sqrt():.4f}")
