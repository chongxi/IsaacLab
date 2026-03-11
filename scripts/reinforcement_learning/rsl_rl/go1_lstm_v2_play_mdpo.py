"""
Play a trained MDPO + LSTM v2 baseline checkpoint for Isaac-Velocity-Flat-Unitree-Go1-v0.

Uses the same inline ActorCriticRecurrent architecture as go1_lstm_v2_train_mdpo.py.

Usage:
    python scripts/reinforcement_learning/rsl_rl/go1_lstm_v2_play_mdpo.py
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
import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.utils import unpad_trajectories

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
)


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


# ==============================================================================
# Network definitions (must match go1_lstm_v2_train_mdpo.py exactly)
# ==============================================================================
class ActorCritic(nn.Module):
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
        super().__init__()
        activation = get_activation(activation)

        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


class Memory(nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        rnn_type = type.lower()
        if rnn_type == "gru":
            rnn_cls = nn.GRU
        elif rnn_type == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError(f"RNN type {type} not supported.")
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


class ActorCriticRecurrent(ActorCritic):
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


# ==============================================================================
# Checkpoint — update this path
# ==============================================================================
CHECKPOINT_PATH = "logs/rsl_rl/go1_lstm_v2_mdpo/2026-03-10_22-12-13/model_1050.pt"

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

policy = ActorCriticRecurrent(
    num_actor_obs=num_obs,
    num_critic_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=[128, 128],
    critic_hidden_dims=[256, 128],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_size=128,
    rnn_num_layers=1,
    init_noise_std=1.0,
).to(device)

policy.load_state_dict(checkpoint[state_dict_key])
policy.eval()

print(f"Loaded MDPO + LSTM v2 checkpoint: {CHECKPOINT_PATH}")
print(f"  Policy index: {POLICY_INDEX} of 2")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  Total params: {sum(p.numel() for p in policy.parameters()):,}")

# ==============================================================================
# Play loop
# ==============================================================================
while simulation_app.is_running():
    with torch.inference_mode():
        obs_flat = _flatten_obs(obs)
        actions = policy.act_inference(obs_flat)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)
