"""
Play a trained MDPO + LSTM baseline checkpoint for Isaac-Velocity-Flat-Unitree-Go1-v0.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/go1_lstm_play_mdpo.py
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
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent

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


# ==============================================================================
# Checkpoint — update this path
# ==============================================================================
# CHECKPOINT_PATH = "logs/rsl_rl/go1_lstm_mdpo/2026-03-03_15-51-50/model_50.pt"
CHECKPOINT_PATH = "logs/rsl_rl/go1_lstm_mdpo/2026-03-03_15-52-55/model_600.pt"

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

print(f"Loaded MDPO + LSTM checkpoint: {CHECKPOINT_PATH}")
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
