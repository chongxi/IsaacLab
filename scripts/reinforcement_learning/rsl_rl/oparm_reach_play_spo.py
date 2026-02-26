"""
Play a trained SPO checkpoint for Isaac-Reach-OpenArm-Bi-v0.

This script loads checkpoints saved by oparm_reach_train_spo.py, which uses a
custom ActorCritic (with log_std) that is incompatible with RSL-RL's OnPolicyRunner.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/oparm_reach_play_spo.py
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
from tensordict import TensorDict
from torch.distributions import Normal

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

# ==============================================================================
# Custom ActorCritic (must match the training script exactly)
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
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

        # log_std (matches training script)
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


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
# Checkpoint — update this path to the checkpoint you want to play
# ==============================================================================
# CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_19-27-23/model_1250.pt"
# CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_21-40-37/model_1500.pt"
# CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_22-09-04/model_1500.pt"
CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_22-26-40/model_1500.pt"

# ==============================================================================
# Network config — must match the training run that produced the checkpoint
# ==============================================================================
ACTOR_HIDDEN_DIMS = [128, 128]
CRITIC_HIDDEN_DIMS = [64, 64]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0

# ==============================================================================
# Environment
# ==============================================================================
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 8
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

# Match training-time action interface (relative joint position / delta action).
env_cfg.actions.left_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_left_joint.*"],
    scale=0.3,
    use_zero_offset=True,
    alpha=0.3,
)
env_cfg.actions.right_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_right_joint.*"],
    scale=0.3,
    use_zero_offset=True,
    alpha=0.3,
)

env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)

# ==============================================================================
# Build policy and load checkpoint
# ==============================================================================
device = env.unwrapped.device
obs, info = env.reset()
obs_flat = _flatten_obs(obs)
num_obs = obs_flat.shape[-1]
num_actions = env.num_actions

policy = ActorCritic(
    num_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=ACTOR_HIDDEN_DIMS,
    critic_hidden_dims=CRITIC_HIDDEN_DIMS,
    activation=ACTIVATION,
    init_noise_std=INIT_NOISE_STD,
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
policy.load_state_dict(checkpoint["model_state_dict"])
policy.eval()

print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  Network: actor={ACTOR_HIDDEN_DIMS}, critic={CRITIC_HIDDEN_DIMS}, act={ACTIVATION}")
print(f"  Obs dim: {num_obs}, Action dim: {num_actions}")

# ==============================================================================
# Play loop (obs already populated from env.reset() above)
# ==============================================================================
while simulation_app.is_running():
    with torch.inference_mode():
        obs_flat = _flatten_obs(obs)
        actions = policy.act_inference(obs_flat)
        obs, _, dones, _ = env.step(actions)
