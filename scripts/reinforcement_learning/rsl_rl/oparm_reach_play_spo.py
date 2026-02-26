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
    OpenArmReachEnvCfgErrObs,
)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

# ==============================================================================
# Custom ActorCritic (must match the training script exactly)
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic_MLP(nn.Module):
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


class ActorCritic_NEW(nn.Module):
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
                "ActorCritic_NEW expects explicit error observation layout "
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
        self.arm_key = layer_init(nn.Linear(2 * self.dof_per_arm, self.err_dim_per_arm * attn_dim), std=0.01)
        self.joint_id_embed = nn.Parameter(torch.zeros(self.dof_per_arm, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)
        self.attn_scale = float(attn_dim) ** -0.5
        self.a_scale = nn.Parameter(torch.ones(self.dof_per_arm, self.err_dim_per_arm))
        critic_layers = []
        in_dim = num_obs
        for h_dim in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            critic_layers.append(act_fn())
            in_dim = h_dim
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(1, num_actions)))
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        left_query = self.joint_query(left_joint_tokens)
        right_query = self.joint_query(right_joint_tokens)
        joint_bias = self.joint_id_embed.unsqueeze(0)
        left_query = left_query + joint_bias
        right_query = right_query + joint_bias

        left_state = torch.cat([left_q, left_dq], dim=-1)
        right_state = torch.cat([right_q, right_dq], dim=-1)
        left_key = self.arm_key(left_state).view(-1, self.err_dim_per_arm, left_query.shape[-1])
        right_key = self.arm_key(right_state).view(-1, self.err_dim_per_arm, right_query.shape[-1])

        left_logits = torch.einsum("bnd,bmd->bnm", left_query, left_key) * self.attn_scale
        right_logits = torch.einsum("bnd,bmd->bnm", right_query, right_key) * self.attn_scale
        a_scale = self.a_scale.unsqueeze(0)
        left_A = left_logits
        right_A = right_logits

        left_u = torch.bmm(left_A, left_err.unsqueeze(-1)).squeeze(-1)
        right_u = torch.bmm(right_A, right_err.unsqueeze(-1)).squeeze(-1)

        raw_u = torch.cat([left_u, right_u], dim=-1)
        return torch.tanh(raw_u)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self._actor_mean(obs)


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
# CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_22-26-40/model_1500.pt"
# CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_23-14-01/model_1500.pt"
# CHECKPOINT_PATH = "logs/rsl_rl/openarm_bi_reach/2026-02-26_09-35-12/model_1500.pt"
# CHECKPOINT_PATH = "logs/rsl_rl/openarm_bi_reach/2026-02-26_12-27-19/model_1500.pt"
CHECKPOINT_PATH = "logs/rsl_rl/openarm_bi_reach/2026-02-26_14-09-34/model_1500.pt"

# ==============================================================================
# Network config — must match the training run that produced the checkpoint
# ==============================================================================
ACTOR_HIDDEN_DIMS = [64, 64]
CRITIC_HIDDEN_DIMS = [64, 64]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0
POLICY_CLASS_NAME = "new"  # "mlp" or "new"

# ==============================================================================
# Environment
# ==============================================================================
USE_ERROR_OBS_EXPERIMENT = True
env_cfg = OpenArmReachEnvCfgErrObs() if USE_ERROR_OBS_EXPERIMENT else OpenArmReachEnvCfg()
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

policy_class_map = {
    "mlp": ActorCritic_MLP,
    "new": ActorCritic_NEW,
}
policy_cls = policy_class_map[POLICY_CLASS_NAME.lower()]

policy = policy_cls(
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
print(f"  Policy class: {policy.__class__.__name__}")
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
