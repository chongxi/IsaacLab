"""
Play a trained MDPO checkpoint for Isaac-Reach-OpenArm-Bi-v0.

MDPO trains TWO policies simultaneously (mutual distillation). The checkpoint
contains both:
    - "model_1_state_dict": policy 1 (trained on odd-indexed envs)
    - "model_2_state_dict": policy 2 (trained on even-indexed envs)

By default this script loads policy 1. Set POLICY_INDEX = 2 to play policy 2.
Both policies should behave similarly due to mutual distillation.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/oparm_reach_play_mdpo.py
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

        actor_layers = []
        in_dim = num_obs
        for h_dim in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            actor_layers.append(act_fn())
            in_dim = h_dim
        actor_layers.append(layer_init(nn.Linear(in_dim, num_actions), std=0.01))
        actor_layers.append(nn.Tanh())
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
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class NeuralJacobianPolicy(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [64, 64],
        critic_hidden_dims: list[int] = [64, 64],
        activation: str = "tanh",
        init_noise_std: float = 1.0,
        gate: bool = False,
    ):
        super().__init__()

        if num_actions % 2 != 0:
            raise ValueError(f"Expected even number of actions (bimanual), got {num_actions}.")

        self.dof_per_arm = num_actions // 2
        self.err_dim_per_arm = 6
        self.expected_obs_dim = 6 * self.dof_per_arm + 2 * self.err_dim_per_arm
        if num_obs != self.expected_obs_dim:
            raise ValueError(
                "NeuralJacobianPolicy expects explicit error observation layout "
                "[l_q, r_q, l_dq, r_dq, l_err6, r_err6, l_prev_a, r_prev_a]. "
                f"Got num_obs={num_obs}, expected={self.expected_obs_dim}."
            )

        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "gelu": nn.GELU,
        }
        act_fn = activations[activation]

        attn_dim = actor_hidden_dims[0] if len(actor_hidden_dims) > 0 else 64
        # Query input: (sin(q_i), cos(q_i), sin(dq_i), cos(dq_i)) — 4 values per joint
        self.joint_query = layer_init(nn.Linear(4, attn_dim), std=0.01)
        # Key input: (sin(q), cos(q), sin(dq), cos(dq)) — 4 * dof_per_arm values
        arm_key_input_dim = 4 * self.dof_per_arm
        self.arm_key = nn.Sequential(
            layer_init(nn.Linear(arm_key_input_dim, attn_dim)),
            act_fn(),
            layer_init(nn.Linear(attn_dim, self.err_dim_per_arm * attn_dim), std=0.01),
        )
        self.joint_id_embed = nn.Parameter(torch.zeros(self.dof_per_arm, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)
        self.attn_scale = float(attn_dim) ** -0.5
        self.a_scale = nn.Parameter(torch.ones(self.dof_per_arm, self.err_dim_per_arm))

        # Error-magnitude gating: sigmoid(linear(||error||)) per arm
        self.use_gate = gate
        if gate:
            self.error_gate = layer_init(nn.Linear(1, 1), std=0.01)

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

    def _split_obs(self, obs: torch.Tensor):
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

        # Query: per-joint trig features (sin(q_i), cos(q_i), sin(dq_i), cos(dq_i))
        left_joint_tokens = torch.stack([torch.sin(left_q), torch.cos(left_q), torch.sin(left_dq), torch.cos(left_dq)], dim=-1)
        right_joint_tokens = torch.stack([torch.sin(right_q), torch.cos(right_q), torch.sin(right_dq), torch.cos(right_dq)], dim=-1)
        left_query = self.joint_query(left_joint_tokens)
        right_query = self.joint_query(right_joint_tokens)
        joint_bias = self.joint_id_embed.unsqueeze(0)
        left_query = left_query + joint_bias
        right_query = right_query + joint_bias

        # Key: global trig features (sin(q_all), cos(q_all), sin(dq_all), cos(dq_all))
        left_state = torch.cat([torch.sin(left_q), torch.cos(left_q), torch.sin(left_dq), torch.cos(left_dq)], dim=-1)
        right_state = torch.cat([torch.sin(right_q), torch.cos(right_q), torch.sin(right_dq), torch.cos(right_dq)], dim=-1)
        left_key = self.arm_key(left_state).view(-1, self.err_dim_per_arm, left_query.shape[-1])
        right_key = self.arm_key(right_state).view(-1, self.err_dim_per_arm, right_query.shape[-1])

        left_logits = torch.einsum("bnd,bmd->bnm", left_query, left_key) * self.attn_scale
        right_logits = torch.einsum("bnd,bmd->bnm", right_query, right_key) * self.attn_scale
        left_A = left_logits
        right_A = right_logits

        left_u = torch.bmm(left_A, left_err.unsqueeze(-1)).squeeze(-1)
        right_u = torch.bmm(right_A, right_err.unsqueeze(-1)).squeeze(-1)

        if self.use_gate:
            left_gate = torch.sigmoid(self.error_gate(left_err.norm(dim=-1, keepdim=True)))
            right_gate = torch.sigmoid(self.error_gate(right_err.norm(dim=-1, keepdim=True)))
            left_u = left_gate * left_u
            right_u = right_gate * right_u

        raw_u = torch.cat([left_u, right_u], dim=-1)
        return raw_u

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self._actor_mean(obs)


class NeuralJacobianLocalPolicy(nn.Module):
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
                "NeuralJacobianLocalPolicy expects explicit error observation layout "
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
        self.joint_id_embed = nn.Parameter(torch.zeros(self.dof_per_arm, attn_dim))
        nn.init.normal_(self.joint_id_embed, mean=0.0, std=0.02)
        self.error_key = layer_init(nn.Linear(1, attn_dim), std=0.01)
        self.error_id_embed = nn.Parameter(torch.zeros(self.err_dim_per_arm, attn_dim))
        nn.init.normal_(self.error_id_embed, mean=0.0, std=0.02)

        self.attn_scale = float(attn_dim) ** -0.5

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

    def _split_obs(self, obs: torch.Tensor):
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
        left_query = self.joint_query(left_joint_tokens) + self.joint_id_embed.unsqueeze(0)
        right_query = self.joint_query(right_joint_tokens) + self.joint_id_embed.unsqueeze(0)

        left_err_tokens = left_err.unsqueeze(-1)
        right_err_tokens = right_err.unsqueeze(-1)
        left_key = self.error_key(left_err_tokens) + self.error_id_embed.unsqueeze(0)
        right_key = self.error_key(right_err_tokens) + self.error_id_embed.unsqueeze(0)

        left_A = torch.einsum("bnd,bmd->bnm", left_query, left_key) * self.attn_scale
        right_A = torch.einsum("bnd,bmd->bnm", right_query, right_key) * self.attn_scale

        left_u = torch.bmm(left_A, left_err.unsqueeze(-1)).squeeze(-1)
        right_u = torch.bmm(right_A, right_err.unsqueeze(-1)).squeeze(-1)

        return torch.cat([left_u, right_u], dim=-1)

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
CHECKPOINT_PATH = "logs/rsl_rl/openarm_bi_reach/2026-02-28_23-44-02/model_1500.pt"

# Which of the two MDPO policies to play (1 or 2)
POLICY_INDEX = 1

# ==============================================================================
# Network config — must match the training run that produced the checkpoint
# ==============================================================================
ACTOR_HIDDEN_DIMS = [64, 64]
CRITIC_HIDDEN_DIMS = [64, 64]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0
POLICY_CLASS_NAME = "njp"  # "mlp" or "njp"
USE_ERROR_GATE = True  # must match training config

# ==============================================================================
# Environment
# ==============================================================================
USE_ERROR_OBS_EXPERIMENT = True
env_cfg = OpenArmReachEnvCfgErrObs() if USE_ERROR_OBS_EXPERIMENT else OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 8
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

env_cfg.actions.left_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_left_joint.*"],
    scale=0.5,
    use_zero_offset=True,
    alpha=0.7,
)
env_cfg.actions.right_arm_action = reach_mdp.EMARelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["openarm_right_joint.*"],
    scale=0.5,
    use_zero_offset=True,
    alpha=0.7,
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
    "njp": NeuralJacobianPolicy,
    "njp_local": NeuralJacobianLocalPolicy,
}
policy_cls = policy_class_map[POLICY_CLASS_NAME.lower()]

policy_kwargs = dict(
    num_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=ACTOR_HIDDEN_DIMS,
    critic_hidden_dims=CRITIC_HIDDEN_DIMS,
    activation=ACTIVATION,
    init_noise_std=INIT_NOISE_STD,
)
if POLICY_CLASS_NAME.lower() == "njp":
    policy_kwargs["gate"] = USE_ERROR_GATE

policy = policy_cls(**policy_kwargs).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
state_dict_key = f"model_{POLICY_INDEX}_state_dict"
policy.load_state_dict(checkpoint[state_dict_key])
policy.eval()

print(f"Loaded MDPO checkpoint: {CHECKPOINT_PATH}")
print(f"  Policy index: {POLICY_INDEX} of 2")
print(f"  Iteration: {checkpoint.get('iter', '?')}")
print(f"  Policy class: {policy.__class__.__name__}")
print(f"  Network: actor={ACTOR_HIDDEN_DIMS}, critic={CRITIC_HIDDEN_DIMS}, act={ACTIVATION}")
print(f"  Obs dim: {num_obs}, Action dim: {num_actions}")

# ==============================================================================
# Play loop
# ==============================================================================
while simulation_app.is_running():
    with torch.inference_mode():
        obs_flat = _flatten_obs(obs)
        actions = policy.act_inference(obs_flat)
        obs, _, dones, _ = env.step(actions)
