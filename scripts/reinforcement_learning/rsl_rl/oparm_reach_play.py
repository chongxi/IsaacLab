"""
Play a trained checkpoint for Isaac-Reach-OpenArm-Bi-v0.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/custom_play.py
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
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.agents.rsl_rl_ppo_cfg import (
    OpenArmReachPPORunnerCfg,
)

# ==============================================================================
# CheckPoint 
# ==============================================================================
CHECKPOINT_PATH = "./logs/rsl_rl/openarm_bi_reach/2026-02-25_15-53-50/model_550.pt"

# ==============================================================================
# Configuration of the environment and agent
# ==============================================================================
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 8  # fewer envs for visualization
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

agent_cfg = OpenArmReachPPORunnerCfg()

# ==============================================================================
# Create environment
# ==============================================================================
env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ==============================================================================
# Load the trained policy
# ==============================================================================
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CHECKPOINT_PATH)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(type(policy))

# ==============================================================================
# Play loop
# ==============================================================================
obs, info = env.reset()

while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        # print(f"Actions: {actions.cpu().numpy()}")  # print actions for debugging
