"""
Simple training script for Isaac-Reach-OpenArm-Bi-v0 with RSL-RL (PPO).

This is a simplified, Hydra-free version of train.py.
It uses the exact same config dataclasses, just instantiated directly in Python.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/simple_train.py
"""

# ==============================================================================
# Step 0: Launch Isaac Sim (must happen before any other Isaac imports)
# ==============================================================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

# ==============================================================================
# Now we can import everything else
# ==============================================================================
import tempfile
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import the config classes directly (no Hydra lookup needed)
from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.agents.rsl_rl_ppo_cfg import (
    OpenArmReachPPORunnerCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ==============================================================================
# Step 1: Configure the environment
# ==============================================================================
env_cfg = OpenArmReachEnvCfg()
env_cfg.scene.num_envs = 4096
env_cfg.seed = 42
env_cfg.sim.device = "cuda:0"

# ==============================================================================
# Step 2: Configure the PPO agent
#
# This is the exact same dataclass that train.py loads via Hydra.
# You can modify any field directly, e.g.:
#   agent_cfg.algorithm.learning_rate = 3e-4
#   agent_cfg.policy.actor_hidden_dims = [256, 256]
#   agent_cfg.max_iterations = 1000
# ==============================================================================
agent_cfg = OpenArmReachPPORunnerCfg()

# ==============================================================================
# Step 3: Create the environment
# ==============================================================================
env = gym.make("Isaac-Reach-OpenArm-Bi-v0", cfg=env_cfg)

# ==============================================================================
# Step 4: Wrap the environment for RSL-RL
# ==============================================================================
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ==============================================================================
# Step 5: Create the OnPolicyRunner
#
# agent_cfg.to_dict() converts the dataclass hierarchy into the nested dict
# that OnPolicyRunner expects. This is identical to what train.py does.
# ==============================================================================
log_dir = tempfile.mkdtemp(prefix="rsl_rl_")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

# ==============================================================================
# Step 6: Train!
# ==============================================================================
start_time = time.time()

runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

print(f"Training time: {round(time.time() - start_time, 2)} seconds")

# ==============================================================================
# Step 7: Clean up
# ==============================================================================
env.close()
simulation_app.close()
