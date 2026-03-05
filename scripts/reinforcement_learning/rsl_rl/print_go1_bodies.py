"""Minimal script to load Go1 and print all body names."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# Create sim
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

# Spawn Go1
cfg = UNITREE_GO1_CFG.replace(prim_path="/World/Go1")
robot = Articulation(cfg)

sim.reset()
robot.update(sim.get_physics_dt())

print("\n=== Go1 Body (Link) Names ===")
for i, name in enumerate(robot.body_names):
    print(f"  [{i:2d}] {name}")

print(f"\n=== Go1 Joint Names ===")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

simulation_app.close()
