# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp import base_height_l2
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import foot_clearance_reward, GaitReward
from .rough_env_cfg import UnitreeGo1RoughEnvCfg


@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        self.rewards.base_height = RewTerm(
            func=base_height_l2,
            weight=-30.0,
            params={
                "target_height": 0.32,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_reward,
            weight=0.8,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.08,  # Go1 is smaller than Spot, so lower target
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Trot gait reward: sync diagonal pairs (FL+RR, FR+RL)
        self.rewards.gait = RewTerm(
            func=GaitReward,
            weight=1.5,
            params={
                "std": 0.1,
                "max_err": 0.2,
                "velocity_threshold": 0.5,
                "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )

        # Penalize feet sliding on ground (prevents hind leg dragging)
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Penalize swing feet that don't lift enough (min 4cm)
        self.rewards.min_foot_lift = RewTerm(
            func=mdp.min_foot_lift,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "min_height": 0.04,
            },
        )

        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
