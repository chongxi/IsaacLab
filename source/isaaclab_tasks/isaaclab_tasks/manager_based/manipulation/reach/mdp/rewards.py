# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_when_close(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    position_threshold: float,
) -> torch.Tensor:
    """Penalize orientation error only when position error is within a threshold.

    If the end-effector position error is larger than ``position_threshold``, this
    term returns zero for that environment.
    """
    # extract the asset
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # position error gate
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    gate = pos_error <= position_threshold

    # orientation error
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ori_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    return torch.where(gate, ori_error, torch.zeros_like(ori_error))


def ee_pose_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Explicit EE pose error w.r.t command: [position_error(3), orientation_error_axis_angle(3)].

    This is useful as an observation term for policies that should consume direct
    task-space errors.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, des_quat_w = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b, des_quat_b)

    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    pos_error, ori_error_axis_angle = compute_pose_error(
        curr_pos_w,
        curr_quat_w,
        des_pos_w,
        des_quat_w,
        rot_error_type="axis_angle",
    )
    return torch.cat([pos_error, ori_error_axis_angle], dim=-1)


def position_command_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward progress in position tracking error (previous distance minus current distance).

    Positive value means the end-effector moved closer to the commanded target
    in this step. Negative value means it moved farther away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_dist = torch.norm(curr_pos_w - des_pos_w, dim=1)

    prev_dist_cache = getattr(env, "_position_progress_prev_dist", None)
    if prev_dist_cache is None:
        prev_dist_cache = {}
        setattr(env, "_position_progress_prev_dist", prev_dist_cache)

    prev_cmd_cache = getattr(env, "_position_progress_prev_cmd", None)
    if prev_cmd_cache is None:
        prev_cmd_cache = {}
        setattr(env, "_position_progress_prev_cmd", prev_cmd_cache)

    cache_key = f"{asset_cfg.name}:{asset_cfg.body_ids}:{command_name}"
    prev_dist = prev_dist_cache.get(cache_key)
    prev_cmd = prev_cmd_cache.get(cache_key)
    if prev_dist is None:
        prev_dist = curr_dist.clone()
    if prev_cmd is None:
        prev_cmd = des_pos_b.detach().clone()

    progress = prev_dist - curr_dist

    # Reset progress at command resampling boundaries to avoid artificial spikes
    # from target jumps (not due to robot motion).
    cmd_changed = (des_pos_b - prev_cmd).abs().max(dim=1).values > 1e-6

    if hasattr(env, "episode_length_buf"):
        is_new_episode = env.episode_length_buf == 0
        progress = torch.where(is_new_episode | cmd_changed, torch.zeros_like(progress), progress)
    else:
        progress = torch.where(cmd_changed, torch.zeros_like(progress), progress)

    prev_dist_cache[cache_key] = curr_dist.detach().clone()
    prev_cmd_cache[cache_key] = des_pos_b.detach().clone()
    return progress
