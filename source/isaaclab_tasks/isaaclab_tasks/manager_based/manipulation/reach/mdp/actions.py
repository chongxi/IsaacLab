# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class EMARelativeJointPositionAction(RelativeJointPositionAction):
    r"""Relative joint position action with exponential moving average (EMA) smoothing.

    This smooths the processed delta action before applying:

    .. math::

        \Delta q_t^{ema} = \alpha \cdot \Delta q_t + (1-\alpha) \cdot \Delta q_{t-1}^{ema}

    The final target remains relative-position control implemented by
    :class:`RelativeJointPositionAction`:

    .. math::

        q_t^{target} = q_t^{current} + \Delta q_t^{ema}
    """

    cfg: EMARelativeJointPositionActionCfg

    def __init__(self, cfg: EMARelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)

        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"EMA alpha must be in [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"EMA alpha must be in [0, 1]. Got {value} for joint {name}.")
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported alpha type: {type(cfg.alpha)}. Use float or dict[str, float].")

        self._prev_smoothed_delta = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._prev_smoothed_delta[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # Base relative action processing: affine transform (+ optional clip)
        super().process_actions(actions)

        # EMA on processed deltas
        ema_delta = self._alpha * self._processed_actions + (1.0 - self._alpha) * self._prev_smoothed_delta
        self._processed_actions[:] = ema_delta

        # If explicit clip is configured, enforce it again after EMA
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        self._prev_smoothed_delta[:] = self._processed_actions


@configclass
class EMARelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for EMA-smoothed relative joint position action term."""

    class_type: type[ActionTerm] = EMARelativeJointPositionAction

    alpha: float | dict[str, float] = 1.0
    """EMA blending factor in [0, 1].

    If set to 1.0, no smoothing is applied.
    """
