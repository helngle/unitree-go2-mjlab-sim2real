from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ModeVelocityCommand(UniformVelocityCommand):
  """Sample explicit locomotion modes, with focused high-speed terrain exposure."""

  cfg: ModeVelocityCommandCfg

  MODE_GENERAL = 0
  MODE_LATERAL = 1
  MODE_YAW = 2
  MODE_HIGH_SPEED = 3
  MODE_NAMES = ("general", "lateral", "yaw", "high_speed")

  def __init__(self, cfg: ModeVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.mode_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._base_probabilities = torch.tensor(
      (
        cfg.general_probability,
        cfg.lateral_probability,
        cfg.yaw_probability,
        cfg.high_speed_probability,
      ),
      device=self.device,
    )
    self._focus_terrain_columns = self._build_focus_terrain_columns()
    for mode_name in self.MODE_NAMES:
      self.metrics[f"mode_{mode_name}"] = torch.zeros(
        self.num_envs, device=self.device
      )
    self.metrics["mode_standing"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def _build_focus_terrain_columns(self) -> torch.Tensor | None:
    terrain = self._env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      return None

    generator_cfg = terrain.cfg.terrain_generator
    terrain_names = list(generator_cfg.sub_terrains)
    proportions = np.asarray(
      [
        sub_terrain.proportion
        for sub_terrain in generator_cfg.sub_terrains.values()
      ],
      dtype=np.float64,
    )
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)
    column_names = [
      terrain_names[
        int(np.where(column / generator_cfg.num_cols + 0.001 < cumulative)[0][0])
      ]
      for column in range(generator_cfg.num_cols)
    ]
    focus_names = set(self.cfg.focus_terrain_names)
    focus_columns = torch.tensor(
      [name in focus_names for name in column_names],
      dtype=torch.bool,
      device=self.device,
    )
    if focus_names and not focus_columns.any():
      raise ValueError(
        "None of the focus terrain names occur in the terrain generator: "
        f"{sorted(focus_names)}."
      )
    return focus_columns

  def _mode_probabilities(self, env_ids: torch.Tensor) -> torch.Tensor:
    probabilities = self._base_probabilities.repeat(len(env_ids), 1)
    terrain = self._env.scene.terrain
    if terrain is None or self._focus_terrain_columns is None:
      return probabilities

    terrain_types = terrain.terrain_types[env_ids]
    terrain_levels = terrain.terrain_levels[env_ids]
    focus_mask = (
      self._focus_terrain_columns[terrain_types]
      & (terrain_levels >= self.cfg.focus_min_terrain_level)
    )
    if not focus_mask.any():
      return probabilities

    base_high_speed = self.cfg.high_speed_probability
    other_scale = (
      (1.0 - self.cfg.focus_high_speed_probability)
      / (1.0 - base_high_speed)
    )
    probabilities[focus_mask] *= other_scale
    probabilities[focus_mask, self.MODE_HIGH_SPEED] = (
      self.cfg.focus_high_speed_probability
    )
    return probabilities

  def _uniform(
    self, count: int, value_range: tuple[float, float]
  ) -> torch.Tensor:
    return torch.empty(count, device=self.device).uniform_(*value_range)

  def _signed_uniform(
    self, count: int, magnitude_range: tuple[float, float]
  ) -> torch.Tensor:
    magnitude = self._uniform(count, magnitude_range)
    sign = torch.where(
      torch.rand(count, device=self.device) < 0.5,
      -torch.ones(count, device=self.device),
      torch.ones(count, device=self.device),
    )
    return magnitude * sign

  def _lateral_speed_range(self) -> tuple[float, float]:
    relative_step = max(
      self._env.common_step_counter - self.cfg.stage_origin_step, 0
    )
    speed_range = self.cfg.lateral_speed
    for step, stage_range in self.cfg.lateral_speed_stages:
      if relative_step >= step:
        speed_range = stage_range
    return speed_range

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Preserve standard standing/heading bookkeeping, then replace the sampled
    # velocity with an explicit locomotion mode.
    super()._resample_command(env_ids)
    probabilities = self._mode_probabilities(env_ids)
    sampled_modes = torch.multinomial(probabilities, 1).squeeze(1)
    self.mode_ids[env_ids] = sampled_modes
    self.vel_command_b[env_ids] = 0.0

    general_ids = env_ids[sampled_modes == self.MODE_GENERAL]
    self.vel_command_b[general_ids, 0] = self._uniform(
      len(general_ids), self.cfg.general_lin_vel_x
    )
    self.vel_command_b[general_ids, 1] = self._uniform(
      len(general_ids), self.cfg.general_lin_vel_y
    )
    self.vel_command_b[general_ids, 2] = self._uniform(
      len(general_ids), self.cfg.general_ang_vel_z
    )

    lateral_ids = env_ids[sampled_modes == self.MODE_LATERAL]
    self.vel_command_b[lateral_ids, 1] = self._signed_uniform(
      len(lateral_ids), self._lateral_speed_range()
    )

    yaw_ids = env_ids[sampled_modes == self.MODE_YAW]
    self.vel_command_b[yaw_ids, 2] = self._signed_uniform(
      len(yaw_ids), self.cfg.yaw_speed
    )

    high_speed_ids = env_ids[sampled_modes == self.MODE_HIGH_SPEED]
    self.vel_command_b[high_speed_ids, 0] = self._uniform(
      len(high_speed_ids), self.cfg.high_speed_lin_vel_x
    )
    self.vel_command_b[high_speed_ids, 1] = self._uniform(
      len(high_speed_ids), self.cfg.high_speed_lin_vel_y
    )
    self.vel_command_b[high_speed_ids, 2] = self._uniform(
      len(high_speed_ids), self.cfg.high_speed_ang_vel_z
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    normalizer = max(self._env.max_episode_length, 1)
    for mode_id, mode_name in enumerate(self.MODE_NAMES):
      self.metrics[f"mode_{mode_name}"] += (
        self.mode_ids == mode_id
      ).float() / normalizer
    self.metrics["mode_standing"] += (
      self.is_standing_env.float() / normalizer
    )


@dataclass(kw_only=True)
class ModeVelocityCommandCfg(UniformVelocityCommandCfg):
  general_probability: float = 0.40
  lateral_probability: float = 0.25
  yaw_probability: float = 0.15
  high_speed_probability: float = 0.20
  focus_high_speed_probability: float = 0.45
  focus_min_terrain_level: int = 7
  focus_terrain_names: tuple[str, ...] = (
    "pyramid_stairs",
    "hf_pyramid_slope_inv",
  )

  general_lin_vel_x: tuple[float, float] = (0.15, 0.8)
  general_lin_vel_y: tuple[float, float] = (-0.1, 0.1)
  general_ang_vel_z: tuple[float, float] = (-0.3, 0.3)
  lateral_speed: tuple[float, float] = (0.1, 0.3)
  lateral_speed_stages: tuple[
    tuple[int, tuple[float, float]], ...
  ] = ()
  stage_origin_step: int = 0
  yaw_speed: tuple[float, float] = (0.2, 0.7)
  high_speed_lin_vel_x: tuple[float, float] = (0.8, 1.0)
  high_speed_lin_vel_y: tuple[float, float] = (-0.05, 0.05)
  high_speed_ang_vel_z: tuple[float, float] = (-0.15, 0.15)

  def build(self, env: ManagerBasedRlEnv) -> ModeVelocityCommand:
    return ModeVelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    probabilities = (
      self.general_probability,
      self.lateral_probability,
      self.yaw_probability,
      self.high_speed_probability,
    )
    if any(probability < 0.0 for probability in probabilities):
      raise ValueError("Mode probabilities must be non-negative.")
    if not np.isclose(sum(probabilities), 1.0):
      raise ValueError("Mode probabilities must sum to 1.0.")
    if not 0.0 <= self.focus_high_speed_probability <= 1.0:
      raise ValueError("focus_high_speed_probability must be in [0, 1].")
    if self.high_speed_probability >= 1.0:
      raise ValueError("high_speed_probability must be less than 1.0.")
    if self.heading_command:
      raise ValueError("ModeVelocityCommand does not support heading commands.")
    if self.init_velocity_prob != 0.0:
      raise ValueError("ModeVelocityCommand requires init_velocity_prob=0.")
    previous_step = -1
    for step, speed_range in self.lateral_speed_stages:
      if step < 0 or step <= previous_step:
        raise ValueError(
          "lateral_speed_stages must have strictly increasing non-negative steps."
        )
      if speed_range[0] < 0.0 or speed_range[0] > speed_range[1]:
        raise ValueError("Invalid lateral speed stage range.")
      previous_step = step
