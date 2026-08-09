from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def lateral_conditioned_joint_std(
  command: torch.Tensor,
  base_std: torch.Tensor,
  hip_mask: torch.Tensor,
  *,
  max_hip_std: float = 0.30,
  full_lateral_command: float = 0.30,
  yaw_scale: float = 1.0,
) -> torch.Tensor:
  """Relax hip tolerance continuously for lateral-dominant commands."""
  if full_lateral_command <= 0.0:
    raise ValueError("full_lateral_command must be positive.")
  if yaw_scale < 0.0:
    raise ValueError("yaw_scale must be non-negative.")
  if command.ndim == 0 or command.shape[-1] != 3:
    raise ValueError("command must have exactly three components on its last axis.")

  lateral_dominance = torch.abs(command[..., 1]) - torch.maximum(
    torch.abs(command[..., 0]), yaw_scale * torch.abs(command[..., 2])
  )
  alpha = torch.clamp(lateral_dominance / full_lateral_command, 0.0, 1.0)
  alpha = alpha.to(device=base_std.device, dtype=base_std.dtype).unsqueeze(-1)
  hip_mask = hip_mask.to(device=base_std.device, dtype=torch.bool)
  relaxed_std = base_std + alpha * (max_hip_std - base_std)
  return torch.where(hip_mask, relaxed_std, base_std)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def contact_force_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  soft_threshold: float = 5.0,
  force_scale: float = 20.0,
  max_cost_per_substep: float = 2.0,
  metric_prefix: str | None = None,
) -> torch.Tensor:
  """Penalize non-foot contact force above a soft threshold.

  The strongest contact is used for each stored simulation substep so the
  value is not inflated by the number of collision geoms. The clipped linear
  excess keeps brief brushes cheap while preserving pressure against loading
  the body or legs on the terrain.
  """
  if force_scale <= 0.0:
    raise ValueError("force_scale must be positive")
  if max_cost_per_substep <= 0.0:
    raise ValueError("max_cost_per_substep must be positive")

  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    peak_force = force_mag.amax(dim=1)  # [B, H]
  else:
    assert data.force is not None
    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    peak_force = force_mag.amax(dim=1, keepdim=True)  # [B, 1]

  cost = torch.clamp(
    (peak_force - soft_threshold).clamp_min(0.0) / force_scale,
    max=max_cost_per_substep,
  ).sum(dim=-1)
  metric_prefix = metric_prefix or "nonfoot_contact"
  active = peak_force > soft_threshold
  active_count = active.float().sum()
  env.extras["log"][f"Metrics/{metric_prefix}_force_mean"] = peak_force.mean()
  env.extras["log"][f"Metrics/{metric_prefix}_contact_rate"] = active.float().mean()
  env.extras["log"][f"Metrics/{metric_prefix}_force_when_active"] = (
    (peak_force * active.float()).sum() / torch.clamp(active_count, min=1.0)
  )
  # Keep the original key for V4 log compatibility.
  if metric_prefix == "nonfoot_contact":
    env.extras["log"]["Metrics/nonfoot_contact_force_mean"] = peak_force.mean()
  return cost


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_clearance_terrain_relative(
  env: ManagerBasedRlEnv,
  target_height: float,
  terrain_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  max_horizontal_distance: float = 0.2,
  contact_sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot clearance relative to the terrain below each foot.

  Each foot uses the closest valid point in the yaw-aligned terrain scan.
  Samples farther than ``max_horizontal_distance`` are ignored, which avoids
  applying a misleading target when a foot leaves the scan footprint.
  """
  if max_horizontal_distance <= 0.0:
    raise ValueError("max_horizontal_distance must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  terrain_sensor: RayCastSensor = env.scene[terrain_sensor_name]

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, F, 3]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  hit_pos_w = terrain_sensor.data.hit_pos_w  # [B, R, 3]
  valid_hits = terrain_sensor.data.distances >= 0.0  # [B, R]

  horizontal_error = foot_pos_w[:, :, None, :2] - hit_pos_w[:, None, :, :2]
  horizontal_distance_sq = torch.sum(torch.square(horizontal_error), dim=-1)
  horizontal_distance_sq = horizontal_distance_sq.masked_fill(
    ~valid_hits[:, None, :], torch.inf
  )
  nearest_distance_sq, nearest_ids = horizontal_distance_sq.min(dim=-1)
  terrain_z = torch.gather(hit_pos_w[..., 2], dim=1, index=nearest_ids)

  valid_terrain = nearest_distance_sq <= max_horizontal_distance**2
  clearance = foot_pos_w[..., 2] - terrain_z
  delta = torch.abs(clearance - target_height)
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(delta * vel_norm * valid_terrain.float(), dim=1)

  if contact_sensor_name is not None:
    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    assert contact_sensor.data.found is not None
    in_air = (contact_sensor.data.found <= 0).float()
    cost = torch.sum(delta * vel_norm * valid_terrain.float() * in_air, dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def terrain_relative_loaded_stance_slip_cost(
  foot_pos_w: torch.Tensor,
  foot_vel_w: torch.Tensor,
  contact_force_w: torch.Tensor,
  in_contact: torch.Tensor,
  hit_pos_w: torch.Tensor,
  hit_normals_w: torch.Tensor,
  hit_distances: torch.Tensor,
  *,
  normal_force_threshold: float = 15.0,
  max_horizontal_distance: float = 0.25,
  slip_deadband: float = 0.03,
  slip_scale: float = 0.10,
  max_cost_per_foot: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute load-normalized local-tangent slip cost.

  The contact sensor used by the Go2 task reports the net force on the terrain
  for foot-primary/terrain-secondary matches.  Consequently, the supporting
  normal load is ``-dot(contact_force_w, terrain_normal_w)``.
  """
  if normal_force_threshold < 0.0:
    raise ValueError("normal_force_threshold must be non-negative")
  if max_horizontal_distance <= 0.0:
    raise ValueError("max_horizontal_distance must be positive")
  if slip_deadband < 0.0:
    raise ValueError("slip_deadband must be non-negative")
  if slip_scale <= 0.0:
    raise ValueError("slip_scale must be positive")
  if max_cost_per_foot <= 0.0:
    raise ValueError("max_cost_per_foot must be positive")

  valid_hits = hit_distances >= 0.0
  horizontal_delta = foot_pos_w[:, :, None, :2] - hit_pos_w[:, None, :, :2]
  horizontal_distance_sq = torch.sum(torch.square(horizontal_delta), dim=-1)
  horizontal_distance_sq = horizontal_distance_sq.masked_fill(
    ~valid_hits[:, None, :], torch.inf
  )
  nearest_distance_sq, nearest_ids = horizontal_distance_sq.min(dim=-1)
  gather_ids = nearest_ids[..., None].expand(-1, -1, 3)
  nearest_normals = torch.gather(hit_normals_w, dim=1, index=gather_ids)

  normal_norm = torch.linalg.vector_norm(nearest_normals, dim=-1, keepdim=True)
  valid_terrain = (
    (nearest_distance_sq <= max_horizontal_distance**2)
    & (normal_norm.squeeze(-1) > 1.0e-8)
  )
  terrain_normal = nearest_normals / normal_norm.clamp_min(1.0e-8)
  terrain_normal = torch.where(
    terrain_normal[..., 2:3] < 0.0, -terrain_normal, terrain_normal
  )

  tangent_velocity = foot_vel_w - (
    foot_vel_w * terrain_normal
  ).sum(dim=-1, keepdim=True) * terrain_normal
  slip_velocity = torch.linalg.vector_norm(tangent_velocity, dim=-1)
  normal_force = (-(contact_force_w * terrain_normal).sum(dim=-1)).clamp_min(0.0)
  loaded = (
    in_contact.bool()
    & valid_terrain
    & (normal_force >= normal_force_threshold)
  )

  slip_excess = (slip_velocity - slip_deadband).clamp_min(0.0)
  per_foot_cost = torch.clamp(
    torch.square(slip_excess / slip_scale), max=max_cost_per_foot
  )
  load_weight = torch.where(loaded, normal_force, torch.zeros_like(normal_force))
  total_load = load_weight.sum(dim=-1)
  cost = (per_foot_cost * load_weight).sum(dim=-1) / total_load.clamp_min(1.0e-8)
  cost = torch.where(total_load > 0.0, cost, torch.zeros_like(cost))
  return cost, slip_velocity, loaded, valid_terrain, normal_force


class terrain_relative_loaded_stance_slip:
  """Penalize terrain-tangent slip only for load-bearing stance feet."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    foot_geom_names = tuple(cfg.params["foot_geom_names"])
    site_names = tuple(asset_cfg.site_names)
    if len(site_names) != len(foot_geom_names):
      raise ValueError("site_names and foot_geom_names must have the same length")

    asset: Entity = env.scene[asset_cfg.name]
    _, found_site_names = asset.find_sites(site_names, preserve_order=True)
    if tuple(found_site_names) != site_names:
      raise ValueError(f"foot site order mismatch: {found_site_names}")

    contact_sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    sensor_geom_names = [
      slot.primary_name
      for slot in contact_sensor._slots
      if slot.field_name == "found"
    ]
    missing = [name for name in foot_geom_names if name not in sensor_geom_names]
    if missing:
      raise ValueError(f"foot contact sensor is missing geoms: {missing}")
    self._sensor_permutation = torch.tensor(
      [sensor_geom_names.index(name) for name in foot_geom_names],
      device=env.device,
      dtype=torch.long,
    )
    self._num_feet = len(site_names)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    terrain_sensor_name: str,
    foot_geom_names: tuple[str, ...],
    command_name: str,
    command_threshold: float,
    normal_force_threshold: float,
    max_horizontal_distance: float,
    slip_deadband: float,
    slip_scale: float,
    max_cost_per_foot: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del foot_geom_names  # Used to establish the contact/site permutation at init.
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    terrain_sensor: RayCastSensor = env.scene[terrain_sensor_name]

    assert contact_sensor.data.found is not None
    assert contact_sensor.data.force is not None
    in_contact = contact_sensor.data.found.reshape(
      env.num_envs, self._num_feet, -1
    ).any(dim=-1).index_select(1, self._sensor_permutation)
    contact_force_w = contact_sensor.data.force.reshape(
      env.num_envs, self._num_feet, 3
    ).index_select(1, self._sensor_permutation)

    cost, slip_velocity, loaded, valid_terrain, normal_force = (
      terrain_relative_loaded_stance_slip_cost(
        asset.data.site_pos_w[:, asset_cfg.site_ids, :],
        asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :],
        contact_force_w,
        in_contact,
        terrain_sensor.data.hit_pos_w,
        terrain_sensor.data.normals_w,
        terrain_sensor.data.distances,
        normal_force_threshold=normal_force_threshold,
        max_horizontal_distance=max_horizontal_distance,
        slip_deadband=slip_deadband,
        slip_scale=slip_scale,
        max_cost_per_foot=max_cost_per_foot,
      )
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_magnitude = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    active = command_magnitude > command_threshold
    active_loaded = loaded & active[:, None]
    active_loaded_count = active_loaded.float().sum()
    active_cost = cost * active.float()
    env.extras["log"]["Metrics/terrain_tangent_stance_slip_mean"] = (
      (slip_velocity * active_loaded.float()).sum()
      / active_loaded_count.clamp_min(1.0)
    )
    env.extras["log"]["Metrics/terrain_tangent_loaded_fraction"] = (
      active_loaded.float().mean()
    )
    env.extras["log"]["Metrics/terrain_tangent_ray_valid_fraction"] = (
      valid_terrain.float().mean()
    )
    env.extras["log"]["Metrics/terrain_tangent_normal_force_mean"] = (
      (normal_force * active_loaded.float()).sum()
      / active_loaded_count.clamp_min(1.0)
    )
    env.extras["log"]["Metrics/terrain_tangent_slip_cost_mean"] = active_cost.mean()
    return active_cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    lateral_hip_joint_pattern = cfg.params.get("lateral_hip_joint_pattern")
    self.lateral_hip_mask: torch.Tensor | None = None
    if lateral_hip_joint_pattern is not None:
      if cfg.params.get("full_lateral_command", 0.30) <= 0.0:
        raise ValueError("full_lateral_command must be positive.")

      self.lateral_hip_mask = torch.tensor(
        [
          re.fullmatch(lateral_hip_joint_pattern, joint_name) is not None
          for joint_name in joint_names
        ],
        device=env.device,
        dtype=torch.bool,
      )
      if not torch.any(self.lateral_hip_mask):
        raise ValueError(
          "lateral_hip_joint_pattern must match at least one configured joint."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    lateral_hip_joint_pattern: str | None = None,
    lateral_hip_std: float = 0.30,
    full_lateral_command: float = 0.30,
    lateral_yaw_scale: float = 1.0,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Resolved during initialization.
    del lateral_hip_joint_pattern  # Used to build the joint mask at initialization.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )
    if self.lateral_hip_mask is not None:
      std = lateral_conditioned_joint_std(
        command,
        std,
        self.lateral_hip_mask,
        max_hip_std=lateral_hip_std,
        full_lateral_command=full_lateral_command,
        yaw_scale=lateral_yaw_scale,
      )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
