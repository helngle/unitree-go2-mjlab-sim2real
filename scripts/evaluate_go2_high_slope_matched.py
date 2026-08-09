"""Strict matched straight/arc/S evaluation on high pyramid slopes.

Each route kind is evaluated in a fresh environment constructed with the same
seed and matched-slot order.  This harness changes only evaluation-time terrain,
episode duration, randomization profile, commands, and initial placement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends

from scripts.evaluate_go2_curved_routes import (
  _configure_episode_length,
  _configure_profile,
)
from src.tasks.velocity.evaluation.curved_routes import (
  ArcRoute,
  SRoute,
  arc_command_controller,
  arc_route_errors,
  make_arc_route,
  make_s_route,
  s_command_controller,
  s_route_errors,
)
from src.tasks.velocity.evaluation.high_slope_matched import (
  PROFILE_NAMES,
  ROUTE_KINDS,
  ROUTE_LENGTH_DEFINITION,
  SLOPE_DIRECTIONS,
  build_matched_scenarios,
  effective_slope_parameters,
  geometry_preflight,
  validate_horizon,
  validate_matched_result_invariants,
  validate_route_footprint,
)
from src.tasks.velocity.evaluation.matched_route_metrics import (
  matched_route_length,
)
from src.tasks.velocity.evaluation.proprio_acceptance import (
  actuator_effort_and_power,
  base_pitch_absolute,
  formal_evaluation_provenance,
  normalized_action_safety,
  processed_joint_target_safety,
)
from src.tasks.velocity.evaluation.routes import (
  route_frame_errors,
  straight_line_controller,
  update_attempt_status,
)
from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  make_high_difficulty_curve_generator,
)
from src.tasks.velocity.evaluation.terrain_curved_routes import (
  PATCH_SIZE,
  ROUTE_START_LOCAL,
  TERRAIN_KIND_TO_TYPE,
  relocate_root_pose,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  ACTIVE_SAMPLE_DEFINITION,
  BASE_PITCH_DEFINITION,
  OnlineTerrainRolloutMetrics,
  OnlineTerrainTangentSlipMetrics,
  TERRAIN_TANGENT_STANCE_SLIP_DEFINITION,
  action_acceleration,
  assert_recursive_json_finite,
  contact_any,
  foot_contact_any,
  foot_slip_velocity,
)
from src.tasks.velocity.mdp.rewards import (
  terrain_relative_loaded_stance_slip_cost,
)
from src.tasks.velocity.evaluation.transient_metrics import (
  OnlineCommandTransientMetrics,
  TransientMetricConfig,
)


@dataclass(frozen=True)
class HighSlopeMatchedConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  profiles: tuple[str, ...] = ("clean",)
  slope_directions: tuple[str, ...] = SLOPE_DIRECTIONS
  levels: tuple[int, ...] = (0, 1)
  radii: tuple[float, ...] = (2.5, 4.0)
  speeds: tuple[float, ...] = (0.3, 0.5)
  turn_signs: tuple[int, ...] = (1, -1)
  repeats: int = 1
  steps: int = 2400
  settle_steps: int = 10
  seed: int = 42
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  max_lateral_speed: float = 0.3
  max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.30
  heading_tolerance: float = math.radians(20.0)
  corridor_half_width: float = 0.4
  output_file: str = "go2_high_slope_matched_evaluation.json"


class _OnlineRouteErrors:
  def __init__(
    self, num_envs: int, max_steps: int, *, device: torch.device | str,
    dtype: torch.dtype,
  ) -> None:
    shape = (num_envs, max_steps)
    self.num_envs = num_envs
    self.max_steps = max_steps
    self.next_step = 0
    self.valid = torch.zeros(shape, dtype=torch.bool, device=device)
    self.cross = torch.zeros(shape, dtype=dtype, device=device)
    self.heading = torch.zeros(shape, dtype=dtype, device=device)

  def update(
    self, cross: torch.Tensor, heading: torch.Tensor, mask: torch.Tensor
  ) -> None:
    if self.next_step >= self.max_steps:
      raise RuntimeError("route error accumulator exceeded max_steps")
    if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
      raise ValueError("route error mask must be bool shape (num_envs,)")
    for name, value in (("cross", cross), ("heading", heading)):
      if value.shape != (self.num_envs,) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite shape (num_envs,)")
    self.valid[:, self.next_step] = mask
    self.cross[:, self.next_step] = cross.abs()
    self.heading[:, self.next_step] = heading.abs()
    self.next_step += 1

  @staticmethod
  def _distribution(values: torch.Tensor) -> dict[str, float | str | None]:
    values = values.to(torch.float64)
    if values.numel() == 0:
      return {
        "rms": None, "p95": None, "max": None,
        "reason": "no_active_control_step_samples",
      }
    return {
      "rms": float(torch.sqrt(torch.mean(values.square()))),
      "p95": float(torch.quantile(values, 0.95)),
      "max": float(values.max()),
    }

  def result(self, index: int) -> dict[str, object]:
    mask = self.valid[index, :self.next_step]
    return {
      "sample_count": int(mask.sum()),
      "cross_track_absolute": self._distribution(
        self.cross[index, :self.next_step][mask]
      ),
      "heading_absolute": self._distribution(
        self.heading[index, :self.next_step][mask]
      ),
    }


def _git_head() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1], text=True,
  ).strip()


def _validate_config(cfg: HighSlopeMatchedConfig) -> dict[str, object]:
  if not cfg.profiles or len(set(cfg.profiles)) != len(cfg.profiles):
    raise ValueError("profiles must be nonempty and unique")
  if any(profile not in PROFILE_NAMES for profile in cfg.profiles):
    raise ValueError(f"profiles must contain only {PROFILE_NAMES}")
  scenarios = build_matched_scenarios(
    slope_directions=cfg.slope_directions,
    levels=cfg.levels,
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    repeats=cfg.repeats,
  )
  if not scenarios:
    raise ValueError("scenario matrix must not be empty")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive")
  if cfg.settle_steps < 0 or cfg.settle_steps >= cfg.steps:
    raise ValueError("settle_steps must be in [0, steps)")
  for name, value in (
    ("cross_track_gain", cfg.cross_track_gain),
    ("heading_gain", cfg.heading_gain),
    ("max_lateral_speed", cfg.max_lateral_speed),
    ("max_yaw_rate", cfg.max_yaw_rate),
    ("cross_track_tolerance", cfg.cross_track_tolerance),
    ("heading_tolerance", cfg.heading_tolerance),
    ("corridor_half_width", cfg.corridor_half_width),
  ):
    if not math.isfinite(value) or value <= 0.0:
      raise ValueError(f"{name} must be finite and positive")
  preflight = geometry_preflight(cfg.radii, cfg.turn_signs)
  if not preflight["all_requested_combinations_valid"]:
    invalid = [
      item for item in preflight["combinations"]  # type: ignore[index]
      if not item["valid"]
    ]
    summary = "; ".join(
      f"{item['route_kind']} r={item['radius']} sign={item['turn_sign']} "
      f"corridor_margin={item['corridor_boundary_margin']:.6f} "
      f"scan_margin={item['scan_boundary_margin']:.6f}"
      for item in invalid
    )
    raise ValueError(
      "requested matched matrix is not geometrically valid on the fixed "
      f"18x18 centre-start patch: {summary}"
    )
  return preflight


def _scenarios(cfg: HighSlopeMatchedConfig) -> list[dict[str, Any]]:
  return [scenario.as_dict() for scenario in build_matched_scenarios(
    slope_directions=cfg.slope_directions,
    levels=cfg.levels,
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    repeats=cfg.repeats,
  )]


def _terrain_assignment(
  env: ManagerBasedRlEnv,
  scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
  """Relocate to exact slope patches and place every route at patch centre."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise RuntimeError("terrain origins are unavailable")
  robot = env.scene["robot"]
  device = env.device
  levels = torch.tensor(
    [item["level"] for item in scenarios], dtype=torch.long, device=device
  )
  types = torch.tensor(
    [TERRAIN_KIND_TO_TYPE[item["slope_direction"]] for item in scenarios],
    dtype=torch.long, device=device,
  )
  old_origins = terrain.env_origins.clone()
  old_root = robot.data.root_link_pose_w.clone()
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  new_origins = terrain.env_origins.clone()
  relative_before = old_root[:, :3] - old_origins
  relocated, arithmetic_error = relocate_root_pose(old_root, old_origins, new_origins)
  robot.write_root_link_pose_to_sim(relocated)
  env.scene.write_data_to_sim()
  env.sim.forward(); env.sim.sense()
  relocation_error_by_env = torch.amax(torch.abs(
    (robot.data.root_link_pos_w - new_origins) - relative_before
  ), dim=-1)
  relocation_error = max(arithmetic_error, float(relocation_error_by_env.max()))
  if relocation_error > 1.0e-4:
    raise RuntimeError(f"terrain relocation error {relocation_error:.6f}")

  clearance = relocated[:, 2] - new_origins[:, 2]
  root = robot.data.root_link_pose_w.clone()
  root[:, :2] = new_origins[:, :2]
  root[:, 2] = new_origins[:, 2] + clearance
  old_heading = robot.data.heading_w.clone()
  zeros = torch.zeros_like(old_heading)
  root[:, 3:7] = quat_mul(
    quat_from_euler_xyz(zeros, zeros, -old_heading), root[:, 3:7]
  )
  robot.write_root_link_pose_to_sim(root)
  env.scene.write_data_to_sim()
  env.sim.forward(); env.sim.sense()
  expected = torch.cat((new_origins[:, :2], root[:, 2:3]), dim=-1)
  placement_error_by_env = torch.maximum(
    torch.amax(torch.abs(robot.data.root_link_pos_w - expected), dim=-1),
    torch.abs(robot.data.heading_w),
  )
  placement_error = float(placement_error_by_env.max())
  if placement_error > 1.0e-4:
    raise RuntimeError(f"route placement error {placement_error:.6f}")
  patch_offset = torch.tensor(
    [ROUTE_START_LOCAL[0], ROUTE_START_LOCAL[1], 0.0],
    device=device, dtype=new_origins.dtype,
  )
  return {
    "route_start": new_origins[:, :2].clone(),
    "terrain_origins": new_origins,
    "patch_origins": new_origins - patch_offset,
    "clearance": clearance,
    "levels": levels,
    "types": types,
    "terrain_assignment_position_error": relocation_error_by_env,
    "route_placement_position_error": placement_error_by_env,
    "terrain_assignment_position_error_max": relocation_error,
    "route_placement_position_error_max": placement_error,
  }


def _contact(
  env: ManagerBasedRlEnv, name: str, num_envs: int
) -> torch.Tensor | None:
  try:
    return contact_any(env.scene[name].data.found, num_envs)
  except KeyError:
    return None


def _catastrophic(env: ManagerBasedRlEnv, num_envs: int) -> torch.Tensor:
  value = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  for name in env.termination_manager.active_terms:
    if not env.termination_manager.get_term_cfg(name).time_out:
      value |= env.termination_manager.get_term(name).bool()
  return value


def _foot_sensor_permutation(
  sensor: Any, foot_geom_names: tuple[str, ...], device: torch.device | str
) -> torch.Tensor:
  sensor_geom_names = [
    slot.primary_name for slot in sensor._slots if slot.field_name == "found"
  ]
  missing = [name for name in foot_geom_names if name not in sensor_geom_names]
  if missing:
    raise ValueError(f"foot contact sensor is missing geoms: {missing}")
  return torch.tensor(
    [sensor_geom_names.index(name) for name in foot_geom_names],
    dtype=torch.long,
    device=device,
  )


def _termination_reason(env: ManagerBasedRlEnv, index: int) -> str:
  names = [
    name for name in env.termination_manager.active_terms
    if bool(env.termination_manager.get_term(name)[index])
  ]
  return names[0] if names else "reset"


def _final_failure_reason(
  *, completed: bool, failed: bool, reason: str | None
) -> str | None:
  """Enforce null-on-success and a nonempty reason on every failed row."""
  if completed and failed:
    raise ValueError("scenario cannot be both completed and failed")
  if completed:
    return None
  if failed:
    if reason is None or not reason.strip():
      raise ValueError("failed scenario must provide a nonempty failure reason")
    return reason
  if reason is not None:
    raise ValueError("unfinished scenario cannot provide a failure reason")
  return None


def _contact_termination_summary(
  termination_counts: dict[str, int],
  body_contacts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
  terms = {
    "fell": "fell_over",
    "base": "illegal_base_contact",
    "upper_leg": "illegal_upper_leg_contact",
    "calf": "illegal_calf_contact",
  }
  result: dict[str, dict[str, Any]] = {}
  for body, term in terms.items():
    entry: dict[str, Any] = {
      "termination_term": term,
      "termination_count": termination_counts.get(term),
      "termination_available": term in termination_counts,
    }
    if body in body_contacts:
      entry.update(body_contacts[body])
    else:
      entry.update({
        "non_terminating_count": None,
        "non_terminating_rate": None,
        "reason": "non_terminating_contact_sensor_not_applicable",
      })
    result[body] = entry
  return result


def _evaluate_route_kind(
  cfg: HighSlopeMatchedConfig,
  profile: str,
  route_kind: str,
  scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
  """Evaluate one kind in a fresh, identically seeded environment."""
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  num_envs = len(scenarios)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  terrain_cfg = env_cfg.scene.terrain
  if terrain_cfg is None:
    raise RuntimeError("task has no terrain configuration")
  terrain_cfg.terrain_generator = make_high_difficulty_curve_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = _configure_profile(env_cfg, profile)
  episode = _configure_episode_length(env_cfg, cfg.steps)
  profile_settings.update(episode)
  required_horizon = validate_horizon(
    cfg.steps, radii=cfg.radii, speeds=cfg.speeds,
    control_dt=float(episode["control_dt"]), settle_steps=cfg.settle_steps,
  )
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command is not UniformVelocityCommand-compatible")
  if hasattr(command_cfg, "focus_terrain_names"):
    command_cfg.focus_terrain_names = ()
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (min(cfg.speeds), max(cfg.speeds))
  command_cfg.ranges.lin_vel_y = (-cfg.max_lateral_speed, cfg.max_lateral_speed)
  command_cfg.ranges.ang_vel_z = (-cfg.max_yaw_rate, cfg.max_yaw_rate)
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
    checkpoint = str(Path(cfg.checkpoint).expanduser().resolve())
    runner.load(
      checkpoint, load_cfg={"actor": True}, strict=True,
      map_location="cuda:0",
    )
    policy = runner.get_inference_policy(device="cuda:0")
    placement = _terrain_assignment(env, scenarios)
    route_start = placement["route_start"]
    robot = env.scene["robot"]
    device = env.device
    command_term = env.command_manager.get_term("twist")
    if not isinstance(command_term, UniformVelocityCommand):
      raise TypeError("twist command term is incompatible")

    route_lengths = torch.zeros(num_envs, device=device)
    endpoints = torch.zeros(num_envs, 2, device=device)
    endpoint_headings = torch.zeros(num_envs, device=device)
    group_indices: dict[tuple[float, float, int], list[int]] = {}
    for index, scenario in enumerate(scenarios):
      key = (scenario["radius"], scenario["speed"], scenario["turn_sign"])
      group_indices.setdefault(key, []).append(index)
    routes: dict[
      tuple[float, float, int], tuple[torch.Tensor, ArcRoute | SRoute | None]
    ] = {}
    for key, indices in group_indices.items():
      radius, _, turn_sign = key
      ids = torch.tensor(indices, dtype=torch.long, device=device)
      starts = route_start.index_select(0, ids)
      length = matched_route_length(radius)
      route_lengths[ids] = length
      if route_kind == "straight":
        route = None
        endpoints[ids] = starts + torch.tensor([length, 0.0], device=device)
      elif route_kind == "arc":
        route = make_arc_route(
          starts, torch.zeros(len(indices), device=device), radius, turn_sign,
          angle=2.0 * math.pi / 3.0,
        )
        endpoints[ids], endpoint_headings[ids] = route.pose_at(length)
      else:
        route = make_s_route(
          starts, torch.zeros(len(indices), device=device), radius, turn_sign
        )
        endpoints[ids], endpoint_headings[ids] = route.pose_at(length)
      routes[key] = (ids, route)

    def route_state(
      positions: torch.Tensor, headings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      progress = torch.zeros(num_envs, device=device)
      cross = torch.zeros_like(progress)
      heading_error = torch.zeros_like(progress)
      for ids, route in routes.values():
        group_positions = positions.index_select(0, ids)
        group_headings = headings.index_select(0, ids)
        if route is None:
          state = route_frame_errors(
            group_positions, group_headings,
            route_start.index_select(0, ids), 0.0,
          )
          p, c, h = state.progress, state.cross_track, state.heading_error
        elif isinstance(route, ArcRoute):
          p, c, h = arc_route_errors(route, group_positions, group_headings)
        else:
          p, c, h = s_route_errors(route, group_positions, group_headings)
        progress[ids], cross[ids], heading_error[ids] = p, c, h
      return progress, cross, heading_error

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    reached_endpoint = torch.zeros_like(active)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    settle_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
    reset_count = torch.zeros_like(settle_remaining)
    sample_count = torch.zeros(num_envs, device=device)
    final_progress = torch.zeros(num_envs, device=device)
    final_cross = torch.zeros_like(final_progress)
    final_heading = torch.zeros_like(final_progress)
    final_position = robot.data.root_link_pos_w[:, :2].clone()
    command_sum = torch.zeros(num_envs, 3, device=device)
    actual_sum = torch.zeros_like(command_sum)
    command_square_sum = torch.zeros_like(command_sum)
    command_actual_sum = torch.zeros_like(command_sum)
    saturation_count = torch.zeros(num_envs, device=device)
    termination_counts = {
      name: torch.zeros(num_envs, device=device)
      for name in env.termination_manager.active_terms
    }
    first_reason: list[str | None] = [None] * num_envs
    route_errors = _OnlineRouteErrors(
      num_envs, cfg.steps, device=device,
      dtype=robot.data.root_link_pos_w.dtype,
    )
    terrain_metrics = OnlineTerrainRolloutMetrics(
      num_envs, cfg.steps, device=device,
      dtype=robot.data.root_link_pos_w.dtype,
      control_dt_s=float(episode["control_dt"]),
    )
    transients = OnlineCommandTransientMetrics(
      num_envs, device=device, dtype=robot.data.root_link_pos_w.dtype,
      config=TransientMetricConfig(
        control_dt=float(episode["control_dt"]),
        num_segments=2 if route_kind == "s_curve" else 1,
      ),
    )
    foot_ids, found_foot_names = robot.find_sites(
      ("FR", "FL", "RR", "RL"), preserve_order=True
    )
    if tuple(found_foot_names) != ("FR", "FL", "RR", "RL"):
      raise RuntimeError(f"foot site order mismatch: {found_foot_names}")
    tangent_metrics = OnlineTerrainTangentSlipMetrics(
      num_envs, cfg.steps, len(foot_ids), device=device,
      dtype=robot.data.root_link_pos_w.dtype,
    )
    try:
      feet_sensor = env.scene["feet_ground_contact"]
    except KeyError:
      feet_sensor = None
    terrain_sensor = env.scene["terrain_scan"]
    foot_geom_names = tuple(
      f"{name}_foot_collision" for name in ("FR", "FL", "RR", "RL")
    )
    if feet_sensor is None:
      raise RuntimeError("loaded-stance evaluation requires feet_ground_contact")
    foot_permutation = _foot_sensor_permutation(
      feet_sensor, foot_geom_names, device
    )
    observation = wrapped.get_observations()
    executed_steps = 0
    for step_index in range(cfg.steps):
      if not bool(active.any()):
        break
      executed_steps = step_index + 1
      pre_pos = robot.data.root_link_pos_w[:, :2].clone()
      pre_heading = robot.data.heading_w.clone()
      pre_progress, pre_cross, pre_heading_error = route_state(
        pre_pos, pre_heading
      )
      motion_active = active & ~reached_endpoint
      command = torch.zeros(num_envs, 3, device=device)
      segment = torch.zeros(num_envs, dtype=torch.long, device=device)
      saturated = torch.zeros(num_envs, dtype=torch.bool, device=device)
      for key, (ids, route) in routes.items():
        radius, speed, _ = key
        group_pos = pre_pos.index_select(0, ids)
        group_heading = pre_heading.index_select(0, ids)
        if route is None:
          group = straight_line_controller(
            group_pos, group_heading, route_start.index_select(0, ids), 0.0,
            target_speed=speed,
            cross_track_gain=cfg.cross_track_gain,
            heading_gain=cfg.heading_gain,
            max_lateral_speed=cfg.max_lateral_speed,
            max_yaw_rate=cfg.max_yaw_rate,
            route_length=matched_route_length(radius),
          )
        else:
          controller = arc_command_controller if isinstance(route, ArcRoute) else s_command_controller
          group = controller(
            route, group_pos, group_heading,
            target_speed=speed,
            cross_track_gain=cfg.cross_track_gain,
            heading_gain=cfg.heading_gain,
            max_lateral_speed=cfg.max_lateral_speed,
            max_yaw_rate=cfg.max_yaw_rate,
          )
          if isinstance(route, SRoute):
            first_progress, _, _ = arc_route_errors(
              route.first, group_pos, group_heading
            )
            segment[ids] = (
              first_progress >= route.first.length - 1.0e-5
            ).long()
        command[ids] = group
        saturated[ids] = (
          (group[:, 1].abs() >= cfg.max_lateral_speed - 1.0e-6)
          | (group[:, 2].abs() >= cfg.max_yaw_rate - 1.0e-6)
        )
      command = torch.where(motion_active.unsqueeze(-1), command, 0.0)
      saturated &= motion_active
      command_term.vel_command_b[:] = command
      observation = wrapped.get_observations()
      with torch.inference_mode():
        action = policy(observation)
      _, _, dones, _ = wrapped.step(action)
      command_term.vel_command_b[:] = command
      observation = wrapped.get_observations()
      reset = dones.bool()
      reset_count += (reset & active).long()
      post_pos = robot.data.root_link_pos_w[:, :2]
      post_heading = robot.data.heading_w
      post_progress, post_cross, post_heading_error = route_state(
        post_pos, post_heading
      )
      progress = torch.where(reset, pre_progress, post_progress)
      cross = torch.where(reset, pre_cross, post_cross)
      heading_error = torch.where(reset, pre_heading_error, post_heading_error)
      lifecycle = update_attempt_status(
        motion_active, progress, cross, heading_error, reset,
        route_length=route_lengths,
        cross_track_tolerance=cfg.cross_track_tolerance,
        heading_tolerance=cfg.heading_tolerance,
      )
      sample_mask = active.clone()
      sample_f = sample_mask.float()
      sample_count += sample_f
      final_progress = torch.where(sample_mask, progress, final_progress)
      final_cross = torch.where(sample_mask, cross, final_cross)
      final_heading = torch.where(sample_mask, heading_error, final_heading)
      candidate_pos = torch.where(reset.unsqueeze(-1), pre_pos, post_pos)
      final_position = torch.where(sample_mask.unsqueeze(-1), candidate_pos, final_position)
      actual = torch.cat((
        robot.data.root_link_lin_vel_b[:, :2],
        robot.data.root_link_ang_vel_b[:, 2:3],
      ), dim=-1)
      actual = torch.where(reset.unsqueeze(-1), torch.zeros_like(actual), actual)
      command_sum += command * sample_f.unsqueeze(-1)
      actual_sum += actual * sample_f.unsqueeze(-1)
      command_square_sum += command.square() * sample_f.unsqueeze(-1)
      command_actual_sum += command * actual * sample_f.unsqueeze(-1)
      saturation_count += saturated.float() * sample_f
      route_errors.update(cross, heading_error, sample_mask)
      transients.update(
        step_index=step_index, command=command, actual=actual,
        segment_index=segment, sample_mask=sample_mask, saturated=saturated,
      )
      assert feet_sensor.data.found is not None
      assert feet_sensor.data.force is not None
      feet = foot_contact_any(
        feet_sensor.data.found, num_envs, len(foot_ids)
      ).index_select(1, foot_permutation)
      slip = foot_slip_velocity(
        robot.data.site_lin_vel_w[:, foot_ids, :2], feet
      )
      contact_force_w = feet_sensor.data.force.reshape(
        num_envs, len(foot_ids), 3
      ).index_select(1, foot_permutation)
      tangent_cost, tangent_slip, loaded, ray_valid, normal_force = (
        terrain_relative_loaded_stance_slip_cost(
          robot.data.site_pos_w[:, foot_ids, :],
          robot.data.site_lin_vel_w[:, foot_ids, :],
          contact_force_w,
          feet,
          terrain_sensor.data.hit_pos_w,
          terrain_sensor.data.normals_w,
          terrain_sensor.data.distances,
          normal_force_threshold=15.0,
          max_horizontal_distance=0.25,
          slip_deadband=0.03,
          slip_scale=0.10,
          max_cost_per_foot=4.0,
        )
      )
      tangent_metrics.update(
        sample_mask=sample_mask,
        cost=tangent_cost,
        slip_velocity=tangent_slip,
        loaded=loaded,
        ray_valid=ray_valid,
        normal_force=normal_force,
      )
      actuator_effort, mechanical_power = actuator_effort_and_power(robot)
      action_abs_max, action_fault = normalized_action_safety(
        env.action_manager.action
      )
      terrain_metrics.update(
        sample_mask=sample_mask,
        action_acceleration=action_acceleration(
          env.action_manager.action,
          env.action_manager.prev_action,
          env.action_manager.prev_prev_action,
        ),
        foot_slip_velocity=slip,
        body_contacts={
          "base": _contact(env, "base_ground_contact", num_envs),
          "upper_leg": _contact(env, "upper_leg_ground_contact", num_envs),
          "calf": _contact(env, "calf_ground_contact", num_envs),
        },
        catastrophic_termination=_catastrophic(env, num_envs),
        base_pitch=base_pitch_absolute(robot),
        actuator_effort_abs=actuator_effort,
        mechanical_power_abs=mechanical_power,
        normalized_action_abs_max=action_abs_max,
        action_safety_fault=action_fault,
        joint_target_safety_fault=processed_joint_target_safety(env),
      )
      for name in termination_counts:
        termination_counts[name] += (
          env.termination_manager.get_term(name).float() * sample_f
        )
      for index in torch.where(lifecycle.failed_now)[0].tolist():
        first_reason[index] = _termination_reason(env, index)
      reached_endpoint |= lifecycle.completed_now
      settle_remaining = torch.where(
        lifecycle.completed_now,
        torch.full_like(settle_remaining, cfg.settle_steps),
        settle_remaining,
      )
      settling_before = reached_endpoint & active
      settle_reset = settling_before & reset
      for index in torch.where(settle_reset)[0].tolist():
        first_reason[index] = _termination_reason(env, index)
      failed |= lifecycle.failed_now | settle_reset
      decrement = settling_before & ~lifecycle.completed_now & ~settle_reset
      settle_remaining = torch.where(
        decrement, (settle_remaining - 1).clamp_min(0), settle_remaining
      )
      settle_done = settling_before & (settle_remaining == 0) & ~settle_reset
      success |= settle_done
      active = (lifecycle.active | settling_before) & ~settle_done & ~settle_reset

    for index in torch.where(active)[0].tolist():
      first_reason[index] = "step_limit"
    failed |= active
    denom = sample_count.clamp_min(1.0)
    outputs: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
      footprint = validate_route_footprint(
        route_kind, scenario["radius"], scenario["turn_sign"],
        corridor_half_width=cfg.corridor_half_width,
      )
      terrain_result = terrain_metrics.result(index)
      tangent_result = tangent_metrics.result(index)
      errors = route_errors.result(index)
      if terrain_result["active_control_step_samples"] != int(sample_count[index]):
        raise RuntimeError("terrain metric sample mask diverged")
      if errors["sample_count"] != int(sample_count[index]):
        raise RuntimeError("route error sample mask diverged")
      response_gain: dict[str, float | str | None] = {}
      for axis, name in enumerate(("vx", "vy", "wz")):
        energy = float(command_square_sum[index, axis])
        if energy <= 1.0e-12:
          response_gain[name] = None
          response_gain[f"{name}_reason"] = "no_nonzero_command_energy"
        else:
          response_gain[name] = float(
            command_actual_sum[index, axis] / command_square_sum[index, axis]
          )
      command_mean = command_sum[index] / denom[index]
      actual_mean = actual_sum[index] / denom[index]
      route_error_cross = errors["cross_track_absolute"]
      route_error_heading = errors["heading_absolute"]
      final_position_error = float(torch.linalg.vector_norm(
        final_position[index] - endpoints[index]
      ))
      patch_origins = placement["patch_origins"]
      terrain_origins = placement["terrain_origins"]
      body_contacts = terrain_result["body_contacts"]
      action_distribution = terrain_result["action_acceleration"]
      slip_distribution = terrain_result["foot_slip_velocity"]
      pitch_distribution = terrain_result["base_pitch_absolute"]
      effort_distribution = terrain_result["actuator_effort_abs"]
      power_distribution = terrain_result["mechanical_power_abs"]
      tangent_slip_distribution = tangent_result["slip_velocity"]
      scenario_termination_counts = {
        name: int(value[index]) for name, value in termination_counts.items()
      }
      outputs.append({
        **scenario,
        "route_kind": route_kind,
        "route_length": float(route_lengths[index]),
        "route_length_definition": ROUTE_LENGTH_DEFINITION,
        "route_start_xy": [float(x) for x in route_start[index]],
        "route_endpoint_xy": [float(x) for x in endpoints[index]],
        "route_endpoint_heading": float(endpoint_headings[index]),
        "terrain_origin_xyz": [float(x) for x in terrain_origins[index]],
        "terrain_patch_origin_xyz": [float(x) for x in patch_origins[index]],
        "terrain_patch_size": list(PATCH_SIZE),
        "terrain_type_index": int(placement["types"][index]),
        "terrain_assignment_position_error": float(
          placement["terrain_assignment_position_error"][index]
        ),
        "route_placement_position_error": float(
          placement["route_placement_position_error"][index]
        ),
        "effective_terrain_parameters": effective_slope_parameters(
          scenario["slope_direction"], scenario["level"]
        ),
        "geometry": {
          "route_start_local": list(ROUTE_START_LOCAL),
          "centerline_bounds_local": footprint.centerline_bounds,
          "corridor_bounds_local": footprint.corridor_bounds,
          "scan_footprint_bounds_local": footprint.scan_footprint_bounds,
          "centerline_inside_patch": footprint.centreline_inside_patch,
          "corridor_inside_patch": footprint.corridor_inside_patch,
          "scan_footprint_inside_patch": footprint.scan_footprint_inside_patch,
          "centerline_boundary_margin": footprint.centerline_boundary_margin,
          "corridor_boundary_margin": footprint.corridor_boundary_margin,
          "scan_boundary_margin": footprint.scan_boundary_margin,
        },
        "completed": bool(success[index]),
        "path_endpoint_reached": bool(reached_endpoint[index]),
        "failed": bool(failed[index]),
        "finished": bool(success[index] | failed[index]),
        "catastrophic_termination": bool(
          terrain_result["catastrophic_termination"]["occurred"]
        ),
        "steps_sampled": int(sample_count[index]),
        "progress": float(final_progress[index]),
        "progress_ratio": float(final_progress[index] / route_lengths[index]),
        "cross_track_rms": route_error_cross["rms"],
        "cross_track_p95": route_error_cross["p95"],
        "cross_track_max": route_error_cross["max"],
        "cross_track_final": float(final_cross[index]),
        "heading_rms": route_error_heading["rms"],
        "heading_p95": route_error_heading["p95"],
        "heading_max": route_error_heading["max"],
        "heading_final": float(final_heading[index]),
        "final_position_xy": [float(x) for x in final_position[index]],
        "final_position_error": final_position_error,
        "commanded_velocity_mean": [float(x) for x in command_mean],
        "actual_velocity_mean": [float(x) for x in actual_mean],
        "response_gain": response_gain,
        "command_response_transients": transients.result(index),
        "controller_saturation_count": int(saturation_count[index]),
        "controller_saturation_fraction": float(
          saturation_count[index] / denom[index]
        ),
        "action_acceleration_mean": action_distribution["mean"],
        "action_acceleration_p95": action_distribution["p95"],
        "action_acceleration_max": action_distribution["max"],
        "slip_velocity_mean": slip_distribution["mean"],
        "slip_velocity_p95": slip_distribution["p95"],
        "slip_velocity_max": slip_distribution["max"],
        "slip_velocity_reason": slip_distribution.get("reason", "available"),
        "base_pitch_absolute_mean": pitch_distribution["mean"],
        "base_pitch_absolute_p95": pitch_distribution["p95"],
        "base_pitch_absolute_max": pitch_distribution["max"],
        "actuator_effort_abs_mean": effort_distribution["mean"],
        "actuator_effort_abs_p95": effort_distribution["p95"],
        "actuator_effort_abs_max": effort_distribution["max"],
        "mechanical_power_abs_mean": power_distribution["mean"],
        "mechanical_power_abs_p95": power_distribution["p95"],
        "mechanical_power_abs_max": power_distribution["max"],
        "terrain_tangent_stance_slip_mean": tangent_slip_distribution["mean"],
        "terrain_tangent_stance_slip_p95": tangent_slip_distribution["p95"],
        "terrain_tangent_stance_slip_max": tangent_slip_distribution["max"],
        "terrain_tangent_loaded_stance": tangent_result,
        "base_contact_count": body_contacts["base"]["non_terminating_count"],
        "base_contact_rate": body_contacts["base"]["non_terminating_rate"],
        "base_contact_reason": body_contacts["base"].get("reason", "available"),
        "upper_leg_contact_count": body_contacts["upper_leg"]["non_terminating_count"],
        "upper_leg_contact_rate": body_contacts["upper_leg"]["non_terminating_rate"],
        "upper_leg_contact_reason": body_contacts["upper_leg"].get(
          "reason", "available"
        ),
        "calf_contact_count": body_contacts["calf"]["non_terminating_count"],
        "calf_contact_rate": body_contacts["calf"]["non_terminating_rate"],
        "calf_contact_reason": body_contacts["calf"].get("reason", "available"),
        "terrain_rollout_metrics": terrain_result,
        "contact_termination_summary": _contact_termination_summary(
          scenario_termination_counts, body_contacts
        ),
        "termination_counts": scenario_termination_counts,
        "reset_count": int(reset_count[index]),
        "first_failure_reason": _final_failure_reason(
          completed=bool(success[index]),
          failed=bool(failed[index]),
          reason=first_reason[index],
        ),
        "initial_root_clearance": float(placement["clearance"][index]),
      })
    return {
      "route_kind": route_kind,
      "route_kind_invariants": {
        "checkpoint": checkpoint,
        "task_id": cfg.task_id,
        "seed": cfg.seed,
        "profile": profile,
        "num_envs": num_envs,
        "steps": cfg.steps,
        "settle_steps": cfg.settle_steps,
        "controller_limits": {
          "cross_track_gain": cfg.cross_track_gain,
          "heading_gain": cfg.heading_gain,
          "max_lateral_speed": cfg.max_lateral_speed,
          "max_yaw_rate": cfg.max_yaw_rate,
          "cross_track_tolerance": cfg.cross_track_tolerance,
          "heading_tolerance": cfg.heading_tolerance,
        },
      },
      "profile_settings": profile_settings,
      "num_envs": num_envs,
      "executed_control_steps": executed_steps,
      "ideal_minimum_horizon_steps": required_horizon,
      "terrain_assignment_position_error_max": placement[
        "terrain_assignment_position_error_max"
      ],
      "route_placement_position_error_max": placement[
        "route_placement_position_error_max"
      ],
      "completion_rate": sum(item["completed"] for item in outputs) / num_envs,
      "scenarios": outputs,
    }
  finally:
    env.close()


def evaluate(cfg: HighSlopeMatchedConfig) -> dict[str, Any]:
  preflight = _validate_config(cfg)
  scenarios = _scenarios(cfg)
  checkpoint = str(Path(cfg.checkpoint).expanduser().resolve())
  profiles: dict[str, Any] = {}
  for profile in cfg.profiles:
    route_results = {
      route_kind: _evaluate_route_kind(
        cfg, profile, route_kind, scenarios
      )
      for route_kind in ROUTE_KINDS
    }
    validate_matched_result_invariants(route_results)
    profile_settings = [
      route_results[kind]["profile_settings"] for kind in ROUTE_KINDS
    ]
    if any(item != profile_settings[0] for item in profile_settings[1:]):
      raise RuntimeError("profile settings differ across fresh route environments")
    profiles[profile] = {
      "matched_invariants": {
        "checkpoint": checkpoint,
        "task_id": cfg.task_id,
        "seed": cfg.seed,
        "profile": profile,
        "route_kinds": list(ROUTE_KINDS),
        "num_envs_per_route_kind": len(scenarios),
        "matched_slot_order": [item["matched_slot"] for item in scenarios],
        "slope_directions": list(cfg.slope_directions),
        "difficulties": [
          effective_slope_parameters(direction, level)
          for direction in cfg.slope_directions for level in cfg.levels
        ],
        "radii": list(cfg.radii),
        "speeds": list(cfg.speeds),
        "turn_signs": list(cfg.turn_signs),
        "steps": cfg.steps,
        "settle_steps": cfg.settle_steps,
        "control_dt": profile_settings[0]["control_dt"],
        "route_length_definition": ROUTE_LENGTH_DEFINITION,
        "fresh_environment_per_route_kind": True,
        "same_seed_environment_reconstruction": True,
        "controller_limits": route_results["straight"][
          "route_kind_invariants"
        ]["controller_limits"],
      },
      "profile_settings": profile_settings[0],
      "route_results": route_results,
    }
  result = {
    "schema_version": 1,
    "evaluation_suite": "high_slope_matched_straight_arc_s_curve",
    "git_head": _git_head(),
    "checkpoint": checkpoint,
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "config": asdict(cfg),
    "geometry_preflight": preflight,
    "metric_invariants": {
      "sample_denominator": ACTIVE_SAMPLE_DEFINITION,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
      "base_pitch_definition": BASE_PITCH_DEFINITION,
      "terrain_tangent_stance_slip_definition": (
        TERRAIN_TANGENT_STANCE_SLIP_DEFINITION
      ),
      "attempt_freeze": (
        "terminal step included; all samples from reset episodes and after "
        "fixed completion settle are excluded"
      ),
      "settle_lifecycle": (
        "endpoint must be reached within tolerance, then exactly settle_steps "
        "additional control steps must complete without reset"
      ),
    },
    "coverage": {
      "evaluation_only_18m_high_extreme_pyramid_slopes": True,
      "matched_straight_120deg_arc_two_60deg_s": True,
      "profiles_evaluated": list(cfg.profiles),
      "training_changed": False,
      "stairs": False,
      "continuous_approach_feature_exit": False,
    },
    "profiles": profiles,
  }
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(HighSlopeMatchedConfig)
  result = evaluate(cfg)
  result["provenance"] = formal_evaluation_provenance(
    cfg.checkpoint, __file__,
    (
      Path(__file__).resolve().parents[1] / "src/tasks/velocity/evaluation/proprio_acceptance.py",
      Path(__file__).resolve().parents[1] / "src/tasks/velocity/evaluation/terrain_rollout_metrics.py",
      Path(__file__).resolve().parents[1] / "src/tasks/velocity/evaluation/high_slope_matched.py",
      Path(__file__).resolve().parents[1] / "src/tasks/velocity/mdp/rewards.py",
    ),
  )
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps(result, indent=2, allow_nan=False))
  print(f"[INFO] Wrote high-slope matched evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
