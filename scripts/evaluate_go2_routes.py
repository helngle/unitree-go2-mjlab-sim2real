"""Evaluate a unified Go2 policy on parameterized straight route attempts.

The default patch suite retains the original terrain-origin diagnostic. The
continuous suite uses evaluation-only approach-flat -> feature -> exit-flat
profiles with exact route origins and does not modify the training task.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.evaluation.routes import (
  route_frame_errors,
  route_normal_velocity,
  straight_route_initial_positions,
  straight_line_controller,
  update_attempt_status,
  validate_initial_route_state,
  validate_route_parameters,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  ACTIVE_SAMPLE_DEFINITION,
  OnlineTerrainRolloutMetrics,
  action_acceleration,
  contact_any,
  foot_contact_any,
  foot_slip_velocity,
)


RANDOMIZATION_EVENTS = ("foot_friction", "encoder_bias", "base_com", "base_payload", "motor_strength")
PROFILE_NAMES = ("clean", "dynamics", "randomized")
CONTINUOUS_NCONMAX = 128


@dataclass(frozen=True)
class RouteConfig:
  checkpoints: tuple[str, ...]
  task_id: str = "Unitree-Go2-Rough-V7"
  mode: str = "line_follow"
  terrain_suite: str = "patch"
  transition_cases: tuple[str, ...] = (
    "stairs_up", "stairs_down", "slope_up", "slope_down",
  )
  terrain_types: tuple[str, ...] = (
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope",
    "hf_pyramid_slope_inv", "random_rough", "discrete_obstacles",
  )
  levels: tuple[int, ...] = (3, 5, 7)
  repeats: int = 2
  cross_track_offsets: tuple[float, ...] = (0.0,)
  yaw_offsets: tuple[float, ...] = (0.0,)
  route_heading: float = 0.0
  start_forward_offset: float = 0.0
  route_length: float | None = None
  target_speed: float = 0.4
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  max_lateral_speed: float = 0.3
  max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.35
  heading_tolerance: float = 0.35
  steps: int = 1000
  seed: int = 42
  profile: str = "clean"
  output_file: str | None = None


def _resolve_route_contract(cfg: RouteConfig) -> tuple[float, float, float]:
  """Resolve route length, heading, and start offset for the selected suite."""
  if cfg.terrain_suite == "patch":
    return (
      2.0 if cfg.route_length is None else cfg.route_length,
      cfg.route_heading,
      cfg.start_forward_offset,
    )
  if cfg.terrain_suite != "continuous":
    raise ValueError("terrain_suite must be 'patch' or 'continuous'")

  from src.tasks.velocity.evaluation.route_terrains import ROUTE_LENGTH

  if cfg.route_length is not None and not np.isclose(cfg.route_length, ROUTE_LENGTH):
    raise ValueError(
      f"continuous terrain requires route_length={ROUTE_LENGTH}, "
      f"got {cfg.route_length}"
    )
  if not np.isclose(cfg.route_heading, 0.0):
    raise ValueError("continuous terrain requires route_heading=0")
  if not np.isclose(cfg.start_forward_offset, 0.0):
    raise ValueError("continuous terrain requires start_forward_offset=0")
  return ROUTE_LENGTH, 0.0, 0.0


def _configure_continuous_sim_capacity(env_cfg: Any) -> dict[str, int | None]:
  """Raise contact capacity on the evaluation copy for custom route geometry."""
  original = env_cfg.sim.nconmax
  effective = (
    CONTINUOUS_NCONMAX
    if original is None
    else max(int(original), CONTINUOUS_NCONMAX)
  )
  env_cfg.sim.nconmax = effective
  return {"original": original, "effective": effective}


def _column_terrain_names(generator_cfg) -> list[str]:
  names = list(generator_cfg.sub_terrains)
  proportions = np.asarray([cfg.proportion for cfg in generator_cfg.sub_terrains.values()], dtype=np.float64)
  proportions /= proportions.sum()
  cumulative = np.cumsum(proportions)
  return [names[int(np.where(column / generator_cfg.num_cols + 0.001 < cumulative)[0][0])] for column in range(generator_cfg.num_cols)]


def _configure_profile(env_cfg: Any, profile: str) -> dict[str, Any]:
  if profile not in PROFILE_NAMES:
    raise ValueError(f"profile must be one of {PROFILE_NAMES}, got {profile!r}")
  if profile == "clean":
    env_cfg.observations["actor"].enable_corruption = False
    for name in RANDOMIZATION_EVENTS + ("push_robot",):
      env_cfg.events.pop(name, None)
  elif profile == "dynamics":
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    env_cfg.events.pop("encoder_bias", None)
  return {
    "actor_observation_corruption": env_cfg.observations["actor"].enable_corruption,
    "startup_randomization_events": [name for name in RANDOMIZATION_EVENTS if name in env_cfg.events],
    "push_enabled": "push_robot" in env_cfg.events,
  }


def _sensor_contact(
  env: ManagerBasedRlEnv, name: str, num_envs: int
) -> torch.Tensor | None:
  try:
    return contact_any(env.scene[name].data.found, num_envs)
  except KeyError:
    return None


def _catastrophic_termination(
  env: ManagerBasedRlEnv, num_envs: int
) -> torch.Tensor:
  result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  for name in env.termination_manager.active_terms:
    if not env.termination_manager.get_term_cfg(name).time_out:
      result |= env.termination_manager.get_term(name).bool()
  return result


def _make_scenarios(cfg: RouteConfig, terrain_columns: dict[str, list[int]]) -> list[dict[str, Any]]:
  scenarios = []
  for terrain_name in cfg.terrain_types:
    if not terrain_columns.get(terrain_name):
      raise ValueError(f"terrain type {terrain_name!r} has no generated columns")
  for terrain_name in cfg.terrain_types:
    for level in cfg.levels:
      for repeat in range(cfg.repeats):
        column = terrain_columns[terrain_name][repeat % len(terrain_columns[terrain_name])]
        for cross in cfg.cross_track_offsets:
          for yaw in cfg.yaw_offsets:
            if cfg.start_forward_offset < 0.0 and "stairs" in terrain_name:
              direction = "stairs_mixed_uncalibrated"
            elif terrain_name == "pyramid_stairs":
              direction = "stairs_down"
            elif terrain_name == "pyramid_stairs_inv":
              direction = "stairs_up"
            else:
              direction = "straight"
            scenarios.append({"terrain_type": terrain_name, "terrain_column": column, "level": level, "repeat": repeat, "cross_track_offset": cross, "yaw_offset": yaw, "direction_semantics": direction})
  return scenarios


def _make_continuous_scenarios(
  cfg: RouteConfig,
  terrain_columns: dict[str, list[int]],
) -> list[dict[str, Any]]:
  from src.tasks.velocity.evaluation.route_terrains import (
    TERRAIN_KIND_TO_KEY,
    continuous_route_difficulty_matrix,
    route_terrain_metadata,
  )

  unknown = sorted(set(cfg.transition_cases) - set(TERRAIN_KIND_TO_KEY))
  if unknown:
    raise ValueError(f"unknown continuous transition cases: {unknown}")
  if not cfg.transition_cases:
    raise ValueError("transition_cases must not be empty for continuous terrain")
  difficulty_matrix = continuous_route_difficulty_matrix(cfg.seed)
  scenarios: list[dict[str, Any]] = []
  for transition_case in cfg.transition_cases:
    terrain_name = TERRAIN_KIND_TO_KEY[transition_case]
    columns = terrain_columns.get(terrain_name, [])
    if len(columns) != 1:
      raise ValueError(
        f"continuous transition {transition_case!r} requires exactly one "
        f"{terrain_name!r} column, got {columns}"
      )
    column = columns[0]
    for requested_level in cfg.levels:
      level = min(max(requested_level, 0), difficulty_matrix.shape[0] - 1)
      difficulty = float(difficulty_matrix[level, column])
      metadata = route_terrain_metadata(transition_case, difficulty)
      for repeat in range(cfg.repeats):
        for cross in cfg.cross_track_offsets:
          for yaw in cfg.yaw_offsets:
            scenarios.append(
              {
                "terrain_type": terrain_name,
                "terrain_column": column,
                "level": level,
                "level_requested": requested_level,
                "difficulty": difficulty,
                "repeat": repeat,
                "cross_track_offset": cross,
                "yaw_offset": yaw,
                "transition_case": transition_case,
                "feature": metadata.family,
                "direction": metadata.direction,
                "direction_semantics": f"{metadata.family}_{metadata.direction}",
                "entry_surface_z_local": metadata.entry_surface_z,
                "exit_surface_z_local": metadata.exit_surface_z,
                "net_height": metadata.exit_surface_z - metadata.entry_surface_z,
                "step_height": metadata.step_height,
                "slope": metadata.slope,
              }
            )
  return scenarios


def _set_terrain_and_route(env: ManagerBasedRlEnv, scenarios: list[dict[str, Any]], route_heading: float, start_forward_offset: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  old_origins = terrain.env_origins.clone()
  old_root_pos = robot.data.root_link_pos_w.clone()
  device = env.device
  levels = torch.tensor([item["level"] for item in scenarios], device=device, dtype=torch.long).clamp(0, terrain.max_terrain_level - 1)
  columns = torch.tensor([item["terrain_column"] for item in scenarios], device=device, dtype=torch.long)
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = columns
  terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :3] += terrain.env_origins - old_origins
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  placement_error = torch.max(torch.abs((robot.data.root_link_pos_w - terrain.env_origins) - (old_root_pos - old_origins)))
  if placement_error > 1e-4:
    raise RuntimeError(f"terrain assignment position error unexpectedly large: {placement_error.item():.6f}")
  root_pose = robot.data.root_link_pose_w.clone()
  nominal_root_xy = root_pose[:, :2].clone()
  route_headings = torch.full((len(scenarios),), route_heading, device=device)
  cross_offsets = torch.tensor(
    [item["cross_track_offset"] for item in scenarios], device=device
  )
  route_start, robot_start = straight_route_initial_positions(
    nominal_root_xy,
    route_headings,
    start_forward_offset,
    cross_offsets,
  )
  root_pose[:, :2] = robot_start
  old_heading = robot.data.heading_w.clone()
  desired_heading = torch.tensor([route_heading + item["yaw_offset"] for item in scenarios], device=device)
  delta_yaw = desired_heading - old_heading
  root_pose[:, 3:7] = quat_mul(quat_from_euler_xyz(torch.zeros_like(delta_yaw), torch.zeros_like(delta_yaw), delta_yaw), root_pose[:, 3:7])
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  yaw_offsets = torch.tensor(
    [item["yaw_offset"] for item in scenarios], device=device
  )
  validate_initial_route_state(
    robot.data.root_link_pos_w[:, :2],
    robot.data.heading_w,
    route_start,
    route_headings,
    cross_offsets,
    yaw_offsets,
  )
  return route_start, route_headings, terrain.env_origins.clone(), float(placement_error)


def _set_continuous_terrain_and_route(
  env: ManagerBasedRlEnv,
  scenarios: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
  """Place roots at exact custom-profile origins after wrapper reset."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  old_origins = terrain.env_origins.clone()
  old_root_pos = robot.data.root_link_pos_w.clone()
  base_clearance = old_root_pos[:, 2] - old_origins[:, 2]
  device = env.device
  levels = torch.tensor(
    [item["level"] for item in scenarios], device=device, dtype=torch.long
  )
  columns = torch.tensor(
    [item["terrain_column"] for item in scenarios],
    device=device,
    dtype=torch.long,
  )
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = columns
  terrain.env_origins[:] = terrain.terrain_origins[levels, columns]

  route_start = terrain.env_origins[:, :2].clone()
  route_headings = torch.zeros(len(scenarios), device=device)
  cross_offsets = torch.tensor(
    [item["cross_track_offset"] for item in scenarios], device=device
  )
  _, robot_start = straight_route_initial_positions(
    route_start, route_headings, 0.0, cross_offsets
  )
  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :2] = robot_start
  root_pose[:, 2] = terrain.env_origins[:, 2] + base_clearance
  old_heading = robot.data.heading_w.clone()
  yaw_offsets = torch.tensor(
    [item["yaw_offset"] for item in scenarios], device=device
  )
  root_pose[:, 3:7] = quat_mul(
    quat_from_euler_xyz(
      torch.zeros_like(yaw_offsets),
      torch.zeros_like(yaw_offsets),
      yaw_offsets - old_heading,
    ),
    root_pose[:, 3:7],
  )
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()

  expected_position = torch.cat(
    (robot_start, (terrain.env_origins[:, 2] + base_clearance).unsqueeze(-1)),
    dim=-1,
  )
  placement_error = torch.max(
    torch.abs(robot.data.root_link_pos_w - expected_position)
  )
  if placement_error > 1.0e-4:
    raise RuntimeError(
      "continuous terrain placement did not preserve the requested root pose: "
      f"{placement_error.item():.6f}"
    )
  validate_initial_route_state(
    robot.data.root_link_pos_w[:, :2],
    robot.data.heading_w,
    route_start,
    route_headings,
    cross_offsets,
    yaw_offsets,
  )
  realized_clearance = robot.data.root_link_pos_w[:, 2] - terrain.env_origins[:, 2]
  return (
    route_start,
    route_headings,
    terrain.env_origins.clone(),
    float(placement_error),
    realized_clearance,
  )


def _evaluate_checkpoint(checkpoint: Path, cfg: RouteConfig) -> dict[str, Any]:
  if cfg.mode not in {"open_loop", "line_follow"}:
    raise ValueError("mode must be 'open_loop' or 'line_follow'")
  if cfg.repeats <= 0 or cfg.steps <= 0:
    raise ValueError("repeats and steps must be positive")
  route_length, route_heading, start_forward_offset = _resolve_route_contract(cfg)
  validate_route_parameters(route_length=route_length, control_dt=0.02, cross_track_tolerance=cfg.cross_track_tolerance, heading_tolerance=cfg.heading_tolerance, target_speed=cfg.target_speed)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  terrain_cfg = env_cfg.scene.terrain
  assert terrain_cfg is not None and terrain_cfg.terrain_generator is not None
  if cfg.terrain_suite == "continuous":
    from src.tasks.velocity.evaluation.route_terrains import (
      make_continuous_route_terrain_generator,
    )

    terrain_cfg.terrain_generator = make_continuous_route_terrain_generator(
      seed=cfg.seed
    )
  column_names = _column_terrain_names(terrain_cfg.terrain_generator)
  requested_terrains = (
    set(terrain_cfg.terrain_generator.sub_terrains)
    if cfg.terrain_suite == "continuous"
    else set(cfg.terrain_types)
  )
  terrain_columns = {name: [i for i, column_name in enumerate(column_names) if column_name == name] for name in requested_terrains}
  scenarios = (
    _make_continuous_scenarios(cfg, terrain_columns)
    if cfg.terrain_suite == "continuous"
    else _make_scenarios(cfg, terrain_columns)
  )
  num_envs = len(scenarios)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = _configure_profile(env_cfg, cfg.profile)
  if cfg.terrain_suite == "continuous":
    capacity_override = _configure_continuous_sim_capacity(env_cfg)
    profile_settings["continuous_sim_nconmax_override"] = capacity_override
    profile_settings["sim_nconmax"] = env_cfg.sim.nconmax
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 task command must implement UniformVelocityCommandCfg")
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (cfg.target_speed, cfg.target_speed)
  command_cfg.ranges.lin_vel_y = (-cfg.max_lateral_speed, cfg.max_lateral_speed)
  command_cfg.ranges.ang_vel_z = (-cfg.max_yaw_rate, cfg.max_yaw_rate)
  command_cfg.ranges.heading = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  terrain = env.scene.terrain
  assert terrain is not None
  robot = env.scene["robot"]
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  # RslRlVecEnvWrapper construction resets the environment. Place routes only
  # after wrapper/runner initialization so the measured initial pose is the
  # pose used by the first rollout observation.
  if cfg.terrain_suite == "continuous":
    (
      route_start,
      route_headings,
      terrain_origins,
      placement_error,
      initial_root_clearance,
    ) = _set_continuous_terrain_and_route(env, scenarios)
  else:
    route_start, route_headings, terrain_origins, placement_error = (
      _set_terrain_and_route(
        env, scenarios, route_heading, start_forward_offset
      )
    )
    initial_root_clearance = robot.data.root_link_pos_w[:, 2] - terrain_origins[:, 2]
  command_term = env.command_manager.get_term("twist")
  if not isinstance(command_term, UniformVelocityCommand):
    raise TypeError("twist command term is not UniformVelocityCommand-compatible")
  device = env.device
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  completed = torch.zeros_like(active)
  failed = torch.zeros_like(active)
  first_reason = [None] * num_envs
  sample_count = torch.zeros(num_envs, device=device)
  cross_sq_sum = torch.zeros(num_envs, device=device)
  heading_sq_sum = torch.zeros(num_envs, device=device)
  position_sq_sum = torch.zeros(num_envs, device=device)
  cross_max = torch.zeros(num_envs, device=device)
  heading_max = torch.zeros(num_envs, device=device)
  cross_abs_samples = torch.zeros(
    (num_envs, cfg.steps), device=device, dtype=cross_max.dtype
  )
  heading_abs_samples = torch.zeros_like(cross_abs_samples)
  path_sample_valid = torch.zeros(
    (num_envs, cfg.steps), device=device, dtype=torch.bool
  )
  final_progress = torch.zeros(num_envs, device=device)
  final_cross = torch.zeros(num_envs, device=device)
  final_heading = torch.zeros(num_envs, device=device)
  final_position = robot.data.root_link_pos_w[:, :2].clone()
  initial_position = final_position.clone()
  initial_heading = robot.data.heading_w.clone()
  completion_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  reset_count = torch.zeros(num_envs, dtype=torch.long, device=device)
  actual_lin_sum = torch.zeros(num_envs, 2, device=device)
  command_lin_sum = torch.zeros_like(actual_lin_sum)
  actual_yaw_sum = torch.zeros(num_envs, device=device)
  command_yaw_sum = torch.zeros_like(actual_yaw_sum)
  cross_axis_sum = torch.zeros(num_envs, device=device)
  slip_sum = torch.zeros(num_envs, device=device)
  action_acc_sum = torch.zeros(num_envs, device=device)
  rollout_metrics = OnlineTerrainRolloutMetrics(
    num_envs,
    cfg.steps,
    device=device,
    dtype=robot.data.root_link_pos_w.dtype,
  )
  termination_counts = {name: torch.zeros(num_envs, device=device) for name in env.termination_manager.active_terms}
  try:
    feet_sensor = env.scene["feet_ground_contact"]
  except KeyError:
    feet_sensor = None
  foot_site_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"))
  observation = wrapped_env.get_observations()
  command_values = torch.zeros(num_envs, 3, device=device)
  for step_index in range(cfg.steps):
    if not active.any():
      break
    pre_pos = robot.data.root_link_pos_w[:, :2].clone()
    pre_heading = robot.data.heading_w.clone()
    state = route_frame_errors(pre_pos, pre_heading, route_start, route_headings)
    if cfg.mode == "open_loop":
      command_values[:] = 0.0
      command_values[:, 0] = cfg.target_speed
    else:
      command_values = straight_line_controller(pre_pos, pre_heading, route_start, route_headings, target_speed=cfg.target_speed, cross_track_gain=cfg.cross_track_gain, heading_gain=cfg.heading_gain, max_lateral_speed=cfg.max_lateral_speed, max_yaw_rate=cfg.max_yaw_rate, route_length=route_length)
    command_values = torch.where(active.unsqueeze(-1), command_values, 0.0)
    command_term.vel_command_b[:] = command_values
    # The command is part of the actor observation. Refresh it before policy
    # inference so line-follow uses this step's controller output, not the
    # command retained in the observation returned by the previous step.
    observation = wrapped_env.get_observations()
    with torch.inference_mode():
      action = policy(observation)
    step_observation, reward, dones, extras = wrapped_env.step(action)
    del step_observation, reward, extras
    command_term.vel_command_b[:] = command_values
    observation = wrapped_env.get_observations()
    failure_terms = [name for name in env.termination_manager.active_terms if not env.termination_manager.get_term_cfg(name).time_out]
    for name in failure_terms:
      term = env.termination_manager.get_term(name)
      termination_counts[name] += term.float() * active.float()
    for name in env.termination_manager.active_terms:
      if env.termination_manager.get_term_cfg(name).time_out:
        termination_counts[name] += env.termination_manager.get_term(name).float() * active.float()
    # For a reset env the simulator has already written a new pose. Freeze the
    # previous state; non-reset envs use the post-step pose for completion.
    reset_mask = dones.bool()
    reset_count += (reset_mask & active).long()
    post_pos = robot.data.root_link_pos_w[:, :2]
    post_heading = robot.data.heading_w
    post_state = route_frame_errors(post_pos, post_heading, route_start, route_headings)
    candidate_progress = torch.where(reset_mask, state.progress, post_state.progress)
    candidate_cross = torch.where(reset_mask, state.cross_track, post_state.cross_track)
    candidate_heading = torch.where(reset_mask, state.heading_error, post_state.heading_error)
    lifecycle = update_attempt_status(active, candidate_progress, candidate_cross, candidate_heading, reset_mask, route_length=route_length, cross_track_tolerance=cfg.cross_track_tolerance, heading_tolerance=cfg.heading_tolerance)
    sample = lifecycle.sample_mask
    sample_count += sample.float()
    cross_sq_sum += candidate_cross.square() * sample.float()
    heading_sq_sum += candidate_heading.square() * sample.float()
    cross_abs_samples[:, step_index] = candidate_cross.abs()
    heading_abs_samples[:, step_index] = candidate_heading.abs()
    path_sample_valid[:, step_index] = sample
    endpoint = route_start + torch.stack((route_length * torch.cos(route_headings), route_length * torch.sin(route_headings)), dim=-1)
    candidate_position = torch.where(reset_mask.unsqueeze(-1), pre_pos, post_pos)
    position_error = torch.norm(candidate_position - endpoint, dim=-1)
    position_sq_sum += position_error.square() * sample.float()
    cross_max = torch.maximum(cross_max, candidate_cross.abs() * sample.float())
    heading_max = torch.maximum(heading_max, candidate_heading.abs() * sample.float())
    final_progress = torch.where(sample, candidate_progress, final_progress)
    final_cross = torch.where(sample, candidate_cross, final_cross)
    final_heading = torch.where(sample, candidate_heading, final_heading)
    final_position = torch.where(sample.unsqueeze(-1), candidate_position, final_position)
    actual_lin = robot.data.root_link_lin_vel_b[:, :2]
    actual_lin_w = robot.data.root_link_lin_vel_w[:, :2]
    actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
    actual_lin = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(actual_lin), actual_lin)
    actual_lin_w = torch.where(
      reset_mask.unsqueeze(-1), torch.zeros_like(actual_lin_w), actual_lin_w
    )
    actual_yaw = torch.where(reset_mask, torch.zeros_like(actual_yaw), actual_yaw)
    actual_lin_sum += actual_lin * sample.unsqueeze(-1).float()
    command_lin_sum += command_values[:, :2] * sample.unsqueeze(-1).float()
    actual_yaw_sum += actual_yaw * sample.float()
    command_yaw_sum += command_values[:, 2] * sample.float()
    cross_axis_sum += (
      torch.abs(route_normal_velocity(actual_lin_w, route_headings))
      * sample.float()
    )
    if feet_sensor is None:
      slip = None
    else:
      foot_contact = foot_contact_any(
        feet_sensor.data.found, num_envs, len(foot_site_ids)
      )
      slip = foot_slip_velocity(
        robot.data.site_lin_vel_w[:, foot_site_ids, :2], foot_contact
      )
      slip_sum += slip * sample.float()
    action_acc = action_acceleration(
      env.action_manager.action,
      env.action_manager.prev_action,
      env.action_manager.prev_prev_action,
    )
    action_acc_sum += action_acc * sample.float()
    rollout_metrics.update(
      sample_mask=sample,
      action_acceleration=action_acc,
      foot_slip_velocity=slip,
      body_contacts={
        "base": _sensor_contact(env, "base_ground_contact", num_envs),
        "upper_leg": _sensor_contact(
          env, "upper_leg_ground_contact", num_envs
        ),
        "calf": _sensor_contact(env, "calf_ground_contact", num_envs),
      },
      catastrophic_termination=_catastrophic_termination(env, num_envs),
    )
    for index in torch.where(lifecycle.completed_now | lifecycle.failed_now)[0].tolist():
      if first_reason[index] is None:
        if bool(lifecycle.completed_now[index]):
          completion_steps[index] = step_index + 1
        else:
          names = [name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[index])]
          first_reason[index] = names[0] if names else "reset"
    completed |= lifecycle.completed_now
    failed |= lifecycle.failed_now
    active = lifecycle.active
  for index in torch.where(active)[0].tolist():
    first_reason[index] = "step_limit"
  failed |= active
  active[:] = False
  env.close()
  denom = sample_count.clamp_min(1.0)
  endpoint = route_start + torch.stack((route_length * torch.cos(route_headings), route_length * torch.sin(route_headings)), dim=-1)
  output_scenarios = []
  for index, scenario in enumerate(scenarios):
    final_position_error = float(torch.norm(final_position[index] - endpoint[index]))
    path_mask = path_sample_valid[index]
    cross_distribution = cross_abs_samples[index][path_mask].to(torch.float64)
    heading_distribution = heading_abs_samples[index][path_mask].to(torch.float64)
    if cross_distribution.numel() == 0 or heading_distribution.numel() == 0:
      raise RuntimeError("finished straight attempt has no retained path samples")
    terrain_metrics = rollout_metrics.result(index)
    if terrain_metrics["active_control_step_samples"] != int(sample_count[index]):
      raise RuntimeError("terrain metric sample mask diverged from route lifecycle")
    action_distribution = terrain_metrics["action_acceleration"]
    slip_distribution = terrain_metrics["foot_slip_velocity"]
    body_contacts = terrain_metrics["body_contacts"]
    scenario_output = {
      **scenario,
      "completed": bool(completed[index]),
      "failed": bool(failed[index]),
      "first_failure_reason": first_reason[index],
      "reset_count": int(reset_count[index]),
      "steps_sampled": int(sample_count[index]),
      "steps_to_completion": (
        int(completion_steps[index]) if completion_steps[index] >= 0 else None
      ),
      "forward_progress": float(final_progress[index]),
      "progress_ratio": float(final_progress[index] / route_length),
      "lateral_rms": float(torch.sqrt(cross_sq_sum[index] / denom[index])),
      "lateral_p95": float(torch.quantile(cross_distribution, 0.95)),
      "lateral_max": float(cross_max[index]),
      "lateral_final": float(final_cross[index]),
      "heading_rms": float(torch.sqrt(heading_sq_sum[index] / denom[index])),
      "heading_p95": float(torch.quantile(heading_distribution, 0.95)),
      "heading_max": float(heading_max[index]),
      "heading_final": float(final_heading[index]),
      "final_heading_error": float(final_heading[index]),
      "position_error_rms": float(torch.sqrt(position_sq_sum[index] / denom[index])),
      "position_error_final": final_position_error,
      "final_position_error": final_position_error,
      "actual_velocity_xy_mean": [
        float(value) for value in (actual_lin_sum[index] / denom[index])
      ],
      "commanded_velocity_xy_mean": [
        float(value) for value in (command_lin_sum[index] / denom[index])
      ],
      "actual_yaw_rate_mean": float(actual_yaw_sum[index] / denom[index]),
      "commanded_yaw_rate_mean": float(command_yaw_sum[index] / denom[index]),
      "cross_axis_velocity_mean": float(cross_axis_sum[index] / denom[index]),
      "slip_velocity_mean": slip_distribution["mean"],
      "slip_velocity_p95": slip_distribution["p95"],
      "slip_velocity_max": slip_distribution["max"],
      "action_acceleration_mean": action_distribution["mean"],
      "action_acceleration_p95": action_distribution["p95"],
      "action_acceleration_max": action_distribution["max"],
      "base_contact_count": body_contacts["base"]["non_terminating_count"],
      "base_contact_rate": body_contacts["base"]["non_terminating_rate"],
      "upper_leg_contact_count": body_contacts["upper_leg"][
        "non_terminating_count"
      ],
      "upper_leg_contact_rate": body_contacts["upper_leg"][
        "non_terminating_rate"
      ],
      "calf_contact_count": body_contacts["calf"]["non_terminating_count"],
      "calf_contact_rate": body_contacts["calf"]["non_terminating_rate"],
      "terrain_rollout_metrics": terrain_metrics,
      "route_start_xy": [float(value) for value in route_start[index]],
      "route_endpoint_xy": [float(value) for value in endpoint[index]],
      "initial_position_xy": [float(value) for value in initial_position[index]],
      "initial_heading": float(initial_heading[index]),
      "initial_root_clearance": float(initial_root_clearance[index]),
      "final_position_xy": [float(value) for value in final_position[index]],
      "terrain_origin_xyz": [float(value) for value in terrain_origins[index]],
      "termination_counts": {
        name: float(values[index]) for name, values in termination_counts.items()
      },
    }
    if cfg.terrain_suite == "continuous":
      from src.tasks.velocity.evaluation.route_terrains import (
        FEATURE_END_X,
        FEATURE_START_X,
        GENERATOR_NUM_ROWS,
        PATCH_SIZE,
        ROUTE_END_X,
        ROUTE_START_X,
        TERRAIN_SCAN_HALF_X,
      )

      feature_start_progress = FEATURE_START_X - ROUTE_START_X
      feature_end_progress = FEATURE_END_X - ROUTE_START_X
      route_y = PATCH_SIZE[1] / 2.0
      net_height = float(scenario["net_height"])
      max_delta = (
        abs(float(scenario["step_height"]))
        if scenario["feature"] == "stairs"
        else abs(float(scenario["slope"])) * 0.05
      )
      scenario_output.update(
        {
          "route_start_local": [ROUTE_START_X, route_y],
          "feature_entry_local": [FEATURE_START_X, route_y],
          "feature_exit_local": [FEATURE_END_X, route_y],
          "route_endpoint_local": [ROUTE_END_X, route_y],
          "feature_start_progress": feature_start_progress,
          "feature_end_progress": feature_end_progress,
          "start_surface_z": float(terrain_origins[index, 2]),
          "endpoint_surface_z": float(terrain_origins[index, 2]) + net_height,
          "difficulty_interval": [
            scenario["level"] / GENERATOR_NUM_ROWS,
            (scenario["level"] + 1) / GENERATOR_NUM_ROWS,
          ],
          "entry_junction_error": 0.0,
          "exit_junction_error": 0.0,
          "max_adjacent_height_delta": max_delta,
          "route_inside_patch": True,
          "min_boundary_margin": min(
            ROUTE_START_X,
            PATCH_SIZE[0] - ROUTE_END_X,
            route_y,
            PATCH_SIZE[1] - route_y,
          ),
          "boundary_margin_x": min(
            ROUTE_START_X, PATCH_SIZE[0] - ROUTE_END_X
          ),
          "boundary_margin_y": min(route_y, PATCH_SIZE[1] - route_y),
          "terrain_scan_half_extent_x": TERRAIN_SCAN_HALF_X,
          "terrain_scan_footprint_inside_patch": (
            ROUTE_START_X - TERRAIN_SCAN_HALF_X >= 0.0
            and ROUTE_END_X + TERRAIN_SCAN_HALF_X <= PATCH_SIZE[0]
          ),
          "terrain_scan_boundary_clearance_x": min(
            ROUTE_START_X - TERRAIN_SCAN_HALF_X,
            PATCH_SIZE[0] - ROUTE_END_X - TERRAIN_SCAN_HALF_X,
          ),
          "start_flat": True,
          "endpoint_flat": True,
        }
      )
    output_scenarios.append(scenario_output)
  completion_rate = sum(item["completed"] for item in output_scenarios) / max(len(output_scenarios), 1)
  if cfg.terrain_suite == "continuous":
    from src.tasks.velocity.evaluation.route_terrains import (
      FEATURE_END_X,
      FEATURE_START_X,
      PATCH_SIZE,
      ROUTE_END_X,
      ROUTE_START_X,
      TERRAIN_SCAN_HALF_X,
    )

    coverage = {
      "independent_generated_patch_straight_routes": False,
      "continuous_intra_patch_transitions": True,
      "continuous_inter_patch_transitions": False,
      "continuous_transition_status": "implemented",
      "transition_cases": list(cfg.transition_cases),
    }
    limitations = [
      "Continuous routes cover approach-flat, one feature, and exit-flat "
      "inside one custom evaluation patch; they do not cross patch boundaries."
    ]
    route_definition = {
      "patch_size": list(PATCH_SIZE),
      "start_local": [ROUTE_START_X, PATCH_SIZE[1] / 2.0],
      "feature_x": [FEATURE_START_X, FEATURE_END_X],
      "endpoint_local": [ROUTE_END_X, PATCH_SIZE[1] / 2.0],
      "route_length": route_length,
      "heading": route_heading,
      "start_forward_offset": start_forward_offset,
      "terrain_scan_half_extent_x": TERRAIN_SCAN_HALF_X,
      "terrain_scan_start_x": [
        ROUTE_START_X - TERRAIN_SCAN_HALF_X,
        ROUTE_START_X + TERRAIN_SCAN_HALF_X,
      ],
      "terrain_scan_endpoint_x": [
        ROUTE_END_X - TERRAIN_SCAN_HALF_X,
        ROUTE_END_X + TERRAIN_SCAN_HALF_X,
      ],
      "cross_track_offsets": list(cfg.cross_track_offsets),
      "yaw_offsets": list(cfg.yaw_offsets),
    }
  else:
    coverage = {
      "independent_generated_patch_straight_routes": True,
      "continuous_inter_patch_transitions": False,
      "continuous_transition_status": "unsupported_deferred",
    }
    limitations = [
      "This evaluator assigns one generated terrain patch per route attempt; "
      "it does not implement or claim flat-to-stairs, stairs-to-flat, "
      "slope-to-flat, or other inter-patch transitions."
    ]
    route_definition = {
      "route_heading": route_heading,
      "start_forward_offset": start_forward_offset,
      "route_length": route_length,
      "cross_track_offsets": cfg.cross_track_offsets,
      "yaw_offsets": cfg.yaw_offsets,
      "terrain_origin_semantics": {
        "pyramid_stairs": "origin central high platform; outward route is descending",
        "pyramid_stairs_inv": "origin central low platform; outward route is ascending",
      },
    }
  return {
    "schema_version": 2,
    "checkpoint": str(checkpoint),
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "mode": cfg.mode,
    "profile": cfg.profile,
    "terrain_suite": cfg.terrain_suite,
    "profile_settings": profile_settings,
    "num_envs": num_envs,
    "steps": cfg.steps,
    "terrain_assignment_position_error_max": placement_error,
    "metric_invariants": {
      "sample_denominator": ACTIVE_SAMPLE_DEFINITION,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
      "body_contact_rate_denominator": (
        "active_control_step_samples for the same environment and original attempt"
      ),
      "attempt_freeze": (
        "terminal control step included; all later/reset-episode steps excluded"
      ),
    },
    "coverage": coverage,
    "limitations": limitations,
    "route_definition": route_definition,
    "completion_rate": completion_rate,
    "scenarios": output_scenarios,
    "termination_totals": {
      name: float(values.sum()) for name, values in termination_counts.items()
    },
  }


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(RouteConfig)
  results = [_evaluate_checkpoint(Path(path).expanduser().resolve(), cfg) for path in cfg.checkpoints]
  config_output = asdict(cfg)
  config_output["route_length"] = _resolve_route_contract(cfg)[0]
  output = {"config": config_output, "results": results}
  output_path = Path(cfg.output_file) if cfg.output_file else Path("go2_route_evaluation.json")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
  print(json.dumps(output, indent=2, allow_nan=False))
  print(f"[INFO] Wrote route evaluation to {output_path}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
