"""Evaluate V7 on parameterized flat-ground arcs and S curves.

This is an evaluation-only harness.  It creates a 16 m flat patch so the
largest configured curve and the actor height-scan footprint remain inside a
single terrain.  It never changes the training task configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

# Support direct execution from an independent Git worktree whose editable
# install may still point at the integration worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.evaluation.curved_routes import (
  ArcRoute,
  SRoute,
  arc_command_controller,
  arc_route_errors,
  make_arc_route,
  make_command_tape_schedule,
  make_s_route,
  s_command_controller,
  s_route_errors,
)
from src.tasks.velocity.evaluation.routes import route_normal_velocity, update_attempt_status
from src.tasks.velocity.evaluation.transient_metrics import (
  OnlineCommandTransientMetrics,
  TransientMetricConfig,
)


PATCH_SIZE = (16.0, 16.0)
ROUTE_START_LOCAL = (2.0, 8.0)
RANDOMIZATION_EVENTS = ("foot_friction", "encoder_bias", "base_com", "base_payload", "motor_strength")
PROFILE_NAMES = ("clean", "dynamics", "randomized")


def _configure_profile(env_cfg: Any, profile: str) -> dict[str, Any]:
  if profile not in PROFILE_NAMES:
    raise ValueError(f"profile must be one of {PROFILE_NAMES}")
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


def _configure_episode_length(
  env_cfg: Any, steps: int, safety_steps: int = 10
) -> dict[str, float]:
  """Extend only the evaluation timeout to cover the requested rollout."""
  if steps <= 0 or safety_steps < 0:
    raise ValueError("steps must be positive and safety_steps nonnegative")
  if hasattr(env_cfg.sim, "dt"):
    physics_dt = env_cfg.sim.dt
  else:
    physics_dt = env_cfg.sim.mujoco.timestep
  control_dt = float(physics_dt * env_cfg.decimation)
  original = float(env_cfg.episode_length_s)
  env_cfg.episode_length_s = max(
    original, (steps + safety_steps) * control_dt
  )
  return {
    "original_episode_length_s": original,
    "effective_episode_length_s": float(env_cfg.episode_length_s),
    "control_dt": control_dt,
  }


class CurvedFlatTerrainCfg(SubTerrainCfg):
  size: tuple[float, float] = PATCH_SIZE

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    plane = make_plane(spec.body("terrain"), self.size, 0.0, center_zero=False)[0]
    return TerrainOutput(
      origin=np.asarray([*ROUTE_START_LOCAL, 0.0], dtype=np.float64),
      geometries=[TerrainGeometry(geom=plane)],
    )


def make_curved_flat_generator(seed: int) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    seed=seed,
    size=PATCH_SIZE,
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    add_lights=True,
    sub_terrains={"evaluation_flat": CurvedFlatTerrainCfg(proportion=1.0)},
  )


@dataclass(frozen=True)
class CurvedRouteConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  route_kind: str = "arc"
  mode: str = "closed_loop"
  radii: tuple[float, ...] = (1.5, 2.5, 4.0)
  speeds: tuple[float, ...] = (0.3, 0.5, 0.6)
  turn_signs: tuple[int, ...] = (1, -1)
  cross_track_offsets: tuple[float, ...] = (0.0,)
  yaw_offsets: tuple[float, ...] = (0.0,)
  repeats: int = 1
  settle_steps: int = 10
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  max_lateral_speed: float = 0.3
  max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.30
  heading_tolerance: float = math.radians(20.0)
  steps: int = 1200
  seed: int = 42
  profile: str = "clean"
  output_file: str = "go2_curved_route_evaluation.json"


def _validate_config(cfg: CurvedRouteConfig) -> None:
  if cfg.route_kind not in {"arc", "s_curve"}:
    raise ValueError("route_kind must be 'arc' or 's_curve'")
  if cfg.mode not in {"command_tape", "closed_loop"}:
    raise ValueError("mode must be 'command_tape' or 'closed_loop'")
  if cfg.repeats <= 0 or cfg.steps <= 0:
    raise ValueError("repeats and steps must be positive")
  if not cfg.radii or any(not math.isfinite(r) or r <= 0 for r in cfg.radii):
    raise ValueError("radii must be finite and positive")
  if not cfg.speeds or any(not math.isfinite(v) or v <= 0 for v in cfg.speeds):
    raise ValueError("speeds must be finite and positive")
  if any(sign not in (-1, 1) for sign in cfg.turn_signs):
    raise ValueError("turn_signs must contain only -1 or +1")
  if cfg.mode == "command_tape" and (any(abs(x) > 1e-8 for x in cfg.cross_track_offsets) or any(abs(x) > 1e-8 for x in cfg.yaw_offsets)):
    raise ValueError("command_tape isolates locomotion and requires zero initial offsets")
  if cfg.settle_steps < 0:
    raise ValueError("settle_steps must be nonnegative")
  if cfg.mode == "command_tape":
    required_steps = max(
      make_command_tape_schedule(
        cfg.route_kind, radius, speed, sign, 0.02, cfg.settle_steps
      ).total_steps
      for radius in cfg.radii
      for speed in cfg.speeds
      for sign in cfg.turn_signs
    )
    if cfg.steps < required_steps:
      raise ValueError(
        f"command_tape requires steps >= {required_steps}, got {cfg.steps}"
      )


def _scenarios(cfg: CurvedRouteConfig) -> list[dict[str, Any]]:
  return [
    {"radius": radius, "speed": speed, "turn_sign": sign, "cross_track_offset": cross, "yaw_offset": yaw, "repeat": repeat}
    for radius in cfg.radii
    for speed in cfg.speeds
    for sign in cfg.turn_signs
    for cross in cfg.cross_track_offsets
    for yaw in cfg.yaw_offsets
    for repeat in range(cfg.repeats)
  ]


def _scenario_group_indices(
  scenarios: list[dict[str, Any]],
) -> dict[tuple[float, float, int], list[int]]:
  """Group scenario indices without changing or duplicating their order."""
  groups: dict[tuple[float, float, int], list[int]] = {}
  for index, scenario in enumerate(scenarios):
    key = (scenario["radius"], scenario["speed"], scenario["turn_sign"])
    groups.setdefault(key, []).append(index)
  covered = [index for indices in groups.values() for index in indices]
  if sorted(covered) != list(range(len(scenarios))) or len(set(covered)) != len(covered):
    raise RuntimeError("scenario groups must cover every index exactly once")
  return groups


def _route_errors(route: ArcRoute | SRoute, pos: torch.Tensor, heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return arc_route_errors(route, pos, heading) if isinstance(route, ArcRoute) else s_route_errors(route, pos, heading)


def _place_routes(env: ManagerBasedRlEnv, cross_offsets: torch.Tensor, yaw_offsets: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  terrain.terrain_levels[:] = 0
  terrain.terrain_types[:] = 0
  terrain.env_origins[:] = terrain.terrain_origins[0, 0]
  clearance = robot.data.root_link_pos_w[:, 2] - terrain.env_origins[:, 2]
  route_start = terrain.env_origins[:, :2].clone()
  root = robot.data.root_link_pose_w.clone()
  root[:, :2] = route_start + torch.stack((torch.zeros_like(cross_offsets), cross_offsets), dim=-1)
  root[:, 2] = terrain.env_origins[:, 2] + clearance
  old_heading = robot.data.heading_w.clone()
  root[:, 3:7] = quat_mul(
    quat_from_euler_xyz(torch.zeros_like(yaw_offsets), torch.zeros_like(yaw_offsets), yaw_offsets - old_heading),
    root[:, 3:7],
  )
  robot.write_root_link_pose_to_sim(root)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  expected = torch.cat((root[:, :2], (terrain.env_origins[:, 2] + clearance).unsqueeze(-1)), dim=-1)
  placement_error = float(torch.max(torch.abs(robot.data.root_link_pos_w - expected)))
  if placement_error > 1e-4:
    raise RuntimeError(f"curved route placement error: {placement_error:.6f}")
  return route_start, placement_error, clearance


def _evaluate_scenarios(cfg: CurvedRouteConfig, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
  """Run all curve parameter combinations in one vectorized environment."""
  num_envs = len(scenarios)
  group_indices = _scenario_group_indices(scenarios)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  terrain_cfg = env_cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_generator = make_curved_flat_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile = _configure_profile(env_cfg, cfg.profile)
  profile.update(_configure_episode_length(env_cfg, cfg.steps))
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command must be UniformVelocityCommand-compatible")
  if hasattr(command_cfg, "focus_terrain_names"):
    command_cfg.focus_terrain_names = ()
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1e9, 1e9)
  command_cfg.ranges.lin_vel_x = (min(cfg.speeds), max(cfg.speeds))
  command_cfg.ranges.lin_vel_y = (-cfg.max_lateral_speed, cfg.max_lateral_speed)
  command_cfg.ranges.ang_vel_z = (-cfg.max_yaw_rate, cfg.max_yaw_rate)
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
  runner.load(str(Path(cfg.checkpoint).expanduser().resolve()), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  device = env.device
  cross_offsets = torch.tensor([s["cross_track_offset"] for s in scenarios], device=device)
  yaw_offsets = torch.tensor([s["yaw_offset"] for s in scenarios], device=device)
  route_start, placement_error, clearance = _place_routes(env, cross_offsets, yaw_offsets)
  robot = env.scene["robot"]

  route_groups: dict[tuple[float, float, int], tuple[torch.Tensor, ArcRoute | SRoute]] = {}
  tape_schedules = {}
  route_lengths = torch.zeros(num_envs, device=device)
  endpoints = torch.zeros(num_envs, 2, device=device)
  endpoint_headings = torch.zeros(num_envs, device=device)
  for key, indices in group_indices.items():
    radius, _, turn_sign = key
    ids = torch.tensor(indices, dtype=torch.long, device=device)
    starts = route_start.index_select(0, ids)
    headings = torch.zeros(len(indices), device=device)
    route: ArcRoute | SRoute = (
      make_arc_route(starts, headings, radius, turn_sign)
      if cfg.route_kind == "arc"
      else make_s_route(starts, headings, radius, turn_sign)
    )
    route_groups[key] = (ids, route)
    tape_schedules[key] = make_command_tape_schedule(
      cfg.route_kind, radius, key[1], turn_sign,
      float(profile["control_dt"]), cfg.settle_steps,
    )
    route_lengths[ids] = route.length
    endpoint, endpoint_heading = route.pose_at(route.length)
    endpoints[ids] = endpoint
    endpoint_headings[ids] = endpoint_heading

  def route_states(
    positions: torch.Tensor, headings: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    progress = torch.zeros(num_envs, device=device)
    cross = torch.zeros_like(progress)
    heading_error = torch.zeros_like(progress)
    for ids, route in route_groups.values():
      p, c, h = _route_errors(
        route, positions.index_select(0, ids), headings.index_select(0, ids)
      )
      progress[ids], cross[ids], heading_error[ids] = p, c, h
    return progress, cross, heading_error

  initial_progress, initial_cross, initial_heading_error = route_states(
    robot.data.root_link_pos_w[:, :2], robot.data.heading_w
  )
  if torch.max(initial_progress.abs()) > 1e-4 or torch.max((initial_cross - cross_offsets).abs()) > 1e-4 or torch.max((initial_heading_error - yaw_offsets).abs()) > 1e-4:
    raise RuntimeError("curved route initialization does not match requested offsets")
  command_term = env.command_manager.get_term("twist")
  if not isinstance(command_term, UniformVelocityCommand):
    raise TypeError("twist command term is not UniformVelocityCommand-compatible")
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  completed = torch.zeros_like(active)
  failed = torch.zeros_like(active)
  reset_count = torch.zeros(num_envs, dtype=torch.long, device=device)
  samples = torch.zeros(num_envs, device=device)
  cross_sq = torch.zeros_like(samples); heading_sq = torch.zeros_like(samples)
  cross_max = torch.zeros_like(samples); heading_max = torch.zeros_like(samples)
  cross_abs_samples = torch.zeros(
    (num_envs, cfg.steps), device=device, dtype=samples.dtype
  )
  heading_abs_samples = torch.zeros_like(cross_abs_samples)
  path_sample_valid = torch.zeros(
    (num_envs, cfg.steps), device=device, dtype=torch.bool
  )
  final_progress = torch.zeros_like(samples); final_cross = torch.zeros_like(samples); final_heading = torch.zeros_like(samples)
  final_position = robot.data.root_link_pos_w[:, :2].clone()
  cmd_sum = torch.zeros(num_envs, 3, device=device); actual_sum = torch.zeros_like(cmd_sum)
  cross_velocity_sum = torch.zeros_like(samples); slip_sum = torch.zeros_like(samples); action_acc_sum = torch.zeros_like(samples)
  first_reason: list[str | None] = [None] * num_envs
  termination_counts = {name: torch.zeros_like(samples) for name in env.termination_manager.active_terms}
  transient_metrics = OnlineCommandTransientMetrics(
    num_envs,
    device=device,
    dtype=robot.data.root_link_pos_w.dtype,
    config=TransientMetricConfig(
      control_dt=float(profile["control_dt"]),
      num_segments=2 if cfg.route_kind == "s_curve" else 1,
    ),
  )
  try:
    feet_sensor = env.scene["feet_ground_contact"]
  except KeyError:
    feet_sensor = None
  foot_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"))
  observation = wrapped.get_observations()
  for _ in range(cfg.steps):
    if not active.any():
      break
    pre_pos = robot.data.root_link_pos_w[:, :2].clone(); pre_heading = robot.data.heading_w.clone()
    pre_progress, pre_cross, pre_heading_error = route_states(pre_pos, pre_heading)
    command = torch.zeros(num_envs, 3, device=device)
    segment_index = torch.zeros(num_envs, dtype=torch.long, device=device)
    controller_saturated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    step_index = _
    for key, (ids, route) in route_groups.items():
      radius, speed, turn_sign = key
      group_progress = pre_progress.index_select(0, ids)
      if cfg.mode == "command_tape":
        tape_command = tape_schedules[key].command_at(
          step_index, device=device, dtype=pre_pos.dtype
        )
        group_command = tape_command.expand(len(ids), -1)
        if cfg.route_kind == "s_curve":
          segment_index[ids] = tape_schedules[key].segment_at(step_index)
      else:
        controller = (
          arc_command_controller if isinstance(route, ArcRoute) else s_command_controller
        )
        group_command = controller(
          route,
          pre_pos.index_select(0, ids),
          pre_heading.index_select(0, ids),
          target_speed=speed,
          cross_track_gain=cfg.cross_track_gain,
          heading_gain=cfg.heading_gain,
          max_lateral_speed=cfg.max_lateral_speed,
          max_yaw_rate=cfg.max_yaw_rate,
        )
        if isinstance(route, SRoute):
          first_progress, _, _ = arc_route_errors(
            route.first,
            pre_pos.index_select(0, ids),
            pre_heading.index_select(0, ids),
          )
          segment_index[ids] = (
            first_progress >= route.first.length - 1.0e-5
          ).long()
        controller_saturated[ids] = (
          (group_command[:, 1].abs() >= cfg.max_lateral_speed - 1.0e-6)
          | (group_command[:, 2].abs() >= cfg.max_yaw_rate - 1.0e-6)
        )
      command[ids] = group_command
    command = torch.where(active.unsqueeze(-1), command, 0.0)
    command_term.vel_command_b[:] = command
    observation = wrapped.get_observations()
    with torch.inference_mode():
      action = policy(observation)
    _, _, dones, _ = wrapped.step(action)
    command_term.vel_command_b[:] = command
    observation = wrapped.get_observations()
    reset = dones.bool()
    reset_count += (reset & active).long()
    post_pos = robot.data.root_link_pos_w[:, :2]; post_heading = robot.data.heading_w
    post_progress, post_cross, post_heading_error = route_states(post_pos, post_heading)
    progress = torch.where(reset, pre_progress, post_progress)
    cross = torch.where(reset, pre_cross, post_cross)
    heading_error = torch.where(reset, pre_heading_error, post_heading_error)
    if cfg.mode == "command_tape":
      completion_allowed = torch.zeros(num_envs, dtype=torch.bool, device=device)
      tape_finished_now = torch.zeros_like(active)
      for key, (ids, _) in route_groups.items():
        schedule = tape_schedules[key]
        completion_allowed[ids] = schedule.completion_allowed(step_index + 1)
        tape_finished_now[ids] = schedule.tape_finished(step_index + 1)
      gated_progress = torch.where(
        completion_allowed, progress, torch.minimum(progress, route_lengths - 1e-5)
      )
    else:
      gated_progress = progress
      tape_finished_now = torch.zeros_like(active)
    lifecycle = update_attempt_status(active, gated_progress, cross, heading_error, reset, route_length=route_lengths, cross_track_tolerance=cfg.cross_track_tolerance, heading_tolerance=cfg.heading_tolerance)
    sample = lifecycle.sample_mask.float(); samples += sample
    cross_sq += cross.square() * sample; heading_sq += heading_error.square() * sample
    cross_abs_samples[:, step_index] = cross.abs()
    heading_abs_samples[:, step_index] = heading_error.abs()
    path_sample_valid[:, step_index] = lifecycle.sample_mask
    cross_max = torch.maximum(cross_max, cross.abs() * sample); heading_max = torch.maximum(heading_max, heading_error.abs() * sample)
    final_progress = torch.where(lifecycle.sample_mask, progress, final_progress); final_cross = torch.where(lifecycle.sample_mask, cross, final_cross); final_heading = torch.where(lifecycle.sample_mask, heading_error, final_heading)
    candidate_position = torch.where(reset.unsqueeze(-1), pre_pos, post_pos)
    final_position = torch.where(lifecycle.sample_mask.unsqueeze(-1), candidate_position, final_position)
    actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1)
    actual = torch.where(reset.unsqueeze(-1), torch.zeros_like(actual), actual)
    transient_metrics.update(
      step_index=step_index,
      command=command,
      actual=actual,
      segment_index=segment_index,
      sample_mask=lifecycle.sample_mask,
      saturated=controller_saturated,
    )
    cmd_sum += command * sample.unsqueeze(-1); actual_sum += actual * sample.unsqueeze(-1)
    expected_h = torch.zeros(num_envs, device=device)
    for ids, route in route_groups.values():
      _, group_heading = route.pose_at(progress.index_select(0, ids))
      expected_h[ids] = group_heading
    world_v = torch.where(reset.unsqueeze(-1), torch.zeros_like(robot.data.root_link_lin_vel_w[:, :2]), robot.data.root_link_lin_vel_w[:, :2])
    cross_velocity_sum += route_normal_velocity(world_v, expected_h).abs() * sample
    if feet_sensor is not None:
      contact = feet_sensor.data.found > 0
      foot_speed = torch.norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1)
      slip_sum += ((foot_speed * contact).sum(-1) / contact.sum(-1).clamp_min(1)) * sample
    action_acc = env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action
    action_acc_sum += torch.mean(torch.abs(action_acc), dim=-1) * sample
    for name in env.termination_manager.active_terms:
      termination_counts[name] += env.termination_manager.get_term(name).float() * lifecycle.sample_mask.float()
    for index in torch.where(lifecycle.failed_now)[0].tolist():
      names = [name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[index])]
      first_reason[index] = names[0] if names else "reset"
    tape_failed = tape_finished_now & lifecycle.active
    for index in torch.where(tape_failed)[0].tolist():
      first_reason[index] = "tape_end"
    completed |= lifecycle.completed_now
    failed |= lifecycle.failed_now | tape_failed
    active = lifecycle.active & ~tape_failed
  for index in torch.where(active)[0].tolist():
    first_reason[index] = "step_limit"
  denom = samples.clamp_min(1.0)
  outputs = []
  for index, scenario in enumerate(scenarios):
    mean_cmd = cmd_sum[index] / denom[index]; mean_actual = actual_sum[index] / denom[index]
    path_mask = path_sample_valid[index]
    cross_distribution = cross_abs_samples[index][path_mask].to(torch.float64)
    heading_distribution = heading_abs_samples[index][path_mask].to(torch.float64)
    if cross_distribution.numel() == 0 or heading_distribution.numel() == 0:
      raise RuntimeError("finished curved attempt has no retained path samples")
    required_yaw_rate = scenario["turn_sign"] * scenario["speed"] / scenario["radius"]
    outputs.append({
      **scenario,
      "route_kind": cfg.route_kind,
      "terrain_type": "evaluation_flat_16m",
      "terrain_level": 0,
      "required_yaw_rate": required_yaw_rate,
      "general_yaw_in_distribution": abs(required_yaw_rate) <= 0.3 + 1e-8,
      "completed": bool(completed[index]),
      "failed": bool(failed[index]),
      "finished": bool(completed[index] | failed[index]),
      "steps_sampled": int(samples[index]),
      "path_completion": bool(completed[index]),
      "arc_length": float(route_lengths[index]),
      "motion_steps": tape_schedules[
        (scenario["radius"], scenario["speed"], scenario["turn_sign"])
      ].motion_steps,
      "settle_steps": tape_schedules[
        (scenario["radius"], scenario["speed"], scenario["turn_sign"])
      ].settle_steps,
      "arc_length_progress": float(final_progress[index]),
      "arc_length_progress_ratio": float(final_progress[index] / route_lengths[index]),
      "lateral_rms": float(torch.sqrt(cross_sq[index] / denom[index])),
      "lateral_p95": float(torch.quantile(cross_distribution, 0.95)),
      "lateral_max": float(cross_max[index]),
      "lateral_final": float(final_cross[index]),
      "heading_rms": float(torch.sqrt(heading_sq[index] / denom[index])),
      "heading_p95": float(torch.quantile(heading_distribution, 0.95)),
      "heading_max": float(heading_max[index]),
      "heading_final": float(final_heading[index]),
      "final_heading_error": float(final_heading[index]),
      "final_position_error": float(torch.norm(final_position[index] - endpoints[index])),
      "commanded_velocity_xy_mean": [float(x) for x in mean_cmd[:2]],
      "actual_velocity_xy_mean": [float(x) for x in mean_actual[:2]],
      "commanded_yaw_rate_mean": float(mean_cmd[2]),
      "actual_yaw_rate_mean": float(mean_actual[2]),
      "commanded_curvature_mean": float(mean_cmd[2] / mean_cmd[0].clamp_min(1e-6)),
      "actual_curvature_response_mean": float(mean_actual[2] / mean_actual[0].abs().clamp_min(1e-6)),
      "cross_axis_velocity_mean": float(cross_velocity_sum[index] / denom[index]),
      "reset_count": int(reset_count[index]),
      "first_failure_reason": first_reason[index],
      "slip_velocity_mean": float(slip_sum[index] / denom[index]),
      "action_acceleration_mean": float(action_acc_sum[index] / denom[index]),
      "command_response_transients": transient_metrics.result(index),
      "termination_counts": {name: float(value[index]) for name, value in termination_counts.items()},
      "route_start_xy": [float(x) for x in route_start[index]],
      "route_endpoint_xy": [float(x) for x in endpoints[index]],
      "route_endpoint_heading": float(endpoint_headings[index]),
      "initial_root_clearance": float(clearance[index]),
    })
  env.close()
  return {
    "profile_settings": profile,
    "terrain_assignment_position_error_max": placement_error,
    "num_envs": num_envs,
    "scenarios": outputs,
  }


def evaluate(cfg: CurvedRouteConfig) -> dict[str, Any]:
  _validate_config(cfg)
  all_scenarios = _scenarios(cfg)
  result = _evaluate_scenarios(cfg, all_scenarios)
  outputs = result["scenarios"]
  completion = sum(item["completed"] for item in outputs) / max(len(outputs), 1)
  return {
    "config": asdict(cfg),
    "checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
    "task_id": cfg.task_id,
    "profile_settings": result["profile_settings"],
    "num_envs": result["num_envs"],
    "terrain_assignment_position_error_max": result["terrain_assignment_position_error_max"],
    "coverage": {"flat_curves": True, "rough_curves": False, "terrain_transitions": False},
    "completion_rate": completion,
    "scenarios": outputs,
  }


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(CurvedRouteConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  print(f"[INFO] Wrote curved-route evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
