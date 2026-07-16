"""Evaluate straight, arc, and S routes under paired randomization profiles.

Each route kind is built in a fresh environment with the same seed, number of
environments, scenario order, checkpoint, horizon, and command limits. The
common path length is ``2*pi*radius/3``: a straight segment, a 120-degree arc,
or two opposite 60-degree arcs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

# Prefer this worktree over an editable install that may target integration.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends

from scripts.evaluate_go2_curved_routes import (
  _configure_episode_length,
  _place_routes,
  make_curved_flat_generator,
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
from src.tasks.velocity.evaluation.matched_route_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  PROFILE_NAMES,
  ROUTE_KINDS,
  MatchedRouteContract,
  OnlineMatchedRouteMetrics,
  action_acceleration,
  aggregate_distributions,
  assert_recursive_finite,
  configure_matched_profile,
  contact_slip_velocity,
  matched_route_length,
  matched_route_local_bounds,
)
from src.tasks.velocity.evaluation.routes import (
  route_frame_errors,
  route_normal_velocity,
  straight_line_controller,
  update_attempt_status,
)
from src.tasks.velocity.evaluation.transient_metrics import (
  OnlineCommandTransientMetrics,
  TransientMetricConfig,
)


@dataclass(frozen=True)
class MatchedRouteConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  profiles: tuple[str, ...] = ("clean", "full_randomized")
  radii: tuple[float, ...] = (1.5, 2.5, 4.0)
  speeds: tuple[float, ...] = (0.3, 0.5, 0.6)
  turn_signs: tuple[int, ...] = (1, -1)
  repeats: int = 1
  steps: int = 2000
  settle_steps: int = 10
  seed: int = 42
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  max_lateral_speed: float = 0.3
  max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.30
  heading_tolerance: float = math.radians(20.0)
  output_file: str = "go2_matched_route_evaluation.json"


def _validate_config(cfg: MatchedRouteConfig) -> None:
  if not cfg.profiles or len(set(cfg.profiles)) != len(cfg.profiles):
    raise ValueError("profiles must be nonempty and unique")
  unknown_profiles = sorted(set(cfg.profiles) - set(PROFILE_NAMES))
  if unknown_profiles:
    raise ValueError(f"unknown profiles: {unknown_profiles}")
  if cfg.repeats <= 0 or cfg.steps <= 0:
    raise ValueError("repeats and steps must be positive")
  if cfg.settle_steps < 0 or cfg.settle_steps >= cfg.steps:
    raise ValueError("settle_steps must be in [0, steps)")
  if not cfg.radii or any(not math.isfinite(x) or x <= 0.0 for x in cfg.radii):
    raise ValueError("radii must be finite and positive")
  if not cfg.speeds or any(not math.isfinite(x) or x <= 0.0 for x in cfg.speeds):
    raise ValueError("speeds must be finite and positive")
  if not cfg.turn_signs or any(sign not in (-1, 1) for sign in cfg.turn_signs):
    raise ValueError("turn_signs must contain only -1 or +1")
  for route_kind in ROUTE_KINDS:
    for radius in cfg.radii:
      for sign in cfg.turn_signs:
        bounds = matched_route_local_bounds(route_kind, radius, sign)
        scan_bounds = (
          bounds[0] - 0.8, bounds[1] + 0.8,
          bounds[2] - 0.8, bounds[3] + 0.8,
        )
        if (
          scan_bounds[0] < 0.0 or scan_bounds[1] > 16.0
          or scan_bounds[2] < 0.0 or scan_bounds[3] > 16.0
        ):
          raise ValueError(
            f"{route_kind} radius={radius} sign={sign} scan footprint "
            f"leaves the 16 m evaluation patch: {scan_bounds}"
          )


def _scenarios(cfg: MatchedRouteConfig) -> list[dict[str, Any]]:
  scenarios: list[dict[str, Any]] = []
  for radius in cfg.radii:
    for speed in cfg.speeds:
      for sign in cfg.turn_signs:
        for repeat in range(cfg.repeats):
          scenarios.append({
            "matched_slot": len(scenarios),
            "radius": radius,
            "speed": speed,
            "turn_sign": sign,
            "repeat": repeat,
          })
  return scenarios


def _git_head() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _route_groups(scenarios: list[dict[str, Any]]) -> dict[tuple[float, float, int], list[int]]:
  groups: dict[tuple[float, float, int], list[int]] = {}
  for index, scenario in enumerate(scenarios):
    key = (scenario["radius"], scenario["speed"], scenario["turn_sign"])
    groups.setdefault(key, []).append(index)
  return groups


def _contact_found(env: ManagerBasedRlEnv, name: str, num_envs: int) -> torch.Tensor:
  try:
    found = env.scene[name].data.found > 0
  except KeyError:
    return torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  return found.reshape(num_envs, -1).any(dim=-1)


def _evaluate_route_kind(
  cfg: MatchedRouteConfig,
  profile_name: str,
  route_kind: str,
  scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
  """Evaluate one route kind; repeated calls intentionally rebuild the env."""
  if route_kind not in ROUTE_KINDS:
    raise ValueError(f"unknown route kind {route_kind!r}")
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  num_envs = len(scenarios)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  terrain_cfg = env_cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_generator = make_curved_flat_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = configure_matched_profile(env_cfg, profile_name)
  episode_settings = _configure_episode_length(env_cfg, cfg.steps)
  profile_settings.update(episode_settings)
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command must be UniformVelocityCommand-compatible")
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
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True,
    map_location="cuda:0",
  )
  policy = runner.get_inference_policy(device="cuda:0")
  device = env.device
  zeros = torch.zeros(num_envs, device=device)
  route_start, placement_error, clearance = _place_routes(env, zeros, zeros)
  robot = env.scene["robot"]
  command_term = env.command_manager.get_term("twist")
  if not isinstance(command_term, UniformVelocityCommand):
    raise TypeError("twist command term is not UniformVelocityCommand-compatible")

  groups = _route_groups(scenarios)
  routes: dict[tuple[float, float, int], tuple[torch.Tensor, ArcRoute | SRoute | None]] = {}
  route_lengths = torch.zeros(num_envs, device=device)
  endpoints = torch.zeros(num_envs, 2, device=device)
  endpoint_headings = torch.zeros(num_envs, device=device)
  for key, indices in groups.items():
    radius, _, sign = key
    ids = torch.tensor(indices, dtype=torch.long, device=device)
    starts = route_start.index_select(0, ids)
    headings = torch.zeros(len(indices), device=device)
    length = matched_route_length(radius)
    if route_kind == "straight":
      route = None
      endpoint = starts + torch.tensor([length, 0.0], device=device)
      endpoint_heading = headings
    elif route_kind == "arc":
      route = make_arc_route(
        starts, headings, radius, sign, angle=2.0 * math.pi / 3.0
      )
      endpoint, endpoint_heading = route.pose_at(length)
    else:
      route = make_s_route(starts, headings, radius, sign)
      endpoint, endpoint_heading = route.pose_at(length)
    routes[key] = (ids, route)
    route_lengths[ids] = length
    endpoints[ids] = endpoint
    endpoint_headings[ids] = endpoint_heading

  def route_states(
    positions: torch.Tensor, headings: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    progress = torch.zeros(num_envs, device=device)
    cross = torch.zeros_like(progress)
    heading_error = torch.zeros_like(progress)
    for ids, route in routes.values():
      if route is None:
        state = route_frame_errors(
          positions.index_select(0, ids), headings.index_select(0, ids),
          route_start.index_select(0, ids), torch.zeros(len(ids), device=device),
        )
        p, c, h = state.progress, state.cross_track, state.heading_error
      elif isinstance(route, ArcRoute):
        p, c, h = arc_route_errors(
          route, positions.index_select(0, ids), headings.index_select(0, ids)
        )
      else:
        p, c, h = s_route_errors(
          route, positions.index_select(0, ids), headings.index_select(0, ids)
        )
      progress[ids], cross[ids], heading_error[ids] = p, c, h
    return progress, cross, heading_error

  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  completed = torch.zeros_like(active)
  failed = torch.zeros_like(active)
  settle_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
  reset_count = torch.zeros(num_envs, dtype=torch.long, device=device)
  sample_count = torch.zeros(num_envs, device=device)
  cross_sq = torch.zeros(num_envs, device=device)
  heading_sq = torch.zeros_like(cross_sq)
  cross_max = torch.zeros_like(cross_sq)
  heading_max = torch.zeros_like(cross_sq)
  final_progress = torch.zeros_like(cross_sq)
  final_cross = torch.zeros_like(cross_sq)
  final_heading = torch.zeros_like(cross_sq)
  final_position = robot.data.root_link_pos_w[:, :2].clone()
  command_sum = torch.zeros(num_envs, 3, device=device)
  command_energy = torch.zeros_like(command_sum)
  actual_sum = torch.zeros_like(command_sum)
  saturation_count = torch.zeros(num_envs, device=device)
  contact_counts = {
    name: torch.zeros(num_envs, device=device)
    for name in ("base_ground_contact", "upper_leg_ground_contact", "calf_ground_contact")
  }
  termination_counts = {
    name: torch.zeros(num_envs, device=device)
    for name in env.termination_manager.active_terms
  }
  first_reason: list[str | None] = [None] * num_envs
  distribution_metrics = OnlineMatchedRouteMetrics(
    num_envs, cfg.steps, device=device,
    dtype=robot.data.root_link_pos_w.dtype,
  )
  transient_metrics = OnlineCommandTransientMetrics(
    num_envs, device=device, dtype=robot.data.root_link_pos_w.dtype,
    config=TransientMetricConfig(
      control_dt=float(episode_settings["control_dt"]),
      num_segments=2 if route_kind == "s_curve" else 1,
    ),
  )
  try:
    feet_sensor = env.scene["feet_ground_contact"]
  except KeyError:
    feet_sensor = None
  foot_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"))
  observation = wrapped.get_observations()
  try:
    for step_index in range(cfg.steps):
      if not active.any():
        break
      pre_pos = robot.data.root_link_pos_w[:, :2].clone()
      pre_heading = robot.data.heading_w.clone()
      pre_progress, pre_cross, pre_heading_error = route_states(pre_pos, pre_heading)
      command = torch.zeros(num_envs, 3, device=device)
      segment_index = torch.zeros(num_envs, dtype=torch.long, device=device)
      saturated = torch.zeros(num_envs, dtype=torch.bool, device=device)
      expected_heading = torch.zeros(num_envs, device=device)
      for key, (ids, route) in routes.items():
        _, speed, _ = key
        group_pos = pre_pos.index_select(0, ids)
        group_heading = pre_heading.index_select(0, ids)
        if route is None:
          group_command = straight_line_controller(
            group_pos, group_heading, route_start.index_select(0, ids),
            torch.zeros(len(ids), device=device), target_speed=speed,
            cross_track_gain=cfg.cross_track_gain,
            heading_gain=cfg.heading_gain,
            max_lateral_speed=cfg.max_lateral_speed,
            max_yaw_rate=cfg.max_yaw_rate,
            route_length=matched_route_length(key[0]),
          )
        else:
          controller = arc_command_controller if isinstance(route, ArcRoute) else s_command_controller
          group_command = controller(
            route, group_pos, group_heading, target_speed=speed,
            cross_track_gain=cfg.cross_track_gain,
            heading_gain=cfg.heading_gain,
            max_lateral_speed=cfg.max_lateral_speed,
            max_yaw_rate=cfg.max_yaw_rate,
          )
          if isinstance(route, SRoute):
            first_progress, _, _ = arc_route_errors(
              route.first, group_pos, group_heading
            )
            segment_index[ids] = (
              first_progress >= route.first.length - 1.0e-5
            ).long()
          _, group_expected = route.pose_at(pre_progress.index_select(0, ids))
          expected_heading[ids] = group_expected
        saturated[ids] = (
          (group_command[:, 1].abs() >= cfg.max_lateral_speed - 1.0e-6)
          | (group_command[:, 2].abs() >= cfg.max_yaw_rate - 1.0e-6)
        )
        command[ids] = group_command
      motion_active = active & ~completed
      command = torch.where(motion_active.unsqueeze(-1), command, 0.0)
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
      post_progress, post_cross, post_heading_error = route_states(post_pos, post_heading)
      progress = torch.where(reset, pre_progress, post_progress)
      cross = torch.where(reset, pre_cross, post_cross)
      heading_error = torch.where(reset, pre_heading_error, post_heading_error)
      lifecycle = update_attempt_status(
        motion_active, progress, cross, heading_error, reset,
        route_length=route_lengths,
        cross_track_tolerance=cfg.cross_track_tolerance,
        heading_tolerance=cfg.heading_tolerance,
      )
      sample = active
      sample_f = sample.float()
      sample_count += sample_f
      cross_sq += cross.square() * sample_f
      heading_sq += heading_error.square() * sample_f
      cross_max = torch.maximum(cross_max, cross.abs() * sample_f)
      heading_max = torch.maximum(heading_max, heading_error.abs() * sample_f)
      final_progress = torch.where(sample, progress, final_progress)
      final_cross = torch.where(sample, cross, final_cross)
      final_heading = torch.where(sample, heading_error, final_heading)
      candidate_position = torch.where(reset.unsqueeze(-1), pre_pos, post_pos)
      final_position = torch.where(sample.unsqueeze(-1), candidate_position, final_position)
      actual = torch.cat(
        (robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]),
        dim=-1,
      )
      actual = torch.where(reset.unsqueeze(-1), torch.zeros_like(actual), actual)
      command_sum += command * sample_f.unsqueeze(-1)
      command_energy += command.square() * sample_f.unsqueeze(-1)
      actual_sum += actual * sample_f.unsqueeze(-1)
      saturation_count += saturated.float() * sample_f
      transient_metrics.update(
        step_index=step_index, command=command, actual=actual,
        segment_index=segment_index, sample_mask=sample, saturated=saturated,
      )
      world_velocity = torch.where(
        reset.unsqueeze(-1),
        torch.zeros_like(robot.data.root_link_lin_vel_w[:, :2]),
        robot.data.root_link_lin_vel_w[:, :2],
      )
      cross_axis = route_normal_velocity(world_velocity, expected_heading).abs()
      if feet_sensor is None:
        slip = torch.zeros(num_envs, device=device)
      else:
        contact = (feet_sensor.data.found > 0).reshape(
          num_envs, len(foot_ids), -1
        ).any(dim=-1)
        foot_velocity = robot.data.site_lin_vel_w[:, foot_ids, :2]
        slip = contact_slip_velocity(foot_velocity, contact)
      action_acc = action_acceleration(
        env.action_manager.action,
        env.action_manager.prev_action,
        env.action_manager.prev_prev_action,
      )
      velocity_error = torch.linalg.vector_norm(actual - command, dim=-1)
      distribution_metrics.update(
        sample_mask=sample, action_acceleration=action_acc,
        slip_velocity=slip, velocity_error=velocity_error,
        cross_axis_velocity=cross_axis,
      )
      for name in termination_counts:
        termination_counts[name] += (
          env.termination_manager.get_term(name).float() * sample_f
        )
      for name in contact_counts:
        contact_counts[name] += _contact_found(env, name, num_envs).float() * sample_f
      for index in torch.where(lifecycle.failed_now)[0].tolist():
        names = [
          name for name in env.termination_manager.active_terms
          if bool(env.termination_manager.get_term(name)[index])
        ]
        first_reason[index] = names[0] if names else "reset"
      completed |= lifecycle.completed_now
      failed |= lifecycle.failed_now
      settle_remaining = torch.where(
        lifecycle.completed_now,
        torch.full_like(settle_remaining, cfg.settle_steps),
        settle_remaining,
      )
      settling = completed & active
      settle_remaining = torch.where(
        settling & ~lifecycle.completed_now,
        (settle_remaining - 1).clamp_min(0),
        settle_remaining,
      )
      settle_failed = settling & reset
      for index in torch.where(settle_failed)[0].tolist():
        names = [
          name for name in env.termination_manager.active_terms
          if bool(env.termination_manager.get_term(name)[index])
        ]
        first_reason[index] = names[0] if names else "reset_during_settle"
      failed |= settle_failed
      settle_done = completed & (settle_remaining == 0)
      active = (lifecycle.active | settling) & ~settle_done & ~settle_failed
  finally:
    env.close()

  for index in torch.where(active)[0].tolist():
    first_reason[index] = "step_limit"
  failed |= active
  denom = sample_count.clamp_min(1.0)
  outputs: list[dict[str, Any]] = []
  for index, scenario in enumerate(scenarios):
    command_mean = command_sum[index] / denom[index]
    actual_mean = actual_sum[index] / denom[index]
    local_bounds = matched_route_local_bounds(
      route_kind, scenario["radius"], scenario["turn_sign"]
    )
    scan_bounds = (
      local_bounds[0] - 0.8, local_bounds[1] + 0.8,
      local_bounds[2] - 0.8, local_bounds[3] + 0.8,
    )
    outputs.append({
      **scenario,
      "route_kind": route_kind,
      "route_length": float(route_lengths[index]),
      "route_corridor_bounds_local_xy": list(local_bounds),
      "terrain_scan_footprint_bounds_local_xy": list(scan_bounds),
      "route_and_scan_inside_patch": True,
      "completed": bool(completed[index]),
      "failed": bool(failed[index]),
      "catastrophic_termination": bool(reset_count[index] > 0),
      "first_failure_reason": first_reason[index],
      "steps_sampled": int(sample_count[index]),
      "progress_ratio": float(final_progress[index] / route_lengths[index]),
      "lateral_rms": float(torch.sqrt(cross_sq[index] / denom[index])),
      "lateral_p95": None,
      "lateral_p95_reason": "not_retained_by_matched_distribution_accumulator",
      "lateral_max": float(cross_max[index]),
      "lateral_final": float(final_cross[index]),
      "heading_rms": float(torch.sqrt(heading_sq[index] / denom[index])),
      "heading_p95": None,
      "heading_p95_reason": "not_retained_by_matched_distribution_accumulator",
      "heading_max": float(heading_max[index]),
      "heading_final": float(final_heading[index]),
      "commanded_velocity_mean": [float(x) for x in command_mean],
      "command_energy": {
        "discrete_sum_squared_by_axis": [
          float(x) for x in command_energy[index]
        ],
        "integral_squared_by_axis": [
          float(x * episode_settings["control_dt"])
          for x in command_energy[index]
        ],
        "definition": "sum(command_axis^2) over motion plus settle samples",
      },
      "actual_velocity_mean": [float(x) for x in actual_mean],
      "response_gain": transient_metrics.result(index),
      "sample_metrics": distribution_metrics.result(index),
      "controller_saturation_fraction": float(
        saturation_count[index] / denom[index]
      ),
      "reset_count": int(reset_count[index]),
      "contact_sample_counts": {
        name: int(values[index]) for name, values in contact_counts.items()
      },
      "termination_counts": {
        name: float(values[index]) for name, values in termination_counts.items()
      },
      "route_start_xy": [float(x) for x in route_start[index]],
      "route_endpoint_xy": [float(x) for x in endpoints[index]],
      "route_endpoint_heading": float(endpoint_headings[index]),
      "final_position_xy": [float(x) for x in final_position[index]],
      "initial_root_clearance": float(clearance[index]),
    })
  catastrophic = sum(item["catastrophic_termination"] for item in outputs)
  return {
    "route_kind": route_kind,
    "profile_settings": profile_settings,
    "num_envs": num_envs,
    "terrain_assignment_position_error_max": placement_error,
    "completion_rate": sum(item["completed"] for item in outputs) / num_envs,
    "catastrophic_termination_fraction": catastrophic / num_envs,
    "action_acceleration": aggregate_distributions(outputs, "action_acceleration"),
    "slip_velocity": aggregate_distributions(outputs, "slip_velocity"),
    "velocity_error": aggregate_distributions(outputs, "velocity_error"),
    "cross_axis_velocity": aggregate_distributions(outputs, "cross_axis_velocity"),
    "scenarios": outputs,
  }


def evaluate(cfg: MatchedRouteConfig) -> dict[str, Any]:
  _validate_config(cfg)
  scenarios = _scenarios(cfg)
  checkpoint = str(Path(cfg.checkpoint).expanduser().resolve())
  profiles: dict[str, Any] = {}
  for profile_name in cfg.profiles:
    route_results = {
      route_kind: _evaluate_route_kind(
        cfg, profile_name, route_kind, scenarios
      )
      for route_kind in ROUTE_KINDS
    }
    invariant_settings = [
      route_results[kind]["profile_settings"] for kind in ROUTE_KINDS
    ]
    if any(settings != invariant_settings[0] for settings in invariant_settings[1:]):
      raise RuntimeError("profile settings differ across matched route kinds")
    contract = MatchedRouteContract(
      checkpoint=checkpoint, task_id=cfg.task_id, seed=cfg.seed,
      profile=profile_name, num_slots=len(scenarios), speeds=cfg.speeds,
      steps=cfg.steps, settle_steps=cfg.settle_steps,
      control_dt=float(invariant_settings[0]["control_dt"]),
    )
    profiles[profile_name] = {
      "matched_invariants": {
        **contract.invariant_fields(),
        "radii": list(cfg.radii),
        "turn_signs": list(cfg.turn_signs),
        "repeats": cfg.repeats,
        "route_length_definition": "2*pi*radius/3",
        "paired_randomization_method": (
          "fresh identical environment construction per route kind with "
          "the same seed, num_envs, and matched_slot order"
        ),
      },
      "profile_settings": invariant_settings[0],
      "route_results": route_results,
    }
  result = {
    "schema_version": 1,
    "git_head": _git_head(),
    "checkpoint": checkpoint,
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "config": asdict(cfg),
    "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
    "coverage": {
      "flat_matched_straight_arc_s_curve": True,
      "rough_curves": False,
      "terrain_transitions": False,
    },
    "profiles": profiles,
  }
  assert_recursive_finite(result)
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(MatchedRouteConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  print(f"[INFO] Wrote matched route evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
