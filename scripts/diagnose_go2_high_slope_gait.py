"""Diagnose V7 foot gait and contact behavior on matched forward scenarios.

This is evaluation-only.  It intentionally uses one fixed V7 checkpoint and
does not register or modify a training task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg

from scripts.evaluate_go2_curved_routes import _configure_episode_length, _configure_profile
from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  HighDifficultyTerrainCurveSubTerrainCfg,
  effective_high_terrain_parameters,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


V7_CHECKPOINT = (
  "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_"
  "focus_probe_2048env_500iter/model_13600.pt"
)
FORBIDDEN_CHECKPOINTS = {"model_13900.pt", "model_13999.pt", "model_14099.pt"}
TERRAIN_CONDITIONS = (
  ("flat", "flat", 0),
  ("slope_up_high", "slope_up", 0),
  ("slope_down_extreme", "slope_down", 1),
)
TERRAIN_TYPES = tuple(item[0] for item in TERRAIN_CONDITIONS)
GENERATOR_TERRAIN_TYPES = ("flat", "slope_up", "slope_down")
GENERATOR_TYPE_INDEX = {name: index for index, name in enumerate(GENERATOR_TERRAIN_TYPES)}
FOOT_NAMES = ("FR", "FL", "RR", "RL")
FOOT_LOAD_ON_N = 20.0
FOOT_LOAD_OFF_N = 10.0


@dataclass(frozen=True)
class GaitConfig:
  checkpoint: str = V7_CHECKPOINT
  task_id: str = "Unitree-Go2-Rough-V7"
  profiles: tuple[str, ...] = ("clean",)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  output_file: str = "go2_high_slope_gait_diagnostics.json"


def _git_head() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1], text=True,
  ).strip()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _validate_config(cfg: GaitConfig) -> None:
  expected = Path(V7_CHECKPOINT).resolve()
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint.name in FORBIDDEN_CHECKPOINTS:
    raise ValueError(f"Rejected checkpoint is not allowed: {checkpoint.name}")
  if checkpoint != expected:
    raise ValueError(
      "This diagnostic is V7-only; checkpoint must be "
      f"{expected}, got {checkpoint}"
    )
  if cfg.task_id != "Unitree-Go2-Rough-V7":
    raise ValueError("task_id must remain Unitree-Go2-Rough-V7")
  if not cfg.profiles or any(p not in {"clean", "dynamics", "randomized"} for p in cfg.profiles):
    raise ValueError("profiles must contain only clean, dynamics, randomized")
  if not cfg.speeds or any(not math.isfinite(v) or v <= 0.0 for v in cfg.speeds):
    raise ValueError("speeds must be finite and positive")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("repeats, sample_steps must be positive and warmup_steps nonnegative")


def _make_gait_generator(seed: int) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    seed=seed,
    size=(18.0, 18.0),
    border_width=20.0,
    num_rows=2,
    num_cols=len(GENERATOR_TERRAIN_TYPES),
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=1.0),
      "slope_up": HighDifficultyTerrainCurveSubTerrainCfg(
        kind="slope_up", proportion=1.0,
      ),
      "slope_down": HighDifficultyTerrainCurveSubTerrainCfg(
        kind="slope_down", proportion=1.0,
      ),
    },
  )


def _scenario_slots(cfg: GaitConfig, condition: str, terrain_kind: str, level: int) -> list[dict[str, Any]]:
  if condition not in TERRAIN_TYPES:
    raise ValueError(f"unknown terrain condition: {condition}")
  rows: list[dict[str, Any]] = []
  slot = 0
  for speed in cfg.speeds:
    for repeat in range(cfg.repeats):
      rows.append({
        "matched_slot": slot,
        "terrain_condition": condition,
        "terrain_kind": terrain_kind,
        "terrain_level": level,
        "speed": float(speed),
        "command_name": f"forward_{speed:g}",
        "repeat": repeat,
      })
      slot += 1
  return rows


def _foot_contact(
  sensor: Any, num_envs: int, num_feet: int, permutation: torch.Tensor,
) -> torch.Tensor:
  found = sensor.data.found
  if found is None:
    return torch.zeros(num_envs, num_feet, dtype=torch.bool, device=sensor.device)
  return found.reshape(num_envs, num_feet, -1).any(dim=-1).index_select(1, permutation)


def _body_contact_any(sensor: Any, num_envs: int) -> torch.Tensor:
  found = sensor.data.found
  if found is None:
    return torch.zeros(num_envs, dtype=torch.bool, device=sensor.device)
  return found.reshape(num_envs, -1).any(dim=-1)


def _foot_force(
  sensor: Any, num_envs: int, num_feet: int, permutation: torch.Tensor,
) -> torch.Tensor | None:
  force = getattr(sensor.data, "force", None)
  if force is None:
    return None
  return force.reshape(num_envs, num_feet, 3).index_select(1, permutation)


def _contact_termination(env: ManagerBasedRlEnv, index: int) -> str:
  names = [
    name for name in env.termination_manager.active_terms
    if bool(env.termination_manager.get_term(name)[index])
  ]
  return names[0] if names else "reset"


def _catastrophic(env: ManagerBasedRlEnv, num_envs: int) -> torch.Tensor:
  value = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  for name in env.termination_manager.active_terms:
    if not env.termination_manager.get_term_cfg(name).time_out:
      value |= env.termination_manager.get_term(name).bool()
  return value


def _require_finite(
  name: str, value: torch.Tensor, mask: torch.Tensor | None = None,
) -> None:
  selected = value if mask is None else value[mask]
  if selected.numel() and not torch.isfinite(selected).all():
    raise RuntimeError(f"non-finite rollout value in {name}")


def _finite_stats(values: torch.Tensor) -> dict[str, Any]:
  values = values[torch.isfinite(values)].to(torch.float64)
  if values.numel() == 0:
    return {"mean": None, "p95": None, "max": None, "count": 0, "reason": "no_valid_samples"}
  return {
    "mean": float(values.mean()),
    "p95": float(torch.quantile(values, 0.95)),
    "max": float(values.max()),
    "count": int(values.numel()),
  }


def _normal_and_clearance(
  terrain_sensor: Any,
  foot_pos: torch.Tensor,
  fallback_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Nearest valid yaw-aligned ray gives terrain-relative height and normal."""
  hit_pos = getattr(terrain_sensor.data, "hit_pos_w", None)
  normals = getattr(terrain_sensor.data, "normals_w", None)
  distances = getattr(terrain_sensor.data, "distances", None)
  nenv, nfeet = foot_pos.shape[:2]
  if hit_pos is None or normals is None or distances is None:
    normal = fallback_normal.expand(nenv, nfeet, 3)
    return torch.zeros(nenv, nfeet, device=foot_pos.device), normal, torch.zeros(nenv, nfeet, dtype=torch.bool, device=foot_pos.device)
  valid = distances >= 0.0
  delta = foot_pos[:, :, None, :2] - hit_pos[:, None, :, :2]
  d2 = delta.square().sum(dim=-1).masked_fill(~valid[:, None, :], torch.inf)
  nearest_d2, nearest = d2.min(dim=-1)
  nearest_hit = torch.gather(hit_pos, 1, nearest[..., None].expand(-1, -1, 3))
  nearest_normal = torch.gather(normals, 1, nearest[..., None].expand(-1, -1, 3))
  valid_nearest = nearest_d2 <= 0.25**2
  normal_norm = torch.linalg.vector_norm(nearest_normal, dim=-1, keepdim=True)
  valid_nearest &= normal_norm.squeeze(-1) > 1.0e-8
  normal = nearest_normal / normal_norm.clamp_min(1.0e-8)
  normal = torch.where(valid_nearest[..., None] & (normal_norm > 1.0e-8), normal, fallback_normal)
  normal = torch.where(normal[..., 2:3] < 0.0, -normal, normal)
  clearance = (foot_pos - nearest_hit) * normal
  return clearance.sum(dim=-1), normal, valid_nearest


def _assign_terrain(
  env: ManagerBasedRlEnv, scenarios: list[dict[str, Any]], device: torch.device,
) -> dict[str, Any]:
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise RuntimeError("terrain origins unavailable")
  robot = env.scene["robot"]
  levels = torch.tensor([s["terrain_level"] for s in scenarios], device=device, dtype=torch.long)
  types = torch.tensor([GENERATOR_TYPE_INDEX[s["terrain_kind"]] for s in scenarios], device=device, dtype=torch.long)
  old_origins = terrain.env_origins.clone()
  old_root = robot.data.root_link_pose_w.clone()
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  new_origins = terrain.env_origins.clone()
  relative = old_root[:, :3] - old_origins
  moved = old_root.clone()
  moved[:, :3] += new_origins - old_origins
  robot.write_root_link_pose_to_sim(moved)
  env.scene.write_data_to_sim(); env.sim.forward(); env.sim.sense()
  relocation_by_env = torch.linalg.vector_norm(
    (robot.data.root_link_pos_w - new_origins) - relative, dim=-1
  )
  relocation = float(torch.maximum(relocation_by_env.max(), torch.tensor(0.0, device=device)))
  if relocation > 1.0e-4:
    raise RuntimeError(f"terrain relocation error {relocation:.6f}")
  clearance = moved[:, 2] - new_origins[:, 2]
  placed = robot.data.root_link_pose_w.clone()
  placed[:, :2] = new_origins[:, :2]
  placed[:, 2] = new_origins[:, 2] + clearance
  heading = robot.data.heading_w.clone()
  zeros = torch.zeros_like(heading)
  placed[:, 3:7] = quat_mul(
    quat_from_euler_xyz(zeros, zeros, -heading), placed[:, 3:7]
  )
  robot.write_root_link_pose_to_sim(placed)
  env.scene.write_data_to_sim(); env.sim.forward(); env.sim.sense()
  placement_by_env = torch.maximum(
    torch.linalg.vector_norm(robot.data.root_link_pos_w - placed[:, :3], dim=-1),
    torch.abs(robot.data.heading_w),
  )
  placement = float(placement_by_env.max())
  if placement > 1.0e-4:
    raise RuntimeError(f"route placement error {placement:.6f}")
  return {
    "terrain_levels": levels,
    "terrain_types": types,
    "terrain_origins": new_origins,
    "clearance": clearance,
    "terrain_assignment_position_error": relocation_by_env,
    "terrain_assignment_position_error_max": relocation,
    "terrain_placement_position_error_max": placement,
  }


def _evaluate_condition(cfg: GaitConfig, profile: str, condition: str, terrain_kind: str, terrain_level: int) -> dict[str, Any]:
  scenarios = _scenario_slots(cfg, condition, terrain_kind, terrain_level)
  num_envs = len(scenarios)
  torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = _make_gait_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = _configure_profile(env_cfg, profile)
  episode_settings = _configure_episode_length(env_cfg, cfg.warmup_steps + cfg.sample_steps + 20)
  control_dt = float(episode_settings["control_dt"])
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command is not UniformVelocityCommand-compatible")
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  if hasattr(command_cfg, "focus_terrain_names"):
    command_cfg.focus_terrain_names = ()
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (min(cfg.speeds), max(cfg.speeds))
  command_cfg.ranges.lin_vel_y = (0.0, 0.0)
  command_cfg.ranges.ang_vel_z = (0.0, 0.0)
  command_cfg.ranges.heading = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(str(Path(cfg.checkpoint).resolve()), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)
    placement = _assign_terrain(env, scenarios, env.device)
    robot = env.scene["robot"]
    command_term = env.command_manager.get_term("twist")
    if not isinstance(command_term, UniformVelocityCommand):
      raise TypeError("twist command term is incompatible")
    foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != FOOT_NAMES:
      raise RuntimeError(f"foot order mismatch: {foot_names}")
    nfeet = len(foot_ids)
    joint_ids, joint_names = robot.find_joints((".*",), preserve_order=True)
    hip_ids, hip_names = robot.find_bodies(tuple(f"{name}_hip" for name in FOOT_NAMES), preserve_order=True)
    if tuple(hip_names) != tuple(f"{name}_hip" for name in FOOT_NAMES):
      raise RuntimeError(f"hip order mismatch: {hip_names}")
    feet_sensor = env.scene["feet_ground_contact"]
    terrain_sensor = env.scene["terrain_scan"]
    desired_geom_names = [f"{name}_foot_collision" for name in FOOT_NAMES]
    sensor_geom_names = [
      slot.primary_name for slot in feet_sensor._slots
      if slot.field_name == "found"
    ]
    if len(sensor_geom_names) != nfeet or len(set(sensor_geom_names)) != nfeet:
      raise RuntimeError(f"unexpected foot sensor slots: {sensor_geom_names}")
    if set(sensor_geom_names) != set(desired_geom_names):
      raise RuntimeError(
        f"foot sensor names do not match Go2 feet: {sensor_geom_names}"
      )
    foot_permutation = torch.tensor(
      [sensor_geom_names.index(name) for name in desired_geom_names],
      device=env.device, dtype=torch.long,
    )
    body_sensors: dict[str, Any] = {}
    for key, name in {
      "base": "base_ground_contact", "upper_leg": "upper_leg_ground_contact", "calf": "calf_ground_contact",
    }.items():
      try:
        body_sensors[key] = env.scene[name]
      except KeyError:
        body_sensors[key] = None
    raw_values = torch.full((num_envs, cfg.sample_steps, len(joint_ids)), torch.nan, device=env.device)
    joint_relative_values = torch.full_like(raw_values, torch.nan)
    acc_values = torch.full((num_envs, cfg.sample_steps), torch.nan, device=env.device)
    clearance_values = torch.full((num_envs, nfeet, cfg.sample_steps), torch.nan, device=env.device)
    slip_values = torch.full_like(clearance_values, torch.nan)
    normal_force_values = torch.full_like(clearance_values, torch.nan)
    signed_normal_force_values = torch.full_like(clearance_values, torch.nan)
    tangent_force_values = torch.full_like(clearance_values, torch.nan)
    pitch_values = torch.full((num_envs, cfg.sample_steps), torch.nan, device=env.device)
    touchdown_body = torch.full((num_envs, nfeet, cfg.sample_steps, 3), torch.nan, device=env.device)
    touchdown_base = torch.full((num_envs, nfeet, cfg.sample_steps, 3), torch.nan, device=env.device)
    step_lengths = torch.full((num_envs, nfeet, cfg.sample_steps), torch.nan, device=env.device)
    step_incline_mask = torch.zeros(
      num_envs, nfeet, cfg.sample_steps, dtype=torch.bool, device=env.device
    )
    incline_mask_values = torch.zeros_like(step_incline_mask)
    raw_contact_mask_values = torch.zeros_like(step_incline_mask)
    loaded_contact_mask_values = torch.zeros_like(step_incline_mask)
    incline_any_values = torch.zeros(
      num_envs, cfg.sample_steps, dtype=torch.bool, device=env.device
    )
    duration_sums = {mode: torch.zeros(num_envs, nfeet, device=env.device) for mode in ("swing", "stance")}
    duration_counts = {mode: torch.zeros(num_envs, nfeet, device=env.device) for mode in ("swing", "stance")}
    touchdown_counts = torch.zeros(num_envs, nfeet, device=env.device)
    incomplete_touchdown_count = torch.zeros(num_envs, nfeet, device=env.device)
    raw_contact_count = torch.zeros(num_envs, nfeet, device=env.device)
    loaded_contact_count = torch.zeros(num_envs, nfeet, device=env.device)
    signed_force_negative_count = torch.zeros(num_envs, nfeet, device=env.device)
    ray_valid_count = torch.zeros(num_envs, nfeet, device=env.device)
    inclined_count = torch.zeros(num_envs, nfeet, device=env.device)
    sample_count = torch.zeros(num_envs, device=env.device)
    reset_count = torch.zeros(num_envs, device=env.device)
    body_contact_count = {key: torch.zeros(num_envs, device=env.device) for key in body_sensors}
    termination_counts = {
      name: torch.zeros(num_envs, device=env.device)
      for name in env.termination_manager.active_terms
    }
    actual_sum = torch.zeros(num_envs, 3, device=env.device)
    command_sum = torch.zeros_like(actual_sum)
    actual_command_sum = torch.zeros_like(actual_sum)
    command_sq_sum = torch.zeros_like(actual_sum)
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    loaded_contact: torch.Tensor | None = None
    first_reason: list[str | None] = [None] * num_envs
    prev_action = torch.zeros(num_envs, len(joint_ids), device=env.device)
    prev_prev_action = torch.zeros_like(prev_action)
    command_values = torch.tensor([[s["speed"], 0.0, 0.0] for s in scenarios], device=env.device)
    command_term.vel_command_b[:] = command_values
    observation = wrapped.get_observations()
    for step in range(cfg.warmup_steps + cfg.sample_steps):
      command_term.vel_command_b[:] = command_values
      with torch.inference_mode():
        action = policy(observation)
      command_term.vel_command_b[:] = command_values
      pre_raw_contact = _foot_contact(
        feet_sensor, num_envs, nfeet, foot_permutation
      )
      pre_foot = robot.data.site_pos_w[:, foot_ids, :].clone()
      pre_foot_vel = robot.data.site_lin_vel_w[:, foot_ids, :].clone()
      pre_joint_relative = (
        robot.data.joint_pos[:, joint_ids]
        - robot.data.default_joint_pos[:, joint_ids]
      ).clone()
      pre_body_contact = {
        body: None if sensor is None else _body_contact_any(sensor, num_envs)
        for body, sensor in body_sensors.items()
      }
      pre_force = _foot_force(
        feet_sensor, num_envs, nfeet, foot_permutation
      )
      pre_force = None if pre_force is None else pre_force.clone()
      pre_normal_fallback = torch.zeros(num_envs, nfeet, 3, device=env.device)
      pre_normal_fallback[..., 2] = 1.0
      if terrain_kind != "flat":
        grad = 0.4 * (0.8 if terrain_kind == "slope_up" and scenarios[0]["terrain_level"] == 0 else 1.0)
        if terrain_kind == "slope_down": grad = -0.4
        pre_normal_fallback[:] = torch.tensor([-grad, 0.0, 1.0], device=env.device)
        pre_normal_fallback /= torch.linalg.vector_norm(pre_normal_fallback, dim=-1, keepdim=True)
      pre_clearance, pre_normal, pre_ray_valid = _normal_and_clearance(terrain_sensor, pre_foot, pre_normal_fallback)
      pre_actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1)
      pre_pitch = torch.atan2(-robot.data.projected_gravity_b[:, 0], torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=-1).clamp_min(1.0e-6))
      _require_finite("policy_action", action, active)
      candidate_acc = action - 2.0 * prev_action + prev_prev_action
      _, _, dones, _ = wrapped.step(action)
      command_term.vel_command_b[:] = command_values
      observation = wrapped.get_observations()
      reset = dones.bool()
      reset_count += (reset & active).float()
      valid = active & ~reset
      post_raw_contact = _foot_contact(
        feet_sensor, num_envs, nfeet, foot_permutation
      )
      post_foot = robot.data.site_pos_w[:, foot_ids, :].clone()
      post_foot_vel = robot.data.site_lin_vel_w[:, foot_ids, :].clone()
      post_joint_relative = (
        robot.data.joint_pos[:, joint_ids]
        - robot.data.default_joint_pos[:, joint_ids]
      ).clone()
      post_force = _foot_force(
        feet_sensor, num_envs, nfeet, foot_permutation
      )
      post_force = None if post_force is None else post_force.clone()
      post_clearance, post_normal, post_ray_valid = _normal_and_clearance(terrain_sensor, post_foot, pre_normal_fallback)
      state_raw_contact = torch.where(
        reset[:, None], pre_raw_contact, post_raw_contact
      )
      state_foot = torch.where(reset[:, None, None], pre_foot, post_foot)
      state_foot_vel = torch.where(reset[:, None, None], pre_foot_vel, post_foot_vel)
      state_force = pre_force if post_force is None else torch.where(reset[:, None, None], pre_force if pre_force is not None else post_force, post_force)
      state_joint_relative = torch.where(
        reset[:, None], pre_joint_relative, post_joint_relative
      )
      state_clearance = torch.where(reset[:, None], pre_clearance, post_clearance)
      state_normal = torch.where(reset[:, None, None], pre_normal, post_normal)
      state_ray_valid = torch.where(reset[:, None], pre_ray_valid, post_ray_valid)
      state_actual = torch.where(reset[:, None], pre_actual, torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1))
      state_pitch = torch.where(reset, pre_pitch, torch.atan2(-robot.data.projected_gravity_b[:, 0], torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=-1).clamp_min(1.0e-6)))
      _require_finite("state_foot", state_foot, active)
      _require_finite("state_foot_velocity", state_foot_vel, active)
      _require_finite("state_joint_relative", state_joint_relative, active)
      _require_finite("state_actual_velocity", state_actual, active)
      if step >= cfg.warmup_steps:
        k = step - cfg.warmup_steps
        sample_count += active.float()
        command_sum += command_values * active[:, None].float()
        actual_sum += state_actual * active[:, None].float()
        command_sq_sum += command_values.square() * active[:, None].float()
        actual_command_sum += command_values * state_actual * active[:, None].float()
        raw_values[:, k] = torch.where(active[:, None], action, torch.nan)
        joint_relative_values[:, k] = torch.where(
          active[:, None], state_joint_relative, torch.nan
        )
        acc_values[:, k] = torch.where(active, candidate_acc.abs().mean(dim=-1), torch.nan)
        pitch_values[:, k] = torch.where(active, state_pitch, torch.nan)
        for body, sensor in body_sensors.items():
          if sensor is not None:
            post_value = _body_contact_any(sensor, num_envs)
            state_value = torch.where(reset, pre_body_contact[body], post_value)
            body_contact_count[body] += (active & state_value).float()
        for name in termination_counts:
          termination_counts[name] += (
            env.termination_manager.get_term(name).float() * active.float()
          )
        force_vec = (
          torch.zeros(num_envs, nfeet, 3, device=env.device)
          if state_force is None else state_force
        )
        signed_fn = (force_vec * state_normal).sum(dim=-1)
        fn = signed_fn.abs()
        raw_contact = state_raw_contact
        if loaded_contact is None:
          loaded_contact = raw_contact & (fn >= FOOT_LOAD_ON_N)
        else:
          loaded_contact = raw_contact & torch.where(
            loaded_contact, fn >= FOOT_LOAD_OFF_N, fn >= FOOT_LOAD_ON_N
          )
        contact = loaded_contact
        ft = torch.linalg.vector_norm(
          force_vec - signed_fn[..., None] * state_normal, dim=-1
        )
        tangent_vel = state_foot_vel - (
          state_foot_vel * state_normal
        ).sum(dim=-1, keepdim=True) * state_normal
        slip = torch.linalg.vector_norm(tangent_vel, dim=-1)
        valid_rows = active[:, None]
        raw_contact_count += (valid_rows & raw_contact).float()
        loaded_contact_count += (valid_rows & contact).float()
        signed_force_negative_count += (
          valid_rows & raw_contact & (signed_fn < 0.0)
        ).float()
        clearance_values[:, :, k] = torch.where(
          valid_rows & ~contact & state_ray_valid, state_clearance, torch.nan
        )
        loaded_valid = valid_rows & contact & state_ray_valid
        inclined = state_ray_valid & (state_normal[..., 2] < 0.99)
        incline_mask_values[:, :, k] = valid_rows & inclined
        raw_contact_mask_values[:, :, k] = valid_rows & raw_contact
        loaded_contact_mask_values[:, :, k] = valid_rows & contact
        incline_any_values[:, k] = active & inclined.any(dim=-1)
        slip_values[:, :, k] = torch.where(loaded_valid, slip, torch.nan)
        normal_force_values[:, :, k] = torch.where(loaded_valid, fn, torch.nan)
        signed_normal_force_values[:, :, k] = torch.where(
          loaded_valid, signed_fn, torch.nan
        )
        tangent_force_values[:, :, k] = torch.where(loaded_valid, ft, torch.nan)
        ray_valid_count += (valid_rows & state_ray_valid).float()
        inclined_count += (
          valid_rows & state_ray_valid & (state_normal[..., 2] < 0.99)
        ).float()
        # Gait event state is updated only for non-reset rows, so a reset cannot
        # leak a new episode's contact or foot position into the old attempt.
        if k == 0:
          prev_contact = contact.clone()
          mode_duration = torch.zeros(num_envs, nfeet, device=env.device)
          liftoff_pos = state_foot.clone()
          has_liftoff = torch.zeros(
            num_envs, nfeet, dtype=torch.bool, device=env.device
          )
          step_count = torch.zeros(num_envs, nfeet, device=env.device)
        transition = valid[:, None] & (contact != prev_contact)
        for mode, mask in (("stance", prev_contact), ("swing", ~prev_contact)):
          completed_intervals = transition & mask
          duration_sums[mode] += completed_intervals.float() * mode_duration
          duration_counts[mode] += completed_intervals.float()
        touchdown = transition & contact
        liftoff = transition & ~contact
        has_liftoff |= liftoff
        if touchdown.any():
          delta = state_foot - liftoff_pos
          forward = torch.zeros(num_envs, nfeet, 3, device=env.device)
          forward[..., 0] = 1.0
          tangent = forward - (
            forward * state_normal
          ).sum(dim=-1, keepdim=True) * state_normal
          tangent /= torch.linalg.vector_norm(
            tangent, dim=-1, keepdim=True
          ).clamp_min(1.0e-8)
          length = (delta * tangent).sum(dim=-1)
          idx = step_count.long().clamp_max(cfg.sample_steps - 1)
          for env_id, foot_id in torch.nonzero(
            touchdown, as_tuple=False
          ).tolist():
            event_index = int(idx[env_id, foot_id])
            complete_swing = bool(
              has_liftoff[env_id, foot_id]
              and state_ray_valid[env_id, foot_id]
            )
            if complete_swing:
              step_lengths[env_id, foot_id, event_index] = length[env_id, foot_id]
              step_incline_mask[env_id, foot_id, event_index] = bool(
                state_normal[env_id, foot_id, 2] < 0.99
              )
            else:
              incomplete_touchdown_count[env_id, foot_id] += 1
            root_quat = robot.data.root_link_quat_w[env_id]
            hip_quat = robot.data.body_link_quat_w[
              env_id, hip_ids[foot_id]
            ]
            rel_hip = (
              state_foot[env_id, foot_id]
              - robot.data.body_link_pos_w[env_id, hip_ids[foot_id]]
            )
            rel_base = state_foot[env_id, foot_id] - robot.data.root_link_pos_w[env_id]
            touchdown_body[env_id, foot_id, event_index] = quat_apply_inverse(
              hip_quat, rel_hip
            )
            touchdown_base[env_id, foot_id, event_index] = quat_apply_inverse(
              root_quat, rel_base
            )
          step_count += touchdown.float()
          touchdown_counts += touchdown.float()
        liftoff_pos = torch.where(liftoff[..., None], state_foot, liftoff_pos)
        has_liftoff &= ~touchdown
        prev_contact = torch.where(valid[:, None], contact, prev_contact)
        mode_duration = torch.where(transition, 0.0, mode_duration)
        mode_duration = torch.where(
          valid[:, None], mode_duration + control_dt, mode_duration
        )
      for env_id in torch.where(reset & active)[0].tolist():
        first_reason[env_id] = _contact_termination(env, env_id)
      active &= ~reset
      prev_prev_action, prev_action = prev_action, action.detach()
    denominator = sample_count.clamp_min(1.0)
    output_rows: list[dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
      response_gain: dict[str, float | str | None] = {}
      for axis, axis_name in enumerate(("vx", "vy", "wz")):
        if float(command_sq_sum[i, axis]) <= 1.0e-12:
          response_gain[axis_name] = None
          response_gain[f"{axis_name}_reason"] = "no_nonzero_command_energy"
        else:
          response_gain[axis_name] = float(
            actual_command_sum[i, axis] / command_sq_sum[i, axis]
          )
      row: dict[str, Any] = {
        **scenario,
        "profile": profile,
        "checkpoint": str(Path(cfg.checkpoint).resolve()),
        "sample_count": int(sample_count[i]),
        "reset_count": int(reset_count[i]),
        "failed": first_reason[i] is not None,
        "first_failure_reason": first_reason[i],
        "commanded_velocity_mean": [float(x) for x in command_sum[i] / denominator[i]],
        "actual_velocity_mean": [float(x) for x in actual_sum[i] / denominator[i]],
        "response_gain": response_gain,
        "base_pitch": _finite_stats(pitch_values[i]),
        "action_acceleration": _finite_stats(acc_values[i]),
        "terrain_relative_clearance": [_finite_stats(clearance_values[i, f]) for f in range(nfeet)],
        "foot_slip_tangent": [_finite_stats(slip_values[i, f]) for f in range(nfeet)],
        "foot_force_normal": [_finite_stats(normal_force_values[i, f]) for f in range(nfeet)],
        "foot_force_normal_signed": [
          _finite_stats(signed_normal_force_values[i, f])
          for f in range(nfeet)
        ],
        "foot_force_tangent": [_finite_stats(tangent_force_values[i, f]) for f in range(nfeet)],
        "normal_force_negative_fraction_when_found": [
          float(
            signed_force_negative_count[i, f]
            / raw_contact_count[i, f].clamp_min(1.0)
          )
          for f in range(nfeet)
        ],
        "terrain_normal_ray_valid_fraction": [
          float(ray_valid_count[i, f] / denominator[i]) for f in range(nfeet)
        ],
        "foot_names": list(foot_names),
        "raw_found_contact_fraction": [
          float(raw_contact_count[i, f] / denominator[i])
          for f in range(nfeet)
        ],
        "loaded_contact_fraction": [
          float(loaded_contact_count[i, f] / denominator[i])
          for f in range(nfeet)
        ],
        "touchdown_count": [int(x) for x in touchdown_counts[i]],
        "incomplete_touchdown_excluded_count": [
          int(x) for x in incomplete_touchdown_count[i]
        ],
        "step_length": [_finite_stats(step_lengths[i, f]) for f in range(nfeet)],
        "step_length_absolute": [
          _finite_stats(step_lengths[i, f].abs()) for f in range(nfeet)
        ],
        "forward_swing_displacement": [
          _finite_stats(step_lengths[i, f]) for f in range(nfeet)
        ],
        "touchdown_relative_hip_body": [
          {
            axis_name: _finite_stats(touchdown_body[i, f, :, axis_index])
            for axis_index, axis_name in enumerate(("x", "y", "z"))
          }
          for f in range(nfeet)
        ],
        "touchdown_relative_base_body": [
          {
            axis_name: _finite_stats(touchdown_base[i, f, :, axis_index])
            for axis_index, axis_name in enumerate(("x", "y", "z"))
          }
          for f in range(nfeet)
        ],
        "swing_duration": [
          {"mean": float(duration_sums["swing"][i, f] / duration_counts["swing"][i, f]) if duration_counts["swing"][i, f] > 0 else None,
           "count": int(duration_counts["swing"][i, f])}
          for f in range(nfeet)
        ],
        "stance_duration": [
          {"mean": float(duration_sums["stance"][i, f] / duration_counts["stance"][i, f]) if duration_counts["stance"][i, f] > 0 else None,
           "count": int(duration_counts["stance"][i, f])}
          for f in range(nfeet)
        ],
        "duty_factor": [float(duration_sums["stance"][i, f] / (duration_sums["stance"][i, f] + duration_sums["swing"][i, f]).clamp_min(1.0e-12)) if duration_counts["stance"][i, f] > 0 and duration_counts["swing"][i, f] > 0 else None for f in range(nfeet)],
        "duty_factor_sample_occupancy": [
          float(loaded_contact_count[i, f] / denominator[i])
          for f in range(nfeet)
        ],
        "actions": {
          "joint_names": list(joint_names),
          "raw_policy_amplitude": [
            _finite_stats(raw_values[i, :, j].abs())
            for j in range(len(joint_ids))
          ],
          "groups": {
            group: [
              _finite_stats(raw_values[i, :, j].abs())
              for j, name in enumerate(joint_names) if group in name
            ]
            for group in ("hip", "thigh", "calf")
          },
          "joint_position_relative_amplitude_rad": {
            group: [
              _finite_stats(joint_relative_values[i, :, j].abs())
              for j, name in enumerate(joint_names) if group in name
            ]
            for group in ("hip", "thigh", "calf")
          },
        },
        "contacts": {
          body: {"count": int(body_contact_count[body][i]), "rate": float(body_contact_count[body][i] / denominator[i])}
          for body in body_sensors
        },
        "termination_counts": {
          name: int(values[i]) for name, values in termination_counts.items()
        },
        "terrain_parameters": (
          {"terrain_kind": "flat", "difficulty_affects_geometry": False}
          if terrain_kind == "flat" else effective_high_terrain_parameters(terrain_kind, scenario["terrain_level"])
        ),
        "terrain_inclined_foot_sample_fraction": [
          float(inclined_count[i, f] / denominator[i]) for f in range(nfeet)
        ],
        "on_incline": {
          "foot_sample_count": [
            int(incline_mask_values[i, f].sum()) for f in range(nfeet)
          ],
          "loaded_contact_sample_count": [
            int((incline_mask_values[i, f] & loaded_contact_mask_values[i, f]).sum())
            for f in range(nfeet)
          ],
          "raw_contact_fraction": [
            float(
              (incline_mask_values[i, f] & raw_contact_mask_values[i, f]).sum()
              / incline_mask_values[i, f].sum().clamp_min(1)
            )
            for f in range(nfeet)
          ],
          "loaded_contact_fraction": [
            float(
              (incline_mask_values[i, f] & loaded_contact_mask_values[i, f]).sum()
              / incline_mask_values[i, f].sum().clamp_min(1)
            )
            for f in range(nfeet)
          ],
          "terrain_relative_clearance": [
            _finite_stats(torch.where(
              incline_mask_values[i, f], clearance_values[i, f], torch.nan
            ))
            for f in range(nfeet)
          ],
          "foot_slip_tangent": [
            _finite_stats(torch.where(
              incline_mask_values[i, f], slip_values[i, f], torch.nan
            ))
            for f in range(nfeet)
          ],
          "foot_force_normal": [
            _finite_stats(torch.where(
              incline_mask_values[i, f], normal_force_values[i, f], torch.nan
            ))
            for f in range(nfeet)
          ],
          "foot_force_tangent": [
            _finite_stats(torch.where(
              incline_mask_values[i, f], tangent_force_values[i, f], torch.nan
            ))
            for f in range(nfeet)
          ],
          "step_length_absolute": [
            _finite_stats(torch.where(
              step_incline_mask[i, f], step_lengths[i, f].abs(), torch.nan
            ))
            for f in range(nfeet)
          ],
          "base_pitch": _finite_stats(torch.where(
            incline_any_values[i], pitch_values[i], torch.nan
          )),
          "action_acceleration": _finite_stats(torch.where(
            incline_any_values[i], acc_values[i], torch.nan
          )),
        },
        "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]),
        "terrain_assignment_position_error": float(
          placement["terrain_assignment_position_error"][i]
        ),
        "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]),
        "terrain_type_index": int(placement["terrain_types"][i]),
        "terrain_origin_xyz": [
          float(x) for x in placement["terrain_origins"][i]
        ],
      }
      output_rows.append(row)
    return {
      "terrain_condition": condition,
      "terrain_kind": terrain_kind,
      "profile": profile,
      "profile_settings": profile_settings,
      "episode_settings": episode_settings,
      "num_envs": num_envs,
      "matched_slots": [s["matched_slot"] for s in scenarios],
      "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]),
      "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]),
      "foot_sensor_primary_names": sensor_geom_names,
      "foot_sensor_to_output_permutation": [int(x) for x in foot_permutation],
      "scenarios": output_rows,
    }
  finally:
    env.close()


def evaluate(cfg: GaitConfig) -> dict[str, Any]:
  _validate_config(cfg)
  results: dict[str, Any] = {}
  for profile in cfg.profiles:
    for condition, terrain_kind, terrain_level in TERRAIN_CONDITIONS:
      results[f"{profile}|{condition}"] = _evaluate_condition(
        cfg, profile, condition, terrain_kind, terrain_level
      )
  by_profile = {}
  for profile in cfg.profiles:
    slots = [results[f"{profile}|{condition}"]["matched_slots"] for condition, _, _ in TERRAIN_CONDITIONS]
    if any(value != slots[0] for value in slots[1:]):
      raise RuntimeError("matched slot order differs across terrain conditions")
    by_profile[profile] = {
      "matched_slots": slots[0],
      "conditions": {
        condition: results[f"{profile}|{condition}"]
        for condition, _, _ in TERRAIN_CONDITIONS
      },
    }
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_high_slope_gait_diagnostic",
    "git_head": _git_head(),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
    "checkpoint_sha256": _sha256(Path(cfg.checkpoint).expanduser().resolve()),
    "config": asdict(cfg),
    "metric_definitions": {
      "foot_order": list(FOOT_NAMES),
      "contact": (
        "raw found>0 is reported separately; loaded contact uses normal-force "
        f"hysteresis {FOOT_LOAD_ON_N:g}N/{FOOT_LOAD_OFF_N:g}N at control-step resolution; "
        "feet have no force history"
      ),
      "swing_stance": "loaded-contact transitions at control-step resolution; incomplete intervals excluded",
      "clearance": "signed foot-to-nearest-valid-yaw-aligned-terrain-ray normal distance; null when no valid ray",
      "on_incline": "normal.z < 0.99 with a valid nearest terrain ray; condition-specific distributions are separated from platform samples",
      "slip": "tangent foot velocity magnitude during loaded contact and valid terrain ray",
      "step_length": "complete swing foot displacement from observed lift-off to touchdown projected on the local tangent plane",
      "action": "raw policy action; joint targets are not inferred from scaled action",
      "terminal_lifecycle": "reset row is counted for command/action/reset metrics but cannot update gait state; later episode samples are frozen",
    },
    "coverage": {
      "terrain_conditions": list(TERRAIN_TYPES),
      "commands": [f"forward_{speed:g}" for speed in cfg.speeds],
      "vy": 0.0,
      "yaw": 0.0,
      "training_changed": False,
      "randomized_added_only_if_requested": True,
    },
    "profiles": by_profile,
  }
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(GaitConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps(payload, indent=2, allow_nan=False))
  print(f"[INFO] Wrote high-slope gait diagnostics to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
