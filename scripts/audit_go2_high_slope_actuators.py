"""Evaluation-only actuator and contact-force audit for the V7 Go2 policy.

The audit deliberately keeps the V7 task and checkpoint fixed.  It records the
processed position target, joint state, actuator force and failure windows on
the same matched matrix used by the gait diagnostic.  No training state or
configuration is changed.
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
from mjlab.utils.torch import configure_torch_backends

from scripts.diagnose_go2_high_slope_gait import (
  FOOT_LOAD_OFF_N,
  FOOT_LOAD_ON_N,
  FOOT_NAMES,
  GENERATOR_TYPE_INDEX,
  TERRAIN_CONDITIONS,
  _assign_terrain,
  _body_contact_any,
  _contact_termination,
  _finite_stats,
  _foot_contact,
  _foot_force,
  _git_head,
  _make_gait_generator,
  _normal_and_clearance,
  _scenario_slots,
  _sha256,
)
from scripts.evaluate_go2_curved_routes import _configure_episode_length, _configure_profile
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


V7_CHECKPOINT = (
  "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_"
  "focus_probe_2048env_500iter/model_13600.pt"
)
FORBIDDEN_CHECKPOINTS = {"model_13900.pt", "model_13999.pt", "model_14099.pt"}


@dataclass(frozen=True)
class AuditConfig:
  checkpoint: str = V7_CHECKPOINT
  task_id: str = "Unitree-Go2-Rough-V7"
  profile: str = "clean"
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  stable_tail_steps: int = 300
  failure_windows: tuple[int, ...] = (50, 100)
  seed: int = 42
  device: str = "cuda:0"
  output_file: str = "go2_high_slope_actuator_audit.json"


def _validate_config(cfg: AuditConfig) -> None:
  expected = Path(V7_CHECKPOINT).resolve()
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != expected or checkpoint.name in FORBIDDEN_CHECKPOINTS:
    raise ValueError(f"V7-only checkpoint required: {expected}")
  if cfg.task_id != "Unitree-Go2-Rough-V7" or cfg.profile != "clean":
    raise ValueError("task_id/profile are fixed to V7 clean evaluation")
  if not cfg.speeds or any(not math.isfinite(v) or v <= 0.0 for v in cfg.speeds):
    raise ValueError("speeds must be finite and positive")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("repeats, sample_steps must be positive")
  if cfg.stable_tail_steps <= 0 or cfg.stable_tail_steps > cfg.sample_steps:
    raise ValueError("stable_tail_steps must be within sample_steps")
  if tuple(sorted(set(cfg.failure_windows))) != cfg.failure_windows or any(
    window <= 0 or window > cfg.sample_steps for window in cfg.failure_windows
  ):
    raise ValueError("failure_windows must be sorted positive sample windows")


def _longest_true_run(mask: torch.Tensor) -> int:
  values = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
  best = current = 0
  for value in values:
    current = current + 1 if value else 0
    best = max(best, current)
  return best


def _stats_with_status(values: torch.Tensor) -> dict[str, Any]:
  return _finite_stats(values)


def _window_indices(
  sample_count: int, failed: bool, window: int, stable_tail: int,
) -> tuple[bool, int, int, str | None]:
  if failed:
    if sample_count < window:
      return False, 0, 0, "insufficient_failure_window"
    return True, sample_count - window, sample_count, None
  if sample_count < stable_tail:
    return False, 0, 0, "no_full_stable_attempt"
  return True, sample_count - stable_tail, sample_count, None


def _group_for_joint(name: str) -> str:
  if "calf" in name:
    return "calf"
  if "thigh" in name:
    return "thigh"
  if "hip" in name:
    return "hip"
  return "other"


def _snapshot(
  env: ManagerBasedRlEnv,
  robot: Any,
  feet_sensor: Any,
  terrain_sensor: Any,
  body_sensors: dict[str, Any],
  foot_ids: torch.Tensor,
  foot_permutation: torch.Tensor,
  joint_ids: torch.Tensor,
  force_ids: torch.Tensor,
  current_action: torch.Tensor,
  terrain_kind: str,
  terrain_level: int,
  loaded_contact: torch.Tensor,
) -> dict[str, torch.Tensor]:
  num_envs = env.num_envs
  nfeet = len(FOOT_NAMES)
  joint_pos = robot.data.joint_pos[:, joint_ids].clone()
  joint_vel = robot.data.joint_vel[:, joint_ids].clone()
  target = robot.data.joint_pos_target[:, joint_ids].clone()
  force = robot.data.actuator_force[:, force_ids].clone()
  normal_fallback = torch.zeros(num_envs, nfeet, 3, device=env.device)
  normal_fallback[..., 2] = 1.0
  if terrain_kind != "flat":
    gradient = 0.4 * (0.8 if terrain_kind == "slope_up" and terrain_level == 0 else 1.0)
    if terrain_kind == "slope_down":
      gradient = -0.4
    normal_fallback[:] = torch.tensor(
      [-gradient, 0.0, 1.0], device=env.device
    )
    normal_fallback /= torch.linalg.vector_norm(
      normal_fallback, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
  foot_pos = robot.data.site_pos_w[:, foot_ids, :].clone()
  foot_vel = robot.data.site_lin_vel_w[:, foot_ids, :].clone()
  _, normal, ray_valid = _normal_and_clearance(
    terrain_sensor, foot_pos, normal_fallback
  )
  foot_force = _foot_force(feet_sensor, num_envs, nfeet, foot_permutation)
  if foot_force is None:
    foot_force = torch.zeros(num_envs, nfeet, 3, device=env.device)
  signed_normal = (foot_force * normal).sum(dim=-1)
  normal_force = signed_normal.abs()
  tangent_force = torch.linalg.vector_norm(
    foot_force - signed_normal[..., None] * normal, dim=-1
  )
  tangent_vel = foot_vel - (foot_vel * normal).sum(dim=-1, keepdim=True) * normal
  slip = torch.linalg.vector_norm(tangent_vel, dim=-1)
  raw_contact = _foot_contact(feet_sensor, num_envs, nfeet, foot_permutation)
  loaded = raw_contact & torch.where(
    loaded_contact, normal_force >= FOOT_LOAD_OFF_N, normal_force >= FOOT_LOAD_ON_N
  )
  q = robot.data.joint_pos[:, joint_ids]
  pitch = torch.atan2(
    -robot.data.projected_gravity_b[:, 0],
    torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=-1).clamp_min(1.0e-6),
  )
  body = torch.stack(
    [
      torch.zeros(num_envs, dtype=torch.bool, device=env.device)
      if sensor is None else _body_contact_any(sensor, num_envs)
      for sensor in body_sensors.values()
    ],
    dim=-1,
  )
  return {
    "action": current_action.clone(),
    "joint_pos": q.clone(),
    "joint_vel": joint_vel,
    "target": target,
    "force": force,
    "normal_force": normal_force,
    "signed_normal_force": signed_normal,
    "tangent_force": tangent_force,
    "slip": slip,
    "loaded": loaded,
    "ray_valid": ray_valid,
    "pitch": pitch,
    "body_contact": body,
  }


def _capture_reset_state(
  env: ManagerBasedRlEnv,
  robot: Any,
  feet_sensor: Any,
  terrain_sensor: Any,
  body_sensors: dict[str, Any],
  foot_ids: torch.Tensor,
  foot_permutation: torch.Tensor,
  joint_ids: torch.Tensor,
  force_ids: torch.Tensor,
  current_action: torch.Tensor,
  terrain_kind: str,
  terrain_level: int,
  loaded_contact: torch.Tensor,
  env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
  snapshot = _snapshot(
    env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
    foot_permutation, joint_ids, force_ids, current_action, terrain_kind,
    terrain_level, loaded_contact,
  )
  return {key: value.index_select(0, env_ids).clone() for key, value in snapshot.items()}


def _window_metrics(
  arrays: dict[str, torch.Tensor],
  env_index: int,
  start: int,
  end: int,
  joint_names: list[str],
  hard_limits: torch.Tensor,
  soft_limits: torch.Tensor,
  effort_limits: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
  clip_bound: float | None,
) -> dict[str, Any]:
  raw = arrays["action"][env_index, start:end]
  q = arrays["joint_pos"][env_index, start:end]
  qd = arrays["joint_vel"][env_index, start:end]
  target = arrays["target"][env_index, start:end]
  force = arrays["force"][env_index, start:end]
  demand = kp * (target - q) - kd * qd
  limit = effort_limits.unsqueeze(0).clamp_min(1.0e-8)
  utilization = force.abs() / limit
  sat = (utilization >= 0.98) & (demand.abs() >= limit)
  soft = soft_limits.unsqueeze(0)
  soft_violation = (q < soft[..., 0]) | (q > soft[..., 1])
  target_soft_violation = (target < soft[..., 0]) | (target > soft[..., 1])
  hard_margin = torch.minimum(q - hard_limits[:, 0], hard_limits[:, 1] - q)
  action_acc = raw[2:] - 2.0 * raw[1:-1] + raw[:-2]
  per_joint: dict[str, Any] = {}
  for index, name in enumerate(joint_names):
    per_joint[name] = {
      "raw_action_abs": _stats_with_status(raw[:, index].abs()),
      "target_position": _stats_with_status(target[:, index]),
      "joint_position": _stats_with_status(q[:, index]),
      "position_error_abs": _stats_with_status((target[:, index] - q[:, index]).abs()),
      "joint_velocity_abs": _stats_with_status(qd[:, index].abs()),
      "actuator_force_abs": _stats_with_status(force[:, index].abs()),
      "pd_demand_abs": _stats_with_status(demand[:, index].abs()),
      "effort_utilization": _stats_with_status(utilization[:, index]),
      "clip_residual_abs": _stats_with_status((demand[:, index] - force[:, index]).abs()),
      "action_second_difference_abs": _stats_with_status(
        action_acc[:, index].abs() if action_acc.numel() else action_acc[:, index]
      ),
      "near_effort_limit_fraction": float((utilization[:, index] >= 0.95).float().mean()),
      "effort_saturation_fraction": float(sat[:, index].float().mean()),
      "effort_saturation_longest_run": _longest_true_run(sat[:, index]),
      "soft_limit_violation_fraction": float(soft_violation[:, index].float().mean()),
      "target_soft_limit_violation_fraction": float(target_soft_violation[:, index].float().mean()),
      "hard_limit_margin_min": float(hard_margin[:, index].min()),
      "hard_limit_near_fraction": float((hard_margin[:, index] <= 0.02).float().mean()),
    }
  groups: dict[str, Any] = {}
  for group in ("hip", "thigh", "calf"):
    ids = [i for i, name in enumerate(joint_names) if _group_for_joint(name) == group]
    groups[group] = {
      "effort_utilization": _stats_with_status(utilization[:, ids].reshape(-1)),
      "effort_saturation_fraction": float(sat[:, ids].float().mean()),
      "soft_limit_violation_fraction": float(soft_violation[:, ids].float().mean()),
    }
  result: dict[str, Any] = {
    "eligible": True,
    "sample_count": int(end - start),
    "per_joint": per_joint,
    "groups": groups,
    "body_contact_fraction": [
      float(arrays["body_contact"][env_index, start:end, i].float().mean())
      for i in range(arrays["body_contact"].shape[-1])
    ],
    "base_pitch": _stats_with_status(arrays["pitch"][env_index, start:end]),
    "foot_normal_force_abs_max": _stats_with_status(
      arrays["normal_force"][env_index, start:end].amax(dim=-1)
    ),
    "foot_tangent_force_max": _stats_with_status(
      arrays["tangent_force"][env_index, start:end].amax(dim=-1)
    ),
    "stance_slip": _stats_with_status(
      torch.where(
        arrays["loaded"][env_index, start:end],
        arrays["slip"][env_index, start:end],
        torch.nan,
      ).reshape(-1)
    ),
    "foot_signed_normal_force": _stats_with_status(
      torch.where(
        arrays["loaded"][env_index, start:end],
        arrays["signed_normal_force"][env_index, start:end],
        torch.nan,
      )
    ),
    "foot_signed_normal_negative_fraction_when_loaded": None,
    "clip_bound": clip_bound,
    "clip_fraction": None if clip_bound is None else float(
      (raw.abs() >= clip_bound - 1.0e-6).float().mean()
    ),
    "measurement_timing": "post-step pre-reset snapshot; terminal rows captured inside _reset_idx before reset",
  }
  loaded = arrays["loaded"][env_index, start:end]
  if loaded.any():
    result["foot_signed_normal_negative_fraction_when_loaded"] = float(
      (arrays["signed_normal_force"][env_index, start:end][loaded] < 0.0)
      .float()
      .mean()
    )
  return result


def _audit_condition(cfg: AuditConfig, condition: str, terrain_kind: str, terrain_level: int) -> dict[str, Any]:
  scenarios = _scenario_slots(cfg, condition, terrain_kind, terrain_level)
  num_envs = len(scenarios)
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = _make_gait_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  _configure_profile(env_cfg, cfg.profile)
  episode_settings = _configure_episode_length(
    env_cfg, cfg.warmup_steps + cfg.sample_steps + 20
  )
  control_dt = float(episode_settings["control_dt"])
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command is incompatible")
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
    joint_ids, joint_names = robot.find_joints((".*",), preserve_order=True)
    joint_names = list(joint_names)
    action_term = env.action_manager.get_term("joint_pos")
    if list(action_term.target_names) != joint_names:
      raise RuntimeError("action target and joint ordering differ")
    force_ids = torch.full(
      (len(joint_names),), -1, dtype=torch.long, device=env.device
    )
    for actuator in robot.actuators:
      for target_name, ctrl_id in zip(actuator.target_names, actuator.ctrl_ids.tolist(), strict=True):
        if target_name in joint_names:
          force_ids[joint_names.index(target_name)] = int(ctrl_id)
    if (force_ids < 0).any():
      raise RuntimeError(f"actuator force mapping incomplete: {force_ids.tolist()}")
    nfeet = len(FOOT_NAMES)
    feet_sensor = env.scene["feet_ground_contact"]
    terrain_sensor = env.scene["terrain_scan"]
    sensor_geom_names = [slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"]
    desired_geom_names = [f"{name}_foot_collision" for name in FOOT_NAMES]
    foot_permutation = torch.tensor(
      [sensor_geom_names.index(name) for name in desired_geom_names],
      dtype=torch.long, device=env.device,
    )
    body_sensors: dict[str, Any] = {}
    for key, name in {"base": "base_ground_contact", "upper_leg": "upper_leg_ground_contact", "calf": "calf_ground_contact"}.items():
      try:
        body_sensors[key] = env.scene[name]
      except KeyError:
        body_sensors[key] = None
    hard_limits = robot.data.joint_pos_limits[0, joint_ids].clone()
    soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].clone()
    effort_limits = torch.tensor(
      [45.0 if "calf" in name else 23.5 for name in joint_names],
      device=env.device,
    )
    kp = torch.tensor([40.0 if "calf" in name else 20.0 for name in joint_names], device=env.device)
    kd = torch.tensor([2.0 if "calf" in name else 1.0 for name in joint_names], device=env.device)
    arrays = {
      "action": torch.full((num_envs, cfg.sample_steps, len(joint_names)), torch.nan, device=env.device),
      "joint_pos": torch.full((num_envs, cfg.sample_steps, len(joint_names)), torch.nan, device=env.device),
      "joint_vel": torch.full((num_envs, cfg.sample_steps, len(joint_names)), torch.nan, device=env.device),
      "target": torch.full((num_envs, cfg.sample_steps, len(joint_names)), torch.nan, device=env.device),
      "force": torch.full((num_envs, cfg.sample_steps, len(joint_names)), torch.nan, device=env.device),
      "normal_force": torch.full((num_envs, cfg.sample_steps, nfeet), torch.nan, device=env.device),
      "signed_normal_force": torch.full((num_envs, cfg.sample_steps, nfeet), torch.nan, device=env.device),
      "tangent_force": torch.full((num_envs, cfg.sample_steps, nfeet), torch.nan, device=env.device),
      "slip": torch.full((num_envs, cfg.sample_steps, nfeet), torch.nan, device=env.device),
      "loaded": torch.zeros((num_envs, cfg.sample_steps, nfeet), dtype=torch.bool, device=env.device),
      "ray_valid": torch.zeros((num_envs, cfg.sample_steps, nfeet), dtype=torch.bool, device=env.device),
      "pitch": torch.full((num_envs, cfg.sample_steps), torch.nan, device=env.device),
      "body_contact": torch.zeros((num_envs, cfg.sample_steps, 3), dtype=torch.bool, device=env.device),
    }
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    sample_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    reset_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    first_reason: list[str | None] = [None] * num_envs
    loaded_contact = torch.zeros(num_envs, nfeet, dtype=torch.bool, device=env.device)
    capture: dict[str, torch.Tensor] = {}
    capture_reasons: dict[int, str] = {}
    current_action = torch.zeros(num_envs, len(joint_names), device=env.device)
    original_reset_idx = env._reset_idx

    def capture_reset(env_ids: torch.Tensor | None = None) -> None:
      nonlocal capture, capture_reasons
      if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)
      capture = _capture_reset_state(
        env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
        foot_permutation, joint_ids, force_ids, current_action, terrain_kind,
        terrain_level, loaded_contact, env_ids,
      )
      capture_reasons = {
        int(env_id): _contact_termination(env, int(env_id))
        for env_id in env_ids.tolist()
      }
      original_reset_idx(env_ids)

    env._reset_idx = capture_reset  # type: ignore[method-assign]
    command_values = torch.tensor([[s["speed"], 0.0, 0.0] for s in scenarios], device=env.device)
    command_term.vel_command_b[:] = command_values
    observation = wrapped.get_observations()
    for step in range(cfg.warmup_steps + cfg.sample_steps):
      command_term.vel_command_b[:] = command_values
      with torch.inference_mode():
        action = policy(observation)
      current_action = action.detach().clone()
      capture = {}
      capture_reasons = {}
      _, _, dones, _ = wrapped.step(action)
      observation = wrapped.get_observations()
      reset = dones.bool()
      post = _snapshot(
        env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
        foot_permutation, joint_ids, force_ids, action, terrain_kind,
        terrain_level, loaded_contact,
      )
      state = {key: value.clone() for key, value in post.items()}
      if capture:
        reset_ids = torch.where(reset)[0]
        for key, value in capture.items():
          state[key][reset_ids] = value
      if step < cfg.warmup_steps:
        loaded_contact = torch.where(reset[:, None], torch.zeros_like(loaded_contact), state["loaded"])
        active &= ~reset
        continue
      k = step - cfg.warmup_steps
      write = active.clone()
      for key, value in state.items():
        if value.ndim == 1:
          arrays[key][:, k] = torch.where(write, value, arrays[key][:, k])
        elif value.ndim == 2:
          arrays[key][:, k] = torch.where(write[:, None], value, arrays[key][:, k])
        else:
          raise RuntimeError(f"unexpected snapshot rank for {key}: {value.ndim}")
      sample_count += active.long()
      reset_count += (reset & active).long()
      for env_id in torch.where(reset & active)[0].tolist():
        first_reason[env_id] = capture_reasons.get(env_id, "reset")
      loaded_contact = torch.where(reset[:, None], torch.zeros_like(loaded_contact), state["loaded"])
      active &= ~reset
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
      count = int(sample_count[index])
      failed = first_reason[index] is not None
      windows: dict[str, Any] = {}
      requested_windows = cfg.failure_windows if failed else (cfg.stable_tail_steps,)
      for requested in requested_windows:
        eligible, start, end, reason = _window_indices(
          count, failed, requested, cfg.stable_tail_steps
        )
        key = f"failure_last_{requested}" if failed else "stable_rollout"
        if not eligible:
          windows[key] = {"eligible": False, "requested_steps": requested, "reason": reason}
        else:
          windows[key] = _window_metrics(
            arrays, index, start, end, joint_names, hard_limits, soft_limits,
            effort_limits, kp, kd, agent_cfg.clip_actions,
          )
          windows[key]["requested_steps"] = requested
      rows.append({
        **scenario,
        "checkpoint": str(Path(cfg.checkpoint).resolve()),
        "sample_count": count,
        "reset_count": int(reset_count[index]),
        "failed": failed,
        "first_failure_reason": first_reason[index],
        "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]),
        "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]),
        "windows": windows,
      })
    return {
      "terrain_condition": condition,
      "terrain_kind": terrain_kind,
      "terrain_level": terrain_level,
      "control_dt": control_dt,
      "joint_names": joint_names,
      "actuator_force_indices": [int(x) for x in force_ids.detach().cpu().tolist()],
      "actuator_limits": {
        "hip": {"kp": 20.0, "kd": 1.0, "effort_limit_Nm": 23.5},
        "thigh": {"kp": 20.0, "kd": 1.0, "effort_limit_Nm": 23.5},
        "calf": {"kp": 40.0, "kd": 2.0, "effort_limit_Nm": 45.0},
      },
      "scenarios": rows,
    }
  finally:
    env.close()


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  failed = [row for row in rows if row["failed"]]
  missing = [
    {
      "terrain_condition": row["terrain_condition"],
      "speed": row["speed"],
      "matched_slot": row["matched_slot"],
      "sample_count": row["sample_count"],
    }
    for row in failed
    if not row["windows"].get("failure_last_50", {}).get("eligible", False)
  ]
  required_stable = {
    "flat_vx_0.3": sum(
      not row["failed"]
      for row in rows
      if row["terrain_condition"] == "flat" and math.isclose(row["speed"], 0.3)
    ),
    "flat_vx_0.5": sum(
      not row["failed"]
      for row in rows
      if row["terrain_condition"] == "flat" and math.isclose(row["speed"], 0.5)
    ),
    "slope_up_high_vx_0.3": sum(
      not row["failed"]
      for row in rows
      if row["terrain_condition"] == "slope_up_high" and math.isclose(row["speed"], 0.3)
    ),
  }
  stable_shortfall = {
    name: count for name, count in required_stable.items() if count < 4
  }
  sat_evidence: list[dict[str, Any]] = []
  for row in rows:
    for window_name, window in row["windows"].items():
      if not window.get("eligible"):
        continue
      for joint, metrics in window["per_joint"].items():
        if metrics["effort_saturation_fraction"] >= 0.10 or metrics["effort_saturation_longest_run"] >= 3:
          sat_evidence.append({
            "terrain_condition": row["terrain_condition"],
            "speed": row["speed"],
            "matched_slot": row["matched_slot"],
            "window": window_name,
            "joint": joint,
            "metrics": metrics,
          })
  if sat_evidence:
    verdict = "SATURATION_CONFIRMED"
    reason = "persistent actuator force saturation met the declared 3-step or 10% window rule"
  elif missing or stable_shortfall:
    verdict = "INCONCLUSIVE"
    reason = "required stable controls or complete 50-step failure windows are missing"
  else:
    verdict = "SATURATION_NOT_CONFIRMED"
    reason = "no eligible stable/failure window met the persistent saturation rule"
  return {
    "verdict": verdict,
    "reason": reason,
    "failed_rows": len(failed),
    "required_stable_counts": required_stable,
    "stable_count_shortfall": stable_shortfall,
    "missing_failure_window_slots": missing,
    "saturation_evidence": sat_evidence,
  }


def evaluate(cfg: AuditConfig) -> dict[str, Any]:
  _validate_config(cfg)
  configure_torch_backends()
  condition_results = []
  for condition, terrain_kind, terrain_level in TERRAIN_CONDITIONS:
    condition_results.append(_audit_condition(cfg, condition, terrain_kind, terrain_level))
  rows = [row for result in condition_results for row in result["scenarios"]]
  source = Path(__file__).resolve()
  checkpoint = Path(cfg.checkpoint).resolve()
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_high_slope_actuator_audit",
    "git_head": _git_head(),
    "evaluator_source": str(source),
    "evaluator_source_sha256": _sha256(source),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint),
    "task_id": cfg.task_id,
    "config": {
      **asdict(cfg),
      "checkpoint": str(checkpoint),
      "matrix_identity": "same 48 matched slots as high_slope_gait_diagnostics_clean_seed42_48slots_1200steps_v2.json",
      "contact_thresholds_N": {"load_on": FOOT_LOAD_ON_N, "load_off": FOOT_LOAD_OFF_N},
      "force_source": "robot.data.actuator_force (actuation-space scalar; not joint_torques)",
      "force_sign": "foot-to-terrain sensor force is terrain-directed; loaded upward-normal projection is expected negative",
      "stable_window_definition": "last stable_tail_steps post-warmup control steps of a no-reset attempt",
      "failure_window_definition": "last requested active control steps ending at the pre-reset terminal snapshot",
    },
    "actuator_identity": {
      "action_type": "joint_position",
      "raw_action_scale": 0.25,
      "raw_action_offset": "default_joint_pos",
      "joint_names": condition_results[0]["joint_names"],
      "limits": condition_results[0]["actuator_limits"],
      "hard_position_limits_source": "robot.data.joint_pos_limits",
      "soft_position_limits_source": "robot.data.soft_joint_pos_limits",
      "raw_action_clip_bound": None,
    },
    "conditions": condition_results,
    "gate1": _gate(rows),
  }
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  cfg = tyro.cli(AuditConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
