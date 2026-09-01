"""Paired evaluation-only actuator headroom counterfactual for Go2 V7."""

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

from scripts.audit_go2_high_slope_actuators import (
  FORBIDDEN_CHECKPOINTS,
  V7_CHECKPOINT,
  _group_for_joint,
  _longest_true_run,
  _snapshot,
  _window_metrics as _base_window_metrics,
)
from scripts.diagnose_go2_high_slope_gait import (
  FOOT_NAMES,
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


BASELINE_AUDIT = (
  "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_"
  "focus_probe_2048env_500iter/high_slope_actuator_audit_clean_seed42_48slots_"
  "1200steps_v1.json"
)
ARM_NAMES = ("control", "headroom")
ARM_MULTIPLIERS = (1.0, 1.25)


@dataclass(frozen=True)
class HeadroomConfig:
  checkpoint: str = V7_CHECKPOINT
  baseline_audit: str = BASELINE_AUDIT
  task_id: str = "Unitree-Go2-Rough-V7"
  profile: str = "clean"
  multipliers: tuple[float, ...] = ARM_MULTIPLIERS
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  stable_tail_steps: int = 300
  failure_windows: tuple[int, ...] = (50, 100)
  seed: int = 42
  device: str = "cuda:0"
  output_file: str = "go2_actuator_headroom_counterfactual.json"


def _validate_config(cfg: HeadroomConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(V7_CHECKPOINT).resolve() or checkpoint.name in FORBIDDEN_CHECKPOINTS:
    raise ValueError("the evaluator is locked to V7 model_13600.pt")
  if cfg.task_id != "Unitree-Go2-Rough-V7" or cfg.profile != "clean":
    raise ValueError("task_id/profile are locked to V7 clean")
  if cfg.multipliers != ARM_MULTIPLIERS:
    raise ValueError("multipliers are locked to control=1.00/headroom=1.25")
  if cfg.speeds != (0.3, 0.5):
    raise ValueError("speeds are locked to 0.3 and 0.5 m/s")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("invalid repeats/warmup/sample_steps")
  if cfg.stable_tail_steps <= 0 or cfg.stable_tail_steps > cfg.sample_steps:
    raise ValueError("stable_tail_steps must be within sample_steps")
  if cfg.failure_windows != (50, 100):
    raise ValueError("failure windows are locked to 50 and 100")


def _sha_tensor(value: torch.Tensor) -> str:
  return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _physical_scenarios(
  cfg: HeadroomConfig, condition: str, terrain_kind: str, terrain_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  base = _scenario_slots(cfg, condition, terrain_kind, terrain_level)
  physical: list[dict[str, Any]] = []
  for row in base:
    for arm_index, (arm, multiplier) in enumerate(zip(ARM_NAMES, ARM_MULTIPLIERS, strict=True)):
      physical.append({
        **row,
        "arm": arm,
        "arm_index": arm_index,
        "effort_limit_multiplier": multiplier,
        "physical_env": len(physical),
      })
  return base, physical


def _joint_control_mapping(robot: Any, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
  device = robot.data.joint_pos.device
  local = torch.full((len(joint_names),), -1, dtype=torch.long, device=device)
  global_ids = torch.full_like(local, -1)
  for actuator in robot.actuators:
    for name, ctrl_id, global_id in zip(
      actuator.target_names,
      actuator.ctrl_ids.tolist(),
      actuator.global_ctrl_ids.tolist(),
      strict=True,
    ):
      if name in joint_names:
        index = joint_names.index(name)
        local[index] = int(ctrl_id)
        global_ids[index] = int(global_id)
  if (local < 0).any() or (global_ids < 0).any():
    raise RuntimeError("incomplete joint/control mapping")
  return local, global_ids


def _apply_runtime_limits(
  env: ManagerBasedRlEnv,
  robot: Any,
  global_ids: torch.Tensor,
  physical: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
  if "motor_strength" in env.cfg.events:
    raise RuntimeError("clean evaluator must remove motor_strength")
  env.sim.expand_model_fields(("actuator_forcerange",))
  defaults = env.sim.get_default_field("actuator_forcerange")
  ranges = env.sim.model.actuator_forcerange
  multipliers = torch.tensor(
    [row["effort_limit_multiplier"] for row in physical],
    dtype=ranges.dtype,
    device=env.device,
  )
  ranges[:, global_ids, :] = defaults[global_ids, :].unsqueeze(0) * multipliers[:, None, None]
  effective = ranges[:, global_ids, 1].clone()
  compiled = torch.as_tensor(
    env.sim.mj_model.actuator_forcerange[global_ids.detach().cpu().numpy()],
    dtype=effective.dtype,
    device=env.device,
  )
  limited = env.sim.mj_model.actuator_forcelimited[
    global_ids.detach().cpu().numpy()
  ]
  if not np.all(limited == 1):
    raise RuntimeError("all Go2 actuators must be force limited")
  expected = compiled[:, 1].unsqueeze(0) * multipliers[:, None]
  max_error = float((effective - expected).abs().max())
  if max_error > 1.0e-6:
    raise RuntimeError(f"runtime effort-limit mismatch: {max_error}")
  identity = {
    "expanded_fields": sorted(env.sim.expanded_fields),
    "compiled_default_ranges": compiled.detach().cpu().tolist(),
    "control_runtime_ranges": ranges[0, global_ids].detach().cpu().tolist(),
    "headroom_runtime_ranges": ranges[1, global_ids].detach().cpu().tolist(),
    "runtime_expected_error_max": max_error,
    "motor_strength_present": False,
  }
  return effective, identity


def _copy_pair_initial_state(
  env: ManagerBasedRlEnv,
  robot: Any,
  joint_ids: torch.Tensor,
  command_term: UniformVelocityCommand,
  command_values: torch.Tensor,
) -> dict[str, Any]:
  control_ids = torch.arange(0, env.num_envs, 2, device=env.device)
  probe_ids = control_ids + 1
  robot.write_root_link_pose_to_sim(
    robot.data.root_link_pose_w[control_ids].clone(), env_ids=probe_ids
  )
  robot.write_root_link_velocity_to_sim(
    robot.data.root_link_vel_w[control_ids].clone(), env_ids=probe_ids
  )
  robot.write_joint_state_to_sim(
    robot.data.joint_pos[control_ids][:, joint_ids].clone(),
    robot.data.joint_vel[control_ids][:, joint_ids].clone(),
    joint_ids=joint_ids,
    env_ids=probe_ids,
  )
  env.action_manager.reset()
  env.observation_manager.reset()
  command_term.vel_command_b[:] = command_values
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(dt=0.0)
  env.sim.sense()
  root_error = float(
    (robot.data.root_link_pose_w[control_ids] - robot.data.root_link_pose_w[probe_ids])
    .abs()
    .max()
  )
  velocity_error = float(
    (robot.data.root_link_vel_w[control_ids] - robot.data.root_link_vel_w[probe_ids])
    .abs()
    .max()
  )
  joint_error = float(
    (robot.data.joint_pos[control_ids][:, joint_ids] - robot.data.joint_pos[probe_ids][:, joint_ids])
    .abs()
    .max()
  )
  return {
    "root_pose_error_max": root_error,
    "root_velocity_error_max": velocity_error,
    "joint_position_error_max": joint_error,
    "control_state_sha256": _sha_tensor(torch.cat((
      robot.data.root_link_pose_w[control_ids],
      robot.data.root_link_vel_w[control_ids],
      robot.data.joint_pos[control_ids][:, joint_ids],
      robot.data.joint_vel[control_ids][:, joint_ids],
    ), dim=-1)),
    "probe_state_sha256": _sha_tensor(torch.cat((
      robot.data.root_link_pose_w[probe_ids],
      robot.data.root_link_vel_w[probe_ids],
      robot.data.joint_pos[probe_ids][:, joint_ids],
      robot.data.joint_vel[probe_ids][:, joint_ids],
    ), dim=-1)),
  }


def _terrain_fallback(
  num_envs: int, num_feet: int, device: torch.device, terrain_kind: str,
  terrain_level: int,
) -> torch.Tensor:
  normal = torch.zeros(num_envs, num_feet, 3, device=device)
  normal[..., 2] = 1.0
  if terrain_kind != "flat":
    gradient = 0.4 * (0.8 if terrain_kind == "slope_up" and terrain_level == 0 else 1.0)
    if terrain_kind == "slope_down":
      gradient = -0.4
    normal[:] = torch.tensor([-gradient, 0.0, 1.0], device=device)
    normal /= torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-8)
  return normal


def _headroom_snapshot(
  env: ManagerBasedRlEnv,
  robot: Any,
  feet_sensor: Any,
  terrain_sensor: Any,
  body_sensors: dict[str, Any],
  foot_ids: torch.Tensor,
  foot_permutation: torch.Tensor,
  joint_ids: torch.Tensor,
  force_ids: torch.Tensor,
  action: torch.Tensor,
  terrain_kind: str,
  terrain_level: int,
  loaded_contact: torch.Tensor,
) -> dict[str, torch.Tensor]:
  state = _snapshot(
    env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
    foot_permutation, joint_ids, force_ids, action, terrain_kind,
    terrain_level, loaded_contact,
  )
  foot_pos = robot.data.site_pos_w[:, foot_ids, :].clone()
  fallback = _terrain_fallback(
    env.num_envs, len(FOOT_NAMES), env.device, terrain_kind, terrain_level
  )
  _, normal, ray_valid = _normal_and_clearance(terrain_sensor, foot_pos, fallback)
  state["foot_pos"] = foot_pos
  state["terrain_normal"] = normal
  state["ray_valid"] = ray_valid
  state["actual_velocity"] = torch.cat((
    robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]
  ), dim=-1).clone()
  return state


def _risk_ratio(probe: float | None, control: float | None) -> dict[str, Any]:
  if probe is None or control is None:
    return {"ratio": None, "passes_1p2x": False, "reason": "missing_metric"}
  if abs(control) <= 1.0e-12:
    return {
      "ratio": None,
      "passes_1p2x": abs(probe) <= 1.0e-12,
      "reason": "zero_baseline_requires_zero_probe",
    }
  ratio = probe / control
  return {"ratio": ratio, "passes_1p2x": ratio <= 1.2 + 1.0e-9}


def _common_prefix_risk(
  arrays: dict[str, torch.Tensor], control_id: int, probe_id: int, end: int,
) -> dict[str, Any]:
  def values(env_id: int) -> dict[str, float | None]:
    loaded = arrays["loaded"][env_id, :end]
    slip_values = arrays["slip"][env_id, :end][loaded]
    raw = arrays["action"][env_id, :end]
    action_acc = raw[2:] - 2.0 * raw[1:-1] + raw[:-2]
    return {
      "slip": None if slip_values.numel() == 0 else float(slip_values.mean()),
      "action_second_difference": None if action_acc.numel() == 0 else float(
        action_acc.abs().mean()
      ),
      "body_contact": float(
        arrays["body_contact"][env_id, :end].float().sum() / max(end, 1)
      ),
    }

  control = values(control_id)
  probe = values(probe_id)
  return {
    name: {
      **_risk_ratio(probe[name], control[name]),
      "control": control[name],
      "probe": probe[name],
      "delta": None if probe[name] is None or control[name] is None else probe[name] - control[name],
    }
    for name in control
  }


def _stats_delta(probe: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
  return {
    key: None
    if probe.get(key) is None or control.get(key) is None
    else probe[key] - control[key]
    for key in ("mean", "p95", "max")
  }


def _aligned_metric_deltas(
  probe: dict[str, Any], control: dict[str, Any], joint_names: list[str],
) -> dict[str, Any]:
  per_joint: dict[str, Any] = {}
  stat_names = (
    "raw_action_abs", "target_position", "joint_position", "position_error_abs",
    "joint_velocity_abs", "actuator_force_abs", "pd_demand_abs",
    "effort_utilization", "clip_residual_abs", "action_second_difference_abs",
  )
  for name in joint_names:
    per_joint[name] = {
      metric: _stats_delta(
        probe["per_joint"][name][metric], control["per_joint"][name][metric]
      )
      for metric in stat_names
    }
    for metric in (
      "near_effort_limit_fraction", "effort_saturation_fraction",
      "soft_limit_violation_fraction", "target_soft_limit_violation_fraction",
      "hard_limit_near_fraction",
    ):
      per_joint[name][metric] = (
        probe["per_joint"][name][metric] - control["per_joint"][name][metric]
      )
  return {
    "per_joint": per_joint,
    "base_pitch": _stats_delta(probe["base_pitch"], control["base_pitch"]),
    "stance_slip": _stats_delta(probe["stance_slip"], control["stance_slip"]),
    "foot_normal_force_abs_max": _stats_delta(
      probe["foot_normal_force_abs_max"], control["foot_normal_force_abs_max"]
    ),
    "foot_tangent_force_max": _stats_delta(
      probe["foot_tangent_force_max"], control["foot_tangent_force_max"]
    ),
    "body_contact_fraction": [
      probe["body_contact_fraction"][index] - control["body_contact_fraction"][index]
      for index in range(3)
    ],
  }


def _response_gain(command: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
  values: dict[str, Any] = {}
  for axis, name in enumerate(("vx", "vy", "wz")):
    denominator = float(command[:, axis].square().sum())
    if denominator <= 1.0e-12:
      values[name] = None
      values[f"{name}_reason"] = "no_nonzero_command_energy"
    else:
      values[name] = float((command[:, axis] * actual[:, axis]).sum() / denominator)
  return values


def _attempt_summary(
  arrays: dict[str, torch.Tensor],
  env_id: int,
  count: int,
  command: torch.Tensor,
  step_events: list[dict[str, Any]],
) -> dict[str, Any]:
  raw = arrays["action"][env_id, :count]
  action_acc = raw[2:] - 2.0 * raw[1:-1] + raw[:-2]
  loaded = arrays["loaded"][env_id, :count]
  slip = torch.where(loaded, arrays["slip"][env_id, :count], torch.nan)
  actual = arrays["actual_velocity"][env_id, :count]
  commands = command.unsqueeze(0).expand(count, -1)
  step_values = torch.tensor(
    [event["length"] for event in step_events],
    device=raw.device,
    dtype=raw.dtype,
  )
  return {
    "response_gain": _response_gain(commands, actual),
    "actual_velocity": {
      name: _finite_stats(actual[:, axis])
      for axis, name in enumerate(("vx", "vy", "wz"))
    },
    "step_length": _finite_stats(step_values),
    "complete_step_count": len(step_events),
    "stance_slip": _finite_stats(slip.reshape(-1)),
    "foot_normal_force": _finite_stats(arrays["normal_force"][env_id, :count].reshape(-1)),
    "foot_tangent_force": _finite_stats(arrays["tangent_force"][env_id, :count].reshape(-1)),
    "base_pitch": _finite_stats(arrays["pitch"][env_id, :count]),
    "action_second_difference": _finite_stats(action_acc.abs().mean(dim=-1)),
    "body_contact_count": [
      int(arrays["body_contact"][env_id, :count, index].sum()) for index in range(3)
    ],
    "sample_count": count,
  }


def _persistent_joint_metrics(
  arrays: dict[str, torch.Tensor],
  env_id: int,
  start: int,
  end: int,
  limits: torch.Tensor,
  old_limits: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
  joint_names: list[str],
) -> dict[str, Any]:
  q = arrays["joint_pos"][env_id, start:end]
  qd = arrays["joint_vel"][env_id, start:end]
  target = arrays["target"][env_id, start:end]
  force = arrays["force"][env_id, start:end]
  demand = kp * (target - q) - kd * qd
  pd_valid = arrays["pd_valid"][env_id, start:end]
  sat = pd_valid[:, None] & (force.abs() / limits.clamp_min(1.0e-8) >= 0.98) & (
    demand.abs() >= limits
  )
  engaged = force.abs() > old_limits + 1.0e-4
  pd_count = int(pd_valid.sum())
  result: dict[str, Any] = {}
  for index, name in enumerate(joint_names):
    count = int(sat[:, index].sum())
    longest = _longest_true_run(sat[:, index])
    result[name] = {
      "saturated_step_count": count,
      "saturation_fraction": count / max(pd_count, 1),
      "saturation_longest_run": longest,
      "persistent": longest >= 3 or count / max(pd_count, 1) >= 0.10,
      "pd_demand_sample_count": pd_count,
      "old_limit_exceedance_count": int(engaged[:, index].sum()),
      "old_limit_exceedance_longest_run": _longest_true_run(engaged[:, index]),
    }
  return result


def _window_metrics(
  arrays: dict[str, torch.Tensor], env_index: int, start: int, end: int,
  joint_names: list[str], hard_limits: torch.Tensor, soft_limits: torch.Tensor,
  effort_limits: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor,
  clip_bound: float | None,
) -> dict[str, Any]:
  result = _base_window_metrics(
    arrays, env_index, start, end, joint_names, hard_limits, soft_limits,
    effort_limits, kp, kd, clip_bound,
  )
  q = arrays["joint_pos"][env_index, start:end]
  qd = arrays["joint_vel"][env_index, start:end]
  target = arrays["target"][env_index, start:end]
  force = arrays["force"][env_index, start:end]
  pd_valid = arrays["pd_valid"][env_index, start:end]
  demand = kp * (target - q) - kd * qd
  limit = effort_limits.unsqueeze(0).clamp_min(1.0e-8)
  sat = pd_valid[:, None] & (force.abs() / limit >= 0.98) & (
    demand.abs() >= limit
  )
  pd_count = int(pd_valid.sum())
  for index, name in enumerate(joint_names):
    metrics = result["per_joint"][name]
    metrics["pd_demand_abs"] = _finite_stats(demand[pd_valid, index].abs())
    metrics["clip_residual_abs"] = _finite_stats(
      (demand[pd_valid, index] - force[pd_valid, index]).abs()
    )
    metrics["effort_saturation_fraction"] = (
      float(sat[:, index].sum()) / max(pd_count, 1)
    )
    metrics["effort_saturation_longest_run"] = _longest_true_run(sat[:, index])
    metrics["pd_demand_sample_count"] = pd_count
  for group in ("hip", "thigh", "calf"):
    ids = [
      i for i, name in enumerate(joint_names) if _group_for_joint(name) == group
    ]
    result["groups"][group]["effort_saturation_fraction"] = (
      float(sat[:, ids].sum()) / max(pd_count * len(ids), 1)
    )
  result["pd_demand_sample_count"] = pd_count
  result["pd_demand_terminal_samples_excluded"] = end - start - pd_count
  result["pd_demand_timing"] = (
    "terminal reset-hook row excluded because actuator_force is one physics "
    "substep older than terminal q/qd"
  )
  return result


def _window_extra(
  arrays: dict[str, torch.Tensor], env_id: int, start: int, end: int,
  command: torch.Tensor, events: list[dict[str, Any]],
) -> dict[str, Any]:
  actual = arrays["actual_velocity"][env_id, start:end]
  commands = command.unsqueeze(0).expand(end - start, -1)
  touchdown = [event for event in events if start <= event["touchdown_step"] < end]
  contained = [event for event in touchdown if event["liftoff_step"] >= start]
  values = torch.tensor(
    [event["length"] for event in touchdown],
    dtype=actual.dtype,
    device=actual.device,
  )
  contained_values = torch.tensor(
    [event["length"] for event in contained],
    dtype=actual.dtype,
    device=actual.device,
  )
  return {
    "window_start": start,
    "window_end": end,
    "response_gain": _response_gain(commands, actual),
    "step_length_by_touchdown": _finite_stats(values),
    "step_length_fully_contained": _finite_stats(contained_values),
  }


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
  n = wins + losses
  if n == 0:
    return None
  return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def _runtime_force_identity_mask(
  force: torch.Tensor, rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, int]:
  """Select force samples whose PD state shares the same forward-call phase."""
  valid = torch.isfinite(force)
  excluded = 0
  for env_id, row in enumerate(rows):
    if row["first_failure_phase"] != "sample" or row["sample_count"] <= 0:
      continue
    terminal_step = row["sample_count"] - 1
    excluded += int(valid[env_id, terminal_step].sum())
    valid[env_id, terminal_step] = False
  return valid, excluded


def _classify(
  pairs: list[dict[str, Any]],
  identity_pass: bool,
  baseline_compatible: bool,
  missing_windows: list[dict[str, Any]],
) -> dict[str, Any]:
  cohort = [
    pair for pair in pairs
    if pair["terrain_condition"] != "flat"
    and pair["control_failed"]
    and pair["aligned_primary"].get("eligible")
    and pair["aligned_primary"]["control_persistent_joint_count"] > 0
  ]
  control_steps = sum(pair["aligned_primary"]["control_saturated_steps"] for pair in cohort)
  probe_steps = sum(pair["aligned_primary"]["probe_saturated_steps"] for pair in cohort)
  saturation_reduction = None if control_steps == 0 else (control_steps - probe_steps) / control_steps
  completion_delta: dict[str, int] = {}
  for speed in (0.3, 0.5):
    cell = [p for p in pairs if p["terrain_condition"] == "slope_up_high" and math.isclose(p["speed"], speed)]
    completion_delta[f"vx_{speed:g}"] = sum(not p["probe_failed"] for p in cell) - sum(not p["control_failed"] for p in cell)
  common_survivor_improvement: dict[str, Any] = {}
  alternate_path = True
  for speed in (0.3, 0.5):
    cell = [
      p for p in pairs
      if p["terrain_condition"] == "slope_up_high"
      and math.isclose(p["speed"], speed)
      and not p["control_failed"] and not p["probe_failed"]
    ]
    cg = [p["control_gain"] for p in cell if p["control_gain"] is not None and p["probe_gain"] is not None]
    pg = [p["probe_gain"] for p in cell if p["control_gain"] is not None and p["probe_gain"] is not None]
    cs = [p["control_step"] for p in cell if p["control_step"] is not None and p["probe_step"] is not None]
    ps = [p["probe_step"] for p in cell if p["control_step"] is not None and p["probe_step"] is not None]
    gain_rel = None if not cg or abs(sum(cg) / len(cg)) <= 1.0e-12 else (
      (sum(pg) / len(pg)) / (sum(cg) / len(cg)) - 1.0
    )
    step_rel = None if not cs or abs(sum(cs) / len(cs)) <= 1.0e-12 else (
      (sum(ps) / len(ps)) / (sum(cs) / len(cs)) - 1.0
    )
    common_survivor_improvement[f"vx_{speed:g}"] = {
      "count": len(cell), "gain_relative": gain_rel, "step_relative": step_rel,
    }
    alternate_path &= gain_rel is not None and step_rel is not None and gain_rel >= 0.20 and step_rel >= 0.20
  completion_path = all(value >= 2 for value in completion_delta.values())
  risk_failures = [pair for pair in pairs if not pair["risk_guardrails_pass"]]
  new_failure_pairs = [pair for pair in pairs if not pair["control_failed"] and pair["probe_failed"]]
  control_classes = {
    pair["control_failure_reason"] for pair in pairs
    if pair["control_failure_reason"] is not None
  }
  probe_classes = {
    pair["probe_failure_reason"] for pair in pairs
    if pair["probe_failure_reason"] is not None
  }
  new_failure_classes = sorted(probe_classes - control_classes)
  guardrails = not risk_failures and not new_failure_pairs and not new_failure_classes
  wins = sum(pair["lifecycle_improved"] for pair in cohort)
  losses = sum(pair["lifecycle_worsened"] for pair in cohort)
  if not identity_pass or not baseline_compatible or missing_windows or len(cohort) < 8:
    verdict = "INCONCLUSIVE"
    reason = "identity, baseline compatibility, aligned-window coverage, or causal cohort failed"
  elif saturation_reduction is None or saturation_reduction < 0.50:
    verdict = "HEADROOM_INSUFFICIENT"
    reason = "1.25x headroom did not reduce aligned persistent saturation by 50%"
  elif (completion_path or alternate_path) and guardrails:
    verdict = "ACTUATOR_CAUSAL"
    reason = "saturation and slope-up locomotion improved without the declared risk regression"
  else:
    verdict = "SATURATION_DOWNSTREAM"
    reason = "saturation fell but locomotion did not meet the predeclared improvement/guardrail gate"
  return {
    "verdict": verdict,
    "reason": reason,
    "causal_cohort_size": len(cohort),
    "control_saturated_joint_steps": control_steps,
    "probe_saturated_joint_steps": probe_steps,
    "saturation_reduction_fraction": saturation_reduction,
    "slope_up_completion_delta": completion_delta,
    "completion_path_pass": completion_path,
    "common_survivor_improvement": common_survivor_improvement,
    "alternate_gain_step_path_pass": alternate_path,
    "risk_guardrails_pass": guardrails,
    "risk_failure_pair_count": len(risk_failures),
    "new_failure_pair_count": len(new_failure_pairs),
    "new_failure_classes": new_failure_classes,
    "lifecycle_wins": wins,
    "lifecycle_losses": losses,
    "one_sided_sign_test_pvalue": _sign_test_pvalue(wins, losses),
  }


def _run_condition(
  cfg: HeadroomConfig, condition: str, terrain_kind: str, terrain_level: int,
) -> dict[str, Any]:
  base_scenarios, physical = _physical_scenarios(cfg, condition, terrain_kind, terrain_level)
  num_envs = len(physical)
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = _make_gait_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = _configure_profile(env_cfg, cfg.profile)
  if profile_settings["startup_randomization_events"]:
    raise RuntimeError("clean profile retained startup randomization")
  episode_settings = _configure_episode_length(
    env_cfg, cfg.warmup_steps + cfg.sample_steps + 20
  )
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
  command_cfg.ranges.lin_vel_x = (0.3, 0.5)
  command_cfg.ranges.lin_vel_y = (0.0, 0.0)
  command_cfg.ranges.ang_vel_z = (0.0, 0.0)
  command_cfg.ranges.heading = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  original_reset_idx = env._reset_idx
  try:
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints((".*",), preserve_order=True)
    joint_names = list(joint_names)
    force_ids, global_ids = _joint_control_mapping(robot, joint_names)
    runtime_limits, runtime_identity = _apply_runtime_limits(
      env, robot, global_ids, physical
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(Path(cfg.checkpoint).resolve()), load_cfg={"actor": True}, strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
    placement = _assign_terrain(env, physical, env.device)
    command_term = env.command_manager.get_term("twist")
    if not isinstance(command_term, UniformVelocityCommand):
      raise TypeError("twist command term is incompatible")
    command_values = torch.tensor(
      [[row["speed"], 0.0, 0.0] for row in physical], device=env.device
    )
    initial_identity = _copy_pair_initial_state(
      env, robot, joint_ids, command_term, command_values
    )
    foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != FOOT_NAMES:
      raise RuntimeError("foot order mismatch")
    feet_sensor = env.scene["feet_ground_contact"]
    terrain_sensor = env.scene["terrain_scan"]
    sensor_names = [
      slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"
    ]
    desired_names = [f"{name}_foot_collision" for name in FOOT_NAMES]
    foot_permutation = torch.tensor(
      [sensor_names.index(name) for name in desired_names],
      dtype=torch.long, device=env.device,
    )
    body_sensors: dict[str, Any] = {}
    for key, name in {
      "base": "base_ground_contact",
      "upper_leg": "upper_leg_ground_contact",
      "calf": "calf_ground_contact",
    }.items():
      try:
        body_sensors[key] = env.scene[name]
      except KeyError:
        body_sensors[key] = None
    hard_limits = robot.data.joint_pos_limits[0, joint_ids].clone()
    soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].clone()
    kp = torch.tensor(
      [40.0 if "calf" in name else 20.0 for name in joint_names], device=env.device
    )
    kd = torch.tensor(
      [2.0 if "calf" in name else 1.0 for name in joint_names], device=env.device
    )
    nfeet = len(FOOT_NAMES)
    arrays: dict[str, torch.Tensor] = {
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
      "foot_pos": torch.full((num_envs, cfg.sample_steps, nfeet, 3), torch.nan, device=env.device),
      "terrain_normal": torch.full((num_envs, cfg.sample_steps, nfeet, 3), torch.nan, device=env.device),
      "actual_velocity": torch.full((num_envs, cfg.sample_steps, 3), torch.nan, device=env.device),
      "pd_valid": torch.zeros((num_envs, cfg.sample_steps), dtype=torch.bool, device=env.device),
    }
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    loaded_contact = torch.zeros(num_envs, nfeet, dtype=torch.bool, device=env.device)
    sample_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    reset_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    first_reason: list[str | None] = [None] * num_envs
    first_failure_phase: list[str | None] = [None] * num_envs
    first_failure_step: list[int | None] = [None] * num_envs
    capture: dict[str, torch.Tensor] = {}
    captured_ids = torch.empty(0, dtype=torch.long, device=env.device)
    capture_reasons: dict[int, str] = {}
    current_action = torch.zeros(num_envs, len(joint_names), device=env.device)
    step_events: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    prev_contact = torch.zeros(num_envs, nfeet, dtype=torch.bool, device=env.device)
    has_liftoff = torch.zeros_like(prev_contact)
    liftoff_pos = torch.zeros(num_envs, nfeet, 3, device=env.device)
    liftoff_step = torch.zeros(num_envs, nfeet, dtype=torch.long, device=env.device)

    def capture_reset(env_ids: torch.Tensor | None = None) -> None:
      nonlocal capture, captured_ids, capture_reasons
      if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)
      snapshot = _headroom_snapshot(
        env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
        foot_permutation, joint_ids, force_ids, current_action, terrain_kind,
        terrain_level, loaded_contact,
      )
      captured_ids = env_ids.clone()
      capture = {
        key: value.index_select(0, env_ids).clone() for key, value in snapshot.items()
      }
      capture_reasons = {
        int(env_id): _contact_termination(env, int(env_id))
        for env_id in env_ids.tolist()
      }
      original_reset_idx(env_ids)

    env._reset_idx = capture_reset  # type: ignore[method-assign]
    command_term.vel_command_b[:] = command_values
    observation = wrapped.get_observations()
    control_ids = torch.arange(0, num_envs, 2, device=env.device)
    probe_ids = control_ids + 1
    observation_error = float(
      (observation["actor"][control_ids] - observation["actor"][probe_ids]).abs().max()
    )
    term_errors: dict[str, float] = {}
    offset = 0
    for term_name, term_dim in zip(
      env.observation_manager.active_terms["actor"],
      env.observation_manager.group_obs_term_dim["actor"],
      strict=True,
    ):
      width = math.prod(term_dim)
      term_errors[term_name] = float(
        (
          observation["actor"][control_ids, offset : offset + width]
          - observation["actor"][probe_ids, offset : offset + width]
        ).abs().max()
      )
      offset += width
    with torch.inference_mode():
      first_action = policy(observation)
    action_error = float(
      (first_action[control_ids] - first_action[probe_ids]).abs().max()
    )
    initial_identity.update({
      "actor_observation_error_max": observation_error,
      "actor_observation_term_error_max": term_errors,
      "first_policy_action_error_max": action_error,
      "pairing_pass": max(
        initial_identity["root_pose_error_max"],
        initial_identity["root_velocity_error_max"],
        initial_identity["joint_position_error_max"],
        observation_error,
        action_error,
      ) <= 1.0e-6,
    })
    if not initial_identity["pairing_pass"]:
      raise RuntimeError(f"initial pair identity failed: {initial_identity}")

    for step in range(cfg.warmup_steps + cfg.sample_steps):
      command_term.vel_command_b[:] = command_values
      with torch.inference_mode():
        action = policy(observation)
      current_action = action.detach().clone()
      capture = {}
      captured_ids = torch.empty(0, dtype=torch.long, device=env.device)
      capture_reasons = {}
      _, _, dones, _ = wrapped.step(action)
      observation = wrapped.get_observations()
      reset = dones.bool()
      state = _headroom_snapshot(
        env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
        foot_permutation, joint_ids, force_ids, action, terrain_kind,
        terrain_level, loaded_contact,
      )
      if capture:
        for key, values in capture.items():
          state[key][captured_ids] = values
      if step < cfg.warmup_steps:
        for env_id in torch.where(reset & active)[0].tolist():
          first_reason[env_id] = capture_reasons.get(env_id, "reset")
          first_failure_phase[env_id] = "warmup"
          first_failure_step[env_id] = step
        loaded_contact = torch.where(
          reset[:, None], torch.zeros_like(loaded_contact), state["loaded"]
        )
        active &= ~reset
        continue
      k = step - cfg.warmup_steps
      write = active.clone()
      for key, value in state.items():
        if value.ndim == 1:
          arrays[key][:, k] = torch.where(write, value, arrays[key][:, k])
        elif value.ndim == 2:
          arrays[key][:, k] = torch.where(write[:, None], value, arrays[key][:, k])
        elif value.ndim == 3:
          arrays[key][:, k] = torch.where(
            write[:, None, None], value, arrays[key][:, k]
          )
        else:
          raise RuntimeError(f"unexpected snapshot rank for {key}: {value.ndim}")
      arrays["pd_valid"][:, k] = write & ~reset
      if k == 0:
        prev_contact = state["loaded"].clone()
        liftoff_pos = state["foot_pos"].clone()
      transition = write[:, None] & (state["loaded"] != prev_contact)
      liftoff = transition & ~state["loaded"]
      touchdown = transition & state["loaded"]
      liftoff_pos = torch.where(liftoff[..., None], state["foot_pos"], liftoff_pos)
      liftoff_step = torch.where(liftoff, torch.full_like(liftoff_step, k), liftoff_step)
      has_liftoff |= liftoff
      for env_id, foot_id in torch.nonzero(touchdown, as_tuple=False).tolist():
        if has_liftoff[env_id, foot_id] and state["ray_valid"][env_id, foot_id]:
          normal = state["terrain_normal"][env_id, foot_id]
          forward = torch.tensor([1.0, 0.0, 0.0], device=env.device)
          tangent = forward - (forward * normal).sum() * normal
          tangent /= torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
          length = float(
            ((state["foot_pos"][env_id, foot_id] - liftoff_pos[env_id, foot_id]) * tangent).sum()
          )
          step_events[env_id].append({
            "foot": FOOT_NAMES[foot_id],
            "liftoff_step": int(liftoff_step[env_id, foot_id]),
            "touchdown_step": k,
            "length": length,
          })
      has_liftoff &= ~touchdown
      prev_contact = torch.where(write[:, None], state["loaded"], prev_contact)
      sample_count += active.long()
      reset_count += (reset & active).long()
      for env_id in torch.where(reset & active)[0].tolist():
        first_reason[env_id] = capture_reasons.get(env_id, "reset")
        first_failure_phase[env_id] = "sample"
        first_failure_step[env_id] = k
      loaded_contact = torch.where(
        reset[:, None], torch.zeros_like(loaded_contact), state["loaded"]
      )
      active &= ~reset

    rows: list[dict[str, Any]] = []
    old_limits = runtime_limits[0]
    for env_id, scenario in enumerate(physical):
      count = int(sample_count[env_id])
      failed = first_reason[env_id] is not None
      windows: dict[str, Any] = {}
      if first_failure_phase[env_id] == "warmup":
        for size in cfg.failure_windows:
          windows[f"failure_last_{size}"] = {
            "eligible": False, "reason": "failed_during_warmup", "requested_steps": size,
          }
      elif failed:
        for size in cfg.failure_windows:
          key = f"failure_last_{size}"
          if count < size:
            windows[key] = {
              "eligible": False, "reason": "insufficient_failure_window",
              "requested_steps": size,
            }
          else:
            start, end = count - size, count
            window = _window_metrics(
              arrays, env_id, start, end, joint_names, hard_limits, soft_limits,
              runtime_limits[env_id], kp, kd, agent_cfg.clip_actions,
            )
            window.update(_window_extra(
              arrays, env_id, start, end, command_values[env_id], step_events[env_id]
            ))
            window["persistent_joints"] = _persistent_joint_metrics(
              arrays, env_id, start, end, runtime_limits[env_id], old_limits,
              kp, kd, joint_names,
            )
            windows[key] = window
      elif count == cfg.sample_steps:
        start, end = count - cfg.stable_tail_steps, count
        window = _window_metrics(
          arrays, env_id, start, end, joint_names, hard_limits, soft_limits,
          runtime_limits[env_id], kp, kd, agent_cfg.clip_actions,
        )
        window.update(_window_extra(
          arrays, env_id, start, end, command_values[env_id], step_events[env_id]
        ))
        window["persistent_joints"] = _persistent_joint_metrics(
          arrays, env_id, start, end, runtime_limits[env_id], old_limits,
          kp, kd, joint_names,
        )
        windows["stable_tail_300"] = window
      else:
        windows["stable_tail_300"] = {
          "eligible": False, "reason": "incomplete_nonfailed_attempt",
        }
      summary = _attempt_summary(
        arrays, env_id, count, command_values[env_id], step_events[env_id]
      ) if count > 0 else {
        "response_gain": {"vx": None, "vx_reason": "no_samples"},
        "step_length": _finite_stats(torch.empty(0, device=env.device)),
        "complete_step_count": 0,
        "sample_count": 0,
      }
      rows.append({
        **scenario,
        "sample_count": count,
        "reset_count": int(reset_count[env_id]),
        "failed": failed,
        "first_failure_reason": first_reason[env_id],
        "first_failure_phase": first_failure_phase[env_id],
        "first_failure_step": first_failure_step[env_id],
        "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]),
        "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]),
        "attempt_summary": summary,
        "windows": windows,
      })

    pairs: list[dict[str, Any]] = []
    for base_index, base in enumerate(base_scenarios):
      control_id, probe_id = 2 * base_index, 2 * base_index + 1
      control, probe = rows[control_id], rows[probe_id]
      aligned: dict[str, Any] = {}
      if control["failed"] and control["first_failure_phase"] == "sample":
        for size, label in ((50, "confirmation"), (100, "primary")):
          end = control["sample_count"]
          start = end - size
          key = f"aligned_{label}"
          if start < 0 or probe["sample_count"] < end:
            aligned[key] = {
              "eligible": False,
              "reason": "aligned_window_not_covered_by_both_arms",
              "window_start": max(start, 0), "window_end": end,
            }
            continue
          control_persistent = _persistent_joint_metrics(
            arrays, control_id, start, end, runtime_limits[control_id], old_limits,
            kp, kd, joint_names,
          )
          probe_persistent = _persistent_joint_metrics(
            arrays, probe_id, start, end, runtime_limits[probe_id], old_limits,
            kp, kd, joint_names,
          )
          implicated = [name for name in joint_names if control_persistent[name]["persistent"]]
          control_window = _window_metrics(
            arrays, control_id, start, end, joint_names, hard_limits, soft_limits,
            runtime_limits[control_id], kp, kd, agent_cfg.clip_actions,
          )
          probe_window = _window_metrics(
            arrays, probe_id, start, end, joint_names, hard_limits, soft_limits,
            runtime_limits[probe_id], kp, kd, agent_cfg.clip_actions,
          )
          aligned[key] = {
            "eligible": True,
            "window_start": start,
            "window_end": end,
            "control_persistent_joint_count": len(implicated),
            "implicated_joints": implicated,
            "control_saturated_steps": sum(control_persistent[name]["saturated_step_count"] for name in implicated),
            "probe_saturated_steps": sum(probe_persistent[name]["saturated_step_count"] for name in implicated),
            "headroom_engaged": any(
              probe_persistent[name]["old_limit_exceedance_longest_run"] >= 3
              for name in implicated
            ),
            "control": control_persistent,
            "probe": probe_persistent,
            "control_window": control_window,
            "probe_window": probe_window,
            "paired_metric_deltas": _aligned_metric_deltas(
              probe_window, control_window, joint_names
            ),
          }
      common = min(control["sample_count"], probe["sample_count"])
      control_summary = control["attempt_summary"]
      probe_summary = probe["attempt_summary"]
      risk = _common_prefix_risk(arrays, control_id, probe_id, common)
      severity = {
        None: 0, "illegal_calf_contact": 1, "illegal_upper_leg_contact": 2,
        "illegal_base_contact": 3, "fell_over": 3,
      }
      lifecycle_improved = control["failed"] and (
        not probe["failed"]
        or (
          probe["sample_count"] - control["sample_count"]
          >= max(100, math.ceil(0.20 * control["sample_count"]))
          and severity.get(probe["first_failure_reason"], 99)
          <= severity.get(control["first_failure_reason"], 99)
        )
      )
      pairs.append({
        **base,
        "control_failed": control["failed"],
        "probe_failed": probe["failed"],
        "control_failure_reason": control["first_failure_reason"],
        "probe_failure_reason": probe["first_failure_reason"],
        "control_sample_count": control["sample_count"],
        "probe_sample_count": probe["sample_count"],
        "common_prefix_steps": common,
        "control_gain": control_summary.get("response_gain", {}).get("vx"),
        "probe_gain": probe_summary.get("response_gain", {}).get("vx"),
        "control_step": control_summary.get("step_length", {}).get("mean"),
        "probe_step": probe_summary.get("step_length", {}).get("mean"),
        "gain_delta": None if control_summary.get("response_gain", {}).get("vx") is None or probe_summary.get("response_gain", {}).get("vx") is None else probe_summary["response_gain"]["vx"] - control_summary["response_gain"]["vx"],
        "step_delta": None if control_summary.get("step_length", {}).get("mean") is None or probe_summary.get("step_length", {}).get("mean") is None else probe_summary["step_length"]["mean"] - control_summary["step_length"]["mean"],
        "risk": risk,
        "risk_guardrails_pass": all(item["passes_1p2x"] for item in risk.values()),
        "lifecycle_improved": lifecycle_improved,
        "lifecycle_worsened": (not control["failed"] and probe["failed"])
        or probe["sample_count"] < control["sample_count"],
        "aligned_confirmation": aligned.get("aligned_confirmation", {"eligible": False, "reason": "control_not_failed"}),
        "aligned_primary": aligned.get("aligned_primary", {"eligible": False, "reason": "control_not_failed"}),
      })
    demand = kp * (arrays["target"] - arrays["joint_pos"]) - kd * arrays["joint_vel"]
    expected_force = torch.clamp(
      demand,
      min=-runtime_limits[:, None, :],
      max=runtime_limits[:, None, :],
    )
    # The reset hook captures the terminal row before ManagerBasedRlEnv.step()
    # performs its final sim.forward(). Its q/qd are post-integration while
    # actuator_force is still from the preceding substep, so that row cannot
    # be used to reconstruct PD clipping. It remains part of all diagnostics
    # and of the independent force-within-runtime-limit check below.
    valid_force = torch.isfinite(arrays["force"]) & arrays["pd_valid"][:, :, None]
    excluded_terminal_values = int(
      (torch.isfinite(arrays["force"]) & ~arrays["pd_valid"][:, :, None]).sum()
    )
    clamp_error = float(
      (arrays["force"][valid_force] - expected_force[valid_force]).abs().max()
    ) if valid_force.any() else None
    all_valid_force = torch.isfinite(arrays["force"])
    force_limit_violation = float(
      (arrays["force"].abs() - runtime_limits[:, None, :])
      .masked_fill(~all_valid_force, -torch.inf)
      .max()
    ) if all_valid_force.any() else None
    runtime_ranges_end = env.sim.model.actuator_forcerange[:, global_ids, :]
    expected_ranges_end = torch.stack((-runtime_limits, runtime_limits), dim=-1)
    runtime_range_drift = float(
      (runtime_ranges_end - expected_ranges_end).abs().max()
    )
    demand_exceeds = valid_force & (
      demand.abs() >= runtime_limits[:, None, :]
    )
    force_at_limit = valid_force & (
      (arrays["force"].abs() - runtime_limits[:, None, :]).abs() <= 1.0e-3
    )
    clipping_observation: dict[str, Any] = {}
    for arm, env_ids in {
      "control": torch.arange(0, num_envs, 2, device=env.device),
      "headroom": torch.arange(1, num_envs, 2, device=env.device),
    }.items():
      exceeds = demand_exceeds[env_ids]
      count = int(exceeds.sum())
      clipping_observation[arm] = {
        "demand_exceeds_limit_count": count,
        "force_at_limit_when_demand_exceeds_count": int(
          (exceeds & force_at_limit[env_ids]).sum()
        ),
        "status": "dynamically_exercised" if count else "not_exercised",
      }
    runtime_identity.update({
      "actuator_force_clamp_error_max": clamp_error,
      "actuator_force_clamp_identity_sample_count": int(valid_force.sum()),
      "actuator_force_clamp_terminal_values_excluded": excluded_terminal_values,
      "actuator_force_clamp_timing": (
        "nonterminal post-step samples after sim.forward; terminal reset-hook "
        "samples excluded from PD reconstruction because actuator_force is "
        "one physics substep stale"
      ),
      "actuator_force_limit_excess_max": force_limit_violation,
      "runtime_range_drift_max": runtime_range_drift,
      "actual_clipping_observation": clipping_observation,
      "runtime_clamp_pass": clamp_error is not None and clamp_error <= 1.0e-3
      and force_limit_violation is not None and force_limit_violation <= 1.0e-4
      and runtime_range_drift <= 1.0e-6,
      "joint_names": joint_names,
      "local_ctrl_ids": force_ids.detach().cpu().tolist(),
      "global_ctrl_ids": global_ids.detach().cpu().tolist(),
    })
    return {
      "terrain_condition": condition,
      "terrain_kind": terrain_kind,
      "terrain_level": terrain_level,
      "profile_settings": profile_settings,
      "episode_settings": episode_settings,
      "runtime_identity": runtime_identity,
      "initial_pair_identity": initial_identity,
      "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]),
      "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]),
      "rows": rows,
      "pairs": pairs,
    }
  finally:
    env._reset_idx = original_reset_idx  # type: ignore[method-assign]
    env.close()


def _baseline_compatibility(
  baseline: dict[str, Any], conditions: list[dict[str, Any]],
) -> dict[str, Any]:
  old: dict[tuple[str, float], int] = {}
  for condition in baseline["conditions"]:
    for speed in (0.3, 0.5):
      old[(condition["terrain_condition"], speed)] = sum(
        not row["failed"] for row in condition["scenarios"]
        if math.isclose(row["speed"], speed)
      )
  new: dict[tuple[str, float], int] = {}
  for condition in conditions:
    for speed in (0.3, 0.5):
      new[(condition["terrain_condition"], speed)] = sum(
        not row["failed"] for row in condition["rows"]
        if row["arm"] == "control" and math.isclose(row["speed"], speed)
      )
  deltas = {
    f"{condition}_vx_{speed:g}": new[(condition, speed)] - value
    for (condition, speed), value in old.items()
  }
  return {
    "formal_v1_success_counts": {f"{c}_vx_{s:g}": v for (c, s), v in old.items()},
    "paired_control_success_counts": {f"{c}_vx_{s:g}": v for (c, s), v in new.items()},
    "success_count_deltas": deltas,
    "allowed_absolute_delta_per_cell": 2,
    "passes": all(abs(value) <= 2 for value in deltas.values()),
  }


def evaluate(cfg: HeadroomConfig) -> dict[str, Any]:
  _validate_config(cfg)
  configure_torch_backends()
  baseline_path = Path(cfg.baseline_audit).expanduser().resolve()
  baseline = json.loads(baseline_path.read_text())
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if baseline.get("checkpoint_sha256") != _sha256(checkpoint):
    raise RuntimeError("baseline/checkpoint identity mismatch")
  conditions = [
    _run_condition(cfg, condition, terrain_kind, terrain_level)
    for condition, terrain_kind, terrain_level in TERRAIN_CONDITIONS
  ]
  pairs = [pair for condition in conditions for pair in condition["pairs"]]
  compatibility = _baseline_compatibility(baseline, conditions)
  identity_pass = all(
    condition["runtime_identity"]["runtime_clamp_pass"]
    and condition["initial_pair_identity"]["pairing_pass"]
    and condition["terrain_assignment_position_error_max"] < 1.0e-4
    and condition["terrain_placement_position_error_max"] < 1.0e-4
    for condition in conditions
  )
  missing_windows = [
    {
      "terrain_condition": pair["terrain_condition"],
      "speed": pair["speed"],
      "matched_slot": pair["matched_slot"],
      "reason": pair["aligned_primary"].get("reason"),
    }
    for pair in pairs
    if pair["control_failed"] and not pair["aligned_primary"].get("eligible")
  ]
  for condition in conditions:
    for row in condition["rows"]:
      if not row["failed"]:
        continue
      for size in cfg.failure_windows:
        key = f"failure_last_{size}"
        if not row["windows"].get(key, {}).get("eligible"):
          missing_windows.append({
            "terrain_condition": row["terrain_condition"],
            "speed": row["speed"],
            "matched_slot": row["matched_slot"],
            "arm": row["arm"],
            "reason": row["windows"].get(key, {}).get("reason", "missing_window"),
            "window": key,
          })
  source = Path(__file__).resolve()
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_actuator_headroom_counterfactual",
    "git_head": _git_head(),
    "evaluator_source": str(source),
    "evaluator_source_sha256": _sha256(source),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint),
    "baseline_audit": str(baseline_path),
    "baseline_audit_sha256": _sha256(baseline_path),
    "config": asdict(cfg),
    "single_variable_contract": {
      "variable": "per-world actuator_forcerange multiplier",
      "control": 1.0,
      "headroom": 1.25,
      "unchanged": [
        "checkpoint", "policy", "terrain", "command", "termination", "gait",
        "observation", "reward", "network", "PPO",
      ],
    },
    "conditions": conditions,
    "baseline_compatibility": compatibility,
    "identity_gate_pass": identity_pass,
    "missing_aligned_primary_windows": missing_windows,
  }
  payload["causal_decision"] = _classify(
    pairs, identity_pass, compatibility["passes"], missing_windows
  )
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  cfg = tyro.cli(HeadroomConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps({
    "output": str(output),
    "causal_decision": payload["causal_decision"],
    "baseline_compatibility": payload["baseline_compatibility"],
    "identity_gate_pass": payload["identity_gate_pass"],
  }, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
