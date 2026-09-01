"""Event-triggered actuator-headroom counterfactual for Go2 V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_go2_actuator_headroom_counterfactual as base
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


BASELINE_COUNTERFACTUAL = (
  "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_"
  "focus_probe_2048env_500iter/actuator_headroom_counterfactual_clean_seed42_"
  "96worlds_1200steps_v2.json"
)
ARMS = ("control", "probe")
MULTIPLIERS = (1.0, 1.25)
TRIGGER_CONDITIONS = tuple(
  row for row in base.TERRAIN_CONDITIONS
  if row[0] in ("flat", "slope_up_high")
)


@dataclass(frozen=True)
class TriggeredConfig:
  checkpoint: str = base.V7_CHECKPOINT
  baseline_counterfactual: str = BASELINE_COUNTERFACTUAL
  task_id: str = "Unitree-Go2-Rough-V7"
  profile: str = "clean"
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  post_windows: tuple[int, ...] = (50, 100, 300)
  trigger_run_steps: int = 3
  saturation_threshold: float = 0.98
  seed: int = 42
  device: str = "cuda:0"
  forced_trigger_step: int | None = None
  output_file: str = "go2_actuator_headroom_triggered.json"


def _validate_config(cfg: TriggeredConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(base.V7_CHECKPOINT).resolve():
    raise ValueError("the evaluator is locked to V7 model_13600.pt")
  if checkpoint.name in base.FORBIDDEN_CHECKPOINTS:
    raise ValueError("forbidden checkpoint")
  if cfg.task_id != "Unitree-Go2-Rough-V7" or cfg.profile != "clean":
    raise ValueError("task/profile are locked to V7 clean")
  if cfg.speeds != (0.3, 0.5):
    raise ValueError("speeds are locked to 0.3 and 0.5 m/s")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("invalid repeats/warmup/sample_steps")
  if cfg.post_windows != (50, 100, 300):
    raise ValueError("post windows are locked to 50, 100 and 300")
  if cfg.trigger_run_steps != 3 or not math.isclose(cfg.saturation_threshold, 0.98):
    raise ValueError("trigger is locked to three consecutive 0.98-limit steps")
  if max(cfg.post_windows) > cfg.sample_steps:
    raise ValueError("sample_steps must cover the primary post window")
  if cfg.forced_trigger_step is not None and not (
    2 <= cfg.forced_trigger_step < cfg.sample_steps - 1
  ):
    raise ValueError("forced trigger must leave a following intervention step")


def _physical_scenarios(
  cfg: TriggeredConfig, condition: str, terrain_kind: str, terrain_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  base_rows = base._scenario_slots(cfg, condition, terrain_kind, terrain_level)
  physical: list[dict[str, Any]] = []
  for row in base_rows:
    for arm_index, arm in enumerate(ARMS):
      physical.append({
        **row,
        "arm": arm,
        "arm_index": arm_index,
        "initial_effort_limit_multiplier": 1.0,
        "physical_env": len(physical),
      })
  return base_rows, physical


def _prepare_limits(
  env: base.ManagerBasedRlEnv, global_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
  if "motor_strength" in env.cfg.events:
    raise RuntimeError("clean evaluator retained motor_strength")
  env.sim.expand_model_fields(("actuator_forcerange",))
  defaults = env.sim.get_default_field("actuator_forcerange")
  ranges = env.sim.model.actuator_forcerange
  ranges[:, global_ids, :] = defaults[global_ids, :].unsqueeze(0)
  compiled = torch.as_tensor(
    env.sim.mj_model.actuator_forcerange[global_ids.detach().cpu().numpy()],
    dtype=ranges.dtype,
    device=env.device,
  )
  limited = env.sim.mj_model.actuator_forcelimited[
    global_ids.detach().cpu().numpy()
  ]
  if not np.all(limited == 1):
    raise RuntimeError("all Go2 actuators must be force limited")
  baseline_limits = compiled[:, 1].clone()
  active_limits = baseline_limits.unsqueeze(0).repeat(env.num_envs, 1)
  initial_error = float(
    (ranges[:, global_ids, :] - compiled.unsqueeze(0)).abs().max()
  )
  if initial_error > 1.0e-6:
    raise RuntimeError(f"initial runtime range mismatch: {initial_error}")
  return baseline_limits, active_limits, ranges, {
    "expanded_fields": sorted(env.sim.expanded_fields),
    "compiled_default_ranges": compiled.detach().cpu().tolist(),
    "initial_all_world_range_error_max": initial_error,
    "motor_strength_present": False,
  }


def _copy_rows(value: torch.Tensor, controls: torch.Tensor, probes: torch.Tensor) -> None:
  if value.ndim > 0 and value.shape[0] > int(probes.max()):
    value[probes] = value[controls].clone()


def _copy_tensor_attributes(
  obj: Any, controls: torch.Tensor, probes: torch.Tensor, num_envs: int,
) -> None:
  for value in vars(obj).values():
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == num_envs:
      value[probes] = value[controls].clone()


def _copy_circular_buffer(buffer: Any, controls: torch.Tensor, probes: torch.Tensor) -> None:
  if getattr(buffer, "_buffer", None) is not None:
    buffer._buffer[:, probes] = buffer._buffer[:, controls].clone()
  if hasattr(buffer, "_num_pushes"):
    buffer._num_pushes[probes] = buffer._num_pushes[controls].clone()


def _copy_observation_buffers(
  manager: Any, controls: torch.Tensor, probes: torch.Tensor,
) -> None:
  for group in manager._group_obs_term_history_buffer.values():
    for buffer in group.values():
      _copy_circular_buffer(buffer, controls, probes)
  for group in manager._group_obs_term_delay_buffer.values():
    for delay in group.values():
      _copy_circular_buffer(delay._buffer, controls, probes)
      for name in ("_current_lags", "_step_count", "_phase_offsets"):
        value = getattr(delay, name)
        value[probes] = value[controls].clone()


def _branch_pairs(
  env: base.ManagerBasedRlEnv,
  wrapped: Any,
  policy: Any,
  robot: Any,
  command_term: Any,
  controls: torch.Tensor,
  probes: torch.Tensor,
) -> tuple[Any, list[dict[str, Any]]]:
  state_fields = (
    "time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl",
    "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
  )
  copied_fields: list[str] = []
  for name in state_fields:
    value = getattr(env.sim.data, name)
    if value.ndim > 0 and value.shape[0] == env.num_envs:
      value[probes] = value[controls].clone()
      copied_fields.append(name)

  for name in (
    "joint_pos_target", "joint_vel_target", "joint_effort_target",
    "tendon_len_target", "tendon_vel_target", "tendon_effort_target",
    "site_effort_target", "encoder_bias",
  ):
    value = getattr(robot.data, name)
    if value.ndim > 0 and value.shape[0] == env.num_envs:
      value[probes] = value[controls].clone()

  env.episode_length_buf[probes] = env.episode_length_buf[controls].clone()
  _copy_tensor_attributes(env.action_manager, controls, probes, env.num_envs)
  for term_name in env.action_manager.active_terms:
    _copy_tensor_attributes(
      env.action_manager.get_term(term_name), controls, probes, env.num_envs
    )
  _copy_tensor_attributes(command_term, controls, probes, env.num_envs)
  _copy_observation_buffers(env.observation_manager, controls, probes)

  for sensor in env.scene._sensors.values():
    air_state = getattr(sensor, "_air_time_state", None)
    if air_state is not None:
      _copy_tensor_attributes(air_state, controls, probes, env.num_envs)
    history_state = getattr(sensor, "_history_state", None)
    if history_state is not None:
      for value in history_state.values():
        _copy_rows(value, controls, probes)
    sensor._invalidate_cache()

  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  for sensor in env.scene._sensors.values():
    sensor._invalidate_cache()
  env.observation_manager._obs_buffer = None
  env.obs_buf = env.observation_manager.compute(update_history=False)
  observation = wrapped.get_observations()
  with torch.inference_mode():
    first_action = policy(observation)

  results: list[dict[str, Any]] = []
  for control, probe in zip(controls.tolist(), probes.tolist(), strict=True):
    errors = {
      "qpos_error_max": float(
        (env.sim.data.qpos[control] - env.sim.data.qpos[probe]).abs().max()
      ),
      "qvel_error_max": float(
        (env.sim.data.qvel[control] - env.sim.data.qvel[probe]).abs().max()
      ),
      "qacc_warmstart_error_max": float(
        (env.sim.data.qacc_warmstart[control] - env.sim.data.qacc_warmstart[probe])
        .abs().max()
      ),
      "episode_length_error": int(
        abs(int(env.episode_length_buf[control]) - int(env.episode_length_buf[probe]))
      ),
      "action_history_error_max": float(max(
        (env.action_manager.action[control] - env.action_manager.action[probe]).abs().max(),
        (env.action_manager.prev_action[control] - env.action_manager.prev_action[probe]).abs().max(),
        (env.action_manager.prev_prev_action[control] - env.action_manager.prev_prev_action[probe]).abs().max(),
      )),
      "actor_observation_error_max": float(
        (observation["actor"][control] - observation["actor"][probe]).abs().max()
      ),
      "first_policy_action_error_max": float(
        (first_action[control] - first_action[probe]).abs().max()
      ),
    }
    state = torch.cat((
      env.sim.data.qpos[control], env.sim.data.qvel[control],
      env.sim.data.qacc_warmstart[control], env.action_manager.action[control],
    ))
    probe_state = torch.cat((
      env.sim.data.qpos[probe], env.sim.data.qvel[probe],
      env.sim.data.qacc_warmstart[probe], env.action_manager.action[probe],
    ))
    maximum = max(float(value) for value in errors.values())
    results.append({
      **errors,
      "control_state_sha256": base._sha_tensor(state),
      "probe_state_sha256": base._sha_tensor(probe_state),
      "copied_integration_fields": copied_fields,
      "branch_pass": maximum <= 1.0e-6,
    })
  return observation, results


def _update_streak(
  streak: torch.Tensor, saturated: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
  return torch.where(
    valid[:, None] & saturated,
    streak + 1,
    torch.zeros_like(streak),
  )


def _control_trigger_valid(
  active: torch.Tensor, reset: torch.Tensor, control_ids: torch.Tensor,
) -> torch.Tensor:
  """Trigger validity depends only on the control trajectory."""
  return active[control_ids] & ~reset[control_ids]


def _window_status(
  desired_end: int, control_end: int, probe_end: int,
  control_failed: bool, probe_failed: bool,
) -> str:
  if control_end == desired_end and probe_end == desired_end:
    return "complete_both"
  if probe_end < control_end:
    return "probe_failed_first_harm"
  if control_end < probe_end:
    return "control_failed_first_benefit"
  if control_failed and probe_failed:
    return "both_failed_same_step"
  if probe_failed:
    return "probe_failed_first_harm"
  if control_failed:
    return "control_failed_first_benefit"
  return "horizon_censored"


def _lifecycle_outcome_flags(
  intervention_applied: bool,
  control_failed: bool,
  probe_failed: bool,
  control_count: int,
  probe_count: int,
) -> tuple[bool, bool]:
  if not intervention_applied:
    return False, False
  probe_earlier = probe_failed and (
    not control_failed or probe_count < control_count
  )
  control_earlier = control_failed and (
    not probe_failed or control_count < probe_count
  )
  return probe_earlier, control_earlier


def _risk_window(
  arrays: dict[str, torch.Tensor], control: int, probe: int, start: int, end: int,
) -> dict[str, Any]:
  def values(env_id: int) -> dict[str, float | None]:
    loaded = arrays["loaded"][env_id, start:end]
    slip_values = arrays["slip"][env_id, start:end][loaded]
    action = arrays["action"][env_id, start:end]
    acceleration = action[2:] - 2.0 * action[1:-1] + action[:-2]
    return {
      "slip": None if slip_values.numel() == 0 else float(slip_values.mean()),
      "action_second_difference": None if acceleration.numel() == 0 else float(
        acceleration.abs().mean()
      ),
      "body_contact": float(
        arrays["body_contact"][env_id, start:end].float().sum() / max(end - start, 1)
      ),
    }
  control_values = values(control)
  probe_values = values(probe)
  return {
    name: {
      **base._risk_ratio(probe_values[name], control_values[name]),
      "control": control_values[name],
      "probe": probe_values[name],
      "delta": None if control_values[name] is None or probe_values[name] is None
      else probe_values[name] - control_values[name],
    }
    for name in control_values
  }


def _risk_regression(risk: dict[str, dict[str, Any]]) -> bool:
  return any(
    not item["passes_1p2x"] and item.get("reason") != "missing_metric"
    for item in risk.values()
  )


def _event_metrics(
  arrays: dict[str, torch.Tensor], env_id: int, start: int, end: int,
  command: torch.Tensor, events: list[dict[str, Any]], joint_names: list[str],
  hard_limits: torch.Tensor, soft_limits: torch.Tensor, kp: torch.Tensor,
  kd: torch.Tensor, clip_bound: float | None,
) -> dict[str, Any]:
  if end <= start:
    return {"eligible": False, "status": "no_observed_samples", "sample_count": 0}
  limits = arrays["effort_limit"][env_id, start]
  result = base._window_metrics(
    arrays, env_id, start, end, joint_names, hard_limits, soft_limits,
    limits, kp, kd, clip_bound,
  )
  result.update(base._window_extra(arrays, env_id, start, end, command, events))
  result["effort_limits"] = limits.detach().cpu().tolist()
  result["effort_limit_drift_within_window_max"] = float(
    (arrays["effort_limit"][env_id, start:end] - limits).abs().max()
  )
  return result


def _saturation_summary(
  arrays: dict[str, torch.Tensor], env_id: int, start: int, end: int,
  joint_indices: list[int], kp: torch.Tensor, kd: torch.Tensor,
  old_limits: torch.Tensor,
) -> dict[str, Any]:
  if end <= start or not joint_indices:
    return {"count": 0, "denominator": 0, "fraction": None, "status": "no_samples"}
  q = arrays["joint_pos"][env_id, start:end][:, joint_indices]
  qd = arrays["joint_vel"][env_id, start:end][:, joint_indices]
  target = arrays["target"][env_id, start:end][:, joint_indices]
  force = arrays["force"][env_id, start:end][:, joint_indices]
  limits = arrays["effort_limit"][env_id, start:end][:, joint_indices]
  valid = arrays["pd_valid"][env_id, start:end, None]
  demand = kp[joint_indices] * (target - q) - kd[joint_indices] * qd
  saturated = valid & (force.abs() / limits.clamp_min(1.0e-8) >= 0.98) & (
    demand.abs() >= limits
  )
  denominator = int(valid.sum()) * len(joint_indices)
  count = int(saturated.sum())
  engaged = valid & (force.abs() > old_limits[joint_indices] + 1.0e-4)
  return {
    "count": count,
    "denominator": denominator,
    "fraction": None if denominator == 0 else count / denominator,
    "old_limit_exceedance_count": int(engaged.sum()),
    "status": "ok" if denominator else "no_samples",
  }


def _pair_window(
  size: int, apply: int, control_count: int, probe_count: int,
  control_failed: bool, probe_failed: bool,
  arrays: dict[str, torch.Tensor], control: int, probe: int,
  command_values: torch.Tensor, step_events: list[list[dict[str, Any]]],
  joint_names: list[str], hard_limits: torch.Tensor, soft_limits: torch.Tensor,
  kp: torch.Tensor, kd: torch.Tensor, clip_bound: float | None,
  trigger_joint_indices: list[int], old_limits: torch.Tensor,
) -> dict[str, Any]:
  desired_end = apply + size
  control_end = min(control_count, desired_end)
  probe_end = min(probe_count, desired_end)
  common_end = min(control_end, probe_end)
  status = _window_status(
    desired_end, control_end, probe_end, control_failed, probe_failed
  )
  control_metrics = _event_metrics(
    arrays, control, apply, control_end, command_values[control],
    step_events[control], joint_names, hard_limits, soft_limits, kp, kd, clip_bound,
  )
  probe_metrics = _event_metrics(
    arrays, probe, apply, probe_end, command_values[probe],
    step_events[probe], joint_names, hard_limits, soft_limits, kp, kd, clip_bound,
  )
  common_control = _event_metrics(
    arrays, control, apply, common_end, command_values[control],
    step_events[control], joint_names, hard_limits, soft_limits, kp, kd, clip_bound,
  )
  common_probe = _event_metrics(
    arrays, probe, apply, common_end, command_values[probe],
    step_events[probe], joint_names, hard_limits, soft_limits, kp, kd, clip_bound,
  )
  deltas = None
  if common_end > apply:
    deltas = base._aligned_metric_deltas(common_probe, common_control, joint_names)
  risk = _risk_window(arrays, control, probe, apply, common_end) if common_end > apply else {}
  missing_risk = [
    name for name, item in risk.items() if item.get("reason") == "missing_metric"
  ]
  return {
    "requested_steps": size,
    "desired_start": apply,
    "desired_end": desired_end,
    "control_observed_end": control_end,
    "probe_observed_end": probe_end,
    "common_end": common_end,
    "common_steps": max(common_end - apply, 0),
    "status": status,
    "control": control_metrics,
    "probe": probe_metrics,
    "common_control": common_control,
    "common_probe": common_probe,
    "paired_metric_deltas": deltas,
    "risk": risk,
    "risk_status": "missing_metric" if missing_risk else "complete",
    "missing_risk_metrics": missing_risk,
    "risk_regression": _risk_regression(risk),
    "risk_guardrails_pass": bool(risk) and all(
      item["passes_1p2x"] for item in risk.values()
    ),
    "control_saturation": _saturation_summary(
      arrays, control, apply, common_end, trigger_joint_indices, kp, kd, old_limits
    ),
    "probe_saturation": _saturation_summary(
      arrays, probe, apply, common_end, trigger_joint_indices, kp, kd, old_limits
    ),
  }


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
  total = wins + losses
  if total == 0:
    return None
  return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total)


def _relative_change(probe: list[float], control: list[float]) -> float | None:
  if not probe or not control:
    return None
  control_mean = sum(control) / len(control)
  if abs(control_mean) <= 1.0e-12:
    return None
  return (sum(probe) / len(probe)) / control_mean - 1.0


def _classify(
  pairs: list[dict[str, Any]], strict_gate: bool, baseline_compatible: bool,
  forced_trigger: bool,
) -> dict[str, Any]:
  event_cohort = [
    pair for pair in pairs
    if pair["terrain_condition"] == "slope_up_high"
    and pair["trigger"]["status"] == "applied"
    and pair["branch_identity"]["branch_pass"]
    and pair["pre_100"]["eligible"]
  ]
  evaluable_50 = [p for p in event_cohort if p["post_50"]["common_steps"] >= 50]
  evaluable_100 = [p for p in event_cohort if p["post_100"]["common_steps"] >= 100]

  def reduction(rows: list[dict[str, Any]], key: str) -> float | None:
    control_count = sum(p[key]["control_saturation"]["count"] for p in rows)
    probe_count = sum(p[key]["probe_saturation"]["count"] for p in rows)
    return None if control_count == 0 else (control_count - probe_count) / control_count

  reduction_50 = reduction(evaluable_50, "post_50")
  reduction_100 = reduction(evaluable_100, "post_100")
  wins = sum(p["outcome"]["lifecycle_win"] for p in event_cohort)
  losses = sum(p["outcome"]["lifecycle_loss"] for p in event_cohort)
  harms = sum(p["outcome"]["harm"] for p in event_cohort)
  engaged = sum(
    p["post_100"]["probe_saturation"].get("old_limit_exceedance_count", 0) > 0
    for p in evaluable_100
  )
  completion_delta: dict[str, int] = {}
  gait: dict[str, Any] = {}
  gait_path = True
  for speed in (0.3, 0.5):
    cell = [
      p for p in event_cohort if math.isclose(p["speed"], speed)
    ]
    completion_delta[f"vx_{speed:g}"] = sum(
      not p["probe_lifecycle"]["failed"] for p in cell
    ) - sum(not p["control_lifecycle"]["failed"] for p in cell)
    common = [
      p for p in event_cohort
      if math.isclose(p["speed"], speed)
      and p["post_300"]["common_steps"] >= 300
    ]
    control_gain = [p["post_300"]["common_control"]["response_gain"]["vx"] for p in common]
    probe_gain = [p["post_300"]["common_probe"]["response_gain"]["vx"] for p in common]
    control_step = [
      p["post_300"]["common_control"]["step_length_fully_contained"]["mean"]
      for p in common
      if p["post_300"]["common_control"]["step_length_fully_contained"]["mean"] is not None
      and p["post_300"]["common_probe"]["step_length_fully_contained"]["mean"] is not None
    ]
    probe_step = [
      p["post_300"]["common_probe"]["step_length_fully_contained"]["mean"]
      for p in common
      if p["post_300"]["common_control"]["step_length_fully_contained"]["mean"] is not None
      and p["post_300"]["common_probe"]["step_length_fully_contained"]["mean"] is not None
    ]
    gain_change = _relative_change(probe_gain, control_gain)
    step_change = _relative_change(probe_step, control_step)
    gait[f"vx_{speed:g}"] = {
      "pair_count": len(common),
      "gain_relative": gain_change,
      "step_relative": step_change,
    }
    gait_path &= (
      len(common) >= 4 and gain_change is not None and step_change is not None
      and gain_change >= 0.20 and step_change >= 0.20
    )
  completion_path = all(value >= 2 for value in completion_delta.values())
  risk_pass = harms == 0 and all(
    p["post_100"]["risk_guardrails_pass"]
    and (p["post_300"]["common_steps"] < 300 or p["post_300"]["risk_guardrails_pass"])
    for p in event_cohort
  )
  sign_p = _sign_test_pvalue(wins, losses)
  lifecycle_gate = (
    wins >= math.ceil(len(event_cohort) / 2)
    and wins > losses and sign_p is not None and sign_p <= 0.05
  )
  coverage_pass = len(event_cohort) >= 8 and len(evaluable_50) >= 8 and len(evaluable_100) >= 8
  engagement_pass = bool(evaluable_100) and engaged >= math.ceil(len(evaluable_100) / 2)

  if forced_trigger:
    verdict = "INCONCLUSIVE"
    reason = "forced trigger is plumbing-only and cannot produce a causal verdict"
  elif not strict_gate:
    verdict = "INCONCLUSIVE"
    reason = "strict identity/provenance/flat-sentinel gate failed"
  elif not baseline_compatible:
    verdict = "INCONCLUSIVE"
    reason = "control rerun was incompatible with the formal v2 baseline"
  elif not coverage_pass:
    verdict = "INCONCLUSIVE"
    reason = "triggered cohort or complete post-50/post-100 coverage was below 8"
  elif not engagement_pass:
    verdict = "INCONCLUSIVE"
    reason = "1.25x headroom was not dynamically engaged in half of the primary cohort"
  elif reduction_50 is None or reduction_100 is None:
    verdict = "INCONCLUSIVE"
    reason = "post-trigger saturation evidence was unavailable"
  elif reduction_50 < 0.50 and reduction_100 < 0.50:
    verdict = "HEADROOM_INSUFFICIENT"
    reason = "post-trigger saturation did not fall by 50%"
  elif (reduction_50 >= 0.50) != (reduction_100 >= 0.50):
    verdict = "INCONCLUSIVE"
    reason = "post-50 and post-100 saturation directions conflict"
  elif (completion_path or gait_path) and lifecycle_gate and risk_pass:
    verdict = "TRIGGERED_HEADROOM_CAUSAL"
    reason = "timely headroom reduced saturation and improved locomotion without harm"
  else:
    verdict = "SATURATION_DOWNSTREAM"
    reason = "timely headroom reduced saturation without the required locomotion/risk improvement"
  return {
    "verdict": verdict,
    "reason": reason,
    "strict_identity_gate_pass": strict_gate,
    "baseline_compatibility_pass": baseline_compatible,
    "coverage_pass": coverage_pass,
    "headroom_engagement_pass": engagement_pass,
    "event_cohort_size": len(event_cohort),
    "evaluable_post_50_count": len(evaluable_50),
    "evaluable_post_100_count": len(evaluable_100),
    "saturation_reduction_post_50": reduction_50,
    "saturation_reduction_post_100": reduction_100,
    "headroom_engaged_pair_count": engaged,
    "lifecycle_wins": wins,
    "lifecycle_losses": losses,
    "harm_pair_count": harms,
    "one_sided_sign_test_pvalue": sign_p,
    "completion_delta": completion_delta,
    "completion_path_pass": completion_path,
    "gait_path": gait,
    "gait_path_pass": gait_path,
    "risk_guardrails_pass": risk_pass,
  }


def _run_condition(
  cfg: TriggeredConfig, condition: str, terrain_kind: str, terrain_level: int,
) -> dict[str, Any]:
  base_scenarios, physical = _physical_scenarios(
    cfg, condition, terrain_kind, terrain_level
  )
  num_envs = len(physical)
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  env_cfg = base.load_env_cfg(cfg.task_id)
  agent_cfg = base.load_rl_cfg(cfg.task_id)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = base._make_gait_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  profile_settings = base._configure_profile(env_cfg, cfg.profile)
  if profile_settings["startup_randomization_events"]:
    raise RuntimeError("clean profile retained startup randomization")
  episode_settings = base._configure_episode_length(
    env_cfg, cfg.warmup_steps + cfg.sample_steps + 20
  )
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, base.UniformVelocityCommandCfg):
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

  env = base.ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  original_reset_idx = env._reset_idx
  try:
    robot = env.scene["robot"]
    joint_ids, joint_names_raw = robot.find_joints((".*",), preserve_order=True)
    joint_names = list(joint_names_raw)
    force_ids, global_ids = base._joint_control_mapping(robot, joint_names)
    old_limits, active_limits, ranges, runtime_identity = _prepare_limits(
      env, global_ids
    )
    wrapped = base.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = base.load_runner_cls(cfg.task_id) or base.MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(Path(cfg.checkpoint).resolve()), load_cfg={"actor": True}, strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
    placement = base._assign_terrain(env, physical, env.device)
    command_term = env.command_manager.get_term("twist")
    if not isinstance(command_term, base.UniformVelocityCommand):
      raise TypeError("twist command term is incompatible")
    command_values = torch.tensor(
      [[row["speed"], 0.0, 0.0] for row in physical], device=env.device
    )
    initial_identity = base._copy_pair_initial_state(
      env, robot, joint_ids, command_term, command_values
    )
    initial_observation = wrapped.get_observations()
    initial_control_ids = torch.arange(0, num_envs, 2, device=env.device)
    initial_probe_ids = initial_control_ids + 1
    with torch.inference_mode():
      initial_action = policy(initial_observation)
    initial_observation_error = float(
      (initial_observation["actor"][initial_control_ids]
       - initial_observation["actor"][initial_probe_ids]).abs().max()
    )
    initial_action_error = float(
      (initial_action[initial_control_ids] - initial_action[initial_probe_ids])
      .abs().max()
    )
    initial_identity.update({
      "actor_observation_error_max": initial_observation_error,
      "first_policy_action_error_max": initial_action_error,
      "pairing_pass": max(
        initial_identity["root_pose_error_max"],
        initial_identity["root_velocity_error_max"],
        initial_identity["joint_position_error_max"],
        initial_observation_error,
        initial_action_error,
      ) <= 1.0e-6,
    })
    if not initial_identity["pairing_pass"]:
      raise RuntimeError(f"initial pair identity failed: {initial_identity}")

    foot_ids, foot_names = robot.find_sites(base.FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != base.FOOT_NAMES:
      raise RuntimeError("foot order mismatch")
    feet_sensor = env.scene["feet_ground_contact"]
    terrain_sensor = env.scene["terrain_scan"]
    sensor_names = [
      slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"
    ]
    desired_names = [f"{name}_foot_collision" for name in base.FOOT_NAMES]
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
    nfeet = len(base.FOOT_NAMES)
    shape = (num_envs, cfg.sample_steps)
    arrays: dict[str, torch.Tensor] = {
      "action": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "joint_pos": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "joint_vel": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "target": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "force": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "effort_limit": torch.full((*shape, len(joint_names)), torch.nan, device=env.device),
      "normal_force": torch.full((*shape, nfeet), torch.nan, device=env.device),
      "signed_normal_force": torch.full((*shape, nfeet), torch.nan, device=env.device),
      "tangent_force": torch.full((*shape, nfeet), torch.nan, device=env.device),
      "slip": torch.full((*shape, nfeet), torch.nan, device=env.device),
      "loaded": torch.zeros((*shape, nfeet), dtype=torch.bool, device=env.device),
      "ray_valid": torch.zeros((*shape, nfeet), dtype=torch.bool, device=env.device),
      "pitch": torch.full(shape, torch.nan, device=env.device),
      "body_contact": torch.zeros((*shape, 3), dtype=torch.bool, device=env.device),
      "foot_pos": torch.full((*shape, nfeet, 3), torch.nan, device=env.device),
      "terrain_normal": torch.full((*shape, nfeet, 3), torch.nan, device=env.device),
      "actual_velocity": torch.full((*shape, 3), torch.nan, device=env.device),
      "pd_valid": torch.zeros(shape, dtype=torch.bool, device=env.device),
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
    prev_contact = torch.zeros(num_envs, nfeet, dtype=torch.bool, device=env.device)
    has_liftoff = torch.zeros_like(prev_contact)
    liftoff_pos = torch.zeros(num_envs, nfeet, 3, device=env.device)
    liftoff_step = torch.zeros(num_envs, nfeet, dtype=torch.long, device=env.device)
    step_events: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]

    pair_count = len(base_scenarios)
    control_ids = torch.arange(0, num_envs, 2, device=env.device)
    probe_ids = control_ids + 1
    streak = torch.zeros(pair_count, len(joint_names), dtype=torch.long, device=env.device)
    trigger_events: list[dict[str, Any] | None] = [None] * pair_count
    branch_identities: list[dict[str, Any] | None] = [None] * pair_count
    pending = torch.zeros(pair_count, dtype=torch.bool, device=env.device)
    applied = torch.zeros(pair_count, dtype=torch.bool, device=env.device)
    preapply_pair_error_max = torch.zeros(pair_count, device=env.device)
    range_events: list[dict[str, Any] | None] = [None] * pair_count

    def capture_reset(env_ids: torch.Tensor | None = None) -> None:
      nonlocal capture, captured_ids, capture_reasons
      if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)
      snapshot = base._headroom_snapshot(
        env, robot, feet_sensor, terrain_sensor, body_sensors, foot_ids,
        foot_permutation, joint_ids, force_ids, current_action, terrain_kind,
        terrain_level, loaded_contact,
      )
      captured_ids = env_ids.clone()
      capture = {
        key: value.index_select(0, env_ids).clone() for key, value in snapshot.items()
      }
      capture_reasons = {
        int(env_id): base._contact_termination(env, int(env_id))
        for env_id in env_ids.tolist()
      }
      original_reset_idx(env_ids)

    env._reset_idx = capture_reset  # type: ignore[method-assign]
    command_term.vel_command_b[:] = command_values
    observation = wrapped.get_observations()

    for step in range(cfg.warmup_steps + cfg.sample_steps):
      command_term.vel_command_b[:] = command_values
      with torch.inference_mode():
        action = policy(observation)
      action_pair_error = (action[control_ids] - action[probe_ids]).abs().amax(dim=1)
      untriggered = ~applied
      preapply_pair_error_max = torch.maximum(
        preapply_pair_error_max,
        torch.where(untriggered, action_pair_error, torch.zeros_like(action_pair_error)),
      )
      if pending.any():
        pair_ids = torch.where(pending)[0]
        if float(action_pair_error[pair_ids].max()) > 1.0e-6:
          raise RuntimeError("first branch action identity failed")
        probes = probe_ids[pair_ids]
        before = ranges[probes[:, None], global_ids, :].clone()
        ranges[probes[:, None], global_ids, :] = (
          env.sim.get_default_field("actuator_forcerange")[global_ids, :]
          .unsqueeze(0) * MULTIPLIERS[1]
        )
        active_limits[probes] = old_limits * MULTIPLIERS[1]
        after = ranges[probes[:, None], global_ids, :].clone()
        sample_step = step - cfg.warmup_steps
        for offset, pair_id in enumerate(pair_ids.tolist()):
          trigger_events[pair_id]["apply_step"] = sample_step  # type: ignore[index]
          trigger_events[pair_id]["status"] = "applied"  # type: ignore[index]
          range_events[pair_id] = {
            "before": before[offset].detach().cpu().tolist(),
            "after": after[offset].detach().cpu().tolist(),
            "control_range_error_at_apply": float(
              (ranges[control_ids[pair_id], global_ids, :]
               - env.sim.get_default_field("actuator_forcerange")[global_ids, :]).abs().max()
            ),
          }
        applied[pair_ids] = True
        pending[pair_ids] = False

      step_limits = active_limits.clone()
      current_action = action.detach().clone()
      capture = {}
      captured_ids = torch.empty(0, dtype=torch.long, device=env.device)
      capture_reasons = {}
      _, _, dones, _ = wrapped.step(action)
      observation = wrapped.get_observations()
      reset = dones.bool()
      state = base._headroom_snapshot(
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
          raise RuntimeError(f"unexpected snapshot rank: {key}")
      arrays["effort_limit"][:, k] = torch.where(
        write[:, None], step_limits, arrays["effort_limit"][:, k]
      )
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
            "foot": base.FOOT_NAMES[foot_id],
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

      control_valid = _control_trigger_valid(active, reset, control_ids)
      pair_state_error = torch.stack((
        (state["joint_pos"][control_ids] - state["joint_pos"][probe_ids]).abs().amax(dim=1),
        (state["joint_vel"][control_ids] - state["joint_vel"][probe_ids]).abs().amax(dim=1),
        (state["target"][control_ids] - state["target"][probe_ids]).abs().amax(dim=1),
        (state["force"][control_ids] - state["force"][probe_ids]).abs().amax(dim=1),
      )).amax(dim=0)
      preapply_pair_error_max = torch.maximum(
        preapply_pair_error_max,
        torch.where(~applied, pair_state_error, torch.zeros_like(pair_state_error)),
      )

      control_q = state["joint_pos"][control_ids]
      control_qd = state["joint_vel"][control_ids]
      control_target = state["target"][control_ids]
      control_force = state["force"][control_ids]
      demand = kp * (control_target - control_q) - kd * control_qd
      saturated = (
        control_force.abs() / old_limits.clamp_min(1.0e-8) >= cfg.saturation_threshold
      ) & (demand.abs() >= old_limits)
      streak = _update_streak(streak, saturated, control_valid & ~applied)
      natural = streak >= cfg.trigger_run_steps
      forced = torch.zeros_like(natural)
      if (
        cfg.forced_trigger_step is not None
        and condition == "slope_up_high"
        and k == cfg.forced_trigger_step
      ):
        forced[:, 0] = control_valid & ~applied
      new_pairs = torch.where((natural | forced).any(dim=1) & ~applied & ~pending)[0]
      if len(new_pairs) > 0:
        controls = control_ids[new_pairs]
        probes = probe_ids[new_pairs]
        discarded_probe_prebranch = [
          {
            "failed": first_reason[probe] is not None,
            "sample_count": int(sample_count[probe]),
            "reset_count": int(reset_count[probe]),
            "reason": first_reason[probe],
            "failure_phase": first_failure_phase[probe],
            "failure_step": first_failure_step[probe],
          }
          for probe in probes.tolist()
        ]
        observation, identities = _branch_pairs(
          env, wrapped, policy, robot, command_term, controls, probes
        )
        for offset, pair_id in enumerate(new_pairs.tolist()):
          control = int(controls[offset])
          probe = int(probes[offset])
          joint_mask = natural[pair_id]
          if forced[pair_id].any() and not joint_mask.any():
            joint_mask = forced[pair_id]
          joint_indices = torch.where(joint_mask)[0].tolist()
          trace: list[dict[str, Any]] = []
          for trace_step in range(max(k - 2, 0), k + 1):
            trace.append({
              "step": trace_step,
              "pd_valid": bool(arrays["pd_valid"][control_ids[pair_id], trace_step]),
              "force": arrays["force"][control_ids[pair_id], trace_step, joint_indices]
              .detach().cpu().tolist(),
              "demand": (
                kp[joint_indices] * (
                  arrays["target"][control_ids[pair_id], trace_step, joint_indices]
                  - arrays["joint_pos"][control_ids[pair_id], trace_step, joint_indices]
                )
                - kd[joint_indices] * arrays["joint_vel"][
                  control_ids[pair_id], trace_step, joint_indices
                ]
              ).detach().cpu().tolist(),
            })
          trigger_events[pair_id] = {
            "status": "pending",
            "forced": bool(forced[pair_id].any()),
            "onset_step": k - cfg.trigger_run_steps + 1,
            "detect_step": k,
            "apply_step": None,
            "joint_indices": joint_indices,
            "joints": [joint_names[index] for index in joint_indices],
            "primary_joint": joint_names[joint_indices[0]],
            "detector_trace": trace,
            "detector_replay_pass": bool(forced[pair_id].any()) or all(
              row["pd_valid"]
              and all(abs(force) >= cfg.saturation_threshold * float(old_limits[index])
                      for force, index in zip(row["force"], joint_indices, strict=True))
              and all(abs(demand_value) >= float(old_limits[index])
                      for demand_value, index in zip(row["demand"], joint_indices, strict=True))
              for row in trace
            ),
          }
          identity = identities[offset]
          identity["preapply_pair_error_max"] = float(
            preapply_pair_error_max[pair_id]
          )
          identity["prebranch_probe_history_used_for_causal_metrics"] = False
          identity["discarded_probe_prebranch"] = discarded_probe_prebranch[offset]
          branch_identities[pair_id] = identity
          for values in arrays.values():
            values[probe, :k + 1] = values[control, :k + 1].clone()
          sample_count[probe] = sample_count[control]
          reset_count[probe] = reset_count[control]
          first_reason[probe] = first_reason[control]
          first_failure_phase[probe] = first_failure_phase[control]
          first_failure_step[probe] = first_failure_step[control]
          step_events[probe] = [dict(event) for event in step_events[control]]
          active[probe] = active[control]
          reset[probe] = False
          loaded_contact[probe] = loaded_contact[control].clone()
          state["loaded"][probe] = state["loaded"][control].clone()
          prev_contact[control] = state["loaded"][control].clone()
          prev_contact[probe] = state["loaded"][control].clone()
          has_liftoff[control] = False
          has_liftoff[probe] = False
          liftoff_pos[control] = state["foot_pos"][control].clone()
          liftoff_pos[probe] = state["foot_pos"][control].clone()
        pending[new_pairs] = True

      loaded_contact = torch.where(
        reset[:, None], torch.zeros_like(loaded_contact), state["loaded"]
      )
      active &= ~reset

    rows: list[dict[str, Any]] = []
    for env_id, scenario in enumerate(physical):
      count = int(sample_count[env_id])
      failed = first_reason[env_id] is not None
      summary = base._attempt_summary(
        arrays, env_id, count, command_values[env_id], step_events[env_id]
      ) if count > 0 else {
        "response_gain": {"vx": None, "vx_reason": "no_samples"},
        "step_length": base._finite_stats(torch.empty(0, device=env.device)),
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
        "attempt_summary": summary,
      })

    pairs: list[dict[str, Any]] = []
    for pair_id, scenario in enumerate(base_scenarios):
      control, probe = 2 * pair_id, 2 * pair_id + 1
      trigger = trigger_events[pair_id]
      identity = branch_identities[pair_id]
      if trigger is None:
        trigger = {
          "status": "not_triggered", "forced": False, "onset_step": None,
          "detect_step": None, "apply_step": None, "joint_indices": [],
          "joints": [], "primary_joint": None, "detector_trace": [],
          "detector_replay_pass": True,
        }
        identity = {
          "branch_pass": True,
          "status": "not_applicable_no_trigger",
          "preapply_pair_error_max": float(preapply_pair_error_max[pair_id]),
          "prebranch_probe_history_used_for_causal_metrics": False,
        }
      elif trigger["status"] == "pending":
        trigger["status"] = "detected_not_applied_horizon_end"
      apply_step = trigger["apply_step"]
      control_count = rows[control]["sample_count"]
      probe_count = rows[probe]["sample_count"]
      control_lifecycle = {
        "failed": rows[control]["failed"],
        "sample_count": control_count,
        "reset_count": rows[control]["reset_count"],
        "reason": rows[control]["first_failure_reason"],
        "failure_phase": rows[control]["first_failure_phase"],
        "failure_step": rows[control]["first_failure_step"],
        "failure_relative_to_apply": None if apply_step is None or not rows[control]["failed"]
        else rows[control]["first_failure_step"] - apply_step,
      }
      probe_lifecycle = {
        "failed": rows[probe]["failed"],
        "sample_count": probe_count,
        "reset_count": rows[probe]["reset_count"],
        "reason": rows[probe]["first_failure_reason"],
        "failure_phase": rows[probe]["first_failure_phase"],
        "failure_step": rows[probe]["first_failure_step"],
        "failure_relative_to_apply": None if apply_step is None or not rows[probe]["failed"]
        else rows[probe]["first_failure_step"] - apply_step,
      }
      if apply_step is None:
        pre_100 = {"eligible": False, "status": "not_applicable_no_trigger"}
        windows = {
          f"post_{size}": {
            "requested_steps": size, "status": "not_applicable_no_trigger",
            "common_steps": 0, "risk_guardrails_pass": True,
          }
          for size in cfg.post_windows
        }
      else:
        detect = trigger["detect_step"]
        pre_start = detect - 99
        pre_100 = {
          "eligible": pre_start >= 0,
          "status": "complete" if pre_start >= 0 else "insufficient_pretrigger_history",
          "desired_start": pre_start,
          "desired_end": detect + 1,
          "control": None if pre_start < 0 else _event_metrics(
            arrays, control, pre_start, detect + 1, command_values[control],
            step_events[control], joint_names, hard_limits, soft_limits, kp, kd,
            agent_cfg.clip_actions,
          ),
        }
        windows = {
          f"post_{size}": _pair_window(
            size, apply_step, control_count, probe_count,
            rows[control]["failed"], rows[probe]["failed"],
            arrays, control, probe,
            command_values, step_events, joint_names, hard_limits, soft_limits,
            kp, kd, agent_cfg.clip_actions, trigger["joint_indices"], old_limits,
          )
          for size in cfg.post_windows
        }
      intervention_applied = apply_step is not None
      probe_earlier, control_earlier = _lifecycle_outcome_flags(
        intervention_applied,
        rows[control]["failed"], rows[probe]["failed"],
        control_count, probe_count,
      )
      lifecycle_win = False
      if apply_step is not None and rows[control]["failed"]:
        control_age = control_count - apply_step
        probe_age = probe_count - apply_step
        lifecycle_win = (
          not rows[probe]["failed"]
          or probe_age - control_age >= max(100, math.ceil(0.20 * max(control_age, 1)))
        )
      risk_failure = apply_step is not None and any(
        windows[f"post_{size}"]["risk_regression"]
        for size in (100, 300)
        if windows[f"post_{size}"]["common_steps"] > 0
      )
      schedule = {
        "control": [{"start": 0, "end": cfg.sample_steps, "multiplier": 1.0}],
        "probe": [{"start": 0, "end": apply_step, "multiplier": 1.0}]
        if apply_step is not None else [
          {"start": 0, "end": cfg.sample_steps, "multiplier": 1.0}
        ],
      }
      if apply_step is not None:
        schedule["probe"].append({
          "start": apply_step, "end": cfg.sample_steps, "multiplier": 1.25
        })
      schedule_sha = hashlib.sha256(
        json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
      ).hexdigest()
      pairs.append({
        **scenario,
        "trigger": trigger,
        "schedule": schedule,
        "schedule_sha256": schedule_sha,
        "range_event": range_events[pair_id],
        "branch_identity": identity,
        "control_lifecycle": control_lifecycle,
        "probe_lifecycle": probe_lifecycle,
        "pre_100": pre_100,
        **windows,
        "outcome": {
          "status": "evaluated" if intervention_applied else "not_applicable_no_trigger",
          "probe_earlier_failure": probe_earlier,
          "control_earlier_failure": control_earlier,
          "lifecycle_win": lifecycle_win,
          "lifecycle_loss": probe_earlier,
          "harm": probe_earlier or risk_failure,
        },
      })

    demand = kp * (arrays["target"] - arrays["joint_pos"]) - kd * arrays["joint_vel"]
    expected_force = torch.clamp(
      demand,
      min=-arrays["effort_limit"],
      max=arrays["effort_limit"],
    )
    pd_valid = torch.isfinite(arrays["force"]) & arrays["pd_valid"][:, :, None]
    all_force_valid = torch.isfinite(arrays["force"])
    clamp_error = float(
      (arrays["force"][pd_valid] - expected_force[pd_valid]).abs().max()
    ) if pd_valid.any() else None
    force_excess = float(
      (arrays["force"].abs() - arrays["effort_limit"])
      .masked_fill(~all_force_valid, -torch.inf).max()
    ) if all_force_valid.any() else None
    expected_end = env.sim.get_default_field("actuator_forcerange").unsqueeze(0).repeat(
      num_envs, 1, 1
    )
    for pair_id in torch.where(applied)[0].tolist():
      expected_end[probe_ids[pair_id], global_ids] *= MULTIPLIERS[1]
    range_drift = float((ranges - expected_end).abs().max())
    control_range_drift = float(
      (ranges[control_ids[:, None], global_ids, :]
       - env.sim.get_default_field("actuator_forcerange")[global_ids, :]).abs().max()
    )
    categories = {
      "control": torch.zeros((num_envs, cfg.sample_steps, 1), dtype=torch.bool, device=env.device),
      "probe_pre_trigger": torch.zeros((num_envs, cfg.sample_steps, 1), dtype=torch.bool, device=env.device),
      "probe_post_trigger": torch.zeros((num_envs, cfg.sample_steps, 1), dtype=torch.bool, device=env.device),
    }
    categories["control"][control_ids] = True
    for pair_id in range(pair_count):
      apply_step = trigger_events[pair_id]["apply_step"] if trigger_events[pair_id] else None
      split = cfg.sample_steps if apply_step is None else apply_step
      categories["probe_pre_trigger"][probe_ids[pair_id], :split] = True
      if apply_step is not None:
        categories["probe_post_trigger"][probe_ids[pair_id], apply_step:] = True
    clipping: dict[str, Any] = {}
    for name, category in categories.items():
      valid = pd_valid & category
      exceeds = valid & (demand.abs() >= arrays["effort_limit"])
      at_limit = valid & (
        (arrays["force"].abs() - arrays["effort_limit"]).abs() <= 1.0e-3
      )
      clipping[name] = {
        "demand_exceeds_limit_count": int(exceeds.sum()),
        "force_at_limit_when_demand_exceeds_count": int((exceeds & at_limit).sum()),
        "status": "dynamically_exercised" if exceeds.any() else "not_exercised",
      }
    runtime_identity.update({
      "actuator_force_clamp_error_max": clamp_error,
      "actuator_force_limit_excess_max": force_excess,
      "runtime_range_drift_max": range_drift,
      "control_range_drift_max": control_range_drift,
      "actual_clipping_observation": clipping,
      "runtime_pass": clamp_error is not None and clamp_error <= 1.0e-3
      and force_excess is not None and force_excess <= 1.0e-4
      and range_drift <= 1.0e-6 and control_range_drift <= 1.0e-6,
    })
    return {
      "terrain_condition": condition,
      "terrain_kind": terrain_kind,
      "terrain_level": terrain_level,
      "profile_settings": profile_settings,
      "episode_settings": episode_settings,
      "runtime_identity": runtime_identity,
      "initial_identity": initial_identity,
      "terrain_assignment_position_error_max": float(
        placement["terrain_assignment_position_error_max"]
      ),
      "terrain_placement_position_error_max": float(
        placement["terrain_placement_position_error_max"]
      ),
      "rows": rows,
      "pairs": pairs,
    }
  finally:
    env._reset_idx = original_reset_idx  # type: ignore[method-assign]
    env.close()


def _baseline_compatibility(
  baseline: dict[str, Any], conditions: list[dict[str, Any]],
) -> dict[str, Any]:
  expected: dict[tuple[str, float], int] = {}
  for condition in baseline["conditions"]:
    if condition["terrain_condition"] not in {row[0] for row in TRIGGER_CONDITIONS}:
      continue
    for speed in (0.3, 0.5):
      expected[(condition["terrain_condition"], speed)] = sum(
        not row["failed"] for row in condition["rows"]
        if row["arm"] == "control" and math.isclose(row["speed"], speed)
      )
  observed: dict[tuple[str, float], int] = {}
  for condition in conditions:
    for speed in (0.3, 0.5):
      observed[(condition["terrain_condition"], speed)] = sum(
        not row["failed"] for row in condition["rows"]
        if row["arm"] == "control" and math.isclose(row["speed"], speed)
      )
  deltas = {f"{c}_vx_{s:g}": observed[(c, s)] - value for (c, s), value in expected.items()}
  return {
    "baseline_control_success_counts": {
      f"{c}_vx_{s:g}": value for (c, s), value in expected.items()
    },
    "triggered_control_success_counts": {
      f"{c}_vx_{s:g}": value for (c, s), value in observed.items()
    },
    "success_count_deltas": deltas,
    "allowed_absolute_delta_per_cell": 2,
    "passes": all(abs(value) <= 2 for value in deltas.values()),
  }


def _no_trigger_lifecycle_match(pair: dict[str, Any]) -> bool:
  control = pair["control_lifecycle"]
  probe = pair["probe_lifecycle"]
  return (
    control["failed"] == probe["failed"]
    and control["sample_count"] == probe["sample_count"]
    and control["reason"] == probe["reason"]
  )


def _strict_identity_gate(conditions: list[dict[str, Any]]) -> bool:
  """Applied branches and the declared flat no-trigger sentinel are hard gates."""
  return all(
    condition["runtime_identity"]["runtime_pass"]
    and condition["initial_identity"]["pairing_pass"]
    and condition["terrain_assignment_position_error_max"] < 1.0e-4
    and condition["terrain_placement_position_error_max"] < 1.0e-4
    and all(
      pair["branch_identity"]["branch_pass"]
      and pair["trigger"]["detector_replay_pass"]
      and (
        pair["trigger"]["status"] != "not_triggered"
        or pair["terrain_condition"] != "flat"
        or _no_trigger_lifecycle_match(pair)
      )
      for pair in condition["pairs"]
    )
    for condition in conditions
  )


def _negative_control_diagnostics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
  no_trigger = [pair for pair in pairs if pair["trigger"]["status"] == "not_triggered"]
  mismatches = [pair for pair in no_trigger if not _no_trigger_lifecycle_match(pair)]
  return {
    "flat_no_trigger_pair_count": sum(
      pair["terrain_condition"] == "flat" for pair in no_trigger
    ),
    "flat_lifecycle_mismatch_count": sum(
      pair["terrain_condition"] == "flat" for pair in mismatches
    ),
    "slope_no_trigger_pair_count": sum(
      pair["terrain_condition"] == "slope_up_high" for pair in no_trigger
    ),
    "slope_lifecycle_mismatch_count": sum(
      pair["terrain_condition"] == "slope_up_high" for pair in mismatches
    ),
    "slope_mismatch_slots": [
      pair["matched_slot"] for pair in mismatches
      if pair["terrain_condition"] == "slope_up_high"
    ],
    "interpretation": (
      "flat no-trigger lifecycle equality is a hard sentinel; slope no-trigger "
      "divergence is a recorded MuJoCo-Warp nondeterminism boundary because the "
      "unused probe history is discarded at an applied branch"
    ),
  }


def evaluate(cfg: TriggeredConfig) -> dict[str, Any]:
  _validate_config(cfg)
  base.configure_torch_backends()
  baseline_path = Path(cfg.baseline_counterfactual).expanduser().resolve()
  baseline = json.loads(baseline_path.read_text())
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if baseline.get("checkpoint_sha256") != base._sha256(checkpoint):
    raise RuntimeError("baseline/checkpoint identity mismatch")
  conditions = [
    _run_condition(cfg, condition, kind, level)
    for condition, kind, level in TRIGGER_CONDITIONS
  ]
  pairs = [pair for condition in conditions for pair in condition["pairs"]]
  compatibility = _baseline_compatibility(baseline, conditions)
  strict_gate = _strict_identity_gate(conditions)
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_actuator_headroom_triggered_counterfactual",
    "git_head": base._git_head(),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": base._sha256(Path(__file__).resolve()),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": base._sha256(checkpoint),
    "baseline_counterfactual": str(baseline_path),
    "baseline_counterfactual_sha256": base._sha256(baseline_path),
    "config": asdict(cfg),
    "single_variable_contract": {
      "variable": "actuator headroom activation timing",
      "control": "1.00x for all steps",
      "probe": "1.00x before control trigger, 1.25x from detect_step+1",
      "unchanged": [
        "checkpoint", "policy", "terrain", "command", "termination", "gait",
        "observation", "action scale", "reward", "network", "PPO",
      ],
    },
    "coordinate_and_timing_contract": {
      "trigger": "same control joint saturated for three consecutive valid post-step rows",
      "terminal_pd": "invalid because reset-hook actuator force is one physics substep stale",
      "step_length": "liftoff-to-touchdown displacement projected on local terrain forward tangent",
      "contact_force": "local terrain normal/tangent projection",
    },
    "conditions": conditions,
    "negative_control_diagnostics": _negative_control_diagnostics(pairs),
    "baseline_compatibility": compatibility,
    "strict_gate_pass": strict_gate,
  }
  payload["causal_decision"] = _classify(
    pairs, strict_gate, compatibility["passes"], cfg.forced_trigger_step is not None
  )
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  cfg = tyro.cli(TriggeredConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps({
    "output": str(output),
    "strict_gate_pass": payload["strict_gate_pass"],
    "baseline_compatibility": payload["baseline_compatibility"],
    "causal_decision": payload["causal_decision"],
  }, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
