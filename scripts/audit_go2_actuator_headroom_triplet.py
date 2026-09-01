"""Matched source/sham/probe actuator-headroom diagnostic for Go2 V7.

This evaluator is deliberately narrower than the production actuator audit.  It
records only the quantities needed to separate an actuator intervention from the
natural branch divergence of MJWarp:

* source: 1.00x effort limits and the only trigger detector;
* sham: 1.00x effort limits, copied from source at the trigger;
* probe: copied from source at the trigger, then 1.25x from detect+1.

No training configuration or robot asset is modified.
"""

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
import scripts.audit_go2_actuator_headroom_triggered as triggered
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


ARMS = ("source", "sham", "probe")
MULTIPLIERS = (1.0, 1.0, 1.25)
TRIGGER_CONDITIONS = tuple(
  row for row in base.TERRAIN_CONDITIONS
  if row[0] in ("flat", "slope_up_high")
)


@dataclass(frozen=True)
class TripletConfig:
  checkpoint: str = base.V7_CHECKPOINT
  task_id: str = "Unitree-Go2-Rough-V7"
  profile: str = "clean"
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  post_windows: tuple[int, ...] = (50, 100)
  trigger_run_steps: int = 3
  saturation_threshold: float = 0.98
  seed: int = 42
  device: str = "cuda:0"
  forced_trigger_step: int | None = None
  output_file: str = "go2_actuator_headroom_triplet.json"


def _validate_config(cfg: TripletConfig) -> None:
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
  if cfg.post_windows != (50, 100):
    raise ValueError("post windows are locked to 50 and 100")
  if cfg.trigger_run_steps != 3 or not math.isclose(cfg.saturation_threshold, 0.98):
    raise ValueError("trigger is locked to three consecutive 0.98-limit steps")
  if max(cfg.post_windows) > cfg.sample_steps:
    raise ValueError("sample_steps must cover post windows")
  if cfg.forced_trigger_step is not None and not (
    2 <= cfg.forced_trigger_step < cfg.sample_steps - 1
  ):
    raise ValueError("forced trigger must leave a following intervention step")


def _physical_scenarios(
  cfg: TripletConfig, condition: str, terrain_kind: str, terrain_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  rows = base._scenario_slots(cfg, condition, terrain_kind, terrain_level)
  physical: list[dict[str, Any]] = []
  for pair_id, row in enumerate(rows):
    for arm_index, (arm, multiplier) in enumerate(zip(ARMS, MULTIPLIERS, strict=True)):
      physical.append({
        **row,
        "pair_id": pair_id,
        "arm": arm,
        "arm_index": arm_index,
        "initial_effort_limit_multiplier": 1.0,
        "post_trigger_effort_limit_multiplier": multiplier,
        "physical_env": len(physical),
      })
  return rows, physical


def _sha_tensor(value: torch.Tensor) -> str:
  return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _copy_manager_state(manager: Any, source: torch.Tensor, target: torch.Tensor, n: int) -> None:
  triggered._copy_tensor_attributes(manager, source, target, n)
  for name in getattr(manager, "active_terms", ()):
    try:
      triggered._copy_tensor_attributes(manager.get_term(name), source, target, n)
    except (AttributeError, KeyError):
      pass


def _branch_triplet(
  env: base.ManagerBasedRlEnv,
  wrapped: Any,
  policy: Any,
  robot: Any,
  command_term: Any,
  source: torch.Tensor,
  sham: torch.Tensor,
  probe: torch.Tensor,
) -> dict[str, Any]:
  """Copy a source state into both 1.0x branches and verify identity."""
  state_fields = (
    "time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl",
    "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
  )
  copied: list[str] = []
  for name in state_fields:
    value = getattr(env.sim.data, name)
    if value.ndim > 0 and value.shape[0] == env.num_envs:
      value[sham] = value[source].clone()
      value[probe] = value[source].clone()
      copied.append(name)

  for name in (
    "joint_pos_target", "joint_vel_target", "joint_effort_target",
    "tendon_len_target", "tendon_vel_target", "tendon_effort_target",
    "site_effort_target", "encoder_bias",
  ):
    value = getattr(robot.data, name, None)
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == env.num_envs:
      value[sham] = value[source].clone()
      value[probe] = value[source].clone()

  env.episode_length_buf[sham] = env.episode_length_buf[source].clone()
  env.episode_length_buf[probe] = env.episode_length_buf[source].clone()
  for name in ("reset_buf", "reset_terminated", "reset_time_outs"):
    value = getattr(env, name, None)
    if isinstance(value, torch.Tensor):
      value[sham] = value[source].clone()
      value[probe] = value[source].clone()
  for target in (sham, probe):
    _copy_manager_state(env.action_manager, source, target, env.num_envs)
    _copy_manager_state(command_term, source, target, env.num_envs)
  # Termination buffers are observable lifecycle state even though clean runs
  # normally have all of them false at a branch point.
  term = env.termination_manager
  for value in (getattr(term, "_terminated_buf", None), getattr(term, "_truncated_buf", None)):
    if isinstance(value, torch.Tensor):
      value[sham] = value[source].clone()
      value[probe] = value[source].clone()
  for value in getattr(term, "_term_dones", {}).values():
    value[sham] = value[source].clone()
    value[probe] = value[source].clone()
  triggered._copy_observation_buffers(env.observation_manager, source, sham)
  triggered._copy_observation_buffers(env.observation_manager, source, probe)

  for sensor in env.scene._sensors.values():
    air_state = getattr(sensor, "_air_time_state", None)
    if air_state is not None:
      triggered._copy_tensor_attributes(air_state, source, sham, env.num_envs)
      triggered._copy_tensor_attributes(air_state, source, probe, env.num_envs)
    history_state = getattr(sensor, "_history_state", None)
    if history_state is not None:
      for value in history_state.values():
        triggered._copy_rows(value, source, sham)
        triggered._copy_rows(value, source, probe)
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
    action = policy(observation)

  identities: list[dict[str, Any]] = []
  for src_id, sham_id, probe_id in zip(source.tolist(), sham.tolist(), probe.tolist(), strict=True):
    identity: dict[str, Any] = {"copied_integration_fields": copied, "branches": {}}
    for branch_id, name in ((sham_id, "sham"), (probe_id, "probe")):
      state_error = max(
        float((env.sim.data.qpos[src_id] - env.sim.data.qpos[branch_id]).abs().max()),
        float((env.sim.data.qvel[src_id] - env.sim.data.qvel[branch_id]).abs().max()),
        float((env.sim.data.qacc_warmstart[src_id] - env.sim.data.qacc_warmstart[branch_id]).abs().max()),
      )
      obs_error = float((observation["actor"][src_id] - observation["actor"][branch_id]).abs().max())
      action_error = float((action[src_id] - action[branch_id]).abs().max())
      identity["branches"][name] = {
        "state_error_max": state_error,
        "actor_observation_error_max": obs_error,
        "first_policy_action_error_max": action_error,
        "source_state_sha256": _sha_tensor(torch.cat((env.sim.data.qpos[src_id], env.sim.data.qvel[src_id]))),
        "branch_state_sha256": _sha_tensor(torch.cat((env.sim.data.qpos[branch_id], env.sim.data.qvel[branch_id]))),
        "branch_pass": max(state_error, obs_error, action_error) <= 1.0e-6,
      }
    identity["branch_pass"] = all(row["branch_pass"] for row in identity["branches"].values())
    identities.append(identity)
  return {"observation": observation, "identities": identities}


def _update_streak(streak: torch.Tensor, saturated: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
  return torch.where(valid[:, None] & saturated, streak + 1, torch.zeros_like(streak))


def _window_status(desired_end: int, source_end: int, sham_end: int, probe_end: int) -> str:
  ends = (source_end, sham_end, probe_end)
  if all(end == desired_end for end in ends):
    return "complete"
  if min(ends) < max(ends):
    return "partial_branch_failure"
  return "horizon_censored"


def _saturation_window(
  arrays: dict[str, torch.Tensor], env_id: int, start: int, end: int,
  limits: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor, threshold: float,
) -> dict[str, Any]:
  valid = arrays["pd_valid"][env_id, start:end]
  if end <= start or not bool(valid.any()):
    return {"count": 0, "denominator": 0, "fraction": None, "status": "no_samples"}
  q = arrays["joint_pos"][env_id, start:end]
  qd = arrays["joint_vel"][env_id, start:end]
  target = arrays["target"][env_id, start:end]
  force = arrays["force"][env_id, start:end]
  demand = kp * (target - q) - kd * qd
  mask = valid[:, None] & (force.abs() / limits.clamp_min(1.0e-8) >= threshold) & (demand.abs() >= limits)
  denominator = int(valid.sum()) * force.shape[-1]
  count = int(mask.sum())
  return {"count": count, "denominator": denominator, "fraction": count / denominator if denominator else None, "status": "ok" if denominator else "no_samples"}


def _actuator_snapshot(
  robot: Any, joint_ids: torch.Tensor, force_ids: torch.Tensor, action: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Capture only actuator quantities used by this diagnostic."""
  return {
    "action": action.clone(),
    "joint_pos": robot.data.joint_pos[:, joint_ids].clone(),
    "joint_vel": robot.data.joint_vel[:, joint_ids].clone(),
    "target": robot.data.joint_pos_target[:, joint_ids].clone(),
    "force": robot.data.actuator_force[:, force_ids].clone(),
  }


def _apply_expected_probe_ranges(
  expected: torch.Tensor,
  probe_ids: torch.Tensor,
  global_ids: torch.Tensor,
  applied_mask: torch.Tensor,
  multiplier: float,
) -> torch.Tensor:
  """Scale the world-by-actuator cross product for applied probe worlds."""
  applied_probe_ids = probe_ids[applied_mask]
  if applied_probe_ids.numel() > 0:
    expected[applied_probe_ids[:, None], global_ids[None, :], :] *= multiplier
  return expected


def _failure_delta(source: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
  source_step, branch_step = source.get("failure_step"), branch.get("failure_step")
  return {
    "failed_delta": int(bool(branch.get("failed"))) - int(bool(source.get("failed"))),
    "failure_step_delta": None if source_step is None or branch_step is None else branch_step - source_step,
    "same_reason": source.get("reason") == branch.get("reason"),
  }


def _same_lifecycle(left: dict[str, Any], right: dict[str, Any]) -> bool:
  return all(left.get(name) == right.get(name) for name in ("failed", "reason", "sample_count"))


def _strict_identity_gate(conditions: list[dict[str, Any]]) -> bool:
  return all(
    condition["runtime_identity"]["runtime_pass"]
    and all(row["branch_pass"] for row in condition["initial_identity"])
    and condition["terrain_assignment_position_error_max"] < 1.0e-4
    and condition["terrain_placement_position_error_max"] < 1.0e-4
    and all(
      (
        pair["trigger"]["status"] != "applied"
        or pair["branch_identity"]["branch_pass"]
      )
      and (
        pair["terrain_condition"] != "flat"
        or pair["trigger"]["status"] != "not_triggered"
        or (
          _same_lifecycle(pair["lifecycle"]["source"], pair["lifecycle"]["sham"])
          and _same_lifecycle(pair["lifecycle"]["sham"], pair["lifecycle"]["probe"])
        )
      )
      for pair in condition["pairs"]
    )
    for condition in conditions
  )


def _classify_triplets(
  pairs: list[dict[str, Any]], strict_gate: bool, forced_trigger: bool = False,
) -> dict[str, Any]:
  eligible = [
    pair for pair in pairs
    if pair["trigger"]["status"] == "applied"
    and pair["branch_identity"]["branch_pass"]
    and pair["post_100"]["status"] == "complete"
  ]
  post50 = [pair for pair in eligible if pair["post_50"]["status"] == "complete"]
  post100 = [pair for pair in eligible if pair["post_100"]["status"] == "complete"]
  source_count = sum(pair["post_100"]["source_saturation"]["count"] for pair in post100)
  probe_count = sum(pair["post_100"]["probe_saturation"]["count"] for pair in post100)
  sham_count = sum(pair["post_100"]["sham_saturation"]["count"] for pair in post100)
  reduction = None if source_count == 0 else (source_count - probe_count) / source_count
  sham_noise = abs(sham_count - source_count) / max(source_count, 1)
  wins = sum(pair["lifecycle"]["probe_vs_sham"] > 0 for pair in eligible)
  losses = sum(pair["lifecycle"]["probe_vs_sham"] < 0 for pair in eligible)
  if forced_trigger:
    verdict, reason = "INCONCLUSIVE", "forced trigger is plumbing-only"
  elif not strict_gate:
    verdict, reason = "INCONCLUSIVE", "strict identity/provenance/lifecycle gate failed"
  elif len(post50) < 8 or len(post100) < 8:
    verdict, reason = "INCONCLUSIVE", "triplet post-window coverage below 8"
  elif reduction is None:
    verdict, reason = "INCONCLUSIVE", "no valid saturation samples"
  elif reduction >= 0.50 and wins > losses:
    verdict, reason = "TRIPLET_HEADROOM_DIRECTIONAL", "probe saturation reduction exceeds sham noise and lifecycle direction is favorable"
  elif reduction < 0.50:
    verdict, reason = "HEADROOM_INSUFFICIENT", "probe saturation did not fall by 50%"
  else:
    verdict, reason = "SATURATION_DOWNSTREAM", "probe saturation changed without consistent lifecycle improvement"
  return {
    "verdict": verdict, "reason": reason,
    "event_cohort_size": len(eligible), "post_50_count": len(post50), "post_100_count": len(post100),
    "source_saturation_count": source_count, "sham_saturation_count": sham_count,
    "probe_saturation_count": probe_count, "probe_saturation_reduction": reduction,
    "source_sham_saturation_relative_delta": sham_noise,
    "lifecycle_probe_vs_sham_wins": wins, "lifecycle_probe_vs_sham_losses": losses,
  }


def _run_condition(cfg: TripletConfig, condition: str, terrain_kind: str, terrain_level: int) -> dict[str, Any]:
  scenarios, physical = _physical_scenarios(cfg, condition, terrain_kind, terrain_level)
  n_pairs, num_envs = len(scenarios), len(physical)
  torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
  env_cfg = base.load_env_cfg(cfg.task_id)
  agent_cfg = base.load_rl_cfg(cfg.task_id)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = base._make_gait_generator(cfg.seed)
  env_cfg.scene.num_envs = num_envs; env_cfg.seed = cfg.seed; env_cfg.curriculum = {}
  profile_settings = base._configure_profile(env_cfg, cfg.profile)
  if profile_settings["startup_randomization_events"]:
    raise RuntimeError("clean profile retained startup randomization")
  episode_settings = base._configure_episode_length(env_cfg, cfg.warmup_steps + cfg.sample_steps + 20)
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, base.UniformVelocityCommandCfg):
    raise TypeError("V7 twist command is incompatible")
  command_cfg.heading_command = False; command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0; command_cfg.init_velocity_prob = 0.0
  if hasattr(command_cfg, "focus_terrain_names"): command_cfg.focus_terrain_names = ()
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (0.3, 0.5); command_cfg.ranges.lin_vel_y = (0.0, 0.0); command_cfg.ranges.ang_vel_z = (0.0, 0.0); command_cfg.ranges.heading = None

  env = base.ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  original_reset_idx = env._reset_idx
  try:
    robot = env.scene["robot"]
    joint_ids, joint_names_raw = robot.find_joints((".*",), preserve_order=True)
    joint_names = list(joint_names_raw); force_ids, global_ids = base._joint_control_mapping(robot, joint_names)
    old_limits, active_limits, ranges, runtime_identity = triggered._prepare_limits(env, global_ids)
    active_limits[:] = old_limits
    wrapped = base.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = base.load_runner_cls(cfg.task_id) or base.MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(str(Path(cfg.checkpoint).resolve()), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)
    placement = base._assign_terrain(env, physical, env.device)
    command_term = env.command_manager.get_term("twist")
    command_values = torch.tensor([[row["speed"], 0.0, 0.0] for row in physical], device=env.device)
    command_term.vel_command_b[:] = command_values
    kp = torch.tensor([40.0 if "calf" in name else 20.0 for name in joint_names], device=env.device)
    kd = torch.tensor([2.0 if "calf" in name else 1.0 for name in joint_names], device=env.device)
    shape = (num_envs, cfg.sample_steps)
    arrays = {name: torch.full((*shape, len(joint_names)), torch.nan, device=env.device) for name in ("action", "joint_pos", "joint_vel", "target", "force", "effort_limit")}
    arrays["pd_valid"] = torch.zeros(shape, dtype=torch.bool, device=env.device)
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device); sample_count = torch.zeros(num_envs, dtype=torch.long, device=env.device); reset_count = torch.zeros_like(sample_count)
    first_reason: list[str | None] = [None] * num_envs; first_phase: list[str | None] = [None] * num_envs; first_step: list[int | None] = [None] * num_envs
    current_action = torch.zeros(num_envs, len(joint_names), device=env.device); capture: dict[str, torch.Tensor] = {}; captured_ids = torch.empty(0, dtype=torch.long, device=env.device); capture_reasons: dict[int, str] = {}
    source_ids = torch.arange(0, num_envs, 3, device=env.device); sham_ids = source_ids + 1; probe_ids = source_ids + 2
    streak = torch.zeros(n_pairs, len(joint_names), dtype=torch.long, device=env.device); pending = torch.zeros(n_pairs, dtype=torch.bool, device=env.device); applied = torch.zeros_like(pending); events: list[dict[str, Any] | None] = [None] * n_pairs; identities: list[dict[str, Any] | None] = [None] * n_pairs

    def capture_reset(env_ids: torch.Tensor | None = None) -> None:
      nonlocal capture, captured_ids, capture_reasons
      if env_ids is None: env_ids = torch.arange(num_envs, device=env.device)
      snapshot = _actuator_snapshot(robot, joint_ids, force_ids, current_action)
      captured_ids = env_ids.clone(); capture = {key: value.index_select(0, env_ids).clone() for key, value in snapshot.items()}; capture_reasons = {int(i): base._contact_termination(env, int(i)) for i in env_ids.tolist()}; original_reset_idx(env_ids)
    env._reset_idx = capture_reset  # type: ignore[method-assign]
    initial = _branch_triplet(
      env, wrapped, policy, robot, command_term, source_ids, sham_ids, probe_ids
    )
    observation = initial["observation"]
    initial_identity = initial["identities"]
    if not all(row["branch_pass"] for row in initial_identity):
      raise RuntimeError("initial triplet identity failed")
    for step in range(cfg.warmup_steps + cfg.sample_steps):
      command_term.vel_command_b[:] = command_values
      with torch.inference_mode(): action = policy(observation)
      pending_ids = torch.where(pending)[0]
      if len(pending_ids):
        probes = probe_ids[pending_ids]
        ranges[probes[:, None], global_ids, :] = env.sim.get_default_field("actuator_forcerange")[global_ids, :].unsqueeze(0) * MULTIPLIERS[2]
        active_limits[probes] = old_limits * MULTIPLIERS[2]; pending[pending_ids] = False; applied[pending_ids] = True
        for pair_id in pending_ids.tolist(): events[pair_id]["apply_step"] = step - cfg.warmup_steps; events[pair_id]["status"] = "applied"  # type: ignore[index]
      step_limits = active_limits.clone(); current_action = action.detach().clone(); capture = {}; captured_ids = torch.empty(0, dtype=torch.long, device=env.device); capture_reasons = {}
      _, _, dones, _ = wrapped.step(action); reset = dones.bool(); observation = wrapped.get_observations()
      state = _actuator_snapshot(robot, joint_ids, force_ids, action)
      if capture:
        for key, value in capture.items(): state[key][captured_ids] = value
      if step < cfg.warmup_steps:
        for env_id in torch.where(reset & active)[0].tolist(): first_reason[env_id] = capture_reasons.get(env_id, "reset"); first_phase[env_id] = "warmup"; first_step[env_id] = step
        active &= ~reset; continue
      k = step - cfg.warmup_steps; write = active.clone()
      for key, value in state.items():
        if key not in arrays: continue
        arrays[key][:, k] = torch.where(write[:, None], value if value.ndim == 2 else value, arrays[key][:, k])
      arrays["effort_limit"][:, k] = torch.where(write[:, None], step_limits, arrays["effort_limit"][:, k]); arrays["pd_valid"][:, k] = write & ~reset
      sample_count += active.long(); reset_count += (reset & active).long()
      for env_id in torch.where(reset & active)[0].tolist(): first_reason[env_id] = capture_reasons.get(env_id, "reset"); first_phase[env_id] = "sample"; first_step[env_id] = k
      source_valid = active[source_ids] & ~reset[source_ids] & ~applied
      demand = kp * (state["target"][source_ids] - state["joint_pos"][source_ids]) - kd * state["joint_vel"][source_ids]
      saturated = (state["force"][source_ids].abs() / old_limits.clamp_min(1.0e-8) >= cfg.saturation_threshold) & (demand.abs() >= old_limits)
      streak = triggered._update_streak(streak, saturated, source_valid)
      forced = torch.zeros(n_pairs, dtype=torch.bool, device=env.device)
      if (
        cfg.forced_trigger_step is not None
        and condition == "slope_up_high"
        and k == cfg.forced_trigger_step
      ):
        forced[0] = bool(source_valid[0])
      natural = (streak >= cfg.trigger_run_steps).any(dim=1)
      new_pairs = torch.where((natural | forced) & ~applied & ~pending)[0]
      if len(new_pairs):
        out = _branch_triplet(env, wrapped, policy, robot, command_term, source_ids[new_pairs], sham_ids[new_pairs], probe_ids[new_pairs])
        observation = out["observation"]
        for offset, pair_id in enumerate(new_pairs.tolist()):
          trigger_joints = streak[pair_id] >= cfg.trigger_run_steps
          if forced[pair_id] and not bool(trigger_joints.any()):
            trigger_joints[0] = True
          events[pair_id] = {"status": "pending", "forced": bool(forced[pair_id]), "onset_step": k - cfg.trigger_run_steps + 1, "detect_step": k, "apply_step": None, "joints": [joint_names[j] for j in torch.where(trigger_joints)[0].tolist()]}
          identities[pair_id] = out["identities"][offset]
        for pair_id in new_pairs.tolist():
          for value in arrays.values(): value[probe_ids[pair_id], :k + 1] = value[source_ids[pair_id], :k + 1].clone(); value[sham_ids[pair_id], :k + 1] = value[source_ids[pair_id], :k + 1].clone()
          sample_count[sham_ids[pair_id]] = sample_count[source_ids[pair_id]]; sample_count[probe_ids[pair_id]] = sample_count[source_ids[pair_id]]; reset_count[sham_ids[pair_id]] = reset_count[source_ids[pair_id]]; reset_count[probe_ids[pair_id]] = reset_count[source_ids[pair_id]]
          source_id = int(source_ids[pair_id])
          for branch_id_tensor in (sham_ids[pair_id], probe_ids[pair_id]):
            branch_id = int(branch_id_tensor)
            first_reason[branch_id] = first_reason[source_id]
            first_phase[branch_id] = first_phase[source_id]
            first_step[branch_id] = first_step[source_id]
            reset[branch_id_tensor] = False
          active[sham_ids[pair_id]] = active[source_ids[pair_id]]; active[probe_ids[pair_id]] = active[source_ids[pair_id]]
        pending[new_pairs] = True
      active &= ~reset
    rows = []
    for env_id, scenario in enumerate(physical):
      rows.append({**scenario, "sample_count": int(sample_count[env_id]), "reset_count": int(reset_count[env_id]), "failed": first_reason[env_id] is not None, "first_failure_reason": first_reason[env_id], "first_failure_phase": first_phase[env_id], "first_failure_step": first_step[env_id]})
    pairs = []
    for pair_id, scenario in enumerate(scenarios):
      source, sham, probe = 3 * pair_id, 3 * pair_id + 1, 3 * pair_id + 2; event = events[pair_id] or {"status": "not_triggered", "onset_step": None, "detect_step": None, "apply_step": None, "joints": []}; identity = identities[pair_id] or {"branch_pass": True, "branches": {}, "status": "not_applicable_no_trigger"}; apply = event.get("apply_step")
      lifecycle = {"source": {"failed": rows[source]["failed"], "reason": rows[source]["first_failure_reason"], "failure_step": rows[source]["first_failure_step"], "sample_count": rows[source]["sample_count"]}, "sham": {"failed": rows[sham]["failed"], "reason": rows[sham]["first_failure_reason"], "failure_step": rows[sham]["first_failure_step"], "sample_count": rows[sham]["sample_count"]}, "probe": {"failed": rows[probe]["failed"], "reason": rows[probe]["first_failure_reason"], "failure_step": rows[probe]["first_failure_step"], "sample_count": rows[probe]["sample_count"]}}
      sham_end = cfg.sample_steps if lifecycle["sham"]["failure_step"] is None else lifecycle["sham"]["failure_step"]
      probe_end = cfg.sample_steps if lifecycle["probe"]["failure_step"] is None else lifecycle["probe"]["failure_step"]
      lifecycle["probe_vs_sham"] = probe_end - sham_end
      lifecycle["source_sham_delta"] = _failure_delta(lifecycle["source"], lifecycle["sham"])
      lifecycle["probe_sham_delta"] = _failure_delta(lifecycle["sham"], lifecycle["probe"])
      windows: dict[str, Any] = {}
      for size in cfg.post_windows:
        if apply is None: windows[f"post_{size}"] = {"status": "not_applicable_no_trigger"}; continue
        ends = [min(rows[e]["sample_count"], apply + size) for e in (source, sham, probe)]; end = min(ends); status = _window_status(apply + size, *ends)
        source_sat = _saturation_window(arrays, source, apply, end, old_limits, kp, kd, cfg.saturation_threshold)
        sham_sat = _saturation_window(arrays, sham, apply, end, old_limits, kp, kd, cfg.saturation_threshold)
        probe_sat = _saturation_window(arrays, probe, apply, end, old_limits * MULTIPLIERS[2], kp, kd, cfg.saturation_threshold)
        windows[f"post_{size}"] = {
          "status": status,
          "common_steps": max(end - apply, 0),
          "source_saturation": source_sat,
          "sham_saturation": sham_sat,
          "probe_saturation": probe_sat,
          "paired_deltas": {
            "source_minus_sham_count": source_sat["count"] - sham_sat["count"],
            "probe_minus_sham_count": probe_sat["count"] - sham_sat["count"],
            "source_minus_sham_fraction": None if source_sat["fraction"] is None or sham_sat["fraction"] is None else source_sat["fraction"] - sham_sat["fraction"],
            "probe_minus_sham_fraction": None if probe_sat["fraction"] is None or sham_sat["fraction"] is None else probe_sat["fraction"] - sham_sat["fraction"],
          },
        }
      pairs.append({**scenario, "trigger": event, "branch_identity": identity, "lifecycle": lifecycle, **windows})
    expected_end = env.sim.get_default_field("actuator_forcerange").unsqueeze(0).repeat(num_envs, 1, 1)
    applied_mask = torch.tensor(
      [event is not None and event.get("apply_step") is not None for event in events],
      dtype=torch.bool,
      device=env.device,
    )
    _apply_expected_probe_ranges(
      expected_end, probe_ids, global_ids, applied_mask, MULTIPLIERS[2]
    )
    range_drift = float((ranges - expected_end).abs().max())
    demand = kp * (arrays["target"] - arrays["joint_pos"]) - kd * arrays["joint_vel"]
    expected_force = torch.clamp(demand, min=-arrays["effort_limit"], max=arrays["effort_limit"])
    pd_valid = torch.isfinite(arrays["force"]) & arrays["pd_valid"][:, :, None]
    clamp_error = float((arrays["force"][pd_valid] - expected_force[pd_valid]).abs().max()) if bool(pd_valid.any()) else None
    force_excess = float((arrays["force"].abs() - arrays["effort_limit"]).masked_fill(~torch.isfinite(arrays["force"]), -torch.inf).max()) if bool(torch.isfinite(arrays["force"]).any()) else None
    runtime_identity.update({
      "runtime_range_drift_max": range_drift,
      "actuator_force_clamp_error_max": clamp_error,
      "actuator_force_limit_excess_max": force_excess,
      "runtime_pass": range_drift <= 1.0e-6 and clamp_error is not None
      and clamp_error <= 1.0e-3 and force_excess is not None and force_excess <= 1.0e-4,
    })
    return {"terrain_condition": condition, "terrain_kind": terrain_kind, "terrain_level": terrain_level, "profile_settings": profile_settings, "episode_settings": episode_settings, "runtime_identity": runtime_identity, "initial_identity": initial_identity, "terrain_assignment_position_error_max": float(placement["terrain_assignment_position_error_max"]), "terrain_placement_position_error_max": float(placement["terrain_placement_position_error_max"]), "rows": rows, "pairs": pairs}
  finally:
    env._reset_idx = original_reset_idx  # type: ignore[method-assign]
    env.close()


def evaluate(cfg: TripletConfig) -> dict[str, Any]:
  _validate_config(cfg); base.configure_torch_backends()
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  conditions = [_run_condition(cfg, condition, kind, level) for condition, kind, level in TRIGGER_CONDITIONS]
  strict = _strict_identity_gate(conditions)
  payload = {"schema_version": 1, "evaluation_suite": "go2_actuator_headroom_triplet", "git_head": base._git_head(), "evaluator_source": str(Path(__file__).resolve()), "evaluator_source_sha256": base._sha256(Path(__file__).resolve()), "checkpoint": str(checkpoint), "checkpoint_sha256": base._sha256(checkpoint), "config": asdict(cfg), "arms": [{"name": name, "multiplier": multiplier} for name, multiplier in zip(ARMS, MULTIPLIERS, strict=True)], "single_variable_contract": {"trigger": "source same joint saturated for three consecutive valid post-step rows", "application": "probe only, detect+1", "unchanged": ["checkpoint", "terrain", "command", "termination", "gait", "observation", "reward", "network", "PPO"]}, "conditions": conditions, "strict_gate_pass": strict}
  payload["causal_decision"] = _classify_triplets(
    [pair for condition in conditions for pair in condition["pairs"]],
    strict,
    cfg.forced_trigger_step is not None,
  )
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  cfg = tyro.cli(TripletConfig); payload = evaluate(cfg); output = Path(cfg.output_file).expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n"); print(json.dumps({"output": str(output), "strict_gate_pass": payload["strict_gate_pass"], "causal_decision": payload["causal_decision"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
