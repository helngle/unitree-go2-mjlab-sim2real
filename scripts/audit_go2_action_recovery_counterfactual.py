"""Strict matched action-stability counterfactual for V7.

Probe worlds apply a pre-registered 0.5 two-tap FIR blend to policy actions.
Source and sham use the same wrapper with no blending.  This is evaluation-only
and does not modify reward, terrain, termination, observations, network, PPO or
assets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import tyro
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_go2_friction_contact_causal as base
from scripts import audit_go2_foot_placement_counterfactual as foot
from scripts import diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


EXPECTED_CHECKPOINT_SHA256 = foot.EXPECTED_CHECKPOINT_SHA256
BLEND = 0.5
FRICTION = 0.6
ACTION_ACCEL_ONSET_THRESHOLD = 0.75
SATURATION_THRESHOLD = 0.98
REQUIRED_TRIPLETS = 8


@dataclass(frozen=True)
class RecoveryConfig:
  checkpoint: str = gait.V7_CHECKPOINT
  profiles: tuple[str, ...] = ("clean",)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  formal: bool = True
  blend: float = BLEND
  output_file: str = "go2_action_recovery_counterfactual_strict.json"


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_config(cfg: RecoveryConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(gait.V7_CHECKPOINT).resolve() or _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("checkpoint is not locked V7 model_13600.pt")
  if cfg.profiles != ("clean",) or cfg.speeds != (0.3, 0.5) or cfg.seed != 42:
    raise ValueError("profile/speeds/seed are locked")
  if not math.isclose(float(cfg.blend), BLEND, abs_tol=1.0e-8):
    raise ValueError("blend is pre-registered at 0.5")
  if cfg.formal and (cfg.repeats < 8 or cfg.warmup_steps != 100 or cfg.sample_steps < 1200):
    raise ValueError("formal matrix requires >=8 repeats, warmup=100, sample>=1200")
  if not cfg.formal and (cfg.repeats <= 0 or cfg.sample_steps < 10):
    raise ValueError("invalid smoke matrix")


def _slots(cfg: RecoveryConfig, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for speed_index, speed in enumerate(cfg.speeds):
    for repeat in range(cfg.repeats):
      matched = speed_index * cfg.repeats + repeat
      for arm_index, arm in enumerate(("source", "sham", "probe")):
        rows.append({
          "matched_slot": matched, "terrain_condition": condition,
          "terrain_kind": kind, "terrain_level": level, "speed": float(speed),
          "command_name": f"forward_{speed:g}", "repeat": repeat,
          "speed_index": speed_index, "arm": arm, "arm_index": arm_index,
          "friction": FRICTION, "intervention": "policy_action_two_tap_fir_blend",
          "blend": BLEND if arm == "probe" else 0.0,
        })
  return rows


def _blend_actions(
  actions: torch.Tensor, previous_policy_action: torch.Tensor,
  probe_mask: torch.Tensor, blend: float,
) -> torch.Tensor:
  """Apply the registered FIR action blend without changing source/sham."""
  expected = (1.0 - blend) * actions + blend * previous_policy_action
  return torch.where(probe_mask[:, None], expected, actions)


class _RecoveryTrace(base._TraceWrapper):
  def __init__(self, env: Any, clip_actions: float | None, traces: list[Any], blend: float):
    super().__init__(env, clip_actions, traces)
    self.blend = float(blend)
    self._probe_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._previous_policy_action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    self._current_raw_action = torch.zeros_like(self._previous_policy_action)
    self._current_executed_action = torch.zeros_like(self._previous_policy_action)
    robot = env.scene["robot"]
    self._joint_ids, joint_names = robot.find_joints((".*",), preserve_order=True)
    self._joint_names = list(joint_names)
    action_term = env.action_manager.get_term("joint_pos")
    if list(action_term.target_names) != self._joint_names:
      raise RuntimeError("action target and joint ordering differ")
    force_ids = torch.full(
      (len(self._joint_names),), -1, dtype=torch.long, device=env.device
    )
    for actuator in robot.actuators:
      for target_name, ctrl_id in zip(
        actuator.target_names, actuator.ctrl_ids.tolist(), strict=True
      ):
        if target_name in self._joint_names:
          force_ids[self._joint_names.index(target_name)] = int(ctrl_id)
    if bool((force_ids < 0).any()):
      raise RuntimeError(f"actuator force mapping incomplete: {force_ids.tolist()}")
    self._force_ids = force_ids
    self._kp = torch.tensor(
      [40.0 if "calf" in name else 20.0 for name in self._joint_names],
      device=env.device,
    )
    self._kd = torch.tensor(
      [2.0 if "calf" in name else 1.0 for name in self._joint_names],
      device=env.device,
    )
    self._effort_limits = torch.tensor(
      [45.0 if "calf" in name else 23.5 for name in self._joint_names],
      device=env.device,
    )

  def set_context(self, scenarios: list[dict[str, Any]]) -> None:
    super().set_context(scenarios)
    self._probe_mask = torch.tensor(
      [row["arm"] == "probe" for row in scenarios], dtype=torch.bool, device=self.env.device
    )

  def _capture(self) -> dict[str, torch.Tensor | None]:
    state = super()._capture()
    robot = self.env.scene["robot"]
    q = robot.data.joint_pos[:, self._joint_ids]
    qd = robot.data.joint_vel[:, self._joint_ids]
    target = robot.data.joint_pos_target[:, self._joint_ids]
    force = robot.data.actuator_force[:, self._force_ids]
    demand = self._kp * (target - q) - self._kd * qd
    utilization = force.abs() / self._effort_limits.clamp_min(1.0e-8)
    saturated = (
      (utilization >= SATURATION_THRESHOLD)
      & (demand.abs() >= self._effort_limits)
    )
    state.update({
      "raw_action_abs": self._current_raw_action.abs().mean(dim=-1),
      "executed_action_abs": self._current_executed_action.abs().mean(dim=-1),
      "position_error": (target - q).abs().mean(dim=-1),
      "joint_velocity": qd.abs().mean(dim=-1),
      "pd_demand": demand.abs().mean(dim=-1),
      "actuator_force": force.abs().mean(dim=-1),
      "effort_utilization": utilization.mean(dim=-1),
      "saturation_fraction": saturated.float().mean(dim=-1),
    })
    return state

  def step(self, actions: torch.Tensor):
    raw = actions.detach()
    expected_probe = (1.0 - self.blend) * raw + self.blend * self._previous_policy_action
    filtered = _blend_actions(raw, self._previous_policy_action, self._probe_mask, self.blend)
    self._current_raw_action = raw
    self._current_executed_action = filtered
    formula_error = torch.where(
      self._probe_mask[:, None], filtered - expected_probe, filtered - raw
    ).abs().amax(dim=-1)
    no_op_error = torch.where(
      (~self._probe_mask)[:, None], filtered - raw, torch.zeros_like(filtered)
    ).abs().amax(dim=-1)
    result = super().step(filtered)
    dones = result[2].bool()
    self._previous_policy_action = torch.where(
      dones[:, None], torch.zeros_like(raw), raw
    )
    self.trace_rows[-1]["recovery_intervention_active"] = self._probe_mask.detach().cpu().tolist()
    self.trace_rows[-1]["recovery_formula_error"] = formula_error.detach().cpu().tolist()
    self.trace_rows[-1]["source_sham_no_op_error"] = no_op_error.detach().cpu().tolist()
    return result

  def summaries(self, warmup_steps: int, sample_steps: int) -> list[dict[str, Any]]:
    output = super().summaries(warmup_steps, sample_steps)
    extra_keys = (
      "raw_action_abs", "executed_action_abs", "position_error", "joint_velocity",
      "pd_demand", "actuator_force", "effort_utilization", "saturation_fraction",
    )
    for i, item in enumerate(output):
      start, end = warmup_steps, warmup_steps + sample_steps
      active = [bool(row["active"][i]) for row in self.trace_rows[start:end]]
      for key in extra_keys:
        values: list[float | None] = []
        for is_active, row in zip(active, self.trace_rows[start:end], strict=True):
          value = float(row[key][i])
          values.append(value if is_active and math.isfinite(value) else None)
        item["series"][key] = values
      formula_error = max(
        (float(row["recovery_formula_error"][i]) for row in self.trace_rows),
        default=float("inf"),
      )
      no_op_error = max(
        (float(row["source_sham_no_op_error"][i]) for row in self.trace_rows),
        default=float("inf"),
      )
      item["recovery_formula_error_max"] = formula_error
      item["source_sham_no_op_error_max"] = no_op_error
      item["recovery_identity_pass"] = bool(
        math.isfinite(formula_error) and formula_error <= 1.0e-7
        and math.isfinite(no_op_error) and no_op_error <= 1.0e-7
      )
      item["recovery_active_steps"] = int(sum(
        bool(row["recovery_intervention_active"][i]) and bool(row["active"][i])
        for row in self.trace_rows[start:end]
      ))
      item["action_instability_onset_step"] = base._onset(
        item["series"]["action_acc"], ACTION_ACCEL_ONSET_THRESHOLD
      )
      item["actuator_saturation_onset_step"] = base._onset(
        item["series"]["saturation_fraction"], 1.0 / len(self._joint_names)
      )
      item["actuator_metrics"] = {
        key: base._stats(item["series"][key]) for key in extra_keys
      }
      item["joint_names"] = self._joint_names
    return output


def _delta_from_pairs(
  pairs: list[dict[str, Any]], metric: str, lower_better: bool,
) -> dict[str, Any]:
  effect: list[float] = []
  noise: list[float] = []
  sign = -1.0 if lower_better else 1.0
  for pair in pairs:
    values = pair["metrics"].get(metric, {})
    if any(values.get(arm) is None for arm in ("source", "sham", "probe")):
      continue
    effect.append(sign * (values["probe"] - values["sham"]))
    noise.append(sign * (values["source"] - values["sham"]))
  return {
    "n_valid": len(effect), "effect_values": effect, "noise_values": noise,
    "effect_mean": None if not effect else float(np.mean(effect)),
    "noise_abs_mean": None if not noise else float(np.mean(np.abs(noise))),
    "direction_fraction": None if not effect else float(np.mean(np.asarray(effect) > 0)),
  }


def _recovery_cell(
  rows: list[dict[str, Any]], traces: dict[tuple[str, int, str], dict[str, Any]],
  condition: str, speed: float,
) -> dict[str, Any]:
  cell = base._paired(rows, traces, condition, speed)
  by_slot = {int(pair["matched_slot"]): pair for pair in cell["pairs"]}
  for slot, pair in by_slot.items():
    trace_arms = {arm: traces[(condition, slot, arm)] for arm in ("source", "sham", "probe")}
    prefix = int(pair["common_prefix_steps"])
    for metric in (
      "raw_action_abs", "executed_action_abs", "position_error", "joint_velocity",
      "pd_demand", "actuator_force", "effort_utilization", "saturation_fraction",
    ):
      pair["metrics"][metric] = {
        arm: base._mean_prefix(trace_arms[arm], metric, prefix)
        for arm in ("source", "sham", "probe")
      }
    pair["action_onset_ordering"] = {
      arm: (
        trace_arms[arm]["failure_step"] is None
        or (
          trace_arms[arm]["action_instability_onset_step"] is not None
          and trace_arms[arm]["action_instability_onset_step"]
          <= trace_arms[arm]["failure_step"] + 1
        )
      ) for arm in ("source", "sham", "probe")
    }
    pair["actuator_onset_ordering"] = {
      arm: (
        trace_arms[arm]["failure_step"] is None
        or (
          trace_arms[arm]["actuator_saturation_onset_step"] is not None
          and trace_arms[arm]["actuator_saturation_onset_step"]
          <= trace_arms[arm]["failure_step"] + 1
        )
      ) for arm in ("source", "sham", "probe")
    }
  for metric, lower in {
    "raw_action_abs": True, "executed_action_abs": True, "position_error": True,
    "joint_velocity": True, "pd_demand": True, "actuator_force": True,
    "effort_utilization": True, "saturation_fraction": True,
  }.items():
    cell["effects"][metric] = _delta_from_pairs(cell["pairs"], metric, lower)
  return cell


def _recovery_gate(cell: dict[str, Any], seed: int) -> dict[str, Any]:
  base_gate = base._cell_gate(cell, seed)
  action = cell["effects"]["action_acc"]
  effect = action["effect_values"]
  noise = action["noise_values"]
  excess = [value - abs(natural) for value, natural in zip(effect, noise, strict=True)]
  action_bootstrap = {
    "effect": base._bootstrap(effect, seed + 401),
    "excess_over_source_sham_noise": base._bootstrap(excess, seed + 402),
  }
  action_ci_pass = bool(
    action_bootstrap["effect"]["ci95"][0] is not None
    and action_bootstrap["effect"]["ci95"][0] > 0
    and action_bootstrap["excess_over_source_sham_noise"]["ci95"][0] > 0
  )
  action_direction_pass = bool(
    action["direction_fraction"] is not None and action["direction_fraction"] >= 0.75
  )
  onset_pass = all(
    all(pair["action_onset_ordering"].values()) for pair in cell["pairs"]
  )
  base_gate.update({
    "action_bootstrap": action_bootstrap,
    "action_direction_pass": action_direction_pass,
    "action_ci_pass": action_ci_pass,
    "action_onset_ordering_pass": onset_pass,
    "recovery_causal_pass": bool(
      base_gate["contact_causal_pass"] and action_direction_pass
      and action_ci_pass and onset_pass
    ),
  })
  return base_gate


def _flat_sentinel_gate(cell: dict[str, Any]) -> dict[str, Any]:
  pairs = cell["pairs"]
  catastrophic = sum(
    pair["completion"]["sham"] and not pair["completion"]["probe"]
    for pair in pairs
  )
  sham_gain = [
    pair["metrics"]["gain"]["sham"] for pair in pairs
    if pair["metrics"]["gain"]["sham"] is not None
  ]
  probe_gain = [
    pair["metrics"]["gain"]["probe"] for pair in pairs
    if pair["metrics"]["gain"]["probe"] is not None
  ]
  gain_ratio = None
  if sham_gain and probe_gain and abs(float(np.mean(sham_gain))) > 1.0e-8:
    gain_ratio = float(np.mean(probe_gain) / np.mean(sham_gain))
  return {
    "coverage_pass": bool(cell["coverage_pass"]),
    "sham_complete_to_probe_fail_count": int(catastrophic),
    "gain_probe_over_sham": gain_ratio,
    "pass": bool(
      cell["coverage_pass"] and catastrophic == 0
      and gain_ratio is not None and gain_ratio >= 0.90
    ),
  }


def evaluate(cfg: RecoveryConfig) -> dict[str, Any]:
  _validate_config(cfg)
  original_slots, original_assign, original_wrapper = (
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper
  )
  traces: list[_RecoveryTrace] = []
  runtime: dict[str, dict[str, Any]] = {}

  def slot_fn(_cfg: Any, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
    return _slots(cfg, condition, kind, level)

  def wrapper_fn(env: Any, clip_actions: float | None = None) -> _RecoveryTrace:
    return _RecoveryTrace(env, clip_actions, traces, cfg.blend)

  def assign_fn(env: Any, scenarios: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    placement = original_assign(env, scenarios, device)
    for sensor in env.scene._sensors.values():
      sensor._invalidate_cache()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    if not traces:
      raise RuntimeError("recovery wrapper missing before assignment")
    traces[-1].set_context(scenarios)
    runtime[str(scenarios[0]["terrain_condition"])] = {
      "runtime_pass": False, "blend": float(cfg.blend),
      "friction": FRICTION, "placement": placement,
      "intervention": "two_tap_fir: executed=(1-blend)*current_policy+blend*previous_policy",
    }
    return placement

  gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = slot_fn, assign_fn, wrapper_fn
  try:
    payload = gait.evaluate(gait.GaitConfig(
      checkpoint=cfg.checkpoint, profiles=cfg.profiles, speeds=cfg.speeds,
      repeats=cfg.repeats, warmup_steps=cfg.warmup_steps, sample_steps=cfg.sample_steps,
      seed=cfg.seed, device=cfg.device, output_file="unused.json",
    ))
  finally:
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = original_slots, original_assign, original_wrapper

  conditions = payload["profiles"]["clean"]["conditions"]
  trace_map: dict[tuple[str, int, str], dict[str, Any]] = {}
  for trace in traces:
    for scenario, summary in zip(trace.scenarios, trace.summaries(cfg.warmup_steps, cfg.sample_steps), strict=True):
      trace_map[(str(scenario["terrain_condition"]), int(scenario["matched_slot"]), str(scenario["arm"]))] = summary
  all_rows: list[dict[str, Any]] = []
  for condition in conditions.values():
    for row in condition["scenarios"]:
      row["causal_trace"] = trace_map[(str(row["terrain_condition"]), int(row["matched_slot"]), str(row["arm"]))]
      all_rows.append(row)
  for condition_name, item in runtime.items():
    summaries = [
      summary for (condition, _slot, _arm), summary in trace_map.items()
      if condition == condition_name
    ]
    item["formula_error_max"] = max(
      (summary["recovery_formula_error_max"] for summary in summaries), default=None
    )
    item["source_sham_no_op_error_max"] = max(
      (summary["source_sham_no_op_error_max"] for summary in summaries), default=None
    )
    item["identity_rows"] = len(summaries)
    item["runtime_pass"] = bool(
      summaries and all(summary["recovery_identity_pass"] for summary in summaries)
    )
  cells = [_recovery_cell(all_rows, trace_map, "slope_up_high", speed) for speed in cfg.speeds]
  gates = [_recovery_gate(cell, cfg.seed + index * 1000) for index, cell in enumerate(cells)]
  flat_cells = [_recovery_cell(all_rows, trace_map, "flat", speed) for speed in cfg.speeds]
  flat_gates = [_flat_sentinel_gate(cell) for cell in flat_cells]
  runtime_pass = bool(runtime) and all(item["runtime_pass"] for item in runtime.values())
  recovery_causal = bool(
    runtime_pass and all(gate["recovery_causal_pass"] for gate in gates)
    and all(gate["pass"] for gate in flat_gates)
  )
  result: dict[str, Any] = {
    "schema_version": 2, "evaluation_suite": "go2_action_recovery_counterfactual_strict",
    "checkpoint": str(Path(cfg.checkpoint).resolve()), "checkpoint_sha256": _sha256(cfg.checkpoint),
    "evaluator_source": str(Path(__file__).resolve()), "evaluator_source_sha256": _sha256(__file__),
    "evaluator_sha256": _sha256(__file__), "config": asdict(cfg),
    "arms": [{"name": arm, "blend": BLEND if arm == "probe" else 0.0, "friction": FRICTION} for arm in ("source", "sham", "probe")],
    "runtime_intervention_identity": runtime,
    "matched_coverage": {cell["cell"]: {"matched_triplets": cell["matched_triplets"], "required": cell["required_triplets"], "pass": cell["coverage_pass"]} for cell in cells},
    "cells": cells, "cell_acceptance": {cell["cell"]: gate for cell, gate in zip(cells, gates, strict=True)},
    "flat_sentinel": {cell["cell"]: gate for cell, gate in zip(flat_cells, flat_gates, strict=True)},
    "verdict": "RECOVERY_CAUSAL" if recovery_causal else "INCONCLUSIVE",
    "training_ready": False,
    "primary_cause": "policy action instability mediated through joint-position/PD demand" if recovery_causal else None,
    "causal_evidence": {"runtime_pass": runtime_pass, "per_cell": gates, "single_intervention": "probe two-tap FIR action blend=0.5"},
    "rejected_alternatives": {"actuator_headroom": "strongly de-prioritized as the sole/primary explanation; not formally excluded", "friction": "direct friction probes show strong speed-dependent benefit but strict guardrails fail", "foot_placement_plus_0p05": "strongly harmful"},
    "effect_vs_sham_noise": {cell["cell"]: cell["effects"] for cell in cells},
    "bootstrap_ci": {cell["cell"]: {"contact": gate["bootstrap"], "action": gate["action_bootstrap"]} for cell, gate in zip(cells, gates, strict=True)},
    "measurement_limits": ["fixed two-tap FIR smoothing is an evaluator action-stability proxy, not a learned recovery policy", "action smoothing also changes PD demand and actuator utilization, which are reported as a mediation path", "no real hardware latency/thermal model"],
    "next_training_variable": "continue contact-causal evaluation; do not train" if recovery_causal else "continue strict matched evaluation; do not train",
    "source_gait_payload": payload,
  }
  result = foot._jsonable(result)
  canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  result["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  gait.configure_torch_backends()
  cfg = tyro.cli(RecoveryConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  output.with_suffix(output.suffix + ".sha256").write_text(f"{_sha256(output)}  {output.name}\n")
  print(json.dumps({"output": str(output), "verdict": result["verdict"], "training_ready": result["training_ready"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  main()
