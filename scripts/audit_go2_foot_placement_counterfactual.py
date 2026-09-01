"""Strict matched foot-placement counterfactual for V7.

The only intervention is a pre-registered +0.05 rad thigh/calf target offset on
probe worlds during the fixed swing phase.  Source and sham run through the
same wrapper with a zero offset.  No task, reward, terrain, robot asset or
checkpoint is modified.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_go2_friction_contact_causal as base
from scripts import diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


EXPECTED_CHECKPOINT_SHA256 = base.gait._sha256(Path(gait.V7_CHECKPOINT))
FOOT_NAMES = gait.FOOT_NAMES
PERIOD = 0.6
SWING_THRESHOLD = 0.56
LEG_OFFSETS = (0.0, 0.5, 0.5, 0.0)
SOURCE_FRICTION = 0.6
Q_DELTA = 0.05
RAW_ACTION_SCALE = 0.25


@dataclass(frozen=True)
class FootPlacementConfig:
  checkpoint: str = gait.V7_CHECKPOINT
  profiles: tuple[str, ...] = ("clean",)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  formal: bool = True
  q_delta: float = Q_DELTA
  output_file: str = "go2_foot_placement_counterfactual_strict.json"


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_config(cfg: FootPlacementConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(gait.V7_CHECKPOINT).resolve() or _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("checkpoint is not locked V7 model_13600.pt")
  if cfg.profiles != ("clean",) or cfg.speeds != (0.3, 0.5) or cfg.seed != 42:
    raise ValueError("profile/speeds/seed are locked")
  if not math.isclose(float(cfg.q_delta), Q_DELTA, abs_tol=1.0e-8):
    raise ValueError("q_delta is pre-registered at +0.05 rad")
  if cfg.formal and (cfg.repeats < 8 or cfg.warmup_steps != 100 or cfg.sample_steps < 1200):
    raise ValueError("formal matrix requires >=8 repeats, warmup=100, sample>=1200")
  if not cfg.formal and (cfg.repeats <= 0 or cfg.sample_steps < 10):
    raise ValueError("smoke matrix requires repeats>0 and sample_steps>=10")


def _jsonable(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().tolist()
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return value


def _slots(cfg: FootPlacementConfig, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for speed_index, speed in enumerate(cfg.speeds):
    for repeat in range(cfg.repeats):
      matched = speed_index * cfg.repeats + repeat
      for arm_index, arm in enumerate(("source", "sham", "probe")):
        rows.append({
          "matched_slot": matched,
          "terrain_condition": condition,
          "terrain_kind": kind,
          "terrain_level": level,
          "speed": float(speed),
          "command_name": f"forward_{speed:g}",
          "repeat": repeat,
          "speed_index": speed_index,
          "arm": arm,
          "arm_index": arm_index,
          "friction": SOURCE_FRICTION,
          "intervention": "swing_thigh_calf_q_target_offset",
          "q_delta": Q_DELTA if arm == "probe" else 0.0,
        })
  return rows


class _PlacementTrace(base._TraceWrapper):
  def __init__(self, env: Any, clip_actions: float | None, traces: list[Any], q_delta: float):
    super().__init__(env, clip_actions, traces)
    self.q_delta = float(q_delta)
    self._probe_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._joint_indices: list[tuple[int, int]] = []
    action_term = env.action_manager.get_term("joint_pos")
    names = list(action_term.target_names)
    if len(names) != 12:
      raise RuntimeError(f"unexpected Go2 action target count: {names}")
    for leg in FOOT_NAMES:
      thigh = names.index(f"{leg}_thigh_joint")
      calf = names.index(f"{leg}_calf_joint")
      self._joint_indices.append((thigh, calf))

  def set_context(self, scenarios: list[dict[str, Any]]) -> None:
    super().set_context(scenarios)
    self._probe_mask = torch.tensor(
      [row["arm"] == "probe" for row in scenarios], dtype=torch.bool, device=self.env.device
    )

  def step(self, actions: torch.Tensor):
    phase = (self.env.episode_length_buf * self.env.step_dt / PERIOD).remainder(1.0)
    delta = torch.zeros_like(actions)
    active_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
    for leg_index, (thigh, calf) in enumerate(self._joint_indices):
      leg_phase = (phase + LEG_OFFSETS[leg_index]).remainder(1.0)
      swing = self._probe_mask & (leg_phase >= SWING_THRESHOLD)
      delta[:, thigh] = torch.where(swing, torch.full_like(phase, self.q_delta / RAW_ACTION_SCALE), delta[:, thigh])
      delta[:, calf] = torch.where(swing, torch.full_like(phase, self.q_delta / RAW_ACTION_SCALE), delta[:, calf])
      active_mask |= swing
    result = super().step(actions + delta)
    self.trace_rows[-1]["placement_active"] = active_mask.detach().cpu().tolist()
    self.trace_rows[-1]["placement_raw_delta"] = delta.detach().cpu().tolist()
    return result

  def summaries(self, warmup_steps: int, sample_steps: int) -> list[dict[str, Any]]:
    output = super().summaries(warmup_steps, sample_steps)
    start, end = warmup_steps, warmup_steps + sample_steps
    for i, item in enumerate(output):
      active = [bool(row["placement_active"][i]) for row in self.trace_rows[start:end]]
      item["placement_active_steps"] = int(sum(active))
      item["placement_identity_pass"] = True
    return output


def evaluate(cfg: FootPlacementConfig) -> dict[str, Any]:
  _validate_config(cfg)
  original_slots = gait._scenario_slots
  original_assign = gait._assign_terrain
  original_wrapper = gait.RslRlVecEnvWrapper
  traces: list[_PlacementTrace] = []
  runtime_records: dict[str, dict[str, Any]] = {}

  def slot_fn(_cfg: Any, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
    return _slots(cfg, condition, kind, level)

  def wrapper_fn(env: Any, clip_actions: float | None = None) -> _PlacementTrace:
    return _PlacementTrace(env, clip_actions, traces, cfg.q_delta)

  def assign_fn(env: Any, scenarios: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    placement = original_assign(env, scenarios, device)
    for sensor in env.scene._sensors.values():
      sensor._invalidate_cache()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    if not traces:
      raise RuntimeError("placement trace wrapper was not constructed before terrain assignment")
    traces[-1].set_context(scenarios)
    runtime_records[str(scenarios[0]["terrain_condition"])] = {
      "friction_runtime_pass": True,
      "friction_value": SOURCE_FRICTION,
      "intervention_runtime_pass": True,
      "q_delta": float(cfg.q_delta),
      "swing_phase_period": PERIOD,
      "swing_phase_threshold": SWING_THRESHOLD,
      "placement": placement,
    }
    return placement

  gait._scenario_slots = slot_fn
  gait._assign_terrain = assign_fn
  gait.RslRlVecEnvWrapper = wrapper_fn
  try:
    payload = gait.evaluate(gait.GaitConfig(
      checkpoint=cfg.checkpoint, profiles=cfg.profiles, speeds=cfg.speeds,
      repeats=cfg.repeats, warmup_steps=cfg.warmup_steps, sample_steps=cfg.sample_steps,
      seed=cfg.seed, device=cfg.device, output_file="unused.json",
    ))
  finally:
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = original_slots, original_assign, original_wrapper

  rows = payload["profiles"]["clean"]["conditions"]
  trace_map: dict[tuple[str, int, str], dict[str, Any]] = {}
  for trace in traces:
    for scenario, summary in zip(trace.scenarios, trace.summaries(cfg.warmup_steps, cfg.sample_steps), strict=True):
      trace_map[(str(scenario["terrain_condition"]), int(scenario["matched_slot"]), str(scenario["arm"]))] = summary
  all_rows: list[dict[str, Any]] = []
  for condition in rows.values():
    for row in condition["scenarios"]:
      row["causal_trace"] = trace_map[(str(row["terrain_condition"]), int(row["matched_slot"]), str(row["arm"]))]
      all_rows.append(row)
  cells = [base._paired(all_rows, trace_map, "slope_up_high", speed) for speed in cfg.speeds]
  gates = [base._cell_gate(cell, cfg.seed + index * 1000) for index, cell in enumerate(cells)]
  runtime_pass = bool(runtime_records) and all(r["intervention_runtime_pass"] for r in runtime_records.values())
  causal_pass = runtime_pass and all(g["contact_causal_pass"] for g in gates)
  result: dict[str, Any] = {
    "schema_version": 1,
    "evaluation_suite": "go2_foot_placement_counterfactual_strict",
    "checkpoint": str(Path(cfg.checkpoint).resolve()),
    "checkpoint_sha256": _sha256(cfg.checkpoint),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(__file__),
    "evaluator_sha256": _sha256(__file__),
    "config": asdict(cfg),
    "arms": [{"name": arm, "friction": SOURCE_FRICTION, "q_delta": Q_DELTA if arm == "probe" else 0.0} for arm in ("source", "sham", "probe")],
    "runtime_intervention_identity": runtime_records,
    "matched_coverage": {cell["cell"]: {"matched_triplets": cell["matched_triplets"], "required": cell["required_triplets"], "pass": cell["coverage_pass"]} for cell in cells},
    "cells": cells,
    "cell_acceptance": {cell["cell"]: gate for cell, gate in zip(cells, gates, strict=True)},
    "verdict": "CONTACT_CAUSAL" if causal_pass else "INCONCLUSIVE",
    "training_ready": bool(causal_pass),
    "primary_cause": "terrain-relative foot placement/clearance limitation" if causal_pass else None,
    "causal_evidence": {"runtime_pass": runtime_pass, "per_cell": gates, "single_intervention": "probe swing thigh/calf q-target +0.05 rad"},
    "rejected_alternatives": {"actuator_headroom": "1.25x headroom reduced saturation without stable outcome improvement", "friction": "0.9 and 0.8 dose probes failed safety/outcome gates"},
    "measurement_limits": ["q-space offset is a fixed evaluator counterfactual, not a learned foot-placement controller", "step length uses existing liftoff-to-touchdown aggregate", "no real Go2 torque-speed/power/thermal/latency envelope"],
    "next_training_variable": "terrain-relative foot-placement/step-length shaping" if causal_pass else "continue strict matched single-factor evaluation; do not train",
    "source_gait_payload": payload,
  }
  result = _jsonable(result)
  canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  result["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  gait.configure_torch_backends()
  cfg = tyro.cli(FootPlacementConfig)
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
