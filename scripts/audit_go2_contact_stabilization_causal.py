"""Strict matched loaded-stance tangential stabilization audit for V7.

The evaluator keeps the locked V7 policy and all task configuration unchanged.
Only probe worlds receive a bounded world-frame wrench at each loaded foot that
opposes local terrain-tangent slip.  Source and sham execute the identical
wrapper with zero wrench.  The intervention is an evaluation-only mechanism
test, not a deployable controller or a training change.
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

from scripts import audit_go2_friction_contact_causal as base
from scripts import audit_go2_foot_placement_counterfactual as foot
from scripts import diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


EXPECTED_CHECKPOINT_SHA256 = foot.EXPECTED_CHECKPOINT_SHA256
ARMS = ("source", "sham", "probe")
FRICTION = 0.6
DAMPING_N_PER_MPS = 20.0
CAP_FRICTION_FRACTION = 0.20
CONTACT_FORCE_THRESHOLD_N = 5.0
PITCH_INSTABILITY_THRESHOLD_RAD = 0.45
REQUIRED_TRIPLETS = 8


@dataclass(frozen=True)
class ContactStabilizationConfig:
  checkpoint: str = gait.V7_CHECKPOINT
  profiles: tuple[str, ...] = ("clean",)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  formal: bool = True
  damping_n_per_mps: float = DAMPING_N_PER_MPS
  cap_friction_fraction: float = CAP_FRICTION_FRACTION
  output_file: str = "go2_contact_stabilization_causal_strict.json"


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_config(cfg: ContactStabilizationConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(gait.V7_CHECKPOINT).resolve():
    raise ValueError("checkpoint path is not locked V7 model_13600.pt")
  if _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("checkpoint SHA is not locked V7 model_13600.pt")
  if cfg.profiles != ("clean",) or cfg.speeds != (0.3, 0.5) or cfg.seed != 42:
    raise ValueError("profile/speeds/seed are locked")
  if not math.isclose(cfg.damping_n_per_mps, DAMPING_N_PER_MPS, abs_tol=1.0e-8):
    raise ValueError("damping is pre-registered at 20 N/(m/s)")
  if not math.isclose(cfg.cap_friction_fraction, CAP_FRICTION_FRACTION, abs_tol=1.0e-8):
    raise ValueError("force cap is pre-registered at 0.20*mu*Fn")
  if cfg.formal and (
    cfg.repeats < REQUIRED_TRIPLETS or cfg.warmup_steps != 100 or cfg.sample_steps < 1200
  ):
    raise ValueError("formal matrix requires >=8 repeats, warmup=100, sample>=1200")
  if not cfg.formal and (cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps < 10):
    raise ValueError("invalid smoke matrix")


def _slots(
  cfg: ContactStabilizationConfig, condition: str, kind: str, level: int,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for speed_index, speed in enumerate(cfg.speeds):
    for repeat in range(cfg.repeats):
      matched = speed_index * cfg.repeats + repeat
      for arm_index, arm in enumerate(ARMS):
        rows.append({
          "matched_slot": matched, "terrain_condition": condition,
          "terrain_kind": kind, "terrain_level": level, "speed": float(speed),
          "command_name": f"forward_{speed:g}", "repeat": repeat,
          "speed_index": speed_index, "arm": arm, "arm_index": arm_index,
          "friction": FRICTION,
          "intervention": "loaded_stance_local_tangent_stabilization",
          "damping_n_per_mps": DAMPING_N_PER_MPS if arm == "probe" else 0.0,
          "cap_friction_fraction": CAP_FRICTION_FRACTION if arm == "probe" else 0.0,
        })
  return rows


def _bounded_tangent_force(
  tangent_velocity: torch.Tensor, normal_force: torch.Tensor,
  damping: float = DAMPING_N_PER_MPS,
  cap_fraction: float = CAP_FRICTION_FRACTION,
  friction: float = FRICTION,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return damping force and scalar cap for (..., 3) tangent velocity."""
  cap = cap_fraction * friction * normal_force.clamp_min(0.0)
  desired = -damping * tangent_velocity
  magnitude = torch.linalg.vector_norm(desired, dim=-1)
  scale = torch.minimum(torch.ones_like(magnitude), cap / magnitude.clamp_min(1.0e-8))
  return desired * scale[..., None], cap


class _ContactTrace(base._TraceWrapper):
  def __init__(self, env: Any, clip_actions: float | None, traces: list[Any]):
    super().__init__(env, clip_actions, traces)
    self._probe_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    self._foot_ids, foot_names = robot.find_sites(gait.FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != gait.FOOT_NAMES:
      raise RuntimeError(f"foot order mismatch: {foot_names}")
    calf_names = tuple(f"{name}_calf" for name in gait.FOOT_NAMES)
    self._calf_ids, found_calf_names = robot.find_bodies(calf_names, preserve_order=True)
    if tuple(found_calf_names) != calf_names:
      raise RuntimeError(f"calf order mismatch: {found_calf_names}")
    self._calf_body_ids = [int(value) for value in self._calf_ids]
    feet_sensor = env.scene["feet_ground_contact"]
    sensor_names = [
      slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"
    ]
    self._foot_permutation = torch.tensor(
      [sensor_names.index(f"{name}_foot_collision") for name in gait.FOOT_NAMES],
      device=env.device,
    )

  def set_context(self, scenarios: list[dict[str, Any]]) -> None:
    super().set_context(scenarios)
    self._probe_mask = torch.tensor(
      [row["arm"] == "probe" for row in scenarios], dtype=torch.bool, device=self.env.device
    )

  def _write_stabilizer(self) -> dict[str, torch.Tensor]:
    env = self.env
    robot = env.scene["robot"]
    nenv = env.num_envs
    foot = robot.data.site_pos_w[:, self._foot_ids, :]
    velocity = robot.data.site_lin_vel_w[:, self._foot_ids, :]
    contact_sensor = env.scene["feet_ground_contact"]
    contact = gait._foot_contact(
      contact_sensor, nenv, len(gait.FOOT_NAMES), self._foot_permutation
    )
    contact_force = gait._foot_force(
      contact_sensor, nenv, len(gait.FOOT_NAMES), self._foot_permutation
    )
    clearance, normal, ray_valid = gait._normal_and_clearance(
      env.scene["terrain_scan"], foot, self.fallback_normal
    )
    del clearance
    if contact_force is None:
      normal_force = torch.zeros(nenv, len(gait.FOOT_NAMES), device=env.device)
    else:
      normal_force = -(contact_force * normal).sum(dim=-1)
    tangent_velocity = velocity - (velocity * normal).sum(dim=-1, keepdim=True) * normal
    force, cap = _bounded_tangent_force(tangent_velocity, normal_force)
    loaded = contact & ray_valid & (normal_force >= CONTACT_FORCE_THRESHOLD_N)
    active_probe = self._active & self._probe_mask
    apply = loaded & active_probe[:, None]
    force = torch.where(apply[..., None], force, torch.zeros_like(force))
    calf_com = robot.data.body_com_pos_w[:, self._calf_ids, :]
    torque = torch.linalg.cross(foot - calf_com, force, dim=-1)
    robot.write_external_wrench_to_sim(
      force, torque, body_ids=self._calf_body_ids
    )
    actual_force = robot.data.body_external_force[:, self._calf_ids, :]
    actual_torque = robot.data.body_external_torque[:, self._calf_ids, :]
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    power = -(force * tangent_velocity).sum(dim=-1)
    normal_component = (force * normal).sum(dim=-1).abs()
    nonprobe = (~self._probe_mask)[:, None, None]
    return {
      "force_mean": magnitude.mean(dim=-1),
      "force_max": magnitude.max(dim=-1).values,
      "cap_mean": cap.mean(dim=-1),
      "active_fraction": apply.float().mean(dim=-1),
      "dissipated_power": torch.where(apply, power, torch.zeros_like(power)).sum(dim=-1),
      "runtime_force_error": (actual_force - force).abs().amax(dim=(-1, -2)),
      "runtime_torque_error": (actual_torque - torque).abs().amax(dim=(-1, -2)),
      "cap_excess": (magnitude - cap).clamp_min(0.0).amax(dim=-1),
      "normal_component_error": normal_component.amax(dim=-1),
      "source_sham_wrench_error": torch.where(
        nonprobe, actual_force, torch.zeros_like(actual_force)
      ).abs().amax(dim=(-1, -2)),
    }

  def step(self, actions: torch.Tensor):
    intervention = self._write_stabilizer()
    result = super().step(actions)
    for key, value in intervention.items():
      self.trace_rows[-1][f"stabilizer_{key}"] = value.detach().cpu().tolist()
    return result

  def summaries(self, warmup_steps: int, sample_steps: int) -> list[dict[str, Any]]:
    output = super().summaries(warmup_steps, sample_steps)
    keys = (
      "force_mean", "force_max", "cap_mean", "active_fraction", "dissipated_power",
      "runtime_force_error", "runtime_torque_error", "cap_excess",
      "normal_component_error", "source_sham_wrench_error",
    )
    for env_id, item in enumerate(output):
      start, end = warmup_steps, warmup_steps + sample_steps
      active = [bool(row["active"][env_id]) for row in self.trace_rows[start:end]]
      for key in keys:
        values: list[float | None] = []
        trace_key = f"stabilizer_{key}"
        for is_active, row in zip(active, self.trace_rows[start:end], strict=True):
          value = float(row[trace_key][env_id])
          values.append(value if is_active and math.isfinite(value) else None)
        item["series"][f"stabilizer_{key}"] = values
      identity_max = {
        key: max(
          (float(row[f"stabilizer_{key}"][env_id]) for row in self.trace_rows),
          default=float("inf"),
        ) for key in (
          "runtime_force_error", "runtime_torque_error", "cap_excess",
          "normal_component_error", "source_sham_wrench_error",
        )
      }
      item["stabilizer_identity"] = identity_max
      item["stabilizer_identity_pass"] = bool(
        all(math.isfinite(value) and value <= 1.0e-5 for value in identity_max.values())
      )
      item["pitch_instability_onset_step"] = base._onset(
        item["series"]["pitch"], PITCH_INSTABILITY_THRESHOLD_RAD
      )
      item["stabilizer_metrics"] = {
        key: base._stats(item["series"][f"stabilizer_{key}"])
        for key in keys[:5]
      }
    return output


def _contact_cell(
  rows: list[dict[str, Any]], traces: dict[tuple[str, int, str], dict[str, Any]],
  condition: str, speed: float,
) -> dict[str, Any]:
  cell = base._paired(rows, traces, condition, speed)
  for pair in cell["pairs"]:
    slot = int(pair["matched_slot"])
    trace_arms = {arm: traces[(condition, slot, arm)] for arm in ARMS}
    prefix = int(pair["common_prefix_steps"])
    for metric in (
      "stabilizer_force_mean", "stabilizer_force_max", "stabilizer_active_fraction",
      "stabilizer_dissipated_power",
    ):
      pair["metrics"][metric] = {
        arm: base._mean_prefix(trace_arms[arm], metric, prefix) for arm in ARMS
      }
    ordering: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    for arm in ARMS:
      contact_onsets = [
        onset for onset in (
          trace_arms[arm]["slip_onset_step"],
          trace_arms[arm]["cone_utilization_onset_step"],
        ) if onset is not None
      ]
      contact_onset = min(contact_onsets) if contact_onsets else None
      endpoints = [
        endpoint for endpoint in (
          trace_arms[arm]["pitch_instability_onset_step"],
          trace_arms[arm]["failure_step"],
        ) if endpoint is not None
      ]
      endpoint = min(endpoints) if endpoints else None
      ordering[arm] = endpoint is None or (
        contact_onset is not None and contact_onset <= endpoint + 1
      )
      detail[arm] = {
        "contact_onset_step": contact_onset,
        "pitch_instability_onset_step": trace_arms[arm]["pitch_instability_onset_step"],
        "failure_step": trace_arms[arm]["failure_step"],
      }
    pair["onset_ordering"] = ordering
    pair["contact_onset_detail"] = detail
  return cell


def _flat_sentinel(cell: dict[str, Any], seed: int) -> dict[str, Any]:
  gate = base._cell_gate(cell, seed)
  catastrophic = sum(
    pair["completion"]["sham"] and not pair["completion"]["probe"]
    for pair in cell["pairs"]
  )
  sham_gain = [
    pair["metrics"]["gain"]["sham"] for pair in cell["pairs"]
    if pair["metrics"]["gain"]["sham"] is not None
  ]
  probe_gain = [
    pair["metrics"]["gain"]["probe"] for pair in cell["pairs"]
    if pair["metrics"]["gain"]["probe"] is not None
  ]
  gain_ratio = None
  if sham_gain and probe_gain and abs(float(np.mean(sham_gain))) > 1.0e-8:
    gain_ratio = float(np.mean(probe_gain) / np.mean(sham_gain))
  return {
    "coverage_pass": cell["coverage_pass"],
    "sham_complete_to_probe_fail_count": int(catastrophic),
    "gain_probe_over_sham": gain_ratio,
    "side_effect_pass": gate["side_effect_pass"],
    "side_effect_ratios": gate["side_effect_ratios"],
    "pass": bool(
      cell["coverage_pass"] and catastrophic == 0
      and gain_ratio is not None and gain_ratio >= 0.90
      and gate["side_effect_pass"]
    ),
  }


def evaluate(cfg: ContactStabilizationConfig) -> dict[str, Any]:
  _validate_config(cfg)
  original_slots, original_assign, original_wrapper = (
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper
  )
  traces: list[_ContactTrace] = []
  runtime: dict[str, dict[str, Any]] = {}

  def slot_fn(_cfg: Any, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
    return _slots(cfg, condition, kind, level)

  def wrapper_fn(env: Any, clip_actions: float | None = None) -> _ContactTrace:
    return _ContactTrace(env, clip_actions, traces)

  def assign_fn(env: Any, scenarios: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    placement = original_assign(env, scenarios, device)
    for sensor in env.scene._sensors.values():
      sensor._invalidate_cache()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    if not traces:
      raise RuntimeError("contact wrapper missing before assignment")
    traces[-1].set_context(scenarios)
    runtime[str(scenarios[0]["terrain_condition"])] = {
      "runtime_pass": False,
      "placement": placement,
      "damping_n_per_mps": DAMPING_N_PER_MPS,
      "force_cap": "0.20 * 0.6 * local_normal_force",
      "application": "world-frame force and equivalent foot-point torque on calf body",
    }
    return placement

  gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = (
    slot_fn, assign_fn, wrapper_fn
  )
  try:
    payload = gait.evaluate(gait.GaitConfig(
      checkpoint=cfg.checkpoint, profiles=cfg.profiles, speeds=cfg.speeds,
      repeats=cfg.repeats, warmup_steps=cfg.warmup_steps,
      sample_steps=cfg.sample_steps, seed=cfg.seed, device=cfg.device,
      output_file="unused.json",
    ))
  finally:
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = (
      original_slots, original_assign, original_wrapper
    )

  conditions = payload["profiles"]["clean"]["conditions"]
  trace_map: dict[tuple[str, int, str], dict[str, Any]] = {}
  for trace in traces:
    summaries = trace.summaries(cfg.warmup_steps, cfg.sample_steps)
    for scenario, summary in zip(trace.scenarios, summaries, strict=True):
      trace_map[(
        str(scenario["terrain_condition"]), int(scenario["matched_slot"]),
        str(scenario["arm"]),
      )] = summary
  all_rows: list[dict[str, Any]] = []
  for condition in conditions.values():
    for row in condition["scenarios"]:
      row["causal_trace"] = trace_map[(
        str(row["terrain_condition"]), int(row["matched_slot"]), str(row["arm"])
      )]
      all_rows.append(row)
  for condition_name, item in runtime.items():
    summaries = [
      summary for (condition, _slot, _arm), summary in trace_map.items()
      if condition == condition_name
    ]
    item["identity_rows"] = len(summaries)
    item["runtime_pass"] = bool(
      summaries and all(summary["stabilizer_identity_pass"] for summary in summaries)
    )
    for key in (
      "runtime_force_error", "runtime_torque_error", "cap_excess",
      "normal_component_error", "source_sham_wrench_error",
    ):
      item[f"{key}_max"] = max(
        (summary["stabilizer_identity"][key] for summary in summaries), default=None
      )

  cells = [_contact_cell(all_rows, trace_map, "slope_up_high", speed) for speed in cfg.speeds]
  gates = [base._cell_gate(cell, cfg.seed + index * 1000) for index, cell in enumerate(cells)]
  flat_cells = [_contact_cell(all_rows, trace_map, "flat", speed) for speed in cfg.speeds]
  flat_gates = [
    _flat_sentinel(cell, cfg.seed + 5000 + index * 1000)
    for index, cell in enumerate(flat_cells)
  ]
  runtime_pass = bool(runtime) and all(item["runtime_pass"] for item in runtime.values())
  causal_pass = bool(
    runtime_pass and all(gate["contact_causal_pass"] for gate in gates)
    and all(gate["pass"] for gate in flat_gates)
  )
  result: dict[str, Any] = {
    "schema_version": 1,
    "evaluation_suite": "go2_loaded_stance_contact_stabilization_causal_strict",
    "checkpoint": str(Path(cfg.checkpoint).resolve()),
    "checkpoint_sha256": _sha256(cfg.checkpoint),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(__file__),
    "evaluator_sha256": _sha256(__file__),
    "config": asdict(cfg),
    "arms": [
      {
        "name": arm, "friction": FRICTION,
        "damping_n_per_mps": DAMPING_N_PER_MPS if arm == "probe" else 0.0,
        "cap_friction_fraction": CAP_FRICTION_FRACTION if arm == "probe" else 0.0,
      } for arm in ARMS
    ],
    "runtime_intervention_identity": runtime,
    "matched_coverage": {
      cell["cell"]: {
        "matched_triplets": cell["matched_triplets"],
        "required": cell["required_triplets"], "pass": cell["coverage_pass"],
      } for cell in cells
    },
    "cells": cells,
    "cell_acceptance": {
      cell["cell"]: gate for cell, gate in zip(cells, gates, strict=True)
    },
    "flat_sentinel": {
      cell["cell"]: gate for cell, gate in zip(flat_cells, flat_gates, strict=True)
    },
    "verdict": "CONTACT_CAUSAL" if causal_pass else "INCONCLUSIVE",
    "training_ready": bool(causal_pass),
    "primary_cause": (
      "loaded-stance local-tangent slip/contact-stability limitation under the MuJoCo contact model"
      if causal_pass else None
    ),
    "causal_evidence": {
      "runtime_pass": runtime_pass, "per_cell": gates,
      "single_intervention": "bounded loaded-stance local-tangent damping wrench",
    },
    "rejected_alternatives": {
      "action_recovery_fir_0p5": "formally harmful on both high-slope speed cells",
      "actuator_headroom": "strongly de-prioritized as the sole/primary explanation; not formally excluded",
      "foot_placement_plus_0p05": "strongly harmful",
      "scalar_friction": "strong causal direction but strict pitch/action/contact guardrails failed",
    },
    "effect_vs_sham_noise": {
      cell["cell"]: {
        metric: cell["effects"][metric]
        for metric in ("slip", "cone_utilization", "gain")
      } for cell in cells
    },
    "bootstrap_ci": {
      cell["cell"]: gate["bootstrap"]
      for cell, gate in zip(cells, gates, strict=True)
    },
    "measurement_limits": [
      "the stabilizer is an evaluator oracle using contact force and terrain normal",
      "the foot-point force is represented by an equivalent force/torque wrench on the calf body",
      "the result identifies a simulation contact mechanism, not real-hardware friction by itself",
      "no real Go2 thermal, latency, battery or motor torque-speed envelope",
    ],
    "next_training_variable": (
      "terrain-relative local-tangent stance-slip/contact-stability shaping"
      if causal_pass else "continue strict matched evaluation; do not train"
    ),
    "source_gait_payload": payload,
  }
  result = foot._jsonable(result)
  canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  result["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  gait.configure_torch_backends()
  cfg = tyro.cli(ContactStabilizationConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  output.with_suffix(output.suffix + ".sha256").write_text(
    f"{_sha256(output)}  {output.name}\n"
  )
  print(json.dumps({
    "output": str(output), "verdict": result["verdict"],
    "training_ready": result["training_ready"],
  }, indent=2, allow_nan=False))


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  main()
