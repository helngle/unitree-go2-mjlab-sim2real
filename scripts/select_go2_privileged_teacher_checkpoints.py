"""Select matched control/candidate teacher stages from high-slope evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.privileged_teacher_schema import (  # noqa: E402
  CANDIDATE_ACTOR_DIM,
  SOURCE_ACTOR_DIM,
)
from src.tasks.velocity.rl.privileged_teacher_transfer import (  # noqa: E402
  SOURCE_CHECKPOINT,
  _validate_probe_checkpoint_infos,
  sha256_file,
)


PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")
STAGES = (100, 200, 300, 400)
SAFETY = (
  "slip", "action_acceleration", "base_pitch", "base_contact",
  "upper_leg_contact", "calf_contact", "failure_risk", "action_fault_rate",
  "joint_target_fault_rate",
)


def _exceeds_safety_guardrail(value: float, reference: float) -> bool:
  if not math.isfinite(value) or not math.isfinite(reference):
    raise ValueError("safety metric must be finite")
  if value < 0.0 or reference < 0.0:
    raise ValueError("safety metric must be non-negative")
  return value > 1.2 * reference


def _weighted(rows: list[dict[str, Any]], field: str, weight) -> float:
  denominator = sum(weight(row) for row in rows)
  if denominator <= 0:
    raise ValueError(f"non-positive denominator for {field}")
  return sum(float(row[field]) * weight(row) for row in rows) / denominator


def _aggregate_route(payload: dict[str, Any], profile: str, route: str) -> dict[str, float]:
  rows = payload["profiles"][profile]["route_results"][route]["scenarios"]
  if len(rows) != 16:
    raise ValueError(f"{profile}/{route} must contain exactly 16 scenarios")
  active_steps = [
    int(row["terrain_rollout_metrics"]["active_control_step_samples"])
    for row in rows
  ]
  if any(
    active != int(row["steps_sampled"])
    for row, active in zip(rows, active_steps, strict=True)
  ):
    raise ValueError(f"{profile}/{route} active-step denominator differs")
  steps = sum(active_steps)
  loaded = sum(
    int(row["terrain_tangent_loaded_stance"]["loaded_stance_foot_samples"])
    for row in rows
  )
  if steps <= 0 or loaded <= 0:
    raise ValueError(f"{profile}/{route} has an empty metric denominator")
  gain_weight = lambda row: int(row["steps_sampled"]) * float(row["speed"]) ** 2
  gain_denominator = sum(gain_weight(row) for row in rows)
  fault_steps = sum(
    int(row["terrain_rollout_metrics"]["action_safety"]["fault_control_step_count"])
    for row in rows
  )
  joint_target_fault_steps = sum(
    int(row["terrain_rollout_metrics"]["action_safety"][
      "joint_target_fault_control_step_count"
    ]) for row in rows
  )
  if not all(
    row["terrain_rollout_metrics"]["action_safety"].get("available") is True
    and row["terrain_rollout_metrics"]["action_safety"].get(
      "joint_target_available"
    ) is True
    for row in rows
  ):
    raise ValueError(f"{profile}/{route} action safety is unavailable")
  return {
    "completion": sum(bool(row["completed"]) for row in rows) / len(rows),
    "progress": sum(float(row.get("progress_ratio", 0.0)) for row in rows) / len(rows),
    "forward_gain": sum(
      float(row["response_gain"]["vx"]) * gain_weight(row) for row in rows
    ) / gain_denominator,
    "slip": sum(
      float(row["terrain_tangent_stance_slip_mean"])
      * int(row["terrain_tangent_loaded_stance"]["loaded_stance_foot_samples"])
      for row in rows
    ) / loaded,
    "action_acceleration": _weighted(
      rows, "action_acceleration_mean", lambda row: int(row["steps_sampled"])
    ),
    "base_pitch": _weighted(
      rows, "base_pitch_absolute_mean", lambda row: int(row["steps_sampled"])
    ),
    "base_contact": sum(int(row["base_contact_count"]) for row in rows) / steps,
    "upper_leg_contact": sum(int(row["upper_leg_contact_count"]) for row in rows) / steps,
    "calf_contact": sum(int(row["calf_contact_count"]) for row in rows) / steps,
    "failure_risk": sum(
      bool(row["failed"]) or bool(row["catastrophic_termination"])
      for row in rows
    ) / len(rows),
    "action_fault_rate": fault_steps / steps,
    "joint_target_fault_rate": joint_target_fault_steps / steps,
  }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
  routes = {
    profile: {
      route: _aggregate_route(payload, profile, route) for route in ROUTES
    } for profile in PROFILES
  }
  values = [routes[profile][route] for profile in PROFILES for route in ROUTES]
  return {
    "routes": routes,
    "macro_completion": sum(item["completion"] for item in values) / 6.0,
    "mean_forward_gain": sum(item["forward_gain"] for item in values) / 6.0,
    "mean_progress": sum(item["progress"] for item in values) / 6.0,
    "mean_slip": sum(item["slip"] for item in values) / 6.0,
    "minimum_clean_completion": min(
      routes["clean"][route]["completion"] for route in ROUTES
    ),
    "minimum_randomized_completion": min(
      routes["randomized"][route]["completion"] for route in ROUTES
    ),
  }


def _load_evaluation(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if payload.get("evaluation_suite") != "high_slope_matched_straight_arc_s_curve":
    raise ValueError(f"unexpected high-slope suite: {path}")
  if tuple(payload.get("profiles", {})) != PROFILES:
    raise ValueError(f"evaluation profiles differ: {path}")
  return payload


def _checkpoint_identity(payload: dict[str, Any]) -> Path:
  checkpoint = Path(payload["checkpoint"]).expanduser().resolve()
  provenance = payload.get("provenance", {}).get("checkpoint", {})
  if (
    provenance.get("path") != str(checkpoint)
    or provenance.get("sha256") != sha256_file(checkpoint)
  ):
    raise ValueError(f"evaluation checkpoint provenance differs: {checkpoint}")
  return checkpoint


def _stage_record(
  *, arm: str, evaluation_path: Path, screening_path: Path,
  baseline: dict[str, Any], require_ability: bool,
) -> dict[str, Any]:
  payload = _load_evaluation(evaluation_path)
  checkpoint = _checkpoint_identity(payload)
  update = int(checkpoint.stem.removeprefix("model_"))
  if update not in STAGES:
    raise ValueError(f"unexpected checkpoint update: {checkpoint}")
  checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_dim = SOURCE_ACTOR_DIM if arm == "control_234" else CANDIDATE_ACTOR_DIM
  _validate_probe_checkpoint_infos(checkpoint_payload.get("infos") or {}, actor_dim)
  screening = json.loads(screening_path.read_text(encoding="utf-8"))
  identity = screening.get("checkpoint", {})
  if identity != {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}:
    raise ValueError(f"screening checkpoint identity differs: {screening_path}")
  required_screen = (
    "provenance_valid", "formal_metadata_valid", "recursive_finite",
    "optimizer_present_finite", "formal_environment_state_valid",
    "tensorboard_finite",
  )
  violations = [name for name in required_screen if screening.get(name) is not True]
  summary = summarize(payload)
  for profile in PROFILES:
    for route in ROUTES:
      candidate = summary["routes"][profile][route]
      reference = baseline["routes"][profile][route]
      for metric in SAFETY:
        if _exceeds_safety_guardrail(candidate[metric], reference[metric]):
          violations.append(f"safety:{profile}:{route}:{metric}")
      if require_ability:
        threshold = 0.75 if profile == "clean" else 0.625
        if candidate["completion"] < threshold:
          violations.append(f"completion:{profile}:{route}")
        if candidate["completion"] < reference["completion"]:
          violations.append(f"v7_completion:{profile}:{route}")
  if require_ability and summary["mean_forward_gain"] < 0.80:
    violations.append("mean_forward_gain")
  ranking = (
    summary["minimum_clean_completion"],
    summary["minimum_randomized_completion"],
    summary["mean_forward_gain"],
    -summary["mean_slip"],
    -update,
  )
  return {
    "arm": arm,
    "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    "update": update,
    "evaluation": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
    "screening": {"path": str(screening_path), "sha256": sha256_file(screening_path)},
    "summary": summary,
    "hard_gate_pass": not violations,
    "violations": violations,
    "ranking_key": ranking,
  }


def select(
  *, v7_json: Path, control_jsons: tuple[Path, ...],
  candidate_jsons: tuple[Path, ...], screening_dir: Path,
) -> dict[str, Any]:
  if len(control_jsons) != 4 or len(candidate_jsons) != 4:
    raise ValueError("exactly four evaluation JSON files are required per arm")
  baseline_payload = _load_evaluation(v7_json)
  baseline_checkpoint = _checkpoint_identity(baseline_payload)
  if baseline_checkpoint != SOURCE_CHECKPOINT.resolve():
    raise ValueError("baseline evaluation is not the locked V7 checkpoint")
  baseline = summarize(baseline_payload)
  records: dict[str, list[dict[str, Any]]] = {"control_234": [], "candidate_237": []}
  for arm, paths in (("control_234", control_jsons), ("candidate_237", candidate_jsons)):
    for path in paths:
      payload = _load_evaluation(path)
      checkpoint = _checkpoint_identity(payload)
      update = int(checkpoint.stem.removeprefix("model_"))
      screening = screening_dir / f"{arm}_model_{update}.screening.json"
      records[arm].append(_stage_record(
        arm=arm, evaluation_path=path.resolve(), screening_path=screening.resolve(),
        baseline=baseline, require_ability=arm == "candidate_237",
      ))
  for arm in records:
    if sorted(item["update"] for item in records[arm]) != list(STAGES):
      raise ValueError(f"{arm} checkpoint schedule differs")
  control_by_update = {item["update"]: item for item in records["control_234"]}
  candidate_by_update = {item["update"]: item for item in records["candidate_237"]}
  paired_survivors = []
  for update in STAGES:
    control = control_by_update[update]
    candidate = candidate_by_update[update]
    delta = candidate["summary"]["macro_completion"] - control["summary"]["macro_completion"]
    if control["hard_gate_pass"] and candidate["hard_gate_pass"] and delta >= 0.10:
      paired_survivors.append((candidate, control, delta))
  paired_survivors.sort(
    key=lambda item: tuple(item[0]["ranking_key"]), reverse=True
  )
  selected_candidate = paired_survivors[0][0] if paired_survivors else None
  selected_control = paired_survivors[0][1] if paired_survivors else None
  causal_delta = paired_survivors[0][2] if paired_survivors else None
  causal_pass = bool(paired_survivors)
  selection_status = (
    "SELECTED_FOR_FULL_ACCEPTANCE" if causal_pass
    else "NO_CAUSAL_SURVIVOR"
  )
  return {
    "schema_version": 1,
    "selection_method": "hard_gates_then_lexicographic_then_control_delta",
    "weighted_score_used": False,
    "baseline": {
      "checkpoint": str(baseline_checkpoint),
      "sha256": sha256_file(baseline_checkpoint),
      "evaluation": {"path": str(v7_json.resolve()), "sha256": sha256_file(v7_json)},
      "summary": baseline,
    },
    "arms": records,
    "selected_control": selected_control,
    "selected_candidate": selected_candidate,
    "candidate_minus_control_macro_completion": causal_delta,
    "causal_delta_required": 0.10,
    "causal_delta_pass": causal_pass,
    "selection_status": selection_status,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--v7-json", type=Path, required=True)
  parser.add_argument("--control-jsons", type=Path, nargs=4, required=True)
  parser.add_argument("--candidate-jsons", type=Path, nargs=4, required=True)
  parser.add_argument("--screening-dir", type=Path, required=True)
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  result = select(
    v7_json=args.v7_json.expanduser().resolve(),
    control_jsons=tuple(path.expanduser().resolve() for path in args.control_jsons),
    candidate_jsons=tuple(path.expanduser().resolve() for path in args.candidate_jsons),
    screening_dir=args.screening_dir.expanduser().resolve(),
  )
  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite selection artifact: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
