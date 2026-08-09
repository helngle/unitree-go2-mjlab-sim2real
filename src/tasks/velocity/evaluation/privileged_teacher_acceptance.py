"""Suite-aware aggregation and hard gates for the Go2 V8 teacher probe."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.tasks.velocity.evaluation.proprio_acceptance import (
  canonical_profile_name,
  sha256_file,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


PROFILES = ("clean", "randomized")
SAFETY_METRICS = (
  "terrain_tangent_slip",
  "action_acceleration",
  "base_pitch_absolute",
  "base_contact",
  "upper_leg_contact",
  "calf_contact",
  "failure_risk",
  "action_fault_rate",
  "joint_target_fault_rate",
)
EXPECTED_ARTIFACTS = frozenset({
  "high_slope_matched",
  "flat_matched_routes",
  *(
    artifact
    for profile in PROFILES
    for artifact in (
      f"continuous_retained_{profile}",
      f"terrain_curves_arc_{profile}",
      f"terrain_curves_s_curve_{profile}",
      f"stairs_level9_seed42_{profile}",
      f"stairs_level9_seed43_{profile}",
      f"stairs_level9_seed44_{profile}",
    )
  ),
})


@dataclass(frozen=True)
class RawArtifact:
  checkpoint_label: str
  checkpoint_path: str
  checkpoint_sha256: str
  task_id: str
  artifact_id: str
  path: str


def _suite_id(artifact_id: str) -> str:
  if artifact_id == "high_slope_matched":
    return "high_slope"
  if artifact_id == "flat_matched_routes":
    return "flat"
  if artifact_id.startswith("continuous_retained_"):
    return "continuous_retained"
  if artifact_id.startswith("terrain_curves_"):
    return "terrain_curves"
  if artifact_id.startswith("stairs_level9_seed"):
    return "stairs_level9"
  raise ValueError(f"unknown teacher acceptance artifact: {artifact_id}")


def validate_artifact_inventory(artifacts: Iterable[RawArtifact]) -> None:
  grouped: dict[str, list[RawArtifact]] = defaultdict(list)
  for artifact in artifacts:
    grouped[artifact.checkpoint_label].append(artifact)
  if not grouped:
    raise ValueError("teacher acceptance inventory is empty")
  for label, items in grouped.items():
    observed = [item.artifact_id for item in items]
    if len(observed) != 14 or set(observed) != EXPECTED_ARTIFACTS:
      raise ValueError(f"teacher acceptance inventory differs for {label}")
    if len(observed) != len(set(observed)):
      raise ValueError(f"teacher acceptance inventory has duplicates for {label}")
    paths = {item.checkpoint_path for item in items}
    shas = {item.checkpoint_sha256 for item in items}
    tasks = {item.task_id for item in items}
    if len(paths) != 1 or len(shas) != 1 or len(tasks) != 1:
      raise ValueError(f"teacher acceptance checkpoint identity drifts for {label}")


def _profile_settings(payload: dict[str, Any], artifact_id: str) -> dict[str, Any]:
  if artifact_id in {"high_slope_matched", "flat_matched_routes"}:
    return {
      canonical_profile_name(name): value["profile_settings"]
      for name, value in payload["profiles"].items()
    }
  profile = "randomized" if artifact_id.endswith("_randomized") else "clean"
  return {profile: payload["profile_settings"]}


def _evaluation_contract(
  payload: dict[str, Any], artifact: RawArtifact
) -> dict[str, Any]:
  if payload.get("task_id") != artifact.task_id:
    raise ValueError(f"raw task identity differs: {artifact.path}")
  provenance = payload.get("provenance")
  if not isinstance(provenance, dict):
    raise ValueError(f"raw provenance is missing: {artifact.path}")
  dirty = provenance.get("dirty_state")
  evaluator = provenance.get("evaluator")
  dependencies = provenance.get("dependencies")
  if (
    not isinstance(dirty, dict)
    or not isinstance(evaluator, dict)
    or set(evaluator) != {"path", "sha256"}
    or not isinstance(dependencies, dict)
  ):
    raise ValueError(f"raw evaluation provenance is incomplete: {artifact.path}")
  contract = {
    "git_branch": provenance.get("git_branch"),
    "git_head": provenance.get("git_head"),
    "tracked_diff_sha256": dirty.get("tracked_diff_sha256"),
    "evaluator": evaluator,
    "dependencies": dependencies,
  }
  if (
    not all(isinstance(contract[name], str) and contract[name]
            for name in ("git_branch", "git_head", "tracked_diff_sha256"))
    or not all(isinstance(value, str) and value for value in dependencies.values())
  ):
    raise ValueError(f"raw evaluation provenance values are invalid: {artifact.path}")
  return contract


def _scenario_rows(
  payload: dict[str, Any], artifact_id: str
) -> list[tuple[str, str, str, str, dict[str, Any]]]:
  suite = _suite_id(artifact_id)
  rows: list[tuple[str, str, str, str, dict[str, Any]]] = []
  if suite in {"high_slope", "flat"}:
    for raw_profile, profile_data in payload["profiles"].items():
      profile = canonical_profile_name(raw_profile)
      for raw_route, result in profile_data["route_results"].items():
        route = "line" if raw_route == "straight" else raw_route
        for scenario in result["scenarios"]:
          if suite == "high_slope":
            scene = (
              f"{scenario['slope_direction']}_level{scenario['level']}"
            )
            category = "complex"
          else:
            scene, category = "flat", "retained"
          rows.append((profile, category, scene, route, scenario))
    return rows
  profile = "randomized" if artifact_id.endswith("_randomized") else "clean"
  for scenario in payload["scenarios"]:
    if suite == "terrain_curves":
      kind = str(scenario["terrain_curve_kind"])
      category = "complex" if kind.startswith("slope") else "retained"
      scene = f"{kind}_level{scenario['terrain_level']}"
      route = "line" if scenario.get("route_kind") == "straight" else str(
        scenario["route_kind"]
      )
    else:
      transition = str(scenario["transition_case"])
      level = int(scenario.get("level", scenario.get("transition_level", 0)))
      scene = f"{transition}_level{level}"
      category = "complex" if suite == "stairs_level9" else "retained"
      route = "line"
    rows.append((profile, category, scene, route, scenario))
  return rows


def _scenario_identity(artifact_id: str, scenario: dict[str, Any]) -> str:
  fields = {
    name: scenario[name]
    for name in (
      "matched_slot", "terrain_type", "terrain_curve_kind", "terrain_level",
      "transition_case", "level", "difficulty", "radius", "speed",
      "turn_sign", "repeat", "seed", "terrain_origin_xyz",
      "terrain_patch_origin_xyz", "route_start_xy", "route_endpoint_xy",
      "effective_terrain_parameters", "geometry",
    )
    if name in scenario
  }
  return hashlib.sha256(json.dumps(
    [artifact_id, fields], sort_keys=True, separators=(",", ":")
  ).encode()).hexdigest()


def _progress(scenario: dict[str, Any]) -> float:
  for name in ("progress_ratio", "arc_length_progress_ratio", "path_completion"):
    if name in scenario:
      return float(scenario[name])
  raise ValueError("scenario is missing normalized progress")


def _active_steps(scenario: dict[str, Any]) -> int:
  value = int(scenario["terrain_rollout_metrics"]["active_control_step_samples"])
  if value <= 0:
    raise ValueError("scenario has no active control samples")
  return value


def _aggregate(items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
  if not items:
    raise ValueError("cannot aggregate an empty teacher group")
  steps = sum(_active_steps(item) for _, item in items)
  loaded = sum(int(item["terrain_tangent_loaded_stance"][
    "loaded_stance_foot_samples"
  ]) for _, item in items)
  if loaded <= 0:
    raise ValueError("teacher group has no loaded-stance samples")
  forward = []
  for _, item in items:
    command = item.get("commanded_velocity_mean")
    actual = item.get("actual_velocity_mean")
    if isinstance(command, list) and isinstance(actual, list) and abs(float(command[0])) > 1e-6:
      forward.append(float(actual[0]) / float(command[0]))
  action_faults = 0
  joint_target_faults = 0
  contacts = {name: [0, 0] for name in ("base", "upper_leg", "calf")}
  slip_total = 0.0
  acceleration_total = 0.0
  pitch_total = 0.0
  for _, item in items:
    terrain = item["terrain_rollout_metrics"]
    count = _active_steps(item)
    safety = terrain["action_safety"]
    if safety.get("available") is not True or safety.get("joint_target_available") is not True:
      raise ValueError("action/joint-target safety telemetry is unavailable")
    action_faults += int(safety["fault_control_step_count"])
    joint_target_faults += int(safety["joint_target_fault_control_step_count"])
    acceleration_total += float(terrain["action_acceleration"]["mean"]) * count
    pitch_total += float(terrain["base_pitch_absolute"]["mean"]) * count
    loaded_count = int(item["terrain_tangent_loaded_stance"][
      "loaded_stance_foot_samples"
    ])
    slip_total += float(item["terrain_tangent_stance_slip_mean"]) * loaded_count
    for name in contacts:
      contact = terrain["body_contacts"][name]
      contacts[name][0] += int(contact["non_terminating_count"])
      contacts[name][1] += int(contact["denominator"])
  result = {
    "scenario_count": len(items),
    "scenario_ids": [identity for identity, _ in sorted(items)],
    "completion": sum(bool(item["completed"]) for _, item in items) / len(items),
    "progress": sum(_progress(item) for _, item in items) / len(items),
    "forward_gain": sum(forward) / len(forward) if forward else None,
    "terrain_tangent_slip": slip_total / loaded,
    "action_acceleration": acceleration_total / steps,
    "base_pitch_absolute": pitch_total / steps,
    "base_contact": contacts["base"][0] / contacts["base"][1],
    "upper_leg_contact": contacts["upper_leg"][0] / contacts["upper_leg"][1],
    "calf_contact": contacts["calf"][0] / contacts["calf"][1],
    "failure_risk": sum(bool(
      item.get("failed") or item.get("catastrophic_termination")
    ) for _, item in items) / len(items),
    "action_fault_rate": action_faults / steps,
    "joint_target_fault_rate": joint_target_faults / steps,
  }
  assert_recursive_json_finite(result)
  return result


def aggregate_checkpoint(artifacts: list[RawArtifact]) -> dict[str, Any]:
  validate_artifact_inventory(artifacts)
  if len({item.checkpoint_label for item in artifacts}) != 1:
    raise ValueError("aggregate_checkpoint requires exactly one checkpoint")
  grouped: dict[tuple[str, str, str, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
  profile_contract: dict[str, Any] = {}
  evaluation_contract: dict[str, Any] = {}
  raw = []
  for artifact in artifacts:
    path = Path(artifact.path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert_recursive_json_finite(payload)
    identity = payload.get("provenance", {}).get("checkpoint", {})
    if identity != {
      "path": artifact.checkpoint_path,
      "sha256": artifact.checkpoint_sha256,
    }:
      raise ValueError(f"raw checkpoint provenance differs: {path}")
    evaluation_contract[artifact.artifact_id] = _evaluation_contract(
      payload, artifact
    )
    profile_contract[artifact.artifact_id] = _profile_settings(
      payload, artifact.artifact_id
    )
    suite = _suite_id(artifact.artifact_id)
    for profile, category, scene, route, scenario in _scenario_rows(
      payload, artifact.artifact_id
    ):
      grouped[(suite, profile, category, scene, route)].append((
        _scenario_identity(artifact.artifact_id, scenario), scenario
      ))
    raw.append({"path": str(path), "sha256": sha256_file(path)})
  if len(grouped) != 106:
    raise ValueError(f"teacher suite-aware group count differs: {len(grouped)}")
  groups = []
  for key in sorted(grouped):
    suite, profile, category, scene, route = key
    groups.append({
      "suite": suite, "profile": profile, "category": category,
      "scene": scene, "route": route, **_aggregate(grouped[key]),
    })
  return {
    "checkpoint": {
      "label": artifacts[0].checkpoint_label,
      "path": artifacts[0].checkpoint_path,
      "sha256": artifacts[0].checkpoint_sha256,
      "task_id": artifacts[0].task_id,
    },
    "raw_artifacts": raw,
    "profile_contract": profile_contract,
    "evaluation_contract": evaluation_contract,
    "groups": groups,
  }


def _indexed(summary: dict[str, Any]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
  return {
    (row["suite"], row["profile"], row["category"], row["scene"], row["route"]): row
    for row in summary["groups"]
  }


def _high_slope_routes(summary: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
  rows = _indexed(summary)
  result: dict[str, dict[str, dict[str, float]]] = {}
  for profile in PROFILES:
    result[profile] = {}
    for route in ("line", "arc", "s_curve"):
      matches = [value for key, value in rows.items() if (
        key[0] == "high_slope" and key[1] == profile and key[4] == route
      )]
      total = sum(int(item["scenario_count"]) for item in matches)
      if total != 16:
        raise ValueError(f"high-slope {profile}/{route} coverage differs: {total}")
      result[profile][route] = {
        "completion": sum(item["completion"] * item["scenario_count"] for item in matches) / total,
        "forward_gain": sum(item["forward_gain"] * item["scenario_count"] for item in matches) / total,
        "slip": sum(item["terrain_tangent_slip"] * item["scenario_count"] for item in matches) / total,
      }
  return result


def decide(
  *, v7: dict[str, Any], control: dict[str, Any], candidate: dict[str, Any],
  required_candidate_minus_control: float = 0.10,
) -> dict[str, Any]:
  if not all(
    value.get("profile_contract") == v7.get("profile_contract")
    for value in (control, candidate)
  ):
    raise ValueError("matched teacher profile contract differs")
  if not all(
    value.get("evaluation_contract") == v7.get("evaluation_contract")
    for value in (control, candidate)
  ):
    raise ValueError("matched teacher evaluation provenance differs")
  v7_rows, control_rows, candidate_rows = map(_indexed, (v7, control, candidate))
  if set(v7_rows) != set(control_rows) or set(v7_rows) != set(candidate_rows):
    raise ValueError("matched teacher group inventory differs")
  violations: list[str] = []
  group_results = []
  for key in sorted(v7_rows):
    reference, observed = v7_rows[key], candidate_rows[key]
    if reference["scenario_ids"] != observed["scenario_ids"]:
      violations.append(f"matched_scenarios:{key}")
    completion_floor = reference["completion"] if key[0] == "high_slope" else reference["completion"] - 0.05
    if observed["completion"] < completion_floor:
      violations.append(f"completion:{key}")
    safety = {}
    for metric in SAFETY_METRICS:
      limit = 1.2 * reference[metric]
      passed = observed[metric] <= limit
      safety[metric] = {"value": observed[metric], "limit": limit, "pass": passed}
      if not passed:
        violations.append(f"safety:{metric}:{key}")
    group_results.append({
      "identity": key, "completion": observed["completion"],
      "v7_completion": reference["completion"], "safety": safety,
    })
  high = {
    "v7": _high_slope_routes(v7),
    "control": _high_slope_routes(control),
    "candidate": _high_slope_routes(candidate),
  }
  candidate_gains = []
  candidate_completions = []
  control_completions = []
  for profile, floor in (("clean", 0.75), ("randomized", 0.625)):
    for route in ("line", "arc", "s_curve"):
      if high["candidate"][profile][route]["completion"] < floor:
        violations.append(f"high_slope_completion:{profile}:{route}")
      candidate_gains.append(high["candidate"][profile][route]["forward_gain"])
      candidate_completions.append(high["candidate"][profile][route]["completion"])
      control_completions.append(high["control"][profile][route]["completion"])
  if sum(candidate_gains) / len(candidate_gains) < 0.80:
    violations.append("high_slope_forward_gain")
  candidate_macro = sum(candidate_completions) / len(candidate_completions)
  control_macro = sum(control_completions) / len(control_completions)
  if candidate_macro < control_macro + required_candidate_minus_control:
    violations.append("causal_delta")
  result = {
    "schema_version": 1,
    "evaluation_suite": "go2_privileged_teacher_full_acceptance",
    "weighted_score_used": False,
    "checkpoint_identities": {
      name: value["checkpoint"] for name, value in (
        ("v7", v7), ("control", control), ("candidate", candidate)
      )
    },
    "high_slope": high,
    "groups": group_results,
    "violations": violations,
    "decision": "ACCEPT" if not violations else "REJECT",
  }
  assert_recursive_json_finite(result)
  return result
