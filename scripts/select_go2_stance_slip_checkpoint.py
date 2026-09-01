"""Select a stance-slip checkpoint from preregistered matched evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")
GUARDRAILS = ("slip", "action_acceleration", "base_pitch", "base_contact", "upper_leg_contact", "calf_contact", "failure_risk")


@dataclass(frozen=True)
class SelectionConfig:
  baseline_json: str
  candidate_jsons: tuple[str, ...]
  output_file: str


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path_string: str) -> tuple[Path, dict[str, Any]]:
  path = Path(path_string).expanduser().resolve()
  payload = json.loads(path.read_text())
  assert_recursive_json_finite(payload)
  if payload.get("evaluation_suite") != "high_slope_matched_straight_arc_s_curve":
    raise ValueError(f"unexpected evaluation suite in {path}")
  if tuple(payload.get("profiles", {})) != PROFILES:
    raise ValueError(f"{path} must contain clean and randomized profiles")
  return path, payload


def _weighted(rows: list[dict[str, Any]], field: str, weight) -> float:
  denominator = sum(weight(row) for row in rows)
  if denominator <= 0:
    raise ValueError(f"non-positive denominator for {field}")
  return sum(float(row[field]) * weight(row) for row in rows) / denominator


def aggregate_route(payload: dict[str, Any], profile: str, route: str) -> dict[str, float]:
  rows = payload["profiles"][profile]["route_results"][route]["scenarios"]
  if len(rows) != 16:
    raise ValueError(f"{profile}/{route} must contain exactly 16 matched scenarios")
  steps = sum(int(row["steps_sampled"]) for row in rows)
  loaded = sum(int(row["terrain_tangent_loaded_stance"]["loaded_stance_foot_samples"]) for row in rows)
  if steps <= 0 or loaded <= 0:
    raise ValueError(f"{profile}/{route} has an empty guardrail denominator")
  gain_weight = lambda row: int(row["steps_sampled"]) * float(row["speed"]) ** 2
  gain_denominator = sum(gain_weight(row) for row in rows)
  return {
    "completion": sum(bool(row["completed"]) for row in rows) / len(rows),
    "forward_gain": sum(float(row["response_gain"]["vx"]) * gain_weight(row) for row in rows) / gain_denominator,
    "slip": sum(float(row["terrain_tangent_stance_slip_mean"]) * int(row["terrain_tangent_loaded_stance"]["loaded_stance_foot_samples"]) for row in rows) / loaded,
    "action_acceleration": _weighted(rows, "action_acceleration_mean", lambda row: int(row["steps_sampled"])),
    "base_pitch": _weighted(rows, "base_pitch_absolute_mean", lambda row: int(row["steps_sampled"])),
    "base_contact": sum(int(row["base_contact_count"]) for row in rows) / steps,
    "upper_leg_contact": sum(int(row["upper_leg_contact_count"]) for row in rows) / steps,
    "calf_contact": sum(int(row["calf_contact_count"]) for row in rows) / steps,
    "failure_risk": sum(bool(row["catastrophic_termination"]) for row in rows) / len(rows),
  }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
  routes = {
    profile: {route: aggregate_route(payload, profile, route) for route in ROUTES}
    for profile in PROFILES
  }
  all_routes = [routes[profile][route] for profile in PROFILES for route in ROUTES]
  return {
    "routes": routes,
    "lexicographic": {
      "minimum_clean_completion": min(routes["clean"][route]["completion"] for route in ROUTES),
      "minimum_randomized_completion": min(routes["randomized"][route]["completion"] for route in ROUTES),
      "minimum_forward_gain": min(item["forward_gain"] for item in all_routes),
      "mean_terrain_tangent_loaded_stance_slip": sum(item["slip"] for item in all_routes) / len(all_routes),
    },
  }


def select(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
  baseline_summary = summarize(baseline)
  stages = []
  survivors = []
  for payload in candidates:
    summary = summarize(payload)
    violations = []
    for profile in PROFILES:
      for route in ROUTES:
        reference = baseline_summary["routes"][profile][route]
        candidate = summary["routes"][profile][route]
        for metric in GUARDRAILS:
          threshold = reference[metric] * 1.2
          if candidate[metric] > threshold + 1.0e-12:
            violations.append({
              "profile": profile, "route": route, "metric": metric,
              "candidate": candidate[metric], "baseline": reference[metric],
              "threshold_1p2x": threshold,
            })
    record = {
      "checkpoint": payload["checkpoint"],
      "summary": summary,
      "guardrail_pass": not violations,
      "guardrail_violations": violations,
    }
    stages.append(record)
    if not violations:
      survivors.append(record)
  survivors.sort(key=lambda item: (
    -item["summary"]["lexicographic"]["minimum_clean_completion"],
    -item["summary"]["lexicographic"]["minimum_randomized_completion"],
    -item["summary"]["lexicographic"]["minimum_forward_gain"],
    item["summary"]["lexicographic"]["mean_terrain_tangent_loaded_stance_slip"],
    int(Path(item["checkpoint"]).stem.removeprefix("model_")),
  ))
  return {
    "baseline": {"checkpoint": baseline["checkpoint"], "summary": baseline_summary},
    "stages": stages,
    "survivor_count": len(survivors),
    "selected_checkpoint": survivors[0]["checkpoint"] if survivors else None,
    "selection_status": "SELECTED" if survivors else "NO_SAFE_SURVIVOR",
  }


def main() -> None:
  cfg = tyro.cli(SelectionConfig)
  baseline_path, baseline = _load(cfg.baseline_json)
  loaded_candidates = [_load(path) for path in cfg.candidate_jsons]
  result = {
    "schema_version": 1,
    "selection_method": "hard_1p2x_guardrails_then_lexicographic",
    "input_artifacts": {
      "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
      "candidates": [{"path": str(path), "sha256": _sha256(path)} for path, _ in loaded_candidates],
    },
    **select(baseline, [payload for _, payload in loaded_candidates]),
  }
  assert_recursive_json_finite(result)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
