"""Diagnose randomization effects from a matched route evaluation JSON."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

# Prefer this worktree over an editable install that may target integration.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro

from src.tasks.velocity.evaluation.matched_route_metrics import (
  CORE_PROFILE_NAMES,
  FACTOR_PROFILE_NAMES,
  PROFILE_NAMES,
  ROUTE_KINDS,
  assert_recursive_finite,
  matched_thresholds,
)


@dataclass(frozen=True)
class DiagnosticConfig:
  input_file: str
  output_file: str = "go2_matched_route_diagnostics.json"
  mirror_relative_tolerance: float = 0.20


def _metric_mean(route_result: Mapping[str, Any], name: str) -> float | None:
  value = route_result.get(name, {}).get("mean")
  return None if value is None else float(value)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
  if numerator is None or denominator is None or denominator <= 0.0:
    return None
  return numerator / denominator


def _effect(profile: float | None, clean: float | None) -> dict[str, Any]:
  if profile is None or clean is None:
    return {"absolute_delta": None, "ratio": None, "reason": "profile_not_run"}
  return {
    "absolute_delta": profile - clean,
    "ratio": _ratio(profile, clean),
    "reason": None,
  }


def _mirror_summary(route_result: Mapping[str, Any], metric: str) -> dict[str, Any]:
  by_sign: dict[int, list[float]] = {-1: [], 1: []}
  for scenario in route_result["scenarios"]:
    value = scenario.get("sample_metrics", {}).get(metric, {}).get("mean")
    if value is not None:
      by_sign[int(scenario["turn_sign"])].append(float(value))
  means = {
    str(sign): (sum(values) / len(values) if values else None)
    for sign, values in by_sign.items()
  }
  left, right = means["1"], means["-1"]
  if left is None or right is None:
    relative = None
  else:
    relative = abs(left - right) / max((left + right) / 2.0, 1.0e-12)
  return {"mean_by_turn_sign": means, "relative_difference": relative}


def _validate_input(payload: Mapping[str, Any]) -> None:
  if payload.get("schema_version") != 1:
    raise ValueError("unsupported or missing matched evaluation schema_version")
  profiles = payload.get("profiles")
  if not isinstance(profiles, Mapping) or not profiles:
    raise ValueError("matched evaluation has no profiles")
  for profile_name, profile in profiles.items():
    if profile_name not in PROFILE_NAMES:
      raise ValueError(f"unknown profile in input: {profile_name!r}")
    route_results = profile.get("route_results", {})
    if tuple(route_results) != ROUTE_KINDS:
      raise ValueError(
        f"profile {profile_name!r} must contain route results in {ROUTE_KINDS} order"
      )
    invariants = profile.get("matched_invariants", {})
    for kind in ROUTE_KINDS:
      if route_results[kind].get("num_envs") != invariants.get("num_envs"):
        raise ValueError(f"num_envs mismatch for {profile_name}/{kind}")
      slots = [item["matched_slot"] for item in route_results[kind]["scenarios"]]
      if slots != list(range(len(slots))):
        raise ValueError(f"matched slots are incomplete for {profile_name}/{kind}")


def diagnose(payload: Mapping[str, Any], mirror_relative_tolerance: float = 0.20) -> dict[str, Any]:
  if not math.isfinite(mirror_relative_tolerance) or mirror_relative_tolerance < 0.0:
    raise ValueError("mirror_relative_tolerance must be finite and nonnegative")
  _validate_input(payload)
  profiles = payload["profiles"]
  profile_summaries: dict[str, Any] = {}
  for profile_name, profile in profiles.items():
    route_results = profile["route_results"]
    route_summary: dict[str, Any] = {}
    for kind in ROUTE_KINDS:
      result = route_results[kind]
      action_mirror = _mirror_summary(result, "action_acceleration")
      slip_mirror = _mirror_summary(result, "slip_velocity")
      mirror_values = (
        action_mirror["relative_difference"], slip_mirror["relative_difference"]
      )
      route_summary[kind] = {
        "completion_rate": result["completion_rate"],
        "catastrophic_termination_fraction": result[
          "catastrophic_termination_fraction"
        ],
        "action_acceleration": result["action_acceleration"],
        "slip_velocity": result["slip_velocity"],
        "velocity_error": result["velocity_error"],
        "cross_axis_velocity": result["cross_axis_velocity"],
        "mirror": {
          "action_acceleration": action_mirror,
          "slip_velocity": slip_mirror,
          "passed": all(
            value is not None and value <= mirror_relative_tolerance
            for value in mirror_values
          ),
        },
      }
    action = {
      kind: _metric_mean(route_results[kind], "action_acceleration")
      for kind in ROUTE_KINDS
    }
    slip = {
      kind: _metric_mean(route_results[kind], "slip_velocity")
      for kind in ROUTE_KINDS
    }
    route_summary["s_curve_ratios"] = {
      "action_vs_arc": _ratio(action["s_curve"], action["arc"]),
      "action_vs_straight": _ratio(action["s_curve"], action["straight"]),
      "slip_vs_arc": _ratio(slip["s_curve"], slip["arc"]),
      "slip_vs_straight": _ratio(slip["s_curve"], slip["straight"]),
    }
    profile_summaries[profile_name] = route_summary

  effects: dict[str, Any] = {}
  clean = profile_summaries.get("clean")
  for profile_name in PROFILE_NAMES:
    if profile_name == "clean":
      continue
    current = profile_summaries.get(profile_name)
    effects[profile_name] = {}
    for kind in ROUTE_KINDS:
      effects[profile_name][kind] = {}
      for metric in ("action_acceleration", "slip_velocity", "velocity_error"):
        current_value = (
          None if current is None else current[kind][metric]["mean"]
        )
        clean_value = None if clean is None else clean[kind][metric]["mean"]
        effects[profile_name][kind][metric] = _effect(
          current_value, clean_value
        )

  full = profile_summaries.get("full_randomized")
  thresholds = None
  attribution = "insufficient_profiles"
  if full is not None:
    s_action = full["s_curve"]["action_acceleration"]["mean"]
    arc_action = full["arc"]["action_acceleration"]["mean"]
    straight_action = full["straight"]["action_acceleration"]["mean"]
    s_slip = full["s_curve"]["slip_velocity"]["mean"]
    reference_slip_values = [
      value for value in (
        full["arc"]["slip_velocity"]["mean"],
        full["straight"]["slip_velocity"]["mean"],
      ) if value is not None
    ]
    catastrophic = max(
      full[kind]["catastrophic_termination_fraction"] for kind in ROUTE_KINDS
    )
    if None not in (s_action, arc_action, straight_action, s_slip) and reference_slip_values:
      thresholds = matched_thresholds(
        s_action=s_action, arc_action=arc_action,
        straight_action=straight_action, s_slip=s_slip,
        reference_slip=sum(reference_slip_values) / len(reference_slip_values),
        catastrophic_fraction=catastrophic,
      )
      if thresholds["s_vs_arc_action_acceleration"] and thresholds["s_vs_straight_action_acceleration"]:
        attribution = "not_s_curve_specific"
      else:
        attribution = "s_curve_specific_candidate"

  missing_ablation = [name for name in CORE_PROFILE_NAMES if name not in profiles]
  core_effects = []
  for profile_name in ("dynamics_only", "observation_only", "push_only"):
    if profile_name not in profiles:
      continue
    effect = effects[profile_name]["s_curve"]["action_acceleration"]
    core_effects.append({
      "profile": profile_name,
      "absolute_delta": effect["absolute_delta"],
      "ratio": effect["ratio"],
    })
  core_effects.sort(
    key=lambda item: (
      item["absolute_delta"] is not None,
      item["absolute_delta"] if item["absolute_delta"] is not None else -math.inf,
    ),
    reverse=True,
  )
  factor_effects = []
  for profile_name in FACTOR_PROFILE_NAMES:
    if profile_name not in profiles:
      continue
    effect = effects[profile_name]["s_curve"]["action_acceleration"]
    factor_effects.append({
      "profile": profile_name,
      "absolute_delta": effect["absolute_delta"],
      "ratio": effect["ratio"],
    })
  factor_effects.sort(
    key=lambda item: (
      item["absolute_delta"] is not None,
      item["absolute_delta"] if item["absolute_delta"] is not None else -math.inf,
    ),
    reverse=True,
  )
  result = {
    "schema_version": 1,
    "source": {
      "checkpoint": payload.get("checkpoint"),
      "git_head": payload.get("git_head"),
      "seed": payload.get("seed"),
      "action_acceleration_definition": payload.get(
        "action_acceleration_definition"
      ),
    },
    "profiles": profile_summaries,
    "effects_vs_clean": effects,
    "core_action_acceleration_ranking_s_curve": core_effects,
    "factor_action_acceleration_ranking_s_curve": factor_effects,
    "full_randomized_thresholds": thresholds,
    "action_acceleration_attribution": attribution,
    "missing_ablation_profiles": missing_ablation,
    "ablation_complete": not missing_ablation and "clean" in profiles,
    "factor_ablation_complete": all(
      name in profiles for name in FACTOR_PROFILE_NAMES
    ),
  }
  assert_recursive_finite(result)
  return result


def main() -> None:
  cfg = tyro.cli(DiagnosticConfig)
  source = Path(cfg.input_file)
  payload = json.loads(source.read_text())
  result = diagnose(payload, cfg.mirror_relative_tolerance)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  print(f"[INFO] Wrote matched diagnostics to {output}")


if __name__ == "__main__":
  main()
