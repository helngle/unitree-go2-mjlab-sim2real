"""Offline attribution for matched high-slope routes and level-9 stairs.

The tool consumes saved JSON only.  It never imports the simulator, creates an
environment, loads a policy, or starts training.  Invalid geometry, unmatched
slots, missing metrics, NaN/Inf, or contract drift are rejected before any
policy attribution is emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROUTE_KINDS = ("straight", "arc", "s_curve")
STAIRS_DIRECTIONS = ("stairs_up", "stairs_down")
STAIRS_SEEDS = (42, 43, 44)
AXES = ("vx", "vy", "wz")
MATCHED_FIELDS = (
  "slope_direction",
  "level",
  "difficulty_label",
  "difficulty",
  "effective_terrain_parameters",
  "radius",
  "speed",
  "turn_sign",
  "repeat",
  "route_length",
)


@dataclass(frozen=True)
class AttributionThresholds:
  pass_completion_rate: float = 0.80
  failure_completion_rate: float = 0.50
  forward_under_response_gain: float = 0.80
  similar_forward_gain_absolute_spread: float = 0.15
  controller_saturation_fraction: float = 0.10
  geometry_error_tolerance: float = 1.0e-4
  near_end_progress_ratio: float = 0.90
  slow_speed_max: float = 0.30
  nominal_steps: int = 2400
  single_retry_steps: int = 3000

  def validate(self) -> None:
    fractions = (
      self.pass_completion_rate,
      self.failure_completion_rate,
      self.forward_under_response_gain,
      self.controller_saturation_fraction,
      self.near_end_progress_ratio,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
      raise ValueError("fraction thresholds must be finite and in [0, 1]")
    if self.failure_completion_rate >= self.pass_completion_rate:
      raise ValueError("failure completion rate must be below pass completion rate")
    if (
      not math.isfinite(self.similar_forward_gain_absolute_spread)
      or self.similar_forward_gain_absolute_spread < 0.0
    ):
      raise ValueError("forward gain spread must be finite and nonnegative")
    if not math.isfinite(self.geometry_error_tolerance) or self.geometry_error_tolerance < 0.0:
      raise ValueError("geometry error tolerance must be finite and nonnegative")
    if self.nominal_steps <= 0 or self.single_retry_steps <= self.nominal_steps:
      raise ValueError("retry steps must exceed positive nominal steps")


def _fail(path: str, message: str) -> None:
  raise ValueError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    _fail(path, "expected an object")
  return value


def _sequence(value: Any, path: str, length: int | None = None) -> Sequence[Any]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
    _fail(path, "expected an array")
  if length is not None and len(value) != length:
    _fail(path, f"expected {length} values, got {len(value)}")
  return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    _fail(path, "expected a number")
  result = float(value)
  if not math.isfinite(result):
    _fail(path, "must be finite")
  if minimum is not None and result < minimum:
    _fail(path, f"must be >= {minimum}")
  return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
  result = _number(value, path, minimum=float(minimum))
  if not result.is_integer():
    _fail(path, "expected an integer")
  return int(result)


def _boolean(value: Any, path: str) -> bool:
  if not isinstance(value, bool):
    _fail(path, "expected a boolean")
  return value


def _required(data: Mapping[str, Any], name: str, path: str) -> Any:
  if name not in data:
    _fail(path, f"missing required field {name!r}")
  return data[name]


def _finite_recursive(value: Any, path: str = "root") -> None:
  if value is None or isinstance(value, (str, bool, int)):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      _fail(path, "contains NaN or Inf")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      _finite_recursive(item, f"{path}.{key}")
    return
  if isinstance(value, Sequence):
    for index, item in enumerate(value):
      _finite_recursive(item, f"{path}[{index}]")
    return
  _fail(path, f"unsupported JSON value type {type(value).__name__}")


def _nullable_metric(data: Mapping[str, Any], name: str, path: str) -> float | None:
  value = _required(data, name, path)
  if value is None:
    reason = data.get(f"{name}_reason")
    if not isinstance(reason, str) or not reason:
      _fail(path, f"null {name!r} requires nonempty {name}_reason")
    return None
  return _number(value, f"{path}.{name}")


def _distribution(data: Mapping[str, Any], prefix: str, path: str) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for statistic in ("mean", "p95", "max"):
    name = f"{prefix}_{statistic}"
    result[statistic] = _nullable_metric(data, name, path)
    if result[statistic] is None:
      result[f"{statistic}_reason"] = data[f"{name}_reason"]
  available = [value for value in (result["mean"], result["p95"], result["max"]) if value is not None]
  if any(value < 0.0 for value in available):
    _fail(path, f"{prefix} statistics must be nonnegative")
  return result


def _mean(values: Iterable[float | None]) -> float | None:
  finite = [float(value) for value in values if value is not None]
  return sum(finite) / len(finite) if finite else None


def _reported(value: float | None, reason: str) -> dict[str, Any]:
  return {"value": value, "reason": None if value is not None else reason}


def _axis_gain(actual: float, commanded: float) -> dict[str, Any]:
  if abs(commanded) <= 1.0e-9:
    return {"value": None, "reason": "mean_command_is_zero"}
  return {"value": actual / commanded, "reason": None}


def _validate_scenario(
  scenario: Mapping[str, Any],
  route_kind: str,
  path: str,
  thresholds: AttributionThresholds,
  terrain_assignment_error_max: float,
  route_placement_error_max: float,
) -> dict[str, Any]:
  slot = _integer(_required(scenario, "matched_slot", path), f"{path}.matched_slot")
  if _required(scenario, "route_kind", path) != route_kind:
    _fail(path, "route_kind disagrees with containing route result")
  matched = {name: _required(scenario, name, path) for name in MATCHED_FIELDS}
  for name in ("level", "repeat"):
    _integer(matched[name], f"{path}.{name}")
  _number(matched["difficulty"], f"{path}.difficulty", minimum=0.0)
  for name in ("radius", "speed", "route_length"):
    if _number(matched[name], f"{path}.{name}", minimum=0.0) <= 0.0:
      _fail(f"{path}.{name}", "must be positive")
  if (
    isinstance(matched["turn_sign"], bool)
    or not isinstance(matched["turn_sign"], int)
    or matched["turn_sign"] not in (-1, 1)
  ):
    _fail(f"{path}.turn_sign", "must be -1 or +1")
  for name in ("slope_direction", "difficulty_label"):
    if not isinstance(matched[name], str) or not matched[name]:
      _fail(f"{path}.{name}", "must be a nonempty string")
  _mapping(matched["effective_terrain_parameters"], f"{path}.effective_terrain_parameters")

  geometry = _mapping(scenario.get("geometry", scenario), f"{path}.geometry")
  for name in ("corridor_inside_patch", "scan_footprint_inside_patch"):
    if not _boolean(_required(geometry, name, f"{path}.geometry"), f"{path}.geometry.{name}"):
      _fail(path, f"unsafe geometry: {name} is false")
  for name in ("corridor_boundary_margin", "scan_boundary_margin"):
    _number(_required(geometry, name, f"{path}.geometry"), f"{path}.geometry.{name}", minimum=0.0)
  placement_error = _number(
    scenario.get(
      "route_placement_position_error",
      scenario.get("route_placement_error", route_placement_error_max),
    ),
    f"{path}.route_placement_position_error",
    minimum=0.0,
  )
  if placement_error > thresholds.geometry_error_tolerance:
    _fail(path, "route placement error exceeds attribution tolerance")
  terrain_assignment_error = _number(
    scenario.get("terrain_assignment_position_error", terrain_assignment_error_max),
    f"{path}.terrain_assignment_position_error",
    minimum=0.0,
  )
  if terrain_assignment_error > thresholds.geometry_error_tolerance:
    _fail(path, "terrain assignment error exceeds attribution tolerance")

  completed = _boolean(_required(scenario, "completed", path), f"{path}.completed")
  failed = _boolean(_required(scenario, "failed", path), f"{path}.failed")
  if completed == failed:
    _fail(path, "exactly one of completed and failed must be true")
  catastrophic = _boolean(
    _required(scenario, "catastrophic_termination", path),
    f"{path}.catastrophic_termination",
  )
  first_reason = _required(scenario, "first_failure_reason", path)
  if failed and (not isinstance(first_reason, str) or not first_reason):
    _fail(path, "failed scenario requires first_failure_reason")
  if completed and first_reason is not None:
    _fail(path, "completed scenario must have null first_failure_reason")

  commanded = [
    _number(value, f"{path}.commanded_velocity_mean[{index}]")
    for index, value in enumerate(
      _sequence(_required(scenario, "commanded_velocity_mean", path), f"{path}.commanded_velocity_mean", 3)
    )
  ]
  actual = [
    _number(value, f"{path}.actual_velocity_mean[{index}]")
    for index, value in enumerate(
      _sequence(_required(scenario, "actual_velocity_mean", path), f"{path}.actual_velocity_mean", 3)
    )
  ]
  response_gain_input = _mapping(
    _required(scenario, "response_gain", path), f"{path}.response_gain"
  )
  response_gain = {
    axis: _reported(
      _nullable_metric(response_gain_input, axis, f"{path}.response_gain"),
      str(response_gain_input.get(f"{axis}_reason", "unavailable")),
    )
    for axis in AXES
  }
  saturation = _number(
    _required(scenario, "controller_saturation_fraction", path),
    f"{path}.controller_saturation_fraction",
    minimum=0.0,
  )
  if saturation > 1.0:
    _fail(path, "controller_saturation_fraction must be <= 1")

  path_metrics = {}
  for prefix in ("cross_track", "heading"):
    path_metrics[prefix] = {
      statistic: _nullable_metric(scenario, f"{prefix}_{statistic}", path)
      for statistic in ("rms", "p95", "max", "final")
    }
  final_position_error = _nullable_metric(scenario, "final_position_error", path)
  distributions = {
    name: _distribution(scenario, name, path)
    for name in ("action_acceleration", "slip_velocity")
  }

  contacts: dict[str, Any] = {}
  for body in ("base", "upper_leg", "calf"):
    count = _integer(
      _required(scenario, f"{body}_contact_count", path),
      f"{path}.{body}_contact_count",
    )
    rate = _number(
      _required(scenario, f"{body}_contact_rate", path),
      f"{path}.{body}_contact_rate",
      minimum=0.0,
    )
    if rate > 1.0:
      _fail(path, f"{body}_contact_rate must be <= 1")
    contacts[body] = {"non_terminating_count": count, "non_terminating_rate": rate}

  terminations = _mapping(
    _required(scenario, "termination_counts", path), f"{path}.termination_counts"
  )
  termination_counts = {
    str(name): _number(value, f"{path}.termination_counts.{name}", minimum=0.0)
    for name, value in terminations.items()
  }
  if "contact_termination_summary" in scenario:
    _mapping(
      scenario["contact_termination_summary"],
      f"{path}.contact_termination_summary",
    )
  reset_count = _integer(_required(scenario, "reset_count", path), f"{path}.reset_count")

  return {
    "matched_slot": slot,
    "matched_fields": matched,
    "completed": completed,
    "failed": failed,
    "catastrophic_termination": catastrophic,
    "first_failure_reason": first_reason,
    "steps_sampled": _integer(
      _required(scenario, "steps_sampled", path), f"{path}.steps_sampled", minimum=1
    ),
    "progress_ratio": _number(_required(scenario, "progress_ratio", path), f"{path}.progress_ratio"),
    "commanded_velocity_mean": dict(zip(AXES, commanded)),
    "actual_velocity_mean": dict(zip(AXES, actual)),
    "calculated_response_gain": {
      axis: _axis_gain(actual[index], commanded[index])
      for index, axis in enumerate(AXES)
    },
    "evaluator_response_gain": response_gain,
    "path_metrics": path_metrics,
    "final_position_error": final_position_error,
    "controller_saturation_fraction": saturation,
    "distributions": distributions,
    "contacts": contacts,
    "termination_counts": termination_counts,
    "reset_count": reset_count,
  }


def _extract_profiles(payload: Mapping[str, Any]) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
  profiles = payload.get("profiles")
  if profiles is not None:
    profiles = _mapping(profiles, "root.profiles")
    if not profiles:
      _fail("root.profiles", "must not be empty")
    return {
      str(name): (
        _mapping(_required(_mapping(value, f"root.profiles.{name}"), "matched_invariants", f"root.profiles.{name}"), f"root.profiles.{name}.matched_invariants"),
        _mapping(_required(_mapping(value, f"root.profiles.{name}"), "route_results", f"root.profiles.{name}"), f"root.profiles.{name}.route_results"),
      )
      for name, value in profiles.items()
    }
  return {
    str(payload.get("profile", "unspecified")): (
      _mapping(_required(payload, "matched_invariants", "root"), "root.matched_invariants"),
      _mapping(_required(payload, "route_results", "root"), "root.route_results"),
    )
  }


def _route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  completion = sum(row["completed"] for row in rows) / len(rows)
  gains = {
    axis: _mean(row["calculated_response_gain"][axis]["value"] for row in rows)
    for axis in AXES
  }
  term_names = sorted({name for row in rows for name in row["termination_counts"]})
  return {
    "num_scenarios": len(rows),
    "completion_rate": completion,
    "progress_ratio": {
      "mean": _mean(row["progress_ratio"] for row in rows),
      "min": min(row["progress_ratio"] for row in rows),
      "max": max(row["progress_ratio"] for row in rows),
    },
    "commanded_velocity_mean": {
      axis: _mean(row["commanded_velocity_mean"][axis] for row in rows) for axis in AXES
    },
    "actual_velocity_mean": {
      axis: _mean(row["actual_velocity_mean"][axis] for row in rows) for axis in AXES
    },
    "response_gain_mean": {
      axis: {
        "value": gains[axis],
        "reason": None if gains[axis] is not None else "mean_command_is_zero_for_all_scenarios",
      }
      for axis in AXES
    },
    "path_metrics_mean": {
      prefix: {
        statistic: _reported(
          _mean(row["path_metrics"][prefix][statistic] for row in rows),
          "metric_unavailable_in_all_scenarios",
        )
        for statistic in ("rms", "p95", "max", "final")
      }
      for prefix in ("cross_track", "heading")
    },
    "final_position_error_mean": _reported(
      _mean(row["final_position_error"] for row in rows),
      "metric_unavailable_in_all_scenarios",
    ),
    "controller_saturation": {
      "mean": _mean(row["controller_saturation_fraction"] for row in rows),
      "max": max(row["controller_saturation_fraction"] for row in rows),
    },
    "action_acceleration": {
      statistic: _reported(
        _mean(row["distributions"]["action_acceleration"][statistic] for row in rows),
        "metric_unavailable_in_all_scenarios",
      )
      for statistic in ("mean", "p95", "max")
    },
    "slip_velocity": {
      statistic: _reported(
        _mean(row["distributions"]["slip_velocity"][statistic] for row in rows),
        "metric_unavailable_in_all_scenarios",
      )
      for statistic in ("mean", "p95", "max")
    },
    "non_terminating_contacts": {
      body: {
        "count": sum(row["contacts"][body]["non_terminating_count"] for row in rows),
        "rate_mean": _mean(row["contacts"][body]["non_terminating_rate"] for row in rows),
      }
      for body in ("base", "upper_leg", "calf")
    },
    "catastrophic_termination_count": sum(row["catastrophic_termination"] for row in rows),
    "reset_count": sum(row["reset_count"] for row in rows),
    "termination_counts": {
      name: sum(row["termination_counts"].get(name, 0.0) for row in rows)
      for name in term_names
    },
    "first_failure_reasons": dict(sorted(Counter(
      row["first_failure_reason"] for row in rows if row["first_failure_reason"] is not None
    ).items())),
  }


def _classify(
  summaries: Mapping[str, Mapping[str, Any]],
  retry_candidates: list[dict[str, Any]],
  thresholds: AttributionThresholds,
) -> dict[str, Any]:
  completion = {kind: float(summaries[kind]["completion_rate"]) for kind in ROUTE_KINDS}
  vx_gain = {
    kind: summaries[kind]["response_gain_mean"]["vx"]["value"] for kind in ROUTE_KINDS
  }
  saturation_ok = all(
    float(summaries[kind]["controller_saturation"]["max"])
    <= thresholds.controller_saturation_fraction
    for kind in ROUTE_KINDS
  )
  evidence = {
    "completion_rate": completion,
    "forward_response_gain_mean": vx_gain,
    "controller_saturation_within_limit": saturation_ok,
  }
  if retry_candidates:
    return {
      "classification": "horizon_retry_required",
      "training_authorized": False,
      "recommended_single_variable": None,
      "evidence": evidence,
      "reason": "near-end slow step-limit cases must receive the single predeclared retry first",
    }
  if all(completion[kind] >= thresholds.pass_completion_rate for kind in ROUTE_KINDS):
    return {
      "classification": "all_routes_passed_no_training",
      "training_authorized": False,
      "recommended_single_variable": None,
      "evidence": evidence,
      "reason": "straight, arc, and S meet the predeclared completion threshold",
    }
  vx_values = [vx_gain[kind] for kind in ROUTE_KINDS]
  sustained = (
    saturation_ok
    and all(completion[kind] <= thresholds.failure_completion_rate for kind in ROUTE_KINDS)
    and all(value is not None and value <= thresholds.forward_under_response_gain for value in vx_values)
    and max(float(value) for value in vx_values) - min(float(value) for value in vx_values)
    <= thresholds.similar_forward_gain_absolute_spread
  )
  if sustained:
    return {
      "classification": "sustained_high_slope_locomotion_limitation",
      "training_authorized": False,
      "recommended_single_variable": "increase_high_extreme_sustained_slope_sampling",
      "evidence": evidence,
      "reason": "all route kinds fail with similar forward under-response",
    }
  curvature = (
    saturation_ok
    and completion["straight"] >= thresholds.pass_completion_rate
    and completion["arc"] <= thresholds.failure_completion_rate
    and completion["s_curve"] <= thresholds.failure_completion_rate
  )
  if curvature:
    return {
      "classification": "high_slope_forward_yaw_curvature_coupling_limitation",
      "training_authorized": False,
      "recommended_single_variable": "increase_high_slope_parameterized_forward_yaw_sampling",
      "evidence": evidence,
      "reason": "straight passes while both curved route kinds consistently fail",
    }
  return {
    "classification": "inconclusive_no_training",
    "training_authorized": False,
    "recommended_single_variable": None,
    "evidence": evidence,
    "reason": "results do not satisfy a predeclared attribution rule",
  }


def analyze_matched(
  payload: Mapping[str, Any], thresholds: AttributionThresholds
) -> dict[str, Any]:
  thresholds.validate()
  _finite_recursive(payload)
  if payload.get("schema_version") != 1:
    _fail("root.schema_version", "must be 1")
  if payload.get("evaluation_suite") != "high_slope_matched_straight_arc_s_curve":
    _fail("root.evaluation_suite", "unexpected evaluation suite")
  metric_invariants = _mapping(
    _required(payload, "metric_invariants", "root"), "root.metric_invariants"
  )
  for name in (
    "sample_denominator", "action_acceleration_definition", "attempt_freeze",
    "settle_lifecycle",
  ):
    if not isinstance(_required(metric_invariants, name, "root.metric_invariants"), str):
      _fail(f"root.metric_invariants.{name}", "must be a string")
  coverage = _mapping(_required(payload, "coverage", "root"), "root.coverage")
  if coverage.get("training_changed") is not False:
    _fail("root.coverage.training_changed", "must be false for evaluation-only evidence")
  root_identity = {
    name: _required(payload, name, "root") for name in ("checkpoint", "task_id", "seed")
  }
  for name in ("checkpoint", "task_id"):
    if not isinstance(root_identity[name], str) or not root_identity[name]:
      _fail(f"root.{name}", "must be a nonempty string")
  root_identity["seed"] = _integer(root_identity["seed"], "root.seed")
  profiles_output: dict[str, Any] = {}
  for profile_name, (invariants, route_results) in _extract_profiles(payload).items():
    if tuple(route_results) != ROUTE_KINDS:
      _fail(f"profile.{profile_name}.route_results", f"must preserve order {ROUTE_KINDS}")
    for name, value in root_identity.items():
      if invariants.get(name) != value:
        _fail(f"profile.{profile_name}.matched_invariants.{name}", "disagrees with top-level identity")
    if invariants.get("profile") != profile_name:
      _fail(f"profile.{profile_name}.matched_invariants.profile", "profile name mismatch")
    num_envs_value = invariants.get(
      "num_envs_per_route_kind", invariants.get("num_envs")
    )
    if num_envs_value is None:
      _fail(
        f"profile.{profile_name}.matched_invariants",
        "missing num_envs_per_route_kind",
      )
    num_envs = _integer(
      num_envs_value,
      f"profile.{profile_name}.matched_invariants.num_envs",
      minimum=1,
    )
    steps = _integer(
      _required(invariants, "steps", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.steps",
      minimum=1,
    )
    route_kinds = list(_sequence(
      _required(invariants, "route_kinds", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.route_kinds",
    ))
    if route_kinds != list(ROUTE_KINDS):
      _fail(f"profile.{profile_name}.matched_invariants.route_kinds", "route order mismatch")
    slot_order = list(_sequence(
      _required(invariants, "matched_slot_order", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.matched_slot_order",
    ))
    if slot_order != list(range(num_envs)):
      _fail(f"profile.{profile_name}.matched_invariants.matched_slot_order", "must be 0..num_envs-1")
    if invariants.get("route_length_definition") != "2*pi*radius/3":
      _fail(f"profile.{profile_name}.matched_invariants.route_length_definition", "unexpected definition")
    for name in ("fresh_environment_per_route_kind", "same_seed_environment_reconstruction"):
      if invariants.get(name) is not True:
        _fail(f"profile.{profile_name}.matched_invariants.{name}", "must be true")
    control_dt = _number(
      _required(invariants, "control_dt", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.control_dt",
      minimum=0.0,
    )
    if control_dt <= 0.0:
      _fail(f"profile.{profile_name}.matched_invariants.control_dt", "must be positive")
    settle_steps = _integer(
      _required(invariants, "settle_steps", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.settle_steps",
    )
    if settle_steps >= steps:
      _fail(f"profile.{profile_name}.matched_invariants.settle_steps", "must be below steps")
    controller_limits = _mapping(
      _required(invariants, "controller_limits", f"profile.{profile_name}.matched_invariants"),
      f"profile.{profile_name}.matched_invariants.controller_limits",
    )
    for name in (
      "cross_track_gain", "heading_gain", "max_lateral_speed", "max_yaw_rate",
      "cross_track_tolerance", "heading_tolerance",
    ):
      if _number(
        _required(controller_limits, name, f"profile.{profile_name}.matched_invariants.controller_limits"),
        f"profile.{profile_name}.matched_invariants.controller_limits.{name}",
        minimum=0.0,
      ) <= 0.0:
        _fail(f"profile.{profile_name}.matched_invariants.controller_limits.{name}", "must be positive")
    rows_by_kind: dict[str, list[dict[str, Any]]] = {}
    common_profile_settings: Mapping[str, Any] | None = None
    for kind in ROUTE_KINDS:
      result = _mapping(route_results[kind], f"profile.{profile_name}.route_results.{kind}")
      if result.get("route_kind") != kind:
        _fail(f"profile.{profile_name}.route_results.{kind}", "route_kind mismatch")
      if _integer(result.get("num_envs"), f"profile.{profile_name}.route_results.{kind}.num_envs", minimum=1) != num_envs:
        _fail(f"profile.{profile_name}.route_results.{kind}.num_envs", "does not match invariant")
      route_invariants = _mapping(
        _required(result, "route_kind_invariants", f"profile.{profile_name}.route_results.{kind}"),
        f"profile.{profile_name}.route_results.{kind}.route_kind_invariants",
      )
      expected_route_invariants = {
        "checkpoint": root_identity["checkpoint"],
        "task_id": root_identity["task_id"],
        "seed": root_identity["seed"],
        "profile": profile_name,
        "num_envs": num_envs,
        "steps": steps,
        "settle_steps": settle_steps,
        "controller_limits": controller_limits,
      }
      if dict(route_invariants) != expected_route_invariants:
        _fail(f"profile.{profile_name}.route_results.{kind}.route_kind_invariants", "does not match common invariants")
      profile_settings = _mapping(
        _required(result, "profile_settings", f"profile.{profile_name}.route_results.{kind}"),
        f"profile.{profile_name}.route_results.{kind}.profile_settings",
      )
      if common_profile_settings is None:
        common_profile_settings = profile_settings
      elif profile_settings != common_profile_settings:
        _fail(f"profile.{profile_name}.route_results.{kind}.profile_settings", "differs across route kinds")
      terrain_assignment_error_max = 0.0
      route_placement_error_max = 0.0
      for error_name in (
        "terrain_assignment_position_error_max", "route_placement_position_error_max"
      ):
        error = _number(_required(result, error_name, f"profile.{profile_name}.route_results.{kind}"), f"profile.{profile_name}.route_results.{kind}.{error_name}", minimum=0.0)
        if error > thresholds.geometry_error_tolerance:
          _fail(f"profile.{profile_name}.route_results.{kind}.{error_name}", "exceeds attribution tolerance")
        if error_name == "route_placement_position_error_max":
          route_placement_error_max = error
        else:
          terrain_assignment_error_max = error
      scenarios = _sequence(
        _required(result, "scenarios", f"profile.{profile_name}.route_results.{kind}"),
        f"profile.{profile_name}.route_results.{kind}.scenarios",
      )
      if len(scenarios) != num_envs:
        _fail(f"profile.{profile_name}.route_results.{kind}.scenarios", "count does not match num_envs")
      rows_by_kind[kind] = [
        _validate_scenario(
          _mapping(row, f"profile.{profile_name}.{kind}.scenarios[{index}]"),
          kind,
          f"profile.{profile_name}.{kind}.scenarios[{index}]",
          thresholds,
          terrain_assignment_error_max,
          route_placement_error_max,
        )
        for index, row in enumerate(scenarios)
      ]
      slots = [row["matched_slot"] for row in rows_by_kind[kind]]
      if slots != list(range(num_envs)):
        _fail(f"profile.{profile_name}.{kind}.matched_slot", "slots must be ordered 0..num_envs-1")

    for slot in range(num_envs):
      reference = rows_by_kind["straight"][slot]["matched_fields"]
      for kind in ROUTE_KINDS[1:]:
        if rows_by_kind[kind][slot]["matched_fields"] != reference:
          _fail(f"profile.{profile_name}.matched_slot[{slot}]", f"{kind} invariants differ from straight")

    retry_candidates = []
    if steps == thresholds.nominal_steps:
      for kind in ROUTE_KINDS:
        for row in rows_by_kind[kind]:
          if (
            row["failed"]
            and row["first_failure_reason"] == "step_limit"
            and float(row["matched_fields"]["speed"]) <= thresholds.slow_speed_max + 1.0e-9
            and row["progress_ratio"] >= thresholds.near_end_progress_ratio
            and not row["catastrophic_termination"]
            and row["reset_count"] == 0
          ):
            retry_candidates.append({
              "route_kind": kind,
              "matched_slot": row["matched_slot"],
              "original_steps": steps,
              "retry_steps": thresholds.single_retry_steps,
              "progress_ratio": row["progress_ratio"],
              "retry_limit": 1,
            })
    summaries = {kind: _route_summary(rows_by_kind[kind]) for kind in ROUTE_KINDS}
    profiles_output[profile_name] = {
      "matched_invariants": dict(invariants),
      "route_summaries": summaries,
      "retry_contract": {
        "candidates": retry_candidates,
        "retry_steps": thresholds.single_retry_steps,
        "maximum_retries_per_slot": 1,
        "all_other_acceptance_fields_must_remain_identical": True,
      },
      "attribution": _classify(summaries, retry_candidates, thresholds),
    }
  return {"identity": root_identity, "profiles": profiles_output}


def _stairs_results(payload: Mapping[str, Any], path: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
  config = _mapping(payload.get("config", {}), f"{path}.config")
  if "results" in payload:
    results = _sequence(payload["results"], f"{path}.results")
    return [(config, _mapping(result, f"{path}.results[{index}]")) for index, result in enumerate(results)]
  return [(config, payload)]


def analyze_stairs(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  contract = {
    "seeds": list(STAIRS_SEEDS),
    "profile": "randomized",
    "terrain_suite": "continuous",
    "directions": list(STAIRS_DIRECTIONS),
    "level": 9,
    "target_speed": 0.5,
    "steps": 2400,
    "route": "continuous_straight_approach_feature_exit",
  }
  if not payloads:
    return {
      "contract": contract,
      "status": "not_provided",
      "training_authorized": False,
      "reason": "stairs JSON files were not supplied",
    }
  records: dict[tuple[int, str], dict[str, Any]] = {}
  identities: set[tuple[str, str]] = set()
  for source_index, payload in enumerate(payloads):
    _finite_recursive(payload, f"stairs[{source_index}]")
    for config, result in _stairs_results(payload, f"stairs[{source_index}]"):
      seed = _integer(_required(result, "seed", "stairs.result"), "stairs.result.seed")
      if seed not in STAIRS_SEEDS:
        _fail("stairs.result.seed", f"must be one of {STAIRS_SEEDS}")
      if result.get("profile") != "randomized":
        _fail("stairs.result.profile", "must be randomized")
      if result.get("terrain_suite") != "continuous":
        _fail("stairs.result.terrain_suite", "must be continuous")
      if result.get("mode") != "line_follow":
        _fail("stairs.result.mode", "must be line_follow")
      if _integer(_required(result, "steps", "stairs.result"), "stairs.result.steps", minimum=1) != 2400:
        _fail("stairs.result.steps", "must be 2400")
      target_speed = config.get("target_speed", result.get("target_speed"))
      if target_speed is None or not math.isclose(_number(target_speed, "stairs.config.target_speed"), 0.5, abs_tol=1.0e-9):
        _fail("stairs.config.target_speed", "must be 0.5 m/s")
      required_config = {
        "levels": [9],
        "transition_cases": list(STAIRS_DIRECTIONS),
        "cross_track_offsets": [0.0],
        "yaw_offsets": [0.0],
        "profile": "randomized",
        "terrain_suite": "continuous",
        "mode": "line_follow",
        "steps": 2400,
        "seed": seed,
      }
      for name, expected_value in required_config.items():
        actual_value = config.get(name)
        if isinstance(expected_value, list):
          if actual_value is None or list(_sequence(actual_value, f"stairs.config.{name}")) != expected_value:
            _fail(f"stairs.config.{name}", f"must equal {expected_value}")
        elif actual_value != expected_value:
          _fail(f"stairs.config.{name}", f"must equal {expected_value!r}")
      for name in ("route_heading", "start_forward_offset"):
        if not math.isclose(
          _number(_required(config, name, "stairs.config"), f"stairs.config.{name}"),
          0.0,
          abs_tol=1.0e-9,
        ):
          _fail(f"stairs.config.{name}", "must be zero")
      checkpoint = str(_required(result, "checkpoint", "stairs.result"))
      task_id = str(_required(result, "task_id", "stairs.result"))
      identities.add((checkpoint, task_id))
      scenarios = _sequence(_required(result, "scenarios", "stairs.result"), "stairs.result.scenarios")
      selected = [row for row in scenarios if isinstance(row, Mapping) and row.get("transition_case") in STAIRS_DIRECTIONS]
      if len(selected) != 2:
        _fail("stairs.result.scenarios", "must contain exactly one stairs_up and one stairs_down scenario")
      for row in selected:
        direction = str(row["transition_case"])
        key = (seed, direction)
        if key in records:
          _fail("stairs.result.scenarios", f"duplicate seed/direction {key}")
        if _integer(_required(row, "level", "stairs.scenario"), "stairs.scenario.level") != 9:
          _fail("stairs.scenario.level", "must be 9")
        if row.get("feature") != "stairs" or row.get("direction_semantics") != direction:
          _fail("stairs.scenario", "stairs direction semantics mismatch")
        completed = _boolean(_required(row, "completed", "stairs.scenario"), "stairs.scenario.completed")
        failed = _boolean(_required(row, "failed", "stairs.scenario"), "stairs.scenario.failed")
        if completed == failed:
          _fail("stairs.scenario", "exactly one of completed and failed must be true")
        reason = _required(row, "first_failure_reason", "stairs.scenario")
        if failed and (not isinstance(reason, str) or not reason):
          _fail("stairs.scenario.first_failure_reason", "failed scenario requires reason")
        if completed and reason is not None:
          _fail("stairs.scenario.first_failure_reason", "completed scenario requires null reason")
        terminations = _mapping(_required(row, "termination_counts", "stairs.scenario"), "stairs.scenario.termination_counts")
        termination_counts = {
          str(name): _number(value, f"stairs.scenario.termination_counts.{name}", minimum=0.0)
          for name, value in terminations.items()
        }
        calf = failed and (
          reason == "illegal_calf_contact"
          or termination_counts.get("illegal_calf_contact", 0.0) > 0.0
        )
        records[key] = {
          "seed": seed,
          "direction": direction,
          "completed": completed,
          "failed": failed,
          "first_failure_reason": reason,
          "calf_termination": calf,
          "progress_ratio": _number(_required(row, "progress_ratio", "stairs.scenario"), "stairs.scenario.progress_ratio"),
          "reset_count": _integer(_required(row, "reset_count", "stairs.scenario"), "stairs.scenario.reset_count"),
          "termination_counts": termination_counts,
        }
  if len(identities) != 1:
    _fail("stairs", "checkpoint and task_id must match across all seeds")
  expected = {(seed, direction) for seed in STAIRS_SEEDS for direction in STAIRS_DIRECTIONS}
  if set(records) != expected:
    _fail("stairs", f"incomplete seed/direction matrix; missing {sorted(expected - set(records))}")

  by_direction = {}
  all_failure_reasons = set()
  failure_directions = set()
  for direction in STAIRS_DIRECTIONS:
    rows = [records[(seed, direction)] for seed in STAIRS_SEEDS]
    calf_count = sum(row["calf_termination"] for row in rows)
    failure_reasons = [row["first_failure_reason"] for row in rows if row["failed"]]
    all_failure_reasons.update(failure_reasons)
    if failure_reasons:
      failure_directions.add(direction)
    if calf_count >= 2:
      classification = "stable_calf_termination_risk"
    elif calf_count == 1:
      classification = "low_confidence_or_incidental_calf_risk"
    elif failure_reasons:
      classification = "non_calf_failures_require_diagnosis"
    else:
      classification = "passed_all_seeds"
    by_direction[direction] = {
      "classification": classification,
      "calf_termination_seed_count": calf_count,
      "failure_seed_count": len(failure_reasons),
      "failure_reasons": dict(sorted(Counter(failure_reasons).items())),
      "seeds": rows,
    }
  stable = any(value["calf_termination_seed_count"] >= 2 for value in by_direction.values())
  low = any(value["calf_termination_seed_count"] == 1 for value in by_direction.values())
  heterogeneous = len(all_failure_reasons) > 1 or len(failure_directions) > 1
  overall = (
    "stable_level9_stairs_calf_risk" if stable else
    "heterogeneous_failures_require_more_diagnosis" if heterogeneous else
    "low_confidence_level9_stairs_risk" if low else
    "no_stable_level9_stairs_calf_risk_observed"
  )
  checkpoint, task_id = next(iter(identities))
  return {
    "contract": contract,
    "status": "complete",
    "identity": {"checkpoint": checkpoint, "task_id": task_id},
    "by_direction": by_direction,
    "overall_classification": overall,
    "training_authorized": False,
    "reason": "stairs risk is recorded separately and cannot be combined with a slope training variable",
  }


def build_report(
  matched_payload: Mapping[str, Any],
  stairs_payloads: Sequence[Mapping[str, Any]],
  thresholds: AttributionThresholds = AttributionThresholds(),
) -> dict[str, Any]:
  matched = analyze_matched(matched_payload, thresholds)
  stairs = analyze_stairs(stairs_payloads)
  if stairs.get("status") == "complete":
    for name in ("checkpoint", "task_id"):
      if stairs["identity"][name] != matched["identity"][name]:
        _fail(f"level9_stairs.identity.{name}", "must match high-slope evaluation")
  result = {
    "schema_version": 1,
    "analysis_type": "offline_high_slope_matched_attribution",
    "thresholds": asdict(thresholds),
    "matched_high_slope": matched,
    "level9_stairs": stairs,
    "training_gate": "NO-GO",
    "training_started": False,
  }
  _finite_recursive(result)
  return result


def _load_json(path: Path) -> Mapping[str, Any]:
  with path.open(encoding="utf-8") as stream:
    value = json.load(stream)
  return _mapping(value, str(path))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--matched-json", type=Path, required=True)
  parser.add_argument("--stairs-json", type=Path, action="append", default=[])
  parser.add_argument("--output", type=Path)
  parser.add_argument("--pass-completion-rate", type=float, default=0.80)
  parser.add_argument("--failure-completion-rate", type=float, default=0.50)
  parser.add_argument("--forward-under-response-gain", type=float, default=0.80)
  parser.add_argument("--similar-forward-gain-absolute-spread", type=float, default=0.15)
  parser.add_argument("--controller-saturation-fraction", type=float, default=0.10)
  parser.add_argument("--near-end-progress-ratio", type=float, default=0.90)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  thresholds = AttributionThresholds(
    pass_completion_rate=args.pass_completion_rate,
    failure_completion_rate=args.failure_completion_rate,
    forward_under_response_gain=args.forward_under_response_gain,
    similar_forward_gain_absolute_spread=args.similar_forward_gain_absolute_spread,
    controller_saturation_fraction=args.controller_saturation_fraction,
    near_end_progress_ratio=args.near_end_progress_ratio,
  )
  report = build_report(
    _load_json(args.matched_json),
    [_load_json(path) for path in args.stairs_json],
    thresholds,
  )
  rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
  print(rendered)


if __name__ == "__main__":
  main()
