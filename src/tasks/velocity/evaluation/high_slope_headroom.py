"""Pure contracts for evaluation-only high-slope controller headroom A/B.

The A/B intervention intentionally changes only the closed-loop controller's
body-frame lateral-speed and yaw-rate clamps.  Policy weights, terrain,
randomization, route geometry, controller gains, and acceptance tolerances are
matched across scales.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .high_slope_matched import (
  ROUTE_KINDS,
  HighSlopeMatchedScenario,
  build_matched_scenarios,
)
from .terrain_rollout_metrics import assert_recursive_json_finite


HEADROOM_SCALES = (1.0, 1.5)
TARGET_STRATA = (("slope_up", 0), ("slope_down", 1))
SATURATION_AXES = ("vx", "vy", "wz")
ONLY_CHANGED_CONTROLLER_FIELDS = ("max_lateral_speed", "max_yaw_rate")
IDENTITY_FLOAT_ABS_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class ControllerLimits:
  """Effective controller clamps for one evaluation arm."""

  vx_limit: None
  max_lateral_speed: float
  max_yaw_rate: float

  def as_dict(self) -> dict[str, float | None]:
    return asdict(self)


def _positive_finite(name: str, value: float) -> float:
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive")
  return value


def scale_controller_limits(
  base_lateral_speed: float,
  base_yaw_rate: float,
  scale: float,
) -> ControllerLimits:
  """Scale only the two controller headroom clamps used by the A/B."""
  lateral = _positive_finite("base_lateral_speed", base_lateral_speed)
  yaw = _positive_finite("base_yaw_rate", base_yaw_rate)
  factor = _positive_finite("scale", scale)
  return ControllerLimits(
    vx_limit=None,
    max_lateral_speed=lateral * factor,
    max_yaw_rate=yaw * factor,
  )


def per_axis_saturation(
  commands: torch.Tensor,
  *,
  max_lateral_speed: float,
  max_yaw_rate: float,
  atol: float = 1.0e-6,
) -> torch.Tensor:
  """Return ``(..., 3)`` saturation flags for body-frame ``vx, vy, wz``.

  ``vx`` is always false because this experiment does not impose or change a
  forward-speed controller clamp.  The comparison matches the existing
  matched evaluator: a value at the limit within ``atol`` is saturated.
  """
  if not isinstance(commands, torch.Tensor):
    raise TypeError("commands must be a torch.Tensor")
  if commands.ndim != 2 or commands.shape[1] != 3:
    raise ValueError("commands must have shape (N, 3) for vx, vy, wz")
  if not torch.isfinite(commands).all():
    raise ValueError("commands must be finite")
  lateral = _positive_finite("max_lateral_speed", max_lateral_speed)
  yaw = _positive_finite("max_yaw_rate", max_yaw_rate)
  if not math.isfinite(atol) or atol < 0.0:
    raise ValueError("atol must be finite and nonnegative")
  flags = torch.zeros_like(commands, dtype=torch.bool)
  flags[..., 1] = commands[..., 1].abs() >= lateral - atol
  flags[..., 2] = commands[..., 2].abs() >= yaw - atol
  return flags


def summarize_axis_saturation(
  saturation: torch.Tensor,
  sample_mask: torch.Tensor,
) -> dict[str, dict[str, int | float]]:
  """Summarize per-axis flags over an explicit sample mask."""
  if not isinstance(saturation, torch.Tensor) or saturation.dtype != torch.bool:
    raise TypeError("saturation must be a bool torch.Tensor")
  if saturation.ndim != 2 or saturation.shape[1] != 3:
    raise ValueError("saturation must have shape (N, 3)")
  if not isinstance(sample_mask, torch.Tensor) or sample_mask.dtype != torch.bool:
    raise TypeError("sample_mask must be a bool torch.Tensor")
  if sample_mask.shape != (saturation.shape[0],):
    raise ValueError("sample_mask must have shape (N,)")
  denominator = int(sample_mask.sum())
  result: dict[str, dict[str, int | float]] = {}
  for axis, name in enumerate(SATURATION_AXES):
    count = int((saturation[..., axis] & sample_mask).sum())
    result[name] = {
      "count": count,
      "rate": count / max(denominator, 1),
      "denominator": denominator,
    }
  return result


def build_headroom_scenarios(
  *,
  radii: Sequence[float] = (2.5,),
  speeds: Sequence[float] = (0.5,),
  turn_signs: Sequence[int] = (1, -1),
  repeats: int = 1,
) -> list[HighSlopeMatchedScenario]:
  """Build only slope-up/high and slope-down/extreme matched slots."""
  scenarios: list[HighSlopeMatchedScenario] = []
  for slope_direction, level in TARGET_STRATA:
    block = build_matched_scenarios(
      slope_directions=(slope_direction,),
      levels=(level,),
      radii=radii,
      speeds=speeds,
      turn_signs=turn_signs,
      repeats=repeats,
    )
    for item in block:
      scenarios.append(HighSlopeMatchedScenario(
        matched_slot=len(scenarios),
        slope_direction=item.slope_direction,
        level=item.level,
        difficulty_label=item.difficulty_label,
        difficulty=item.difficulty,
        radius=item.radius,
        speed=item.speed,
        turn_sign=item.turn_sign,
        repeat=item.repeat,
      ))
  return scenarios


def _scale_key(scale: float) -> str:
  return f"{float(scale):.1f}"


def _scenario_identity(row: Mapping[str, Any]) -> dict[str, Any]:
  keys = (
    "matched_slot", "slope_direction", "level", "difficulty_label",
    "difficulty", "radius", "speed", "turn_sign", "repeat", "route_kind",
    "route_length", "route_length_definition", "route_start_xy",
    "route_endpoint_xy", "route_endpoint_heading", "terrain_origin_xyz",
    "terrain_patch_origin_xyz", "terrain_patch_size", "terrain_type_index",
    "effective_terrain_parameters", "geometry", "initial_root_clearance",
  )
  missing = [key for key in keys if key not in row]
  if missing:
    raise ValueError(f"scenario identity is missing {missing}")
  return {key: row[key] for key in keys}


def _compare_identity(
  expected: Any,
  actual: Any,
  *,
  path: str,
  float_abs_tolerance: float = IDENTITY_FLOAT_ABS_TOLERANCE,
) -> None:
  """Recursively compare static identity and report its first exact path.

  Integer/string/boolean identity is exact.  Floating values produced by GPU
  placement and geometry tensors use a small absolute tolerance; this avoids
  rejecting matched reconstructions solely due to float32 round-off while
  still rejecting a meaningful initial-state or terrain mismatch.
  """
  if isinstance(expected, Mapping):
    if not isinstance(actual, Mapping):
      raise ValueError(f"A/B identity mismatch at {path}: mapping type changed")
    expected_keys = tuple(expected)
    actual_keys = tuple(actual)
    if set(expected_keys) != set(actual_keys):
      missing = sorted(set(expected_keys) - set(actual_keys))
      extra = sorted(set(actual_keys) - set(expected_keys))
      raise ValueError(
        f"A/B identity mismatch at {path}: missing={missing}, extra={extra}"
      )
    for key in expected_keys:
      _compare_identity(
        expected[key], actual[key], path=f"{path}.{key}",
        float_abs_tolerance=float_abs_tolerance,
      )
    return
  if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
      raise ValueError(f"A/B identity mismatch at {path}: sequence type changed")
    if len(expected) != len(actual):
      raise ValueError(
        f"A/B identity mismatch at {path}: length {len(expected)} != {len(actual)}"
      )
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
      _compare_identity(
        left, right, path=f"{path}[{index}]",
        float_abs_tolerance=float_abs_tolerance,
      )
    return
  if isinstance(expected, bool) or isinstance(actual, bool):
    if type(expected) is not type(actual) or expected != actual:
      raise ValueError(
        f"A/B identity mismatch at {path}: {expected!r} != {actual!r}"
      )
    return
  if isinstance(expected, float) or isinstance(actual, float):
    if (
      isinstance(expected, (int, float))
      and not isinstance(expected, bool)
      and isinstance(actual, (int, float))
      and not isinstance(actual, bool)
      and math.isfinite(float(expected))
      and math.isfinite(float(actual))
      and math.isclose(
        float(expected), float(actual), rel_tol=0.0,
        abs_tol=float_abs_tolerance,
      )
    ):
      return
    raise ValueError(
      f"A/B identity mismatch at {path}: {expected!r} != {actual!r} "
      f"(abs_tol={float_abs_tolerance})"
    )
  if type(expected) is not type(actual) or expected != actual:
    raise ValueError(
      f"A/B identity mismatch at {path}: {expected!r} != {actual!r}"
    )


def _validate_axis_summary(row: Mapping[str, Any]) -> None:
  completed = row.get("completed")
  failed = row.get("failed")
  reason = row.get("first_failure_reason")
  if completed is True:
    if failed is not False or reason is not None:
      raise ValueError("completed scenario must have null first_failure_reason")
  elif failed is True:
    if (
      not isinstance(reason, str)
      or not reason.strip()
      or reason.strip().lower() in {"none", "null", "unknown", "n/a"}
    ):
      raise ValueError("failed scenario must have a real first_failure_reason")
  else:
    raise ValueError("scenario must be exactly completed or failed")
  summary = row.get("controller_saturation_by_axis")
  if not isinstance(summary, Mapping) or tuple(summary) != SATURATION_AXES:
    raise ValueError("controller_saturation_by_axis must preserve vx/vy/wz order")
  expected_denominator = row["steps_sampled"]
  if (
    isinstance(expected_denominator, bool)
    or not isinstance(expected_denominator, int)
    or expected_denominator <= 0
  ):
    raise ValueError("steps_sampled must be a positive integer")
  for axis in SATURATION_AXES:
    item = summary[axis]
    if not isinstance(item, Mapping):
      raise ValueError(f"{axis} saturation summary must be a mapping")
    count = item["count"]
    denominator = item["denominator"]
    rate = item["rate"]
    if isinstance(count, bool) or not isinstance(count, int):
      raise ValueError(f"{axis} saturation count must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
      raise ValueError(f"{axis} saturation denominator must be an integer")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
      raise ValueError(f"{axis} saturation rate must be numeric")
    rate = float(rate)
    if not math.isfinite(rate):
      raise ValueError(f"{axis} saturation rate must be finite")
    if denominator != expected_denominator:
      raise ValueError(f"{axis} saturation denominator differs from steps_sampled")
    if count < 0 or count > denominator:
      raise ValueError(f"{axis} saturation count is out of range")
    if not math.isclose(rate, count / max(denominator, 1), abs_tol=1.0e-12):
      raise ValueError(f"{axis} saturation rate is inconsistent")
  if summary["vx"]["count"] != 0:
    raise ValueError("vx saturation must be zero because vx is not clamped")


def validate_headroom_pair(
  scale_results: Mapping[str, Mapping[str, Any]],
  *,
  base_lateral_speed: float,
  base_yaw_rate: float,
  scales: Sequence[float] = HEADROOM_SCALES,
) -> None:
  """Validate strict A/B identity while allowing only the two headroom limits."""
  expected_keys = tuple(_scale_key(scale) for scale in scales)
  if tuple(scale_results) != expected_keys:
    raise ValueError(f"scale results must preserve order {expected_keys}")
  reference: Mapping[str, Any] | None = None
  environment_ids: set[str] = set()
  for scale, key in zip(scales, expected_keys, strict=True):
    arm = scale_results[key]
    if float(arm.get("controller_scale", math.nan)) != float(scale):
      raise ValueError(f"controller_scale mismatch for arm {key}")
    effective = scale_controller_limits(base_lateral_speed, base_yaw_rate, scale)
    if arm.get("effective_controller_limits") != effective.as_dict():
      raise ValueError(f"effective controller limits mismatch for arm {key}")
    routes = arm.get("route_results")
    if not isinstance(routes, Mapping) or tuple(routes) != ROUTE_KINDS:
      raise ValueError(f"arm {key} route results must preserve {ROUTE_KINDS}")
    arm_identity: dict[str, Any] = {}
    for route_kind in ROUTE_KINDS:
      result = routes[route_kind]
      env_id = result.get("fresh_environment_identity")
      if not isinstance(env_id, str) or not env_id:
        raise ValueError("fresh_environment_identity must be nonempty")
      if env_id in environment_ids:
        raise ValueError("fresh environment identity was reused")
      environment_ids.add(env_id)
      invariant = dict(result["route_kind_invariants"])
      for field in (
        "checkpoint", "task_id", "seed", "profile", "num_envs", "steps",
        "settle_steps", "controller_limits",
      ):
        if field not in invariant:
          raise ValueError(f"route invariant is missing {field}")
      if invariant["profile"] != result["profile_settings"].get("profile", invariant["profile"]):
        # Some existing profile settings use a different descriptive key; if
        # present, an explicit profile identity must nevertheless agree.
        raise ValueError("route/profile identity mismatch")
      if not isinstance(invariant["seed"], int) or isinstance(invariant["seed"], bool):
        raise ValueError("route invariant seed must be an integer")
      if not isinstance(invariant["steps"], int) or invariant["steps"] <= 0:
        raise ValueError("route invariant steps must be a positive integer")
      limits = dict(invariant.pop("controller_limits"))
      if effective.vx_limit is not None:
        raise ValueError("vx_limit must remain None")
      for field, value in (
        ("max_lateral_speed", effective.max_lateral_speed),
        ("max_yaw_rate", effective.max_yaw_rate),
      ):
        if not math.isclose(float(limits.pop(field)), value, abs_tol=1.0e-12):
          raise ValueError(f"{field} is not the requested scaled limit")
      invariant["controller_limits"] = limits
      arm_identity[route_kind] = {
        "route_kind_invariants": invariant,
        "profile_settings": result["profile_settings"],
        "scenarios": [
          _scenario_identity(row) for row in result["scenarios"]
        ],
      }
      for row in result["scenarios"]:
        _validate_axis_summary(row)
    if reference is None:
      reference = arm_identity
    else:
      for route_kind in ROUTE_KINDS:
        _compare_identity(
          reference[route_kind]["route_kind_invariants"],
          arm_identity[route_kind]["route_kind_invariants"],
          path=f"{key}.{route_kind}.route_kind_invariants",
        )
        _compare_identity(
          reference[route_kind]["profile_settings"],
          arm_identity[route_kind]["profile_settings"],
          path=f"{key}.{route_kind}.profile_settings",
        )
        expected_scenarios = reference[route_kind]["scenarios"]
        actual_scenarios = arm_identity[route_kind]["scenarios"]
        if len(expected_scenarios) != len(actual_scenarios):
          raise ValueError(
            f"A/B identity mismatch at {key}.{route_kind}.scenarios: "
            f"length {len(expected_scenarios)} != {len(actual_scenarios)}"
          )
        for index, (expected, actual) in enumerate(zip(
          expected_scenarios, actual_scenarios, strict=True
        )):
          _compare_identity(
            expected,
            actual,
            path=f"{key}.{route_kind}.scenarios[{index}]",
          )


def validate_headroom_result(payload: Mapping[str, Any]) -> None:
  """Validate the top-level formal headroom result contract."""
  assert_recursive_json_finite(payload)
  if payload.get("evaluation_suite") != "high_slope_controller_headroom_ab":
    raise ValueError("unexpected evaluation_suite")
  invariant = payload.get("ab_invariants")
  if not isinstance(invariant, Mapping):
    raise ValueError("ab_invariants are missing")
  if tuple(invariant.get("only_changed_controller_fields", ())) != ONLY_CHANGED_CONTROLLER_FIELDS:
    raise ValueError("A/B intervention fields changed")
  scales = tuple(float(x) for x in invariant.get("controller_scales", ()))
  if scales != HEADROOM_SCALES:
    raise ValueError(f"controller scales must be {HEADROOM_SCALES}")
  base = invariant.get("base_controller_limits")
  if not isinstance(base, Mapping):
    raise ValueError("base_controller_limits are missing")
  profiles = payload.get("profiles")
  if not isinstance(profiles, Mapping) or not profiles:
    raise ValueError("profiles are missing")
  for profile_name, profile in profiles.items():
    if profile.get("profile") != profile_name:
      raise ValueError("profile identity mismatch")
    validate_headroom_pair(
      profile["scale_results"],
      base_lateral_speed=float(base["max_lateral_speed"]),
      base_yaw_rate=float(base["max_yaw_rate"]),
      scales=scales,
    )


__all__ = [
  "ControllerLimits", "HEADROOM_SCALES", "ONLY_CHANGED_CONTROLLER_FIELDS",
  "IDENTITY_FLOAT_ABS_TOLERANCE", "SATURATION_AXES", "TARGET_STRATA",
  "build_headroom_scenarios",
  "per_axis_saturation", "scale_controller_limits",
  "summarize_axis_saturation", "validate_headroom_pair",
  "validate_headroom_result",
]
