"""Pure contracts and geometry for matched routes on high pyramid slopes.

The helpers in this module are evaluation-only.  In particular, they do not
register a task or mutate any training configuration.  The canonical route
starts at the centre of an 18 m square pyramid primitive and initially points
along local +x.  Straight, 120-degree arc, and two-opposite-60-degree-arc S
routes all have length ``2*pi*radius/3``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from .curved_routes import make_arc_route, make_s_route
from .matched_route_metrics import matched_route_length
from .terrain_curved_routes import (
  DEFAULT_CORRIDOR_HALF_WIDTH,
  HEIGHT_SCAN_HALF_EXTENTS,
  PATCH_SIZE,
  ROUTE_START_LOCAL,
)


ROUTE_KINDS = ("straight", "arc", "s_curve")
SLOPE_DIRECTIONS = ("slope_up", "slope_down")
DIFFICULTY_LEVELS = (0, 1)
DIFFICULTY_LABELS = ("high", "extreme")
DIFFICULTIES = (0.8, 1.0)
PROFILE_NAMES = ("clean", "randomized")
ROUTE_LENGTH_DEFINITION = "2*pi*radius/3"
ROUTE_START_HEADING = 0.0
FOOTPRINT_SAMPLES = 4097


def difficulty_for_level(level: int) -> tuple[str, float]:
  if isinstance(level, bool) or not isinstance(level, int):
    raise TypeError("high-slope level must be an integer")
  if level not in DIFFICULTY_LEVELS:
    raise ValueError("high-slope level must be 0 (high) or 1 (extreme)")
  return DIFFICULTY_LABELS[level], DIFFICULTIES[level]


def effective_slope_parameters(
  slope_direction: str, level: int
) -> dict[str, object]:
  """Return the exact V7 pyramid primitive settings used by the evaluator."""
  if slope_direction not in SLOPE_DIRECTIONS:
    raise ValueError(f"unknown slope direction: {slope_direction!r}")
  label, difficulty = difficulty_for_level(level)
  inverted = slope_direction == "slope_up"
  return {
    "terrain_kind": slope_direction,
    "difficulty_label": label,
    "requested_difficulty": difficulty,
    "primitive": "HfPyramidSlopedTerrainCfg",
    "slope_gradient": (1.0 if inverted else -1.0) * difficulty * 0.4,
    "slope_magnitude": difficulty * 0.4,
    "inverted": inverted,
    "platform_width": 2.0,
    "primitive_border_width": 0.25,
    "difficulty_affects_geometry": True,
    "route_direction_semantics": (
      "centre_low_to_outward_high" if inverted
      else "centre_high_to_outward_low"
    ),
  }


@dataclass(frozen=True)
class HighSlopeMatchedScenario:
  matched_slot: int
  slope_direction: str
  level: int
  difficulty_label: str
  difficulty: float
  radius: float
  speed: float
  turn_sign: int
  repeat: int

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)


def build_matched_scenarios(
  *,
  slope_directions: Sequence[str] = SLOPE_DIRECTIONS,
  levels: Sequence[int] = DIFFICULTY_LEVELS,
  radii: Sequence[float] = (2.5, 4.0),
  speeds: Sequence[float] = (0.3, 0.5),
  turn_signs: Sequence[int] = (1, -1),
  repeats: int = 1,
) -> list[HighSlopeMatchedScenario]:
  """Build the stable matched-slot order shared by all three route kinds."""
  if repeats <= 0:
    raise ValueError("repeats must be positive")
  if not slope_directions or any(x not in SLOPE_DIRECTIONS for x in slope_directions):
    raise ValueError(f"slope_directions must contain only {SLOPE_DIRECTIONS}")
  if not levels:
    raise ValueError("levels must not be empty")
  if not radii or any(not math.isfinite(x) or x <= 0.0 for x in radii):
    raise ValueError("radii must be finite and positive")
  if not speeds or any(not math.isfinite(x) or x <= 0.0 for x in speeds):
    raise ValueError("speeds must be finite and positive")
  if not turn_signs or any(x not in (-1, 1) for x in turn_signs):
    raise ValueError("turn_signs must contain only -1 and +1")
  scenarios: list[HighSlopeMatchedScenario] = []
  for slope_direction in slope_directions:
    for level in levels:
      label, difficulty = difficulty_for_level(level)
      for radius in radii:
        for speed in speeds:
          for turn_sign in turn_signs:
            for repeat in range(repeats):
              scenarios.append(HighSlopeMatchedScenario(
                matched_slot=len(scenarios),
                slope_direction=slope_direction,
                level=level,
                difficulty_label=label,
                difficulty=difficulty,
                radius=float(radius),
                speed=float(speed),
                turn_sign=int(turn_sign),
                repeat=repeat,
              ))
  return scenarios


@dataclass(frozen=True)
class HighSlopeRouteFootprint:
  route_kind: str
  radius: float
  turn_sign: int
  route_length: float
  patch_size: tuple[float, float]
  route_start_local: tuple[float, float]
  route_start_heading: float
  corridor_half_width: float
  scan_half_extents: tuple[float, float]
  centerline_bounds: tuple[tuple[float, float], tuple[float, float]]
  corridor_bounds: tuple[tuple[float, float], tuple[float, float]]
  scan_footprint_bounds: tuple[tuple[float, float], tuple[float, float]]
  centreline_inside_patch: bool
  corridor_inside_patch: bool
  scan_footprint_inside_patch: bool
  centerline_boundary_margin: float
  corridor_boundary_margin: float
  scan_boundary_margin: float


def _route_pose(
  route_kind: str,
  radius: float,
  turn_sign: int,
  progress: torch.Tensor,
  *,
  route_start_local: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
  start = torch.tensor(route_start_local, dtype=progress.dtype, device=progress.device)
  if route_kind == "straight":
    positions = start + torch.stack((progress, torch.zeros_like(progress)), dim=-1)
    return positions, torch.zeros_like(progress)
  if route_kind == "arc":
    route = make_arc_route(
      start, ROUTE_START_HEADING, radius, turn_sign,
      angle=2.0 * math.pi / 3.0,
    )
  elif route_kind == "s_curve":
    route = make_s_route(start, ROUTE_START_HEADING, radius, turn_sign)
  else:
    raise ValueError(f"route_kind must be one of {ROUTE_KINDS}")
  return route.pose_at(progress)


def _bounds(points: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
  return (
    (float(points[..., 0].min()), float(points[..., 0].max())),
    (float(points[..., 1].min()), float(points[..., 1].max())),
  )


def _margin(
  bounds: tuple[tuple[float, float], tuple[float, float]],
  patch_size: tuple[float, float],
) -> float:
  return min(
    bounds[0][0], patch_size[0] - bounds[0][1],
    bounds[1][0], patch_size[1] - bounds[1][1],
  )


def compute_route_footprint(
  route_kind: str,
  radius: float,
  turn_sign: int,
  *,
  patch_size: tuple[float, float] = PATCH_SIZE,
  route_start_local: tuple[float, float] = ROUTE_START_LOCAL,
  corridor_half_width: float = DEFAULT_CORRIDOR_HALF_WIDTH,
  scan_half_extents: tuple[float, float] = HEIGHT_SCAN_HALF_EXTENTS,
  samples: int = FOOTPRINT_SAMPLES,
) -> HighSlopeRouteFootprint:
  """Compute the actual swept corridor and yaw-aligned scan footprint."""
  if route_kind not in ROUTE_KINDS:
    raise ValueError(f"route_kind must be one of {ROUTE_KINDS}")
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("radius must be finite and positive")
  if turn_sign not in (-1, 1):
    raise ValueError("turn_sign must be -1 or +1")
  if len(patch_size) != 2 or not all(math.isfinite(x) and x > 0 for x in patch_size):
    raise ValueError("patch_size must contain two finite positive values")
  if len(route_start_local) != 2 or not all(math.isfinite(x) for x in route_start_local):
    raise ValueError("route_start_local must contain two finite values")
  if not math.isfinite(corridor_half_width) or corridor_half_width <= 0.0:
    raise ValueError("corridor_half_width must be finite and positive")
  if len(scan_half_extents) != 2 or not all(math.isfinite(x) and x > 0 for x in scan_half_extents):
    raise ValueError("scan_half_extents must contain two finite positive values")
  if samples < 3:
    raise ValueError("samples must be at least 3")

  length = matched_route_length(radius)
  progress = torch.linspace(0.0, length, samples, dtype=torch.float64)
  positions, headings = _route_pose(
    route_kind, radius, turn_sign, progress,
    route_start_local=route_start_local,
  )
  normals = torch.stack((-torch.sin(headings), torch.cos(headings)), dim=-1)
  corridor = torch.cat((
    positions,
    positions + corridor_half_width * normals,
    positions - corridor_half_width * normals,
  ))
  scan_x, scan_y = scan_half_extents
  corners = torch.tensor(
    ((scan_x, scan_y), (scan_x, -scan_y), (-scan_x, scan_y), (-scan_x, -scan_y)),
    dtype=positions.dtype,
  )
  cos_h = torch.cos(headings).unsqueeze(-1)
  sin_h = torch.sin(headings).unsqueeze(-1)
  rotated = torch.stack((
    cos_h * corners[:, 0] - sin_h * corners[:, 1],
    sin_h * corners[:, 0] + cos_h * corners[:, 1],
  ), dim=-1)
  scan = positions.unsqueeze(1) + rotated
  centerline_bounds = _bounds(positions)
  corridor_bounds = _bounds(corridor)
  scan_bounds = _bounds(scan)
  center_margin = _margin(centerline_bounds, patch_size)
  corridor_margin = _margin(corridor_bounds, patch_size)
  scan_margin = _margin(scan_bounds, patch_size)
  tolerance = 1.0e-9
  return HighSlopeRouteFootprint(
    route_kind=route_kind,
    radius=radius,
    turn_sign=turn_sign,
    route_length=length,
    patch_size=patch_size,
    route_start_local=route_start_local,
    route_start_heading=ROUTE_START_HEADING,
    corridor_half_width=corridor_half_width,
    scan_half_extents=scan_half_extents,
    centerline_bounds=centerline_bounds,
    corridor_bounds=corridor_bounds,
    scan_footprint_bounds=scan_bounds,
    centreline_inside_patch=center_margin >= -tolerance,
    corridor_inside_patch=corridor_margin >= -tolerance,
    scan_footprint_inside_patch=scan_margin >= -tolerance,
    centerline_boundary_margin=center_margin,
    corridor_boundary_margin=corridor_margin,
    scan_boundary_margin=scan_margin,
  )


def validate_route_footprint(
  route_kind: str, radius: float, turn_sign: int, **kwargs: object
) -> HighSlopeRouteFootprint:
  footprint = compute_route_footprint(
    route_kind, radius, turn_sign, **kwargs  # type: ignore[arg-type]
  )
  failures = []
  if not footprint.centreline_inside_patch:
    failures.append("centerline")
  if not footprint.corridor_inside_patch:
    failures.append("corridor")
  if not footprint.scan_footprint_inside_patch:
    failures.append("yaw-aligned height-scan footprint")
  if failures:
    raise ValueError(
      f"{', '.join(failures)} leaves the 18x18 evaluation patch for "
      f"{route_kind} radius={radius} turn_sign={turn_sign}; "
      f"corridor_margin={footprint.corridor_boundary_margin:.9f}, "
      f"scan_margin={footprint.scan_boundary_margin:.9f}"
    )
  return footprint


def geometry_preflight(
  radii: Iterable[float], turn_signs: Iterable[int]
) -> dict[str, object]:
  """Return truthful runnable coverage without hiding invalid combinations."""
  combinations = []
  all_valid = True
  for route_kind in ROUTE_KINDS:
    for radius in radii:
      for turn_sign in turn_signs:
        footprint = compute_route_footprint(route_kind, radius, turn_sign)
        valid = (
          footprint.centreline_inside_patch
          and footprint.corridor_inside_patch
          and footprint.scan_footprint_inside_patch
        )
        all_valid &= valid
        combinations.append({**asdict(footprint), "valid": valid})
  return {
    "all_requested_combinations_valid": all_valid,
    "combinations": combinations,
  }


def minimum_horizon_steps(
  radius: float, speed: float, control_dt: float, settle_steps: int
) -> int:
  """Ideal-command lower bound used to reject trivially short horizons."""
  for name, value in (("radius", radius), ("speed", speed), ("control_dt", control_dt)):
    if not math.isfinite(value) or value <= 0.0:
      raise ValueError(f"{name} must be finite and positive")
  if settle_steps < 0:
    raise ValueError("settle_steps must be nonnegative")
  return math.ceil(matched_route_length(radius) / (speed * control_dt)) + settle_steps


def validate_horizon(
  steps: int,
  *,
  radii: Iterable[float],
  speeds: Iterable[float],
  control_dt: float,
  settle_steps: int,
) -> int:
  required = max(
    minimum_horizon_steps(radius, speed, control_dt, settle_steps)
    for radius in radii for speed in speeds
  )
  if steps < required:
    raise ValueError(
      f"steps={steps} is shorter than ideal matched route plus settle "
      f"lower bound {required}"
    )
  return required


def validate_matched_result_invariants(
  route_results: Mapping[str, Mapping[str, Any]],
) -> None:
  """Fail if route subprocess-like results do not preserve matched slots."""
  if tuple(route_results) != ROUTE_KINDS:
    raise ValueError(f"route results must preserve order {ROUTE_KINDS}")
  slots: list[list[int]] = []
  invariants: list[object] = []
  for kind in ROUTE_KINDS:
    result = route_results[kind]
    scenario_values = result.get("scenarios")
    if not isinstance(scenario_values, Sequence):
      raise ValueError(f"{kind} scenarios are missing")
    slots.append([int(item["matched_slot"]) for item in scenario_values])
    invariants.append(result.get("route_kind_invariants"))
  if any(value != slots[0] for value in slots[1:]):
    raise ValueError("matched_slot order differs across route kinds")
  if any(value != invariants[0] for value in invariants[1:]):
    raise ValueError("route-kind invariant settings differ")


__all__ = [
  "DIFFICULTIES", "DIFFICULTY_LABELS", "DIFFICULTY_LEVELS",
  "FOOTPRINT_SAMPLES", "HighSlopeMatchedScenario", "HighSlopeRouteFootprint",
  "PROFILE_NAMES", "ROUTE_KINDS", "ROUTE_LENGTH_DEFINITION",
  "ROUTE_START_HEADING", "SLOPE_DIRECTIONS", "build_matched_scenarios",
  "compute_route_footprint", "difficulty_for_level", "effective_slope_parameters",
  "geometry_preflight", "minimum_horizon_steps", "validate_horizon",
  "validate_matched_result_invariants", "validate_route_footprint",
]
