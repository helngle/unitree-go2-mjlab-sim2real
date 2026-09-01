"""Evaluation-only terrain and footprint helpers for curved Go2 routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfDiscreteObstaclesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput

from .curved_routes import ArcRoute, SRoute, make_arc_route, make_s_route


PATCH_SIZE = (18.0, 18.0)
ROUTE_START_LOCAL = (PATCH_SIZE[0] / 2.0, PATCH_SIZE[1] / 2.0)
HEIGHT_SCAN_SIZE = (1.6, 1.0)
HEIGHT_SCAN_HALF_EXTENTS = (HEIGHT_SCAN_SIZE[0] / 2.0, HEIGHT_SCAN_SIZE[1] / 2.0)
DEFAULT_CORRIDOR_HALF_WIDTH = 0.4
LOW_DIFFICULTY = 0.25
MEDIUM_DIFFICULTY = 0.55
DIFFICULTY_LABELS = ("low", "medium")
BASE_PATCH_SIZE = (8.0, 8.0)
BASE_OBSTACLE_COUNT = 32
SCALED_OBSTACLE_COUNT = round(
  BASE_OBSTACLE_COUNT
  * PATCH_SIZE[0] * PATCH_SIZE[1]
  / (BASE_PATCH_SIZE[0] * BASE_PATCH_SIZE[1])
)

TerrainCurveKind = Literal[
  "slope_up",
  "slope_down",
  "random_rough",
  "discrete_obstacle",
  "stairs_up",
  "stairs_down",
]

TERRAIN_CURVE_KINDS: tuple[TerrainCurveKind, ...] = (
  "slope_up",
  "slope_down",
  "random_rough",
  "discrete_obstacle",
)
PROPRIO_TERRAIN_CURVE_KINDS: tuple[TerrainCurveKind, ...] = (
  *TERRAIN_CURVE_KINDS,
  "stairs_up",
  "stairs_down",
)
TERRAIN_KIND_TO_TYPE = {
  kind: index for index, kind in enumerate(TERRAIN_CURVE_KINDS)
}
PROPRIO_TERRAIN_KIND_TO_TYPE = {
  kind: index for index, kind in enumerate(PROPRIO_TERRAIN_CURVE_KINDS)
}


@dataclass(frozen=True)
class RouteFootprint:
  route_kind: str
  radius: float
  turn_sign: int
  patch_size: tuple[float, float]
  route_start_local: tuple[float, float]
  corridor_half_width: float
  centerline_bounds: tuple[tuple[float, float], tuple[float, float]]
  corridor_bounds: tuple[tuple[float, float], tuple[float, float]]
  scan_footprint_bounds: tuple[tuple[float, float], tuple[float, float]]
  corridor_inside_patch: bool
  scan_footprint_inside_patch: bool
  corridor_boundary_margin: float
  scan_boundary_margin: float


def difficulty_for_level(level: int) -> tuple[str, float]:
  if level == 0:
    return "low", LOW_DIFFICULTY
  if level == 1:
    return "medium", MEDIUM_DIFFICULTY
  raise ValueError("terrain curve level must be 0 (low) or 1 (medium)")


def effective_terrain_parameters(kind: TerrainCurveKind, level: int) -> dict[str, object]:
  label, difficulty = difficulty_for_level(level)
  if kind == "slope_up":
    return {
      "difficulty_label": label,
      "difficulty": difficulty,
      "slope_gradient": difficulty * 0.4,
      "inverted": True,
      "difficulty_affects_geometry": True,
    }
  if kind == "slope_down":
    return {
      "difficulty_label": label,
      "difficulty": difficulty,
      "slope_gradient": -difficulty * 0.4,
      "inverted": False,
      "difficulty_affects_geometry": True,
    }
  if kind == "random_rough":
    return {
      "difficulty_label": label,
      "difficulty": difficulty,
      "noise_range": [0.01, 0.06],
      "noise_step": 0.01,
      "difficulty_affects_geometry": False,
      "difficulty_reason": "V7 HfRandomUniformTerrainCfg ignores difficulty",
    }
  if kind == "discrete_obstacle":
    return {
      "difficulty_label": label,
      "difficulty": difficulty,
      "obstacle_height": 0.02 + difficulty * (0.10 - 0.02),
      "obstacle_width_range": [0.30, 0.80],
      "num_obstacles": SCALED_OBSTACLE_COUNT,
      "difficulty_affects_geometry": True,
    }
  if kind in {"stairs_up", "stairs_down"}:
    return {
      "difficulty_label": label,
      "difficulty": difficulty,
      "step_height": 0.02 + difficulty * (0.12 - 0.02),
      "step_width": 0.30,
      "platform_width": 3.0,
      "inverted": kind == "stairs_up",
      "difficulty_affects_geometry": True,
      "route_direction_semantics": (
        "centre_low_to_outward_high" if kind == "stairs_up"
        else "centre_high_to_outward_low"
      ),
    }
  raise ValueError(f"unknown terrain curve kind: {kind!r}")


def _base_terrain_cfg(kind: TerrainCurveKind) -> SubTerrainCfg:
  if kind == "slope_up":
    return HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    )
  if kind == "slope_down":
    return HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=False,
    )
  if kind == "random_rough":
    return HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.01, 0.06),
      noise_step=0.01,
      border_width=0.25,
    )
  if kind == "discrete_obstacle":
    return HfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      obstacle_width_range=(0.30, 0.80),
      obstacle_height_range=(0.02, 0.10),
      num_obstacles=SCALED_OBSTACLE_COUNT,
      platform_width=2.0,
      border_width=0.25,
      origin_z_offset=0.02,
    )
  if kind == "stairs_up":
    return BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.02, 0.12),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    )
  if kind == "stairs_down":
    return BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.02, 0.12),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    )
  raise ValueError(f"unknown terrain curve kind: {kind!r}")


@dataclass(kw_only=True)
class TerrainCurveSubTerrainCfg(SubTerrainCfg):
  """Delegate to a V7 primitive at deterministic low/medium difficulty."""

  kind: TerrainCurveKind
  size: tuple[float, float] = PATCH_SIZE

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    if self.kind not in PROPRIO_TERRAIN_CURVE_KINDS:
      raise ValueError(f"unknown terrain curve kind: {self.kind!r}")
    if self.size != PATCH_SIZE:
      raise ValueError(f"terrain curve patch size must be {PATCH_SIZE}")
    fixed_difficulty = LOW_DIFFICULTY if difficulty < 0.5 else MEDIUM_DIFFICULTY
    delegate = _base_terrain_cfg(self.kind)
    delegate.size = self.size
    output = delegate.function(fixed_difficulty, spec, rng)
    expected_xy = np.asarray(ROUTE_START_LOCAL)
    if not np.allclose(output.origin[:2], expected_xy, rtol=0.0, atol=1.0e-12):
      raise RuntimeError(
        f"{self.kind} primitive origin {output.origin[:2]} is not patch center "
        f"{expected_xy}"
      )
    return output


def make_terrain_curve_generator(
  seed: int | None = 42, *, include_stairs: bool = False
) -> TerrainGeneratorCfg:
  """Build a two-level, four-type evaluation-only terrain grid."""
  kinds = PROPRIO_TERRAIN_CURVE_KINDS if include_stairs else TERRAIN_CURVE_KINDS
  return TerrainGeneratorCfg(
    seed=seed,
    size=PATCH_SIZE,
    border_width=20.0,
    num_rows=2,
    num_cols=len(kinds),
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      kind: TerrainCurveSubTerrainCfg(kind=kind, proportion=1.0)
      for kind in kinds
    },
  )


def _route(
  route_kind: str,
  radius: float,
  turn_sign: int,
  start_xy: torch.Tensor,
) -> ArcRoute | SRoute:
  if route_kind == "arc":
    return make_arc_route(start_xy, 0.0, radius, turn_sign)
  if route_kind == "s_curve":
    return make_s_route(start_xy, 0.0, radius, turn_sign)
  raise ValueError("route_kind must be 'arc' or 's_curve'")


def _bounds(points: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
  return (
    (float(points[:, 0].min()), float(points[:, 0].max())),
    (float(points[:, 1].min()), float(points[:, 1].max())),
  )


def _inside_and_margin(
  bounds: tuple[tuple[float, float], tuple[float, float]],
  patch_size: tuple[float, float],
) -> tuple[bool, float]:
  x_bounds, y_bounds = bounds
  margin = min(
    x_bounds[0],
    patch_size[0] - x_bounds[1],
    y_bounds[0],
    patch_size[1] - y_bounds[1],
  )
  return margin >= -1.0e-6, margin


def compute_route_footprint(
  route_kind: str,
  radius: float,
  turn_sign: int,
  *,
  patch_size: tuple[float, float] = PATCH_SIZE,
  route_start_local: tuple[float, float] = ROUTE_START_LOCAL,
  corridor_half_width: float = DEFAULT_CORRIDOR_HALF_WIDTH,
  samples: int = 4097,
) -> RouteFootprint:
  """Compute swept centerline, corridor, and yaw-aligned scan bounds."""
  if (
    len(patch_size) != 2
    or not all(math.isfinite(value) and value > 0.0 for value in patch_size)
  ):
    raise ValueError("patch_size must contain two finite positive values")
  if len(route_start_local) != 2 or not all(
    math.isfinite(value) for value in route_start_local
  ):
    raise ValueError("route_start_local must contain two finite values")
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("radius must be finite and positive")
  if turn_sign not in (-1, 1):
    raise ValueError("turn_sign must be -1 or +1")
  if not math.isfinite(corridor_half_width) or corridor_half_width <= 0.0:
    raise ValueError("corridor_half_width must be finite and positive")
  if samples < 3:
    raise ValueError("samples must be at least 3")

  start = torch.tensor(route_start_local, dtype=torch.float64)
  route = _route(route_kind, radius, turn_sign, start)
  progress = torch.linspace(0.0, route.length, samples, dtype=torch.float64)
  positions, headings = route.pose_at(progress)
  normals = torch.stack((-torch.sin(headings), torch.cos(headings)), dim=-1)
  corridor_points = torch.cat(
    (
      positions,
      positions + corridor_half_width * normals,
      positions - corridor_half_width * normals,
    ),
    dim=0,
  )

  half_x, half_y = HEIGHT_SCAN_HALF_EXTENTS
  local_corners = torch.tensor(
    [
      [half_x, half_y],
      [half_x, -half_y],
      [-half_x, half_y],
      [-half_x, -half_y],
    ],
    dtype=torch.float64,
  )
  cos_h = torch.cos(headings).unsqueeze(-1)
  sin_h = torch.sin(headings).unsqueeze(-1)
  corner_x = (
    cos_h * local_corners[:, 0] - sin_h * local_corners[:, 1]
  )
  corner_y = (
    sin_h * local_corners[:, 0] + cos_h * local_corners[:, 1]
  )
  scan_points = positions.unsqueeze(1) + torch.stack((corner_x, corner_y), dim=-1)
  centerline_bounds = _bounds(positions)
  corridor_bounds = _bounds(corridor_points)
  scan_bounds = _bounds(scan_points.reshape(-1, 2))
  corridor_inside, corridor_margin = _inside_and_margin(corridor_bounds, patch_size)
  scan_inside, scan_margin = _inside_and_margin(scan_bounds, patch_size)
  return RouteFootprint(
    route_kind=route_kind,
    radius=radius,
    turn_sign=turn_sign,
    patch_size=patch_size,
    route_start_local=route_start_local,
    corridor_half_width=corridor_half_width,
    centerline_bounds=centerline_bounds,
    corridor_bounds=corridor_bounds,
    scan_footprint_bounds=scan_bounds,
    corridor_inside_patch=corridor_inside,
    scan_footprint_inside_patch=scan_inside,
    corridor_boundary_margin=corridor_margin,
    scan_boundary_margin=scan_margin,
  )


def validate_route_footprint(
  route_kind: str,
  radius: float,
  turn_sign: int,
  **kwargs: object,
) -> RouteFootprint:
  footprint = compute_route_footprint(
    route_kind, radius, turn_sign, **kwargs  # type: ignore[arg-type]
  )
  if not footprint.corridor_inside_patch:
    raise ValueError(
      f"route corridor leaves patch: bounds={footprint.corridor_bounds}, "
      f"patch={footprint.patch_size}"
    )
  if not footprint.scan_footprint_inside_patch:
    raise ValueError(
      "height-scan footprint leaves patch: "
      f"bounds={footprint.scan_footprint_bounds}, patch={footprint.patch_size}"
    )
  return footprint


def slope_direction_is_compatible(
  route_kind: str,
  radius: float,
  turn_sign: int,
  *,
  samples: int = 4097,
) -> bool:
  """Whether the route moves monotonically outward from the pyramid center."""
  start = torch.tensor(ROUTE_START_LOCAL, dtype=torch.float64)
  route = _route(route_kind, radius, turn_sign, start)
  progress = torch.linspace(0.0, route.length, samples, dtype=torch.float64)
  positions, _ = route.pose_at(progress)
  radial_distance = torch.norm(positions - start, dim=-1)
  return bool(torch.all(torch.diff(radial_distance) >= -1.0e-9))


def continuous_transition_coverage() -> dict[str, object]:
  return {
    "implemented": False,
    "coverage": False,
    "reason": (
      "The existing 8x4 m approach-feature-exit patch cannot contain r=2.5/4.0 "
      "curves plus the rotated height-scan footprint. A faithful route-aligned "
      "curved transition terrain has not been implemented; patch scaling is "
      "intentionally rejected."
    ),
  }


def relocate_root_pose(
  root_pose: torch.Tensor,
  old_origins: torch.Tensor,
  new_origins: torch.Tensor,
) -> tuple[torch.Tensor, float]:
  """Translate root poses while preserving their patch-relative XYZ."""
  if root_pose.ndim != 2 or root_pose.shape[1] != 7:
    raise ValueError("root_pose must have shape (N, 7)")
  if old_origins.shape != (root_pose.shape[0], 3):
    raise ValueError("old_origins must have shape (N, 3)")
  if new_origins.shape != old_origins.shape:
    raise ValueError("new_origins must match old_origins")
  if not all(
    torch.isfinite(value).all()
    for value in (root_pose, old_origins, new_origins)
  ):
    raise ValueError("root poses and origins must be finite")
  relative_before = root_pose[:, :3] - old_origins
  relocated = root_pose.clone()
  relocated[:, :3] += new_origins - old_origins
  relative_after = relocated[:, :3] - new_origins
  error = float(torch.max(torch.abs(relative_after - relative_before)))
  return relocated, error


__all__ = [
  "BASE_OBSTACLE_COUNT",
  "BASE_PATCH_SIZE",
  "DEFAULT_CORRIDOR_HALF_WIDTH",
  "DIFFICULTY_LABELS",
  "HEIGHT_SCAN_HALF_EXTENTS",
  "HEIGHT_SCAN_SIZE",
  "LOW_DIFFICULTY",
  "MEDIUM_DIFFICULTY",
  "PATCH_SIZE",
  "PROPRIO_TERRAIN_CURVE_KINDS",
  "PROPRIO_TERRAIN_KIND_TO_TYPE",
  "ROUTE_START_LOCAL",
  "RouteFootprint",
  "SCALED_OBSTACLE_COUNT",
  "TERRAIN_CURVE_KINDS",
  "TERRAIN_KIND_TO_TYPE",
  "TerrainCurveKind",
  "TerrainCurveSubTerrainCfg",
  "compute_route_footprint",
  "continuous_transition_coverage",
  "difficulty_for_level",
  "effective_terrain_parameters",
  "make_terrain_curve_generator",
  "relocate_root_pose",
  "slope_direction_is_compatible",
  "validate_route_footprint",
]
