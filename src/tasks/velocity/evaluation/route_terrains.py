"""Evaluation-only continuous terrain profiles for Go2 route rollouts.

The profiles deliberately use the terrain keys already present in V7.  This
keeps ``ModeVelocityCommand`` focus-column discovery valid while the metadata
exposes the actual route family and direction to an evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
import uuid

import mujoco
import numpy as np

from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane


PATCH_SIZE = (8.0, 4.0)
TERRAIN_SCAN_SIZE_X = 1.6
TERRAIN_SCAN_HALF_X = TERRAIN_SCAN_SIZE_X / 2.0
ROUTE_START_X = 1.0
FEATURE_START_X = 2.0
FEATURE_END_X = 4.4
ROUTE_END_X = 7.0
ROUTE_LENGTH = ROUTE_END_X - ROUTE_START_X
STEP_WIDTH = 0.3
STEP_COUNT = int(round((FEATURE_END_X - FEATURE_START_X) / STEP_WIDTH))
GENERATOR_NUM_ROWS = 10
GENERATOR_NUM_COLS = 4

STAIRS_UP_DOWN_PATCH_SIZE = (12.0, 4.0)
STAIRS_UP_DOWN_START_X = 1.0
STAIRS_UP_DOWN_ASCENT_X = (2.0, 4.4)
STAIRS_UP_DOWN_TOP_END_X = 7.6
STAIRS_UP_DOWN_DESCENT_END_X = 10.0
STAIRS_UP_DOWN_END_X = 11.0

RouteTerrainKind = Literal[
  "stairs_up",
  "stairs_down",
  "slope_up",
  "slope_down",
]

TERRAIN_KIND_TO_KEY: dict[RouteTerrainKind, str] = {
  "stairs_up": "pyramid_stairs",
  "stairs_down": "pyramid_stairs_inv",
  "slope_up": "hf_pyramid_slope",
  "slope_down": "hf_pyramid_slope_inv",
}
TERRAIN_KEY_TO_KIND = {value: key for key, value in TERRAIN_KIND_TO_KEY.items()}


@dataclass(frozen=True)
class RouteTerrainMetadata:
  """Geometry-independent route metadata for one generated profile."""

  kind: RouteTerrainKind
  terrain_key: str
  family: Literal["stairs", "slope"]
  direction: Literal["up", "down"]
  difficulty: float
  patch_size: tuple[float, float]
  start_x: float
  feature_start_x: float
  feature_end_x: float
  end_x: float
  route_length: float
  entry_surface_z: float
  exit_surface_z: float
  step_height: float
  slope: float


def validate_route_terrain_parameters(
  kind: str,
  difficulty: float,
  *,
  patch_size: tuple[float, float] = PATCH_SIZE,
) -> None:
  """Validate profile names, difficulty, and fixed route geometry bounds."""
  if kind not in TERRAIN_KIND_TO_KEY:
    raise ValueError(f"unknown route terrain kind: {kind!r}")
  if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
    raise ValueError(f"difficulty must be finite and in [0, 1], got {difficulty}")
  if (
    len(patch_size) != 2
    or not all(math.isfinite(value) and value > 0.0 for value in patch_size)
    or patch_size != PATCH_SIZE
  ):
    raise ValueError(f"patch_size must be exactly {PATCH_SIZE}, got {patch_size}")
  if not (
    0.0 <= ROUTE_START_X < FEATURE_START_X < FEATURE_END_X < ROUTE_END_X <= patch_size[0]
  ):
    raise ValueError("route feature bounds must lie inside the patch")
  if (
    ROUTE_START_X - TERRAIN_SCAN_HALF_X < 0.0
    or ROUTE_END_X + TERRAIN_SCAN_HALF_X > patch_size[0]
  ):
    raise ValueError("terrain scan footprint must remain inside the patch")
  if STEP_COUNT <= 0 or not math.isclose(
    FEATURE_START_X + STEP_COUNT * STEP_WIDTH, FEATURE_END_X, abs_tol=1.0e-9
  ):
    raise ValueError("feature length must contain an integral number of steps")


def route_terrain_bounds() -> dict[str, tuple[float, float]]:
  """Return the patch and route x bounds used by all profiles."""
  return {
    "patch_x": (0.0, PATCH_SIZE[0]),
    "patch_y": (0.0, PATCH_SIZE[1]),
    "route_x": (ROUTE_START_X, ROUTE_END_X),
    "feature_x": (FEATURE_START_X, FEATURE_END_X),
    "start_scan_x": (
      ROUTE_START_X - TERRAIN_SCAN_HALF_X,
      ROUTE_START_X + TERRAIN_SCAN_HALF_X,
    ),
    "end_scan_x": (
      ROUTE_END_X - TERRAIN_SCAN_HALF_X,
      ROUTE_END_X + TERRAIN_SCAN_HALF_X,
    ),
  }


def continuous_route_difficulty_matrix(
  seed: int,
  difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
  """Reproduce generator curriculum difficulties as ``[row, column]``.

  The terrain profiles intentionally do not consume the generator RNG.  This
  makes this column-major loop identical to ``TerrainGenerator`` for the same
  seed and lets evaluators report the exact jittered difficulty.
  """
  if not isinstance(seed, (int, np.integer)):
    raise TypeError("seed must be an integer")
  if (
    len(difficulty_range) != 2
    or not all(math.isfinite(value) for value in difficulty_range)
    or difficulty_range[0] > difficulty_range[1]
  ):
    raise ValueError("difficulty_range must contain finite ascending bounds")
  lower, upper = difficulty_range
  rng = np.random.default_rng(int(seed))
  difficulties = np.empty(
    (GENERATOR_NUM_ROWS, GENERATOR_NUM_COLS), dtype=np.float64
  )
  for column in range(GENERATOR_NUM_COLS):
    for row in range(GENERATOR_NUM_ROWS):
      normalized = (row + rng.uniform()) / GENERATOR_NUM_ROWS
      difficulties[row, column] = lower + (upper - lower) * normalized
  return difficulties


def _profile_parameters(kind: RouteTerrainKind, difficulty: float) -> tuple[float, float]:
  step_height = 0.02 + difficulty * (0.12 - 0.02)
  slope = difficulty * 0.4
  if kind.startswith("stairs"):
    return step_height, 0.0
  return 0.0, slope


def route_terrain_metadata(
  kind: RouteTerrainKind,
  difficulty: float,
) -> RouteTerrainMetadata:
  """Return deterministic metadata for a profile at ``difficulty``."""
  validate_route_terrain_parameters(kind, difficulty)
  step_height, slope = _profile_parameters(kind, difficulty)
  rise = step_height * STEP_COUNT if kind.startswith("stairs") else slope * (
    FEATURE_END_X - FEATURE_START_X
  )
  direction: Literal["up", "down"] = "up" if kind.endswith("_up") else "down"
  family: Literal["stairs", "slope"] = (
    "stairs" if kind.startswith("stairs") else "slope"
  )
  return RouteTerrainMetadata(
    kind=kind,
    terrain_key=TERRAIN_KIND_TO_KEY[kind],
    family=family,
    direction=direction,
    difficulty=float(difficulty),
    patch_size=PATCH_SIZE,
    start_x=ROUTE_START_X,
    feature_start_x=FEATURE_START_X,
    feature_end_x=FEATURE_END_X,
    end_x=ROUTE_END_X,
    route_length=ROUTE_LENGTH,
    entry_surface_z=rise if direction == "down" else 0.0,
    exit_surface_z=rise if direction == "up" else 0.0,
    step_height=step_height,
    slope=slope,
  )


def route_surface_height(
  kind: RouteTerrainKind,
  difficulty: float,
  x: float | np.ndarray,
) -> float | np.ndarray:
  """Evaluate the exact one-dimensional route surface height at x."""
  metadata = route_terrain_metadata(kind, difficulty)
  values = np.asarray(x, dtype=np.float64)
  if not np.isfinite(values).all():
    raise ValueError("x must contain only finite values")
  if (values < 0.0).any() or (values > PATCH_SIZE[0]).any():
    raise ValueError(f"x must lie in [0, {PATCH_SIZE[0]}]")

  result = np.zeros_like(values)
  if metadata.family == "stairs":
    total = metadata.step_height * STEP_COUNT
    if metadata.direction == "down":
      result.fill(total)
      inside = (values >= FEATURE_START_X) & (values < FEATURE_END_X)
      index = np.floor((values[inside] - FEATURE_START_X) / STEP_WIDTH).astype(int)
      result[inside] = np.maximum(STEP_COUNT - index - 1, 0) * metadata.step_height
      result[values >= FEATURE_END_X] = 0.0
    else:
      inside = (values >= FEATURE_START_X) & (values < FEATURE_END_X)
      index = np.floor((values[inside] - FEATURE_START_X) / STEP_WIDTH).astype(int)
      result[inside] = (index + 1) * metadata.step_height
      result[values >= FEATURE_END_X] = total
  elif metadata.direction == "down":
    result.fill(metadata.slope * (FEATURE_END_X - FEATURE_START_X))
    inside = (values >= FEATURE_START_X) & (values < FEATURE_END_X)
    progress = (values[inside] - FEATURE_START_X) / (
      FEATURE_END_X - FEATURE_START_X
    )
    result[inside] = (1.0 - progress) * metadata.slope * (
      FEATURE_END_X - FEATURE_START_X
    )
    result[values >= FEATURE_END_X] = 0.0
  else:
    inside = (values >= FEATURE_START_X) & (values < FEATURE_END_X)
    progress = (values[inside] - FEATURE_START_X) / (
      FEATURE_END_X - FEATURE_START_X
    )
    result[inside] = progress * metadata.slope * (
      FEATURE_END_X - FEATURE_START_X
    )
    result[values >= FEATURE_END_X] = metadata.slope * (
      FEATURE_END_X - FEATURE_START_X
    )
  return float(result) if values.ndim == 0 else result


def _add_box(
  body: mujoco.MjsBody,
  x0: float,
  x1: float,
  height: float,
  width: float,
) -> mujoco.MjsGeom:
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=((x1 - x0) / 2.0, width / 2.0, max(height, 1.0e-6) / 2.0),
    pos=((x0 + x1) / 2.0, width / 2.0, max(height, 1.0e-6) / 2.0),
  )
  return geom


def stairs_up_down_surface_height(
  difficulty: float, x: float | np.ndarray
) -> float | np.ndarray:
  """Evaluate the playback-only stairs-up/platform/stairs-down profile."""
  if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
    raise ValueError(f"difficulty must be finite and in [0, 1], got {difficulty}")
  values = np.asarray(x, dtype=np.float64)
  if not np.isfinite(values).all():
    raise ValueError("x must contain only finite values")
  if (values < 0.0).any() or (values > STAIRS_UP_DOWN_PATCH_SIZE[0]).any():
    raise ValueError(f"x must lie in [0, {STAIRS_UP_DOWN_PATCH_SIZE[0]}]")

  step_height = 0.02 + difficulty * (0.12 - 0.02)
  result = np.zeros_like(values)
  up_start, up_end = STAIRS_UP_DOWN_ASCENT_X
  ascending = (values >= up_start) & (values < up_end)
  up_index = np.floor((values[ascending] - up_start) / STEP_WIDTH).astype(int)
  result[ascending] = (up_index + 1) * step_height
  top_height = STEP_COUNT * step_height
  top = (values >= up_end) & (values < STAIRS_UP_DOWN_TOP_END_X)
  result[top] = top_height
  descending = (
    (values >= STAIRS_UP_DOWN_TOP_END_X)
    & (values < STAIRS_UP_DOWN_DESCENT_END_X)
  )
  down_index = np.floor(
    (values[descending] - STAIRS_UP_DOWN_TOP_END_X) / STEP_WIDTH
  ).astype(int)
  result[descending] = np.maximum(STEP_COUNT - down_index - 1, 0) * step_height
  return float(result) if values.ndim == 0 else result


@dataclass(kw_only=True)
class StairsUpDownDemoTerrainCfg(SubTerrainCfg):
  """Playback-only route with full ascent, top platform, and descent."""

  size: tuple[float, float] = STAIRS_UP_DOWN_PATCH_SIZE

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del rng
    if self.size != STAIRS_UP_DOWN_PATCH_SIZE:
      raise ValueError(
        f"stairs up/down demo size must be {STAIRS_UP_DOWN_PATCH_SIZE}"
      )
    step_height = 0.02 + difficulty * (0.12 - 0.02)
    body = spec.body("terrain")
    geometries = [
      TerrainGeometry(
        geom=make_plane(body, STAIRS_UP_DOWN_PATCH_SIZE, 0.0, center_zero=False)[0]
      )
    ]
    up_start, up_end = STAIRS_UP_DOWN_ASCENT_X
    for index in range(STEP_COUNT):
      geometries.append(
        TerrainGeometry(
          geom=_add_box(
            body,
            up_start + index * STEP_WIDTH,
            up_start + (index + 1) * STEP_WIDTH,
            (index + 1) * step_height,
            STAIRS_UP_DOWN_PATCH_SIZE[1],
          )
        )
      )
    top_height = STEP_COUNT * step_height
    geometries.append(
      TerrainGeometry(
        geom=_add_box(
          body,
          up_end,
          STAIRS_UP_DOWN_TOP_END_X,
          top_height,
          STAIRS_UP_DOWN_PATCH_SIZE[1],
        )
      )
    )
    for index in range(STEP_COUNT):
      height = max(STEP_COUNT - index - 1, 0) * step_height
      if height > 0.0:
        geometries.append(
          TerrainGeometry(
            geom=_add_box(
              body,
              STAIRS_UP_DOWN_TOP_END_X + index * STEP_WIDTH,
              STAIRS_UP_DOWN_TOP_END_X + (index + 1) * STEP_WIDTH,
              height,
              STAIRS_UP_DOWN_PATCH_SIZE[1],
            )
          )
        )
    return TerrainOutput(
      origin=np.array(
        [STAIRS_UP_DOWN_START_X, STAIRS_UP_DOWN_PATCH_SIZE[1] / 2.0, 0.0],
        dtype=np.float64,
      ),
      geometries=geometries,
    )


def make_stairs_up_down_demo_terrain_generator(
  *, seed: int | None = 42
) -> TerrainGeneratorCfg:
  """Build a deterministic playback generator for an up/down stair route."""
  return TerrainGeneratorCfg(
    seed=seed,
    size=STAIRS_UP_DOWN_PATCH_SIZE,
    border_width=20.0,
    num_rows=GENERATOR_NUM_ROWS,
    num_cols=1,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      "pyramid_stairs": StairsUpDownDemoTerrainCfg(proportion=1.0),
    },
  )


def _stairs_output(
  kind: RouteTerrainKind,
  difficulty: float,
  spec: mujoco.MjSpec,
) -> TerrainOutput:
  metadata = route_terrain_metadata(kind, difficulty)
  body = spec.body("terrain")
  geometries: list[TerrainGeometry] = []
  ground = make_plane(body, PATCH_SIZE, 0.0, center_zero=False)[0]
  geometries.append(TerrainGeometry(geom=ground))
  total = metadata.step_height * STEP_COUNT

  if metadata.direction == "down":
    if total > 0.0:
      geometries.append(
        TerrainGeometry(
          geom=_add_box(body, 0.0, FEATURE_START_X, total, PATCH_SIZE[1])
        )
      )
    for index in range(STEP_COUNT):
      height = max(STEP_COUNT - index - 1, 0) * metadata.step_height
      if height > 0.0:
        geometries.append(
          TerrainGeometry(
            geom=_add_box(
              body,
              FEATURE_START_X + index * STEP_WIDTH,
              FEATURE_START_X + (index + 1) * STEP_WIDTH,
              height,
              PATCH_SIZE[1],
            )
          )
        )
  else:
    for index in range(STEP_COUNT):
      height = (index + 1) * metadata.step_height
      geometries.append(
        TerrainGeometry(
          geom=_add_box(
            body,
            FEATURE_START_X + index * STEP_WIDTH,
            FEATURE_START_X + (index + 1) * STEP_WIDTH,
            height,
            PATCH_SIZE[1],
          )
        )
      )
    if total > 0.0:
      geometries.append(
        TerrainGeometry(
          geom=_add_box(body, FEATURE_END_X, PATCH_SIZE[0], total, PATCH_SIZE[1])
        )
      )
  return TerrainOutput(
    origin=np.array(
      [ROUTE_START_X, PATCH_SIZE[1] / 2.0, metadata.entry_surface_z],
      dtype=np.float64,
    ),
    geometries=geometries,
  )


def _slope_output(
  kind: RouteTerrainKind,
  difficulty: float,
  spec: mujoco.MjSpec,
) -> TerrainOutput:
  metadata = route_terrain_metadata(kind, difficulty)
  resolution = 0.05
  x_count = int(round(PATCH_SIZE[0] / resolution)) + 1
  y_count = int(round(PATCH_SIZE[1] / resolution)) + 1
  x_values = np.linspace(0.0, PATCH_SIZE[0], x_count)
  heights = np.asarray(route_surface_height(kind, difficulty, x_values))
  heightfield = np.broadcast_to(heights[None, :], (y_count, x_count)).copy()
  minimum = float(heightfield.min())
  maximum = float(heightfield.max())
  vertical_range = max(maximum - minimum, 1.0e-6)
  normalized = ((heightfield - minimum) / vertical_range).astype(np.float32)

  unique_id = uuid.uuid4().hex
  field = spec.add_hfield(
    name=f"route_hfield_{unique_id}",
    size=[PATCH_SIZE[0] / 2.0, PATCH_SIZE[1] / 2.0, vertical_range, 1.0],
    nrow=y_count,
    ncol=x_count,
    userdata=normalized.flatten().tolist(),
  )
  body = spec.body("terrain")
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname=field.name,
    pos=[PATCH_SIZE[0] / 2.0, PATCH_SIZE[1] / 2.0, minimum],
  )
  return TerrainOutput(
    origin=np.array(
      [ROUTE_START_X, PATCH_SIZE[1] / 2.0, metadata.entry_surface_z],
      dtype=np.float64,
    ),
    geometries=[TerrainGeometry(geom=geom, hfield=field)],
  )


@dataclass(kw_only=True)
class ContinuousRouteTerrainCfg(SubTerrainCfg):
  """One evaluation-only one-dimensional transition profile."""

  kind: RouteTerrainKind
  size: tuple[float, float] = PATCH_SIZE

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del rng
    validate_route_terrain_parameters(self.kind, difficulty, patch_size=self.size)
    if self.kind.startswith("stairs"):
      return _stairs_output(self.kind, difficulty, spec)
    return _slope_output(self.kind, difficulty, spec)


def make_continuous_route_terrain_generator(
  *,
  seed: int | None = 42,
) -> TerrainGeneratorCfg:
  """Build a 10-row/4-column generator for evaluation route profiles."""
  return TerrainGeneratorCfg(
    seed=seed,
    size=PATCH_SIZE,
    border_width=20.0,
    num_rows=GENERATOR_NUM_ROWS,
    num_cols=GENERATOR_NUM_COLS,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      TERRAIN_KIND_TO_KEY["stairs_up"]: ContinuousRouteTerrainCfg(
        kind="stairs_up", proportion=1.0
      ),
      TERRAIN_KIND_TO_KEY["stairs_down"]: ContinuousRouteTerrainCfg(
        kind="stairs_down", proportion=1.0
      ),
      TERRAIN_KIND_TO_KEY["slope_up"]: ContinuousRouteTerrainCfg(
        kind="slope_up", proportion=1.0
      ),
      TERRAIN_KIND_TO_KEY["slope_down"]: ContinuousRouteTerrainCfg(
        kind="slope_down", proportion=1.0
      ),
    },
  )


__all__ = [
  "ContinuousRouteTerrainCfg",
  "FEATURE_END_X",
  "FEATURE_START_X",
  "GENERATOR_NUM_COLS",
  "GENERATOR_NUM_ROWS",
  "PATCH_SIZE",
  "ROUTE_END_X",
  "ROUTE_LENGTH",
  "ROUTE_START_X",
  "RouteTerrainKind",
  "RouteTerrainMetadata",
  "STEP_COUNT",
  "STEP_WIDTH",
  "STAIRS_UP_DOWN_ASCENT_X",
  "STAIRS_UP_DOWN_DESCENT_END_X",
  "STAIRS_UP_DOWN_END_X",
  "STAIRS_UP_DOWN_PATCH_SIZE",
  "STAIRS_UP_DOWN_START_X",
  "STAIRS_UP_DOWN_TOP_END_X",
  "StairsUpDownDemoTerrainCfg",
  "TERRAIN_SCAN_HALF_X",
  "TERRAIN_SCAN_SIZE_X",
  "TERRAIN_KEY_TO_KIND",
  "TERRAIN_KIND_TO_KEY",
  "continuous_route_difficulty_matrix",
  "make_continuous_route_terrain_generator",
  "make_stairs_up_down_demo_terrain_generator",
  "route_surface_height",
  "route_terrain_bounds",
  "route_terrain_metadata",
  "stairs_up_down_surface_height",
  "validate_route_terrain_parameters",
]
