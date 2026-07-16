"""Evaluation-only high-difficulty and continuous terrain boundary scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
import uuid

import mujoco
import numpy as np

from mjlab.terrains import (
  HfDiscreteObstaclesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

from .route_terrains import (
  FEATURE_END_X,
  FEATURE_START_X,
  GENERATOR_NUM_ROWS,
  PATCH_SIZE as TRANSITION_PATCH_SIZE,
  ROUTE_END_X,
  ROUTE_LENGTH,
  ROUTE_START_X,
  TERRAIN_KIND_TO_KEY as BASE_TRANSITION_KEYS,
  ContinuousRouteTerrainCfg,
  route_terrain_metadata,
)
from .terrain_curved_routes import (
  PATCH_SIZE as CURVE_PATCH_SIZE,
  ROUTE_START_LOCAL as CURVE_ROUTE_START_LOCAL,
  SCALED_OBSTACLE_COUNT,
  TERRAIN_CURVE_KINDS,
  TerrainCurveKind,
)


HIGH_DIFFICULTIES = (0.8, 1.0)
HIGH_DIFFICULTY_LABELS = ("high", "extreme")
TRANSITION_HORIZONTAL_SCALE = 0.1
TRANSITION_GUARD_WIDTH = 0.1
TRANSITION_CORRIDOR_HALF_WIDTH = 0.4
TRANSITION_SCAN_HALF_EXTENTS = (0.8, 0.5)
V7_ROUGH_NOISE_RANGE = (0.01, 0.06)
V7_ROUGH_NOISE_STEP = 0.01
V7_OBSTACLE_HEIGHT_RANGE = (0.02, 0.10)
V7_OBSTACLE_WIDTH_RANGE = (0.30, 0.80)
V7_OBSTACLE_AREA_DENSITY = 32 / (8.0 * 8.0)

ContinuousTransitionKind = Literal[
  "slope_up",
  "slope_down",
  "stairs_up",
  "stairs_down",
  "random_rough",
  "discrete_obstacle",
]
CONTINUOUS_TRANSITION_KINDS: tuple[ContinuousTransitionKind, ...] = (
  "slope_up",
  "slope_down",
  "stairs_up",
  "stairs_down",
  "random_rough",
  "discrete_obstacle",
)
CONTINUOUS_TRANSITION_KEYS: dict[ContinuousTransitionKind, str] = {
  "slope_up": BASE_TRANSITION_KEYS["slope_up"],
  "slope_down": BASE_TRANSITION_KEYS["slope_down"],
  "stairs_up": BASE_TRANSITION_KEYS["stairs_up"],
  "stairs_down": BASE_TRANSITION_KEYS["stairs_down"],
  "random_rough": "random_rough",
  "discrete_obstacle": "discrete_obstacles",
}


def difficulty_for_high_level(level: int) -> tuple[str, float]:
  """Map the two evaluation rows to explicit high and maximum difficulty."""
  if not isinstance(level, (int, np.integer)) or isinstance(level, bool):
    raise TypeError("high terrain level must be an integer")
  if level not in (0, 1):
    raise ValueError("high terrain level must be 0 (high) or 1 (extreme)")
  return HIGH_DIFFICULTY_LABELS[level], HIGH_DIFFICULTIES[level]


def effective_high_terrain_parameters(
  kind: TerrainCurveKind, level: int
) -> dict[str, object]:
  """Report the geometry actually consumed by the V7 terrain primitive."""
  label, difficulty = difficulty_for_high_level(level)
  common: dict[str, object] = {
    "difficulty_label": label,
    "requested_difficulty": difficulty,
  }
  if kind == "slope_up":
    return {
      **common,
      "primitive": "HfPyramidSlopedTerrainCfg",
      "slope_gradient": difficulty * 0.4,
      "inverted": True,
      "difficulty_affects_geometry": True,
    }
  if kind == "slope_down":
    return {
      **common,
      "primitive": "HfPyramidSlopedTerrainCfg",
      "slope_gradient": -difficulty * 0.4,
      "inverted": False,
      "difficulty_affects_geometry": True,
    }
  if kind == "random_rough":
    return {
      **common,
      "primitive": "HfRandomUniformTerrainCfg",
      "noise_range": list(V7_ROUGH_NOISE_RANGE),
      "noise_step": V7_ROUGH_NOISE_STEP,
      "difficulty_affects_geometry": False,
      "difficulty_invariant": True,
      "difficulty_reason": (
        "V7 HfRandomUniformTerrainCfg deletes its difficulty argument; high "
        "and extreme rows are independent samples of the same distribution"
      ),
    }
  if kind == "discrete_obstacle":
    return {
      **common,
      "primitive": "HfDiscreteObstaclesTerrainCfg",
      "obstacle_height": V7_OBSTACLE_HEIGHT_RANGE[0]
      + difficulty * (V7_OBSTACLE_HEIGHT_RANGE[1] - V7_OBSTACLE_HEIGHT_RANGE[0]),
      "obstacle_width_range": list(V7_OBSTACLE_WIDTH_RANGE),
      "num_obstacles": SCALED_OBSTACLE_COUNT,
      "difficulty_affects_geometry": True,
    }
  raise ValueError(f"unknown terrain curve kind: {kind!r}")


def _high_curve_delegate(kind: TerrainCurveKind) -> SubTerrainCfg:
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
    )
  if kind == "random_rough":
    return HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=V7_ROUGH_NOISE_RANGE,
      noise_step=V7_ROUGH_NOISE_STEP,
      border_width=0.25,
    )
  if kind == "discrete_obstacle":
    return HfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      obstacle_width_range=V7_OBSTACLE_WIDTH_RANGE,
      obstacle_height_range=V7_OBSTACLE_HEIGHT_RANGE,
      num_obstacles=SCALED_OBSTACLE_COUNT,
      platform_width=2.0,
      border_width=0.25,
      origin_z_offset=0.02,
    )
  raise ValueError(f"unknown terrain curve kind: {kind!r}")


@dataclass(kw_only=True)
class HighDifficultyTerrainCurveSubTerrainCfg(SubTerrainCfg):
  """Delegate to one V7 primitive at an exact high evaluation difficulty."""

  kind: TerrainCurveKind
  size: tuple[float, float] = CURVE_PATCH_SIZE

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    if self.kind not in TERRAIN_CURVE_KINDS:
      raise ValueError(f"unknown terrain curve kind: {self.kind!r}")
    if self.size != CURVE_PATCH_SIZE:
      raise ValueError(f"high curve patch size must be {CURVE_PATCH_SIZE}")
    level = 0 if difficulty < 0.5 else 1
    fixed_difficulty = HIGH_DIFFICULTIES[level]
    delegate = _high_curve_delegate(self.kind)
    delegate.size = self.size
    output = delegate.function(fixed_difficulty, spec, rng)
    if not np.allclose(
      output.origin[:2], CURVE_ROUTE_START_LOCAL, rtol=0.0, atol=1.0e-12
    ):
      raise RuntimeError(
        f"{self.kind} origin {output.origin[:2]} is not curve start "
        f"{CURVE_ROUTE_START_LOCAL}"
      )
    return output


def make_high_difficulty_curve_generator(
  seed: int | None = 42,
) -> TerrainGeneratorCfg:
  """Build exact 0.8/1.0 rows for all four V7 curve primitives."""
  return TerrainGeneratorCfg(
    seed=seed,
    size=CURVE_PATCH_SIZE,
    border_width=20.0,
    num_rows=2,
    num_cols=len(TERRAIN_CURVE_KINDS),
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      kind: HighDifficultyTerrainCurveSubTerrainCfg(kind=kind, proportion=1.0)
      for kind in TERRAIN_CURVE_KINDS
    },
  )


@dataclass(frozen=True)
class StraightTransitionFootprint:
  patch_size: tuple[float, float]
  route_bounds: tuple[tuple[float, float], tuple[float, float]]
  feature_bounds: tuple[tuple[float, float], tuple[float, float]]
  corridor_bounds: tuple[tuple[float, float], tuple[float, float]]
  scan_footprint_bounds: tuple[tuple[float, float], tuple[float, float]]
  route_inside_patch: bool
  corridor_inside_patch: bool
  scan_footprint_inside_patch: bool
  corridor_boundary_margin: float
  scan_boundary_margin: float


def validate_straight_transition_footprint(
  *,
  corridor_half_width: float = TRANSITION_CORRIDOR_HALF_WIDTH,
  scan_half_extents: tuple[float, float] = TRANSITION_SCAN_HALF_EXTENTS,
) -> StraightTransitionFootprint:
  """Validate the straight route, swept corridor, and yaw-aligned height scan."""
  if not math.isfinite(corridor_half_width) or corridor_half_width <= 0.0:
    raise ValueError("corridor_half_width must be finite and positive")
  if len(scan_half_extents) != 2 or not all(
    math.isfinite(value) and value > 0.0 for value in scan_half_extents
  ):
    raise ValueError("scan_half_extents must contain two finite positive values")
  route_y = TRANSITION_PATCH_SIZE[1] / 2.0
  route_bounds = ((ROUTE_START_X, ROUTE_END_X), (route_y, route_y))
  feature_bounds = (
    (FEATURE_START_X, FEATURE_END_X),
    (0.0, TRANSITION_PATCH_SIZE[1]),
  )
  corridor_bounds = (
    (ROUTE_START_X, ROUTE_END_X),
    (route_y - corridor_half_width, route_y + corridor_half_width),
  )
  scan_x, scan_y = scan_half_extents
  scan_bounds = (
    (ROUTE_START_X - scan_x, ROUTE_END_X + scan_x),
    (route_y - scan_y, route_y + scan_y),
  )

  def inside(bounds: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    return (
      bounds[0][0] >= 0.0
      and bounds[0][1] <= TRANSITION_PATCH_SIZE[0]
      and bounds[1][0] >= 0.0
      and bounds[1][1] <= TRANSITION_PATCH_SIZE[1]
    )

  def margin(bounds: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return min(
      bounds[0][0],
      TRANSITION_PATCH_SIZE[0] - bounds[0][1],
      bounds[1][0],
      TRANSITION_PATCH_SIZE[1] - bounds[1][1],
    )

  footprint = StraightTransitionFootprint(
    patch_size=TRANSITION_PATCH_SIZE,
    route_bounds=route_bounds,
    feature_bounds=feature_bounds,
    corridor_bounds=corridor_bounds,
    scan_footprint_bounds=scan_bounds,
    route_inside_patch=inside(route_bounds),
    corridor_inside_patch=inside(corridor_bounds),
    scan_footprint_inside_patch=inside(scan_bounds),
    corridor_boundary_margin=margin(corridor_bounds),
    scan_boundary_margin=margin(scan_bounds),
  )
  if not footprint.route_inside_patch or not footprint.corridor_inside_patch:
    raise ValueError("straight route corridor leaves the transition patch")
  if not footprint.scan_footprint_inside_patch:
    raise ValueError("height-scan footprint leaves the transition patch")
  return footprint


@dataclass(frozen=True)
class BoundaryTransitionMetadata:
  kind: ContinuousTransitionKind
  terrain_key: str
  family: str
  direction: str
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
  difficulty_affects_geometry: bool
  effective_parameters: dict[str, object]
  geometry_contract: str


def boundary_transition_metadata(
  kind: ContinuousTransitionKind, difficulty: float
) -> BoundaryTransitionMetadata:
  """Describe one real, intra-patch approach-feature-exit surface."""
  if kind not in CONTINUOUS_TRANSITION_KINDS:
    raise ValueError(f"unknown continuous transition kind: {kind!r}")
  if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
    raise ValueError("difficulty must be finite and in [0, 1]")
  if kind in BASE_TRANSITION_KEYS:
    base = route_terrain_metadata(kind, difficulty)  # type: ignore[arg-type]
    parameters: dict[str, object]
    if base.family == "stairs":
      parameters = {
        "step_height": base.step_height,
        "step_width": 0.3,
        "step_count": 8,
      }
    else:
      parameters = {"slope_gradient": base.slope}
    return BoundaryTransitionMetadata(
      kind=kind,
      terrain_key=base.terrain_key,
      family=base.family,
      direction=base.direction,
      difficulty=difficulty,
      patch_size=base.patch_size,
      start_x=base.start_x,
      feature_start_x=base.feature_start_x,
      feature_end_x=base.feature_end_x,
      end_x=base.end_x,
      route_length=base.route_length,
      entry_surface_z=base.entry_surface_z,
      exit_surface_z=base.exit_surface_z,
      step_height=base.step_height,
      slope=base.slope,
      difficulty_affects_geometry=True,
      effective_parameters=parameters,
      geometry_contract="single intra-patch approach-feature-exit geometry",
    )
  if kind == "random_rough":
    parameters = {
      "noise_range": list(V7_ROUGH_NOISE_RANGE),
      "noise_step": V7_ROUGH_NOISE_STEP,
      "junction_guard_width": TRANSITION_GUARD_WIDTH,
      "difficulty_invariant": True,
      "difficulty_reason": "V7 random rough ignores difficulty",
    }
    family = "rough"
    affects = False
  else:
    obstacle_height = V7_OBSTACLE_HEIGHT_RANGE[0] + difficulty * (
      V7_OBSTACLE_HEIGHT_RANGE[1] - V7_OBSTACLE_HEIGHT_RANGE[0]
    )
    feature_area = (FEATURE_END_X - FEATURE_START_X) * TRANSITION_PATCH_SIZE[1]
    parameters = {
      "obstacle_height": obstacle_height,
      "obstacle_height_mode": "choice",
      "obstacle_width_range": list(V7_OBSTACLE_WIDTH_RANGE),
      "num_obstacles": max(1, round(feature_area * V7_OBSTACLE_AREA_DENSITY)),
      "junction_guard_width": TRANSITION_GUARD_WIDTH,
    }
    family = "obstacle"
    affects = True
  return BoundaryTransitionMetadata(
    kind=kind,
    terrain_key=CONTINUOUS_TRANSITION_KEYS[kind],
    family=family,
    direction="level",
    difficulty=difficulty,
    patch_size=TRANSITION_PATCH_SIZE,
    start_x=ROUTE_START_X,
    feature_start_x=FEATURE_START_X,
    feature_end_x=FEATURE_END_X,
    end_x=ROUTE_END_X,
    route_length=ROUTE_LENGTH,
    entry_surface_z=0.0,
    exit_surface_z=0.0,
    step_height=0.0,
    slope=0.0,
    difficulty_affects_geometry=affects,
    effective_parameters=parameters,
    geometry_contract="single heightfield with flat approach and exit",
  )


def _local_rng(seed: int, kind: ContinuousTransitionKind, difficulty: float) -> np.random.Generator:
  kind_index = CONTINUOUS_TRANSITION_KINDS.index(kind)
  difficulty_key = int(round(difficulty * 1_000_000))
  return np.random.default_rng(np.random.SeedSequence([seed, kind_index, difficulty_key]))


def continuous_feature_heightfield(
  kind: Literal["random_rough", "discrete_obstacle"],
  difficulty: float,
  seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Create one deterministic surface; no independent terrain patches are joined."""
  if kind not in ("random_rough", "discrete_obstacle"):
    raise ValueError("feature heightfield supports only rough and obstacle")
  metadata = boundary_transition_metadata(kind, difficulty)
  x = np.linspace(
    0.0,
    TRANSITION_PATCH_SIZE[0],
    int(round(TRANSITION_PATCH_SIZE[0] / TRANSITION_HORIZONTAL_SCALE)) + 1,
  )
  y = np.linspace(
    0.0,
    TRANSITION_PATCH_SIZE[1],
    int(round(TRANSITION_PATCH_SIZE[1] / TRANSITION_HORIZONTAL_SCALE)) + 1,
  )
  heights = np.zeros((len(y), len(x)), dtype=np.float64)
  feature_columns = np.where(
    (x > FEATURE_START_X + TRANSITION_GUARD_WIDTH - 1.0e-12)
    & (x < FEATURE_END_X - TRANSITION_GUARD_WIDTH + 1.0e-12)
  )[0]
  rng = _local_rng(seed, kind, difficulty)
  if kind == "random_rough":
    choices = np.arange(
      V7_ROUGH_NOISE_RANGE[0],
      V7_ROUGH_NOISE_RANGE[1] + V7_ROUGH_NOISE_STEP / 2.0,
      V7_ROUGH_NOISE_STEP,
    )
    heights[:, feature_columns] = rng.choice(
      choices, size=(len(y), len(feature_columns))
    )
  else:
    height = float(metadata.effective_parameters["obstacle_height"])
    width_min = max(
      1, round(V7_OBSTACLE_WIDTH_RANGE[0] / TRANSITION_HORIZONTAL_SCALE)
    )
    width_max = max(
      width_min,
      round(V7_OBSTACLE_WIDTH_RANGE[1] / TRANSITION_HORIZONTAL_SCALE),
    )
    count = int(metadata.effective_parameters["num_obstacles"])
    if len(feature_columns) > width_min:
      for obstacle_index in range(count):
        width = int(rng.integers(width_min, width_max + 1))
        length = int(rng.integers(width_min, width_max + 1))
        x0 = int(
          rng.integers(
            feature_columns[0],
            max(
              feature_columns[-1] - width + 2,
              feature_columns[0] + 1,
            ),
          )
        )
        y0 = int(rng.integers(0, max(len(y) - length + 1, 1)))
        signed_height = rng.choice((-height, -height / 2.0, height / 2.0, height))
        heights[
          y0 : min(y0 + length, len(y)),
          x0 : min(x0 + width, feature_columns[-1] + 1),
        ] = signed_height
        if obstacle_index == 0:
          center_y = len(y) // 2
          heights[
            max(center_y - length // 2, 0) : min(center_y + (length + 1) // 2, len(y)),
            x0 : min(x0 + width, feature_columns[-1] + 1),
          ] = height
  # The exact junction columns and both scan-safe endpoint flats remain zero.
  heights[:, x <= FEATURE_START_X + TRANSITION_GUARD_WIDTH / 2.0] = 0.0
  heights[:, x >= FEATURE_END_X - TRANSITION_GUARD_WIDTH / 2.0] = 0.0
  return x, y, heights


def _heightfield_output(
  kind: Literal["random_rough", "discrete_obstacle"],
  difficulty: float,
  seed: int,
  spec: mujoco.MjSpec,
) -> TerrainOutput:
  _, _, heights = continuous_feature_heightfield(kind, difficulty, seed)
  minimum = float(heights.min())
  maximum = float(heights.max())
  vertical_range = max(maximum - minimum, 1.0e-6)
  normalized = ((heights - minimum) / vertical_range).astype(np.float32)
  field = spec.add_hfield(
    name=f"boundary_hfield_{uuid.uuid4().hex}",
    size=[
      TRANSITION_PATCH_SIZE[0] / 2.0,
      TRANSITION_PATCH_SIZE[1] / 2.0,
      vertical_range,
      max(vertical_range, 1.0e-3),
    ],
    nrow=heights.shape[0],
    ncol=heights.shape[1],
    userdata=normalized.flatten().tolist(),
  )
  geom = spec.body("terrain").add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname=field.name,
    pos=[
      TRANSITION_PATCH_SIZE[0] / 2.0,
      TRANSITION_PATCH_SIZE[1] / 2.0,
      minimum,
    ],
  )
  return TerrainOutput(
    origin=np.array(
      [ROUTE_START_X, TRANSITION_PATCH_SIZE[1] / 2.0, 0.0], dtype=np.float64
    ),
    geometries=[TerrainGeometry(geom=geom, hfield=field)],
  )


@dataclass(kw_only=True)
class ContinuousBoundaryTerrainCfg(SubTerrainCfg):
  """One straight, single-patch transition surface for boundary evaluation."""

  kind: ContinuousTransitionKind
  seed: int = 42
  size: tuple[float, float] = TRANSITION_PATCH_SIZE

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del rng
    if self.size != TRANSITION_PATCH_SIZE:
      raise ValueError(
        f"continuous transition patch size must be {TRANSITION_PATCH_SIZE}"
      )
    boundary_transition_metadata(self.kind, difficulty)
    if self.kind in BASE_TRANSITION_KEYS:
      delegate = ContinuousRouteTerrainCfg(
        kind=self.kind, proportion=1.0, size=self.size  # type: ignore[arg-type]
      )
      return delegate.function(difficulty, spec, np.random.default_rng(self.seed))
    return _heightfield_output(self.kind, difficulty, self.seed, spec)


def boundary_transition_difficulty_matrix(
  seed: int,
  difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
  """Reproduce generator difficulties; boundary terrain functions consume no RNG."""
  if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
    raise TypeError("seed must be an integer")
  if len(difficulty_range) != 2 or not all(
    math.isfinite(value) for value in difficulty_range
  ) or difficulty_range[0] > difficulty_range[1]:
    raise ValueError("difficulty_range must contain finite ascending bounds")
  lower, upper = difficulty_range
  rng = np.random.default_rng(int(seed))
  matrix = np.empty(
    (GENERATOR_NUM_ROWS, len(CONTINUOUS_TRANSITION_KINDS)), dtype=np.float64
  )
  for column in range(matrix.shape[1]):
    for row in range(matrix.shape[0]):
      normalized = (row + rng.uniform()) / GENERATOR_NUM_ROWS
      matrix[row, column] = lower + (upper - lower) * normalized
  return matrix


def make_boundary_transition_generator(seed: int | None = 42) -> TerrainGeneratorCfg:
  """Build all six credible straight intra-patch transition families."""
  effective_seed = 42 if seed is None else int(seed)
  validate_straight_transition_footprint()
  return TerrainGeneratorCfg(
    seed=seed,
    size=TRANSITION_PATCH_SIZE,
    border_width=20.0,
    num_rows=GENERATOR_NUM_ROWS,
    num_cols=len(CONTINUOUS_TRANSITION_KINDS),
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      CONTINUOUS_TRANSITION_KEYS[kind]: ContinuousBoundaryTerrainCfg(
        kind=kind, seed=effective_seed, proportion=1.0
      )
      for kind in CONTINUOUS_TRANSITION_KINDS
    },
  )


def continuous_boundary_coverage() -> dict[str, object]:
  """Truthful coverage declaration for the boundary transition suite."""
  return {
    "implemented": True,
    "continuous_intra_patch_transitions": True,
    "continuous_inter_patch_transitions": False,
    "route_kind": "straight",
    "transition_cases": list(CONTINUOUS_TRANSITION_KINDS),
    "single_surface_contract": True,
    "curve_transitions": False,
    "stairs_curves": False,
    "reason": (
      "Each straight route uses one generated approach-feature-exit patch. "
      "The old 8x4 m patch is not scaled or claimed to support curves."
    ),
  }


def reject_curved_transition(kind: str, route_kind: str) -> None:
  """Reject unsupported curved transition claims, especially curved stairs."""
  if route_kind == "straight":
    if kind not in CONTINUOUS_TRANSITION_KINDS:
      raise ValueError(f"unknown continuous transition kind: {kind!r}")
    return
  if route_kind not in ("arc", "s_curve"):
    raise ValueError("route_kind must be 'straight', 'arc', or 's_curve'")
  if kind.startswith("stairs"):
    raise ValueError(
      "stairs curves are unsupported: the 8x4 m tread geometry is route-aligned "
      "for a straight crossing and cannot provide a valid curved stair corridor"
    )
  raise ValueError(
    "continuous curved transitions are unsupported: the validated 8x4 m patch "
    "cannot contain the curve corridor and rotated height-scan footprint"
  )


__all__ = [
  "BoundaryTransitionMetadata",
  "CONTINUOUS_TRANSITION_KEYS",
  "CONTINUOUS_TRANSITION_KINDS",
  "CURVE_PATCH_SIZE",
  "CURVE_ROUTE_START_LOCAL",
  "ContinuousBoundaryTerrainCfg",
  "ContinuousTransitionKind",
  "HIGH_DIFFICULTIES",
  "HIGH_DIFFICULTY_LABELS",
  "HighDifficultyTerrainCurveSubTerrainCfg",
  "StraightTransitionFootprint",
  "TRANSITION_CORRIDOR_HALF_WIDTH",
  "TRANSITION_GUARD_WIDTH",
  "TRANSITION_PATCH_SIZE",
  "TRANSITION_SCAN_HALF_EXTENTS",
  "boundary_transition_difficulty_matrix",
  "boundary_transition_metadata",
  "continuous_boundary_coverage",
  "continuous_feature_heightfield",
  "difficulty_for_high_level",
  "effective_high_terrain_parameters",
  "make_boundary_transition_generator",
  "make_high_difficulty_curve_generator",
  "reject_curved_transition",
  "validate_straight_transition_footprint",
]
