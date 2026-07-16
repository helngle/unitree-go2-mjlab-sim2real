"""CPU contracts for high terrain and continuous boundary scenarios."""

from __future__ import annotations

import dataclasses
import unittest

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGenerator

from scripts.evaluate_go2_terrain_boundary import (
  TerrainBoundaryConfig,
  _strict_json_finite,
  _validate_config,
)
from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  CONTINUOUS_TRANSITION_KINDS,
  CURVE_PATCH_SIZE,
  CURVE_ROUTE_START_LOCAL,
  HIGH_DIFFICULTIES,
  TRANSITION_PATCH_SIZE,
  ContinuousBoundaryTerrainCfg,
  HighDifficultyTerrainCurveSubTerrainCfg,
  boundary_transition_difficulty_matrix,
  boundary_transition_metadata,
  continuous_boundary_coverage,
  continuous_feature_heightfield,
  difficulty_for_high_level,
  effective_high_terrain_parameters,
  make_boundary_transition_generator,
  make_high_difficulty_curve_generator,
  reject_curved_transition,
  validate_straight_transition_footprint,
)
from src.tasks.velocity.evaluation.terrain_curved_routes import (
  MEDIUM_DIFFICULTY,
  TERRAIN_CURVE_KINDS,
  effective_terrain_parameters,
  validate_route_footprint,
)


def _compile(kind: str, difficulty: float) -> tuple[mujoco.MjModel, np.ndarray]:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg = ContinuousBoundaryTerrainCfg(kind=kind, proportion=1.0)  # type: ignore[arg-type]
  output = cfg.function(difficulty, spec, np.random.default_rng(999))
  return spec.compile(), np.asarray(output.origin)


def _ray_height(model: mujoco.MjModel, x: float, y: float = 2.0) -> float:
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geom_id = np.array([-1], dtype=np.int32)
  distance = mujoco.mj_ray(
    model,
    data,
    np.array([x, y, 3.0]),
    np.array([0.0, 0.0, -1.0]),
    np.ones(6, dtype=np.uint8),
    1,
    -1,
    geom_id,
  )
  if distance < 0:
    raise AssertionError(f"ray missed terrain at {(x, y)}")
  return 3.0 - float(distance)


class HighDifficultyCurveTest(unittest.TestCase):
  def test_high_mapping_and_primitive_semantics_are_truthful(self) -> None:
    self.assertEqual(difficulty_for_high_level(0), ("high", 0.8))
    self.assertEqual(difficulty_for_high_level(1), ("extreme", 1.0))
    with self.assertRaises(ValueError):
      difficulty_for_high_level(2)
    for level in (0, 1):
      slope = effective_high_terrain_parameters("slope_up", level)
      obstacle = effective_high_terrain_parameters("discrete_obstacle", level)
      self.assertGreater(
        slope["slope_gradient"], MEDIUM_DIFFICULTY * 0.4  # type: ignore[operator]
      )
      self.assertGreater(
        obstacle["obstacle_height"],  # type: ignore[arg-type]
        effective_terrain_parameters("discrete_obstacle", 1)["obstacle_height"],
      )
    rough = effective_high_terrain_parameters("random_rough", 1)
    self.assertFalse(rough["difficulty_affects_geometry"])
    self.assertTrue(rough["difficulty_invariant"])

  def test_full_high_arc_and_s_matrix_has_safe_corridor_and_scan(self) -> None:
    for route_kind in ("arc", "s_curve"):
      for radius in (2.5, 4.0):
        for turn_sign in (-1, 1):
          footprint = validate_route_footprint(route_kind, radius, turn_sign)
          self.assertTrue(footprint.corridor_inside_patch)
          self.assertTrue(footprint.scan_footprint_inside_patch)
          self.assertGreater(footprint.scan_boundary_margin, 1.0)

  def test_high_generator_has_exact_rows_and_compiling_primitives(self) -> None:
    generator = make_high_difficulty_curve_generator(42)
    self.assertEqual(generator.size, CURVE_PATCH_SIZE)
    self.assertEqual(generator.num_rows, len(HIGH_DIFFICULTIES))
    self.assertEqual(tuple(generator.sub_terrains), TERRAIN_CURVE_KINDS)
    for kind in TERRAIN_CURVE_KINDS:
      for incoming in (0.1, 0.9):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        cfg = HighDifficultyTerrainCurveSubTerrainCfg(
          kind=kind, proportion=1.0
        )
        output = cfg.function(incoming, spec, np.random.default_rng(42))
        np.testing.assert_allclose(output.origin[:2], CURVE_ROUTE_START_LOCAL)
        self.assertTrue(np.isfinite(output.origin).all())
        self.assertEqual(len(output.geometries), 1)


class StraightContinuousTransitionTest(unittest.TestCase):
  def test_footprint_reports_route_corridor_scan_bounds_and_margins(self) -> None:
    footprint = validate_straight_transition_footprint()
    self.assertEqual(footprint.patch_size, TRANSITION_PATCH_SIZE)
    self.assertEqual(footprint.route_bounds, ((1.0, 7.0), (2.0, 2.0)))
    self.assertTrue(footprint.route_inside_patch)
    self.assertTrue(footprint.corridor_inside_patch)
    self.assertTrue(footprint.scan_footprint_inside_patch)
    self.assertAlmostEqual(footprint.scan_boundary_margin, 0.2)
    _strict_json_finite(dataclasses.asdict(footprint))

  def test_all_six_cases_have_truthful_metadata_and_flat_scan_endpoints(self) -> None:
    for kind in CONTINUOUS_TRANSITION_KINDS:
      with self.subTest(kind=kind):
        metadata = boundary_transition_metadata(kind, 0.9)
        self.assertEqual(metadata.kind, kind)
        self.assertEqual(metadata.patch_size, TRANSITION_PATCH_SIZE)
        self.assertEqual(metadata.start_x, 1.0)
        self.assertEqual(metadata.end_x, 7.0)
        model, origin = _compile(kind, 0.9)
        self.assertAlmostEqual(origin[0], 1.0)
        self.assertAlmostEqual(origin[1], 2.0)
        self.assertAlmostEqual(origin[2], metadata.entry_surface_z)
        for x in (0.2, 1.0, 1.8):
          self.assertAlmostEqual(
            _ray_height(model, x), metadata.entry_surface_z, delta=0.011
          )
        for x in (5.2, 6.2, 7.0, 7.8):
          self.assertAlmostEqual(
            _ray_height(model, x), metadata.exit_surface_z, delta=0.011
          )

  def test_rough_and_obstacle_are_single_arrays_with_zero_junctions(self) -> None:
    for kind in ("random_rough", "discrete_obstacle"):
      x, y, heights = continuous_feature_heightfield(kind, 0.9, seed=42)
      self.assertEqual(x.shape, (81,))
      self.assertEqual(y.shape, (41,))
      self.assertEqual(heights.shape, (41, 81))
      self.assertTrue(np.isfinite(heights).all())
      np.testing.assert_allclose(heights[:, x <= 2.0], 0.0)
      np.testing.assert_allclose(heights[:, x >= 4.4], 0.0)
      self.assertGreater(np.max(np.abs(heights[:, (x > 2.1) & (x < 4.3)])), 0.0)
      x2, y2, heights2 = continuous_feature_heightfield(kind, 0.9, seed=42)
      np.testing.assert_array_equal(x, x2)
      np.testing.assert_array_equal(y, y2)
      np.testing.assert_array_equal(heights, heights2)

  def test_difficulty_matrix_and_generator_column_order_agree(self) -> None:
    matrix = boundary_transition_difficulty_matrix(42)
    generator = make_boundary_transition_generator(42)
    self.assertEqual(matrix.shape, (10, 6))
    self.assertEqual(generator.num_rows, 10)
    self.assertEqual(generator.num_cols, 6)
    self.assertEqual(len(generator.sub_terrains), 6)
    self.assertTrue(np.all(matrix >= 0.0))
    self.assertTrue(np.all(matrix <= 1.0))
    self.assertTrue(np.all(np.diff(matrix, axis=0) > 0.0))

  def test_full_six_column_generator_compiles_with_expected_entry_heights(self) -> None:
    matrix = boundary_transition_difficulty_matrix(42)
    generator = TerrainGenerator(make_boundary_transition_generator(42))
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    self.assertGreater(model.ngeom, 0)
    self.assertEqual(generator.terrain_origins.shape, (10, 6, 3))
    self.assertTrue(np.isfinite(generator.terrain_origins).all())
    for column, kind in enumerate(CONTINUOUS_TRANSITION_KINDS):
      for row in (0, 7, 9):
        expected = boundary_transition_metadata(
          kind, float(matrix[row, column])
        ).entry_surface_z
        self.assertAlmostEqual(generator.terrain_origins[row, column, 2], expected)

  def test_coverage_and_curved_rejections_cannot_overclaim(self) -> None:
    coverage = continuous_boundary_coverage()
    self.assertTrue(coverage["continuous_intra_patch_transitions"])
    self.assertFalse(coverage["continuous_inter_patch_transitions"])
    self.assertFalse(coverage["curve_transitions"])
    reject_curved_transition("stairs_up", "straight")
    with self.assertRaisesRegex(ValueError, "stairs curves are unsupported"):
      reject_curved_transition("stairs_up", "arc")
    with self.assertRaisesRegex(ValueError, "8x4"):
      reject_curved_transition("random_rough", "s_curve")


class BoundaryCliContractTest(unittest.TestCase):
  def test_high_and_continuous_configs_validate_without_gpu(self) -> None:
    _validate_config(TerrainBoundaryConfig(checkpoint="unused"))
    _validate_config(
      TerrainBoundaryConfig(
        checkpoint="unused", suite="continuous_straight", route_kind="straight"
      )
    )
    with self.assertRaisesRegex(ValueError, "stairs curves"):
      _validate_config(
        TerrainBoundaryConfig(
          checkpoint="unused",
          suite="continuous_straight",
          route_kind="arc",
          transition_cases=("stairs_up",),
        )
      )

  def test_strict_json_rejects_nested_nonfinite_values(self) -> None:
    _strict_json_finite({"ok": [1.0, None, {"value": 2.0}]})
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.assertRaises(ValueError):
        _strict_json_finite({"bad": [value]})


if __name__ == "__main__":
  unittest.main()
