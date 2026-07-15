"""CPU-only tests for the evaluation-only continuous terrain profiles."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from mjlab.terrains import TerrainGenerator
from src.tasks.velocity.evaluation.route_terrains import (
  FEATURE_END_X,
  FEATURE_START_X,
  PATCH_SIZE,
  ROUTE_END_X,
  ROUTE_LENGTH,
  ROUTE_START_X,
  STEP_COUNT,
  STEP_WIDTH,
  TERRAIN_SCAN_HALF_X,
  TERRAIN_SCAN_SIZE_X,
  TERRAIN_KEY_TO_KIND,
  TERRAIN_KIND_TO_KEY,
  ContinuousRouteTerrainCfg,
  continuous_route_difficulty_matrix,
  make_continuous_route_terrain_generator,
  route_surface_height,
  route_terrain_bounds,
  route_terrain_metadata,
  validate_route_terrain_parameters,
  STAIRS_UP_DOWN_ASCENT_X,
  STAIRS_UP_DOWN_DESCENT_END_X,
  STAIRS_UP_DOWN_END_X,
  STAIRS_UP_DOWN_START_X,
  STAIRS_UP_DOWN_TOP_END_X,
  StairsUpDownDemoTerrainCfg,
  stairs_up_down_surface_height,
)


class RouteTerrainProfileTest(unittest.TestCase):
  def test_fixed_bounds_and_route_length(self) -> None:
    bounds = route_terrain_bounds()
    self.assertEqual(bounds["patch_x"], (0.0, 8.0))
    self.assertEqual(bounds["patch_y"], (0.0, 4.0))
    self.assertEqual(bounds["route_x"], (1.0, 7.0))
    self.assertEqual(bounds["feature_x"], (FEATURE_START_X, FEATURE_END_X))
    self.assertEqual(PATCH_SIZE, (8.0, 4.0))
    self.assertAlmostEqual(ROUTE_LENGTH, 6.0)
    self.assertAlmostEqual(TERRAIN_SCAN_SIZE_X, 1.6)
    self.assertAlmostEqual(TERRAIN_SCAN_HALF_X, 0.8)
    np.testing.assert_allclose(bounds["start_scan_x"], (0.2, 1.8))
    np.testing.assert_allclose(bounds["end_scan_x"], (6.2, 7.8))
    self.assertGreaterEqual(bounds["start_scan_x"][0], bounds["patch_x"][0])
    self.assertLessEqual(bounds["end_scan_x"][1], bounds["patch_x"][1])
    self.assertEqual(STEP_COUNT, 8)
    self.assertAlmostEqual(STEP_COUNT * STEP_WIDTH, 2.4)

  def test_metadata_direction_and_difficulty_interpolation(self) -> None:
    stairs = route_terrain_metadata("stairs_up", 0.5)
    self.assertEqual(stairs.terrain_key, "pyramid_stairs")
    self.assertEqual(stairs.family, "stairs")
    self.assertEqual(stairs.direction, "up")
    self.assertAlmostEqual(stairs.step_height, 0.07)
    self.assertAlmostEqual(stairs.entry_surface_z, 0.0)
    self.assertAlmostEqual(stairs.exit_surface_z, 0.56)

    slope = route_terrain_metadata("slope_down", 0.5)
    self.assertEqual(slope.terrain_key, "hf_pyramid_slope_inv")
    self.assertEqual(slope.family, "slope")
    self.assertEqual(slope.direction, "down")
    self.assertAlmostEqual(slope.slope, 0.2)
    self.assertAlmostEqual(slope.entry_surface_z, 0.48)
    self.assertAlmostEqual(slope.exit_surface_z, 0.0)

  def test_profiles_have_flat_entry_and_exit(self) -> None:
    for kind in TERRAIN_KIND_TO_KEY:
      metadata = route_terrain_metadata(kind, 1.0)
      entry = route_surface_height(kind, 1.0, ROUTE_START_X)
      exit_height = route_surface_height(kind, 1.0, ROUTE_END_X)
      self.assertAlmostEqual(entry, metadata.entry_surface_z)
      self.assertAlmostEqual(exit_height, metadata.exit_surface_z)
      self.assertAlmostEqual(
        route_surface_height(kind, 1.0, FEATURE_START_X - 0.01),
        metadata.entry_surface_z,
      )
      self.assertAlmostEqual(
        route_surface_height(kind, 1.0, FEATURE_END_X + 0.01),
        metadata.exit_surface_z,
      )

      start_scan = np.linspace(
        ROUTE_START_X - TERRAIN_SCAN_HALF_X,
        ROUTE_START_X + TERRAIN_SCAN_HALF_X,
        9,
      )
      end_scan = np.linspace(
        ROUTE_END_X - TERRAIN_SCAN_HALF_X,
        ROUTE_END_X + TERRAIN_SCAN_HALF_X,
        9,
      )
      np.testing.assert_allclose(
        route_surface_height(kind, 1.0, start_scan),
        metadata.entry_surface_z,
      )
      np.testing.assert_allclose(
        route_surface_height(kind, 1.0, end_scan),
        metadata.exit_surface_z,
      )

  def test_profile_is_monotonic_in_declared_direction(self) -> None:
    x = np.linspace(FEATURE_START_X, FEATURE_END_X, 801, endpoint=False)
    for kind in TERRAIN_KIND_TO_KEY:
      heights = np.asarray(route_surface_height(kind, 0.8, x))
      difference = np.diff(heights)
      if kind.endswith("_up"):
        self.assertGreaterEqual(float(difference.min()), -1.0e-9)
      else:
        self.assertLessEqual(float(difference.max()), 1.0e-9)

  def test_invalid_kind_difficulty_and_bounds_are_rejected(self) -> None:
    with self.assertRaises(ValueError):
      validate_route_terrain_parameters("stairs", 0.5)
    with self.assertRaises(ValueError):
      validate_route_terrain_parameters("stairs_up", -0.01)
    with self.assertRaises(ValueError):
      validate_route_terrain_parameters("stairs_up", 1.01)
    with self.assertRaises(ValueError):
      route_surface_height("stairs_up", 0.5, -1.0)
    with self.assertRaises(ValueError):
      route_surface_height("stairs_up", 0.5, 8.01)

  def test_difficulty_matrix_reproduces_generator_column_major_sampling(self) -> None:
    matrix = continuous_route_difficulty_matrix(7)
    self.assertEqual(matrix.shape, (10, 4))
    self.assertTrue(np.all(matrix[1:] >= matrix[:-1]))
    self.assertTrue(np.all((matrix >= 0.0) & (matrix <= 1.0)))
    with self.assertRaises(TypeError):
      continuous_route_difficulty_matrix(7.5)  # type: ignore[arg-type]


def _ray_height(model: mujoco.MjModel, x: float, y: float = 2.0) -> float:
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geomid = np.array([-1], dtype=np.int32)
  distance = mujoco.mj_ray(
    model,
    data,
    np.array([x, y, 3.0]),
    np.array([0.0, 0.0, -1.0]),
    np.ones(6, dtype=np.uint8),
    1,
    -1,
    geomid,
  )
  if distance < 0.0:
    raise AssertionError(f"ray missed terrain at x={x}")
  return 3.0 - float(distance)


class RouteTerrainSpecTest(unittest.TestCase):
  def _compile_profile(self, kind: str, difficulty: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = ContinuousRouteTerrainCfg(kind=kind)  # type: ignore[arg-type]
    cfg.function(difficulty, spec, np.random.default_rng(0))
    return spec.compile()

  def test_each_profile_compiles_and_entry_origin_is_surface(self) -> None:
    for kind in TERRAIN_KIND_TO_KEY:
      with self.subTest(kind=kind):
        model = self._compile_profile(kind, 0.75)
        metadata = route_terrain_metadata(kind, 0.75)
        measured = _ray_height(model, ROUTE_START_X)
        self.assertAlmostEqual(measured, metadata.entry_surface_z, delta=0.06)

  def test_generated_geometry_matches_profile_at_safe_samples(self) -> None:
    samples = (
      ROUTE_START_X - TERRAIN_SCAN_HALF_X,
      ROUTE_START_X + TERRAIN_SCAN_HALF_X,
      2.15,
      2.75,
      3.65,
      4.55,
      ROUTE_END_X - TERRAIN_SCAN_HALF_X,
      ROUTE_END_X + TERRAIN_SCAN_HALF_X,
    )
    for kind in TERRAIN_KIND_TO_KEY:
      with self.subTest(kind=kind):
        model = self._compile_profile(kind, 0.6)
        for x in samples:
          measured = _ray_height(model, x)
          expected = route_surface_height(kind, 0.6, x)
          self.assertAlmostEqual(measured, expected, delta=0.06)

  def test_generator_has_v7_focus_keys_and_expected_shape(self) -> None:
    cfg = make_continuous_route_terrain_generator(seed=7)
    self.assertEqual(cfg.size, PATCH_SIZE)
    self.assertEqual((cfg.num_rows, cfg.num_cols), (10, 4))
    self.assertTrue(cfg.curriculum)
    self.assertEqual(set(cfg.sub_terrains), set(TERRAIN_KEY_TO_KIND))
    generator = TerrainGenerator(cfg, device="cpu")
    spec = mujoco.MjSpec()
    generator.compile(spec)
    self.assertEqual(generator.terrain_origins.shape, (10, 4, 3))
    difficulties = continuous_route_difficulty_matrix(7)
    for row in range(10):
      expected = route_terrain_metadata("stairs_down", float(difficulties[row, 1]))
      self.assertAlmostEqual(
        float(generator.terrain_origins[row, 1, 2]), expected.entry_surface_z
      )
    spec.compile()

  def test_stairs_up_down_demo_has_two_full_staircases(self) -> None:
    difficulty = 0.5
    step_height = 0.07
    self.assertAlmostEqual(
      stairs_up_down_surface_height(difficulty, STAIRS_UP_DOWN_START_X), 0.0
    )
    self.assertAlmostEqual(
      stairs_up_down_surface_height(difficulty, STAIRS_UP_DOWN_ASCENT_X[1]),
      STEP_COUNT * step_height,
    )
    self.assertAlmostEqual(
      stairs_up_down_surface_height(difficulty, STAIRS_UP_DOWN_TOP_END_X - 0.01),
      STEP_COUNT * step_height,
    )
    self.assertAlmostEqual(
      stairs_up_down_surface_height(
        difficulty, STAIRS_UP_DOWN_DESCENT_END_X + 0.01
      ),
      0.0,
    )
    self.assertAlmostEqual(
      stairs_up_down_surface_height(difficulty, STAIRS_UP_DOWN_END_X), 0.0
    )

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = StairsUpDownDemoTerrainCfg().function(
      difficulty, spec, np.random.default_rng(0)
    )
    model = spec.compile()
    self.assertEqual(output.origin.tolist(), [1.0, 2.0, 0.0])
    for x in (1.0, 2.15, 4.5, 7.75, 10.1, 11.0):
      self.assertAlmostEqual(
        _ray_height(model, x),
        stairs_up_down_surface_height(difficulty, x),
        delta=1.0e-5,
      )


if __name__ == "__main__":
  unittest.main()
