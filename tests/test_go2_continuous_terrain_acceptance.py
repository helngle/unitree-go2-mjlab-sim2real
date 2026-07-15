"""Independent acceptance tests for continuous intra-patch route terrains."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch

from src.tasks.velocity.evaluation.route_terrains import (
  FEATURE_END_X,
  FEATURE_START_X,
  PATCH_SIZE,
  ROUTE_END_X,
  ROUTE_LENGTH,
  ROUTE_START_X,
  STEP_COUNT,
  STEP_WIDTH,
  TERRAIN_KIND_TO_KEY,
  ContinuousRouteTerrainCfg,
  route_surface_height,
  route_terrain_bounds,
  route_terrain_metadata,
)
from src.tasks.velocity.evaluation.routes import update_attempt_status


KINDS = tuple(TERRAIN_KIND_TO_KEY)
SCAN_HALF_EXTENT_X = 0.8


def _compile_profile(kind: str, difficulty: float) -> tuple[mujoco.MjModel, np.ndarray]:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg = ContinuousRouteTerrainCfg(kind=kind)  # type: ignore[arg-type]
  output = cfg.function(difficulty, spec, np.random.default_rng(42))
  return spec.compile(), np.asarray(output.origin, dtype=np.float64)


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
  if distance < 0.0:
    raise AssertionError(f"ray missed terrain at x={x}, y={y}")
  return 3.0 - float(distance)


class ContinuousHeightProfileAcceptanceTest(unittest.TestCase):
  def test_route_feature_and_footprint_corridor_remain_inside_patch(self) -> None:
    bounds = route_terrain_bounds()
    self.assertEqual(bounds["patch_x"], (0.0, 8.0))
    self.assertEqual(bounds["patch_y"], (0.0, 4.0))
    self.assertEqual(bounds["route_x"], (1.0, 7.0))
    self.assertEqual(bounds["feature_x"], (2.0, 4.4))
    self.assertAlmostEqual(ROUTE_LENGTH, 6.0)

    self.assertGreaterEqual(ROUTE_START_X, SCAN_HALF_EXTENT_X)
    self.assertLessEqual(ROUTE_END_X, PATCH_SIZE[0] - SCAN_HALF_EXTENT_X)
    route_y = PATCH_SIZE[1] / 2.0
    self.assertGreaterEqual(route_y, SCAN_HALF_EXTENT_X)
    self.assertLessEqual(route_y, PATCH_SIZE[1] - SCAN_HALF_EXTENT_X)
    self.assertLess(ROUTE_START_X, FEATURE_START_X)
    self.assertLess(FEATURE_START_X, FEATURE_END_X)
    self.assertLess(FEATURE_END_X, ROUTE_END_X)

  def test_height_scan_footprint_stays_in_patch_and_on_endpoint_flats(self) -> None:
    start_scan = (
      ROUTE_START_X - SCAN_HALF_EXTENT_X,
      ROUTE_START_X + SCAN_HALF_EXTENT_X,
    )
    end_scan = (
      ROUTE_END_X - SCAN_HALF_EXTENT_X,
      ROUTE_END_X + SCAN_HALF_EXTENT_X,
    )

    self.assertGreaterEqual(start_scan[0], 0.0)
    self.assertLessEqual(start_scan[1], FEATURE_START_X)
    self.assertGreaterEqual(end_scan[0], FEATURE_END_X)
    self.assertLessEqual(end_scan[1], PATCH_SIZE[0])
    for kind in KINDS:
      with self.subTest(kind=kind):
        metadata = route_terrain_metadata(kind, 0.8)
        start_heights = route_surface_height(
          kind, 0.8, np.linspace(*start_scan, 33)
        )
        end_heights = route_surface_height(
          kind, 0.8, np.linspace(*end_scan, 33)
        )
        np.testing.assert_allclose(
          start_heights, metadata.entry_surface_z, rtol=0.0, atol=1.0e-10
        )
        np.testing.assert_allclose(
          end_heights, metadata.exit_surface_z, rtol=0.0, atol=1.0e-10
        )

  def test_stair_risers_are_bounded_quantized_and_directional(self) -> None:
    difficulty = 0.73
    for kind, sign in (("stairs_up", 1.0), ("stairs_down", -1.0)):
      with self.subTest(kind=kind):
        metadata = route_terrain_metadata(kind, difficulty)
        tread_centers = FEATURE_START_X + (np.arange(STEP_COUNT) + 0.5) * STEP_WIDTH
        samples = np.concatenate(
          ([FEATURE_START_X - 0.05], tread_centers, [FEATURE_END_X + 0.05])
        )
        heights = np.asarray(route_surface_height(kind, difficulty, samples))
        deltas = np.diff(heights)
        expected_riser = sign * metadata.step_height

        self.assertTrue(np.all(np.isfinite(heights)))
        self.assertTrue(
          np.all(
            np.isclose(deltas, 0.0, atol=1.0e-10)
            | np.isclose(deltas, expected_riser, atol=1.0e-10)
          )
        )
        self.assertEqual(np.count_nonzero(np.isclose(deltas, expected_riser)), STEP_COUNT)
        self.assertLessEqual(float(np.max(np.abs(deltas))), metadata.step_height + 1.0e-10)

    x = np.linspace(0.0, PATCH_SIZE[0], 1601)
    up = np.asarray(route_surface_height("stairs_up", difficulty, x))
    down = np.asarray(route_surface_height("stairs_down", difficulty, x))
    total_rise = route_terrain_metadata("stairs_up", difficulty).exit_surface_z
    np.testing.assert_allclose(up + down, total_rise, rtol=0.0, atol=1.0e-10)

  def test_slope_junctions_are_c0_and_gradient_has_correct_sign(self) -> None:
    difficulty = 0.65
    epsilon = 1.0e-7
    for kind, sign in (("slope_up", 1.0), ("slope_down", -1.0)):
      with self.subTest(kind=kind):
        metadata = route_terrain_metadata(kind, difficulty)
        junction_x = np.array(
          [
            FEATURE_START_X - epsilon,
            FEATURE_START_X,
            FEATURE_END_X - epsilon,
            FEATURE_END_X,
          ]
        )
        junction_z = np.asarray(route_surface_height(kind, difficulty, junction_x))
        self.assertAlmostEqual(junction_z[0], junction_z[1], delta=1.0e-6)
        self.assertAlmostEqual(junction_z[2], junction_z[3], delta=1.0e-6)

        feature_x = np.linspace(FEATURE_START_X, FEATURE_END_X, 25)
        feature_z = np.asarray(route_surface_height(kind, difficulty, feature_x))
        measured_gradient = np.diff(feature_z) / np.diff(feature_x)
        np.testing.assert_allclose(
          measured_gradient,
          sign * metadata.slope,
          rtol=0.0,
          atol=1.0e-10,
        )

    x = np.linspace(0.0, PATCH_SIZE[0], 321)
    up = np.asarray(route_surface_height("slope_up", difficulty, x))
    down = np.asarray(route_surface_height("slope_down", difficulty, x))
    total_rise = route_terrain_metadata("slope_up", difficulty).exit_surface_z
    np.testing.assert_allclose(up + down, total_rise, rtol=0.0, atol=1.0e-10)

  def test_all_cases_have_flat_approach_exit_and_declared_net_direction(self) -> None:
    difficulty = 0.8
    approach_x = np.linspace(ROUTE_START_X, FEATURE_START_X - 0.01, 21)
    exit_x = np.linspace(FEATURE_END_X + 0.01, ROUTE_END_X, 21)
    for kind in KINDS:
      with self.subTest(kind=kind):
        metadata = route_terrain_metadata(kind, difficulty)
        approach_z = np.asarray(route_surface_height(kind, difficulty, approach_x))
        exit_z = np.asarray(route_surface_height(kind, difficulty, exit_x))
        np.testing.assert_allclose(approach_z, metadata.entry_surface_z, atol=1.0e-10)
        np.testing.assert_allclose(exit_z, metadata.exit_surface_z, atol=1.0e-10)
        net_height = metadata.exit_surface_z - metadata.entry_surface_z
        if metadata.direction == "up":
          self.assertGreater(net_height, 0.0)
        else:
          self.assertLess(net_height, 0.0)


class ContinuousTerrainSpecAcceptanceTest(unittest.TestCase):
  def test_mujoco_surface_matches_profile_and_entry_origin(self) -> None:
    difficulty = 0.6
    safe_samples = (1.0, 1.5, 2.15, 2.75, 3.65, 4.55, 6.5, 7.0)
    for kind in KINDS:
      with self.subTest(kind=kind):
        model, origin = _compile_profile(kind, difficulty)
        metadata = route_terrain_metadata(kind, difficulty)
        self.assertAlmostEqual(origin[0], ROUTE_START_X, delta=1.0e-10)
        self.assertAlmostEqual(origin[1], PATCH_SIZE[1] / 2.0, delta=1.0e-10)
        self.assertAlmostEqual(origin[2], metadata.entry_surface_z, delta=1.0e-10)
        self.assertAlmostEqual(
          _ray_height(model, ROUTE_START_X), metadata.entry_surface_z, delta=0.01
        )
        for x in safe_samples:
          self.assertAlmostEqual(
            _ray_height(model, x),
            route_surface_height(kind, difficulty, x),
            delta=0.01,
          )

  def test_entry_root_clearance_and_xyz_relocation_are_invariant(self) -> None:
    nominal_clearance = 0.32
    old_patch_translation = np.array([12.0, -8.0, 0.56])
    new_patch_translation = np.array([-3.0, 21.0, 1.14])
    relocation = new_patch_translation - old_patch_translation
    for kind in KINDS:
      with self.subTest(kind=kind):
        _, local_origin = _compile_profile(kind, 0.7)
        metadata = route_terrain_metadata(kind, 0.7)
        local_endpoint = np.array(
          [ROUTE_END_X, PATCH_SIZE[1] / 2.0, metadata.exit_surface_z]
        )
        start_before = old_patch_translation + local_origin
        endpoint_before = old_patch_translation + local_endpoint
        root_before = start_before + np.array([0.0, 0.0, nominal_clearance])
        start_after = start_before + relocation
        endpoint_after = endpoint_before + relocation
        root_after = root_before + relocation

        np.testing.assert_allclose(
          start_after - new_patch_translation,
          local_origin,
          rtol=0.0,
          atol=1.0e-12,
        )
        np.testing.assert_allclose(
          endpoint_after - new_patch_translation,
          local_endpoint,
          rtol=0.0,
          atol=1.0e-12,
        )
        np.testing.assert_allclose(
          root_after - start_after,
          np.array([0.0, 0.0, nominal_clearance]),
          rtol=0.0,
          atol=1.0e-12,
        )


class ContinuousAttemptFreezeAcceptanceTest(unittest.TestCase):
  def test_failure_precedes_completion_and_reset_episode_cannot_leak_success(self) -> None:
    first = update_attempt_status(
      active=torch.tensor([True]),
      progress=torch.tensor([ROUTE_LENGTH]),
      cross_track=torch.zeros(1),
      heading_error=torch.zeros(1),
      failure_mask=torch.tensor([True]),
      route_length=ROUTE_LENGTH,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    self.assertTrue(bool(first.failed_now[0]))
    self.assertFalse(bool(first.completed_now[0]))
    self.assertFalse(bool(first.active[0]))

    frozen_progress = torch.tensor([3.2])
    sample_count = torch.tensor([17.0])
    second = update_attempt_status(
      active=first.active,
      progress=torch.tensor([ROUTE_LENGTH + 10.0]),
      cross_track=torch.zeros(1),
      heading_error=torch.zeros(1),
      failure_mask=torch.tensor([False]),
      route_length=ROUTE_LENGTH,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    frozen_progress = torch.where(second.sample_mask, torch.tensor([99.0]), frozen_progress)
    sample_count += second.sample_mask.float()

    self.assertFalse(bool(second.sample_mask[0]))
    self.assertFalse(bool(second.completed_now[0]))
    self.assertFalse(bool(second.active[0]))
    torch.testing.assert_close(frozen_progress, torch.tensor([3.2]))
    torch.testing.assert_close(sample_count, torch.tensor([17.0]))


if __name__ == "__main__":
  unittest.main()
