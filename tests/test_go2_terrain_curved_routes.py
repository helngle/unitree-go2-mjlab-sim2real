"""CPU geometry contracts for evaluation-only terrain curves."""

from __future__ import annotations

import dataclasses
import unittest

import mujoco
import numpy as np
import torch

from scripts.evaluate_go2_terrain_curves import (
  TerrainCurvedRouteConfig,
  _contact_termination_summary,
  _scenarios,
  _strict_json_finite,
  _validate_config,
)

from src.tasks.velocity.evaluation.terrain_curved_routes import (
  BASE_OBSTACLE_COUNT,
  BASE_PATCH_SIZE,
  LOW_DIFFICULTY,
  MEDIUM_DIFFICULTY,
  PATCH_SIZE,
  ROUTE_START_LOCAL,
  SCALED_OBSTACLE_COUNT,
  TERRAIN_CURVE_KINDS,
  TERRAIN_KIND_TO_TYPE,
  TerrainCurveSubTerrainCfg,
  compute_route_footprint,
  continuous_transition_coverage,
  difficulty_for_level,
  effective_terrain_parameters,
  make_terrain_curve_generator,
  relocate_root_pose,
  slope_direction_is_compatible,
  validate_route_footprint,
)


class TerrainCurveFootprintTest(unittest.TestCase):
  def test_full_first_stage_matrix_fits_patch_and_scan(self) -> None:
    for route_kind in ("arc", "s_curve"):
      for radius in (2.5, 4.0):
        for turn_sign in (-1, 1):
          with self.subTest(
            route_kind=route_kind, radius=radius, turn_sign=turn_sign
          ):
            footprint = validate_route_footprint(
              route_kind, radius, turn_sign
            )
            self.assertTrue(footprint.corridor_inside_patch)
            self.assertTrue(footprint.scan_footprint_inside_patch)
            self.assertGreater(footprint.corridor_boundary_margin, 1.0)
            self.assertGreater(footprint.scan_boundary_margin, 1.0)

  def test_left_right_bounds_are_mirrored_about_patch_center(self) -> None:
    center_y = ROUTE_START_LOCAL[1]
    for route_kind in ("arc", "s_curve"):
      left = compute_route_footprint(route_kind, 4.0, 1)
      right = compute_route_footprint(route_kind, 4.0, -1)
      self.assertEqual(left.centerline_bounds[0], right.centerline_bounds[0])
      self.assertAlmostEqual(
        left.centerline_bounds[1][0],
        2.0 * center_y - right.centerline_bounds[1][1],
      )
      self.assertAlmostEqual(
        left.centerline_bounds[1][1],
        2.0 * center_y - right.centerline_bounds[1][0],
      )
      self.assertAlmostEqual(
        left.scan_boundary_margin, right.scan_boundary_margin
      )

  def test_old_continuous_patch_is_rejected_not_scaled(self) -> None:
    old_patch = (8.0, 4.0)
    old_start = (1.0, 2.0)
    footprint = compute_route_footprint(
      "s_curve",
      4.0,
      1,
      patch_size=old_patch,
      route_start_local=old_start,
    )
    self.assertFalse(footprint.corridor_inside_patch)
    self.assertFalse(footprint.scan_footprint_inside_patch)
    with self.assertRaisesRegex(ValueError, "route corridor leaves patch"):
      validate_route_footprint(
        "s_curve",
        4.0,
        1,
        patch_size=old_patch,
        route_start_local=old_start,
      )

  def test_pyramid_slope_direction_matches_outward_routes(self) -> None:
    for route_kind in ("arc", "s_curve"):
      for radius in (2.5, 4.0):
        for turn_sign in (-1, 1):
          self.assertTrue(
            slope_direction_is_compatible(route_kind, radius, turn_sign)
          )

  def test_invalid_footprint_inputs_raise(self) -> None:
    invalid = (
      ("unknown", 2.5, 1),
      ("arc", 0.0, 1),
      ("arc", 2.5, 0),
    )
    for args in invalid:
      with self.subTest(args=args), self.assertRaises(ValueError):
        compute_route_footprint(*args)
    with self.assertRaises(ValueError):
      compute_route_footprint("arc", 2.5, 1, corridor_half_width=0.0)


class TerrainCurveGeneratorTest(unittest.TestCase):
  def test_generator_has_two_levels_four_columns_and_stable_order(self) -> None:
    cfg = make_terrain_curve_generator(seed=42)
    self.assertEqual(cfg.size, PATCH_SIZE)
    self.assertEqual(cfg.num_rows, 2)
    self.assertEqual(cfg.num_cols, 4)
    self.assertTrue(cfg.curriculum)
    self.assertEqual(tuple(cfg.sub_terrains), TERRAIN_CURVE_KINDS)
    self.assertEqual(
      TERRAIN_KIND_TO_TYPE,
      {kind: index for index, kind in enumerate(TERRAIN_CURVE_KINDS)},
    )

  def test_low_medium_mapping_and_effective_parameters(self) -> None:
    self.assertEqual(difficulty_for_level(0), ("low", LOW_DIFFICULTY))
    self.assertEqual(difficulty_for_level(1), ("medium", MEDIUM_DIFFICULTY))
    with self.assertRaises(ValueError):
      difficulty_for_level(2)
    self.assertLess(
      effective_terrain_parameters("slope_up", 0)["slope_gradient"],
      effective_terrain_parameters("slope_up", 1)["slope_gradient"],
    )
    self.assertLess(
      effective_terrain_parameters("discrete_obstacle", 0)["obstacle_height"],
      effective_terrain_parameters("discrete_obstacle", 1)["obstacle_height"],
    )
    rough = effective_terrain_parameters("random_rough", 0)
    self.assertFalse(rough["difficulty_affects_geometry"])
    self.assertIn("ignores difficulty", rough["difficulty_reason"])

  def test_obstacle_count_preserves_original_area_density(self) -> None:
    original_density = BASE_OBSTACLE_COUNT / (
      BASE_PATCH_SIZE[0] * BASE_PATCH_SIZE[1]
    )
    scaled_density = SCALED_OBSTACLE_COUNT / (PATCH_SIZE[0] * PATCH_SIZE[1])
    self.assertAlmostEqual(original_density, scaled_density, delta=0.002)

  def test_each_primitive_compiles_with_center_origin(self) -> None:
    for kind in TERRAIN_CURVE_KINDS:
      for incoming, expected in ((0.1, LOW_DIFFICULTY), (0.9, MEDIUM_DIFFICULTY)):
        with self.subTest(kind=kind, incoming=incoming):
          spec = mujoco.MjSpec()
          spec.worldbody.add_body(name="terrain")
          cfg = TerrainCurveSubTerrainCfg(kind=kind, proportion=1.0)
          output = cfg.function(
            incoming, spec, np.random.default_rng(42)
          )
          np.testing.assert_allclose(
            output.origin[:2], ROUTE_START_LOCAL, rtol=0.0, atol=1.0e-12
          )
          self.assertTrue(np.isfinite(output.origin).all())
          self.assertEqual(len(output.geometries), 1)
          self.assertIsNotNone(output.geometries[0].geom)
          self.assertIsNotNone(output.geometries[0].hfield)
          parameters = effective_terrain_parameters(kind, int(incoming >= 0.5))
          self.assertEqual(parameters["difficulty"], expected)

  def test_wrong_patch_size_is_rejected(self) -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = TerrainCurveSubTerrainCfg(
      kind="random_rough", proportion=1.0, size=(8.0, 8.0)
    )
    with self.assertRaisesRegex(ValueError, "patch size"):
      cfg.function(0.2, spec, np.random.default_rng(42))


class TerrainCurveCoverageTest(unittest.TestCase):
  def test_continuous_transition_is_explicitly_not_covered(self) -> None:
    coverage = continuous_transition_coverage()
    self.assertFalse(coverage["implemented"])
    self.assertFalse(coverage["coverage"])
    self.assertIn("8x4", coverage["reason"])
    self.assertIn("intentionally rejected", coverage["reason"])

  def test_footprint_metadata_is_serializable(self) -> None:
    footprint = validate_route_footprint("s_curve", 4.0, 1)
    data = dataclasses.asdict(footprint)
    self.assertEqual(data["patch_size"], PATCH_SIZE)
    self.assertGreater(data["scan_boundary_margin"], 1.0)

  def test_relocation_preserves_patch_relative_xyz_and_orientation(self) -> None:
    root = torch.tensor(
      [
        [1.2, -2.3, 0.7, 1.0, 0.0, 0.0, 0.0],
        [-4.0, 3.5, 1.2, 0.7, 0.0, 0.0, 0.7],
      ]
    )
    old = torch.tensor([[1.0, -2.0, 0.1], [-5.0, 3.0, 0.4]])
    new = torch.tensor([[20.0, 9.0, 1.1], [4.0, -8.0, -0.2]])
    relocated, error = relocate_root_pose(root, old, new)
    torch.testing.assert_close(relocated[:, :3] - new, root[:, :3] - old)
    torch.testing.assert_close(relocated[:, 3:7], root[:, 3:7])
    self.assertLess(error, 1.0e-6)

  def test_relocation_rejects_bad_shapes(self) -> None:
    with self.assertRaises(ValueError):
      relocate_root_pose(
        torch.zeros((2, 6)), torch.zeros((2, 3)), torch.zeros((2, 3))
      )


class TerrainCurveEvaluatorContractTest(unittest.TestCase):
  def test_first_stage_scenario_matrix(self) -> None:
    cfg = TerrainCurvedRouteConfig(checkpoint="unused")
    _validate_config(cfg)
    scenarios = _scenarios(cfg)
    self.assertEqual(len(scenarios), 4 * 2 * 2 * 2 * 2)
    self.assertEqual(
      {scenario["terrain_kind"] for scenario in scenarios},
      set(TERRAIN_CURVE_KINDS),
    )
    self.assertEqual({scenario["level"] for scenario in scenarios}, {0, 1})

  def test_invalid_terrain_kind_level_and_old_patch_equivalent_fail(self) -> None:
    with self.assertRaises(ValueError):
      _validate_config(
        TerrainCurvedRouteConfig(
          checkpoint="unused", terrain_kinds=("stairs",)
        )
      )
    with self.assertRaises(ValueError):
      _validate_config(
        TerrainCurvedRouteConfig(checkpoint="unused", terrain_levels=(2,))
      )

  def test_command_tape_still_requires_scheduled_horizon(self) -> None:
    with self.assertRaisesRegex(ValueError, "command_tape requires steps"):
      _validate_config(
        TerrainCurvedRouteConfig(
          checkpoint="unused",
          route_kind="s_curve",
          mode="command_tape",
          radii=(4.0,),
          speeds=(0.3,),
          turn_signs=(1,),
          steps=100,
        )
      )

  def test_contact_termination_summary_is_explicit_about_scope(self) -> None:
    result = _contact_termination_summary(
      {
        "fell_over": 1.0,
        "illegal_base_contact": 2.0,
        "illegal_upper_leg_contact": 3.0,
        "illegal_calf_contact": 4.0,
      }
    )
    self.assertEqual(result["base"]["termination_count"], 2.0)
    self.assertEqual(result["calf"]["termination_count"], 4.0)
    self.assertIn("non-terminating", result["upper_leg"]["scope"])

  def test_strict_json_rejects_nested_nan_and_inf(self) -> None:
    _strict_json_finite({"ok": [1.0, None, {"value": 2.0}]})
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(value=value), self.assertRaises(ValueError):
        _strict_json_finite({"nested": [0.0, value]})


if __name__ == "__main__":
  unittest.main()
