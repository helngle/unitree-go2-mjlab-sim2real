"""Independent CPU acceptance contracts for terrain boundary evaluation."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import unittest

import numpy as np
import torch

import scripts.evaluate_go2_terrain_curves as terrain_evaluator
from src.tasks.velocity.config.go2 import env_cfgs
from src.tasks.velocity.evaluation.routes import update_attempt_status
from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  CONTINUOUS_TRANSITION_KINDS,
  HIGH_DIFFICULTIES,
  HIGH_DIFFICULTY_LABELS,
  boundary_transition_difficulty_matrix,
  boundary_transition_metadata,
  continuous_boundary_coverage,
  continuous_feature_heightfield,
  difficulty_for_high_level,
  effective_high_terrain_parameters,
  make_high_difficulty_curve_generator,
  reject_curved_transition,
  validate_straight_transition_footprint,
)
from src.tasks.velocity.evaluation.terrain_curved_routes import (
  MEDIUM_DIFFICULTY,
  TERRAIN_CURVE_KINDS,
  effective_terrain_parameters,
  relocate_root_pose,
  validate_route_footprint,
)
from src.tasks.velocity.evaluation.route_terrains import (
  FEATURE_END_X,
  FEATURE_START_X,
  GENERATOR_NUM_ROWS,
  PATCH_SIZE as TRANSITION_PATCH_SIZE,
  ROUTE_END_X,
  ROUTE_START_X,
  route_surface_height,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  BODY_CONTACT_NAMES,
  OnlineTerrainRolloutMetrics,
  action_acceleration,
  assert_recursive_json_finite,
  contact_any,
  foot_contact_any,
  foot_slip_velocity,
)


EXPECTED_SENSOR_NAMES = {
  "base": "base_ground_contact",
  "upper_leg": "upper_leg_ground_contact",
  "calf": "calf_ground_contact",
}


def _sensor_mapping_from_evaluator() -> dict[str, str]:
  """Extract the label-to-sensor mapping without depending on a private helper."""
  tree = ast.parse(inspect.getsource(terrain_evaluator))
  mappings: dict[str, str] = {}
  for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
      continue
    for key, value in zip(node.keys, node.values, strict=True):
      if not (
        isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value in EXPECTED_SENSOR_NAMES
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_sensor_contact"
        and len(value.args) >= 2
        and isinstance(value.args[1], ast.Constant)
        and isinstance(value.args[1].value, str)
      ):
        continue
      mappings[key.value] = value.args[1].value
  return mappings


class TerrainMetricMathAcceptanceTest(unittest.TestCase):
  def test_action_acceleration_is_unscaled_second_action_difference(self) -> None:
    current = torch.tensor([[3.0, -1.0, 2.0], [1.0, 2.0, 7.0]])
    previous = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 4.0]])
    older = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    expected = (current - 2.0 * previous + older).abs().mean(dim=-1)
    torch.testing.assert_close(
      action_acceleration(current, previous, older), expected
    )
    self.assertIn("without control-dt scaling", ACTION_ACCELERATION_DEFINITION)

  def test_contact_reduction_preserves_env_and_foot_axes(self) -> None:
    found = torch.tensor(
      [
        [[0, 0], [0, 2], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [1, 0], [0, 0]],
      ]
    )
    torch.testing.assert_close(contact_any(found, 2), torch.tensor([True, True]))
    torch.testing.assert_close(
      foot_contact_any(found, 2, 4),
      torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
      ),
    )

  def test_slip_averages_only_contacting_feet(self) -> None:
    velocity = torch.tensor(
      [
        [[3.0, 4.0], [0.0, 2.0], [100.0, 0.0]],
        [[9.0, 0.0], [8.0, 0.0], [7.0, 0.0]],
      ]
    )
    contact = torch.tensor(
      [[True, True, False], [False, False, False]]
    )
    torch.testing.assert_close(
      foot_slip_velocity(velocity, contact), torch.tensor([3.5, 0.0])
    )


class TerrainMetricLifecycleAcceptanceTest(unittest.TestCase):
  @staticmethod
  def _update(
    metrics: OnlineTerrainRolloutMetrics,
    *,
    mask: torch.Tensor,
    action: tuple[float, float],
    slip: tuple[float, float],
    base: tuple[bool, bool],
    upper: tuple[bool, bool],
    calf: tuple[bool, bool],
    catastrophic: tuple[bool, bool] = (False, False),
  ) -> None:
    metrics.update(
      sample_mask=mask,
      action_acceleration=torch.tensor(action, dtype=torch.float64),
      foot_slip_velocity=torch.tensor(slip, dtype=torch.float64),
      body_contacts={
        "base": torch.tensor(base),
        "upper_leg": torch.tensor(upper),
        "calf": torch.tensor(calf),
      },
      catastrophic_termination=torch.tensor(catastrophic),
    )

  def test_mean_p95_max_contacts_and_active_step_denominator(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(2, 3, dtype=torch.float64)
    self._update(
      metrics,
      mask=torch.tensor([True, True]),
      action=(1.0, 10.0),
      slip=(2.0, 20.0),
      base=(True, False),
      upper=(False, True),
      calf=(True, False),
    )
    self._update(
      metrics,
      mask=torch.tensor([True, True]),
      action=(2.0, 20.0),
      slip=(4.0, 40.0),
      base=(True, True),
      upper=(True, True),
      calf=(False, True),
      catastrophic=(False, True),
    )
    self._update(
      metrics,
      mask=torch.tensor([False, False]),
      action=(999.0, 999.0),
      slip=(999.0, 999.0),
      base=(True, True),
      upper=(True, True),
      calf=(True, True),
      catastrophic=(True, True),
    )

    first = metrics.result(0)
    self.assertEqual(
      first["action_acceleration"],
      {"mean": 1.5, "p95": 1.95, "max": 2.0},
    )
    self.assertEqual(first["foot_slip_velocity"]["mean"], 3.0)
    self.assertAlmostEqual(first["foot_slip_velocity"]["p95"], 3.9)
    self.assertEqual(first["foot_slip_velocity"]["max"], 4.0)
    self.assertEqual(first["active_control_step_samples"], 2)
    self.assertIn("terminal step is included", first["sample_denominator_definition"])
    self.assertEqual(tuple(first["body_contacts"]), BODY_CONTACT_NAMES)
    self.assertEqual(
      first["body_contacts"]["base"],
      {
        "non_terminating_count": 2,
        "non_terminating_rate": 1.0,
        "all_contact_count": 2,
        "all_contact_rate": 1.0,
        "denominator": 2,
      },
    )

    second = metrics.result(1)
    self.assertEqual(second["active_control_step_samples"], 2)
    self.assertEqual(
      second["body_contacts"]["upper_leg"]["all_contact_count"], 2
    )
    self.assertEqual(
      second["body_contacts"]["upper_leg"]["non_terminating_count"], 1
    )
    self.assertEqual(
      second["body_contacts"]["upper_leg"]["non_terminating_rate"], 0.5
    )
    self.assertEqual(
      second["catastrophic_termination"],
      {"control_step_count": 1, "occurred": True},
    )

  def test_completion_failure_and_reset_episode_samples_are_frozen(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(2, 3, dtype=torch.float64)
    active = torch.tensor([True, True])
    self._update(
      metrics,
      mask=active,
      action=(1.0, 2.0),
      slip=(1.0, 2.0),
      base=(False, False),
      upper=(False, False),
      calf=(False, False),
    )
    terminal = update_attempt_status(
      active=active,
      progress=torch.tensor([1.0, 0.4]),
      cross_track=torch.zeros(2),
      heading_error=torch.zeros(2),
      failure_mask=torch.tensor([False, True]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    torch.testing.assert_close(terminal.sample_mask, torch.tensor([True, True]))
    torch.testing.assert_close(terminal.completed_now, torch.tensor([True, False]))
    torch.testing.assert_close(terminal.failed_now, torch.tensor([False, True]))
    self._update(
      metrics,
      mask=terminal.sample_mask,
      action=(3.0, 4.0),
      slip=(3.0, 4.0),
      base=(False, True),
      upper=(False, False),
      calf=(False, False),
      catastrophic=(False, True),
    )
    post_reset = update_attempt_status(
      active=terminal.active,
      progress=torch.tensor([99.0, 99.0]),
      cross_track=torch.zeros(2),
      heading_error=torch.zeros(2),
      failure_mask=torch.tensor([False, False]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    self.assertFalse(bool(post_reset.sample_mask.any()))
    self._update(
      metrics,
      mask=post_reset.sample_mask,
      action=(999.0, 999.0),
      slip=(999.0, 999.0),
      base=(True, True),
      upper=(True, True),
      calf=(True, True),
      catastrophic=(True, True),
    )
    self.assertEqual(metrics.result(0)["action_acceleration"]["max"], 3.0)
    self.assertEqual(metrics.result(1)["action_acceleration"]["max"], 4.0)
    self.assertEqual(
      metrics.result(1)["body_contacts"]["base"]["all_contact_count"], 1
    )

  def test_empty_attempt_and_unavailable_sensors_use_null_with_reason(self) -> None:
    empty = OnlineTerrainRolloutMetrics(1, 1)
    empty.update(
      sample_mask=torch.tensor([False]),
      action_acceleration=torch.tensor([1.0]),
      foot_slip_velocity=None,
      body_contacts={},
      catastrophic_termination=torch.tensor([False]),
    )
    result = empty.result(0)
    for name in ("action_acceleration", "foot_slip_velocity"):
      self.assertEqual(
        {key: result[name][key] for key in ("mean", "p95", "max")},
        {"mean": None, "p95": None, "max": None},
      )
      self.assertTrue(result[name]["reason"])
    for name in BODY_CONTACT_NAMES:
      contact = result["body_contacts"][name]
      self.assertIsNone(contact["non_terminating_count"])
      self.assertIsNone(contact["non_terminating_rate"])
      self.assertIsNone(contact["all_contact_count"])
      self.assertIsNone(contact["all_contact_rate"])
      self.assertEqual(contact["denominator"], 0)
      self.assertTrue(contact["reason"])

    unavailable = OnlineTerrainRolloutMetrics(1, 1)
    unavailable.update(
      sample_mask=torch.tensor([True]),
      action_acceleration=torch.tensor([1.0]),
      foot_slip_velocity=None,
      body_contacts={"upper_leg": torch.tensor([False])},
      catastrophic_termination=torch.tensor([False]),
    )
    unavailable_result = unavailable.result(0)
    self.assertIsNone(unavailable_result["foot_slip_velocity"]["mean"])
    self.assertEqual(
      unavailable_result["foot_slip_velocity"]["reason"],
      "foot_contact_sensor_unavailable",
    )
    self.assertIsNone(
      unavailable_result["body_contacts"]["base"]["all_contact_rate"]
    )
    self.assertEqual(
      unavailable_result["body_contacts"]["base"]["reason"],
      "contact_sensor_unavailable",
    )
    self.assertEqual(
      unavailable_result["body_contacts"]["upper_leg"]["all_contact_rate"],
      0.0,
    )


class TerrainContactWiringAcceptanceTest(unittest.TestCase):
  def test_evaluator_labels_map_to_registered_v7_contact_sensors(self) -> None:
    self.assertEqual(_sensor_mapping_from_evaluator(), EXPECTED_SENSOR_NAMES)

  def test_registered_sensor_collision_patterns_cover_exact_body_groups(self) -> None:
    source = inspect.getsource(env_cfgs.unitree_go2_rough_v5_env_cfg)
    expected = (
      ("base_ground_contact", r"base[123]_collision"),
      ("upper_leg_ground_contact", r".*_(hip|thigh)_collision"),
      ("calf_ground_contact", r".*_calf[12]_collision"),
    )
    for sensor, pattern in expected:
      self.assertIn(sensor, source)
      self.assertIn(pattern, source)


class HighDifficultyTerrainAcceptanceTest(unittest.TestCase):
  def test_high_levels_are_explicit_and_above_existing_medium(self) -> None:
    self.assertEqual(len(HIGH_DIFFICULTIES), 2)
    self.assertEqual(len(HIGH_DIFFICULTY_LABELS), 2)
    self.assertGreater(HIGH_DIFFICULTIES[0], MEDIUM_DIFFICULTY)
    self.assertGreater(HIGH_DIFFICULTIES[1], HIGH_DIFFICULTIES[0])
    for level, (label, difficulty) in enumerate(
      zip(HIGH_DIFFICULTY_LABELS, HIGH_DIFFICULTIES, strict=True)
    ):
      self.assertEqual(difficulty_for_high_level(level), (label, difficulty))
    for invalid in (-1, 2):
      with self.assertRaises(ValueError):
        difficulty_for_high_level(invalid)

  def test_effective_high_geometry_is_harder_or_truthfully_invariant(self) -> None:
    for kind in TERRAIN_CURVE_KINDS:
      with self.subTest(kind=kind):
        medium = effective_terrain_parameters(kind, 1)
        high = effective_high_terrain_parameters(kind, 0)
        extreme = effective_high_terrain_parameters(kind, 1)
        self.assertEqual(high["requested_difficulty"], HIGH_DIFFICULTIES[0])
        self.assertEqual(extreme["requested_difficulty"], HIGH_DIFFICULTIES[1])
        if kind.startswith("slope"):
          self.assertGreater(
            abs(float(high["slope_gradient"])),
            abs(float(medium["slope_gradient"])),
          )
          self.assertGreater(
            abs(float(extreme["slope_gradient"])),
            abs(float(high["slope_gradient"])),
          )
          self.assertTrue(high["difficulty_affects_geometry"])
        elif kind == "discrete_obstacle":
          self.assertGreater(
            float(high["obstacle_height"]),
            float(medium["obstacle_height"]),
          )
          self.assertGreater(
            float(extreme["obstacle_height"]),
            float(high["obstacle_height"]),
          )
          self.assertTrue(high["difficulty_affects_geometry"])
        else:
          self.assertFalse(high["difficulty_affects_geometry"])
          self.assertTrue(high["difficulty_invariant"])
          self.assertEqual(high["noise_range"], extreme["noise_range"])
          self.assertEqual(high["noise_step"], extreme["noise_step"])
          self.assertIn("ignores", str(high["difficulty_reason"]))

  def test_high_curve_generator_and_arc_s_footprints_are_valid(self) -> None:
    generator = make_high_difficulty_curve_generator(seed=42)
    self.assertEqual(generator.num_rows, len(HIGH_DIFFICULTIES))
    self.assertEqual(generator.num_cols, len(TERRAIN_CURVE_KINDS))
    self.assertEqual(tuple(generator.sub_terrains), TERRAIN_CURVE_KINDS)
    for terrain_kind in TERRAIN_CURVE_KINDS:
      for route_kind in ("arc", "s_curve"):
        for turn_sign in (-1, 1):
          with self.subTest(
            terrain_kind=terrain_kind,
            route_kind=route_kind,
            turn_sign=turn_sign,
          ):
            footprint = validate_route_footprint(
              route_kind, 4.0, turn_sign
            )
            self.assertTrue(footprint.corridor_inside_patch)
            self.assertTrue(footprint.scan_footprint_inside_patch)
            self.assertGreater(footprint.corridor_boundary_margin, 0.0)
            self.assertGreater(footprint.scan_boundary_margin, 0.0)


class ContinuousTransitionGeometryAcceptanceTest(unittest.TestCase):
  def test_straight_route_corridor_and_rotated_scan_fit_single_patch(self) -> None:
    footprint = validate_straight_transition_footprint()
    self.assertEqual(footprint.patch_size, TRANSITION_PATCH_SIZE)
    self.assertEqual(
      footprint.route_bounds,
      ((ROUTE_START_X, ROUTE_END_X), (2.0, 2.0)),
    )
    self.assertEqual(
      footprint.feature_bounds,
      ((FEATURE_START_X, FEATURE_END_X), (0.0, TRANSITION_PATCH_SIZE[1])),
    )
    self.assertTrue(footprint.route_inside_patch)
    self.assertTrue(footprint.corridor_inside_patch)
    self.assertTrue(footprint.scan_footprint_inside_patch)
    self.assertGreater(footprint.corridor_boundary_margin, 0.0)
    self.assertGreater(footprint.scan_boundary_margin, 0.0)
    with self.assertRaisesRegex(ValueError, "height-scan footprint"):
      validate_straight_transition_footprint(scan_half_extents=(1.1, 0.5))

  def test_slope_and_stair_profiles_have_flat_entry_exit_and_valid_junctions(self) -> None:
    difficulty = 0.8
    approach = np.linspace(ROUTE_START_X, FEATURE_START_X - 0.01, 21)
    exit_flat = np.linspace(FEATURE_END_X + 0.01, ROUTE_END_X, 21)
    for kind in ("slope_up", "slope_down", "stairs_up", "stairs_down"):
      with self.subTest(kind=kind):
        metadata = boundary_transition_metadata(kind, difficulty)
        self.assertEqual(
          metadata.geometry_contract,
          "single intra-patch approach-feature-exit geometry",
        )
        self.assertEqual(metadata.patch_size, TRANSITION_PATCH_SIZE)
        entry = np.asarray(route_surface_height(kind, difficulty, approach))
        exit_values = np.asarray(
          route_surface_height(kind, difficulty, exit_flat)
        )
        np.testing.assert_allclose(
          entry, metadata.entry_surface_z, rtol=0.0, atol=1.0e-10
        )
        np.testing.assert_allclose(
          exit_values, metadata.exit_surface_z, rtol=0.0, atol=1.0e-10
        )
        self.assertTrue(np.isfinite(entry).all())
        self.assertTrue(np.isfinite(exit_values).all())

        if metadata.family == "slope":
          epsilon = 1.0e-7
          junctions = np.asarray(
            route_surface_height(
              kind,
              difficulty,
              np.array(
                [
                  FEATURE_START_X - epsilon,
                  FEATURE_START_X,
                  FEATURE_END_X - epsilon,
                  FEATURE_END_X,
                ]
              ),
            )
          )
          self.assertAlmostEqual(junctions[0], junctions[1], delta=1.0e-6)
          self.assertAlmostEqual(junctions[2], junctions[3], delta=1.0e-6)
        else:
          samples = np.linspace(FEATURE_START_X - 0.01, FEATURE_END_X + 0.01, 241)
          heights = np.asarray(route_surface_height(kind, difficulty, samples))
          self.assertLessEqual(
            float(np.max(np.abs(np.diff(heights)))),
            metadata.step_height + 1.0e-10,
          )

  def test_rough_and_obstacle_are_one_heightfield_with_flat_guarded_boundaries(self) -> None:
    for kind in ("random_rough", "discrete_obstacle"):
      with self.subTest(kind=kind):
        x, y, heights = continuous_feature_heightfield(kind, 0.8, seed=42)
        metadata = boundary_transition_metadata(kind, 0.8)
        self.assertEqual(x.shape, (81,))
        self.assertEqual(y.shape, (41,))
        self.assertEqual(heights.shape, (41, 81))
        self.assertTrue(np.isfinite(heights).all())
        self.assertEqual(
          metadata.geometry_contract,
          "single heightfield with flat approach and exit",
        )
        np.testing.assert_allclose(
          heights[:, x <= FEATURE_START_X], 0.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
          heights[:, x >= FEATURE_END_X], 0.0, rtol=0.0, atol=0.0
        )
        self.assertGreater(
          float(np.max(np.abs(heights[:, (x > FEATURE_START_X) & (x < FEATURE_END_X)]))),
          0.0,
        )
        x_again, y_again, heights_again = continuous_feature_heightfield(
          kind, 0.8, seed=42
        )
        np.testing.assert_array_equal(x_again, x)
        np.testing.assert_array_equal(y_again, y)
        np.testing.assert_array_equal(heights_again, heights)

  def test_transition_difficulty_matrix_is_seeded_and_row_bounded(self) -> None:
    matrix = boundary_transition_difficulty_matrix(42)
    self.assertEqual(
      matrix.shape, (GENERATOR_NUM_ROWS, len(CONTINUOUS_TRANSITION_KINDS))
    )
    np.testing.assert_array_equal(matrix, boundary_transition_difficulty_matrix(42))
    self.assertTrue(np.isfinite(matrix).all())
    self.assertGreaterEqual(float(matrix.min()), 0.0)
    self.assertLessEqual(float(matrix.max()), 1.0)
    for row in range(GENERATOR_NUM_ROWS):
      self.assertTrue(np.all(matrix[row] >= row / GENERATOR_NUM_ROWS))
      self.assertTrue(np.all(matrix[row] <= (row + 1) / GENERATOR_NUM_ROWS))

  def test_transition_coverage_is_truthful_and_curves_are_rejected(self) -> None:
    coverage = continuous_boundary_coverage()
    self.assertTrue(coverage["implemented"])
    self.assertTrue(coverage["continuous_intra_patch_transitions"])
    self.assertFalse(coverage["continuous_inter_patch_transitions"])
    self.assertEqual(coverage["route_kind"], "straight")
    self.assertEqual(
      tuple(coverage["transition_cases"]), CONTINUOUS_TRANSITION_KINDS
    )
    self.assertTrue(coverage["single_surface_contract"])
    self.assertFalse(coverage["curve_transitions"])
    self.assertFalse(coverage["stairs_curves"])
    self.assertIn("not scaled", str(coverage["reason"]))
    for kind in CONTINUOUS_TRANSITION_KINDS:
      self.assertIsNone(reject_curved_transition(kind, "straight"))
      for route_kind in ("arc", "s_curve"):
        expected = (
          "stairs curves are unsupported"
          if kind.startswith("stairs")
          else "continuous curved transitions are unsupported"
        )
        with self.assertRaisesRegex(ValueError, expected):
          reject_curved_transition(kind, route_kind)

  def test_xyz_relocation_preserves_patch_relative_pose_and_orientation(self) -> None:
    root = torch.tensor(
      [[1.2, -2.3, 0.7, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    old_origin = torch.tensor([[1.0, -2.0, 0.1]], dtype=torch.float64)
    new_origin = torch.tensor([[20.0, 9.0, 1.1]], dtype=torch.float64)
    relocated, error = relocate_root_pose(root, old_origin, new_origin)
    torch.testing.assert_close(
      relocated[:, :3] - new_origin, root[:, :3] - old_origin
    )
    torch.testing.assert_close(relocated[:, 3:7], root[:, 3:7])
    self.assertLess(error, 1.0e-9)

  def test_metadata_footprint_and_coverage_are_strict_json(self) -> None:
    payload = {
      "coverage": continuous_boundary_coverage(),
      "footprint": dataclasses.asdict(validate_straight_transition_footprint()),
      "scenarios": [
        dataclasses.asdict(boundary_transition_metadata(kind, 0.8))
        for kind in CONTINUOUS_TRANSITION_KINDS
      ],
    }
    assert_recursive_json_finite(payload)
    json.dumps(payload, allow_nan=False)


class TerrainStrictJsonAcceptanceTest(unittest.TestCase):
  def test_metric_payload_is_strict_json_and_recursive_finite(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(1, 1)
    metrics.update(
      sample_mask=torch.tensor([True]),
      action_acceleration=torch.tensor([1.0]),
      foot_slip_velocity=torch.tensor([2.0]),
      body_contacts={name: torch.tensor([False]) for name in BODY_CONTACT_NAMES},
      catastrophic_termination=torch.tensor([False]),
    )
    payload = {"scenario": metrics.result(0), "empty": None}
    assert_recursive_json_finite(payload)
    json.dumps(payload, allow_nan=False)

  def test_recursive_finite_rejects_all_nonfinite_float_variants(self) -> None:
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(value=value), self.assertRaisesRegex(
        ValueError, "root.scenarios"
      ):
        assert_recursive_json_finite({"scenarios": [{"metric": value}]})


if __name__ == "__main__":
  unittest.main()
