"""CPU-only tests for the strict matched high-slope evaluator."""

from __future__ import annotations

import math
import unittest

import torch

from scripts.evaluate_go2_high_slope_matched import (
  HighSlopeMatchedConfig,
  _OnlineRouteErrors,
  _contact_termination_summary,
  _final_failure_reason,
  _scenarios,
  _validate_config,
)
from src.tasks.velocity.evaluation.high_slope_matched import (
  DIFFICULTIES,
  ROUTE_KINDS,
  ROUTE_LENGTH_DEFINITION,
  SLOPE_DIRECTIONS,
  build_matched_scenarios,
  compute_route_footprint,
  difficulty_for_level,
  effective_slope_parameters,
  geometry_preflight,
  minimum_horizon_steps,
  validate_horizon,
  validate_matched_result_invariants,
  validate_route_footprint,
)


class ScenarioContractTest(unittest.TestCase):
  def test_default_slot_order_is_stable_and_route_independent(self) -> None:
    scenarios = build_matched_scenarios()
    self.assertEqual(len(scenarios), 2 * 2 * 2 * 2 * 2)
    self.assertEqual(
      [scenario.matched_slot for scenario in scenarios],
      list(range(len(scenarios))),
    )
    first = scenarios[:8]
    self.assertEqual({item.slope_direction for item in first}, {"slope_up"})
    self.assertEqual({item.level for item in first}, {0})
    self.assertEqual(
      [(item.radius, item.speed, item.turn_sign) for item in first[:4]],
      [(2.5, 0.3, 1), (2.5, 0.3, -1),
       (2.5, 0.5, 1), (2.5, 0.5, -1)],
    )

  def test_script_scenarios_preserve_dataclass_contract(self) -> None:
    cfg = HighSlopeMatchedConfig(
      checkpoint="model.pt", radii=(2.5,), repeats=2
    )
    values = _scenarios(cfg)
    self.assertEqual(len(values), 32)
    self.assertEqual(
      [item["matched_slot"] for item in values], list(range(32))
    )
    self.assertEqual({item["difficulty"] for item in values}, set(DIFFICULTIES))

  def test_invalid_matrix_members_are_rejected(self) -> None:
    with self.assertRaises(ValueError):
      build_matched_scenarios(slope_directions=("stairs_up",))
    with self.assertRaises(ValueError):
      build_matched_scenarios(levels=(2,))
    with self.assertRaises(ValueError):
      build_matched_scenarios(turn_signs=(0,))


class TerrainSemanticsTest(unittest.TestCase):
  def test_high_and_extreme_have_exact_effective_gradients(self) -> None:
    for level, difficulty in enumerate(DIFFICULTIES):
      label, actual = difficulty_for_level(level)
      self.assertEqual(label, ("high", "extreme")[level])
      self.assertEqual(actual, difficulty)
      up = effective_slope_parameters("slope_up", level)
      down = effective_slope_parameters("slope_down", level)
      self.assertAlmostEqual(up["slope_gradient"], difficulty * 0.4)
      self.assertAlmostEqual(down["slope_gradient"], -difficulty * 0.4)
      self.assertTrue(up["inverted"])
      self.assertFalse(down["inverted"])
      self.assertEqual(up["primitive"], down["primitive"])

  def test_direction_and_level_type_checks_are_strict(self) -> None:
    self.assertEqual(SLOPE_DIRECTIONS, ("slope_up", "slope_down"))
    with self.assertRaises(TypeError):
      difficulty_for_level(True)
    with self.assertRaises(ValueError):
      effective_slope_parameters("random_rough", 0)


class GeometryContractTest(unittest.TestCase):
  def test_all_routes_have_exact_common_length(self) -> None:
    for radius in (2.5, 4.0):
      expected = 2.0 * math.pi * radius / 3.0
      lengths = {
        compute_route_footprint(kind, radius, 1).route_length
        for kind in ROUTE_KINDS
      }
      self.assertEqual(len(lengths), 1)
      self.assertAlmostEqual(lengths.pop(), expected)
    self.assertEqual(ROUTE_LENGTH_DEFINITION, "2*pi*radius/3")

  def test_arc_is_120_degrees_and_s_is_two_60_degree_arcs(self) -> None:
    radius = 2.5
    arc = compute_route_footprint("arc", radius, 1)
    s_curve = compute_route_footprint("s_curve", radius, 1)
    self.assertAlmostEqual(
      arc.centerline_bounds[0][1], 9.0 + radius, places=5
    )
    self.assertAlmostEqual(
      arc.centerline_bounds[1][1], 9.0 + 1.5 * radius, places=5
    )
    self.assertAlmostEqual(
      s_curve.centerline_bounds[0][1],
      9.0 + math.sqrt(3.0) * radius,
      places=5,
    )
    self.assertAlmostEqual(
      s_curve.centerline_bounds[1][1], 9.0 + radius, places=5
    )

  def test_left_and_right_are_mirrored_with_equal_margins(self) -> None:
    for kind in ROUTE_KINDS:
      left = compute_route_footprint(kind, 2.5, 1)
      right = compute_route_footprint(kind, 2.5, -1)
      self.assertEqual(left.centerline_bounds[0], right.centerline_bounds[0])
      self.assertAlmostEqual(
        left.centerline_bounds[1][0],
        18.0 - right.centerline_bounds[1][1],
      )
      self.assertAlmostEqual(
        left.centerline_bounds[1][1],
        18.0 - right.centerline_bounds[1][0],
      )
      self.assertAlmostEqual(
        left.scan_boundary_margin, right.scan_boundary_margin
      )

  def test_r2p5_full_matrix_has_valid_corridor_and_rotated_scan(self) -> None:
    for kind in ROUTE_KINDS:
      for sign in (-1, 1):
        footprint = validate_route_footprint(kind, 2.5, sign)
        self.assertTrue(footprint.centreline_inside_patch)
        self.assertTrue(footprint.corridor_inside_patch)
        self.assertTrue(footprint.scan_footprint_inside_patch)
        self.assertGreater(footprint.scan_boundary_margin, 0.0)

  def test_r4_straight_scan_conflict_is_truthfully_rejected(self) -> None:
    footprint = compute_route_footprint("straight", 4.0, 1)
    self.assertTrue(footprint.centreline_inside_patch)
    self.assertTrue(footprint.corridor_inside_patch)
    self.assertFalse(footprint.scan_footprint_inside_patch)
    self.assertAlmostEqual(
      footprint.scan_boundary_margin,
      18.0 - (9.0 + 8.0 * math.pi / 3.0 + 0.8),
    )
    with self.assertRaisesRegex(ValueError, "height-scan footprint"):
      validate_route_footprint("straight", 4.0, 1)

  def test_preflight_and_config_never_overclaim_r4(self) -> None:
    preflight = geometry_preflight((2.5, 4.0), (-1, 1))
    self.assertFalse(preflight["all_requested_combinations_valid"])
    invalid = [item for item in preflight["combinations"] if not item["valid"]]
    self.assertEqual(
      {(item["route_kind"], item["radius"], item["turn_sign"]) for item in invalid},
      {("straight", 4.0, -1), ("straight", 4.0, 1)},
    )
    with self.assertRaisesRegex(ValueError, "not geometrically valid"):
      _validate_config(HighSlopeMatchedConfig(checkpoint="model.pt"))
    valid = _validate_config(HighSlopeMatchedConfig(
      checkpoint="model.pt", radii=(2.5,)
    ))
    self.assertTrue(valid["all_requested_combinations_valid"])


class HorizonAndMetricTest(unittest.TestCase):
  def test_slow_case_horizon_lower_bound_includes_settle(self) -> None:
    required = minimum_horizon_steps(4.0, 0.3, 0.02, 10)
    self.assertEqual(required, math.ceil((8.0 * math.pi / 3.0) / 0.006) + 10)
    self.assertEqual(
      validate_horizon(
        required, radii=(2.5, 4.0), speeds=(0.3, 0.5),
        control_dt=0.02, settle_steps=10,
      ),
      required,
    )
    with self.assertRaisesRegex(ValueError, "shorter than"):
      validate_horizon(
        required - 1, radii=(2.5, 4.0), speeds=(0.3, 0.5),
        control_dt=0.02, settle_steps=10,
      )

  def test_route_error_metrics_freeze_mask_and_compute_p95(self) -> None:
    metrics = _OnlineRouteErrors(2, 3, device="cpu", dtype=torch.float32)
    metrics.update(
      torch.tensor([1.0, 10.0]), torch.tensor([2.0, 20.0]),
      torch.tensor([True, True]),
    )
    metrics.update(
      torch.tensor([100.0, 12.0]), torch.tensor([200.0, 22.0]),
      torch.tensor([False, True]),
    )
    first = metrics.result(0)
    second = metrics.result(1)
    self.assertEqual(first["sample_count"], 1)
    self.assertEqual(first["cross_track_absolute"]["p95"], 1.0)
    self.assertAlmostEqual(second["cross_track_absolute"]["p95"], 11.9)
    self.assertAlmostEqual(second["heading_absolute"]["p95"], 21.9)


class ResultInvariantTest(unittest.TestCase):
  @staticmethod
  def _result(kind: str, slots: list[int]) -> dict[str, object]:
    return {
      "route_kind": kind,
      "route_kind_invariants": {"seed": 42, "steps": 2400},
      "scenarios": [{"matched_slot": slot} for slot in slots],
    }

  def test_route_results_require_order_slots_and_same_invariants(self) -> None:
    valid = {kind: self._result(kind, [0, 1]) for kind in ROUTE_KINDS}
    validate_matched_result_invariants(valid)
    invalid_slots = dict(valid)
    invalid_slots["arc"] = self._result("arc", [1, 0])
    with self.assertRaisesRegex(ValueError, "matched_slot"):
      validate_matched_result_invariants(invalid_slots)
    invalid_settings = dict(valid)
    invalid_settings["s_curve"] = {
      **self._result("s_curve", [0, 1]),
      "route_kind_invariants": {"seed": 43, "steps": 2400},
    }
    with self.assertRaisesRegex(ValueError, "invariant settings"):
      validate_matched_result_invariants(invalid_settings)

  def test_contact_termination_summary_keeps_contacts_and_terms_separate(self) -> None:
    contacts = {
      body: {
        "non_terminating_count": index,
        "non_terminating_rate": index / 10.0,
        "denominator": 10,
      }
      for index, body in enumerate(("base", "upper_leg", "calf"), start=1)
    }
    summary = _contact_termination_summary(
      {
        "fell_over": 1,
        "illegal_base_contact": 2,
        "illegal_upper_leg_contact": 3,
        "illegal_calf_contact": 4,
      },
      contacts,
    )
    self.assertEqual(summary["fell"]["termination_count"], 1)
    self.assertEqual(summary["base"]["termination_count"], 2)
    self.assertEqual(summary["base"]["non_terminating_count"], 1)
    self.assertEqual(summary["calf"]["termination_count"], 4)
    self.assertEqual(summary["calf"]["non_terminating_count"], 3)

  def test_failure_reason_is_null_on_success_and_required_on_failure(self) -> None:
    self.assertIsNone(_final_failure_reason(
      completed=True, failed=False, reason=None
    ))
    self.assertEqual(
      _final_failure_reason(
        completed=False, failed=True, reason="illegal_calf_contact"
      ),
      "illegal_calf_contact",
    )
    for kwargs in (
      {"completed": True, "failed": True, "reason": "reset"},
      {"completed": False, "failed": True, "reason": None},
      {"completed": False, "failed": True, "reason": ""},
      {"completed": False, "failed": False, "reason": "step_limit"},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        _final_failure_reason(**kwargs)


if __name__ == "__main__":
  unittest.main()
