"""Independent CPU acceptance contracts for the high-slope matched suite.

These tests intentionally do not launch MuJoCo/GPU rollouts.  They check the
geometry, command/lifecycle math, metric denominators, and the evidence schema
that a formal JSON result must satisfy.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Mapping

import torch

from scripts.evaluate_go2_high_slope_matched import (
  HighSlopeMatchedConfig,
  _OnlineRouteErrors,
  _contact_termination_summary,
  _validate_config,
)
from src.tasks.velocity.evaluation.curved_routes import (
  arc_command_controller,
  make_arc_route,
  make_s_route,
  s_command_controller,
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
  matched_route_length,
  minimum_horizon_steps,
  validate_horizon,
  validate_matched_result_invariants,
  validate_route_footprint,
)
from src.tasks.velocity.evaluation.routes import (
  straight_line_controller,
  update_attempt_status,
)
from src.tasks.velocity.evaluation.terrain_curved_routes import relocate_root_pose
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTIVE_SAMPLE_DEFINITION,
  OnlineTerrainRolloutMetrics,
  assert_recursive_json_finite,
)


def _assert_inside(bounds: Any, patch: Any) -> None:
  (xmin, xmax), (ymin, ymax) = bounds
  pxmin, pxmax = patch[0]
  pymin, pymax = patch[1]
  if not (pxmin <= xmin <= xmax <= pxmax and pymin <= ymin <= ymax <= pymax):
    raise AssertionError(f"bounds {bounds} leave patch {patch}")


def _assert_formal_result(payload: Mapping[str, Any]) -> None:
  """Acceptance-only schema check used for post-GPU JSON review."""
  assert_recursive_json_finite(payload)
  for key in ("schema_version", "geometry_preflight", "metric_invariants", "coverage", "profiles"):
    if key not in payload:
      raise AssertionError(f"formal result missing {key}")
  if payload["metric_invariants"]["sample_denominator"] != ACTIVE_SAMPLE_DEFINITION:
    raise AssertionError("sample denominator changed")
  preflight = payload["geometry_preflight"]
  if not preflight["all_requested_combinations_valid"]:
    # A rejected request is honest; it must not claim complete matched coverage.
    if any(bool(value) for value in payload["coverage"].values()):
      raise AssertionError("invalid geometry may not be reported as coverage")
  for profile_name, profile in payload["profiles"].items():
    routes = profile["route_results"]
    if tuple(routes) != ROUTE_KINDS:
      raise AssertionError("route result order is not straight/arc/S")
    slots = None
    invariant = None
    for kind in ROUTE_KINDS:
      result = routes[kind]
      if slots is None:
        slots = [item["matched_slot"] for item in result["scenarios"]]
        invariant = result["route_kind_invariants"]
      elif slots != [item["matched_slot"] for item in result["scenarios"]]:
        raise AssertionError("matched slots differ across route kinds")
      if invariant != result["route_kind_invariants"]:
        raise AssertionError("route invariants differ across route kinds")
      for scenario in result["scenarios"]:
        required = {
          "completed", "progress_ratio", "route_length", "geometry",
          "commanded_velocity_mean", "actual_velocity_mean", "response_gain",
          "controller_saturation_fraction", "cross_track_p95", "cross_track_max",
          "heading_p95", "heading_max", "action_acceleration_p95",
          "action_acceleration_max", "slip_velocity_p95", "slip_velocity_max",
          "contact_termination_summary", "termination_counts", "reset_count",
          "first_failure_reason",
        }
        missing = required.difference(scenario)
        if missing:
          raise AssertionError(f"{profile_name}/{kind} missing {sorted(missing)}")
        geometry = scenario["geometry"]
        if not (geometry["centerline_inside_patch"] and geometry["corridor_inside_patch"] and geometry["scan_footprint_inside_patch"]):
          raise AssertionError("an invalid route was emitted as a runnable scenario")
        _assert_inside(geometry["corridor_bounds_local"], geometry["patch_size"] if "patch_size" in geometry else ((0.0, 18.0), (0.0, 18.0)))
        for metric in ("cross_track_p95", "cross_track_max", "heading_p95", "heading_max", "action_acceleration_p95", "action_acceleration_max", "slip_velocity_p95", "slip_velocity_max"):
          value = scenario[metric]
          if value is not None and not math.isfinite(float(value)):
            raise AssertionError(f"non-finite metric {metric}")


class GeometryAndScenarioAcceptanceTest(unittest.TestCase):
  def test_common_length_slot_order_and_mirror_contract(self) -> None:
    for radius in (2.5, 4.0):
      expected = 2.0 * math.pi * radius / 3.0
      self.assertAlmostEqual(matched_route_length(radius), expected)
      lengths = {compute_route_footprint(kind, radius, 1).route_length for kind in ROUTE_KINDS}
      self.assertEqual(len(lengths), 1)
      self.assertAlmostEqual(lengths.pop(), expected)
    scenarios = build_matched_scenarios()
    self.assertEqual([item.matched_slot for item in scenarios], list(range(len(scenarios))))
    self.assertEqual(
      [(item.radius, item.speed, item.turn_sign) for item in scenarios[:4]],
      [(2.5, 0.3, 1), (2.5, 0.3, -1), (2.5, 0.5, 1), (2.5, 0.5, -1)],
    )
    self.assertEqual(ROUTE_LENGTH_DEFINITION, "2*pi*radius/3")

  def test_high_extreme_and_up_down_semantics_are_exact(self) -> None:
    self.assertEqual(SLOPE_DIRECTIONS, ("slope_up", "slope_down"))
    self.assertEqual(DIFFICULTIES, (0.8, 1.0))
    for level, difficulty in enumerate(DIFFICULTIES):
      label, value = difficulty_for_level(level)
      self.assertEqual(label, ("high", "extreme")[level])
      self.assertEqual(value, difficulty)
      up = effective_slope_parameters("slope_up", level)
      down = effective_slope_parameters("slope_down", level)
      self.assertAlmostEqual(up["slope_gradient"], difficulty * 0.4)
      self.assertAlmostEqual(down["slope_gradient"], -difficulty * 0.4)
      self.assertTrue(up["inverted"])
      self.assertFalse(down["inverted"])
      self.assertNotEqual(up["route_direction_semantics"], down["route_direction_semantics"])

  def test_r2p5_is_formally_valid_but_r4_straight_scan_is_rejected(self) -> None:
    for kind in ROUTE_KINDS:
      for sign in (-1, 1):
        footprint = validate_route_footprint(kind, 2.5, sign)
        self.assertTrue(footprint.corridor_inside_patch)
        self.assertTrue(footprint.scan_footprint_inside_patch)
    footprint = compute_route_footprint("straight", 4.0, 1)
    self.assertTrue(footprint.centreline_inside_patch)
    self.assertTrue(footprint.corridor_inside_patch)
    self.assertFalse(footprint.scan_footprint_inside_patch)
    with self.assertRaisesRegex(ValueError, "height-scan footprint"):
      validate_route_footprint("straight", 4.0, 1)
    preflight = geometry_preflight((2.5, 4.0), (-1, 1))
    self.assertFalse(preflight["all_requested_combinations_valid"])
    self.assertEqual(
      {(item["route_kind"], item["radius"], item["turn_sign"])
       for item in preflight["combinations"] if not item["valid"]},
      {("straight", 4.0, -1), ("straight", 4.0, 1)},
    )
    with self.assertRaisesRegex(ValueError, "not geometrically valid"):
      _validate_config(HighSlopeMatchedConfig(checkpoint="model.pt"))
    self.assertTrue(_validate_config(HighSlopeMatchedConfig(checkpoint="model.pt", radii=(2.5,)))["all_requested_combinations_valid"])

  def test_r4_rejection_cannot_be_hidden_as_coverage(self) -> None:
    payload = {
      "schema_version": 1,
      "geometry_preflight": {"all_requested_combinations_valid": False},
      "metric_invariants": {"sample_denominator": ACTIVE_SAMPLE_DEFINITION},
      "coverage": {"matched_straight_120deg_arc_two_60deg_s": True},
      "profiles": {},
    }
    with self.assertRaisesRegex(AssertionError, "invalid geometry"):
      _assert_formal_result(payload)


class PlacementControllerAndLifecycleAcceptanceTest(unittest.TestCase):
  def test_relocation_preserves_relative_xyz_and_quaternion(self) -> None:
    root = torch.tensor([[1.2, -2.3, 0.7, 1.0, 0.0, 0.0, 0.0], [-4.0, 3.5, 1.2, 0.7, 0.0, 0.0, 0.7]])
    old = torch.tensor([[1.0, -2.0, 0.1], [-5.0, 3.0, 0.4]])
    new = torch.tensor([[20.0, 9.0, 1.1], [4.0, -8.0, -0.2]])
    relocated, error = relocate_root_pose(root, old, new)
    torch.testing.assert_close(relocated[:, :3] - new, root[:, :3] - old)
    torch.testing.assert_close(relocated[:, 3:7], root[:, 3:7])
    self.assertLess(error, 1.0e-6)

  def test_straight_and_curved_controller_math_and_limits(self) -> None:
    start = torch.tensor([[2.0, 8.0]])
    heading = torch.tensor([0.0])
    straight = straight_line_controller(start, heading, start, 0.0, target_speed=0.5, cross_track_gain=1.2, heading_gain=1.0, max_lateral_speed=0.3, max_yaw_rate=0.7, route_length=5.0)
    torch.testing.assert_close(straight, torch.tensor([[0.5, 0.0, 0.0]]))
    off = torch.tensor([[2.0, 8.5]])
    corrected = straight_line_controller(off, heading, start, 0.0, target_speed=0.5, cross_track_gain=1.2, heading_gain=1.0, max_lateral_speed=0.3, max_yaw_rate=0.7, route_length=5.0)
    self.assertAlmostEqual(float(corrected[0, 1]), -0.3)  # clipped -gain*cross
    route = make_arc_route(start, 0.0, 2.5, 1, angle=2.0 * math.pi / 3.0)
    arc = arc_command_controller(route, start, heading, target_speed=0.5, max_lateral_speed=0.3, max_yaw_rate=0.7)
    self.assertAlmostEqual(float(arc[0, 0]), 0.5, places=6)
    self.assertAlmostEqual(float(arc[0, 2]), 0.2, places=6)
    s_route = make_s_route(start, 0.0, 2.5, 1)
    at_second, _ = s_route.second.pose_at(0.01)
    s_command = s_command_controller(s_route, at_second, torch.tensor([s_route.second.start_heading]), target_speed=0.5)
    self.assertLess(float(s_command[0, 2]), 0.0)

  def test_attempt_freeze_includes_terminal_step_and_excludes_later_steps(self) -> None:
    active = torch.tensor([True, True])
    result = update_attempt_status(active, torch.tensor([5.0, 1.0]), torch.zeros(2), torch.zeros(2), torch.tensor([False, True]), route_length=5.0, cross_track_tolerance=0.3, heading_tolerance=0.3)
    self.assertTrue(bool(result.completed_now[0]))
    self.assertTrue(bool(result.failed_now[1]))
    self.assertTrue(bool(result.sample_mask.all()))
    self.assertFalse(bool(result.active[0]))
    metrics = _OnlineRouteErrors(
      1, 3, device="cpu", dtype=torch.float32
    )
    metrics.update(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([True]))
    metrics.update(torch.tensor([100.0]), torch.tensor([200.0]), torch.tensor([False]))
    self.assertEqual(metrics.result(0)["sample_count"], 1)
    self.assertEqual(metrics.result(0)["cross_track_absolute"]["max"], 1.0)

  def test_horizon_and_settle_are_explicit(self) -> None:
    required = minimum_horizon_steps(2.5, 0.3, 0.02, 10)
    self.assertEqual(validate_horizon(required, radii=(2.5,), speeds=(0.3,), control_dt=0.02, settle_steps=10), required)
    with self.assertRaisesRegex(ValueError, "shorter than"):
      validate_horizon(required - 1, radii=(2.5,), speeds=(0.3,), control_dt=0.02, settle_steps=10)


class MetricsAndJsonAcceptanceTest(unittest.TestCase):
  def test_p95_max_and_contact_rates_use_active_sample_denominator(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(1, 3, device="cpu")
    metrics.update(sample_mask=torch.tensor([True]), action_acceleration=torch.tensor([1.0]), foot_slip_velocity=torch.tensor([2.0]), body_contacts={"base": torch.tensor([True]), "upper_leg": torch.tensor([False]), "calf": torch.tensor([True])}, catastrophic_termination=torch.tensor([False]))
    metrics.update(sample_mask=torch.tensor([False]), action_acceleration=torch.tensor([100.0]), foot_slip_velocity=torch.tensor([200.0]), body_contacts={"base": torch.tensor([True]), "upper_leg": torch.tensor([True]), "calf": torch.tensor([True])}, catastrophic_termination=torch.tensor([True]))
    result = metrics.result(0)
    self.assertEqual(result["active_control_step_samples"], 1)
    self.assertEqual(result["action_acceleration"]["p95"], 1.0)
    self.assertEqual(result["action_acceleration"]["max"], 1.0)
    self.assertEqual(result["body_contacts"]["base"]["denominator"], 1)
    self.assertEqual(result["body_contacts"]["base"]["non_terminating_count"], 1)
    self.assertEqual(result["body_contacts"]["base"]["non_terminating_rate"], 1.0)

  def test_contact_termination_summary_separates_termination_and_contact(self) -> None:
    summary = _contact_termination_summary({"fell_over": 1, "illegal_calf_contact": 2}, {"calf": {"non_terminating_count": 3, "non_terminating_rate": 0.25, "denominator": 12}})
    self.assertEqual(summary["fell"]["termination_count"], 1)
    self.assertEqual(summary["calf"]["termination_count"], 2)
    self.assertEqual(summary["calf"]["non_terminating_count"], 3)
    self.assertIsNone(summary["base"]["non_terminating_count"])

  def test_recursive_finite_and_matched_result_invariants(self) -> None:
    assert_recursive_json_finite({"ok": [1.0, None, {"x": 2.0}]})
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.assertRaises(ValueError):
        assert_recursive_json_finite({"nested": [value]})
    def result(kind: str, slots: list[int]) -> dict[str, Any]:
      return {"route_kind": kind, "scenarios": [{"matched_slot": x} for x in slots], "route_kind_invariants": {"seed": 42, "steps": 2400}}
    validate_matched_result_invariants({kind: result(kind, [0, 1]) for kind in ROUTE_KINDS})
    bad = {kind: result(kind, [0, 1]) for kind in ROUTE_KINDS}
    bad["arc"] = result("arc", [1, 0])
    with self.assertRaisesRegex(ValueError, "matched_slot"):
      validate_matched_result_invariants(bad)


if __name__ == "__main__":
  unittest.main()
