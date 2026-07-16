"""CPU-only tests for matched route profiles and distribution metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import torch

from src.tasks.velocity.evaluation.matched_route_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  CORE_PROFILE_NAMES,
  DYNAMICS_EVENTS,
  MatchedRouteContract,
  OBSERVATION_EVENTS,
  OnlineMatchedRouteMetrics,
  PROFILE_NAMES,
  PROFILES,
  ROUTE_KINDS,
  action_acceleration,
  assert_recursive_finite,
  configure_matched_profile,
  contact_slip_velocity,
  matched_route_length,
  matched_route_local_bounds,
  matched_thresholds,
)
from scripts.diagnose_go2_route_randomization import diagnose
from scripts.evaluate_go2_matched_routes import (
  MatchedRouteConfig,
  _scenarios,
  _validate_config,
)


@dataclass
class _Actor:
  enable_corruption: bool = True


class _Cfg:
  def __init__(self) -> None:
    self.observations = {"actor": _Actor()}
    self.events = {
      name: object()
      for name in DYNAMICS_EVENTS + OBSERVATION_EVENTS + ("push_robot",)
    }


class MatchedProfileTests(unittest.TestCase):
  def test_profiles_partition_expected_sources(self) -> None:
    expected = {
      "clean": (False, (), False),
      "dynamics_only": (False, DYNAMICS_EVENTS, False),
      "observation_only": (True, OBSERVATION_EVENTS, False),
      "push_only": (False, (), True),
      "full_randomized": (
        True, DYNAMICS_EVENTS + OBSERVATION_EVENTS, True
      ),
    }
    self.assertEqual(tuple(PROFILES), PROFILE_NAMES)
    self.assertEqual(tuple(expected), CORE_PROFILE_NAMES)
    for name, (corruption, events, push) in expected.items():
      cfg = _Cfg()
      settings = configure_matched_profile(cfg, name)
      self.assertEqual(settings["actor_observation_corruption"], corruption)
      self.assertEqual(tuple(settings["startup_randomization_events"]), events)
      self.assertEqual(settings["push_enabled"], push)
      self.assertEqual(cfg.observations["actor"].enable_corruption, corruption)
    for name, profile in PROFILES.items():
      cfg = _Cfg()
      settings = configure_matched_profile(cfg, name)
      self.assertEqual(settings["name"], name)
      self.assertEqual(
        tuple(settings["startup_randomization_events"]),
        profile.startup_events,
      )

  def test_missing_required_event_is_rejected(self) -> None:
    cfg = _Cfg()
    cfg.events.pop("base_payload")
    with self.assertRaisesRegex(ValueError, "base_payload"):
      configure_matched_profile(cfg, "dynamics_only")

  def test_unknown_profile_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "profile must be one of"):
      configure_matched_profile(_Cfg(), "randomized")


class MatchedContractTests(unittest.TestCase):
  def test_contract_exposes_route_invariants(self) -> None:
    contract = MatchedRouteContract(
      checkpoint="model.pt",
      task_id="Unitree-Go2-Rough-V7",
      seed=42,
      profile="clean",
      num_slots=8,
      speeds=(0.3, 0.5),
      steps=1200,
      settle_steps=10,
      control_dt=0.02,
    )
    fields = contract.invariant_fields()
    self.assertEqual(contract.route_kinds, ROUTE_KINDS)
    self.assertEqual(fields["num_envs"], 8)
    self.assertEqual(fields["action_acceleration_definition"], ACTION_ACCELERATION_DEFINITION)

  def test_route_length_matches_all_three_geometries(self) -> None:
    radius = 2.5
    length = matched_route_length(radius)
    self.assertAlmostEqual(length, radius * 2.0 * math.pi / 3.0)
    self.assertAlmostEqual(length, 2.0 * radius * math.pi / 3.0)

  def test_default_route_and_scan_bounds_fit_flat_patch(self) -> None:
    for kind in ROUTE_KINDS:
      for sign in (-1, 1):
        bounds = matched_route_local_bounds(kind, 4.0, sign)
        self.assertGreaterEqual(min(bounds[0], bounds[2]) - 0.8, 0.0)
        self.assertLessEqual(max(bounds[1], bounds[3]) + 0.8, 16.0)

  def test_evaluator_rejects_scan_footprint_outside_patch(self) -> None:
    cfg = MatchedRouteConfig(checkpoint="model.pt", radii=(8.0,))
    with self.assertRaisesRegex(ValueError, "leaves the 16 m evaluation patch"):
      _validate_config(cfg)

  def test_invalid_contract_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "matched order"):
      MatchedRouteContract(
        checkpoint="x", task_id="task", seed=42, profile="clean",
        num_slots=1, speeds=(0.5,), steps=10, settle_steps=0,
        control_dt=0.02, route_kinds=("arc", "straight", "s_curve"),
      )


class MatchedMetricTests(unittest.TestCase):
  def test_action_acceleration_matches_existing_discrete_definition(self) -> None:
    current = torch.tensor([[3.0, -1.0], [1.0, 2.0]])
    previous = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    older = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = action_acceleration(current, previous, older)
    torch.testing.assert_close(result, torch.tensor([0.5, 0.0]))

  def test_contact_slip_uses_only_contacting_feet(self) -> None:
    velocity = torch.tensor(
      [[[3.0, 4.0], [0.0, 2.0]], [[1.0, 0.0], [2.0, 0.0]]]
    )
    contact = torch.tensor([[True, False], [False, False]])
    torch.testing.assert_close(
      contact_slip_velocity(velocity, contact), torch.tensor([5.0, 0.0])
    )

  def test_online_metrics_freeze_inactive_attempts_and_compute_p95(self) -> None:
    metrics = OnlineMatchedRouteMetrics(2, 3)
    metrics.update(
      sample_mask=torch.tensor([True, True]),
      action_acceleration=torch.tensor([1.0, 10.0]),
      slip_velocity=torch.tensor([2.0, 20.0]),
      velocity_error=torch.tensor([3.0, 30.0]),
      cross_axis_velocity=torch.tensor([4.0, 40.0]),
    )
    metrics.update(
      sample_mask=torch.tensor([False, True]),
      action_acceleration=torch.tensor([100.0, 12.0]),
      slip_velocity=torch.tensor([200.0, 22.0]),
      velocity_error=torch.tensor([300.0, 32.0]),
      cross_axis_velocity=torch.tensor([400.0, 42.0]),
    )
    first = metrics.result(0)
    second = metrics.result(1)
    self.assertEqual(first["action_acceleration"], {"mean": 1.0, "p95": 1.0, "max": 1.0})
    self.assertAlmostEqual(second["action_acceleration"]["mean"], 11.0)
    self.assertAlmostEqual(second["action_acceleration"]["p95"], 11.9)
    self.assertEqual(second["action_acceleration"]["max"], 12.0)

  def test_thresholds_match_decision_contract(self) -> None:
    result = matched_thresholds(
      s_action=1.19, arc_action=1.0, straight_action=1.0,
      s_slip=1.2, reference_slip=1.0, catastrophic_fraction=0.05,
    )
    self.assertTrue(all(result.values()))

  def test_recursive_finite_allows_null_and_rejects_nan(self) -> None:
    assert_recursive_finite({"value": [1.0, None, {"reason": "undefined"}]})
    with self.assertRaisesRegex(ValueError, "root.value"):
      assert_recursive_finite({"value": float("nan")})


class MatchedEvaluatorContractTests(unittest.TestCase):
  def test_scenarios_have_stable_paired_slots(self) -> None:
    cfg = MatchedRouteConfig(
      checkpoint="model.pt", radii=(2.5,), speeds=(0.3, 0.5),
      turn_signs=(1, -1), repeats=2,
    )
    _validate_config(cfg)
    scenarios = _scenarios(cfg)
    self.assertEqual(len(scenarios), 8)
    self.assertEqual(
      [scenario["matched_slot"] for scenario in scenarios], list(range(8))
    )
    self.assertEqual(
      [(item["speed"], item["turn_sign"], item["repeat"]) for item in scenarios[:4]],
      [(0.3, 1, 0), (0.3, 1, 1), (0.3, -1, 0), (0.3, -1, 1)],
    )

  @staticmethod
  def _route_result(kind: str, action: float, slip: float) -> dict[str, object]:
    scenarios = [
      {
        "matched_slot": index,
        "turn_sign": sign,
        "sample_metrics": {
          "action_acceleration": {"mean": action, "p95": action, "max": action},
          "slip_velocity": {"mean": slip, "p95": slip, "max": slip},
        },
      }
      for index, sign in enumerate((1, -1))
    ]
    distribution = lambda value: {"mean": value, "p95": value, "max": value}
    return {
      "route_kind": kind,
      "num_envs": 2,
      "completion_rate": 1.0,
      "catastrophic_termination_fraction": 0.0,
      "action_acceleration": distribution(action),
      "slip_velocity": distribution(slip),
      "velocity_error": distribution(0.1),
      "cross_axis_velocity": distribution(0.01),
      "scenarios": scenarios,
    }

  def test_diagnostics_classify_matched_s_curve(self) -> None:
    route_results = {
      "straight": self._route_result("straight", 1.0, 1.0),
      "arc": self._route_result("arc", 1.05, 1.0),
      "s_curve": self._route_result("s_curve", 1.1, 1.1),
    }
    payload = {
      "schema_version": 1,
      "checkpoint": "model.pt",
      "git_head": "abc",
      "seed": 42,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
      "profiles": {
        name: {
          "matched_invariants": {"num_envs": 2},
          "route_results": route_results,
        }
        for name in ("clean", "full_randomized")
      },
    }
    result = diagnose(payload)
    self.assertEqual(
      result["action_acceleration_attribution"], "not_s_curve_specific"
    )
    self.assertTrue(all(result["full_randomized_thresholds"].values()))
    self.assertEqual(
      result["missing_ablation_profiles"],
      ["dynamics_only", "observation_only", "push_only"],
    )


if __name__ == "__main__":
  unittest.main()
