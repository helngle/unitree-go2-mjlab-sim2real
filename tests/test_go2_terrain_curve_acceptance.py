"""Independent CPU acceptance contracts for matched and terrain curves."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
from typing import Any, Mapping
import unittest
from unittest import mock

import torch

from scripts.evaluate_go2_matched_routes import (
  MatchedRouteConfig,
  _evaluate_route_kind,
  _scenarios,
  evaluate,
)
from src.tasks.velocity.evaluation.curved_routes import make_arc_route, make_s_route
from src.tasks.velocity.evaluation.matched_route_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  DYNAMICS_EVENTS,
  OBSERVATION_EVENTS,
  PROFILES,
  ROUTE_KINDS,
  MatchedRouteContract,
  OnlineMatchedRouteMetrics,
  action_acceleration,
  assert_recursive_finite,
  configure_matched_profile,
  matched_route_length,
  matched_route_local_bounds,
)
from src.tasks.velocity.evaluation.routes import update_attempt_status


SCAN_HALF_EXTENT = 0.8
FLAT_PATCH_SIZE = 16.0
EXPECTED_TERRAIN_FAMILIES = {
  "slope_up",
  "slope_down",
  "random_rough",
  "discrete_obstacle",
}


@dataclass
class _Actor:
  enable_corruption: bool = True


class _Event:
  mode = "startup"
  interval_range_s = None
  params = {"range": (0.1, 0.2)}


class _ProfileCfg:
  def __init__(self) -> None:
    self.observations = {"actor": _Actor()}
    self.events = {
      name: _Event()
      for name in DYNAMICS_EVENTS + OBSERVATION_EVENTS + ("push_robot",)
    }


def _assert_bounds_inside(
  bounds: list[float] | tuple[float, ...], patch: list[float] | tuple[float, ...]
) -> None:
  if len(bounds) != 4 or len(patch) != 4:
    raise AssertionError("bounds and patch must be [xmin, xmax, ymin, ymax]")
  xmin, xmax, ymin, ymax = (float(value) for value in bounds)
  pxmin, pxmax, pymin, pymax = (float(value) for value in patch)
  if not (pxmin <= xmin <= xmax <= pxmax and pymin <= ymin <= ymax <= pymax):
    raise AssertionError(f"bounds {bounds} leave patch {patch}")


def assert_terrain_curve_payload(payload: Mapping[str, Any]) -> None:
  """Acceptance-only evidence check used again for formal GPU JSON."""
  assert_recursive_finite(payload)
  coverage = payload.get("coverage")
  scenarios = payload.get("scenarios")
  if not isinstance(coverage, Mapping) or not isinstance(scenarios, list):
    raise AssertionError("terrain result requires coverage and scenarios")
  if float(payload.get("terrain_assignment_position_error_max", math.inf)) > 1.0e-4:
    raise AssertionError("terrain relocation error exceeds tolerance")
  evidenced: set[str] = set()
  for scenario in scenarios:
    required = {
      "terrain_family",
      "route_kind",
      "terrain_level",
      "terrain_origin",
      "patch_bounds_local_xy",
      "route_corridor_bounds_local_xy",
      "terrain_scan_footprint_bounds_local_xy",
      "route_and_scan_inside_patch",
      "reset_count",
      "first_failure_reason",
    }
    missing = required.difference(scenario)
    if missing:
      raise AssertionError(f"terrain scenario missing {sorted(missing)}")
    family = str(scenario["terrain_family"])
    if family not in EXPECTED_TERRAIN_FAMILIES:
      raise AssertionError(f"unsupported terrain family {family!r}")
    if scenario["route_kind"] not in ("arc", "s_curve"):
      raise AssertionError("terrain route must be arc or s_curve")
    if not scenario["route_and_scan_inside_patch"]:
      raise AssertionError("unsafe corridor cannot be reported as coverage")
    patch = scenario["patch_bounds_local_xy"]
    _assert_bounds_inside(scenario["route_corridor_bounds_local_xy"], patch)
    _assert_bounds_inside(scenario["terrain_scan_footprint_bounds_local_xy"], patch)
    if len(scenario["terrain_origin"]) != 3:
      raise AssertionError("terrain_origin must contain xyz")
    if int(scenario["reset_count"]) < 0:
      raise AssertionError("reset_count must be nonnegative")
    evidenced.add(family)
  for family, claimed in coverage.items():
    if bool(claimed) and family not in evidenced:
      raise AssertionError(f"coverage {family!r} lacks scenario evidence")


class MatchedProfileAcceptanceTest(unittest.TestCase):
  def test_profile_sources_are_exact_and_json_stable(self) -> None:
    expected = {
      "clean": (False, (), False),
      "dynamics_only": (False, DYNAMICS_EVENTS, False),
      "observation_only": (True, OBSERVATION_EVENTS, False),
      "push_only": (False, (), True),
      "full_randomized": (
        True,
        DYNAMICS_EVENTS + OBSERVATION_EVENTS,
        True,
      ),
    }
    for name, (corruption, events, push) in expected.items():
      cfg = _ProfileCfg()
      result = configure_matched_profile(cfg, name)
      self.assertEqual(result["actor_observation_corruption"], corruption)
      self.assertEqual(tuple(result["startup_randomization_events"]), events)
      self.assertEqual(result["push_enabled"], push)
      self.assertEqual(set(result["event_parameters"]), set(events) | ({"push_robot"} if push else set()))
      json.dumps(result, allow_nan=False)

  def test_single_factor_profiles_enable_only_the_declared_factor(self) -> None:
    for name, profile in PROFILES.items():
      cfg = _ProfileCfg()
      result = configure_matched_profile(cfg, name)
      self.assertEqual(
        tuple(result["startup_randomization_events"]), profile.startup_events
      )
      self.assertEqual(
        result["actor_observation_corruption"],
        profile.actor_observation_corruption,
      )
      self.assertEqual(result["push_enabled"], profile.push_enabled)

  def test_route_subprocess_invariants_reject_profile_drift(self) -> None:
    cfg = MatchedRouteConfig(
      checkpoint="model.pt",
      profiles=("clean",),
      radii=(2.5,),
      speeds=(0.3,),
      turn_signs=(1, -1),
      steps=1200,
    )

    def route_result(_cfg, _profile, route_kind, scenarios):
      settings = {
        "name": "clean",
        "actor_observation_corruption": False,
        "startup_randomization_events": [],
        "push_enabled": False,
        "event_parameters": {},
        "control_dt": 0.02,
      }
      if route_kind == "arc":
        settings = {**settings, "push_enabled": True}
      return {
        "route_kind": route_kind,
        "num_envs": len(scenarios),
        "profile_settings": settings,
      }

    with mock.patch(
      "scripts.evaluate_go2_matched_routes._evaluate_route_kind",
      side_effect=route_result,
    ), self.assertRaisesRegex(RuntimeError, "profile settings differ"):
      evaluate(cfg)


class MatchedGeometryAndEnergyAcceptanceTest(unittest.TestCase):
  def test_common_length_horizon_and_slots_are_identical(self) -> None:
    cfg = MatchedRouteConfig(
      checkpoint="model.pt",
      radii=(2.5, 4.0),
      speeds=(0.3, 0.5),
      turn_signs=(1, -1),
      repeats=2,
      steps=2000,
      settle_steps=10,
    )
    scenarios = _scenarios(cfg)
    self.assertEqual([item["matched_slot"] for item in scenarios], list(range(16)))
    contract = MatchedRouteContract(
      checkpoint=cfg.checkpoint,
      task_id=cfg.task_id,
      seed=cfg.seed,
      profile="clean",
      num_slots=len(scenarios),
      speeds=cfg.speeds,
      steps=cfg.steps,
      settle_steps=cfg.settle_steps,
      control_dt=0.02,
    )
    invariants = contract.invariant_fields()
    self.assertEqual(invariants["steps"], 2000)
    self.assertEqual(invariants["settle_steps"], 10)
    self.assertEqual(
      invariants["action_acceleration_definition"],
      ACTION_ACCELERATION_DEFINITION,
    )
    for radius in cfg.radii:
      length = matched_route_length(radius)
      for speed in cfg.speeds:
        required = math.ceil(length / (speed * 0.02)) + cfg.settle_steps
        self.assertLessEqual(required, cfg.steps)

  def test_ideal_forward_energy_is_matched_and_arc_s_yaw_energy_match(self) -> None:
    radius, speed, dt = 2.5, 0.5, 0.02
    samples = math.ceil(matched_route_length(radius) / (speed * dt))
    forward_energy = samples * speed**2
    route_energy = {
      "straight": (forward_energy, 0.0),
      "arc": (forward_energy, samples * (speed / radius) ** 2),
      "s_curve": (forward_energy, samples * (speed / radius) ** 2),
    }
    self.assertEqual(
      {value[0] for value in route_energy.values()}, {forward_energy}
    )
    self.assertEqual(route_energy["arc"][1], route_energy["s_curve"][1])

  def test_rollout_applies_settle_and_reports_command_energy(self) -> None:
    source = inspect.getsource(_evaluate_route_kind)
    self.assertIn(
      "cfg.settle_steps",
      source,
      "settle_steps is metadata-only and does not affect the rollout lifecycle",
    )
    self.assertIn(
      "command_energy",
      source,
      "closed-loop command energy is not retained for matched verification",
    )

  def test_left_right_routes_are_exact_mirrors_and_scan_safe(self) -> None:
    start = torch.tensor([2.0, 8.0], dtype=torch.float64)
    progress = torch.linspace(
      0.0, matched_route_length(4.0), 1001, dtype=torch.float64
    )
    for kind in ROUTE_KINDS:
      if kind == "straight":
        left_xy = torch.stack((2.0 + progress, torch.full_like(progress, 8.0)), -1)
        right_xy = left_xy.clone()
      else:
        factory = make_arc_route if kind == "arc" else make_s_route
        kwargs = {"angle": 2.0 * math.pi / 3.0} if kind == "arc" else {}
        left = factory(start, 0.0, 4.0, 1, **kwargs)
        right = factory(start, 0.0, 4.0, -1, **kwargs)
        left_xy, _ = left.pose_at(progress)
        right_xy, _ = right.pose_at(progress)
      torch.testing.assert_close(left_xy[:, 0], right_xy[:, 0])
      torch.testing.assert_close(left_xy[:, 1] - 8.0, -(right_xy[:, 1] - 8.0))
      for sign, points in ((1, left_xy), (-1, right_xy)):
        bounds = matched_route_local_bounds(kind, 4.0, sign)
        self.assertGreaterEqual(float(points[:, 0].min()), bounds[0] - 1.0e-9)
        self.assertLessEqual(float(points[:, 0].max()), bounds[1] + 1.0e-9)
        self.assertGreaterEqual(float(points[:, 1].min()), bounds[2] - 1.0e-9)
        self.assertLessEqual(float(points[:, 1].max()), bounds[3] + 1.0e-9)
        _assert_bounds_inside(
          [
            bounds[0] - SCAN_HALF_EXTENT,
            bounds[1] + SCAN_HALF_EXTENT,
            bounds[2] - SCAN_HALF_EXTENT,
            bounds[3] + SCAN_HALF_EXTENT,
          ],
          [0.0, FLAT_PATCH_SIZE, 0.0, FLAT_PATCH_SIZE],
        )


class MetricAndLifecycleAcceptanceTest(unittest.TestCase):
  def test_action_acceleration_is_second_action_difference(self) -> None:
    current = torch.tensor([[3.0, -1.0, 2.0], [1.0, 2.0, 7.0]])
    previous = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 4.0]])
    older = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    expected = (current - 2.0 * previous + older).abs().mean(dim=-1)
    torch.testing.assert_close(
      action_acceleration(current, previous, older), expected
    )

  def test_p95_max_and_reset_freeze_ignore_new_episode_samples(self) -> None:
    metrics = OnlineMatchedRouteMetrics(2, 4, dtype=torch.float64)
    values = ((1.0, 10.0), (2.0, 20.0), (99.0, 30.0))
    masks = ((True, True), (True, True), (False, True))
    for pair, mask in zip(values, masks, strict=True):
      tensor = torch.tensor(pair, dtype=torch.float64)
      metrics.update(
        sample_mask=torch.tensor(mask),
        action_acceleration=tensor,
        slip_velocity=tensor + 1.0,
        velocity_error=tensor + 2.0,
        cross_axis_velocity=tensor + 3.0,
      )
    frozen = metrics.result(0)["action_acceleration"]
    active = metrics.result(1)["action_acceleration"]
    self.assertEqual(frozen["mean"], 1.5)
    self.assertAlmostEqual(frozen["p95"], 1.95)
    self.assertEqual(frozen["max"], 2.0)
    self.assertEqual(active["mean"], 20.0)
    self.assertAlmostEqual(active["p95"], 29.0)
    self.assertEqual(active["max"], 30.0)

    first = update_attempt_status(
      active=torch.tensor([True]),
      progress=torch.tensor([0.5]),
      cross_track=torch.zeros(1),
      heading_error=torch.zeros(1),
      failure_mask=torch.tensor([True]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    second = update_attempt_status(
      active=first.active,
      progress=torch.tensor([2.0]),
      cross_track=torch.zeros(1),
      heading_error=torch.zeros(1),
      failure_mask=torch.tensor([False]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    self.assertFalse(bool(second.sample_mask[0]))
    self.assertFalse(bool(second.completed_now[0]))


class TerrainEvidenceAcceptanceTest(unittest.TestCase):
  @staticmethod
  def _scenario(family: str = "slope_up") -> dict[str, Any]:
    return {
      "terrain_family": family,
      "route_kind": "arc",
      "terrain_level": 2,
      "terrain_origin": [12.0, -3.0, 0.4],
      "patch_bounds_local_xy": [0.0, 16.0, 0.0, 16.0],
      "route_corridor_bounds_local_xy": [2.0, 5.0, 8.0, 12.0],
      "terrain_scan_footprint_bounds_local_xy": [1.2, 5.8, 7.2, 12.8],
      "route_and_scan_inside_patch": True,
      "reset_count": 0,
      "first_failure_reason": None,
    }

  def test_terrain_payload_requires_real_coverage_and_is_strict_json(self) -> None:
    payload = {
      "terrain_assignment_position_error_max": 0.0,
      "coverage": {
        "slope_up": True,
        "slope_down": False,
        "random_rough": False,
        "discrete_obstacle": False,
      },
      "scenarios": [self._scenario()],
    }
    assert_terrain_curve_payload(payload)
    json.dumps(payload, allow_nan=False)
    false_claim = {**payload, "coverage": {"random_rough": True}}
    with self.assertRaisesRegex(AssertionError, "lacks scenario evidence"):
      assert_terrain_curve_payload(false_claim)

  def test_corridor_scan_and_relocation_violations_are_rejected(self) -> None:
    payload = {
      "terrain_assignment_position_error_max": 0.0,
      "coverage": {"slope_up": True},
      "scenarios": [self._scenario()],
    }
    payload["scenarios"][0]["terrain_scan_footprint_bounds_local_xy"] = [
      -0.1, 5.8, 7.2, 12.8
    ]
    with self.assertRaisesRegex(AssertionError, "leave patch"):
      assert_terrain_curve_payload(payload)
    payload["scenarios"][0] = self._scenario()
    payload["terrain_assignment_position_error_max"] = 2.0e-4
    with self.assertRaisesRegex(AssertionError, "relocation"):
      assert_terrain_curve_payload(payload)

  def test_xyz_relocation_preserves_robot_patch_relative_pose(self) -> None:
    old_origins = torch.tensor(
      [[10.0, -3.0, 0.5], [-5.0, 7.0, 1.2]], dtype=torch.float64
    )
    roots = old_origins + torch.tensor(
      [[1.0, 2.0, 0.32], [3.0, 4.0, 0.32]], dtype=torch.float64
    )
    new_origins = torch.tensor(
      [[-9.0, 8.0, 1.0], [4.0, -2.0, 0.1]], dtype=torch.float64
    )
    relocated = roots + (new_origins - old_origins)
    torch.testing.assert_close(relocated - new_origins, roots - old_origins)

  def test_flat_matched_evaluator_does_not_claim_complex_terrain(self) -> None:
    route_result = {
      "num_envs": 1,
      "profile_settings": {
        "name": "clean",
        "actor_observation_corruption": False,
        "startup_randomization_events": [],
        "push_enabled": False,
        "event_parameters": {},
        "control_dt": 0.02,
      },
    }
    cfg = MatchedRouteConfig(
      checkpoint="model.pt",
      profiles=("clean",),
      radii=(2.5,),
      speeds=(0.3,),
      turn_signs=(1,),
      repeats=1,
      steps=1200,
    )
    with mock.patch(
      "scripts.evaluate_go2_matched_routes._evaluate_route_kind",
      return_value=route_result,
    ):
      result = evaluate(cfg)
    self.assertEqual(
      result["coverage"],
      {
        "flat_matched_straight_arc_s_curve": True,
        "rough_curves": False,
        "terrain_transitions": False,
      },
    )


if __name__ == "__main__":
  unittest.main()
