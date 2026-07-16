"""Independent pre-training acceptance contracts for curved locomotion.

The tests in this module deliberately avoid a GPU environment.  They exercise
the production curriculum and the pure curved-route lifecycle directly, then
record the matched control/probe contract that a future curve sampler must
satisfy.  Production sampler wiring is intentionally not guessed here.
"""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace
import unittest

import torch

from src.tasks.velocity.evaluation.curved_routes import (
  make_command_tape_schedule,
)
from src.tasks.velocity.evaluation.routes import update_attempt_status
from src.tasks.velocity.mdp.curriculums import terrain_levels_vel
from src.tasks.velocity.mdp.mode_velocity_command import ModeVelocityCommandCfg


class _FakeScene:
  def __init__(self, asset: SimpleNamespace, terrain: "_FakeTerrain") -> None:
    self._asset = asset
    self.terrain = terrain
    self.env_origins = torch.zeros((1, 3))

  def __getitem__(self, name: str) -> SimpleNamespace:
    if name != "robot":
      raise KeyError(name)
    return self._asset


class _FakeTerrain:
  def __init__(self, size_x: float) -> None:
    generator = SimpleNamespace(size=(size_x, size_x))
    self.cfg = SimpleNamespace(terrain_generator=generator)
    self.terrain_levels = torch.zeros(1, dtype=torch.long)
    self.last_move_up: torch.Tensor | None = None
    self.last_move_down: torch.Tensor | None = None

  def update_env_origins(
    self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
  ) -> None:
    del env_ids
    self.last_move_up = move_up.clone()
    self.last_move_down = move_down.clone()


def _production_curriculum_decision(
  final_xy: tuple[float, float],
  command_xy: tuple[float, float],
  *,
  episode_length_s: float,
  terrain_size_x: float,
) -> tuple[bool, bool]:
  """Call the real curriculum with a minimal one-environment test double."""
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[final_xy[0], final_xy[1], 0.0]])
    )
  )
  terrain = _FakeTerrain(terrain_size_x)
  command = torch.tensor([[command_xy[0], command_xy[1], 0.0]])
  env = SimpleNamespace(
    scene=_FakeScene(asset, terrain),
    command_manager=SimpleNamespace(get_command=lambda _: command),
    max_episode_length_s=episode_length_s,
  )
  terrain_levels_vel(env, torch.tensor([0]), "twist")
  assert terrain.last_move_up is not None and terrain.last_move_down is not None
  return bool(terrain.last_move_up[0]), bool(terrain.last_move_down[0])


def _assert_matched_terrain_assignments(
  testcase: unittest.TestCase,
  control_levels: torch.Tensor,
  control_types: torch.Tensor,
  probe_levels: torch.Tensor,
  probe_types: torch.Tensor,
) -> None:
  """Contract adapter for future control/probe terrain state snapshots."""
  testcase.assertTrue(torch.equal(control_levels, probe_levels))
  testcase.assertTrue(torch.equal(control_types, probe_types))


def _assert_curve_probability_is_only_sampler_change(
  testcase: unittest.TestCase,
  control: dict[str, float],
  probe: dict[str, float],
) -> None:
  """Validate the agreed 15% curve quota without naming a production API."""
  testcase.assertEqual(
    set(control), {"general", "lateral", "yaw", "high_speed", "curve"}
  )
  testcase.assertEqual(set(control), set(probe))
  testcase.assertAlmostEqual(sum(control.values()), 1.0)
  testcase.assertAlmostEqual(sum(probe.values()), 1.0)
  testcase.assertEqual(control["curve"], 0.0)
  testcase.assertEqual(probe["curve"], 0.15)
  testcase.assertAlmostEqual(control["general"] - probe["general"], 0.15)
  for unchanged in ("lateral", "yaw", "high_speed"):
    testcase.assertEqual(control[unchanged], probe[unchanged])


class CurriculumEndpointDisplacementRiskTest(unittest.TestCase):
  def test_perfect_full_circle_is_downgraded_while_equal_length_straight_moves_up(self) -> None:
    radius = 2.0
    speed = 0.5
    traveled = 2.0 * math.pi * radius
    duration = traveled / speed

    straight = _production_curriculum_decision(
      (traveled, 0.0),
      (speed, 0.0),
      episode_length_s=duration,
      terrain_size_x=8.0,
    )
    full_circle = _production_curriculum_decision(
      (0.0, 0.0),
      (speed, 0.0),
      episode_length_s=duration,
      terrain_size_x=8.0,
    )

    self.assertEqual(straight, (True, False))
    self.assertEqual(full_circle, (False, True))

  def test_equal_arc_length_has_smaller_curriculum_progress_than_straight(self) -> None:
    radius = 3.0
    arc_length = math.pi * radius / 2.0
    arc_chord = math.sqrt(2.0) * radius
    speed = 0.5
    duration = arc_length / speed
    terrain_size_x = 9.0

    straight = _production_curriculum_decision(
      (arc_length, 0.0),
      (speed, 0.0),
      episode_length_s=duration,
      terrain_size_x=terrain_size_x,
    )
    quarter_arc = _production_curriculum_decision(
      (radius, radius),
      (speed, 0.0),
      episode_length_s=duration,
      terrain_size_x=terrain_size_x,
    )

    self.assertGreater(arc_length, terrain_size_x / 2.0)
    self.assertLess(arc_chord, terrain_size_x / 2.0)
    self.assertEqual(straight, (True, False))
    self.assertEqual(quarter_arc, (False, False))

  def test_pure_yaw_has_no_curriculum_success_or_failure_signal(self) -> None:
    decision = _production_curriculum_decision(
      (0.0, 0.0),
      (0.0, 0.0),
      episode_length_s=20.0,
      terrain_size_x=8.0,
    )
    self.assertEqual(decision, (False, False))


class CurveDistributionBoundaryTest(unittest.TestCase):
  def test_required_yaw_rates_are_split_at_v7_general_limit(self) -> None:
    general_range = ModeVelocityCommandCfg.__dataclass_fields__[
      "general_ang_vel_z"
    ].default
    yaw_limit = max(abs(value) for value in general_range)
    self.assertEqual(yaw_limit, 0.3)

    observed: dict[tuple[float, float], bool] = {}
    for radius in (1.5, 2.5, 4.0):
      for speed in (0.3, 0.5, 0.6):
        observed[(radius, speed)] = speed / radius <= yaw_limit + 1.0e-8

    expected_ood = {(1.5, 0.5), (1.5, 0.6)}
    actual_ood = {case for case, is_id in observed.items() if not is_id}
    self.assertEqual(actual_ood, expected_ood)
    self.assertTrue(observed[(1.5, 0.3)])
    self.assertTrue(0.3 <= yaw_limit + 1.0e-8)

  def test_left_and_right_have_equal_distribution_classification(self) -> None:
    yaw_limit = 0.3
    for sign in (-1, 1):
      for radius in (1.5, 2.5, 4.0):
        for speed in (0.3, 0.5, 0.6):
          required = sign * speed / radius
          self.assertEqual(
            abs(required) <= yaw_limit + 1.0e-8,
            speed / radius <= yaw_limit + 1.0e-8,
          )


class TapeAndAttemptIsolationTest(unittest.TestCase):
  def test_command_tape_public_inputs_and_evaluator_are_step_only(self) -> None:
    from scripts.evaluate_go2_curved_routes import _evaluate_scenarios

    schedule = make_command_tape_schedule(
      "s_curve", 2.0, 0.5, 1, 0.02, settle_steps=10
    )
    parameters = inspect.signature(schedule.command_at).parameters
    self.assertEqual(tuple(parameters), ("step_index", "device", "dtype"))

    step = schedule.first_motion_steps
    torch.testing.assert_close(
      schedule.command_at(step), torch.tensor([0.5, 0.0, -0.25]),
      rtol=0.0, atol=1.0e-7,
    )

    tree = ast.parse(inspect.getsource(_evaluate_scenarios))
    calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "command_at"
    ]
    self.assertEqual(len(calls), 1)
    self.assertEqual(len(calls[0].args), 1)
    self.assertIsInstance(calls[0].args[0], ast.Name)
    self.assertEqual(calls[0].args[0].id, "step_index")
    self.assertEqual({item.arg for item in calls[0].keywords}, {"device", "dtype"})

  def test_reset_freezes_attempt_against_new_episode_progress(self) -> None:
    first = update_attempt_status(
      active=torch.tensor([True]),
      progress=torch.tensor([0.4]),
      cross_track=torch.tensor([0.0]),
      heading_error=torch.tensor([0.0]),
      failure_mask=torch.tensor([True]),
      route_length=1.0,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    self.assertTrue(bool(first.failed_now[0]))
    self.assertFalse(bool(first.active[0]))

    after_reset = update_attempt_status(
      active=first.active,
      progress=torch.tensor([10.0]),
      cross_track=torch.tensor([0.0]),
      heading_error=torch.tensor([0.0]),
      failure_mask=torch.tensor([False]),
      route_length=1.0,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    self.assertFalse(bool(after_reset.sample_mask[0]))
    self.assertFalse(bool(after_reset.completed_now[0]))
    self.assertFalse(bool(after_reset.failed_now[0]))
    self.assertFalse(bool(after_reset.active[0]))


class FutureMatchedExperimentContractTest(unittest.TestCase):
  def test_agreed_sampler_probability_contract(self) -> None:
    _assert_curve_probability_is_only_sampler_change(
      self,
      control={
        "general": 0.40,
        "lateral": 0.25,
        "yaw": 0.15,
        "high_speed": 0.20,
        "curve": 0.0,
      },
      probe={
        "general": 0.25,
        "lateral": 0.25,
        "yaw": 0.15,
        "high_speed": 0.20,
        "curve": 0.15,
      },
    )

  def test_control_and_probe_terrain_snapshots_must_match_exactly(self) -> None:
    levels = torch.tensor([0, 2, 5, 9], dtype=torch.long)
    terrain_types = torch.tensor([1, 3, 3, 7], dtype=torch.long)
    _assert_matched_terrain_assignments(
      self, levels, terrain_types, levels.clone(), terrain_types.clone()
    )
    with self.assertRaises(AssertionError):
      _assert_matched_terrain_assignments(
        self,
        levels,
        terrain_types,
        torch.tensor([0, 2, 4, 9]),
        terrain_types.clone(),
      )

  @unittest.skip(
    "Enable after Integration defines the production curve-sampler and frozen-terrain APIs"
  )
  def test_production_sampler_and_frozen_terrain_wiring(self) -> None:
    # Deliberately no speculative import or attribute name: Integration should
    # replace this body with adapters to the reviewed production interfaces.
    self.fail("production integration contract has not been wired")


if __name__ == "__main__":
  unittest.main()
