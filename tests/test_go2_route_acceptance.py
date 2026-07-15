"""Acceptance tests for the parameterized route-evaluation helper contract.

These tests intentionally exercise pure tensor helpers only.  The environment
smoke/evaluator commands are documented in the handoff to the integration
agent and are not run from this CPU-only acceptance worktree.
"""

from __future__ import annotations

import math
import unittest

import torch

from src.tasks.velocity.evaluation.routes import (
  route_frame_errors,
  straight_line_controller,
  update_attempt_status,
  world_to_body_velocity,
  wrap_to_pi,
)


class RouteFrameTransformTest(unittest.TestCase):
  def assertTensorClose(
    self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6
  ) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

  def test_wrap_to_pi_handles_both_boundaries(self) -> None:
    angles = torch.tensor(
      [-math.pi - 1e-5, -math.pi + 1e-5, math.pi - 1e-5, math.pi, math.pi + 1e-5]
    )
    actual = wrap_to_pi(angles)
    expected = torch.tensor(
      [math.pi - 1e-5, -math.pi + 1e-5, math.pi - 1e-5, -math.pi, -math.pi + 1e-5]
    )
    self.assertTensorClose(actual, expected, atol=2e-6)
    self.assertTrue(bool(torch.all(actual >= -math.pi)))
    self.assertTrue(bool(torch.all(actual < math.pi)))

  def test_route_frame_signs_at_zero_and_quarter_turns(self) -> None:
    # A point one metre to the route's left has positive cross-track error.
    state = route_frame_errors(
      torch.tensor([[2.0, 1.0], [2.0, -1.0]]),
      torch.tensor([0.0, 0.2]),
      torch.zeros((2, 2)),
      torch.tensor([0.0, 0.0]),
    )
    self.assertTensorClose(state.progress, torch.tensor([2.0, 2.0]))
    self.assertTensorClose(state.cross_track, torch.tensor([1.0, -1.0]))
    self.assertTensorClose(state.heading_error, torch.tensor([0.0, 0.2]))

    # For a +pi/2 route, left is world -x; for -pi/2, left is world +x.
    plus = route_frame_errors(
      torch.tensor([[1.0, 2.0]]), torch.tensor([math.pi / 2]),
      torch.zeros((1, 2)), math.pi / 2,
    )
    minus = route_frame_errors(
      torch.tensor([[1.0, 2.0]]), torch.tensor([-math.pi / 2]),
      torch.zeros((1, 2)), -math.pi / 2,
    )
    self.assertTensorClose(plus.progress, torch.tensor([2.0]))
    self.assertTensorClose(plus.cross_track, torch.tensor([-1.0]))
    self.assertTensorClose(minus.progress, torch.tensor([-2.0]))
    self.assertTensorClose(minus.cross_track, torch.tensor([1.0]))

  def test_world_to_body_rotation(self) -> None:
    world = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    heading = torch.tensor([0.0, math.pi / 2, -math.pi / 2])
    actual = world_to_body_velocity(world, heading)
    expected = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-2.0, 1.0]])
    self.assertTensorClose(actual, expected)


class RouteControllerTest(unittest.TestCase):
  def test_zero_error_tracks_forward_and_completion_outputs_zero(self) -> None:
    position = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    heading = torch.zeros(2)
    command = straight_line_controller(
      position,
      heading,
      torch.zeros((2, 2)),
      0.0,
      target_speed=0.6,
      cross_track_gain=1.0,
      heading_gain=1.0,
      max_lateral_speed=0.4,
      max_yaw_rate=1.0,
      route_length=2.0,
    )
    torch.testing.assert_close(command[0], torch.tensor([0.6, 0.0, 0.0]))
    torch.testing.assert_close(command[1], torch.zeros(3))

  def test_cross_track_and_heading_corrections_have_stabilizing_signs(self) -> None:
    command = straight_line_controller(
      torch.tensor([[0.0, 0.5], [0.0, -0.5], [0.0, 0.0]]),
      torch.tensor([0.2, -0.2, 0.0]),
      torch.zeros((3, 2)),
      0.0,
      target_speed=0.6,
      cross_track_gain=1.0,
      heading_gain=1.0,
      max_lateral_speed=0.4,
      max_yaw_rate=1.0,
      route_length=10.0,
    )
    self.assertLess(float(command[0, 1]), 0.0)
    self.assertGreater(float(command[1, 1]), 0.0)
    self.assertLess(float(command[0, 2]), 0.0)
    self.assertGreater(float(command[1, 2]), 0.0)

  def test_batch_broadcast_preserves_dtype_device_and_shape(self) -> None:
    position = torch.tensor([[0.0, 0.0], [0.0, 0.2]], dtype=torch.float64)
    command = straight_line_controller(
      position,
      torch.zeros(2, dtype=torch.float64),
      torch.zeros((2, 2), dtype=torch.float64),
      torch.tensor(0.0, dtype=torch.float64),
      target_speed=torch.tensor([0.4, 0.5], dtype=torch.float64),
      cross_track_gain=2.0,
      heading_gain=1.0,
      max_lateral_speed=0.3,
      max_yaw_rate=1.0,
      route_length=2.0,
    )
    self.assertEqual(command.shape, (2, 3))
    self.assertEqual(command.dtype, position.dtype)
    self.assertEqual(command.device, position.device)
    self.assertAlmostEqual(float(command[1, 1]), -0.3)

  def test_invalid_shapes_and_limits_raise_value_error(self) -> None:
    common = dict(
      target_speed=0.5,
      cross_track_gain=1.0,
      heading_gain=1.0,
      max_lateral_speed=0.3,
      max_yaw_rate=1.0,
      route_length=2.0,
    )
    with self.assertRaises(ValueError):
      straight_line_controller(torch.zeros(2), torch.zeros(1), torch.zeros((1, 2)), 0.0, **common)
    for key, value in (("max_lateral_speed", 0.0), ("max_yaw_rate", -1.0), ("route_length", 0.0)):
      bad = dict(common)
      bad[key] = value
      with self.subTest(parameter=key), self.assertRaises(ValueError):
        straight_line_controller(
          torch.zeros((1, 2)), torch.zeros(1), torch.zeros((1, 2)), 0.0, **bad
        )


class AttemptStateTest(unittest.TestCase):
  def test_completion_requires_position_and_heading_tolerances(self) -> None:
    result = update_attempt_status(
      torch.tensor([True, True, True]),
      torch.tensor([2.0, 2.0, 1.9]),
      torch.tensor([0.01, 0.20, 0.01]),
      torch.tensor([0.01, 0.01, 0.01]),
      torch.tensor([False, False, False]),
      route_length=2.0,
      cross_track_tolerance=0.05,
      heading_tolerance=0.05,
    )
    self.assertTrue(bool(result.completed_now[0]))
    self.assertFalse(bool(result.completed_now[1]))
    self.assertFalse(bool(result.completed_now[2]))
    self.assertFalse(bool(result.active[0]))
    self.assertTrue(bool(result.active[1]))

  def test_failure_precedes_simultaneous_completion(self) -> None:
    result = update_attempt_status(
      torch.tensor([True]),
      torch.tensor([2.0]),
      torch.tensor([0.0]),
      torch.tensor([0.0]),
      torch.tensor([True]),
      route_length=2.0,
      cross_track_tolerance=0.05,
      heading_tolerance=0.05,
    )
    self.assertTrue(bool(result.failed_now[0]))
    self.assertFalse(bool(result.completed_now[0]))
    self.assertFalse(bool(result.active[0]))

  def test_inactive_attempts_are_frozen_and_cannot_reactivate(self) -> None:
    result = update_attempt_status(
      torch.tensor([False, True, False]),
      torch.tensor([9.0, 2.0, 9.0]),
      torch.zeros(3),
      torch.zeros(3),
      torch.tensor([False, False, True]),
      route_length=2.0,
      cross_track_tolerance=0.05,
      heading_tolerance=0.05,
    )
    self.assertTrue(torch.equal(result.sample_mask, torch.tensor([False, True, False])))
    self.assertFalse(bool(result.completed_now[0]))
    self.assertFalse(bool(result.failed_now[0]))
    self.assertFalse(bool(result.active[0]))
    self.assertFalse(bool(result.active[2]))
    self.assertTrue(bool(result.completed_now[1]))


if __name__ == "__main__":
  unittest.main()
