"""Acceptance contract tests for curved route geometry and controllers.

These tests intentionally exercise the pure tensor API only.  They keep the
route evaluator honest before any GPU rollout is attempted: signed curvature,
world/body transforms, endpoint geometry, and attempt lifecycle must all be
correct independently of a policy checkpoint.
"""

from __future__ import annotations

import math
import unittest

import torch

from src.tasks.velocity.evaluation.curved_routes import (
  ArcSpec,
  arc_command_controller,
  arc_route_errors,
  make_arc_route,
  make_s_route,
  s_command_controller,
)
from src.tasks.velocity.evaluation.routes import (
  update_attempt_status,
  world_to_body_velocity,
)


class ArcGeometryAcceptanceTest(unittest.TestCase):
  def assertTensorClose(
    self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6
  ) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

  def test_quarter_turn_endpoint_tangent_and_length(self) -> None:
    left = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    right = make_arc_route(torch.zeros(2), 0.0, 2.0, -1)
    self.assertAlmostEqual(left.length, math.pi, places=6)
    self.assertAlmostEqual(right.length, math.pi, places=6)

    left_xy, left_h = left.pose_at(left.length)
    right_xy, right_h = right.pose_at(right.length)
    self.assertTensorClose(left_xy, torch.tensor([2.0, 2.0]))
    self.assertTensorClose(right_xy, torch.tensor([2.0, -2.0]))
    self.assertAlmostEqual(float(left_h), math.pi / 2.0, places=6)
    self.assertAlmostEqual(float(right_h), -math.pi / 2.0, places=6)

  def test_left_right_mirror_and_signed_curvature(self) -> None:
    left = make_arc_route(torch.tensor([0.4, -0.7]), 0.0, 2.5, 1)
    right = make_arc_route(torch.tensor([0.4, -0.7]), 0.0, 2.5, -1)
    self.assertAlmostEqual(left.spec.curvature, 0.4, places=7)
    self.assertAlmostEqual(right.spec.curvature, -0.4, places=7)
    left_xy, left_h = left.pose_at(left.length)
    right_xy, right_h = right.pose_at(right.length)
    self.assertTensorClose(left_xy[0], right_xy[0])
    self.assertTensorClose(left_xy[1], 2.0 * torch.tensor(-0.7) - right_xy[1])
    self.assertAlmostEqual(float(left_h), -float(right_h), places=6)

  def test_command_tape_uses_vx_and_wz_v_over_r(self) -> None:
    left = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    right = make_arc_route(torch.zeros(2), 0.0, 2.0, -1)
    self.assertTensorClose(left.command_tape(0.5), torch.tensor([0.5, 0.0, 0.25]))
    self.assertTensorClose(right.command_tape(0.5), torch.tensor([0.5, 0.0, -0.25]))
    batch = left.command_tape(torch.tensor([0.3, 0.6]))
    self.assertEqual(batch.shape, (2, 3))
    self.assertTensorClose(batch[:, 2], torch.tensor([0.15, 0.30]))

  def test_arc_error_and_controller_are_zero_at_endpoint(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    endpoint, endpoint_h = route.pose_at(route.length)
    progress, cross, heading_error = arc_route_errors(
      route, endpoint.unsqueeze(0), endpoint_h.unsqueeze(0)
    )
    self.assertTensorClose(progress, torch.tensor([route.length]))
    self.assertTensorClose(cross, torch.zeros(1))
    self.assertTensorClose(heading_error, torch.zeros(1))
    command = arc_command_controller(
      route, endpoint.unsqueeze(0), endpoint_h.unsqueeze(0), target_speed=0.5
    )
    self.assertTensorClose(command, torch.zeros((1, 3)))

  def test_arc_controller_has_expected_signed_yaw_at_start(self) -> None:
    for turn_sign, expected_yaw in ((1, 0.25), (-1, -0.25)):
      with self.subTest(turn_sign=turn_sign):
        route = make_arc_route(torch.zeros(2), 0.0, 2.0, turn_sign)
        command = arc_command_controller(
          route, torch.zeros((1, 2)), torch.zeros(1), target_speed=0.5
        )
        self.assertTensorClose(command[0], torch.tensor([0.5, 0.0, expected_yaw]))


class SRouteAcceptanceTest(unittest.TestCase):
  def test_two_sixty_degree_arcs_are_position_and_tangent_continuous(self) -> None:
    route = make_s_route(torch.zeros(2), 0.0, 2.0, 1, angle=math.pi / 3.0)
    junction_xy, junction_h = route.first.pose_at(route.first.length)
    second_start_xy, second_start_h = route.second.pose_at(0.0)
    torch.testing.assert_close(junction_xy, second_start_xy, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(junction_h, second_start_h, rtol=0.0, atol=1e-6)
    final_xy, final_h = route.pose_at(route.length)
    self.assertAlmostEqual(float(final_h), 0.0, places=6)
    self.assertAlmostEqual(float(final_xy[0]), 4.0 * math.sqrt(3.0) / 2.0, places=5)
    self.assertAlmostEqual(float(final_xy[1]), 2.0, places=5)
    self.assertAlmostEqual(route.length, 4.0 * math.pi / 3.0, places=6)

  def test_s_controller_changes_curvature_sign_between_segments(self) -> None:
    route = make_s_route(torch.zeros(2), 0.0, 2.0, 1, angle=math.pi / 3.0)
    first_xy, first_h = route.pose_at(route.first.length * 0.5)
    second_xy, second_h = route.pose_at(route.first.length + route.second.length * 0.5)
    first_cmd = s_command_controller(
      route, first_xy.unsqueeze(0), first_h.unsqueeze(0), target_speed=0.5
    )
    second_cmd = s_command_controller(
      route, second_xy.unsqueeze(0), second_h.unsqueeze(0), target_speed=0.5
    )
    # With zero geometric error, the route tangent must be rotated into the
    # robot frame: forward speed stays target_speed and lateral command is zero
    # even when the world heading is not zero.
    self.assertAlmostEqual(float(first_cmd[0, 0]), 0.5, places=5)
    self.assertAlmostEqual(float(first_cmd[0, 1]), 0.0, places=5)
    self.assertAlmostEqual(float(second_cmd[0, 0]), 0.5, places=5)
    self.assertAlmostEqual(float(second_cmd[0, 1]), 0.0, places=5)
    self.assertGreater(float(first_cmd[0, 2]), 0.0)
    self.assertLess(float(second_cmd[0, 2]), 0.0)


class CurvedRouteBatchAndValidationTest(unittest.TestCase):
  def test_batch_dtype_device_and_world_body_transform(self) -> None:
    starts = torch.zeros((3, 2), dtype=torch.float64)
    headings = torch.tensor([0.0, math.pi / 2.0, -math.pi / 2.0], dtype=torch.float64)
    route = make_arc_route(starts, headings, 2.0, 1)
    progress = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    positions, expected_heading = route.pose_at(progress)
    self.assertEqual(positions.shape, (3, 2))
    self.assertEqual(positions.dtype, torch.float64)
    self.assertEqual(expected_heading.shape, (3,))
    errors = arc_route_errors(route, positions, expected_heading)
    for value in errors:
      self.assertEqual(value.dtype, torch.float64)
      self.assertEqual(value.device, positions.device)
    body = world_to_body_velocity(
      torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float64), headings
    )
    torch.testing.assert_close(
      body,
      torch.tensor([[1.0, 0.0], [1.0, 0.0], [-2.0, 1.0]], dtype=torch.float64),
      rtol=0.0,
      atol=1e-6,
    )

  def test_invalid_radius_turn_angle_speed_and_shapes_raise(self) -> None:
    for kwargs in (
      dict(radius=0.0, turn_sign=1),
      dict(radius=-1.0, turn_sign=1),
      dict(radius=1.0, turn_sign=0),
      dict(radius=1.0, turn_sign=1, angle=0.0),
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        ArcSpec(**kwargs)
    route = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    for speed in (0.0, -0.1, float("nan")):
      with self.subTest(speed=speed), self.assertRaises(ValueError):
        route.command_tape(speed)
    with self.assertRaises(ValueError):
      arc_route_errors(route, torch.zeros((2, 3)), torch.zeros(2))
    with self.assertRaises(ValueError):
      arc_route_errors(route, torch.zeros((2, 2)), torch.zeros(3))


class AttemptLifecycleAcceptanceTest(unittest.TestCase):
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

  def test_completed_and_failed_attempts_freeze_after_reset(self) -> None:
    result = update_attempt_status(
      torch.tensor([True, True, False]),
      torch.tensor([2.0, 1.0, 99.0]),
      torch.tensor([0.0, 0.5, 0.0]),
      torch.tensor([0.0, 0.0, 0.0]),
      torch.tensor([False, True, True]),
      route_length=1.0,
      cross_track_tolerance=0.05,
      heading_tolerance=0.05,
    )
    self.assertTrue(bool(result.completed_now[0]))
    self.assertTrue(bool(result.failed_now[1]))
    self.assertFalse(bool(result.active[0]))
    self.assertFalse(bool(result.active[1]))
    self.assertFalse(bool(result.active[2]))
    self.assertTrue(torch.equal(result.sample_mask, torch.tensor([True, True, False])))


if __name__ == "__main__":
  unittest.main()
