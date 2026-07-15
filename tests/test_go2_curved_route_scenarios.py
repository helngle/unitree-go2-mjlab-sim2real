import math
import unittest

import torch

from scripts.evaluate_go2_curved_routes import (
  PATCH_SIZE,
  ROUTE_START_LOCAL,
  CurvedRouteConfig,
  _scenarios,
  _validate_config,
)
from src.tasks.velocity.evaluation.curved_routes import (
  ArcSpec,
  arc_command_controller,
  arc_route_errors,
  make_arc_route,
  make_s_route,
  s_command_controller,
)


class ArcGeometryTest(unittest.TestCase):
  def test_left_quarter_arc_endpoint_and_heading(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    xy, heading = route.pose_at(route.length)
    torch.testing.assert_close(xy, torch.tensor([2.0, 2.0]), atol=1e-6, rtol=0)
    self.assertAlmostEqual(float(heading), math.pi / 2.0, places=6)
    self.assertAlmostEqual(route.length, math.pi, places=6)

  def test_right_quarter_arc_mirrors_left(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 2.0, -1)
    xy, heading = route.pose_at(route.length)
    torch.testing.assert_close(xy, torch.tensor([2.0, -2.0]), atol=1e-6, rtol=0)
    self.assertAlmostEqual(float(heading), -math.pi / 2.0, places=6)

  def test_command_tape_curvature_sign_and_batch(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 2.5, -1)
    command = route.command_tape(torch.tensor([0.3, 0.6]))
    self.assertEqual(tuple(command.shape), (2, 3))
    torch.testing.assert_close(command[:, 0], torch.tensor([0.3, 0.6]))
    torch.testing.assert_close(command[:, 1], torch.zeros(2))
    torch.testing.assert_close(command[:, 2], -torch.tensor([0.3, 0.6]) / 2.5)

  def test_arc_errors_and_controller_zero_error(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 2.0, 1)
    xy, heading = route.pose_at(torch.tensor([0.0, 0.4]))
    progress, cross, heading_error = arc_route_errors(route, xy, heading)
    torch.testing.assert_close(progress, torch.tensor([0.0, 0.4]), atol=1e-5, rtol=0)
    torch.testing.assert_close(cross, torch.zeros(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(heading_error, torch.zeros(2), atol=1e-5, rtol=0)
    command = arc_command_controller(route, xy, heading, target_speed=0.5)
    torch.testing.assert_close(command[:, 0], torch.full((2,), 0.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(command[:, 1], torch.zeros(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(command[:, 2], torch.full((2,), 0.25), atol=1e-5, rtol=0)

  def test_controller_corrects_nonzero_cross_and_heading(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.4, 2.5, 1)
    xy, heading = route.pose_at(0.7)
    tangent = torch.stack((torch.cos(heading), torch.sin(heading)))
    normal = torch.stack((-torch.sin(heading), torch.cos(heading)))
    position = (xy + 0.15 * normal).unsqueeze(0)
    actual_heading = (heading + 0.1).unsqueeze(0)
    command = arc_command_controller(route, position, actual_heading, target_speed=0.4)
    self.assertLess(float(command[0, 1]), 0.0)
    self.assertLess(float(command[0, 2]), 0.4 / 2.5)

  def test_controller_stops_at_float_endpoint(self) -> None:
    route = make_arc_route(torch.zeros(2), 0.0, 1.5, 1)
    xy, heading = route.pose_at(route.length)
    command = arc_command_controller(route, xy.unsqueeze(0), heading.unsqueeze(0), target_speed=0.4)
    torch.testing.assert_close(command, torch.zeros_like(command))


class SRouteGeometryTest(unittest.TestCase):
  def test_s_route_is_continuous_and_restores_heading(self) -> None:
    route = make_s_route(torch.zeros(2), 0.0, 2.0, 1)
    first_xy, first_heading = route.first.pose_at(route.first.length)
    second_xy, second_heading = route.second.pose_at(0.0)
    torch.testing.assert_close(first_xy, second_xy, atol=1e-6, rtol=0)
    torch.testing.assert_close(first_heading, second_heading, atol=1e-6, rtol=0)
    end_xy, end_heading = route.pose_at(route.length)
    self.assertAlmostEqual(float(end_heading), 0.0, places=5)
    self.assertGreater(float(end_xy[0]), 0.0)

  def test_s_controller_has_opposite_yaw_signs(self) -> None:
    route = make_s_route(torch.zeros(2), 0.0, 2.0, 1)
    first_xy, first_h = route.first.pose_at(0.2)
    second_xy, second_h = route.second.pose_at(0.2)
    first_command = s_command_controller(route, first_xy.unsqueeze(0), first_h.unsqueeze(0), target_speed=0.4)
    second_command = s_command_controller(route, second_xy.unsqueeze(0), second_h.unsqueeze(0), target_speed=0.4)
    self.assertGreater(float(first_command[0, 2]), 0.0)
    self.assertLess(float(second_command[0, 2]), 0.0)

  def test_s_controller_uses_world_tangent_at_nonzero_heading(self) -> None:
    route = make_s_route(torch.tensor([1.0, -2.0]), 0.6, 2.0, 1)
    xy, heading = route.second.pose_at(0.3)
    command = s_command_controller(
      route, xy.unsqueeze(0), heading.unsqueeze(0), target_speed=0.4
    )
    torch.testing.assert_close(command[0, :2], torch.tensor([0.4, 0.0]), atol=1e-5, rtol=0)
    self.assertLess(float(command[0, 2]), 0.0)

  def test_invalid_arc_parameters(self) -> None:
    with self.assertRaises(ValueError):
      ArcSpec(0.0, 1)
    with self.assertRaises(ValueError):
      ArcSpec(1.0, 0)


class CurvedScenarioTest(unittest.TestCase):
  def test_parameter_matrix_and_flat_patch_margin(self) -> None:
    cfg = CurvedRouteConfig(
      checkpoint="unused",
      radii=(1.5, 4.0),
      speeds=(0.3, 0.6),
      turn_signs=(1, -1),
      cross_track_offsets=(-0.2, 0.2),
      yaw_offsets=(-0.2, 0.2),
      repeats=2,
    )
    _validate_config(cfg)
    self.assertEqual(len(_scenarios(cfg)), 2 * 2 * 2 * 2 * 2 * 2)
    self.assertGreaterEqual(ROUTE_START_LOCAL[0], 0.8)
    self.assertGreaterEqual(ROUTE_START_LOCAL[1] - 4.0, 0.8)
    self.assertGreaterEqual(PATCH_SIZE[1] - (ROUTE_START_LOCAL[1] + 4.0), 0.8)

  def test_command_tape_rejects_initial_offsets(self) -> None:
    with self.assertRaises(ValueError):
      _validate_config(
        CurvedRouteConfig(
          checkpoint="unused", mode="command_tape", yaw_offsets=(0.2,)
        )
      )


if __name__ == "__main__":
  unittest.main()
