"""CPU-only tests for route-frame math and attempt lifecycle."""

import math
from types import SimpleNamespace
import unittest

import torch

from scripts.evaluate_go2_routes import (
  RouteConfig,
  _column_terrain_names,
  _configure_continuous_sim_capacity,
  _make_continuous_scenarios,
  _make_scenarios,
  _resolve_route_contract,
)
from src.tasks.velocity.evaluation.routes import (
  route_frame_errors,
  route_normal_velocity,
  straight_route_initial_positions,
  straight_line_controller,
  update_attempt_status,
  validate_initial_route_state,
  world_to_body_velocity,
  wrap_to_pi,
)


class RouteGeometryTest(unittest.TestCase):
  def test_wrap_sign_and_boundary(self) -> None:
    values = torch.tensor([-math.pi, math.pi, 3 * math.pi, -3 * math.pi, 0.25])
    wrapped = wrap_to_pi(values)
    self.assertTrue(torch.all(wrapped >= -math.pi))
    self.assertTrue(torch.all(wrapped < math.pi))
    self.assertAlmostEqual(float(wrapped[1]), -math.pi, places=6)
    self.assertAlmostEqual(float(wrapped[-1]), 0.25, places=6)

  def test_route_frame_progress_and_left_cross_track(self) -> None:
    position = torch.tensor([[2.0, 1.0], [0.0, -2.0]])
    start = torch.zeros(2, 2)
    state = route_frame_errors(position, torch.tensor([0.0, math.pi / 2]), start, torch.tensor([0.0, math.pi / 2]))
    torch.testing.assert_close(state.progress, torch.tensor([2.0, -2.0]))
    torch.testing.assert_close(state.cross_track, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(state.heading_error, torch.zeros(2))

  def test_world_to_body_rotation_and_batch_broadcast(self) -> None:
    world = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    body = world_to_body_velocity(world, torch.tensor([0.0, math.pi / 2]))
    torch.testing.assert_close(body, torch.tensor([[1.0, 0.0], [1.0, 0.0]]), atol=1e-6, rtol=1e-6)

  def test_cross_axis_velocity_uses_route_normal_not_body_y(self) -> None:
    velocity_w = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    projected = route_normal_velocity(
      velocity_w, torch.tensor([0.0, math.pi / 2])
    )
    torch.testing.assert_close(projected, torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0)

  def test_controller_corrects_left_error_and_stops_after_length(self) -> None:
    command = straight_line_controller(
      torch.tensor([[0.0, 0.5], [1.1, 0.0]]), torch.zeros(2), torch.zeros(2, 2), 0.0,
      target_speed=0.5, cross_track_gain=1.0, heading_gain=1.0,
      max_lateral_speed=0.3, max_yaw_rate=0.5, route_length=1.0,
    )
    self.assertLess(float(command[0, 1]), 0.0)
    torch.testing.assert_close(command[1], torch.zeros(3))

  def test_initial_cross_offset_preserves_nominal_root_offset(self) -> None:
    nominal = torch.tensor([[11.0, -4.0], [3.0, 8.0]])
    heading = torch.tensor([0.0, math.pi / 2])
    requested_cross = torch.tensor([0.25, -0.4])
    route_start, robot_start = straight_route_initial_positions(
      nominal, heading, 0.7, requested_cross
    )
    state = route_frame_errors(robot_start, heading, route_start, heading)
    torch.testing.assert_close(state.progress, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
      state.cross_track, requested_cross, atol=1e-6, rtol=0
    )
    self.assertFalse(torch.allclose(route_start, torch.zeros_like(route_start)))

  def test_realized_initial_pose_validation_includes_heading(self) -> None:
    nominal = torch.tensor([[11.0, -4.0], [3.0, 8.0]])
    route_heading = torch.tensor([0.2, -1.0])
    requested_cross = torch.tensor([0.25, -0.4])
    requested_yaw = torch.tensor([0.3, -0.2])
    route_start, robot_start = straight_route_initial_positions(
      nominal, route_heading, 0.7, requested_cross
    )
    state = validate_initial_route_state(
      robot_start,
      route_heading + requested_yaw,
      route_start,
      route_heading,
      requested_cross,
      requested_yaw,
    )
    torch.testing.assert_close(state.progress, torch.zeros(2), atol=1e-6, rtol=0)
    with self.assertRaises(RuntimeError):
      validate_initial_route_state(
        robot_start,
        route_heading,
        route_start,
        route_heading,
        requested_cross,
        requested_yaw,
      )

  def test_bad_shapes_are_rejected(self) -> None:
    with self.assertRaises(ValueError):
      route_frame_errors(torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, 2), 0.0)
    with self.assertRaises(ValueError):
      world_to_body_velocity(torch.zeros(2, 2), torch.zeros(3))


class RouteLifecycleTest(unittest.TestCase):
  def test_completion_and_failure_freeze(self) -> None:
    active = torch.tensor([True, True, False])
    result = update_attempt_status(
      active,
      progress=torch.tensor([1.0, 0.9, 2.0]),
      cross_track=torch.tensor([0.01, 0.0, 0.0]),
      heading_error=torch.tensor([0.01, 0.0, 0.0]),
      failure_mask=torch.tensor([False, True, True]),
      route_length=1.0, cross_track_tolerance=0.1, heading_tolerance=0.1,
    )
    torch.testing.assert_close(result.sample_mask, active)
    torch.testing.assert_close(result.completed_now, torch.tensor([True, False, False]))
    torch.testing.assert_close(result.failed_now, torch.tensor([False, True, False]))
    torch.testing.assert_close(result.active, torch.tensor([False, False, False]))

  def test_inactive_env_cannot_reactivate(self) -> None:
    result = update_attempt_status(
      torch.tensor([False]), torch.tensor([5.0]), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
      route_length=1.0, cross_track_tolerance=0.1, heading_tolerance=0.1,
    )
    self.assertFalse(bool(result.completed_now.item()))
    self.assertFalse(bool(result.active.item()))


class ScenarioCoverageTest(unittest.TestCase):
  def test_scenarios_are_independent_patches_not_transition_claims(self) -> None:
    cfg = RouteConfig(
      checkpoints=("unused",),
      terrain_types=("flat", "pyramid_stairs"),
      levels=(3,),
      repeats=1,
    )
    scenarios = _make_scenarios(
      cfg, {"flat": [0], "pyramid_stairs": [1]}
    )
    self.assertEqual(len(scenarios), 2)
    self.assertEqual(scenarios[0]["direction_semantics"], "straight")
    self.assertEqual(scenarios[1]["direction_semantics"], "stairs_down")
    self.assertNotIn("transition", scenarios[0])

  def test_continuous_contract_locks_geometry(self) -> None:
    cfg = RouteConfig(checkpoints=("unused",), terrain_suite="continuous")
    self.assertEqual(_resolve_route_contract(cfg), (6.5, 0.0, 0.0))
    self.assertEqual(
      _resolve_route_contract(RouteConfig(checkpoints=("unused",))),
      (2.0, 0.0, 0.0),
    )
    invalid = (
      RouteConfig(
        checkpoints=("unused",), terrain_suite="continuous", route_length=6.4
      ),
      RouteConfig(
        checkpoints=("unused",), terrain_suite="continuous", route_heading=0.1
      ),
      RouteConfig(
        checkpoints=("unused",),
        terrain_suite="continuous",
        start_forward_offset=0.1,
      ),
    )
    for invalid_cfg in invalid:
      with self.assertRaises(ValueError):
        _resolve_route_contract(invalid_cfg)

  def test_continuous_scenario_matrix_and_metadata(self) -> None:
    from src.tasks.velocity.evaluation.route_terrains import (
      make_continuous_route_terrain_generator,
    )

    cfg = RouteConfig(
      checkpoints=("unused",),
      terrain_suite="continuous",
      levels=(3, 7),
      repeats=2,
      cross_track_offsets=(0.0, 0.2),
      yaw_offsets=(0.0, -0.1),
    )
    generator = make_continuous_route_terrain_generator(seed=cfg.seed)
    column_names = _column_terrain_names(generator)
    terrain_columns = {
      name: [index for index, value in enumerate(column_names) if value == name]
      for name in generator.sub_terrains
    }
    scenarios = _make_continuous_scenarios(cfg, terrain_columns)
    self.assertEqual(len(scenarios), 4 * 2 * 2 * 2 * 2)
    self.assertEqual(
      {scenario["transition_case"] for scenario in scenarios},
      {"stairs_up", "stairs_down", "slope_up", "slope_down"},
    )
    for scenario in scenarios:
      self.assertEqual(scenario["direction"], scenario["transition_case"].split("_")[1])
      self.assertGreaterEqual(scenario["difficulty"], scenario["level"] / 10)
      self.assertLess(scenario["difficulty"], (scenario["level"] + 1) / 10)
      expected_sign = 1 if scenario["direction"] == "up" else -1
      self.assertGreater(expected_sign * scenario["net_height"], 0.0)

  def test_continuous_suite_raises_contact_capacity_only_when_called(self) -> None:
    env_cfg = SimpleNamespace(sim=SimpleNamespace(nconmax=35))
    override = _configure_continuous_sim_capacity(env_cfg)
    self.assertEqual(override, {"original": 35, "effective": 128})
    self.assertEqual(env_cfg.sim.nconmax, 128)

    already_large = SimpleNamespace(sim=SimpleNamespace(nconmax=256))
    override = _configure_continuous_sim_capacity(already_large)
    self.assertEqual(override, {"original": 256, "effective": 256})
    self.assertEqual(already_large.sim.nconmax, 256)


if __name__ == "__main__":
  unittest.main()
