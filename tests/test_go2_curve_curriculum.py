"""Contracts exposing how the velocity terrain curriculum treats routes.

The production curriculum intentionally remains unchanged in this audit.  The
tests call it through a minimal environment double so the observed move-up and
move-down decisions cannot drift away from the implementation under review.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from src.tasks.velocity.evaluation.curved_routes import make_s_route
from src.tasks.velocity.mdp.curriculums import terrain_levels_vel


class _Scene(dict):
  def __init__(self, robot: object, terrain: object, origins: torch.Tensor):
    super().__init__(robot=robot)
    self.terrain = terrain
    self.env_origins = origins


class _Terrain:
  def __init__(self, num_envs: int, patch_length: float):
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(patch_length, patch_length))
    )
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.move_up: torch.Tensor | None = None
    self.move_down: torch.Tensor | None = None

  def update_env_origins(
    self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
  ) -> None:
    del env_ids
    self.move_up = move_up.clone()
    self.move_down = move_down.clone()


class _CommandManager:
  def __init__(self, command: torch.Tensor):
    self.command = command

  def get_command(self, command_name: str) -> torch.Tensor:
    if command_name != "twist":
      raise KeyError(command_name)
    return self.command


def _curriculum_decisions(
  final_xy: torch.Tensor,
  command: torch.Tensor,
  *,
  episode_s: float,
  patch_length: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  final_xy = torch.as_tensor(final_xy, dtype=torch.float32)
  command = torch.as_tensor(command, dtype=torch.float32)
  if final_xy.ndim == 1:
    final_xy = final_xy.unsqueeze(0)
  if command.ndim == 1:
    command = command.unsqueeze(0)
  if final_xy.shape[0] != command.shape[0]:
    raise ValueError("final_xy and command batch sizes must match")

  num_envs = final_xy.shape[0]
  origins = torch.zeros((num_envs, 3), dtype=torch.float32)
  root_positions = torch.zeros((num_envs, 3), dtype=torch.float32)
  root_positions[:, :2] = final_xy
  robot = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=root_positions)
  )
  terrain = _Terrain(num_envs, patch_length)
  env = SimpleNamespace(
    scene=_Scene(robot, terrain, origins),
    command_manager=_CommandManager(command),
    max_episode_length_s=episode_s,
  )

  terrain_levels_vel(env, torch.arange(num_envs), "twist")
  assert terrain.move_up is not None
  assert terrain.move_down is not None
  return terrain.move_up, terrain.move_down


class CurveTerrainCurriculumContractTest(unittest.TestCase):
  def test_straight_and_lateral_use_only_net_displacement(self) -> None:
    move_up, move_down = _curriculum_decisions(
      torch.tensor([[10.0, 0.0], [0.0, -10.0], [4.9, 0.0], [0.0, 4.9]]),
      torch.tensor(
        [
          [1.0, 0.0, 0.0],
          [0.0, -1.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
        ]
      ),
      episode_s=10.0,
    )
    self.assertTrue(torch.equal(move_up, torch.zeros(4, dtype=torch.bool)))
    self.assertTrue(
      torch.equal(move_down, torch.tensor([False, False, True, True]))
    )

  def test_pure_yaw_failure_cannot_move_down_and_drift_can_move_up(self) -> None:
    move_up, move_down = _curriculum_decisions(
      torch.tensor([[0.0, 0.0], [4.1, 0.0]]),
      torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7]]),
      episode_s=10.0,
      patch_length=8.0,
    )
    self.assertTrue(torch.equal(move_up, torch.tensor([False, True])))
    self.assertTrue(torch.equal(move_down, torch.tensor([False, False])))

  def test_quarter_half_and_full_circle_depend_on_endpoint_chord(self) -> None:
    radius = 2.0
    speed = 0.5
    angles = torch.tensor([math.pi / 2.0, math.pi, 2.0 * math.pi])
    chord = 2.0 * radius * torch.sin(angles / 2.0)
    final_xy = torch.stack((chord, torch.zeros_like(chord)), dim=1)
    commands = torch.tensor([[speed, 0.0, speed / radius]]).repeat(3, 1)

    # Use each route's ideal duration.  The down threshold is half its arc
    # length, while the measured "distance" is only the endpoint chord.
    decisions = []
    for index, angle in enumerate(angles):
      _, move_down = _curriculum_decisions(
        final_xy[index],
        commands[index],
        episode_s=float(radius * angle / speed),
      )
      decisions.append(bool(move_down.item()))
    self.assertEqual(decisions, [False, False, True])

  def test_s_curve_is_judged_by_endpoint_not_traversed_arc_length(self) -> None:
    radius = 2.0
    speed = 0.5
    route = make_s_route(torch.zeros(2), 0.0, radius, 1, angle=math.pi / 3.0)
    endpoint, _ = route.pose_at(route.length)
    _, move_down = _curriculum_decisions(
      endpoint,
      torch.tensor([speed, 0.0, -speed / radius]),
      episode_s=route.length / speed,
    )
    endpoint_ratio = float(torch.linalg.vector_norm(endpoint) / route.length)
    self.assertAlmostEqual(endpoint_ratio, 3.0 / math.pi, places=6)
    self.assertFalse(bool(move_down.item()))

  def test_equal_traversed_length_can_produce_opposite_level_decisions(self) -> None:
    # Both ideal trajectories cover 5 m.  A straight endpoint crosses the
    # patch half-length and moves up; a closed trajectory returns to origin
    # and moves down.
    move_up, move_down = _curriculum_decisions(
      torch.tensor([[5.0, 0.0], [0.0, 0.0]]),
      torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 2.0 * math.pi / 5.0]]),
      episode_s=5.0,
      patch_length=8.0,
    )
    self.assertTrue(torch.equal(move_up, torch.tensor([True, False])))
    self.assertTrue(torch.equal(move_down, torch.tensor([False, True])))

  def test_command_cancellation_and_zero_settle_are_inconsistent(self) -> None:
    # Identical zero-net-displacement attempts receive different decisions
    # solely because the command tensor at termination differs.
    move_up, move_down = _curriculum_decisions(
      torch.zeros((2, 2)),
      torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
      episode_s=10.0,
    )
    self.assertTrue(torch.equal(move_up, torch.tensor([False, False])))
    self.assertTrue(torch.equal(move_down, torch.tensor([True, False])))

  def test_v7_mode_archetypes_treat_yaw_differently(self) -> None:
    commands = torch.tensor(
      [
        [0.5, 0.0, 0.2],  # general
        [0.0, -0.3, 0.0],  # lateral
        [0.0, 0.0, 0.5],  # yaw
        [0.9, 0.0, 0.1],  # high speed
      ]
    )
    _, move_down = _curriculum_decisions(
      torch.zeros((4, 2)), commands, episode_s=10.0
    )
    self.assertTrue(torch.equal(move_down, torch.tensor([True, True, False, True])))

  def test_down_threshold_uses_only_last_command_not_command_history(self) -> None:
    move_up, move_down = _curriculum_decisions(
      torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
      torch.tensor([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]),
      episode_s=10.0,
    )
    self.assertTrue(torch.equal(move_up, torch.tensor([False, False])))
    self.assertTrue(torch.equal(move_down, torch.tensor([False, True])))


if __name__ == "__main__":
  unittest.main()
