from __future__ import annotations

from dataclasses import asdict
import math
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.env_cfgs import (
  STANCE_SLIP_REWARD_WEIGHT,
  unitree_go2_rough_v7_env_cfg,
  unitree_go2_rough_v7_stance_slip_env_cfg,
)
from src.tasks.velocity.mdp.rewards import (
  terrain_relative_loaded_stance_slip_cost,
)
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner


TASK_ID = "Unitree-Go2-Rough-V7-StanceSlip"


def _flat_inputs(
  *, num_feet: int = 2,
) -> tuple[torch.Tensor, ...]:
  foot_pos = torch.zeros(1, num_feet, 3)
  foot_pos[0, :, 0] = torch.arange(num_feet, dtype=torch.float32) * 0.2
  foot_vel = torch.zeros_like(foot_pos)
  contact_force = torch.zeros_like(foot_pos)
  contact_force[..., 2] = -30.0
  in_contact = torch.ones(1, num_feet, dtype=torch.bool)
  hit_pos = foot_pos.clone()
  hit_normals = torch.zeros_like(hit_pos)
  hit_normals[..., 2] = 1.0
  hit_distances = torch.ones(1, num_feet)
  return (
    foot_pos,
    foot_vel,
    contact_force,
    in_contact,
    hit_pos,
    hit_normals,
    hit_distances,
  )


def _cost(*values: torch.Tensor, **kwargs: float) -> tuple[torch.Tensor, ...]:
  return terrain_relative_loaded_stance_slip_cost(*values, **kwargs)


def _canonical(value: object) -> object:
  if isinstance(value, dict):
    return {key: _canonical(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_canonical(item) for item in value]
  if callable(value):
    return (
      getattr(value, "__module__", type(value).__module__),
      getattr(value, "__qualname__", type(value).__qualname__),
    )
  return value


class TestTerrainRelativeLoadedStanceSlipMath(unittest.TestCase):
  def test_flat_cost_uses_deadband_and_load_normalized_average(self) -> None:
    values = list(_flat_inputs())
    values[1][0, 0, 0] = 0.13
    values[1][0, 1, 0] = 0.03
    cost, slip, loaded, valid, normal_force = _cost(*values)
    torch.testing.assert_close(cost, torch.tensor([0.5]))
    torch.testing.assert_close(slip, torch.tensor([[0.13, 0.03]]))
    self.assertTrue(bool(loaded.all()))
    self.assertTrue(bool(valid.all()))
    torch.testing.assert_close(normal_force, torch.full((1, 2), 30.0))

  def test_slope_projection_removes_normal_velocity(self) -> None:
    values = list(_flat_inputs(num_feet=1))
    gradient = 0.4
    normal = torch.tensor([-gradient, 0.0, 1.0])
    normal /= torch.linalg.vector_norm(normal)
    tangent = torch.tensor([1.0, 0.0, gradient])
    tangent /= torch.linalg.vector_norm(tangent)
    values[1][0, 0] = 0.5 * normal
    values[2][0, 0] = -30.0 * normal
    values[5][0, 0] = normal
    cost, slip, *_ = _cost(*values)
    torch.testing.assert_close(cost, torch.zeros(1), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(slip, torch.zeros(1, 1), atol=1.0e-6, rtol=0.0)

    values[1][0, 0] = 0.13 * tangent
    cost, slip, *_ = _cost(*values)
    torch.testing.assert_close(cost, torch.ones(1), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(slip, torch.tensor([[0.13]]), atol=1.0e-6, rtol=0.0)

  def test_only_contact_with_valid_ray_and_sufficient_load_is_charged(self) -> None:
    values = list(_flat_inputs(num_feet=4))
    values[1][..., 0] = 0.13
    values[3][0, 1] = False
    values[6][0, 2] = -1.0
    values[2][0, 3, 2] = -14.99
    cost, _, loaded, valid, normal_force = _cost(
      *values, max_horizontal_distance=0.1
    )
    torch.testing.assert_close(cost, torch.ones(1))
    self.assertEqual(loaded.tolist(), [[True, False, False, False]])
    self.assertEqual(valid.tolist(), [[True, True, False, True]])
    self.assertAlmostEqual(float(normal_force[0, 3]), 14.99, places=5)

  def test_no_loaded_feet_is_finite_zero(self) -> None:
    values = list(_flat_inputs())
    values[3].zero_()
    values[6].fill_(-1.0)
    cost, slip, loaded, valid, normal_force = _cost(*values)
    torch.testing.assert_close(cost, torch.zeros(1))
    self.assertTrue(bool(torch.isfinite(cost).all()))
    self.assertTrue(bool(torch.isfinite(slip).all()))
    self.assertFalse(bool(loaded.any()))
    self.assertFalse(bool(valid.any()))
    self.assertTrue(bool(torch.isfinite(normal_force).all()))

  def test_cost_is_yaw_rotation_invariant(self) -> None:
    values = list(_flat_inputs())
    values[1][0, 0] = torch.tensor([0.13, 0.04, 0.0])
    values[1][0, 1] = torch.tensor([-0.06, 0.11, 0.0])
    expected = _cost(*values)[0]
    angle = math.radians(73.0)
    rotation = torch.tensor(
      [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
      ]
    )
    for index in (0, 1, 2, 4, 5):
      values[index] = values[index] @ rotation.T
    actual = _cost(*values)[0]
    torch.testing.assert_close(actual, expected)

  def test_cost_is_clipped_and_parameter_contract_is_strict(self) -> None:
    values = list(_flat_inputs(num_feet=1))
    values[1][0, 0, 0] = 10.0
    cost, *_ = _cost(*values, max_cost_per_foot=4.0)
    torch.testing.assert_close(cost, torch.tensor([4.0]))
    invalid = {
      "normal_force_threshold": -1.0,
      "max_horizontal_distance": 0.0,
      "slip_deadband": -0.01,
      "slip_scale": 0.0,
      "max_cost_per_foot": 0.0,
    }
    for name, value in invalid.items():
      with self.subTest(name=name), self.assertRaises(ValueError):
        _cost(*values, **{name: value})


class TestStanceSlipTrainingContract(unittest.TestCase):
  def test_probe_differs_from_v7_only_by_one_reward_term(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    probe = unitree_go2_rough_v7_stance_slip_env_cfg()
    reward = probe.rewards.pop("terrain_tangent_stance_slip")
    self.assertEqual(_canonical(asdict(probe)), _canonical(asdict(baseline)))
    self.assertEqual(reward.weight, STANCE_SLIP_REWARD_WEIGHT)
    self.assertEqual(reward.func.__name__, "terrain_relative_loaded_stance_slip")
    self.assertEqual(reward.params["normal_force_threshold"], 15.0)
    self.assertEqual(reward.params["slip_deadband"], 0.03)
    self.assertEqual(reward.params["slip_scale"], 0.10)
    self.assertTrue(reward.params["asset_cfg"].preserve_order)
    self.assertNotIn("high_slope_sampling", probe.events)

  def test_registered_task_uses_unchanged_runner(self) -> None:
    cfg = load_env_cfg(TASK_ID)
    self.assertIn("terrain_tangent_stance_slip", cfg.rewards)
    self.assertIs(load_runner_cls(TASK_ID), VelocityOnPolicyRunner)

  def test_zero_weight_control_has_exact_v7_configuration_after_term_removal(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    control = unitree_go2_rough_v7_stance_slip_env_cfg()
    control.rewards["terrain_tangent_stance_slip"].weight = 0.0
    control.rewards.pop("terrain_tangent_stance_slip")
    self.assertEqual(_canonical(asdict(control)), _canonical(asdict(baseline)))


if __name__ == "__main__":
  unittest.main()
