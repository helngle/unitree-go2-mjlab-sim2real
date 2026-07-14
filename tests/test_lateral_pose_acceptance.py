"""Acceptance tests for lateral-conditioned pose tolerance tensor math."""

from __future__ import annotations

import unittest

import torch

from src.tasks.velocity.mdp.rewards import lateral_conditioned_joint_std


class LateralConditionedJointStdTest(unittest.TestCase):
  def setUp(self) -> None:
    self.base_std = torch.tensor([0.15, 0.35, 0.50, 0.15, 0.35, 0.50])
    self.hip_mask = torch.tensor([True, False, False, True, False, False])

  def assertTensorClose(
    self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-7
  ) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

  def adjusted(self, command: torch.Tensor, **kwargs) -> torch.Tensor:
    return lateral_conditioned_joint_std(
      command,
      self.base_std.to(device=command.device, dtype=command.dtype),
      self.hip_mask.to(device=command.device),
      **kwargs,
    )

  def test_forward_standing_and_yaw_leave_tolerance_unchanged(self) -> None:
    commands = torch.tensor(
      [
        [0.60, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.70],
        [-0.60, 0.00, 0.00],
      ]
    )

    actual = self.adjusted(commands)
    expected = self.base_std.expand(commands.shape[0], -1)

    self.assertTensorClose(actual, expected)
    self.assertTensorClose(actual[:, self.hip_mask], torch.full((4, 2), 0.15))

  def test_full_lateral_command_relaxes_hips_for_both_directions(self) -> None:
    commands = torch.tensor([[0.0, 0.30, 0.0], [0.0, -0.30, 0.0]])

    actual = self.adjusted(commands)

    self.assertTensorClose(actual[:, self.hip_mask], torch.full((2, 2), 0.30))
    self.assertTensorClose(
      actual[:, ~self.hip_mask],
      self.base_std[~self.hip_mask].expand(commands.shape[0], -1),
    )

  def test_pure_lateral_interpolation_is_continuous_and_linear(self) -> None:
    lateral_speed = torch.tensor([0.0, 0.10, 0.20, 0.30])
    commands = torch.zeros((lateral_speed.numel(), 3))
    commands[:, 1] = lateral_speed

    actual = self.adjusted(commands)
    expected_hip_std = torch.tensor([0.15, 0.20, 0.25, 0.30])

    self.assertTensorClose(actual[:, 0], expected_hip_std)
    self.assertTensorClose(actual[:, 3], expected_hip_std)
    increments = actual[1:, 0] - actual[:-1, 0]
    self.assertTensorClose(increments, torch.full((3,), 0.05))

  def test_dominance_uses_lateral_margin_over_forward_and_yaw(self) -> None:
    commands = torch.tensor(
      [
        [0.05, 0.20, 0.00],  # margin 0.15, alpha 0.5
        [0.20, 0.20, 0.00],  # tied with forward: no relaxation
        [0.00, 0.20, 0.10],  # margin 0.10, alpha 1/3
        [0.00, 0.10, 0.20],  # yaw dominant: no relaxation
        [0.00, -0.20, -0.10],  # signs do not affect the margin
      ]
    )

    actual = self.adjusted(commands)
    expected_hip_std = torch.tensor([0.225, 0.15, 0.20, 0.15, 0.20])

    self.assertTensorClose(actual[:, 0], expected_hip_std)
    self.assertTensorClose(actual[:, 3], expected_hip_std)
    self.assertTensorClose(
      actual[:, ~self.hip_mask],
      self.base_std[~self.hip_mask].expand(commands.shape[0], -1),
    )

  def test_multidimensional_batches_preserve_shape_dtype_and_device(self) -> None:
    commands = torch.zeros((2, 3, 3), dtype=torch.float64)
    commands[..., 1] = torch.tensor(
      [[0.0, 0.1, 0.3], [-0.3, -0.2, 0.0]], dtype=commands.dtype
    )

    actual = self.adjusted(commands)

    self.assertEqual(actual.shape, (2, 3, self.base_std.numel()))
    self.assertEqual(actual.dtype, commands.dtype)
    self.assertEqual(actual.device, commands.device)
    expected_first_hip = torch.tensor(
      [[0.15, 0.20, 0.30], [0.30, 0.25, 0.15]], dtype=commands.dtype
    )
    self.assertTensorClose(actual[..., 0], expected_first_hip)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
  def test_cuda_inputs_remain_on_device(self) -> None:
    commands = torch.tensor([[0.0, 0.3, 0.0]], device="cuda")

    actual = self.adjusted(commands)

    self.assertEqual(actual.device, commands.device)
    self.assertEqual(actual.dtype, commands.dtype)
    self.assertTensorClose(
      actual[:, self.hip_mask], torch.full((1, 2), 0.30, device="cuda")
    )

  def test_nonpositive_full_lateral_command_raises(self) -> None:
    command = torch.tensor([[0.0, 0.3, 0.0]])

    for value in (0.0, -0.3):
      with self.subTest(full_lateral_command=value):
        with self.assertRaises(ValueError):
          self.adjusted(command, full_lateral_command=value)

  def test_malformed_command_last_dimension_raises(self) -> None:
    for shape in ((2, 2), (2, 4), (2, 3, 2), (2, 3, 4)):
      with self.subTest(shape=shape):
        with self.assertRaises(ValueError):
          self.adjusted(torch.zeros(shape))

  def test_negative_yaw_scale_raises(self) -> None:
    command = torch.tensor([[0.0, 0.3, 0.0]])

    with self.assertRaises(ValueError):
      self.adjusted(command, yaw_scale=-1.0)


if __name__ == "__main__":
  unittest.main()
