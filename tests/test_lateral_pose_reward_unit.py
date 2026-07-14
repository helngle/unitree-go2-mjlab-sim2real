import unittest

import torch

from src.tasks.velocity.mdp.rewards import lateral_conditioned_joint_std


class LateralConditionedJointStdTest(unittest.TestCase):
  def setUp(self) -> None:
    self.base_std = torch.tensor([0.15, 0.35, 0.50])
    self.hip_mask = torch.tensor([True, False, False])

  def effective_std(self, commands: list[list[float]]) -> torch.Tensor:
    return lateral_conditioned_joint_std(
      torch.tensor(commands), self.base_std, self.hip_mask
    )

  def test_forward_yaw_and_standing_leave_tolerances_unchanged(self) -> None:
    actual = self.effective_std(
      [[0.6, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(actual, self.base_std.expand(3, -1))

  def test_full_lateral_command_relaxes_only_hips_for_both_directions(self) -> None:
    actual = self.effective_std([[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]])
    expected = torch.tensor([[0.30, 0.35, 0.50], [0.30, 0.35, 0.50]])
    torch.testing.assert_close(actual, expected)

  def test_interpolation_is_continuous_and_respects_lateral_dominance(self) -> None:
    actual = self.effective_std(
      [[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.1, 0.2, 0.0]]
    )
    expected_hip_std = torch.tensor([0.20, 0.25, 0.20])
    torch.testing.assert_close(actual[:, 0], expected_hip_std)
    torch.testing.assert_close(actual[:, 1:], self.base_std[1:].expand(3, -1))

  def test_batched_joint_std_broadcasts_without_changing_shape(self) -> None:
    base_std = self.base_std.expand(2, -1).clone()
    actual = lateral_conditioned_joint_std(
      torch.tensor([[0.0, 0.3, 0.0], [0.6, 0.0, 0.0]]),
      base_std,
      self.hip_mask,
    )
    self.assertEqual(actual.shape, (2, 3))
    torch.testing.assert_close(actual[0], torch.tensor([0.30, 0.35, 0.50]))
    torch.testing.assert_close(actual[1], self.base_std)

  def test_non_positive_full_lateral_command_is_rejected(self) -> None:
    command = torch.tensor([[0.0, 0.3, 0.0]])
    for value in (0.0, -0.3):
      with self.subTest(value=value), self.assertRaises(ValueError):
        lateral_conditioned_joint_std(
          command,
          self.base_std,
          self.hip_mask,
          full_lateral_command=value,
        )


if __name__ == "__main__":
  unittest.main()
