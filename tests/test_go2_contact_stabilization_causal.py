"""CPU contracts for the loaded-stance contact stabilizer."""

import unittest

import torch

from scripts.audit_go2_contact_stabilization_causal import (
  ContactStabilizationConfig, _bounded_tangent_force, _slots, _validate_config,
)


class ContactStabilizationCausalTest(unittest.TestCase):
  def test_locked_config(self) -> None:
    _validate_config(ContactStabilizationConfig())
    _validate_config(ContactStabilizationConfig(
      repeats=1, warmup_steps=0, sample_steps=10, formal=False
    ))
    with self.assertRaises(ValueError):
      _validate_config(ContactStabilizationConfig(damping_n_per_mps=10.0))

  def test_source_sham_probe_slots(self) -> None:
    rows = _slots(ContactStabilizationConfig(), "slope_up_high", "slope_up", 0)
    self.assertEqual([row["arm"] for row in rows[:3]], ["source", "sham", "probe"])
    self.assertEqual(
      [row["damping_n_per_mps"] for row in rows[:3]], [0.0, 0.0, 20.0]
    )
    self.assertEqual(len(rows), 48)

  def test_force_opposes_velocity_and_respects_cap(self) -> None:
    velocity = torch.tensor([[[1.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])
    normal_force = torch.tensor([[50.0, 50.0]])
    force, cap = _bounded_tangent_force(velocity, normal_force)
    self.assertTrue(bool(((force * velocity).sum(dim=-1) <= 0.0).all()))
    self.assertTrue(bool((torch.linalg.vector_norm(force, dim=-1) <= cap + 1.0e-7).all()))
    torch.testing.assert_close(cap, torch.tensor([[6.0, 6.0]]))

  def test_zero_normal_force_means_zero_wrench(self) -> None:
    force, cap = _bounded_tangent_force(
      torch.tensor([[[1.0, 2.0, 3.0]]]), torch.zeros(1, 1)
    )
    torch.testing.assert_close(force, torch.zeros_like(force))
    torch.testing.assert_close(cap, torch.zeros_like(cap))


if __name__ == "__main__":
  unittest.main()
