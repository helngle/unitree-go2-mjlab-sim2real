"""CPU contracts for the action-recovery counterfactual."""

import unittest

import torch

from scripts.audit_go2_action_recovery_counterfactual import (
  RecoveryConfig, _blend_actions, _flat_sentinel_gate, _slots, _validate_config,
)


class ActionRecoveryCounterfactualTest(unittest.TestCase):
  def test_locked_config(self) -> None:
    _validate_config(RecoveryConfig())
    _validate_config(RecoveryConfig(repeats=1, warmup_steps=0, sample_steps=10, formal=False))
    with self.assertRaises(ValueError):
      _validate_config(RecoveryConfig(blend=0.25))

  def test_source_sham_probe_identity(self) -> None:
    rows = _slots(RecoveryConfig(), "slope_up_high", "slope_up", 0)
    self.assertEqual([row["arm"] for row in rows[:3]], ["source", "sham", "probe"])
    self.assertEqual([row["blend"] for row in rows[:3]], [0.0, 0.0, 0.5])
    self.assertEqual(len(rows), 48)

  def test_two_tap_fir_and_noop(self) -> None:
    current = torch.tensor([[2.0, -2.0], [4.0, 8.0], [6.0, 10.0]])
    previous = torch.tensor([[0.0, 0.0], [2.0, 4.0], [3.0, 5.0]])
    probe = torch.tensor([False, False, True])
    actual = _blend_actions(current, previous, probe, 0.5)
    torch.testing.assert_close(actual[:2], current[:2])
    torch.testing.assert_close(actual[2], torch.tensor([4.5, 7.5]))

  def test_flat_sentinel_vetoes_new_failure(self) -> None:
    cell = {
      "coverage_pass": True,
      "pairs": [{
        "completion": {"sham": True, "probe": False},
        "metrics": {"gain": {"sham": 1.0, "probe": 1.1}},
      }],
    }
    gate = _flat_sentinel_gate(cell)
    self.assertFalse(gate["pass"])
    self.assertEqual(gate["sham_complete_to_probe_fail_count"], 1)


if __name__ == "__main__":
  unittest.main()
