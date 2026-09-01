import unittest

from scripts.audit_go2_high_slope_actuators import (
  AuditConfig,
  _gate,
  _longest_true_run,
  _validate_config,
  _window_indices,
)


class ActuatorAuditConfigTest(unittest.TestCase):
  def test_v7_checkpoint_and_windows_are_locked(self) -> None:
    _validate_config(AuditConfig())

  def test_forbidden_checkpoint_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      _validate_config(AuditConfig(checkpoint="model_13900.pt"))

  def test_windows_distinguish_failure_and_stable_attempts(self) -> None:
    self.assertEqual(_window_indices(120, True, 50, 300), (True, 70, 120, None))
    self.assertEqual(_window_indices(49, True, 50, 300)[3], "insufficient_failure_window")
    self.assertEqual(_window_indices(300, False, 50, 300), (True, 0, 300, None))
    self.assertEqual(_window_indices(299, False, 50, 300)[3], "no_full_stable_attempt")

  def test_longest_run(self) -> None:
    import torch
    self.assertEqual(_longest_true_run(torch.tensor([False, True, True, False, True])), 2)

  def test_gate_requires_stable_controls(self) -> None:
    rows = []
    for condition, speed in (
      ("flat", 0.3), ("flat", 0.5), ("slope_up_high", 0.3)
    ):
      for matched_slot in range(4):
        rows.append({
          "terrain_condition": condition,
          "speed": speed,
          "matched_slot": matched_slot,
          "sample_count": 1200,
          "failed": False,
          "windows": {"stable_rollout": {"eligible": True, "per_joint": {}}},
        })
    self.assertEqual(_gate(rows)["verdict"], "SATURATION_NOT_CONFIRMED")
    rows.pop()
    self.assertEqual(_gate(rows)["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
