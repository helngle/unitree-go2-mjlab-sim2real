from __future__ import annotations

import copy
import unittest

from scripts.select_go2_privileged_teacher_checkpoints import (
  _aggregate_route,
  _exceeds_safety_guardrail,
)


def _scenario(*, steps: int = 10, joint_faults: int = 2) -> dict:
  return {
    "steps_sampled": steps,
    "speed": 1.0,
    "completed": True,
    "progress_ratio": 1.0,
    "response_gain": {"vx": 0.9},
    "terrain_tangent_stance_slip_mean": 0.1,
    "terrain_tangent_loaded_stance": {"loaded_stance_foot_samples": 4},
    "action_acceleration_mean": 0.2,
    "base_pitch_absolute_mean": 0.3,
    "base_contact_count": 0,
    "upper_leg_contact_count": 0,
    "calf_contact_count": 0,
    "catastrophic_termination": False,
    "failed": False,
    "terrain_rollout_metrics": {
      "active_control_step_samples": steps,
      "action_safety": {
        "available": True,
        "fault_control_step_count": 0,
        "joint_target_available": True,
        "joint_target_fault_control_step_count": joint_faults,
      }
    },
  }


def _payload(row: dict) -> dict:
  return {
    "profiles": {
      "clean": {
        "route_results": {
          "straight": {"scenarios": [copy.deepcopy(row) for _ in range(16)]}
        }
      }
    }
  }


class PrivilegedTeacherSelectionTest(unittest.TestCase):
  def test_joint_target_fault_rate_uses_active_control_steps(self) -> None:
    result = _aggregate_route(_payload(_scenario()), "clean", "straight")
    self.assertEqual(result["joint_target_fault_rate"], 0.2)

  def test_missing_joint_target_telemetry_fails_closed(self) -> None:
    row = _scenario()
    row["terrain_rollout_metrics"]["action_safety"][
      "joint_target_available"
    ] = False
    with self.assertRaisesRegex(ValueError, "action safety is unavailable"):
      _aggregate_route(_payload(row), "clean", "straight")

  def test_active_step_denominator_mismatch_fails_closed(self) -> None:
    row = _scenario()
    row["terrain_rollout_metrics"]["active_control_step_samples"] = 9
    with self.assertRaisesRegex(ValueError, "active-step denominator differs"):
      _aggregate_route(_payload(row), "clean", "straight")

  def test_failure_risk_includes_noncatastrophic_failures(self) -> None:
    row = _scenario()
    row["failed"] = True
    result = _aggregate_route(_payload(row), "clean", "straight")
    self.assertEqual(result["failure_risk"], 1.0)

  def test_guardrail_has_exact_zero_and_exact_boundary_semantics(self) -> None:
    self.assertFalse(_exceeds_safety_guardrail(0.0, 0.0))
    self.assertTrue(_exceeds_safety_guardrail(1e-15, 0.0))
    self.assertFalse(_exceeds_safety_guardrail(1.2, 1.0))
    self.assertTrue(_exceeds_safety_guardrail(1.2000001, 1.0))

  def test_guardrail_rejects_nonfinite_or_negative_metrics(self) -> None:
    for value, reference in ((float("nan"), 1.0), (1.0, float("inf")), (-1.0, 1.0)):
      with self.subTest(value=value, reference=reference):
        with self.assertRaises(ValueError):
          _exceeds_safety_guardrail(value, reference)


if __name__ == "__main__":
  unittest.main()
