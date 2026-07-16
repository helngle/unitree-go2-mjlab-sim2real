"""Offline contracts for curved-command response diagnosis.

No curve sampler exists in this phase. These tests protect the evidence used
to decide whether implementing one is justified.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
  Path(__file__).resolve().parents[1] / "scripts" / "diagnose_go2_command_response.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_go2_command_response", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def _arc_scenario(*, required_yaw_rate: float, completed: bool = False) -> dict:
  in_distribution = abs(required_yaw_rate) <= 0.3
  return {
    "radius": 2.5,
    "speed": 0.5,
    "turn_sign": 1,
    "cross_track_offset": 0.0,
    "yaw_offset": 0.0,
    "repeat": 0,
    "required_yaw_rate": required_yaw_rate,
    "general_yaw_in_distribution": in_distribution,
    "completed": completed,
    "failed": not completed,
    "finished": True,
    "motion_steps": 100,
    "settle_steps": 10,
    "arc_length_progress_ratio": 0.85,
    "actual_velocity_xy_mean": [0.4, 0.01],
    "commanded_velocity_xy_mean": [0.5, 0.0],
    "actual_yaw_rate_mean": required_yaw_rate * 0.95,
    "commanded_yaw_rate_mean": required_yaw_rate,
    "cross_axis_velocity_mean": 0.02,
    "slip_velocity_mean": 0.03,
    "action_acceleration_mean": 0.07,
    "reset_count": 0,
    "termination_counts": {"fell_over": 0.0, "illegal_calf_contact": 0.0},
  }


class ScheduledTapeDiagnosisTest(unittest.TestCase):
  def test_id_ood_split_and_response_gains(self) -> None:
    data = {
      "config": {"route_kind": "arc", "mode": "command_tape", "settle_steps": 10},
      "scenarios": [
        _arc_scenario(required_yaw_rate=0.3),
        _arc_scenario(required_yaw_rate=0.4),
      ],
    }
    report = diagnostics.summarize_arc_tape(data)
    self.assertEqual(report["in_distribution"]["num_scenarios"], 1)
    self.assertEqual(report["out_of_distribution"]["num_scenarios"], 1)
    self.assertAlmostEqual(
      report["in_distribution"]["forward_response_gain_mean"], 0.8
    )
    self.assertAlmostEqual(
      report["in_distribution"]["yaw_response_gain_mean"], 0.95
    )

  def test_obsolete_pose_extended_tape_schema_is_rejected(self) -> None:
    obsolete = {
      "config": {"route_kind": "arc", "mode": "command_tape"},
      "scenarios": [_arc_scenario(required_yaw_rate=0.2)],
    }
    with self.assertRaisesRegex(ValueError, "fixed-time scheduled tape"):
      diagnostics.validate_scheduled_arc_tape(obsolete)

  def test_inconsistent_id_label_is_rejected(self) -> None:
    scenario = _arc_scenario(required_yaw_rate=0.4)
    scenario["general_yaw_in_distribution"] = True
    data = {
      "config": {"route_kind": "arc", "mode": "command_tape", "settle_steps": 10},
      "scenarios": [scenario],
    }
    with self.assertRaisesRegex(ValueError, "inconsistent"):
      diagnostics.validate_scheduled_arc_tape(data)


class ReferenceExtractionTest(unittest.TestCase):
  def test_missing_actual_yaw_is_reported_as_unavailable(self) -> None:
    data = {
      "results": [
        {
          "checkpoint": "/tmp/model_13600.pt",
          "profile": "clean",
          "by_command": {
            "yaw_left": {
              "command": [0.0, 0.0, 0.5],
              "num_envs": 16,
              "yaw_velocity_error_mean": 0.08,
              "slip_velocity_mean": 0.02,
              "action_acceleration_mean": 0.05,
              "terminations_per_env": {},
            }
          },
        }
      ]
    }
    record = diagnostics.extract_reference_commands(data, "model_13600.pt")[0]
    self.assertIsNone(record["yaw_response_gain"])
    self.assertEqual(record["yaw_absolute_error_mean"], 0.08)

  def test_closed_loop_retry_replaces_original_attempt(self) -> None:
    failed = _arc_scenario(required_yaw_rate=0.125, completed=False)
    passed = _arc_scenario(required_yaw_rate=0.125, completed=True)
    base = {"config": {"route_kind": "arc", "mode": "closed_loop"}, "scenarios": [failed]}
    retry = {"config": {"route_kind": "arc", "mode": "closed_loop"}, "scenarios": [passed]}
    result = diagnostics.merge_closed_loop_attempts([base, retry])
    self.assertEqual(result["num_unique_attempts"], 1)
    self.assertEqual(result["completion_rate"], 1.0)


if __name__ == "__main__":
  unittest.main()
