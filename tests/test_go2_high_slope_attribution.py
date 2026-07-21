"""CPU-only tests for strict high-slope and stairs attribution."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from scripts.diagnose_go2_high_slope_attribution import (
  AttributionThresholds,
  analyze_matched,
  analyze_stairs,
  build_report,
)


def _scenario(
  kind: str,
  slot: int,
  *,
  completed: bool,
  vx_gain: float,
  reason: str | None = None,
  progress: float | None = None,
  speed: float = 0.5,
) -> dict[str, object]:
  command = [speed, 0.0, 0.0 if kind == "straight" else 0.2]
  actual = [speed * vx_gain, 0.01, command[2] * 0.95]
  row: dict[str, object] = {
    "matched_slot": slot,
    "slope_direction": "slope_up",
    "level": 1,
    "difficulty_label": "extreme",
    "difficulty": 1.0,
    "effective_terrain_parameters": {"slope": 0.45},
    "radius": 2.5,
    "speed": speed,
    "turn_sign": 1 if slot == 0 else -1,
    "repeat": 0,
    "route_kind": kind,
    "route_length": 5.235987755982989,
    "geometry": {
      "corridor_inside_patch": True,
      "scan_footprint_inside_patch": True,
      "corridor_boundary_margin": 0.5,
      "scan_boundary_margin": 0.2,
    },
    "terrain_assignment_position_error": 0.0,
    "route_placement_position_error": 0.0,
    "completed": completed,
    "failed": not completed,
    "catastrophic_termination": reason == "illegal_calf_contact",
    "first_failure_reason": None if completed else (reason or "fell_over"),
    "steps_sampled": 1000,
    "progress_ratio": (1.0 if completed else 0.5) if progress is None else progress,
    "commanded_velocity_mean": command,
    "actual_velocity_mean": actual,
    "response_gain": {
      "vx": vx_gain,
      "vy": None,
      "vy_reason": "no_nonzero_command_energy",
      "wz": None if kind == "straight" else 0.95,
      **(
        {"wz_reason": "no_nonzero_command_energy"}
        if kind == "straight" else {}
      ),
    },
    "controller_saturation_fraction": 0.01,
    "cross_track_rms": 0.05,
    "cross_track_p95": 0.10,
    "cross_track_max": 0.15,
    "cross_track_final": 0.02,
    "heading_rms": 0.04,
    "heading_p95": 0.08,
    "heading_max": 0.12,
    "heading_final": 0.01,
    "final_position_error": 0.1,
    "action_acceleration_mean": 0.1,
    "action_acceleration_p95": 0.2,
    "action_acceleration_max": 0.3,
    "slip_velocity_mean": 0.03,
    "slip_velocity_p95": 0.06,
    "slip_velocity_max": 0.09,
    "base_contact_count": 0,
    "base_contact_rate": 0.0,
    "upper_leg_contact_count": 0,
    "upper_leg_contact_rate": 0.0,
    "calf_contact_count": 1,
    "calf_contact_rate": 0.001,
    "contact_termination_summary": {
      "illegal_base_contact": 0,
      "illegal_upper_leg_contact": 0,
      "illegal_calf_contact": int(reason == "illegal_calf_contact"),
    },
    "termination_counts": {
      "fell_over": float(reason == "fell_over"),
      "illegal_calf_contact": float(reason == "illegal_calf_contact"),
    },
    "reset_count": int(not completed and reason != "step_limit"),
  }
  return row


def _matched_payload(
  completion: dict[str, bool],
  gains: dict[str, float],
  *,
  reason: str = "fell_over",
  progress: float | None = None,
  speed: float = 0.5,
) -> dict[str, object]:
  controller_limits = {
    "cross_track_gain": 1.2,
    "heading_gain": 1.0,
    "max_lateral_speed": 0.3,
    "max_yaw_rate": 0.7,
    "cross_track_tolerance": 0.3,
    "heading_tolerance": 0.35,
  }
  route_results = {}
  for kind in ("straight", "arc", "s_curve"):
    scenarios = [
      _scenario(
        kind,
        slot,
        completed=completion[kind],
        vx_gain=gains[kind],
        reason=None if completion[kind] else reason,
        progress=progress,
        speed=speed,
      )
      for slot in range(2)
    ]
    route_results[kind] = {
      "route_kind": kind,
      "route_kind_invariants": {
        "checkpoint": "/logs/model_13600.pt",
        "task_id": "Unitree-Go2-Rough-V7",
        "seed": 42,
        "profile": "clean",
        "num_envs": 2,
        "steps": 2400,
        "settle_steps": 10,
        "controller_limits": controller_limits,
      },
      "profile_settings": {"name": "clean", "control_dt": 0.02},
      "num_envs": 2,
      "terrain_assignment_position_error_max": 0.0,
      "route_placement_position_error_max": 0.0,
      "scenarios": scenarios,
    }
  return {
    "schema_version": 1,
    "evaluation_suite": "high_slope_matched_straight_arc_s_curve",
    "checkpoint": "/logs/model_13600.pt",
    "task_id": "Unitree-Go2-Rough-V7",
    "seed": 42,
    "metric_invariants": {
      "sample_denominator": "active attempt samples",
      "action_acceleration_definition": "discrete second action difference",
      "attempt_freeze": "terminal included, later samples excluded",
      "settle_lifecycle": "fixed completion settle",
    },
    "coverage": {"training_changed": False},
    "profiles": {
      "clean": {
        "matched_invariants": {
          "checkpoint": "/logs/model_13600.pt",
          "task_id": "Unitree-Go2-Rough-V7",
          "seed": 42,
          "profile": "clean",
          "num_envs_per_route_kind": 2,
          "steps": 2400,
          "settle_steps": 10,
          "control_dt": 0.02,
          "route_kinds": ["straight", "arc", "s_curve"],
          "matched_slot_order": [0, 1],
          "route_length_definition": "2*pi*radius/3",
          "fresh_environment_per_route_kind": True,
          "same_seed_environment_reconstruction": True,
          "controller_limits": controller_limits,
        },
        "route_results": route_results,
      }
    },
  }


def _stairs_payload(seed: int, failures: dict[str, str | None]) -> dict[str, object]:
  scenarios = []
  for direction in ("stairs_up", "stairs_down"):
    reason = failures.get(direction)
    scenarios.append({
      "transition_case": direction,
      "feature": "stairs",
      "direction_semantics": direction,
      "level": 9,
      "completed": reason is None,
      "failed": reason is not None,
      "first_failure_reason": reason,
      "progress_ratio": 1.0 if reason is None else 0.6,
      "reset_count": int(reason is not None),
      "termination_counts": {
        "illegal_calf_contact": float(reason == "illegal_calf_contact"),
        "fell_over": float(reason == "fell_over"),
      },
    })
  return {
    "config": {
      "target_speed": 0.5,
      "levels": [9],
      "transition_cases": ["stairs_up", "stairs_down"],
      "cross_track_offsets": [0.0],
      "yaw_offsets": [0.0],
      "profile": "randomized",
      "terrain_suite": "continuous",
      "mode": "line_follow",
      "steps": 2400,
      "seed": seed,
      "route_heading": 0.0,
      "start_forward_offset": 0.0,
    },
    "results": [{
      "checkpoint": "/logs/model_13600.pt",
      "task_id": "Unitree-Go2-Rough-V7",
      "seed": seed,
      "profile": "randomized",
      "terrain_suite": "continuous",
      "mode": "line_follow",
      "steps": 2400,
      "scenarios": scenarios,
    }],
  }


class HighSlopeAttributionTests(unittest.TestCase):
  def test_all_route_failure_with_similar_vx_underresponse_is_sustained(self) -> None:
    payload = _matched_payload(
      {kind: False for kind in ("straight", "arc", "s_curve")},
      {"straight": 0.68, "arc": 0.64, "s_curve": 0.61},
    )
    result = analyze_matched(payload, AttributionThresholds())
    attribution = result["profiles"]["clean"]["attribution"]
    self.assertEqual(
      attribution["classification"],
      "sustained_high_slope_locomotion_limitation",
    )
    self.assertEqual(
      attribution["recommended_single_variable"],
      "increase_high_extreme_sustained_slope_sampling",
    )
    self.assertFalse(attribution["training_authorized"])

  def test_straight_pass_and_curve_failure_is_curvature_coupling(self) -> None:
    payload = _matched_payload(
      {"straight": True, "arc": False, "s_curve": False},
      {"straight": 0.90, "arc": 0.75, "s_curve": 0.72},
    )
    result = analyze_matched(payload, AttributionThresholds())
    attribution = result["profiles"]["clean"]["attribution"]
    self.assertEqual(
      attribution["classification"],
      "high_slope_forward_yaw_curvature_coupling_limitation",
    )
    self.assertEqual(
      attribution["recommended_single_variable"],
      "increase_high_slope_parameterized_forward_yaw_sampling",
    )

  def test_all_routes_pass_does_not_authorize_training(self) -> None:
    payload = _matched_payload(
      {kind: True for kind in ("straight", "arc", "s_curve")},
      {kind: 0.9 for kind in ("straight", "arc", "s_curve")},
    )
    result = build_report(payload, [])
    attribution = result["matched_high_slope"]["profiles"]["clean"]["attribution"]
    self.assertEqual(attribution["classification"], "all_routes_passed_no_training")
    self.assertEqual(result["training_gate"], "NO-GO")
    json.dumps(result, allow_nan=False)

  def test_near_end_slow_step_limit_requests_exactly_one_3000_step_retry(self) -> None:
    payload = _matched_payload(
      {kind: False for kind in ("straight", "arc", "s_curve")},
      {kind: 0.78 for kind in ("straight", "arc", "s_curve")},
      reason="step_limit",
      progress=0.95,
      speed=0.3,
    )
    for route in payload["profiles"]["clean"]["route_results"].values():
      for scenario in route["scenarios"]:
        scenario["reset_count"] = 0
    result = analyze_matched(payload, AttributionThresholds())
    clean = result["profiles"]["clean"]
    self.assertEqual(clean["attribution"]["classification"], "horizon_retry_required")
    self.assertEqual(len(clean["retry_contract"]["candidates"]), 6)
    self.assertTrue(all(
      row["retry_steps"] == 3000 and row["retry_limit"] == 1
      for row in clean["retry_contract"]["candidates"]
    ))

  def test_unmatched_slot_and_unsafe_geometry_are_rejected(self) -> None:
    payload = _matched_payload(
      {kind: True for kind in ("straight", "arc", "s_curve")},
      {kind: 0.9 for kind in ("straight", "arc", "s_curve")},
    )
    mismatch = deepcopy(payload)
    mismatch["profiles"]["clean"]["route_results"]["arc"]["scenarios"][0]["speed"] = 0.3
    with self.assertRaisesRegex(ValueError, "invariants differ"):
      analyze_matched(mismatch, AttributionThresholds())
    unsafe = deepcopy(payload)
    unsafe["profiles"]["clean"]["route_results"]["straight"]["scenarios"][0]["geometry"]["scan_boundary_margin"] = -0.177
    with self.assertRaisesRegex(ValueError, "must be >= 0.0"):
      analyze_matched(unsafe, AttributionThresholds())

  def test_null_metric_requires_reason_and_nan_is_rejected(self) -> None:
    payload = _matched_payload(
      {kind: True for kind in ("straight", "arc", "s_curve")},
      {kind: 0.9 for kind in ("straight", "arc", "s_curve")},
    )
    scenario = payload["profiles"]["clean"]["route_results"]["arc"]["scenarios"][0]
    scenario["cross_track_p95"] = None
    with self.assertRaisesRegex(ValueError, "cross_track_p95_reason"):
      analyze_matched(payload, AttributionThresholds())
    scenario["cross_track_p95_reason"] = "not_retained"
    scenario["slip_velocity_mean"] = float("nan")
    with self.assertRaisesRegex(ValueError, "NaN or Inf"):
      analyze_matched(payload, AttributionThresholds())


class StairsAttributionTests(unittest.TestCase):
  def test_two_of_three_same_direction_calf_failures_are_stable(self) -> None:
    payloads = [
      _stairs_payload(42, {"stairs_up": "illegal_calf_contact"}),
      _stairs_payload(43, {"stairs_up": "illegal_calf_contact"}),
      _stairs_payload(44, {}),
    ]
    result = analyze_stairs(payloads)
    self.assertEqual(
      result["by_direction"]["stairs_up"]["classification"],
      "stable_calf_termination_risk",
    )
    self.assertEqual(result["overall_classification"], "stable_level9_stairs_calf_risk")
    self.assertFalse(result["training_authorized"])

  def test_one_of_three_calf_failures_is_low_confidence(self) -> None:
    payloads = [
      _stairs_payload(42, {"stairs_down": "illegal_calf_contact"}),
      _stairs_payload(43, {}),
      _stairs_payload(44, {}),
    ]
    result = analyze_stairs(payloads)
    self.assertEqual(
      result["by_direction"]["stairs_down"]["classification"],
      "low_confidence_or_incidental_calf_risk",
    )
    self.assertEqual(result["overall_classification"], "low_confidence_level9_stairs_risk")

  def test_heterogeneous_failures_require_more_diagnosis(self) -> None:
    payloads = [
      _stairs_payload(42, {"stairs_up": "fell_over"}),
      _stairs_payload(43, {"stairs_down": "illegal_calf_contact"}),
      _stairs_payload(44, {}),
    ]
    result = analyze_stairs(payloads)
    self.assertEqual(
      result["overall_classification"],
      "heterogeneous_failures_require_more_diagnosis",
    )

  def test_calf_failures_that_move_between_directions_are_heterogeneous(self) -> None:
    payloads = [
      _stairs_payload(42, {"stairs_up": "illegal_calf_contact"}),
      _stairs_payload(43, {"stairs_down": "illegal_calf_contact"}),
      _stairs_payload(44, {}),
    ]
    result = analyze_stairs(payloads)
    self.assertEqual(
      result["overall_classification"],
      "heterogeneous_failures_require_more_diagnosis",
    )

  def test_incomplete_seed_matrix_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "incomplete seed/direction matrix"):
      analyze_stairs([
        _stairs_payload(42, {}),
        _stairs_payload(43, {}),
      ])


if __name__ == "__main__":
  unittest.main()
