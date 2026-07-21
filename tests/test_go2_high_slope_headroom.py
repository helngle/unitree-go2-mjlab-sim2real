"""CPU-only tests for the high-slope controller-headroom A/B harness."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest

import torch

from scripts.evaluate_go2_high_slope_headroom import (
  HighSlopeHeadroomConfig,
  _command_injection_line,
  _validate_config,
)
from src.tasks.velocity.evaluation.high_slope_headroom import (
  HEADROOM_SCALES,
  ONLY_CHANGED_CONTROLLER_FIELDS,
  ROUTE_KINDS,
  TARGET_STRATA,
  build_headroom_scenarios,
  per_axis_saturation,
  scale_controller_limits,
  summarize_axis_saturation,
  validate_headroom_pair,
  validate_headroom_result,
)


def _identity_row(kind: str, slot: int) -> dict[str, object]:
  sign = 1 if slot % 2 == 0 else -1
  direction, level = TARGET_STRATA[slot // 2]
  completed = slot != 3
  return {
    "matched_slot": slot,
    "slope_direction": direction,
    "level": level,
    "difficulty_label": "high" if level == 0 else "extreme",
    "difficulty": 0.8 if level == 0 else 1.0,
    "radius": 2.5,
    "speed": 0.5,
    "turn_sign": sign,
    "repeat": 0,
    "route_kind": kind,
    "route_length": 2.0 * math.pi * 2.5 / 3.0,
    "route_length_definition": "2*pi*radius/3",
    "route_start_xy": [9.0, 9.0],
    "route_endpoint_xy": [14.0, 9.0],
    "route_endpoint_heading": 0.0,
    "terrain_origin_xyz": [0.0, 0.0, 0.0],
    "terrain_patch_origin_xyz": [-9.0, -9.0, 0.0],
    "terrain_patch_size": [18.0, 18.0],
    "terrain_type_index": 0 if direction == "slope_up" else 1,
    "terrain_assignment_position_error": 0.0,
    "route_placement_position_error": 0.0,
    "effective_terrain_parameters": {"slope_gradient": 0.32},
    "geometry": {
      "centerline_inside_patch": True,
      "corridor_inside_patch": True,
      "scan_footprint_inside_patch": True,
    },
    "initial_root_clearance": 0.35,
    "completed": completed,
    "failed": not completed,
    "first_failure_reason": None if completed else "step_limit",
    "steps_sampled": 100,
    "controller_saturation_by_axis": {
      "vx": {"count": 0, "rate": 0.0, "denominator": 100},
      "vy": {"count": 5, "rate": 0.05, "denominator": 100},
      "wz": {"count": 10, "rate": 0.10, "denominator": 100},
    },
  }


def _scale_results() -> dict[str, dict[str, object]]:
  results: dict[str, dict[str, object]] = {}
  env_index = 0
  for scale in HEADROOM_SCALES:
    limits = scale_controller_limits(0.3, 0.7, scale)
    routes = {}
    for kind in ROUTE_KINDS:
      env_index += 1
      routes[kind] = {
        "fresh_environment_identity": f"env-{env_index}",
        "route_kind_invariants": {
          "checkpoint": "/logs/model_13600.pt",
          "task_id": "Unitree-Go2-Rough-V7",
          "seed": 42,
          "profile": "clean",
          "num_envs": 4,
          "steps": 2400,
          "settle_steps": 10,
          "controller_limits": {
            "cross_track_gain": 1.2,
            "heading_gain": 1.0,
            "max_lateral_speed": limits.max_lateral_speed,
            "max_yaw_rate": limits.max_yaw_rate,
            "cross_track_tolerance": 0.3,
            "heading_tolerance": math.radians(20.0),
          },
        },
        "profile_settings": {"profile": "clean", "control_dt": 0.02},
        "scenarios": [_identity_row(kind, slot) for slot in range(4)],
      }
    results[f"{scale:.1f}"] = {
      "controller_scale": scale,
      "effective_controller_limits": limits.as_dict(),
      "route_results": routes,
    }
  return results


class ControllerScaleContractTest(unittest.TestCase):
  def test_only_lateral_and_yaw_limits_scale(self) -> None:
    base = scale_controller_limits(0.3, 0.7, 1.0)
    larger = scale_controller_limits(0.3, 0.7, 1.5)
    self.assertIsNone(base.vx_limit)
    self.assertIsNone(larger.vx_limit)
    self.assertAlmostEqual(base.max_lateral_speed, 0.3)
    self.assertAlmostEqual(base.max_yaw_rate, 0.7)
    self.assertAlmostEqual(larger.max_lateral_speed, 0.45)
    self.assertAlmostEqual(larger.max_yaw_rate, 1.05)
    self.assertEqual(
      ONLY_CHANGED_CONTROLLER_FIELDS,
      ("max_lateral_speed", "max_yaw_rate"),
    )

  def test_nonfinite_or_nonpositive_limits_are_rejected(self) -> None:
    for args in ((0.0, 0.7, 1.0), (0.3, -0.7, 1.0), (0.3, 0.7, 0.0),
                 (math.nan, 0.7, 1.0), (0.3, math.inf, 1.0)):
      with self.subTest(args=args), self.assertRaises(ValueError):
        scale_controller_limits(*args)


class PerAxisSaturationTest(unittest.TestCase):
  def test_batch_shape_axis_boundaries_and_vx_unclamped(self) -> None:
    commands = torch.tensor([
      [100.0, 0.29, 0.69],
      [-100.0, 0.31, -0.71],
      [0.5, -0.30, 0.70],
    ])
    flags = per_axis_saturation(
      commands, max_lateral_speed=0.3, max_yaw_rate=0.7
    )
    self.assertEqual(flags.shape, (3, 3))
    self.assertFalse(bool(flags[:, 0].any()))
    torch.testing.assert_close(flags[:, 1], torch.tensor([False, True, True]))
    torch.testing.assert_close(flags[:, 2], torch.tensor([False, True, True]))

  def test_summary_uses_explicit_mask_and_denominator(self) -> None:
    flags = torch.tensor([
      [False, False, True],
      [False, True, True],
      [False, True, False],
    ])
    summary = summarize_axis_saturation(
      flags, torch.tensor([True, False, True])
    )
    self.assertEqual(tuple(summary), ("vx", "vy", "wz"))
    self.assertEqual(summary["vx"], {"count": 0, "rate": 0.0, "denominator": 2})
    self.assertEqual(summary["vy"], {"count": 1, "rate": 0.5, "denominator": 2})
    self.assertEqual(summary["wz"], {"count": 1, "rate": 0.5, "denominator": 2})

  def test_shape_broadcast_and_finite_fail_closed(self) -> None:
    with self.assertRaisesRegex(ValueError, "shape"):
      per_axis_saturation(torch.zeros(3), max_lateral_speed=0.3, max_yaw_rate=0.7)
    with self.assertRaisesRegex(ValueError, "shape"):
      per_axis_saturation(torch.zeros(2, 1, 3), max_lateral_speed=0.3, max_yaw_rate=0.7)
    with self.assertRaisesRegex(ValueError, "finite"):
      per_axis_saturation(torch.tensor([[0.0, math.nan, 0.0]]), max_lateral_speed=0.3, max_yaw_rate=0.7)


class ScenarioAndPreflightTest(unittest.TestCase):
  def test_target_strata_slots_and_route_independence(self) -> None:
    scenarios = build_headroom_scenarios()
    self.assertEqual(len(scenarios), 4)
    self.assertEqual([row.matched_slot for row in scenarios], list(range(4)))
    self.assertEqual(
      {(row.slope_direction, row.level) for row in scenarios},
      set(TARGET_STRATA),
    )
    self.assertEqual({row.radius for row in scenarios}, {2.5})
    self.assertEqual({row.speed for row in scenarios}, {0.5})
    self.assertEqual({row.turn_sign for row in scenarios}, {-1, 1})

  def test_strict_default_config_and_r4_preflight_rejection(self) -> None:
    preflight = _validate_config(HighSlopeHeadroomConfig(checkpoint="model.pt"))
    self.assertTrue(preflight["all_requested_combinations_valid"])
    with self.assertRaisesRegex(ValueError, "strict A/B pair"):
      _validate_config(HighSlopeHeadroomConfig(
        checkpoint="model.pt", controller_scales=(1.0, 2.0)
      ))
    with self.assertRaisesRegex(ValueError, "not geometrically valid"):
      _validate_config(HighSlopeHeadroomConfig(
        checkpoint="model.pt", radii=(2.5, 4.0)
      ))

  def test_capture_marker_tracks_exact_pre_policy_command_write(self) -> None:
    self.assertGreater(_command_injection_line(), 0)


class PairAndFormalResultTest(unittest.TestCase):
  def test_strict_pair_accepts_only_scaled_limits(self) -> None:
    validate_headroom_pair(
      _scale_results(), base_lateral_speed=0.3, base_yaw_rate=0.7
    )

  def test_pair_rejects_terrain_identity_or_fresh_env_reuse(self) -> None:
    terrain_mismatch = _scale_results()
    terrain_mismatch["1.5"]["route_results"]["arc"]["scenarios"][0][
      "terrain_origin_xyz"
    ] = [1.0, 0.0, 0.0]
    with self.assertRaisesRegex(ValueError, "identity mismatch"):
      validate_headroom_pair(
        terrain_mismatch, base_lateral_speed=0.3, base_yaw_rate=0.7
      )
    reused = _scale_results()
    reused["1.5"]["route_results"]["straight"][
      "fresh_environment_identity"
    ] = reused["1.0"]["route_results"]["straight"][
      "fresh_environment_identity"
    ]
    with self.assertRaisesRegex(ValueError, "reused"):
      validate_headroom_pair(reused, base_lateral_speed=0.3, base_yaw_rate=0.7)

  def test_pair_rejects_bad_axis_denominator_or_failure_reason(self) -> None:
    bad_denom = _scale_results()
    bad_denom["1.0"]["route_results"]["straight"]["scenarios"][0][
      "controller_saturation_by_axis"
    ]["vy"]["denominator"] = 99
    with self.assertRaisesRegex(ValueError, "denominator"):
      validate_headroom_pair(bad_denom, base_lateral_speed=0.3, base_yaw_rate=0.7)
    bad_reason = _scale_results()
    bad_reason["1.5"]["route_results"]["s_curve"]["scenarios"][3][
      "first_failure_reason"
    ] = None
    with self.assertRaisesRegex(ValueError, "real first_failure_reason"):
      validate_headroom_pair(bad_reason, base_lateral_speed=0.3, base_yaw_rate=0.7)
    bad_integer = _scale_results()
    bad_integer["1.0"]["route_results"]["straight"]["scenarios"][0][
      "controller_saturation_by_axis"
    ]["vy"]["count"] = 1.5
    with self.assertRaisesRegex(ValueError, "count must be an integer"):
      validate_headroom_pair(bad_integer, base_lateral_speed=0.3, base_yaw_rate=0.7)

  def test_top_level_formal_contract(self) -> None:
    payload = {
      "evaluation_suite": "high_slope_controller_headroom_ab",
      "ab_invariants": {
        "only_changed_controller_fields": list(ONLY_CHANGED_CONTROLLER_FIELDS),
        "controller_scales": list(HEADROOM_SCALES),
        "base_controller_limits": {
          "max_lateral_speed": 0.3,
          "max_yaw_rate": 0.7,
        },
      },
      "profiles": {
        "clean": {"profile": "clean", "scale_results": _scale_results()}
      },
    }
    validate_headroom_result(payload)
    changed = deepcopy(payload)
    changed["ab_invariants"]["only_changed_controller_fields"].append(
      "cross_track_gain"
    )
    with self.assertRaisesRegex(ValueError, "fields changed"):
      validate_headroom_result(changed)
    nonfinite = deepcopy(payload)
    nonfinite["profiles"]["clean"]["scale_results"]["1.0"][
      "route_results"]["straight"]["scenarios"][0]["radius"] = math.nan
    with self.assertRaisesRegex(ValueError, "non-finite"):
      validate_headroom_result(nonfinite)


if __name__ == "__main__":
  unittest.main()
