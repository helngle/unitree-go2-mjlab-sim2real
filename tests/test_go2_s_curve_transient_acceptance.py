"""Independent acceptance contracts for S-curve transient evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import torch

from src.tasks.velocity.evaluation.curved_routes import make_command_tape_schedule
from src.tasks.velocity.evaluation.routes import update_attempt_status
from src.tasks.velocity.evaluation.transient_metrics import (
  OnlineCommandTransientMetrics,
  TransientMetricConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
  path = ROOT / relative_path
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _collector(num_envs: int = 1) -> OnlineCommandTransientMetrics:
  return OnlineCommandTransientMetrics(
    num_envs,
    device="cpu",
    dtype=torch.float64,
    config=TransientMetricConfig(control_dt=0.1, num_segments=2),
  )


class STransientMathAcceptanceTest(unittest.TestCase):
  def test_exact_two_segment_response_math_and_json_finiteness(self) -> None:
    collector = _collector()
    commands = torch.tensor(
      [
        [0.5, 0.0, 0.2],
        [0.5, 0.0, 0.2],
        [0.5, 0.0, -0.2],
        [0.5, 0.0, -0.2],
        [0.5, 0.0, -0.2],
      ],
      dtype=torch.float64,
    )
    actuals = torch.tensor(
      [
        [0.25, 0.0, 0.10],
        [0.50, 0.0, 0.20],
        [0.40, 0.0, 0.05],
        [0.40, 0.0, -0.22],
        [0.40, 0.0, -0.20],
      ],
      dtype=torch.float64,
    )
    for step in range(commands.shape[0]):
      collector.update(
        step_index=step,
        command=commands[step].unsqueeze(0),
        actual=actuals[step].unsqueeze(0),
        segment_index=torch.tensor([int(step >= 2)]),
        sample_mask=torch.tensor([True]),
        saturated=torch.tensor([step in (2, 3)]),
      )

    result = collector.result(0)
    first, second = result["segments"]
    self.assertAlmostEqual(first["response_gain"]["vx"], 0.75)
    self.assertAlmostEqual(second["response_gain"]["vx"], 0.8)
    self.assertAlmostEqual(first["response_gain"]["wz"], 0.75)
    self.assertAlmostEqual(second["response_gain"]["wz"], 0.6166666667)
    self.assertAlmostEqual(
      second["integrated_absolute_error"]["wz"], 0.27 * 0.1
    )
    self.assertEqual(result["transition_step"], 2)
    self.assertEqual(result["yaw_sign_switch_latency_steps"], 1)
    self.assertAlmostEqual(result["controller_saturation_fraction"], 0.4)
    self.assertEqual(second["transient"]["wz"]["rise_time_90_steps"], 1)
    self.assertAlmostEqual(
      second["transient"]["wz"]["overshoot_ratio_max"], 0.1
    )
    self.assertEqual(second["transient"]["wz"]["settling_time_10pct_steps"], 1)
    json.dumps(result, allow_nan=False)

  def test_left_right_mirror_batch_freeze_and_slew(self) -> None:
    collector = _collector(num_envs=3)
    for step, yaw in enumerate((0.2, -0.2, -0.2)):
      command = torch.tensor(
        [[0.5, 0.0, yaw], [0.5, 0.0, -yaw], [0.5, 0.0, yaw]],
        dtype=torch.float64,
      )
      actual = torch.tensor(
        [[0.4, 0.0, 0.8 * yaw], [0.4, 0.0, -0.8 * yaw], [0.4, 0.0, yaw]],
        dtype=torch.float64,
      )
      collector.update(
        step_index=step,
        command=command,
        actual=actual,
        segment_index=torch.tensor([int(step >= 1)] * 3),
        sample_mask=torch.tensor([True, True, step == 0]),
        saturated=torch.tensor([False, False, True]),
      )

    left, right, frozen = (collector.result(index) for index in range(3))
    self.assertEqual(left["yaw_sign_switch_latency_steps"], 0)
    self.assertEqual(left["yaw_sign_switch_latency_steps"], right["yaw_sign_switch_latency_steps"])
    self.assertEqual(left["command_delta_abs_max"], right["command_delta_abs_max"])
    self.assertAlmostEqual(left["command_delta_linf_max"], 0.4)
    self.assertAlmostEqual(left["command_delta_linf_mean"], 0.2)
    self.assertEqual(frozen["segments"][0]["sample_count"], 1)
    self.assertEqual(frozen["segments"][1]["sample_count"], 0)
    self.assertEqual(frozen["command_delta_abs_max"], [0.0, 0.0, 0.0])

  def test_zero_near_zero_nan_and_bad_batches_are_handled(self) -> None:
    collector = _collector()
    collector.update(
      step_index=0,
      command=torch.tensor([[0.5, 1.0e-4, 0.2]], dtype=torch.float64),
      actual=torch.tensor([[0.4, 100.0, 0.18]], dtype=torch.float64),
      segment_index=torch.tensor([0]),
      sample_mask=torch.tensor([True]),
    )
    result = collector.result(0)
    self.assertIsNone(result["segments"][0]["response_gain"]["vy"])
    self.assertEqual(result["controller_saturation_fraction"], 0.0)
    self.assertTrue(math.isfinite(result["command_delta_linf_mean"]))
    json.dumps(result, allow_nan=False)

    common = dict(
      step_index=1,
      command=torch.zeros((1, 3), dtype=torch.float64),
      actual=torch.zeros((1, 3), dtype=torch.float64),
      segment_index=torch.tensor([0]),
      sample_mask=torch.tensor([True]),
    )
    with self.assertRaisesRegex(ValueError, "finite"):
      collector.update(
        **{**common, "actual": torch.tensor([[0.0, math.nan, 0.0]])}
      )
    with self.assertRaisesRegex(ValueError, "shape"):
      collector.update(**{**common, "command": torch.zeros((2, 3))})


class SScheduleAndLifecycleAcceptanceTest(unittest.TestCase):
  def test_step_index_switch_is_pose_independent_and_mirrored(self) -> None:
    left = make_command_tape_schedule("s_curve", 2.5, 0.5, 1, 0.02, 10)
    right = make_command_tape_schedule("s_curve", 2.5, 0.5, -1, 0.02, 10)
    boundary = left.first_motion_steps
    self.assertEqual(left.segment_at(boundary - 1), 0)
    self.assertEqual(left.segment_at(boundary), 1)
    self.assertTrue(torch.equal(
      left.command_at(boundary - 1),
      torch.tensor([0.5, 0.0, 0.2]),
    ))
    self.assertTrue(torch.equal(
      left.command_at(boundary),
      torch.tensor([0.5, 0.0, -0.2]),
    ))
    self.assertTrue(torch.equal(
      left.command_at(boundary),
      right.command_at(boundary) * torch.tensor([1.0, 1.0, -1.0]),
    ))

  def test_reset_and_inactive_attempts_cannot_accrue_a_new_episode(self) -> None:
    first = update_attempt_status(
      active=torch.tensor([True, True]),
      progress=torch.tensor([0.4, 0.4]),
      cross_track=torch.zeros(2),
      heading_error=torch.zeros(2),
      failure_mask=torch.tensor([True, False]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    self.assertTrue(torch.equal(first.sample_mask, torch.tensor([True, True])))
    self.assertTrue(torch.equal(first.active, torch.tensor([False, True])))
    second = update_attempt_status(
      active=first.active,
      progress=torch.tensor([1.0, 0.8]),
      cross_track=torch.zeros(2),
      heading_error=torch.zeros(2),
      failure_mask=torch.tensor([False, False]),
      route_length=1.0,
      cross_track_tolerance=0.3,
      heading_tolerance=0.3,
    )
    self.assertFalse(bool(second.sample_mask[0]))
    self.assertFalse(bool(second.completed_now[0]))

  def test_v7_general_yaw_boundary_splits_matrix_into_id_and_ood(self) -> None:
    rates = [
      speed / radius
      for radius in (1.5, 2.5, 4.0)
      for speed in (0.3, 0.5, 0.6)
      for _ in (1, -1)
    ]
    in_distribution = [rate for rate in rates if rate <= 0.3 + 1.0e-8]
    out_of_distribution = [rate for rate in rates if rate > 0.3 + 1.0e-8]
    self.assertEqual((len(in_distribution), len(out_of_distribution)), (14, 4))
    source = (ROOT / "scripts/evaluate_go2_curved_routes.py").read_text()
    self.assertIn('"general_yaw_in_distribution": abs(required_yaw_rate) <= 0.3 + 1e-8', source)


class EvaluatorWiringAcceptanceTest(unittest.TestCase):
  def test_command_is_refreshed_into_actor_observation_before_policy(self) -> None:
    source = (ROOT / "scripts/evaluate_go2_curved_routes.py").read_text()
    assignment = "command_term.vel_command_b[:] = command"
    refresh = "observation = wrapped.get_observations()"
    inference = "action = policy(observation)"
    start = source.index(assignment)
    refresh_index = source.index(refresh, start)
    inference_index = source.index(inference, refresh_index)
    self.assertLess(start, refresh_index)
    self.assertLess(refresh_index, inference_index)

    env_source = (ROOT / "src/tasks/velocity/velocity_env_cfg.py").read_text()
    self.assertIn('"command": ObservationTermCfg(', env_source)
    self.assertIn("func=mdp.generated_commands", env_source)
    self.assertIn('params={"command_name": "twist"}', env_source)

  def test_obsolete_pose_extended_tape_json_is_rejected(self) -> None:
    diagnostics = _load_script(
      "command_response_diagnostics_acceptance",
      "scripts/diagnose_go2_command_response.py",
    )
    obsolete = {
      "config": {"route_kind": "arc", "mode": "command_tape"},
      "scenarios": [{"required_yaw_rate": 0.2}],
    }
    with self.assertRaisesRegex(ValueError, "fixed-time scheduled tape"):
      diagnostics.validate_scheduled_arc_tape(obsolete)


if __name__ == "__main__":
  unittest.main()
