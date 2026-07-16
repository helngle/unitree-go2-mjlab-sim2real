"""Pure CPU contracts for online curved-route transient metrics."""

from __future__ import annotations

import json
import math
import unittest

import torch

from src.tasks.velocity.evaluation.curved_routes import make_command_tape_schedule
from src.tasks.velocity.evaluation.transient_metrics import (
  OnlineCommandTransientMetrics,
  TransientMetricConfig,
)


def _collector(num_envs: int = 1, num_segments: int = 2) -> OnlineCommandTransientMetrics:
  return OnlineCommandTransientMetrics(
    num_envs,
    device="cpu",
    dtype=torch.float32,
    config=TransientMetricConfig(control_dt=0.02, num_segments=num_segments),
  )


class ScheduledSegmentContractTest(unittest.TestCase):
  def test_s_curve_segment_switch_is_step_only(self) -> None:
    schedule = make_command_tape_schedule(
      "s_curve", radius=2.0, speed=0.5, turn_sign=1,
      control_dt=0.1, settle_steps=3,
    )
    self.assertEqual(schedule.segment_at(0), 0)
    self.assertEqual(schedule.segment_at(schedule.first_motion_steps - 1), 0)
    self.assertEqual(schedule.segment_at(schedule.first_motion_steps), 1)
    self.assertEqual(schedule.segment_at(schedule.total_steps + 100), 1)
    with self.assertRaises(ValueError):
      schedule.segment_at(-1)

  def test_arc_never_reports_a_second_segment(self) -> None:
    schedule = make_command_tape_schedule(
      "arc", radius=2.0, speed=0.5, turn_sign=-1,
      control_dt=0.1, settle_steps=3,
    )
    self.assertEqual(schedule.segment_at(0), 0)
    self.assertEqual(schedule.segment_at(schedule.total_steps + 100), 0)


class OnlineTransientMetricTest(unittest.TestCase):
  def _run_sign_switch(self, sign: float) -> dict[str, object]:
    collector = _collector()
    commands = (
      [0.5, 0.0, 0.25 * sign],
      [0.5, 0.0, 0.25 * sign],
      [0.5, 0.0, -0.25 * sign],
      [0.5, 0.0, -0.25 * sign],
      [0.5, 0.0, -0.25 * sign],
    )
    actuals = (
      [0.2, 0.0, 0.10 * sign],
      [0.5, 0.0, 0.25 * sign],
      [0.5, 0.0, 0.08 * sign],
      [0.5, 0.0, -0.30 * sign],
      [0.5, 0.0, -0.25 * sign],
    )
    for step, (command, actual) in enumerate(zip(commands, actuals, strict=True)):
      collector.update(
        step_index=step,
        command=torch.tensor([command]),
        actual=torch.tensor([actual]),
        segment_index=torch.tensor([int(step >= 2)]),
        sample_mask=torch.tensor([True]),
        saturated=torch.tensor([step == 3]),
      )
    return collector.result(0)

  def test_sign_switch_latency_rise_settling_and_overshoot(self) -> None:
    result = self._run_sign_switch(1.0)
    self.assertEqual(result["transition_step"], 2)
    self.assertAlmostEqual(result["transition_time_s"], 0.04)
    self.assertEqual(result["yaw_sign_switch_latency_steps"], 1)
    self.assertAlmostEqual(result["yaw_sign_switch_latency_s"], 0.02)
    self.assertAlmostEqual(result["controller_saturation_fraction"], 0.2)
    second_yaw = result["segments"][1]["transient"]["wz"]
    self.assertEqual(second_yaw["rise_time_90_steps"], 1)
    self.assertEqual(second_yaw["settling_time_10pct_steps"], 2)
    self.assertAlmostEqual(second_yaw["overshoot_ratio_max"], 0.2, places=5)
    self.assertAlmostEqual(
      result["segments"][0]["response_gain"]["vx"], 0.7, places=6
    )
    self.assertAlmostEqual(
      result["segments"][1]["integrated_absolute_error"]["wz"],
      (0.33 + 0.05) * 0.02,
      places=6,
    )

  def test_left_right_transients_are_mirrors(self) -> None:
    left = self._run_sign_switch(1.0)
    right = self._run_sign_switch(-1.0)
    self.assertEqual(
      left["yaw_sign_switch_latency_steps"],
      right["yaw_sign_switch_latency_steps"],
    )
    for segment in range(2):
      left_segment = left["segments"][segment]
      right_segment = right["segments"][segment]
      self.assertAlmostEqual(
        left_segment["response_gain"]["wz"],
        right_segment["response_gain"]["wz"],
      )
      self.assertEqual(
        left_segment["transient"]["wz"],
        right_segment["transient"]["wz"],
      )

  def test_no_switch_and_zero_target_have_explicit_null_reasons(self) -> None:
    collector = _collector(num_segments=1)
    for step in range(3):
      collector.update(
        step_index=step,
        command=torch.tensor([[0.5, 0.0, 0.2]]),
        actual=torch.tensor([[0.5, 0.0, 0.2]]),
        segment_index=torch.tensor([0]),
        sample_mask=torch.tensor([True]),
      )
    result = collector.result(0)
    self.assertIsNone(result["transition_step"])
    self.assertIsNone(result["yaw_sign_switch_latency_steps"])
    self.assertEqual(
      result["yaw_sign_switch_latency_reason"], "no_segment_transition"
    )
    lateral = result["segments"][0]["transient"]["vy"]
    self.assertIsNone(lateral["rise_time_90_s"])
    self.assertEqual(lateral["rise_time_reason"], "no_nonzero_target")
    self.assertIsNone(result["segments"][0]["response_gain"]["vy"])

  def test_near_zero_and_sign_changing_commands_do_not_create_ratio_spikes(self) -> None:
    near_zero = _collector(num_segments=1)
    for step in range(2):
      near_zero.update(
        step_index=step,
        command=torch.tensor([[0.5, (-1.0 if step else 1.0) * 1.2e-4, 0.2]]),
        actual=torch.tensor([[0.4, 0.01, 0.18]]),
        segment_index=torch.tensor([0]),
        sample_mask=torch.tensor([True]),
      )
    self.assertIsNone(near_zero.result(0)["segments"][0]["response_gain"]["vy"])

    varying = _collector(num_segments=1)
    for step, target in enumerate((0.1, -0.1)):
      varying.update(
        step_index=step,
        command=torch.tensor([[0.5, target, 0.2]]),
        actual=torch.tensor([[0.4, 0.8 * target, 0.18]]),
        segment_index=torch.tensor([0]),
        sample_mask=torch.tensor([True]),
      )
    self.assertAlmostEqual(
      varying.result(0)["segments"][0]["response_gain"]["vy"], 0.8,
      places=6,
    )

  def test_never_reaches_target_reports_unavailable(self) -> None:
    collector = _collector(num_segments=1)
    for step in range(4):
      collector.update(
        step_index=step,
        command=torch.tensor([[0.5, 0.0, 0.2]]),
        actual=torch.zeros((1, 3)),
        segment_index=torch.tensor([0]),
        sample_mask=torch.tensor([True]),
      )
    result = collector.result(0)["segments"][0]["transient"]
    self.assertEqual(result["vx"]["rise_time_reason"], "target_not_reached")
    self.assertEqual(result["wz"]["rise_time_reason"], "target_not_reached")
    self.assertEqual(result["vx"]["settling_time_reason"], "outside_band_at_end")
    self.assertIsNone(result["wz"]["settling_time_10pct_steps"])

  def test_zero_command_settle_does_not_erase_prior_settling(self) -> None:
    collector = _collector(num_segments=1)
    collector.update(
      step_index=0,
      command=torch.tensor([[0.5, 0.0, 0.2]]),
      actual=torch.tensor([[0.5, 0.0, 0.2]]),
      segment_index=torch.tensor([0]),
      sample_mask=torch.tensor([True]),
    )
    collector.update(
      step_index=1,
      command=torch.zeros((1, 3)),
      actual=torch.zeros((1, 3)),
      segment_index=torch.tensor([0]),
      sample_mask=torch.tensor([True]),
    )
    transient = collector.result(0)["segments"][0]["transient"]
    self.assertEqual(transient["vx"]["settling_time_10pct_steps"], 0)
    self.assertEqual(transient["wz"]["settling_time_10pct_steps"], 0)
    self.assertEqual(collector.result(0)["segments"][0]["sample_count"], 1)

  def test_batch_mask_freezes_failed_attempt_and_json_is_strict(self) -> None:
    collector = _collector(num_envs=2, num_segments=1)
    collector.update(
      step_index=0,
      command=torch.tensor([[0.5, 0.0, 0.2], [0.5, 0.0, -0.2]]),
      actual=torch.tensor([[0.4, 0.0, 0.1], [0.4, 0.0, -0.1]]),
      segment_index=torch.tensor([0, 0]),
      sample_mask=torch.tensor([True, True]),
    )
    collector.update(
      step_index=1,
      command=torch.tensor([[99.0, 99.0, 99.0], [0.5, 0.0, -0.2]]),
      actual=torch.tensor([[99.0, 99.0, 99.0], [0.5, 0.0, -0.2]]),
      segment_index=torch.tensor([0, 0]),
      sample_mask=torch.tensor([False, True]),
    )
    first = collector.result(0)
    second = collector.result(1)
    self.assertEqual(first["segments"][0]["sample_count"], 1)
    self.assertEqual(second["segments"][0]["sample_count"], 2)
    self.assertEqual(first["command_delta_abs_max"], [0.0, 0.0, 0.0])
    json.dumps({"first": first, "second": second}, allow_nan=False)

  def test_invalid_batch_and_segment_are_rejected(self) -> None:
    collector = _collector()
    common = dict(
      step_index=0,
      command=torch.zeros((1, 3)),
      actual=torch.zeros((1, 3)),
      sample_mask=torch.tensor([True]),
    )
    with self.assertRaises(ValueError):
      collector.update(**common, segment_index=torch.tensor([2]))
    with self.assertRaises(ValueError):
      collector.update(**common, segment_index=torch.tensor([0.0]))
    with self.assertRaises(ValueError):
      collector.update(
        **{**common, "command": torch.zeros((2, 3))},
        segment_index=torch.tensor([0]),
      )

  def test_config_validation(self) -> None:
    for kwargs in (
      {"control_dt": 0.0, "num_segments": 1},
      {"control_dt": 0.02, "num_segments": 0},
      {"control_dt": 0.02, "num_segments": 1, "rise_fraction": 1.1},
      {"control_dt": 0.02, "num_segments": 1, "settling_fraction": 1.0},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        TransientMetricConfig(**kwargs)


if __name__ == "__main__":
  unittest.main()
