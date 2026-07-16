"""CPU contracts for terrain rollout distribution and contact metrics."""

from __future__ import annotations

import json
import unittest

import torch

from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  OnlineTerrainRolloutMetrics,
  action_acceleration,
  assert_recursive_json_finite,
  contact_any,
  foot_contact_any,
  foot_slip_velocity,
)


class TerrainRolloutMetricMathTest(unittest.TestCase):
  def test_action_acceleration_preserves_discrete_second_difference(self) -> None:
    current = torch.tensor([[3.0, -1.0], [4.0, 8.0]])
    previous = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
    older = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = (current - 2 * previous + older).abs().mean(dim=-1)
    torch.testing.assert_close(
      action_acceleration(current, previous, older), expected
    )

  def test_contact_reduction_supports_multi_match_sensor_shapes(self) -> None:
    found = torch.tensor(
      [
        [[0, 0], [0, 2], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [1, 0], [0, 0]],
      ]
    )
    torch.testing.assert_close(contact_any(found, 2), torch.tensor([True, True]))
    torch.testing.assert_close(
      foot_contact_any(found, 2, 4),
      torch.tensor([[False, True, False, False], [False, False, True, False]]),
    )

  def test_foot_slip_averages_only_contacting_feet(self) -> None:
    velocity = torch.tensor([[[3.0, 4.0], [0.0, 2.0]], [[9.0, 0.0], [8.0, 0.0]]])
    contact = torch.tensor([[True, True], [False, False]])
    torch.testing.assert_close(
      foot_slip_velocity(velocity, contact), torch.tensor([3.5, 0.0])
    )

  def test_shape_and_finite_validation(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(2, 1)
    with self.assertRaises(ValueError):
      metrics.update(
        sample_mask=torch.tensor([True]),
        action_acceleration=torch.ones(2),
        foot_slip_velocity=torch.ones(2),
        body_contacts={},
        catastrophic_termination=torch.zeros(2, dtype=torch.bool),
      )
    with self.assertRaises(ValueError):
      contact_any(torch.zeros(3, 2), 2)
    with self.assertRaises(ValueError):
      foot_contact_any(torch.zeros(2, 3), 2, 4)


class TerrainRolloutAccumulatorTest(unittest.TestCase):
  def _update(
    self,
    metrics: OnlineTerrainRolloutMetrics,
    values: tuple[float, float],
    mask: tuple[bool, bool],
    *,
    catastrophic: tuple[bool, bool] = (False, False),
  ) -> None:
    vector = torch.tensor(values, dtype=torch.float64)
    metrics.update(
      sample_mask=torch.tensor(mask),
      action_acceleration=vector,
      foot_slip_velocity=vector + 0.5,
      body_contacts={
        "base": torch.tensor([values[0] >= 2.0, values[1] >= 20.0]),
        "upper_leg": torch.tensor([False, True]),
        "calf": torch.tensor([True, False]),
      },
      catastrophic_termination=torch.tensor(catastrophic),
    )

  def test_p95_max_mask_freeze_and_contact_denominator(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(2, 4, dtype=torch.float64)
    self._update(metrics, (1.0, 10.0), (True, True))
    self._update(metrics, (2.0, 20.0), (True, True), catastrophic=(True, False))
    self._update(metrics, (999.0, 30.0), (False, True))

    frozen = metrics.result(0)
    distribution = frozen["action_acceleration"]
    self.assertEqual(distribution["mean"], 1.5)
    self.assertAlmostEqual(distribution["p95"], 1.95)
    self.assertEqual(distribution["max"], 2.0)
    self.assertEqual(frozen["active_control_step_samples"], 2)
    base = frozen["body_contacts"]["base"]
    self.assertEqual(base["all_contact_count"], 1)
    self.assertEqual(base["non_terminating_count"], 0)
    self.assertEqual(base["denominator"], 2)
    self.assertEqual(frozen["catastrophic_termination"]["control_step_count"], 1)

    active = metrics.result(1)["action_acceleration"]
    self.assertEqual(active["mean"], 20.0)
    self.assertAlmostEqual(active["p95"], 29.0)
    self.assertEqual(active["max"], 30.0)

  def test_empty_samples_and_unavailable_sensors_are_null_with_reason(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(1, 1)
    metrics.update(
      sample_mask=torch.tensor([False]),
      action_acceleration=torch.tensor([1.0]),
      foot_slip_velocity=None,
      body_contacts={},
      catastrophic_termination=torch.tensor([False]),
    )
    result = metrics.result(0)
    self.assertIsNone(result["action_acceleration"]["mean"])
    self.assertEqual(
      result["action_acceleration"]["reason"], "no_active_control_step_samples"
    )
    self.assertIsNone(result["body_contacts"]["base"]["non_terminating_rate"])
    assert_recursive_json_finite(result)
    json.dumps(result, allow_nan=False)

  def test_recursive_json_rejects_nonfinite(self) -> None:
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.assertRaises(ValueError):
        assert_recursive_json_finite({"nested": [value]})


if __name__ == "__main__":
  unittest.main()
