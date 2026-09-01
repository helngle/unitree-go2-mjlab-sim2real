"""CPU contracts for the matched source/sham/probe evaluator."""

import json
import unittest

import torch

from scripts.audit_go2_actuator_headroom_triplet import (
  ARMS,
  MULTIPLIERS,
  TripletConfig,
  _apply_expected_probe_ranges,
  _classify_triplets,
  _failure_delta,
  _physical_scenarios,
  _same_lifecycle,
  _validate_config,
  _window_status,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


class TripletContractTest(unittest.TestCase):
  def test_locked_config_and_arms(self) -> None:
    _validate_config(TripletConfig())
    self.assertEqual(ARMS, ("source", "sham", "probe"))
    self.assertEqual(MULTIPLIERS, (1.0, 1.0, 1.25))
    with self.assertRaises(ValueError):
      _validate_config(TripletConfig(post_windows=(50, 300)))
    _validate_config(TripletConfig(forced_trigger_step=2))
    with self.assertRaises(ValueError):
      _validate_config(TripletConfig(forced_trigger_step=1))

  def test_physical_slots_are_triplets(self) -> None:
    rows, physical = _physical_scenarios(TripletConfig(repeats=1), "flat", "flat", 0)
    self.assertEqual(len(physical), 3 * len(rows))
    for index, row in enumerate(rows):
      triplet = physical[3 * index:3 * index + 3]
      self.assertEqual([item["arm"] for item in triplet], list(ARMS))
      self.assertEqual([item["matched_slot"] for item in triplet], [row["matched_slot"]] * 3)
      self.assertEqual([item["initial_effort_limit_multiplier"] for item in triplet], [1.0] * 3)
      self.assertEqual([item["post_trigger_effort_limit_multiplier"] for item in triplet], list(MULTIPLIERS))

  def test_window_requires_all_three_branches(self) -> None:
    self.assertEqual(_window_status(100, 100, 100, 100), "complete")
    self.assertEqual(_window_status(100, 100, 90, 100), "partial_branch_failure")
    self.assertEqual(_window_status(100, 80, 80, 80), "horizon_censored")

  def test_expected_probe_ranges_use_world_joint_cross_product(self) -> None:
    expected = torch.ones(9, 5, 2)
    probe_ids = torch.tensor([2, 5, 8])
    global_ids = torch.tensor([1, 3])
    result = _apply_expected_probe_ranges(
      expected.clone(), probe_ids, global_ids, torch.tensor([True, False, True]), 1.25
    )
    self.assertTrue(torch.equal(result[torch.tensor([2, 8])[:, None], global_ids], torch.full((2, 2, 2), 1.25)))
    self.assertEqual(float(result[5, 1, 0]), 1.0)
    empty = _apply_expected_probe_ranges(
      expected.clone(), probe_ids, global_ids, torch.zeros(3, dtype=torch.bool), 1.25
    )
    self.assertTrue(torch.equal(empty, expected))

  def test_failure_delta_and_reason(self) -> None:
    source = {"failed": True, "reason": "fell_over", "failure_step": 50}
    sham = {"failed": False, "reason": None, "failure_step": None}
    delta = _failure_delta(source, sham)
    self.assertEqual(delta["failed_delta"], -1)
    self.assertIsNone(delta["failure_step_delta"])
    self.assertFalse(delta["same_reason"])
    self.assertFalse(_same_lifecycle(source, sham))
    self.assertTrue(_same_lifecycle(source, dict(source)))

  def test_classification_requires_triplet_coverage_and_finite_schema(self) -> None:
    pairs = []
    for slot in range(8):
      pairs.append({
        "matched_slot": slot,
        "trigger": {"status": "applied"},
        "branch_identity": {"branch_pass": True},
        "post_50": {"status": "complete"},
        "post_100": {
          "status": "complete",
          "source_saturation": {"count": 10},
          "sham_saturation": {"count": 10},
          "probe_saturation": {"count": 2},
        },
        "lifecycle": {"probe_vs_sham": 1},
      })
    result = _classify_triplets(pairs, True)
    self.assertEqual(result["verdict"], "TRIPLET_HEADROOM_DIRECTIONAL")
    self.assertEqual(_classify_triplets(pairs, True, True)["verdict"], "INCONCLUSIVE")
    assert_recursive_json_finite(result)
    json.dumps(result, allow_nan=False)

  def test_failed_identity_is_inconclusive(self) -> None:
    self.assertEqual(_classify_triplets([], False)["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
