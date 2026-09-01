"""CPU contracts for the strict matched friction causal evaluator."""

import json
import math
import unittest

from scripts.audit_go2_friction_contact_causal import (
  ARMS,
  FrictionCausalConfig,
  _bootstrap,
  _cell_gate,
  _finite,
  _onset,
  _paired,
  _sample_lifecycle,
  _triplet_slots,
  _validate_config,
)


def _trace(value: float, *, failed: bool = False, failure_step: int = 5) -> dict:
  length = 12
  series = {
    "slip": [value] * length,
    "cone_utilization": [value] * length,
    "tangent_force": [10.0] * length,
    "normal_force": [30.0] * length,
    "clearance_all": [0.03] * length,
    "clearance_swing": [0.06] * length,
    "raw_contact_occupancy": [0.5] * length,
    "loaded_contact_occupancy": [0.4] * length,
    "base_contact": [0.0] * length,
    "upper_leg_contact": [0.0] * length,
    "calf_contact": [0.0] * length,
    "pitch": [0.2] * length,
    "actual_vx": [value] * length,
    "action_acc": [0.1] * length,
  }
  return {
    "failure_step": failure_step if failed else None,
    "failure_before_sample": False,
    "sample_prefix_steps": failure_step + 1 if failed else length,
    "failure_status": "failed" if failed else "right_censored",
    "cone_utilization_onset_step": 1 if failed else None,
    "slip_onset_step": 1 if failed else None,
    "series": series,
  }


class FrictionCausalContractTest(unittest.TestCase):
  def test_locked_config(self) -> None:
    _validate_config(FrictionCausalConfig())
    with self.assertRaises(ValueError):
      _validate_config(FrictionCausalConfig(repeats=7))
    with self.assertRaises(ValueError):
      _validate_config(FrictionCausalConfig(seed=43))
    _validate_config(FrictionCausalConfig(repeats=1, warmup_steps=0, sample_steps=10, formal=False))
    _validate_config(FrictionCausalConfig(repeats=16))
    _validate_config(FrictionCausalConfig(probe_friction=0.8))
    _validate_config(FrictionCausalConfig(probe_friction=1.2))

  def test_triplet_order_and_identity(self) -> None:
    rows = _triplet_slots(FrictionCausalConfig(), "slope_up_high", "slope_up", 0)
    self.assertEqual([row["arm"] for row in rows[:3]], list(ARMS))
    self.assertEqual([row["friction"] for row in rows[:3]], [0.6, 0.6, 0.9])
    self.assertEqual(len(rows), 48)

  def test_failure_is_retained_as_paired_outcome(self) -> None:
    rows = _triplet_slots(FrictionCausalConfig(), "slope_up_high", "slope_up", 0)
    rows = [row | {"first_failure_reason": "fell_over" if row["arm"] == "sham" else None} for row in rows if math.isclose(row["speed"], 0.3)]
    traces = {}
    for row in rows:
      traces[(row["terrain_condition"], row["matched_slot"], row["arm"])] = _trace(
        {"source": 0.6, "sham": 0.8, "probe": 0.3}[row["arm"]],
        failed=row["arm"] == "sham",
      )
    cell = _paired(rows, traces, "slope_up_high", 0.3)
    self.assertEqual(cell["matched_triplets"], 8)
    self.assertTrue(cell["coverage_pass"])
    self.assertTrue(all(pair["completion"]["sham"] is False for pair in cell["pairs"]))
    self.assertEqual(cell["pairs"][0]["common_prefix_steps"], 6)

  def test_onset_and_null_contract(self) -> None:
    self.assertEqual(_onset([None, 0.2, 0.3], 0.2), 1)
    self.assertIsNone(_onset([None, 0.1, 0.3], 0.2))
    self.assertTrue(_finite({"x": [1.0, None]}))
    self.assertFalse(_finite({"x": float("nan")}))
    json.dumps({"x": None}, allow_nan=False)

  def test_warmup_failure_is_failed_with_zero_sample_prefix(self) -> None:
    self.assertEqual(_sample_lifecycle(12, 100, 1200), (None, True, 0, "failed"))
    self.assertEqual(_sample_lifecycle(-1, 100, 1200), (None, False, 1200, "right_censored"))

  def test_paired_uses_swing_clearance_not_all_ray_clearance(self) -> None:
    rows = _triplet_slots(FrictionCausalConfig(), "slope_up_high", "slope_up", 0)
    traces = {}
    for row in rows:
      trace = _trace(0.3)
      trace["series"]["clearance_all"] = [0.001] * 12
      trace["series"]["clearance_swing"] = [0.06] * 12
      traces[(row["terrain_condition"], row["matched_slot"], row["arm"])] = trace
    cell = _paired(rows, traces, "slope_up_high", 0.3)
    self.assertAlmostEqual(cell["pairs"][0]["metrics"]["clearance_swing"]["sham"], 0.06)

  def test_bootstrap_is_deterministic(self) -> None:
    first = _bootstrap([1.0, 2.0, 3.0], 42)
    second = _bootstrap([1.0, 2.0, 3.0], 42)
    self.assertEqual(first, second)
    self.assertGreater(first["ci95"][0], 0.0)

  def test_cell_gate_requires_noise_excess_and_onset(self) -> None:
    pairs = []
    for repeat in range(8):
      pairs.append({
        "completion": {"source": False, "sham": False, "probe": True},
        "onset_ordering": {"source": True, "sham": True, "probe": True},
        "metrics": {
          "action_acc": {"source": 0.1, "sham": 0.1, "probe": 0.1},
          "clearance_swing": {"source": 0.06, "sham": 0.06, "probe": 0.06},
          "base_contact": {"source": 0.0, "sham": 0.1, "probe": 0.1},
          "upper_leg_contact": {"source": 0.0, "sham": 0.1, "probe": 0.1},
          "calf_contact": {"source": 0.0, "sham": 0.1, "probe": 0.1},
        },
      })
    cell = {
      "coverage_pass": True,
      "pairs": pairs,
      "effects": {
        "slip": {"effect_values": [0.2] * 8, "noise_values": [0.01] * 8, "effect_mean": 0.2, "direction_fraction": 1.0},
        "cone_utilization": {"effect_values": [0.3] * 8, "noise_values": [0.01] * 8, "effect_mean": 0.3, "direction_fraction": 1.0},
        "gain": {"effect_values": [0.25] * 8, "noise_values": [0.01] * 8, "effect_mean": 0.25, "direction_fraction": 1.0},
      },
    }
    self.assertTrue(_cell_gate(cell, 42)["contact_causal_pass"])
    for pair in cell["pairs"]:
      pair["completion"]["sham"] = True
      pair["completion"]["probe"] = False
    lifecycle_gate = _cell_gate(cell, 42)
    self.assertFalse(lifecycle_gate["side_effect_pass"])
    self.assertFalse(lifecycle_gate["side_effect_status"]["failure_risk_pass"])
    for pair in cell["pairs"]:
      pair["completion"]["sham"] = False
      pair["completion"]["probe"] = True
    cell["effects"]["slip"]["noise_values"] = [0.3] * 8
    self.assertFalse(_cell_gate(cell, 42)["contact_causal_pass"])


if __name__ == "__main__":
  unittest.main()
