"""CPU contracts for strict causal-coverage gates."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_go2_actuator_headroom_causal_coverage import (
  CRITICAL_CELLS,
  EXPECTED_CHECKPOINT_SHA256,
  CausalCoverageConfig,
  _cell_stats,
  _finite,
  _validate_config,
  _validate_identity,
)


def _pair(cell: str, repeat: int, source: float = 0.8, sham: float = 0.8, probe: float = 0.2) -> dict:
  terrain, speed_name = cell.split("|vx_")
  return {
    "terrain_condition": terrain,
    "terrain_kind": "slope_up",
    "terrain_level": 0,
    "speed": float(speed_name),
    "repeat": repeat,
    "matched_slot": repeat,
    "trigger": {"status": "applied"},
    "branch_identity": {"branch_pass": True},
    "post_100": {"status": "complete", "source_saturation": {"fraction": source}, "sham_saturation": {"fraction": sham}, "probe_saturation": {"fraction": probe}},
    "post_300": {"status": "complete"},
  }


class CausalCoverageContractTest(unittest.TestCase):
  def test_locked_config_and_sha(self) -> None:
    _validate_config(CausalCoverageConfig())
    self.assertEqual(len(EXPECTED_CHECKPOINT_SHA256), 64)
    with self.assertRaises(ValueError):
      _validate_config(CausalCoverageConfig(repeats=8))
    with self.assertRaises(ValueError):
      _validate_config(CausalCoverageConfig(post_windows=(50, 100)))

  def test_finite_and_json_strict(self) -> None:
    self.assertTrue(_finite({"x": [1.0, None]}))
    self.assertFalse(_finite({"x": float("nan")}))
    json.dumps({"x": None}, allow_nan=False)

  def test_per_cell_coverage_and_sham_effect(self) -> None:
    pairs = [_pair(CRITICAL_CELLS[0], i) for i in range(8)]
    stats = _cell_stats(pairs, CRITICAL_CELLS[0])
    self.assertTrue(stats["coverage_pass"])
    self.assertTrue(stats["sham_effect_pass"])
    self.assertEqual(stats["valid_post100_triplets"], 8)

  def test_incomplete_post_window_does_not_count(self) -> None:
    pairs = [_pair(CRITICAL_CELLS[1], i) for i in range(8)]
    pairs[0]["post_300"] = {"status": "partial_branch_failure"}
    stats = _cell_stats(pairs, CRITICAL_CELLS[1])
    self.assertTrue(stats["coverage_pass"])
    self.assertEqual(stats["valid_post100_triplets"], 8)
    self.assertEqual(stats["valid_post300_triplets"], 7)

  def test_identity_rejects_duplicate_or_wrong_count(self) -> None:
    condition = {"terrain_condition": "slope_up_high", "pairs": [_pair(CRITICAL_CELLS[0], i) for i in range(16)]}
    result = _validate_identity([condition], 16)
    self.assertFalse(result["pass"])
    self.assertTrue(result["errors"])


if __name__ == "__main__":
  unittest.main()
