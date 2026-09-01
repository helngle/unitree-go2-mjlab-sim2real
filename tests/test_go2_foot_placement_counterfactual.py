"""CPU contracts for the strict foot-placement counterfactual."""

import unittest

import torch

from scripts.audit_go2_foot_placement_counterfactual import (
  FootPlacementConfig,
  _slots,
  _jsonable,
  _validate_config,
)


class FootPlacementCounterfactualTest(unittest.TestCase):
  def test_locked_config(self) -> None:
    _validate_config(FootPlacementConfig())
    _validate_config(FootPlacementConfig(repeats=1, warmup_steps=0, sample_steps=10, formal=False))
    with self.assertRaises(ValueError):
      _validate_config(FootPlacementConfig(q_delta=-0.05))
    with self.assertRaises(ValueError):
      _validate_config(FootPlacementConfig(seed=43))

  def test_source_sham_probe_identity(self) -> None:
    rows = _slots(FootPlacementConfig(), "slope_up_high", "slope_up", 0)
    self.assertEqual([row["arm"] for row in rows[:3]], ["source", "sham", "probe"])
    self.assertEqual([row["friction"] for row in rows[:3]], [0.6, 0.6, 0.6])
    self.assertEqual([row["q_delta"] for row in rows[:3]], [0.0, 0.0, 0.05])
    self.assertEqual(len(rows), 48)

  def test_jsonable_tensor(self) -> None:
    self.assertEqual(_jsonable({"x": torch.tensor([1.0, 2.0])}), {"x": [1.0, 2.0]})


if __name__ == "__main__":
  unittest.main()
