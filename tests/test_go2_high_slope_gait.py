import unittest

import torch

from scripts.diagnose_go2_high_slope_gait import (
  FOOT_NAMES,
  GaitConfig,
  TERRAIN_CONDITIONS,
  _finite_stats,
  _foot_contact,
  _normal_and_clearance,
  _require_finite,
  _scenario_slots,
  _validate_config,
)


class HighSlopeGaitContractTest(unittest.TestCase):
  def test_v7_checkpoint_and_forbidden_models(self) -> None:
    _validate_config(GaitConfig())
    with self.assertRaises(ValueError):
      _validate_config(GaitConfig(checkpoint="model_13999.pt"))

  def test_matched_slots_are_identical_across_conditions(self) -> None:
    cfg = GaitConfig(repeats=2, speeds=(0.3, 0.5))
    slots = [
      [row["matched_slot"] for row in _scenario_slots(cfg, condition, kind, level)]
      for condition, kind, level in TERRAIN_CONDITIONS
    ]
    self.assertEqual(slots[0], slots[1])
    self.assertEqual(slots[1], slots[2])
    self.assertEqual(len(slots[0]), 4)

  def test_foot_order_is_explicit(self) -> None:
    self.assertEqual(FOOT_NAMES, ("FR", "FL", "RR", "RL"))

  def test_contact_sensor_natural_order_is_reordered(self) -> None:
    class Data:
      found = torch.tensor([[[1], [0], [0], [1]]], dtype=torch.int32)

    class Sensor:
      data = Data()

    permutation = torch.tensor([1, 0, 3, 2])
    contact = _foot_contact(Sensor(), 1, 4, permutation)
    self.assertEqual(contact.tolist(), [[False, True, True, False]])

  def test_finite_stats_excludes_unavailable_values(self) -> None:
    result = _finite_stats(torch.tensor([1.0, float("nan"), 3.0]))
    self.assertEqual(result["count"], 2)
    self.assertEqual(result["max"], 3.0)
    self.assertEqual(_finite_stats(torch.tensor([float("nan")]))["count"], 0)
    with self.assertRaises(RuntimeError):
      _require_finite("actor", torch.tensor([1.0, float("nan")]))

  def test_clearance_fallback_is_explicitly_unavailable(self) -> None:
    class Data:
      hit_pos_w = None
      normals_w = None
      distances = None

    class Sensor:
      data = Data()

    foot_pos = torch.zeros(2, 4, 3)
    normal = torch.zeros(2, 4, 3)
    normal[..., 2] = 1.0
    clearance, _, valid = _normal_and_clearance(Sensor(), foot_pos, normal)
    self.assertTrue(torch.equal(clearance, torch.zeros_like(clearance)))
    self.assertFalse(valid.any())


if __name__ == "__main__":
  unittest.main()
