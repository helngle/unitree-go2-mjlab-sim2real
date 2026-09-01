import unittest

import torch

from scripts.audit_go2_actuator_headroom_counterfactual import (
  HeadroomConfig,
  _classify,
  _physical_scenarios,
  _persistent_joint_metrics,
  _risk_ratio,
  _runtime_force_identity_mask,
  _sign_test_pvalue,
  _validate_config,
)


def _pair(*, saturated=10, probe_saturated=2, control_failed=True,
          probe_failed=False, condition="slope_up_high", speed=0.3,
          risk=True):
  return {
    "terrain_condition": condition,
    "speed": speed,
    "control_failed": control_failed,
    "probe_failed": probe_failed,
    "control_failure_reason": "illegal_upper_leg_contact" if control_failed else None,
    "probe_failure_reason": "illegal_upper_leg_contact" if probe_failed else None,
    "aligned_primary": {
      "eligible": True,
      "control_persistent_joint_count": 1,
      "control_saturated_steps": saturated,
      "probe_saturated_steps": probe_saturated,
    },
    "control_gain": 0.2,
    "probe_gain": 0.25,
    "control_step": 0.05,
    "probe_step": 0.065,
    "risk_guardrails_pass": risk,
    "lifecycle_improved": control_failed and not probe_failed,
    "lifecycle_worsened": not control_failed and probe_failed,
  }


class HeadroomContractTest(unittest.TestCase):
  def test_locked_config(self):
    _validate_config(HeadroomConfig())
    with self.assertRaises(ValueError):
      _validate_config(HeadroomConfig(multipliers=(1.0, 1.5)))

  def test_physical_slots_are_interleaved(self):
    cfg = HeadroomConfig(repeats=1)
    base, physical = _physical_scenarios(cfg, "flat", "flat", 0)
    self.assertEqual(len(physical), 2 * len(base))
    for index, row in enumerate(base):
      self.assertEqual(physical[2 * index]["matched_slot"], row["matched_slot"])
      self.assertEqual(physical[2 * index + 1]["matched_slot"], row["matched_slot"])
      self.assertEqual(physical[2 * index]["arm"], "control")
      self.assertEqual(physical[2 * index + 1]["arm"], "headroom")

  def test_zero_baseline_risk_requires_zero_probe(self):
    self.assertTrue(_risk_ratio(0.0, 0.0)["passes_1p2x"])
    self.assertFalse(_risk_ratio(1.0, 0.0)["passes_1p2x"])
    self.assertTrue(_risk_ratio(1.2, 1.0)["passes_1p2x"])
    self.assertFalse(_risk_ratio(1.2001, 1.0)["passes_1p2x"])

  def test_sign_test(self):
    self.assertAlmostEqual(_sign_test_pvalue(8, 0), 1 / 256)
    self.assertIsNone(_sign_test_pvalue(0, 0))

  def test_runtime_identity_excludes_only_sample_terminal_rows(self):
    force = torch.ones(3, 4, 2)
    force[0, 3] = torch.nan
    rows = [
      {"first_failure_phase": "sample", "sample_count": 3},
      {"first_failure_phase": None, "sample_count": 4},
      {"first_failure_phase": "warmup", "sample_count": 0},
    ]
    valid, excluded = _runtime_force_identity_mask(force, rows)
    self.assertFalse(valid[0, 2].any())
    self.assertTrue(valid[1].all())
    self.assertTrue(valid[2].all())
    self.assertEqual(excluded, 2)

  def test_persistence_excludes_terminal_pd_sample(self):
    arrays = {
      "joint_pos": torch.zeros(1, 3, 1),
      "joint_vel": torch.zeros(1, 3, 1),
      "target": torch.full((1, 3, 1), 2.0),
      "force": torch.full((1, 3, 1), 1.0),
      "pd_valid": torch.tensor([[True, True, False]]),
    }
    result = _persistent_joint_metrics(
      arrays, 0, 0, 3, torch.tensor([1.0]), torch.tensor([0.5]),
      torch.tensor([1.0]), torch.tensor([0.0]), ["joint"],
    )["joint"]
    self.assertEqual(result["saturated_step_count"], 2)
    self.assertEqual(result["saturation_longest_run"], 2)
    self.assertEqual(result["pd_demand_sample_count"], 2)
    self.assertEqual(result["old_limit_exceedance_count"], 3)

  def test_all_four_verdicts(self):
    causal = []
    for speed in (0.3, 0.5):
      causal.extend(_pair(speed=speed) for _ in range(4))
    result = _classify(causal, True, True, [])
    self.assertEqual(result["verdict"], "ACTUATOR_CAUSAL")

    insufficient = [_pair(probe_saturated=8) for _ in range(8)]
    self.assertEqual(
      _classify(insufficient, True, True, [])["verdict"],
      "HEADROOM_INSUFFICIENT",
    )

    downstream = [_pair(probe_failed=True) for _ in range(8)]
    self.assertEqual(
      _classify(downstream, True, True, [])["verdict"],
      "SATURATION_DOWNSTREAM",
    )

    self.assertEqual(
      _classify(causal, False, True, [])["verdict"], "INCONCLUSIVE"
    )


if __name__ == "__main__":
  unittest.main()
