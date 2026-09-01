import unittest

import torch

from scripts.audit_go2_actuator_headroom_triggered import (
  TriggeredConfig,
  _classify,
  _control_trigger_valid,
  _lifecycle_outcome_flags,
  _physical_scenarios,
  _risk_regression,
  _risk_window,
  _sign_test_pvalue,
  _strict_identity_gate,
  _update_streak,
  _validate_config,
  _window_status,
)


def _pair(*, speed=0.3, saturation=(10, 2), win=True, loss=False,
          harm=False, risk=True, completion=(True, False)):
  control_failed, probe_failed = completion
  window = {
    "common_steps": 100,
    "control_saturation": {"count": saturation[0]},
    "probe_saturation": {
      "count": saturation[1], "old_limit_exceedance_count": 1,
    },
    "risk_guardrails_pass": risk,
  }
  post_300 = {
    **window,
    "common_steps": 300,
    "common_control": {
      "response_gain": {"vx": 0.2},
      "step_length_fully_contained": {"mean": 0.05},
    },
    "common_probe": {
      "response_gain": {"vx": 0.25},
      "step_length_fully_contained": {"mean": 0.065},
    },
  }
  return {
    "terrain_condition": "slope_up_high",
    "speed": speed,
    "trigger": {"status": "applied"},
    "branch_identity": {"branch_pass": True},
    "pre_100": {"eligible": True},
    "post_50": dict(window),
    "post_100": dict(window),
    "post_300": post_300,
    "control_lifecycle": {"failed": control_failed},
    "probe_lifecycle": {"failed": probe_failed},
    "outcome": {
      "lifecycle_win": win, "lifecycle_loss": loss, "harm": harm,
    },
  }


class TriggeredHeadroomContractTest(unittest.TestCase):
  def test_locked_config(self):
    _validate_config(TriggeredConfig())
    with self.assertRaises(ValueError):
      _validate_config(TriggeredConfig(trigger_run_steps=2))

  def test_physical_slots_start_at_baseline(self):
    base_rows, physical = _physical_scenarios(
      TriggeredConfig(repeats=1), "flat", "flat", 0
    )
    self.assertEqual(len(physical), 2 * len(base_rows))
    for index in range(len(base_rows)):
      self.assertEqual(physical[2 * index]["arm"], "control")
      self.assertEqual(physical[2 * index + 1]["arm"], "probe")
      self.assertEqual(physical[2 * index + 1]["initial_effort_limit_multiplier"], 1.0)

  def test_trigger_requires_same_joint_consecutive_steps(self):
    streak = torch.zeros(1, 2, dtype=torch.long)
    valid = torch.tensor([True])
    streak = _update_streak(streak, torch.tensor([[True, False]]), valid)
    streak = _update_streak(streak, torch.tensor([[False, True]]), valid)
    streak = _update_streak(streak, torch.tensor([[True, False]]), valid)
    self.assertLess(int(streak.max()), 3)
    for _ in range(3):
      streak = _update_streak(streak, torch.tensor([[True, False]]), valid)
    self.assertGreaterEqual(int(streak[0, 0]), 3)

  def test_invalid_row_breaks_streak(self):
    streak = torch.tensor([[2]], dtype=torch.long)
    result = _update_streak(streak, torch.tensor([[True]]), torch.tensor([False]))
    self.assertEqual(int(result[0, 0]), 0)

  def test_control_trigger_does_not_depend_on_unused_probe(self):
    active = torch.tensor([True, False, True, True])
    reset = torch.tensor([False, True, True, False])
    valid = _control_trigger_valid(active, reset, torch.tensor([0, 2]))
    self.assertEqual(valid.tolist(), [True, False])

  def test_horizon_censor_is_not_failure(self):
    self.assertEqual(
      _window_status(1300, 1200, 1200, False, False),
      "horizon_censored",
    )
    self.assertEqual(
      _window_status(1300, 900, 900, True, True),
      "both_failed_same_step",
    )

  def test_no_trigger_cannot_be_labeled_intervention_harm(self):
    self.assertEqual(
      _lifecycle_outcome_flags(False, False, True, 1200, 500),
      (False, False),
    )
    self.assertEqual(
      _lifecycle_outcome_flags(True, False, True, 1200, 500),
      (True, False),
    )

  def test_missing_risk_metric_is_not_a_regression(self):
    self.assertFalse(_risk_regression({
      "slip": {"passes_1p2x": False, "reason": "missing_metric"},
    }))
    self.assertTrue(_risk_regression({
      "slip": {"passes_1p2x": False, "ratio": 1.3},
    }))

  def test_only_flat_no_trigger_lifecycle_is_a_hard_sentinel(self):
    def condition(terrain_condition):
      return {
        "runtime_identity": {"runtime_pass": True},
        "initial_identity": {"pairing_pass": True},
        "terrain_assignment_position_error_max": 0.0,
        "terrain_placement_position_error_max": 0.0,
        "pairs": [{
          "terrain_condition": terrain_condition,
          "trigger": {"status": "not_triggered", "detector_replay_pass": True},
          "branch_identity": {"branch_pass": True},
          "control_lifecycle": {"failed": False, "sample_count": 1200, "reason": None},
          "probe_lifecycle": {"failed": True, "sample_count": 500, "reason": "fell_over"},
        }],
      }
    self.assertFalse(_strict_identity_gate([condition("flat")]))
    self.assertTrue(_strict_identity_gate([condition("slope_up_high")]))

  def test_sign_test(self):
    self.assertAlmostEqual(_sign_test_pvalue(8, 0), 1 / 256)
    self.assertIsNone(_sign_test_pvalue(0, 0))

  def test_all_four_verdicts(self):
    causal = []
    for speed in (0.3, 0.5):
      causal.extend(_pair(speed=speed) for _ in range(4))
    self.assertEqual(
      _classify(causal, True, True, False)["verdict"],
      "TRIGGERED_HEADROOM_CAUSAL",
    )
    downstream = [_pair(harm=True, risk=False) for _ in range(8)]
    self.assertEqual(
      _classify(downstream, True, True, False)["verdict"],
      "SATURATION_DOWNSTREAM",
    )
    insufficient = [_pair(saturation=(10, 8)) for _ in range(8)]
    self.assertEqual(
      _classify(insufficient, True, True, False)["verdict"],
      "HEADROOM_INSUFFICIENT",
    )
    self.assertEqual(
      _classify(causal, False, True, False)["verdict"], "INCONCLUSIVE"
    )
    self.assertEqual(
      _classify(causal, True, True, True)["verdict"], "INCONCLUSIVE"
    )

  def test_completion_delta_ignores_no_trigger_probe_divergence(self):
    causal = []
    for speed in (0.3, 0.5):
      causal.extend(_pair(speed=speed) for _ in range(4))
    unused_probe = _pair(speed=0.3, completion=(False, True))
    unused_probe["trigger"] = {"status": "not_triggered"}
    result = _classify(causal + [unused_probe], True, True, False)
    self.assertEqual(result["completion_delta"], {"vx_0.3": 4, "vx_0.5": 4})


if __name__ == "__main__":
  unittest.main()
