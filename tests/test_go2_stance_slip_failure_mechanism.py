import unittest

from scripts.diagnose_go2_stance_slip_failure_mechanism import (
  CHECKPOINTS,
  MechanismConfig,
  _direction_summary,
  _pair_rows,
  _validate_config,
  _weighted_mean,
)


class StanceSlipFailureMechanismTest(unittest.TestCase):
  def test_formal_contract_is_fixed(self) -> None:
    _validate_config(MechanismConfig(), verify_files=False)
    with self.assertRaises(ValueError):
      _validate_config(
        MechanismConfig(checkpoint_labels=("v7",), formal=True),
        verify_files=False,
      )
    with self.assertRaises(ValueError):
      _validate_config(MechanismConfig(sample_steps=1200), verify_files=False)
    _validate_config(
      MechanismConfig(checkpoint_labels=("v7",), chunk_mode=True),
      verify_files=False,
    )
    with self.assertRaises(ValueError):
      _validate_config(
        MechanismConfig(checkpoint_labels=("v7", "model_13700"), chunk_mode=True),
        verify_files=False,
      )
    self.assertEqual(tuple(CHECKPOINTS), ("v7", "model_13700", "model_13900", "model_13999"))

  def test_weighted_mean_honors_sample_counts(self) -> None:
    self.assertEqual(
      _weighted_mean([{"mean": 1.0, "count": 1}, {"mean": 3.0, "count": 3}]),
      2.5,
    )
    self.assertIsNone(_weighted_mean([{"mean": None, "count": 0}]))

  def test_paired_direction_requires_equal_identity_and_gait_change(self) -> None:
    base = {
      "matched_slot": 0, "speed": 0.3, "repeat": 0, "sample_count": 2400,
      "tangent_slip_loaded_20_10": 0.10, "response_gain_vx": 0.5,
      "step_length_absolute": 0.20, "duty_factor_completed_intervals": 0.5,
      "stance_duration_s": 0.2,
    }
    candidate = dict(base)
    candidate.update({
      "tangent_slip_loaded_20_10": 0.05,
      "response_gain_vx": 0.4,
      "step_length_absolute": 0.15,
    })
    pairs = _pair_rows([base], [candidate])
    self.assertTrue(pairs[0]["direction_flags"]["reward_avoidance_pattern"])
    summary = _direction_summary(pairs)
    self.assertEqual(summary["reward_avoidance_pattern_count"], 1)
    self.assertFalse(summary["time_order_available"])
    bad = dict(candidate, matched_slot=1)
    with self.assertRaises(RuntimeError):
      _pair_rows([base], [bad])


if __name__ == "__main__":
  unittest.main()
