import json
import math
import unittest

from scripts.diagnose_go2_complex_terrain_causes import (
  ComplexTerrainConfig,
  _aggregate_rows,
  _configure_factor_profile,
  _paired_delta,
  _validate_config,
)


class _Term:
  def __init__(self):
    self.scale = 0.2


class _Group:
  def __init__(self):
    self.terms = {"height_scan": _Term()}


class _Event:
  def __init__(self):
    self.params = {
      "ranges": (0.3, 1.6),
      "operation": "abs",
      "shared_random": True,
    }


class _Cfg:
  def __init__(self):
    self.events = {"foot_friction": _Event()}
    self.observations = {"actor": _Group()}


def _clean_profile(cfg, _profile):
  cfg.events.pop("foot_friction", None)
  return {"startup_randomization_events": []}


def _payload(completed=2, gain=0.5, slip=0.2):
  scenarios = []
  for repeat in range(2):
    scenarios.append({
      "terrain_condition": "slope_up_high",
      "speed": 0.3,
      "repeat": repeat,
      "failed": repeat >= completed,
      "first_failure_reason": None if repeat < completed else "fell_over",
      "response_gain": {"vx": gain},
      "foot_slip_tangent": [{"mean": slip}],
      "step_length": [{"mean": 0.1}],
      "terrain_relative_clearance": [{"mean": 0.06}],
      "base_pitch": {"mean": 0.2},
      "action_acceleration": {"mean": 0.3},
    })
  return {"profiles": {"clean": {"conditions": {"slope_up_high": {"scenarios": scenarios}}}}}


class ComplexTerrainDiagnosticTest(unittest.TestCase):
  def test_locked_config(self):
    _validate_config(ComplexTerrainConfig())
    with self.assertRaises(ValueError):
      _validate_config(ComplexTerrainConfig(checkpoint="model_13900.pt"))
    with self.assertRaises(ValueError):
      _validate_config(ComplexTerrainConfig(factors=("unknown",)))

  def test_fixed_friction_is_single_startup_event(self):
    cfg = _Cfg()
    settings = _configure_factor_profile(
      cfg, "friction_high_1p2", _clean_profile
    )
    self.assertEqual(cfg.events["foot_friction"].params["ranges"], (1.2, 1.2))
    self.assertEqual(settings["startup_randomization_events"], ["foot_friction"])
    self.assertTrue(settings["single_variable"])

  def test_height_scan_mask_preserves_term(self):
    cfg = _Cfg()
    _configure_factor_profile(cfg, "height_scan_masked", _clean_profile)
    self.assertEqual(cfg.observations["actor"].terms["height_scan"].scale, 0.0)

  def test_aggregate_and_delta(self):
    control = _aggregate_rows(_payload(completed=1, gain=0.5, slip=0.2))
    probe = _aggregate_rows(_payload(completed=2, gain=0.6, slip=0.1))
    delta = _paired_delta(control, probe)["slope_up_high|vx_0.3"]
    self.assertEqual(delta["completion_delta"], 1)
    self.assertAlmostEqual(delta["response_gain_vx_relative_change"], 0.2)
    self.assertAlmostEqual(delta["foot_slip_tangent_relative_change"], -0.5)

  def test_attempt_mismatch_is_not_comparable(self):
    control = _aggregate_rows(_payload())
    probe_payload = _payload()
    probe_payload["profiles"]["clean"]["conditions"]["slope_up_high"]["scenarios"].pop()
    probe = _aggregate_rows(probe_payload)
    delta = _paired_delta(control, probe)["slope_up_high|vx_0.3"]
    self.assertFalse(delta["comparable"])
    self.assertIsNone(delta["completion_delta"])

  def test_empty_metric_is_null_and_strict_json(self):
    payload = _payload()
    for row in payload["profiles"]["clean"]["conditions"]["slope_up_high"]["scenarios"]:
      row.pop("step_length")
    value = _aggregate_rows(payload)["slope_up_high|vx_0.3"]["step_length"]
    self.assertIsNone(value["mean"])
    self.assertEqual(value["status"], "no_valid_samples")
    json.dumps(value, allow_nan=False)
    self.assertFalse(any(math.isnan(item) for item in []))


if __name__ == "__main__":
  unittest.main()
