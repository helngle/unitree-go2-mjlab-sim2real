"""CPU contracts for stance-slip stage guardrails and ordering."""

from __future__ import annotations

import unittest

from scripts.select_go2_stance_slip_checkpoint import ROUTES, select


def payload(checkpoint: str, *, completion: float, gain: float, slip: float, risk: bool = False):
  profiles = {}
  for profile in ("clean", "randomized"):
    route_results = {}
    for route in ROUTES:
      rows = []
      for index in range(16):
        rows.append({
          "completed": index < round(completion * 16), "steps_sampled": 10,
          "speed": 0.5, "response_gain": {"vx": gain},
          "terrain_tangent_stance_slip_mean": slip,
          "terrain_tangent_loaded_stance": {"loaded_stance_foot_samples": 4},
          "action_acceleration_mean": 1.0, "base_pitch_absolute_mean": 1.0,
          "base_contact_count": 1, "upper_leg_contact_count": 1,
          "calf_contact_count": 1, "catastrophic_termination": risk,
        })
      route_results[route] = {"scenarios": rows}
    profiles[profile] = {"route_results": route_results}
  return {"checkpoint": checkpoint, "profiles": profiles}


class CheckpointSelectionTest(unittest.TestCase):
  def test_guardrail_is_applied_before_lexicographic_ranking(self) -> None:
    baseline = payload("model_13600.pt", completion=0.25, gain=0.5, slip=1.0)
    unsafe_best = payload("model_13999.pt", completion=1.0, gain=1.0, slip=1.21)
    safe = payload("model_13700.pt", completion=0.75, gain=0.8, slip=1.0)
    result = select(baseline, [unsafe_best, safe])
    self.assertEqual(result["selected_checkpoint"], "model_13700.pt")
    self.assertFalse(result["stages"][0]["guardrail_pass"])

  def test_no_survivor_is_explicit(self) -> None:
    baseline = payload("model_13600.pt", completion=0.25, gain=0.5, slip=1.0)
    result = select(baseline, [payload("model_13700.pt", completion=1.0, gain=1.0, slip=1.3)])
    self.assertEqual(result["selection_status"], "NO_SAFE_SURVIVOR")
    self.assertIsNone(result["selected_checkpoint"])


if __name__ == "__main__":
  unittest.main()
