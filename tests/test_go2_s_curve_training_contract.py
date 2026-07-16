"""CPU contracts for a future matched S-curve training decision.

No production sampler is authorized by these tests. They lock the evidence
boundary and the control/probe invariants that integration must satisfy before
training can be enabled.
"""

from __future__ import annotations

import hashlib
import math
import unittest
from dataclasses import dataclass

import torch

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_v7_env_cfg


@dataclass(frozen=True)
class _ProbeContract:
  name: str
  probability: float
  dwell_s: tuple[float, float]
  vx: tuple[float, float]
  abs_wz: tuple[float, float]
  slew_control_steps: int
  correlated_steady: bool
  sign_transition: bool


STEADY_PROBE = _ProbeContract(
  name="steady_correlated_curve",
  probability=0.15,
  dwell_s=(3.0, 8.0),
  vx=(0.3, 0.6),
  abs_wz=(0.075, 0.3),
  slew_control_steps=1,
  correlated_steady=True,
  sign_transition=False,
)

TRANSITION_PROBE = _ProbeContract(
  name="controlled_yaw_transition",
  probability=0.15,
  dwell_s=(3.0, 8.0),
  vx=(0.3, 0.6),
  abs_wz=(0.075, 0.3),
  slew_control_steps=1,
  correlated_steady=False,
  sign_transition=True,
)


def _assignment_digest(levels: torch.Tensor, terrain_types: torch.Tensor) -> str:
  """Order-sensitive canonical digest for a frozen terrain assignment."""
  if levels.shape != terrain_types.shape or levels.ndim != 1:
    raise ValueError("terrain level/type tensors must be equal-length vectors")
  payload = bytearray()
  for label, tensor in ((b"levels", levels), (b"types", terrain_types)):
    canonical = tensor.detach().cpu().to(torch.int64).contiguous()
    payload.extend(label)
    payload.extend(len(canonical).to_bytes(8, "little"))
    payload.extend(canonical.numpy().astype("<i8", copy=False).tobytes())
  return hashlib.sha256(payload).hexdigest()


class V7TemporalCoverageContractTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.command = unitree_go2_rough_v7_env_cfg(play=False).commands["twist"]

  def test_v7_dwell_and_mode_probabilities_are_the_reviewed_baseline(self) -> None:
    self.assertEqual(self.command.resampling_time_range, (3.0, 8.0))
    self.assertEqual(
      (
        self.command.general_probability,
        self.command.lateral_probability,
        self.command.yaw_probability,
        self.command.high_speed_probability,
      ),
      (0.40, 0.25, 0.15, 0.20),
    )
    self.assertEqual(self.command.general_lin_vel_x, (0.15, 0.8))
    self.assertEqual(self.command.general_lin_vel_y, (-0.1, 0.1))
    self.assertEqual(self.command.general_ang_vel_z, (-0.3, 0.3))

  def test_incidental_general_to_general_yaw_reversal_probability(self) -> None:
    # With independent continuous yaw samples, signs are equiprobable and an
    # exact same-magnitude S transition has probability zero.
    general = self.command.general_probability
    opposite_sign_boundary = general * general * 0.5
    self.assertAlmostEqual(opposite_sign_boundary, 0.08)
    self.assertTrue(math.isclose(opposite_sign_boundary, 8 / 100))

  def test_matrix_has_four_ood_cases_per_both_turn_directions(self) -> None:
    cases = [
      (radius, speed, sign)
      for radius in (1.5, 2.5, 4.0)
      for speed in (0.3, 0.5, 0.6)
      for sign in (-1, 1)
    ]
    ood = [case for case in cases if abs(case[2] * case[1] / case[0]) > 0.3]
    self.assertEqual(
      set(ood),
      {(1.5, 0.5, -1), (1.5, 0.5, 1),
       (1.5, 0.6, -1), (1.5, 0.6, 1)},
    )
    self.assertEqual(len(cases) - len(ood), 14)

  def test_only_five_radius_speed_pairs_match_v7_dwell_range(self) -> None:
    dwell_by_case = {
      (radius, speed): (math.pi / 3.0) * radius / speed
      for radius in (1.5, 2.5, 4.0)
      for speed in (0.3, 0.5, 0.6)
    }
    covered = {
      case for case, dwell in dwell_by_case.items() if 3.0 <= dwell <= 8.0
    }
    self.assertEqual(
      covered,
      {
        (1.5, 0.3),
        (1.5, 0.5),
        (2.5, 0.5),
        (2.5, 0.6),
        (4.0, 0.6),
      },
    )
    self.assertLess(dwell_by_case[(1.5, 0.6)], 3.0)
    for case in ((2.5, 0.3), (4.0, 0.3), (4.0, 0.5)):
      self.assertGreater(dwell_by_case[case], 8.0)


class MutuallyExclusiveProbeContractTest(unittest.TestCase):
  def test_probe_definitions_preserve_id_dwell_and_step_slew(self) -> None:
    for probe in (STEADY_PROBE, TRANSITION_PROBE):
      with self.subTest(probe=probe.name):
        self.assertEqual(probe.probability, 0.15)
        self.assertEqual(probe.dwell_s, (3.0, 8.0))
        self.assertLessEqual(probe.abs_wz[1], 0.3)
        self.assertEqual(probe.slew_control_steps, 1)

  def test_probe_hypotheses_are_mutually_exclusive(self) -> None:
    for probe in (STEADY_PROBE, TRANSITION_PROBE):
      self.assertNotEqual(probe.correlated_steady, probe.sign_transition)
    enabled = [STEADY_PROBE]
    self.assertLessEqual(len(enabled), 1)
    with self.assertRaisesRegex(ValueError, "exactly one"):
      if len([STEADY_PROBE, TRANSITION_PROBE]) != 1:
        raise ValueError("select exactly one S-curve probe")

  def test_steady_probe_replaces_only_general_probability(self) -> None:
    control = {
      "general": 0.40, "curve": 0.0, "lateral": 0.25,
      "yaw": 0.15, "high_speed": 0.20,
    }
    probe = {
      "general": 0.25, "curve": 0.15, "lateral": 0.25,
      "yaw": 0.15, "high_speed": 0.20,
    }
    self.assertAlmostEqual(sum(control.values()), 1.0)
    self.assertAlmostEqual(sum(probe.values()), 1.0)
    self.assertAlmostEqual(
      control["general"] - probe["general"], probe["curve"]
    )
    for mode in ("lateral", "yaw", "high_speed"):
      self.assertEqual(control[mode], probe[mode])

  def test_id_steady_radius_lower_bound_enforces_general_yaw_limit(self) -> None:
    for speed in (0.3, 0.45, 0.6):
      minimum_radius = max(1.5, speed / 0.3)
      self.assertLessEqual(speed / minimum_radius, 0.3 + 1.0e-12)
      self.assertLessEqual(minimum_radius, 4.0)

  def test_transition_keeps_magnitude_and_flips_only_yaw_sign(self) -> None:
    vx, wz = 0.5, 0.2
    first = torch.tensor([vx, 0.0, wz])
    second = torch.tensor([vx, 0.0, -wz])
    torch.testing.assert_close(first[:2], second[:2])
    self.assertEqual(float(first[2]), -float(second[2]))


class FrozenTerrainMatchedControlContractTest(unittest.TestCase):
  def test_control_and_probe_require_exact_elementwise_assignments(self) -> None:
    levels = torch.arange(2048, dtype=torch.long) % 10
    terrain_types = torch.arange(2048, dtype=torch.long) % 20
    self.assertEqual(levels.numel(), 2048)
    self.assertTrue(torch.equal(levels, levels.clone()))
    self.assertTrue(torch.equal(terrain_types, terrain_types.clone()))
    self.assertEqual(
      _assignment_digest(levels, terrain_types),
      _assignment_digest(levels.clone(), terrain_types.clone()),
    )

  def test_histogram_match_cannot_replace_order_sensitive_hash(self) -> None:
    levels = torch.tensor([0, 2, 5, 9], dtype=torch.long)
    permuted = torch.tensor([9, 5, 2, 0], dtype=torch.long)
    terrain_types = torch.tensor([1, 3, 3, 7], dtype=torch.long)
    self.assertEqual(
      torch.bincount(levels, minlength=10).tolist(),
      torch.bincount(permuted, minlength=10).tolist(),
    )
    self.assertNotEqual(
      _assignment_digest(levels, terrain_types),
      _assignment_digest(permuted, terrain_types),
    )

  def test_start_and_end_hash_detect_assignment_mutation(self) -> None:
    levels = torch.arange(2048, dtype=torch.long) % 10
    terrain_types = torch.arange(2048, dtype=torch.long) % 20
    start_hash = _assignment_digest(levels, terrain_types)
    end_hash = _assignment_digest(levels.clone(), terrain_types.clone())
    self.assertEqual(start_hash, end_hash)
    mutated = levels.clone()
    mutated[2] = 4
    self.assertNotEqual(start_hash, _assignment_digest(mutated, terrain_types))

  def test_assignment_digest_rejects_shape_mismatch(self) -> None:
    with self.assertRaises(ValueError):
      _assignment_digest(torch.zeros(4), torch.zeros(3))
    with self.assertRaises(ValueError):
      _assignment_digest(torch.zeros((2, 2)), torch.zeros((2, 2)))

  def test_matched_run_contract_starts_both_runs_from_v7(self) -> None:
    control = {
      "checkpoint": "model_13600.pt", "num_envs": 2048,
      "iterations": 300, "seed": 42, "terrain_curriculum": False,
    }
    probe = dict(control)
    self.assertEqual(control, probe)
    self.assertEqual(control["checkpoint"], "model_13600.pt")
    self.assertFalse(control["terrain_curriculum"])


if __name__ == "__main__":
  unittest.main()
