"""Independent CPU acceptance contracts for the final slope headroom A/B.

This suite never creates a simulator environment and never touches the GPU.
It records the reviewed A/B identity, lifecycle, geometry, metric-denominator,
and JSON contracts.  The production headroom helpers are exercised as soon as
the implementation commit is integrated; until then only that one API-facing
test class is skipped and the independent contracts remain active.
"""

from __future__ import annotations

import copy
import math
import unittest
from typing import Any, Mapping

import torch

from src.tasks.velocity.evaluation.high_slope_matched import (
  ROUTE_KINDS,
  compute_route_footprint,
  geometry_preflight,
)
from src.tasks.velocity.evaluation.routes import update_attempt_status
from src.tasks.velocity.evaluation.terrain_curved_routes import relocate_root_pose
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTIVE_SAMPLE_DEFINITION,
  OnlineTerrainRolloutMetrics,
  assert_recursive_json_finite,
)

try:
  from src.tasks.velocity.evaluation.high_slope_headroom import (
    per_axis_saturation,
    scale_controller_limits,
    summarize_axis_saturation,
  )
except ImportError:
  _HEADROOM_API_AVAILABLE = False
else:
  _HEADROOM_API_AVAILABLE = True


SCALES = (1.0, 1.5)
AXES = ("vx", "vy", "wz")
IDENTITY_FIELDS = (
  "checkpoint",
  "profile",
  "seed",
  "route_kind",
  "matched_slot",
  "slope_direction",
  "level",
  "difficulty",
  "radius",
  "speed",
  "turn_sign",
  "repeat",
  "route_length",
  "terrain_level",
  "terrain_type",
  "terrain_origin",
  "route_start",
  "initial_root_pose",
)
ONLY_CHANGED_FIELDS = (
  "controller_scale",
  "effective_controller_limits.max_lateral_speed",
  "effective_controller_limits.max_yaw_rate",
)


def _assert_reason_semantics(row: Mapping[str, Any]) -> None:
  completed = row.get("completed")
  failed = row.get("failed")
  reason = row.get("first_failure_reason")
  if not isinstance(completed, bool) or not isinstance(failed, bool):
    raise AssertionError("completed and failed must be booleans")
  if completed == failed:
    raise AssertionError("exactly one of completed and failed must be true")
  if completed and reason is not None:
    raise AssertionError("completed row must have null first_failure_reason")
  if failed and (
    not isinstance(reason, str)
    or not reason.strip()
    or reason.strip().lower() in {"none", "null", "unknown"}
  ):
    raise AssertionError("failed row requires a real nonempty failure reason")


def _assert_axis_summary(summary: Mapping[str, Any]) -> None:
  if tuple(summary) != AXES:
    raise AssertionError("per-axis saturation must be ordered vx/vy/wz")
  denominators = set()
  for axis in AXES:
    item = summary[axis]
    if set(item) != {"count", "rate", "denominator"}:
      raise AssertionError(f"invalid saturation summary for {axis}")
    count = item["count"]
    rate = item["rate"]
    denominator = item["denominator"]
    if not isinstance(count, int) or not isinstance(denominator, int):
      raise AssertionError("saturation counts and denominators must be integers")
    if denominator < 0 or not 0 <= count <= denominator:
      raise AssertionError("invalid saturation count/denominator")
    expected = None if denominator == 0 else count / denominator
    if expected is None:
      if rate is not None:
        raise AssertionError("zero denominator requires null rate")
    elif not math.isclose(float(rate), expected, abs_tol=1.0e-12):
      raise AssertionError("saturation rate uses the wrong denominator")
    denominators.add(denominator)
  if len(denominators) != 1:
    raise AssertionError("all saturation axes must share the active-step denominator")


def _assert_ab_identity(control: Mapping[str, Any], probe: Mapping[str, Any]) -> None:
  if control.get("controller_scale") != 1.0:
    raise AssertionError("control scale must be 1.0")
  if probe.get("controller_scale") != 1.5:
    raise AssertionError("probe scale must be 1.5")
  if tuple(control.get("only_changed_fields", ())) != ONLY_CHANGED_FIELDS:
    raise AssertionError("control only_changed_fields contract drifted")
  if tuple(probe.get("only_changed_fields", ())) != ONLY_CHANGED_FIELDS:
    raise AssertionError("probe only_changed_fields contract drifted")
  for field in IDENTITY_FIELDS:
    if field not in control or field not in probe:
      raise AssertionError(f"missing matched identity field {field!r}")
    if control[field] != probe[field]:
      raise AssertionError(f"A/B identity mismatch for {field!r}")
  if control.get("base_controller_limits") != probe.get("base_controller_limits"):
    raise AssertionError("base controller limits must be identical")

  base = control["base_controller_limits"]
  control_limits = control["effective_controller_limits"]
  probe_limits = probe["effective_controller_limits"]
  for name in ("max_lateral_speed", "max_yaw_rate"):
    if not math.isclose(float(control_limits[name]), float(base[name])):
      raise AssertionError(f"scale=1.0 changed {name}")
    if not math.isclose(float(probe_limits[name]), 1.5 * float(base[name])):
      raise AssertionError(f"scale=1.5 did not scale {name}")
  if any("vx" in str(key).lower() or "forward" in str(key).lower() for key in probe_limits):
    raise AssertionError("headroom A/B must not introduce or scale a forward limit")

  identities = []
  for row in (control, probe):
    fresh = row.get("fresh_environment_identity")
    if not isinstance(fresh, Mapping) or fresh.get("fresh_environment") is not True:
      raise AssertionError("every A/B rollout must declare a fresh environment")
    identity = fresh.get("instance_id")
    if not isinstance(identity, str) or not identity:
      raise AssertionError("fresh environment requires a stable nonempty instance_id")
    identities.append(identity)
  if identities[0] == identities[1]:
    raise AssertionError("A and B must use separate fresh environment instances")


def _synthetic_row(*, scale: float, instance_id: str) -> dict[str, Any]:
  row: dict[str, Any] = {
    "checkpoint": "model_13600.pt",
    "profile": "clean",
    "seed": 42,
    "route_kind": "arc",
    "matched_slot": 3,
    "slope_direction": "slope_up",
    "level": 0,
    "difficulty": 0.8,
    "radius": 2.5,
    "speed": 0.5,
    "turn_sign": -1,
    "repeat": 0,
    "route_length": 2.0 * math.pi * 2.5 / 3.0,
    "terrain_level": 0,
    "terrain_type": 2,
    "terrain_origin": [9.0, 9.0, 0.0],
    "route_start": [9.0, 9.0, 0.0],
    "initial_root_pose": [9.0, 9.0, 0.32, 1.0, 0.0, 0.0, 0.0],
    "controller_scale": scale,
    "only_changed_fields": list(ONLY_CHANGED_FIELDS),
    "base_controller_limits": {
      "max_lateral_speed": 0.3,
      "max_yaw_rate": 0.7,
    },
    "effective_controller_limits": {
      "max_lateral_speed": 0.3 * scale,
      "max_yaw_rate": 0.7 * scale,
    },
    "fresh_environment_identity": {
      "fresh_environment": True,
      "instance_id": instance_id,
    },
    "completed": True,
    "failed": False,
    "first_failure_reason": None,
    "active_control_step_samples": 10,
    "controller_saturation_by_axis": {
      axis: {"count": 0, "rate": 0.0, "denominator": 10}
      for axis in AXES
    },
    "commanded_velocity_mean": [0.5, 0.0, -0.2],
    "actual_velocity_mean": [0.4, 0.0, -0.18],
    "response_gain": {"vx": 0.8, "vy": None, "wz": 0.9},
    "cross_track_rms": 0.1,
    "cross_track_max": 0.2,
    "cross_track_final": 0.05,
    "heading_rms": 0.08,
    "heading_max": 0.16,
    "heading_final": 0.03,
    "progress_ratio": 1.0,
    "reset_count": 0,
    "action_acceleration_p95": 0.3,
    "slip_velocity_p95": 0.04,
  }
  return row


@unittest.skipUnless(
  _HEADROOM_API_AVAILABLE,
  "headroom implementation commit has not yet been integrated",
)
class ProductionHeadroomMathAcceptanceTest(unittest.TestCase):
  def test_scale_changes_only_lateral_and_yaw_limits(self) -> None:
    control = scale_controller_limits(0.3, 0.7, 1.0)
    probe = scale_controller_limits(0.3, 0.7, 1.5)
    self.assertIsNone(control.vx_limit)
    self.assertIsNone(probe.vx_limit)
    self.assertAlmostEqual(control.max_lateral_speed, 0.3)
    self.assertAlmostEqual(control.max_yaw_rate, 0.7)
    self.assertAlmostEqual(probe.max_lateral_speed, 0.45)
    self.assertAlmostEqual(probe.max_yaw_rate, 1.05)
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
      with self.subTest(scale=invalid), self.assertRaises((TypeError, ValueError)):
        scale_controller_limits(0.3, 0.7, invalid)

  def test_per_axis_saturation_is_batched_and_vx_is_never_limited(self) -> None:
    command = torch.tensor([
      [99.0, 0.29, 0.69],
      [-99.0, 0.31, -0.71],
      [0.5, -0.31, 0.20],
    ])
    mask = per_axis_saturation(
      command, max_lateral_speed=0.3, max_yaw_rate=0.7
    )
    self.assertEqual(mask.shape, command.shape)
    self.assertEqual(mask.dtype, torch.bool)
    torch.testing.assert_close(
      mask,
      torch.tensor([
        [False, False, False],
        [False, True, True],
        [False, True, False],
      ]),
    )
    summary = summarize_axis_saturation(
      mask, torch.tensor([True, True, False])
    )
    _assert_axis_summary(summary)
    self.assertEqual(summary["vx"], {"count": 0, "rate": 0.0, "denominator": 2})
    self.assertEqual(summary["vy"], {"count": 1, "rate": 0.5, "denominator": 2})
    self.assertEqual(summary["wz"], {"count": 1, "rate": 0.5, "denominator": 2})

  def test_per_axis_saturation_rejects_shape_and_broadcast_drift(self) -> None:
    for invalid in (
      torch.zeros(3),
      torch.zeros(2, 2),
      torch.zeros(2, 3, 1),
    ):
      with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
        per_axis_saturation(
          invalid, max_lateral_speed=0.3, max_yaw_rate=0.7
        )
    mask = torch.zeros((2, 3), dtype=torch.bool)
    with self.assertRaises(ValueError):
      summarize_axis_saturation(mask, torch.tensor([True]))


class MatchedIdentityAndFreshEnvironmentAcceptanceTest(unittest.TestCase):
  def test_exact_ab_pair_passes_and_declares_only_reviewed_changes(self) -> None:
    _assert_ab_identity(
      _synthetic_row(scale=1.0, instance_id="clean-arc-slot3-scale1"),
      _synthetic_row(scale=1.5, instance_id="clean-arc-slot3-scale1p5"),
    )

  def test_every_matched_identity_field_is_enforced(self) -> None:
    control = _synthetic_row(scale=1.0, instance_id="a")
    for field in IDENTITY_FIELDS:
      probe = _synthetic_row(scale=1.5, instance_id="b")
      value = probe[field]
      probe[field] = value + 1 if isinstance(value, (int, float)) else ["changed"]
      with self.subTest(field=field), self.assertRaisesRegex(
        AssertionError, "identity mismatch"
      ):
        _assert_ab_identity(control, probe)

  def test_shared_environment_or_extra_limit_change_is_rejected(self) -> None:
    control = _synthetic_row(scale=1.0, instance_id="same")
    probe = _synthetic_row(scale=1.5, instance_id="same")
    with self.assertRaisesRegex(AssertionError, "separate fresh"):
      _assert_ab_identity(control, probe)
    probe = _synthetic_row(scale=1.5, instance_id="different")
    probe["effective_controller_limits"]["forward_vx_limit"] = 0.9
    with self.assertRaisesRegex(AssertionError, "forward limit"):
      _assert_ab_identity(control, probe)

  def test_route_kind_runs_use_distinct_fresh_environment_identities(self) -> None:
    rows = []
    for route_kind in ROUTE_KINDS:
      for scale in SCALES:
        row = _synthetic_row(
          scale=scale, instance_id=f"{route_kind}-{scale}"
        )
        row["route_kind"] = route_kind
        rows.append(row)
    identities = [row["fresh_environment_identity"]["instance_id"] for row in rows]
    self.assertEqual(len(identities), len(set(identities)))


class GeometryPlacementAndLifecycleAcceptanceTest(unittest.TestCase):
  def test_r4_straight_scan_margin_is_rejected_before_gpu(self) -> None:
    footprint = compute_route_footprint("straight", 4.0, 1)
    expected = 18.0 - (9.0 + 8.0 * math.pi / 3.0 + 0.8)
    self.assertAlmostEqual(footprint.scan_boundary_margin, expected, places=9)
    self.assertAlmostEqual(footprint.scan_boundary_margin, -0.1775804096, places=9)
    self.assertFalse(footprint.scan_footprint_inside_patch)
    preflight = geometry_preflight((2.5, 4.0), (-1, 1))
    rejected = [item for item in preflight["combinations"] if not item["valid"]]
    self.assertEqual(
      {(item["route_kind"], item["radius"], item["turn_sign"]) for item in rejected},
      {("straight", 4.0, -1), ("straight", 4.0, 1)},
    )

  def test_terrain_relocation_preserves_root_relative_pose(self) -> None:
    root = torch.tensor([[1.2, -2.3, 0.7, 1.0, 0.0, 0.0, 0.0]])
    old_origin = torch.tensor([[1.0, -2.0, 0.1]])
    new_origin = torch.tensor([[20.0, 9.0, 1.1]])
    relocated, error = relocate_root_pose(root, old_origin, new_origin)
    torch.testing.assert_close(
      relocated[:, :3] - new_origin, root[:, :3] - old_origin
    )
    torch.testing.assert_close(relocated[:, 3:7], root[:, 3:7])
    self.assertLess(error, 1.0e-6)

  def test_reset_freezes_attempt_and_post_reset_episode_cannot_complete(self) -> None:
    terminal = update_attempt_status(
      active=torch.tensor([True]),
      progress=torch.tensor([0.4]),
      cross_track=torch.tensor([0.0]),
      heading_error=torch.tensor([0.0]),
      failure_mask=torch.tensor([True]),
      route_length=1.0,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    self.assertTrue(bool(terminal.sample_mask[0]))
    self.assertTrue(bool(terminal.failed_now[0]))
    self.assertFalse(bool(terminal.active[0]))
    after_reset = update_attempt_status(
      active=terminal.active,
      progress=torch.tensor([10.0]),
      cross_track=torch.tensor([0.0]),
      heading_error=torch.tensor([0.0]),
      failure_mask=torch.tensor([False]),
      route_length=1.0,
      cross_track_tolerance=0.1,
      heading_tolerance=0.1,
    )
    self.assertFalse(bool(after_reset.sample_mask[0]))
    self.assertFalse(bool(after_reset.completed_now[0]))
    self.assertFalse(bool(after_reset.failed_now[0]))


class JsonReasonAndMetricAcceptanceTest(unittest.TestCase):
  def test_completed_reason_is_null_and_failure_reason_is_real(self) -> None:
    _assert_reason_semantics({
      "completed": True, "failed": False, "first_failure_reason": None
    })
    _assert_reason_semantics({
      "completed": False,
      "failed": True,
      "first_failure_reason": "illegal_calf_contact",
    })
    invalid = (
      {"completed": True, "failed": False, "first_failure_reason": "none"},
      {"completed": False, "failed": True, "first_failure_reason": None},
      {"completed": False, "failed": True, "first_failure_reason": ""},
      {"completed": False, "failed": True, "first_failure_reason": "unknown"},
      {"completed": False, "failed": False, "first_failure_reason": "step_limit"},
    )
    for row in invalid:
      with self.subTest(row=row), self.assertRaises(AssertionError):
        _assert_reason_semantics(row)

  def test_json_is_finite_and_axis_schema_has_active_denominator(self) -> None:
    row = _synthetic_row(scale=1.0, instance_id="finite")
    assert_recursive_json_finite(row)
    _assert_axis_summary(row["controller_saturation_by_axis"])
    for value in (float("nan"), float("inf"), -float("inf")):
      invalid = copy.deepcopy(row)
      invalid["cross_track_rms"] = value
      with self.subTest(value=value), self.assertRaises(ValueError):
        assert_recursive_json_finite(invalid)

  def test_action_slip_and_contacts_share_original_attempt_denominator(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(1, 3, device="cpu")
    metrics.update(
      sample_mask=torch.tensor([True]),
      action_acceleration=torch.tensor([1.0]),
      foot_slip_velocity=torch.tensor([2.0]),
      body_contacts={
        "base": torch.tensor([True]),
        "upper_leg": torch.tensor([False]),
        "calf": torch.tensor([True]),
      },
      catastrophic_termination=torch.tensor([False]),
    )
    metrics.update(
      sample_mask=torch.tensor([False]),
      action_acceleration=torch.tensor([100.0]),
      foot_slip_velocity=torch.tensor([200.0]),
      body_contacts={
        "base": torch.tensor([True]),
        "upper_leg": torch.tensor([True]),
        "calf": torch.tensor([True]),
      },
      catastrophic_termination=torch.tensor([True]),
    )
    result = metrics.result(0)
    self.assertEqual(result["active_control_step_samples"], 1)
    self.assertEqual(result["sample_denominator_definition"], ACTIVE_SAMPLE_DEFINITION)
    self.assertEqual(result["action_acceleration"]["max"], 1.0)
    self.assertEqual(result["foot_slip_velocity"]["max"], 2.0)
    for body in ("base", "upper_leg", "calf"):
      self.assertEqual(result["body_contacts"][body]["denominator"], 1)


if __name__ == "__main__":
  unittest.main()
