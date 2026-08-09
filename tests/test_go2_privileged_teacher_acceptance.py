"""CPU contracts for the privileged-teacher full acceptance reducer."""

from __future__ import annotations

import unittest

from src.tasks.velocity.evaluation.privileged_teacher_acceptance import (
  EXPECTED_ARTIFACTS,
  RawArtifact,
  _suite_id,
  decide,
  validate_artifact_inventory,
)


class TeacherAcceptanceContractTest(unittest.TestCase):
  def test_inventory_requires_exact_fourteen_artifacts(self) -> None:
    items = [RawArtifact("x", "/x/model_100.pt", "a" * 64, "task", name, f"/x/{name}.json")
             for name in sorted(EXPECTED_ARTIFACTS)]
    validate_artifact_inventory(items)
    with self.assertRaises(ValueError):
      validate_artifact_inventory(items[:-1])

  def test_suite_names_prevent_high_curve_collision(self) -> None:
    self.assertEqual(_suite_id("high_slope_matched"), "high_slope")
    self.assertEqual(_suite_id("terrain_curves_arc_clean"), "terrain_curves")
    self.assertEqual(_suite_id("stairs_level9_seed42_clean"), "stairs_level9")
    self.assertNotEqual(
      _suite_id("high_slope_matched"), _suite_id("terrain_curves_arc_clean")
    )

  def test_decide_uses_hard_gates_without_weighted_score(self) -> None:
    def summary(label: str, completion: float) -> dict:
      groups = []
      for profile in ("clean", "randomized"):
        for scene in ("slope_up_level0", "slope_up_level1", "slope_down_level0", "slope_down_level1"):
          for route in ("line", "arc", "s_curve"):
            groups.append(self._group("high_slope", profile, "complex", scene, route, completion, 4, label))
        for suite, count in (("flat", 3), ("continuous_retained", 12), ("terrain_curves", 16), ("stairs_level9", 2)):
          for index in range(count):
            route = ("line", "arc", "s_curve")[index % 3] if suite in {"flat", "terrain_curves"} else "line"
            groups.append(self._group(suite, profile, "retained", f"{suite}_{index}", route, completion, 1, label))
        for index in range(8):
          route = ("arc", "s_curve")[index % 2]
          groups.append(self._group("terrain_curves", profile, "complex", f"terrain_curve_slope_{index}", route, completion, 1, label))
      return {
        "checkpoint": {
          "label": label, "path": f"/{label}.pt",
          "sha256": "a" * 64, "task_id": label,
        },
        "profile_contract": {"suite": {"clean": True}},
        "evaluation_contract": {"suite": {"evaluator": "same"}},
        "groups": groups,
      }

    result = decide(v7=summary("v7", 0.5), control=summary("control", 0.5), candidate=summary("candidate", 1.0))
    self.assertEqual(result["decision"], "ACCEPT")
    self.assertFalse(result["weighted_score_used"])
    unsafe = summary("candidate", 1.0)
    unsafe["groups"][0]["terrain_tangent_slip"] = 1.0
    rejected = decide(v7=summary("v7", 0.5), control=summary("control", 0.5), candidate=unsafe)
    self.assertEqual(rejected["decision"], "REJECT")

  def test_decide_rejects_profile_or_evaluation_contract_drift(self) -> None:
    def minimal(contract: str) -> dict:
      return {
        "checkpoint": {}, "groups": [],
        "profile_contract": {"suite": contract},
        "evaluation_contract": {"suite": "same"},
      }

    with self.assertRaisesRegex(ValueError, "profile contract differs"):
      decide(
        v7=minimal("a"), control=minimal("a"), candidate=minimal("b")
      )
    drift = minimal("a")
    drift["evaluation_contract"] = {"suite": "different"}
    with self.assertRaisesRegex(ValueError, "evaluation provenance differs"):
      decide(v7=minimal("a"), control=minimal("a"), candidate=drift)

  @staticmethod
  def _group(suite, profile, category, scene, route, completion, count, label):
    return {
      "suite": suite, "profile": profile, "category": category,
      "scene": scene, "route": route, "scenario_count": count,
      "scenario_ids": [f"{suite}:{profile}:{scene}:{route}"],
      "completion": completion, "progress": completion, "forward_gain": 1.0,
      "terrain_tangent_slip": 0.0, "action_acceleration": 0.0,
      "base_pitch_absolute": 0.0, "base_contact": 0.0,
      "upper_leg_contact": 0.0, "calf_contact": 0.0, "failure_risk": 0.0,
      "action_fault_rate": 0.0, "joint_target_fault_rate": 0.0,
    }


if __name__ == "__main__":
  unittest.main()
