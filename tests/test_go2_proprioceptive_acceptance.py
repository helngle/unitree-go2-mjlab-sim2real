from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest import mock
import sys

import torch

import scripts.build_go2_proprioceptive_checkpoint_lineage as lineage_builder
import scripts.evaluate_go2_proprioceptive_acceptance as orchestrator
import scripts.screen_go2_proprioceptive_checkpoints as screener
from src.tasks.velocity.evaluation import proprio_acceptance as acceptance
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  OnlineTerrainRolloutMetrics,
)


WORKSPACE = Path(__file__).resolve().parents[1]


class ProprioProfileContractTest(unittest.TestCase):
  def _env(self):
    return SimpleNamespace(
      observations={"actor": SimpleNamespace(enable_corruption=True)},
      events={"push_robot": SimpleNamespace(mode="interval", params={})},
    )

  def _install(self, env_cfg) -> None:
    for name in acceptance.SIM2REAL_RANDOMIZATION_EVENTS:
      env_cfg.events[name] = SimpleNamespace(
        mode="startup", interval_range_s=None, params={"name": name}
      )

  def test_clean_and_randomized_are_exact_seven_event_contracts(self) -> None:
    with mock.patch.object(
      acceptance, "install_sim2real_randomization_contract", self._install
    ):
      clean_env = self._env()
      clean = acceptance.configure_sim2real_profile(clean_env, "clean")
      self.assertEqual(clean["canonical_profile"], "clean")
      self.assertEqual(clean["startup_randomization_events"], [])
      self.assertFalse(clean["actor_observation_corruption"])
      self.assertFalse(clean["push_enabled"])

      randomized_env = self._env()
      randomized = acceptance.configure_sim2real_profile(
        randomized_env, "randomized"
      )
      self.assertEqual(
        randomized["startup_randomization_events"],
        list(acceptance.SIM2REAL_RANDOMIZATION_EVENTS),
      )
      self.assertIn("pd_gains", randomized["event_parameters"])
      self.assertIn("limb_pseudo_inertia", randomized["event_parameters"])
      self.assertTrue(randomized["actor_observation_corruption"])

  def test_effort_and_power_are_finite_absolute_mechanical_definitions(self) -> None:
    robot = SimpleNamespace(data=SimpleNamespace(
      actuator_force=torch.tensor([[2.0, -3.0]]),
      joint_vel=torch.tensor([[4.0, 5.0]]),
    ))
    effort, power = acceptance.actuator_effort_and_power(robot)
    self.assertEqual(effort.tolist(), [2.5])
    self.assertEqual(power.tolist(), [23.0])

  def test_normalized_and_processed_action_limits_share_deploy_contract(self) -> None:
    action = torch.zeros((3, 12))
    action[1, 0] = 4.01
    action[2, 2] = -4.0
    max_abs, fault = acceptance.normalized_action_safety(action)
    self.assertAlmostEqual(max_abs[0].item(), 0.0)
    self.assertAlmostEqual(max_abs[1].item(), 4.01, places=5)
    self.assertAlmostEqual(max_abs[2].item(), 4.0)
    self.assertEqual(fault.tolist(), [False, True, True])

  def test_effort_and_power_use_only_active_attempt_samples(self) -> None:
    metrics = OnlineTerrainRolloutMetrics(1, 2, control_dt_s=0.02)
    for active, effort, power in ((True, 2.0, 20.0), (False, 999.0, 999.0)):
      metrics.update(
        sample_mask=torch.tensor([active]),
        action_acceleration=torch.tensor([0.0]),
        foot_slip_velocity=torch.tensor([0.0]),
        body_contacts={},
        catastrophic_termination=torch.tensor([False]),
        base_pitch=torch.tensor([0.1]),
        actuator_effort_abs=torch.tensor([effort]),
        mechanical_power_abs=torch.tensor([power]),
      )
    result = metrics.result(0)
    self.assertEqual(result["actuator_effort_abs"]["mean"], 2.0)
    self.assertEqual(result["mechanical_power_abs"]["mean"], 20.0)
    self.assertEqual(result["mechanical_energy_abs"], 0.4)


class ProprioRolloutCallContractTest(unittest.TestCase):
  def test_every_rollout_update_supplies_pitch_effort_and_power(self) -> None:
    files = (
      "scripts/evaluate_go2_routes.py",
      "scripts/evaluate_go2_matched_routes.py",
      "scripts/evaluate_go2_terrain_curves.py",
      "scripts/evaluate_go2_terrain_boundary.py",
      "scripts/evaluate_go2_high_slope_matched.py",
    )
    expected = {
      "base_pitch", "actuator_effort_abs", "mechanical_power_abs",
      "normalized_action_abs_max", "action_safety_fault",
    }
    found = 0
    for relative in files:
      tree = ast.parse((WORKSPACE / relative).read_text())
      for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
          continue
        if node.func.attr != "update":
          continue
        names = {keyword.arg for keyword in node.keywords}
        if "catastrophic_termination" not in names:
          continue
        found += 1
        self.assertTrue(expected <= names, relative)
    self.assertEqual(found, 5)


class ProprioSelectorContractTest(unittest.TestCase):
  def test_lineage_builder_supports_monolithic_run(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run = root / "run"
      excluded = root / "excluded"
      run.mkdir()
      excluded.mkdir()
      source = root / "source.json"
      source.write_text('{"source":"frozen"}\n')
      source_sha = acceptance.sha256_file(source)
      safe_schema = __import__(
        "src.tasks.velocity.config.go2.sim2real_safe_action_schema",
        fromlist=["schema_sha256"],
      )
      for iteration in acceptance.EXPECTED_PPO_ITERATIONS:
        torch.save({
          "iter": iteration,
          "actor_state_dict": {"weight": torch.zeros(1)},
          "infos": {
            "proprioceptive_stage": "ppo",
            "student_schema_sha256": safe_schema.schema_sha256(),
            "source_manifest_sha256": source_sha,
            "action_interface": acceptance.V2_SAFE_ACTION_INTERFACE,
            "action_mean_bound": safe_schema.ACTION_MEAN_BOUND,
          },
        }, run / f"model_{iteration}.pt")
      output = root / "lineage.json"
      arguments = (
        "build_go2_proprioceptive_checkpoint_lineage.py",
        "--run", str(run),
        "--source-manifest", str(source),
        "--exclude-run", str(excluded),
        "--output-file", str(output),
      )
      with mock.patch.object(sys, "argv", arguments), mock.patch("builtins.print"):
        lineage_builder.main()
      lineage = acceptance.load_checkpoint_lineage(output)
      self.assertEqual(lineage.payload["execution_mode"], "monolithic")
      self.assertEqual(Path(lineage.payload["run"]["path"]), run.resolve())
      self.assertNotIn("resume_anchor", lineage.payload)
      self.assertTrue(all(path.parent == run.resolve() for path in lineage.checkpoints))

  def test_lineage_builder_supports_exact_250_251_split(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      original = root / "original"
      resumed = root / "resumed"
      excluded = root / "excluded"
      for path in (original, resumed, excluded):
        path.mkdir()
      source = root / "source.json"
      source.write_text('{"source":"frozen"}\n')
      source_sha = acceptance.sha256_file(source)
      safe_schema = __import__(
        "src.tasks.velocity.config.go2.sim2real_safe_action_schema",
        fromlist=["schema_sha256"],
      )

      def payload(iteration: int) -> dict:
        return {
          "iter": iteration,
          "actor_state_dict": {"weight": torch.zeros(1)},
          "infos": {
            "proprioceptive_stage": "ppo",
            "student_schema_sha256": safe_schema.schema_sha256(),
            "source_manifest_sha256": source_sha,
            "action_interface": acceptance.V2_SAFE_ACTION_INTERFACE,
            "action_mean_bound": safe_schema.ACTION_MEAN_BOUND,
          },
        }

      for iteration in acceptance.EXPECTED_PPO_ITERATIONS:
        target = original if iteration <= 250 else resumed
        torch.save(payload(iteration), target / f"model_{iteration}.pt")
      anchor = root / "resume_anchor_iter_251_from_model_250.pt"
      torch.save(payload(251), anchor)
      output = root / "lineage.json"
      arguments = (
        "build_go2_proprioceptive_checkpoint_lineage.py",
        "--original-run", str(original),
        "--resume-run", str(resumed),
        "--resume-anchor", str(anchor),
        "--source-manifest", str(source),
        "--split-iteration", "250",
        "--exclude-run", str(excluded),
        "--output-file", str(output),
      )
      with mock.patch.object(sys, "argv", arguments), mock.patch("builtins.print"):
        lineage_builder.main()
      lineage = acceptance.load_checkpoint_lineage(output)
      self.assertEqual(
        lineage.payload["resume_anchor"]["semantic_change"]["iter"], [250, 251]
      )
      roots = {
        int(entry["iteration"]): Path(entry["path"]).parent
        for entry in lineage.payload["checkpoints"]
      }
      self.assertEqual(roots[0], original.resolve())
      self.assertEqual(roots[250], original.resolve())
      self.assertTrue(all(
        roots[iteration] == resumed.resolve()
        for iteration in acceptance.EXPECTED_PPO_ITERATIONS
        if iteration >= 500
      ))

  def test_split_run_lineage_is_strict_and_schedule_complete(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source.json"
      source.write_text('{"source":"frozen"}\n')
      source_sha = acceptance.sha256_file(source)
      excluded = root / "excluded"
      excluded.mkdir()
      checkpoints = []
      payload_by_path = {}
      schema = __import__(
        "src.tasks.velocity.config.go2.sim2real_schema",
        fromlist=["schema_sha256"],
      ).schema_sha256()
      for iteration in acceptance.EXPECTED_PPO_ITERATIONS:
        path = root / f"model_{iteration}.pt"
        path.write_bytes(f"checkpoint:{iteration}".encode())
        checkpoints.append({
          "iteration": iteration,
          "path": str(path.resolve()),
          "sha256": acceptance.sha256_file(path),
        })
        payload_by_path[path.resolve()] = {
          "iter": iteration,
          "infos": {
            "proprioceptive_stage": "ppo",
            "student_schema_sha256": schema,
            "source_manifest_sha256": source_sha,
          },
        }
      anchor = root / "anchor.pt"
      anchor.write_bytes(b"anchor")
      parent = root / "model_250.pt"
      payload_by_path[anchor.resolve()] = deepcopy(payload_by_path[parent.resolve()])
      payload_by_path[anchor.resolve()]["iter"] = 251
      manifest = root / "lineage.json"
      manifest.write_text(json.dumps({
        "schema_version": 1,
        "evaluation_suite": "go2_proprioceptive_checkpoint_lineage",
        "source_manifest": {
          "path": str(source.resolve()), "sha256": source_sha,
        },
        "resume_anchor": {
          "path": {
            "path": str(anchor.resolve()),
            "sha256": acceptance.sha256_file(anchor),
          },
          "derived_from": {
            "path": str(parent.resolve()),
            "sha256": acceptance.sha256_file(parent),
          },
          "semantic_change": {"iter": [250, 251]},
        },
        "excluded_technical_runs": [{
          "path": str(excluded.resolve()), "reason": "duplicate iteration",
        }],
        "checkpoints": checkpoints,
      }))
      with mock.patch.object(
        acceptance.torch, "load",
        side_effect=lambda path, **_: payload_by_path[Path(path).resolve()],
      ):
        lineage = acceptance.load_checkpoint_lineage(manifest)
        self.assertEqual(
          tuple(acceptance.checkpoint_iteration(path) for path in lineage.checkpoints),
          acceptance.EXPECTED_PPO_ITERATIONS,
        )
        broken = json.loads(manifest.read_text())
        broken["checkpoints"][-1]["sha256"] = "0" * 64
        manifest.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
          acceptance.load_checkpoint_lineage(manifest)

  def test_checkpoint_task_inference_is_v2_aware_and_fail_closed(self) -> None:
    v1_schema = __import__(
      "src.tasks.velocity.config.go2.sim2real_schema",
      fromlist=["schema_sha256"],
    )
    v2_schema = __import__(
      "src.tasks.velocity.config.go2.sim2real_safe_action_schema",
      fromlist=["schema_sha256"],
    )
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      v1 = root / "model_1.pt"
      v2 = root / "model_2.pt"
      unknown = root / "model_3.pt"
      torch.save({"infos": {
        "proprioceptive_stage": "ppo",
        "student_schema_sha256": v1_schema.schema_sha256(),
      }}, v1)
      torch.save({"infos": {
        "proprioceptive_stage": "ppo",
        "student_schema_sha256": v2_schema.schema_sha256(),
        "action_interface": acceptance.V2_SAFE_ACTION_INTERFACE,
        "action_mean_bound": v2_schema.ACTION_MEAN_BOUND,
      }}, v2)
      torch.save({"infos": {
        "proprioceptive_stage": "ppo",
        "student_schema_sha256": v2_schema.schema_sha256(),
        "action_interface": None,
      }}, unknown)
      self.assertEqual(
        acceptance.checkpoint_task_id(v1), acceptance.V1_STUDENT_TASK
      )
      self.assertEqual(
        acceptance.checkpoint_task_id(v2), acceptance.V2_SAFE_ACTION_TASK
      )
      with self.assertRaisesRegex(ValueError, "unsupported checkpoint"):
        acceptance.checkpoint_task_id(unknown)

  def test_formal_checkpoint_schedule_rejects_missing_final(self) -> None:
    paths = [f"model_{iteration}.pt" for iteration in acceptance.EXPECTED_PPO_ITERATIONS]
    acceptance.validate_formal_checkpoint_schedule(paths)
    with self.assertRaisesRegex(ValueError, "schedule mismatch"):
      acceptance.validate_formal_checkpoint_schedule(paths[:-1])

  def setUp(self) -> None:
    self.temp = tempfile.TemporaryDirectory()
    root = Path(self.temp.name)
    self.manifest = root / "proprioceptive_source_manifest.json"
    self.evaluator = root / "evaluator.py"
    self.dependency = root / "dependency.py"
    self.manifest.write_text('{"source":"frozen"}\n')
    self.evaluator.write_text("# evaluator\n")
    self.dependency.write_text("# dependency\n")

  def tearDown(self) -> None:
    self.temp.cleanup()

  def _bundle(self, iteration: int, randomized_complex: float = 0.80):
    checkpoint = Path(self.temp.name) / f"model_{iteration}.pt"
    torch.save({
      "infos": {"source_manifest_sha256": acceptance.sha256_file(self.manifest)}
    }, checkpoint)
    provenance = acceptance.formal_evaluation_provenance(
      checkpoint, self.evaluator, (self.dependency,),
      workspace=WORKSPACE, source_manifest=self.manifest,
    )
    metric = {name: 0.5 for name in acceptance.SAFETY_METRICS}
    rows = [
      ("clean", "retained", "flat", "line", 0.90),
      ("randomized", "retained", "rough", "arc", 0.80),
      ("clean", "complex", "slope", "s_curve", 0.80),
      ("randomized", "complex", "stairs", "line", randomized_complex),
    ]
    groups = []
    for profile, category, scene, route, completion in rows:
      identity = f"{profile}:{category}:{scene}:{route}"
      groups.append({
        "profile": profile,
        "category": category,
        "scene": scene,
        "route_kind": route,
        "completion": completion,
        "moving_forward": True,
        "forward_gain": 0.85,
        "command_tracking_error": 0.10,
        "metrics": deepcopy(metric),
        "v7_reference": deepcopy(metric),
        "metric_availability": {
          name: {
            "available": True,
            "available_scenarios": 1,
            "expected_scenarios": 1,
          }
          for name in acceptance.SAFETY_METRICS
        },
        "v7_metric_availability": {
          name: {
            "available": True,
            "available_scenarios": 1,
            "expected_scenarios": 1,
          }
          for name in acceptance.SAFETY_METRICS
        },
        "scene_identity": identity,
        "matched_reference_identity": identity,
      })
    return {
      "evaluation_suite": "go2_proprioceptive_checkpoint_acceptance",
      "checkpoint": {
        "path": str(checkpoint.resolve()),
        "sha256": acceptance.sha256_file(checkpoint),
      },
      "provenance": provenance,
      "contract": {
        "provenance_valid": True,
        "lifecycle_valid": True,
        "recursive_finite": True,
        "schema_425_valid": True,
        "onnx_single_input": True,
        "onnx_no_privileged_inputs": True,
        "onnx_action_contract_valid": True,
        "no_reset_storm": True,
        "placement_valid": True,
        "action_limits_valid": True,
        "screening_action_fault_count": 0,
        "profile_contract_valid": True,
        "same_scene_reference_valid": True,
        "unified_metrics_valid": True,
        "coverage_complete": True,
        "onnx_max_abs_error": 1.0e-6,
      },
      "groups": groups,
    }

  def test_hard_gates_run_before_randomized_complex_ranking(self) -> None:
    earlier = self._bundle(100, randomized_complex=0.75)
    later = self._bundle(200, randomized_complex=0.85)
    selected, decisions = acceptance.select_checkpoint_bundles([earlier, later])
    self.assertTrue(all(item.passed for item in decisions))
    self.assertIsNotNone(selected)
    self.assertEqual(selected.iteration, 200)

    unsafe = deepcopy(later)
    unsafe["groups"][0]["metrics"]["base_pitch_absolute"] = 0.61
    decision = acceptance.evaluate_checkpoint_bundle(unsafe)
    self.assertFalse(decision.passed)
    self.assertTrue(any("safety:base_pitch_absolute" in x for x in decision.violations))
    self.assertIsNone(decision.lexicographic_key)

  def test_ranking_uses_worst_forward_gain_after_mean_gain_gate(self) -> None:
    balanced = self._bundle(100, randomized_complex=0.80)
    uneven = self._bundle(200, randomized_complex=0.80)
    balanced["groups"][0]["forward_gain"] = 0.80
    balanced["groups"][1]["forward_gain"] = 0.80
    uneven["groups"][0]["forward_gain"] = 0.75
    uneven["groups"][1]["forward_gain"] = 0.90
    selected, decisions = acceptance.select_checkpoint_bundles(
      [balanced, uneven]
    )
    self.assertTrue(all(item.passed for item in decisions))
    self.assertIsNotNone(selected)
    self.assertEqual(selected.iteration, 100)

  def test_body_contacts_rank_only_after_primary_safety_ties(self) -> None:
    safer_primary = self._bundle(100)
    safer_body = self._bundle(200)
    for row in safer_primary["groups"]:
      row["metrics"]["failure_risk"] = 0.10
      row["metrics"]["base_contact"] = 0.59
    for row in safer_body["groups"]:
      row["metrics"]["failure_risk"] = 0.20
      row["metrics"]["base_contact"] = 0.10
    selected, decisions = acceptance.select_checkpoint_bundles(
      [safer_primary, safer_body]
    )
    self.assertTrue(all(item.passed for item in decisions))
    self.assertIsNotNone(selected)
    self.assertEqual(selected.iteration, 100)

    for row in safer_body["groups"]:
      row["metrics"]["failure_risk"] = 0.10
    selected, decisions = acceptance.select_checkpoint_bundles(
      [safer_primary, safer_body]
    )
    self.assertTrue(all(item.passed for item in decisions))
    self.assertIsNotNone(selected)
    self.assertEqual(selected.iteration, 200)

  def test_parity_matched_identity_and_provenance_are_hard_gates(self) -> None:
    bundle = self._bundle(300)
    bundle["contract"]["onnx_max_abs_error"] = 2.0e-5
    bundle["groups"][0]["matched_reference_identity"] = "different"
    bundle["provenance"]["source_manifest"]["sha256"] = "0" * 64
    decision = acceptance.evaluate_checkpoint_bundle(bundle)
    self.assertFalse(decision.passed)
    self.assertTrue(any(x.startswith("onnx_parity") for x in decision.violations))
    self.assertTrue(any(x.startswith("matched_reference") for x in decision.violations))
    self.assertIn("provenance:source_manifest_sha256", decision.violations)

  def test_nonzero_screening_action_fault_is_fail_closed(self) -> None:
    bundle = self._bundle(100)
    bundle["contract"]["screening_action_fault_count"] = 2
    bundle["contract"]["action_limits_valid"] = False
    decision = acceptance.evaluate_checkpoint_bundle(bundle)
    self.assertFalse(decision.passed)
    self.assertIn("contract:action_limits_valid", decision.violations)
    self.assertIn("action_limits:screening_fault_count:2", decision.violations)

    inconsistent = self._bundle(101)
    inconsistent["contract"]["screening_action_fault_count"] = 2
    inconsistent_decision = acceptance.evaluate_checkpoint_bundle(inconsistent)
    self.assertIn(
      "contract:action_limit_screening_consistency", inconsistent_decision.violations
    )

  def test_missing_loaded_stance_slip_is_a_hard_rejection_not_zero(self) -> None:
    bundle = self._bundle(100)
    row = bundle["groups"][0]
    del row["metrics"]["terrain_tangent_slip"]
    row["metric_availability"]["terrain_tangent_slip"] = {
      "available": False,
      "available_scenarios": 0,
      "expected_scenarios": 1,
    }
    bundle["contract"]["unified_metrics_valid"] = False

    decision = acceptance.evaluate_checkpoint_bundle(bundle)

    self.assertFalse(decision.passed)
    self.assertIsNone(decision.lexicographic_key)
    self.assertIn("contract:unified_metrics_valid", decision.violations)
    self.assertTrue(any(
      item.startswith("unified_metric:candidate_unavailable:terrain_tangent_slip")
      for item in decision.violations
    ))


class ProprioOrchestratorContractTest(unittest.TestCase):
  def test_cpu_screening_exports_static_actor_and_checks_parity(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run = root / "run"
      run.mkdir()
      manifest = run / screener.MANIFEST_NAME
      manifest.write_text('{"source":"frozen"}\n')
      model = screener.FrozenActor({
        "obs_normalizer._mean": torch.zeros((1, 425)),
        "obs_normalizer._std": torch.ones((1, 425)),
        "mlp.0.weight": torch.zeros((512, 425)),
        "mlp.0.bias": torch.zeros(512),
        "mlp.2.weight": torch.zeros((256, 512)),
        "mlp.2.bias": torch.zeros(256),
        "mlp.4.weight": torch.zeros((128, 256)),
        "mlp.4.bias": torch.zeros(128),
        "mlp.6.weight": torch.zeros((12, 128)),
        "mlp.6.bias": torch.zeros(12),
      })
      actor_state = {
        "obs_normalizer._mean": model.mean.clone(),
        "obs_normalizer._std": model.std.clone(),
        **{f"mlp.{key}": value for key, value in model.mlp.state_dict().items()},
      }
      checkpoint = run / "model_0.pt"
      safe_schema = __import__(
        "src.tasks.velocity.config.go2.sim2real_safe_action_schema",
        fromlist=["schema_sha256"],
      )
      torch.save({
        "actor_state_dict": actor_state,
        "critic_state_dict": {"value": torch.zeros(1)},
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": {
          "proprioceptive_stage": "ppo",
          "student_schema_sha256": safe_schema.schema_sha256(),
          "source_manifest_sha256": acceptance.sha256_file(manifest),
          "action_interface": acceptance.V2_SAFE_ACTION_INTERFACE,
          "action_mean_bound": safe_schema.ACTION_MEAN_BOUND,
        },
      }, checkpoint)
      result = screener.screen_checkpoint(checkpoint, root / "screening")
      self.assertTrue(result["onnx_single_input"])
      self.assertTrue(result["onnx_no_privileged_inputs"])
      self.assertLessEqual(result["onnx_max_abs_error"], 1.0e-5)
      self.assertEqual(result["screening_action_fault_count"], 0)
      self.assertTrue(result["action_limits_valid"])
      self.assertTrue(result["onnx_action_contract_valid"])
      self.assertEqual(
        result["policy_contract"]["task_id"], acceptance.V2_SAFE_ACTION_TASK
      )

  def test_matrix_discovers_every_checkpoint_and_is_serially_explicit(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / "run"
      output = Path(directory) / "acceptance"
      run.mkdir()
      (run / "model_10.pt").write_bytes(b"10")
      (run / "model_2.pt").write_bytes(b"2")
      with mock.patch.object(
        orchestrator, "checkpoint_task_id", return_value=acceptance.V2_SAFE_ACTION_TASK
      ):
        matrix = orchestrator.build_matrix(run, output)
    candidate_paths = {str((run / name).resolve()) for name in ("model_2.pt", "model_10.pt")}
    self.assertTrue(candidate_paths <= {item.checkpoint_path for item in matrix})
    self.assertEqual(len(matrix), 42)
    self.assertTrue(all(item.command[0] == sys.executable for item in matrix))
    self.assertTrue(all("randomized" in item.canonical_profiles or item.canonical_profiles == ("clean",) for item in matrix))
    high_slope = [item for item in matrix if item.artifact_id == "high_slope_matched"]
    self.assertTrue(high_slope)
    self.assertTrue(all(
      item.command[item.command.index("--radii") + 1] == "2.5"
      for item in high_slope
    ))
    boundary = [
      item for item in matrix
      if Path(item.command[1]).name == "evaluate_go2_terrain_boundary.py"
    ]
    self.assertTrue(boundary)
    self.assertTrue(all(
      item.command[item.command.index("--route-kind") + 1] == "straight"
      for item in boundary
    ))

    by_checkpoint = {}
    for item in matrix:
      by_checkpoint.setdefault(item.checkpoint_path, []).append(item)
    self.assertTrue(all(len(items) == 14 for items in by_checkpoint.values()))
    candidate_items = by_checkpoint[next(iter(candidate_paths))]
    with self.assertRaisesRegex(ValueError, "inventory mismatch"):
      orchestrator.validate_invocation_inventory(candidate_items[:-1])
    extra = deepcopy(candidate_items)
    extra[-1] = orchestrator.Invocation(
      **{**extra[-1].__dict__, "artifact_id": "unexpected"}
    )
    with self.assertRaisesRegex(ValueError, "inventory mismatch"):
      orchestrator.validate_invocation_inventory(extra)
    with self.assertRaisesRegex(ValueError, "duplicate"):
      orchestrator.validate_invocation_inventory(
        [*candidate_items[:-1], candidate_items[0]]
      )
    substituted = deepcopy(candidate_items)
    substituted[0] = orchestrator.Invocation(
      **{**substituted[0].__dict__, "command": ("python", "replacement.py")}
    )
    with mock.patch.object(
      orchestrator, "checkpoint_task_id", return_value=acceptance.V2_SAFE_ACTION_TASK
    ), mock.patch.object(
      orchestrator, "sha256_file", return_value=substituted[0].checkpoint_sha256
    ), self.assertRaisesRegex(ValueError, "command mismatch"):
      orchestrator.validate_invocation_inventory(substituted)

  def test_bundle_assembly_rejects_incomplete_inventory(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate = root / "run" / "model_2.pt"
      candidate.parent.mkdir()
      manifest = candidate.parent / "proprioceptive_source_manifest.json"
      manifest.write_text(
        '{"source":"frozen"}\n'
      )
      torch.save({
        "infos": {"source_manifest_sha256": acceptance.sha256_file(manifest)}
      }, candidate)
      screening_dir = root / "screening"
      screening_dir.mkdir()
      screening = {
        "lifecycle_valid": True, "schema_425_valid": True,
        "onnx_single_input": True, "onnx_no_privileged_inputs": True,
        "no_reset_storm": True, "placement_valid": True,
        "action_limits_valid": True, "screening_action_fault_count": 0,
        "onnx_max_abs_error": 1.0e-6,
      }
      (screening_dir / "model_2.screening.json").write_text(
        json.dumps(screening)
      )

      def scenario(slot: int) -> dict:
        distribution = {"mean": 0.1, "p95": 0.1, "max": 0.1}
        return {
          "matched_slot": slot, "completed": True, "failed": False,
          "reset_count": 0,
          "catastrophic_termination": False,
          "commanded_velocity_mean": [0.4, 0.0, 0.0],
          "actual_velocity_mean": [0.34, 0.0, 0.0],
          "terrain_tangent_stance_slip_mean": 0.1,
          "terrain_rollout_metrics": {
            "active_control_step_samples": 100,
            "foot_slip_velocity": distribution,
            "base_pitch_absolute": distribution,
            "action_acceleration": distribution,
            "actuator_effort_abs": distribution,
            "mechanical_power_abs": distribution,
            "mechanical_energy_abs": 2.0,
            "action_safety": {
              "available": True,
              "fault_occurred": False,
              "fault_control_step_count": 0,
              "normalized_action_abs_max": 1.0,
            },
            "body_contacts": {
              name: {"non_terminating_rate": 0.0}
              for name in ("base", "upper_leg", "calf")
            },
          },
        }

      def raw(checkpoint: Path, sha: str, artifact: str) -> dict:
        profiles = {}
        for profile in ("clean", "randomized"):
          settings = {
            "actor_observation_corruption": profile == "randomized",
            "startup_randomization_events": (
              list(acceptance.SIM2REAL_RANDOMIZATION_EVENTS)
              if profile == "randomized" else []
            ),
            "push_enabled": profile == "randomized",
          }
          profiles[profile] = {
            "profile_settings": settings,
            "route_results": {
              route: {"scenarios": [scenario(index)]}
              for index, route in enumerate(("straight", "arc", "s_curve"))
            },
          }
        return {
          "profiles": profiles,
          "provenance": {"checkpoint": {"path": str(checkpoint), "sha256": sha}},
          "artifact": artifact,
        }

      invocations = []
      for checkpoint, sha in (
        (orchestrator.V7_CHECKPOINT.resolve(), orchestrator.V7_SHA256),
        (candidate.resolve(), acceptance.sha256_file(candidate)),
      ):
        for artifact in ("high_slope_matched", "flat_matched_routes"):
          output = root / f"{checkpoint.stem}_{artifact}.json"
          output.write_text(json.dumps(raw(checkpoint, sha, artifact)))
          invocations.append(orchestrator.Invocation(
            checkpoint_label=checkpoint.stem,
            checkpoint_path=str(checkpoint), checkpoint_sha256=sha,
            artifact_id=artifact,
            canonical_profiles=("clean", "randomized"), command=(),
            output_file=str(output),
          ))
      with self.assertRaisesRegex(ValueError, "inventory mismatch"):
        orchestrator.assemble_bundles(
          invocations, screening_dir, root / "bundles"
        )


if __name__ == "__main__":
  unittest.main()
