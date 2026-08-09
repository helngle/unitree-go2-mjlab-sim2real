from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2 import sim2real_schema as v1_schema
from src.tasks.velocity.config.go2 import sim2real_safe_action_schema as v2_schema
from src.tasks.velocity.rl.bounded_action_distribution import (
  AsymmetricBoundedGaussianDistribution,
)
from src.tasks.velocity.rl.runner import (
  SAFE_ACTION_INTERFACE,
  VelocityDistillationRunner,
  VelocityOnPolicyRunner,
  VelocitySafeActionDistillationRunner,
  VelocitySafeActionOnPolicyRunner,
  _SafeActionTelemetry,
  _proprio_metadata,
  _validate_checkpoint_contract,
)


V1_DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill"
V1_PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1"
V2_DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill"
V2_PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction"
V2_PARAMS = Path(
  "deploy/robots/go2/config/policy/velocity/"
  "v2_safe_action_candidate/params"
)


class TestSafeActionTaskContract(unittest.TestCase):
  def test_v2_tasks_use_safe_runners_and_bounded_configs(self) -> None:
    self.assertIs(load_runner_cls(V2_DISTILL_TASK), VelocitySafeActionDistillationRunner)
    self.assertIs(load_runner_cls(V2_PPO_TASK), VelocitySafeActionOnPolicyRunner)
    distill = load_rl_cfg(V2_DISTILL_TASK)
    ppo = load_rl_cfg(V2_PPO_TASK)
    self.assertIn("AsymmetricBoundedGaussianDistribution", distill.student.distribution_cfg["class_name"])
    self.assertIn("AsymmetricBoundedGaussianDistribution", ppo.actor.distribution_cfg["class_name"])

  def test_v1_registration_and_schema_remain_unchanged(self) -> None:
    self.assertIs(load_runner_cls(V1_DISTILL_TASK), VelocityDistillationRunner)
    self.assertIs(load_runner_cls(V1_PPO_TASK), VelocityOnPolicyRunner)
    self.assertEqual(v1_schema.SCHEMA_VERSION, "go2-sim2real-proprio-v1")
    self.assertNotEqual(v1_schema.schema_sha256(), v2_schema.schema_sha256())

  def test_v2_environment_preserves_v1_training_variables(self) -> None:
    v1 = load_env_cfg(V1_PPO_TASK)
    v2 = load_env_cfg(V2_PPO_TASK)
    self.assertEqual(v1.observations, v2.observations)
    self.assertEqual(v1.rewards, v2.rewards)
    self.assertEqual(v1.commands, v2.commands)
    self.assertEqual(v1.events, v2.events)
    self.assertEqual(v1.curriculum, v2.curriculum)


class TestSafeActionSchema(unittest.TestCase):
  def test_bounds_are_exactly_derived_and_targets_stay_in_limits(self) -> None:
    expected_low = tuple(
      max(-4.0, (lower - q0) / v1_schema.ACTION_SCALE)
      for q0, (lower, _upper) in zip(
        v1_schema.DEFAULT_JOINT_POS, v1_schema.JOINT_POS_LIMITS, strict=True
      )
    )
    expected_high = tuple(
      min(4.0, (upper - q0) / v1_schema.ACTION_SCALE)
      for q0, (_lower, upper) in zip(
        v1_schema.DEFAULT_JOINT_POS, v1_schema.JOINT_POS_LIMITS, strict=True
      )
    )
    self.assertEqual(v2_schema.ACTION_LOW, expected_low)
    self.assertEqual(v2_schema.ACTION_HIGH, expected_high)
    for joint_index, (low, high) in enumerate(zip(expected_low, expected_high, strict=True)):
      for action in (low, 0.0, high):
        applied = [0.0] * 12
        applied[joint_index] = action
        sdk_targets = v2_schema.applied_action_to_sdk_targets(applied)
        target = sdk_targets[v1_schema.SDK_JOINT_IDS_MAP[joint_index]]
        lower, upper = v1_schema.JOINT_POS_LIMITS[joint_index]
        self.assertGreaterEqual(target, lower)
        self.assertLessEqual(target, upper)

  def test_payload_freezes_applied_action_semantics(self) -> None:
    payload = v2_schema.schema_payload()
    self.assertEqual(payload["action"]["interface"], SAFE_ACTION_INTERFACE)
    self.assertEqual(payload["action"]["latent"], "z")
    self.assertEqual(payload["action"]["squashed"], "u=tanh(z)")
    self.assertEqual(payload["action"]["mapping"]["nonnegative"], "a_applied=u*a_high")
    self.assertEqual(payload["onnx"]["output_semantics"], "applied_normalized_action")
    self.assertIn("applied normalized action", payload["previous_action"]["timing"])
    self.assertTrue(payload["action"]["no_deployment_only_clipping"])

  def test_deploy_yaml_and_canonical_json_match(self) -> None:
    deploy_yaml = V2_PARAMS / "deploy.yaml"
    v2_schema.validate_deploy_yaml(deploy_yaml)
    yaml_payload = yaml.safe_load(deploy_yaml.read_text(encoding="utf-8"))
    self.assertIsNone(yaml_payload["actions"]["JointPositionAction"]["clip"])
    artifact = json.loads((V2_PARAMS / "observation_schema.json").read_text(encoding="utf-8"))
    self.assertEqual(artifact.pop("schema_sha256"), v2_schema.schema_sha256())
    canonical_payload = json.loads(
      json.dumps(v2_schema.schema_payload(), allow_nan=False)
    )
    self.assertEqual(artifact, canonical_payload)

  def test_deterministic_distribution_output_is_applied_action(self) -> None:
    distribution = AsymmetricBoundedGaussianDistribution(
      12,
      action_low=v2_schema.ACTION_LOW,
      action_high=v2_schema.ACTION_HIGH,
    )
    latent = torch.tensor([[0.0, 1.0, -1.0] + [0.5] * 9])
    eager = distribution.deterministic_output(latent)
    exported = distribution.as_deterministic_output_module()(latent)
    self.assertTrue(torch.equal(eager, exported))
    self.assertFalse(torch.equal(eager, latent))
    self.assertEqual(eager[0, 0].item(), 0.0)
    self.assertLessEqual(eager.abs().max().item(), 4.0)


class TestSafeActionRunnerContract(unittest.TestCase):
  @staticmethod
  def _fake_env():
    return SimpleNamespace(
      observation_manager=SimpleNamespace(group_obs_dim={"actor": (425,)})
    )

  def test_runner_metadata_selects_v1_or_v2_schema(self) -> None:
    v1 = _proprio_metadata(self._fake_env())
    v2 = _proprio_metadata(
      self._fake_env(), VelocitySafeActionOnPolicyRunner.schema_module_path
    )
    self.assertEqual(v1["observation_schema_sha256"], v1_schema.schema_sha256())
    self.assertNotIn("action_interface", v1)
    self.assertEqual(v2["observation_schema_sha256"], v2_schema.schema_sha256())
    self.assertEqual(v2["action_interface"], SAFE_ACTION_INTERFACE)
    self.assertEqual(v2["action_output_semantics"], "applied_normalized_action")

  def test_v2_checkpoint_requires_schema_and_action_interface(self) -> None:
    valid = {
      "proprioceptive_stage": "ppo",
      "student_schema_sha256": v2_schema.schema_sha256(),
      "action_interface": SAFE_ACTION_INTERFACE,
      "action_mean_bound": v2_schema.ACTION_MEAN_BOUND,
    }
    _validate_checkpoint_contract(
      valid,
      schema_module_path=VelocitySafeActionOnPolicyRunner.schema_module_path,
      expected_stage="ppo",
      action_interface=SAFE_ACTION_INTERFACE,
    )
    for changed, message in (
      ({"student_schema_sha256": "0" * 64}, "schema SHA256"),
      ({"action_interface": "legacy"}, "action interface"),
      ({"action_interface": None}, "action interface"),
      ({"action_mean_bound": 4.0}, "action mean bound"),
    ):
      invalid = valid | changed
      with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, message):
        _validate_checkpoint_contract(
          invalid,
          schema_module_path=VelocitySafeActionOnPolicyRunner.schema_module_path,
          expected_stage="ppo",
          action_interface=SAFE_ACTION_INTERFACE,
        )

  def test_v1_checkpoint_contract_does_not_require_action_interface(self) -> None:
    _validate_checkpoint_contract(
      {
        "proprioceptive_stage": "ppo",
        "student_schema_sha256": v1_schema.schema_sha256(),
      },
      schema_module_path=VelocityOnPolicyRunner.schema_module_path,
      expected_stage="ppo",
      action_interface=None,
    )

  def test_action_chain_telemetry_is_complete_and_fail_closed(self) -> None:
    distribution = AsymmetricBoundedGaussianDistribution(
      12,
      action_low=v2_schema.ACTION_LOW,
      action_high=v2_schema.ACTION_HIGH,
    )
    latent = torch.linspace(-1.0, 1.0, 24).reshape(2, 12)
    actions = distribution.transform(latent)
    q0 = torch.tensor(v1_schema.DEFAULT_JOINT_POS)
    targets = q0 + v1_schema.ACTION_SCALE * actions
    limits = torch.tensor(v1_schema.JOINT_POS_LIMITS).expand(2, -1, -1)
    term = SimpleNamespace(
      _processed_actions=targets,
      target_ids=torch.arange(12),
    )

    class Writer:
      def __init__(self) -> None:
        self.scalars = {}
        self.histograms = {}

      def add_scalar(self, name, value, iteration) -> None:
        self.scalars[name] = (value, iteration)

      def add_histogram(self, name, value, iteration) -> None:
        self.histograms[name] = (value, iteration)

    writer = Writer()
    algorithm = SimpleNamespace(
      last_teacher_raw_actions=latent,
      get_policy=lambda: SimpleNamespace(distribution=distribution),
    )
    env = SimpleNamespace(
      unwrapped=SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _name: term),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos_limits=limits))},
      )
    )
    runner = SimpleNamespace(
      alg=algorithm,
      env=env,
      cfg={"num_steps_per_env": 1},
      logger=SimpleNamespace(writer=writer),
    )
    telemetry = _SafeActionTelemetry(runner)
    telemetry.record_action(actions)
    telemetry.record_target()
    telemetry.write(7)
    self.assertEqual(
      set(writer.histograms),
      {
        "ActionTelemetry/latent_raw",
        "ActionTelemetry/u",
        "ActionTelemetry/a_applied",
        "ActionTelemetry/q_target",
        "ActionTelemetry/q_target_limit_margin",
      },
    )
    self.assertEqual(writer.scalars["ActionTelemetry/action_fault_rows"], (0, 7))
    self.assertEqual(
      writer.scalars["ActionTelemetry/joint_target_fault_rows"], (0, 7)
    )
    with self.assertRaisesRegex(RuntimeError, "incomplete"):
      telemetry.write(8)


if __name__ == "__main__":
  unittest.main()
