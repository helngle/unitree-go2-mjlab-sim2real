from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx.reference import ReferenceEvaluator
import numpy as np
import torch

from scripts import screen_go2_proprioceptive_checkpoints as screener
from src.tasks.velocity.config.go2 import sim2real_safe_action_schema as v2_schema


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _actor_state(output_bias: float) -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.zeros((1, screener.actor_dim())),
    "obs_normalizer._std": torch.ones((1, screener.actor_dim())),
    "mlp.0.weight": torch.zeros((512, screener.actor_dim())),
    "mlp.0.bias": torch.zeros(512),
    "mlp.2.weight": torch.zeros((256, 512)),
    "mlp.2.bias": torch.zeros(256),
    "mlp.4.weight": torch.zeros((128, 256)),
    "mlp.4.bias": torch.zeros(128),
    "mlp.6.weight": torch.zeros((12, 128)),
    "mlp.6.bias": torch.full((12,), output_bias),
    "distribution.std_param": torch.ones(12),
  }


def _checkpoint(
  root: Path,
  *,
  schema_sha256: str,
  action_interface: str | None,
  output_bias: float,
) -> Path:
  manifest = root / screener.MANIFEST_NAME
  manifest.write_text('{"source":"test"}\n', encoding="utf-8")
  infos: dict[str, object] = {
    "proprioceptive_stage": "ppo",
    "student_schema_sha256": schema_sha256,
    "source_manifest_sha256": _sha256(manifest),
  }
  if action_interface is not None:
    infos["action_interface"] = action_interface
    infos["action_mean_bound"] = v2_schema.ACTION_MEAN_BOUND
  path = root / "model_1.pt"
  torch.save(
    {
      "actor_state_dict": _actor_state(output_bias),
      "critic_state_dict": {"finite": torch.zeros(1)},
      "optimizer_state_dict": {},
      "iter": 1,
      "infos": infos,
    },
    path,
  )
  return path


class SafeActionScreeningTest(unittest.TestCase):
  def test_v2_exports_applied_action_and_action_interface_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = _checkpoint(
        root,
        schema_sha256=v2_schema.schema_sha256(),
        action_interface=v2_schema.ACTION_INTERFACE,
        output_bias=100.0,
      )
      result = screener.screen_checkpoint(checkpoint, root / "screening")
      contract = result["policy_contract"]
      self.assertEqual(contract["schema_sha256"], v2_schema.schema_sha256())
      self.assertEqual(contract["action_interface"], v2_schema.ACTION_INTERFACE)
      self.assertEqual(
        contract["action_output_semantics"], v2_schema.ACTION_OUTPUT_SEMANTICS
      )
      self.assertTrue(result["action_limits_valid"])
      self.assertEqual(result["screening_action_fault_count"], 0)
      graph = onnx.load(result["screening_onnx"]["path"])
      metadata = {item.key: item.value for item in graph.metadata_props}
      self.assertEqual(metadata["action_interface"], v2_schema.ACTION_INTERFACE)
      self.assertEqual(
        metadata["action_mean_bound"], str(v2_schema.ACTION_MEAN_BOUND)
      )
      self.assertEqual(
        metadata["observation_schema_sha256"], v2_schema.schema_sha256()
      )
      self.assertEqual(
        metadata["action_output_semantics"], v2_schema.ACTION_OUTPUT_SEMANTICS
      )
      output = ReferenceEvaluator(graph).run(
        None, {"actor": np.zeros((1, screener.actor_dim()), dtype=np.float32)}
      )[0][0]
      expected_scale = np.tanh(v2_schema.ACTION_MEAN_BOUND)
      np.testing.assert_allclose(
        output,
        expected_scale * np.asarray(v2_schema.ACTION_HIGH, dtype=np.float32),
        atol=1.0e-6,
      )

  def test_v1_regression_keeps_unbounded_output(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = _checkpoint(
        root,
        schema_sha256=screener.schema_sha256(),
        action_interface=None,
        output_bias=100.0,
      )
      result = screener.screen_checkpoint(checkpoint, root / "screening")
      self.assertIsNone(result["policy_contract"]["action_interface"])
      self.assertFalse(result["action_limits_valid"])
      self.assertGreater(result["screening_action_fault_count"], 0)
      graph = onnx.load(result["screening_onnx"]["path"])
      metadata = {item.key: item.value for item in graph.metadata_props}
      self.assertNotIn("action_interface", metadata)
      output = ReferenceEvaluator(graph).run(
        None, {"actor": np.zeros((1, screener.actor_dim()), dtype=np.float32)}
      )[0]
      np.testing.assert_allclose(output, np.full((1, 12), 100.0, dtype=np.float32))

  def test_v2_interface_with_v1_schema_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = _checkpoint(
        root,
        schema_sha256=screener.schema_sha256(),
        action_interface=v2_schema.ACTION_INTERFACE,
        output_bias=0.0,
      )
      with self.assertRaisesRegex(ValueError, "schema SHA256"):
        screener.screen_checkpoint(checkpoint, root / "screening")


if __name__ == "__main__":
  unittest.main()
