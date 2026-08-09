from __future__ import annotations

import hashlib
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.config.go2.sim2real_schema import schema_sha256
from src.tasks.velocity.rl import runner as runner_module


ORCHESTRATOR_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/train_go2_proprioceptive.py"
)
_SPEC = importlib.util.spec_from_file_location(
  "go2_proprioceptive_training_orchestrator", ORCHESTRATOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(orchestrator)


class TestSourceManifest(unittest.TestCase):
  def test_manifest_is_deterministic_and_covers_training_sources(self) -> None:
    first = orchestrator._build_source_manifest()
    second = orchestrator._build_source_manifest()
    self.assertEqual(first, second)
    self.assertEqual(first["git"]["head"], orchestrator._git("rev-parse", "HEAD"))
    for relative in (
      "scripts/train_go2_proprioceptive.py",
      "src/tasks/velocity/config/go2/sim2real_schema.py",
      "src/tasks/velocity/rl/teacher_rollout_distillation.py",
    ):
      self.assertIn(relative, first["files"])
      entry = first["files"][relative]
      self.assertEqual(len(entry["sha256"]), 64)
      self.assertIn(entry["git_state"], ("tracked", "untracked"))
      if entry["git_state"] == "untracked":
        self.assertEqual(
          hashlib.sha256(base64.b64decode(entry["content_base64"])).hexdigest(),
          entry["sha256"],
        )
      else:
        self.assertNotIn("content_base64", entry)
    self.assertTrue(first["packages"]["rsl_rl"])

  def test_generate_and_validate_canonical_manifest(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "source_manifest.json"
      generated, expected_sha = orchestrator._generate_source_manifest(path)
      self.assertEqual(generated, path.resolve())
      self.assertEqual(orchestrator._sha256(path), expected_sha)
      self.assertEqual(orchestrator._validate_source_manifest(path), expected_sha)

  def test_manifest_rejects_source_drift_and_noncanonical_json(self) -> None:
    payload = orchestrator._build_source_manifest()
    with tempfile.TemporaryDirectory() as directory:
      drifted = Path(directory) / "drifted.json"
      changed = json.loads(json.dumps(payload))
      changed["files"]["scripts/train.py"]["sha256"] = "0" * 64
      drifted.write_bytes(orchestrator._source_manifest_bytes(changed))
      with self.assertRaisesRegex(RuntimeError, "does not match"):
        orchestrator._validate_source_manifest(drifted)

      noncanonical = Path(directory) / "noncanonical.json"
      noncanonical.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
      with self.assertRaisesRegex(RuntimeError, "not canonical"):
        orchestrator._validate_source_manifest(noncanonical)

  def test_existing_different_manifest_is_never_overwritten(self) -> None:
    payload = orchestrator._build_source_manifest()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "source_manifest.json"
      path.write_text("{}\n", encoding="utf-8")
      with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
        orchestrator._write_source_manifest(path, payload)
      self.assertEqual(path.read_text(encoding="utf-8"), "{}\n")

  def test_manifest_is_installed_verbatim_in_each_run(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source.json"
      source.write_bytes(b'{"source":true}\n')
      run = root / "run"
      run.mkdir()
      installed = orchestrator._install_source_manifest(source, run)
      self.assertEqual(installed.read_bytes(), source.read_bytes())
      self.assertEqual(orchestrator._install_source_manifest(source, run), installed)
      source.write_bytes(b'{"source":false}\n')
      with self.assertRaisesRegex(RuntimeError, "different source manifest"):
        orchestrator._install_source_manifest(source, run)


class TestExclusiveTrainingLock(unittest.TestCase):
  def test_second_process_fails_immediately_while_lock_is_held(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      lock_path = Path(directory) / "training.lock"
      code = "\n".join(
        (
          "import importlib.util",
          "from pathlib import Path",
          f"spec = importlib.util.spec_from_file_location('lock_test', {str(ORCHESTRATOR_PATH)!r})",
          "module = importlib.util.module_from_spec(spec)",
          "spec.loader.exec_module(module)",
          "try:",
          f"  with module._exclusive_training_lock(Path({str(lock_path)!r})):",
          "    raise SystemExit(2)",
          "except RuntimeError as exc:",
          "  print(exc)",
        )
      )
      with orchestrator._exclusive_training_lock(lock_path):
        result = subprocess.run(
          [sys.executable, "-c", code],
          cwd=orchestrator.ROOT,
          capture_output=True,
          text=True,
          timeout=30,
          check=False,
        )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("refusing duplicate launch", result.stdout)

      with orchestrator._exclusive_training_lock(lock_path):
        pass

  def test_only_registered_zero_checkpoint_failure_is_retryable(self) -> None:
    run = orchestrator.LOG_ROOT / next(iter(orchestrator.REGISTERED_TECHNICAL_FAILURES))
    self.assertTrue(orchestrator._is_registered_prelearning_failure(run))
    with tempfile.TemporaryDirectory() as directory:
      unregistered = Path(directory) / "unregistered_run"
      unregistered.mkdir()
      self.assertFalse(orchestrator._is_registered_prelearning_failure(unregistered))


class TestCheckpointProvenance(unittest.TestCase):
  def _write_manifest(self, directory: str) -> tuple[Path, str]:
    path = Path(directory) / "source_manifest.json"
    path.write_bytes(b'{"test":true}\n')
    return path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()

  def test_runner_validates_active_manifest_and_checkpoint_identity(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path, manifest_sha = self._write_manifest(directory)
      environment = {
        runner_module.SOURCE_MANIFEST_PATH_ENV: str(path),
        runner_module.SOURCE_MANIFEST_SHA256_ENV: manifest_sha,
      }
      with patch.dict(os.environ, environment, clear=False):
        self.assertEqual(
          runner_module._active_source_manifest(required=True),
          (str(path), manifest_sha),
        )
        runner_module._validate_checkpoint_source_manifest(
          {
            "source_manifest": str(path),
            "source_manifest_sha256": manifest_sha,
          },
          required=True,
        )
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
          runner_module._validate_checkpoint_source_manifest(
            {"source_manifest_sha256": "f" * 64}, required=True
          )

      bad_environment = {
        runner_module.SOURCE_MANIFEST_PATH_ENV: str(path),
        runner_module.SOURCE_MANIFEST_SHA256_ENV: "0" * 64,
      }
      with patch.dict(os.environ, bad_environment, clear=False):
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
          runner_module._active_source_manifest(required=True)

  def test_checkpoint_validators_require_exact_source_manifest_sha(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      distill = Path(directory) / "model_299.pt"
      ppo = Path(directory) / "model_3999.pt"
      manifest_sha = "1" * 64
      torch.save(
        {
          "student_state_dict": {},
          "teacher_state_dict": {},
          "iter": 299,
          "infos": {
            "proprioceptive_stage": "distillation",
            "student_schema_sha256": schema_sha256(),
            "teacher_sha256": orchestrator.TEACHER_SHA256,
            "source_manifest_sha256": manifest_sha,
          },
        },
        distill,
      )
      torch.save(
        {
          "actor_state_dict": {},
          "critic_state_dict": {},
          "iter": 3999,
          "infos": {
            "proprioceptive_stage": "ppo",
            "student_schema_sha256": schema_sha256(),
            "source_manifest_sha256": manifest_sha,
          },
        },
        ppo,
      )
      orchestrator._validate_distillation_checkpoint(distill, manifest_sha)
      orchestrator._validate_ppo_checkpoint(ppo, manifest_sha)
      with self.assertRaisesRegex(RuntimeError, "source_manifest_sha256"):
        orchestrator._validate_distillation_checkpoint(distill, "2" * 64)
      with self.assertRaisesRegex(RuntimeError, "source_manifest_sha256"):
        orchestrator._validate_ppo_checkpoint(ppo, "2" * 64)


if __name__ == "__main__":
  unittest.main()
