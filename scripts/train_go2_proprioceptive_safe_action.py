"""Run the frozen Go2 safe-action V2 distillation and PPO training arm."""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import scripts.train_go2_proprioceptive as v1_orchestrator
from src.tasks.velocity.config.go2.sim2real_safe_action_schema import schema_sha256


LOG_ROOT = ROOT / "logs/rsl_rl/go2_velocity"
TEACHER = ROOT / (
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
TEACHER_SHA256 = "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
ACTION_INTERFACE = "bounded_asymmetric_per_joint_v2"
DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill"
PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction"
DISTILL_RUN_NAME = (
  "go2_sim2real_proprio_v2_safe_action_meanbound5_"
  "v7_teacher_distill_2048env_300iter"
)
PPO_RUN_NAME = (
  "go2_sim2real_proprio_v2_safe_action_meanbound5_ppo_2048env_4000iter"
)
LOCK_PATH = v1_orchestrator.LOCK_PATH
PROVENANCE_ROOT = LOG_ROOT / "provenance"
SOURCE_MANIFEST_PATH_ENV = v1_orchestrator.SOURCE_MANIFEST_PATH_ENV
SOURCE_MANIFEST_SHA256_ENV = v1_orchestrator.SOURCE_MANIFEST_SHA256_ENV

# This is deliberately explicit. A formal manifest must not absorb unrelated
# dirty files, but every module capable of changing V2 training is bound here.
SOURCE_FILES = tuple(
  path
  for path in v1_orchestrator.SOURCE_FILES
  if path != "scripts/train_go2_proprioceptive.py"
) + (
  "scripts/train_go2_proprioceptive.py",
  "scripts/train_go2_proprioceptive_safe_action.py",
  "src/tasks/velocity/config/go2/sim2real_safe_action_schema.py",
  "src/tasks/velocity/rl/bounded_action_distribution.py",
  "tests/test_go2_bounded_action_distribution.py",
  "tests/test_go2_safe_action_sim2real.py",
  "docs/reviews/go2_safe_action_v2_design.md",
  "docs/reviews/go2_safe_action_v2_training_contract.md",
)
PACKAGE_DISTRIBUTIONS = v1_orchestrator.PACKAGE_DISTRIBUTIONS


def _sha256(path: Path) -> str:
  return v1_orchestrator._sha256(path)


def _git(*args: str) -> str:
  return v1_orchestrator._git(*args)


def _runs_with_suffix(suffix: str) -> set[Path]:
  return {path.resolve() for path in LOG_ROOT.glob(f"*_{suffix}") if path.is_dir()}


def _build_source_manifest() -> dict[str, Any]:
  tracked_files = set(_git("ls-files").splitlines())
  source_files = list(SOURCE_FILES)
  for relative in _git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
    if (
      ("safe_action" in relative or "safe-action" in relative)
      and relative not in source_files
      and (ROOT / relative).is_file()
    ):
      source_files.append(relative)
  files: dict[str, dict[str, Any]] = {}
  for relative in source_files:
    path = ROOT / relative
    if not path.is_file():
      raise FileNotFoundError(f"required V2 training source is missing: {relative}")
    entry: dict[str, Any] = {
      "git_state": "tracked" if relative in tracked_files else "untracked",
      "sha256": _sha256(path),
      "size_bytes": path.stat().st_size,
    }
    if relative not in tracked_files:
      entry["content_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    files[relative] = entry
  return {
    "schema_version": 2,
    "experiment": ACTION_INTERFACE,
    "workspace": str(ROOT),
    "git": {
      "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
      "head": _git("rev-parse", "HEAD"),
    },
    "python": {
      "implementation": platform.python_implementation(),
      "version": platform.python_version(),
    },
    "packages": {
      name: metadata.version(distribution)
      for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    },
    "student_schema_sha256": schema_sha256(),
    "source_files": sorted(source_files),
    "files": files,
  }


def _source_manifest_bytes(payload: dict[str, Any]) -> bytes:
  return (
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    + "\n"
  ).encode("utf-8")


def _write_source_manifest(path: Path, payload: dict[str, Any]) -> str:
  serialized = _source_manifest_bytes(payload)
  expected_sha = hashlib.sha256(serialized).hexdigest()
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    with path.open("xb") as stream:
      stream.write(serialized)
  except FileExistsError:
    if path.read_bytes() != serialized:
      raise RuntimeError(f"refusing to overwrite different source manifest: {path}")
  if _sha256(path) != expected_sha:
    raise RuntimeError("source manifest write verification failed")
  return expected_sha


def _generate_source_manifest(path: Path | None = None) -> tuple[Path, str]:
  payload = _build_source_manifest()
  serialized = _source_manifest_bytes(payload)
  manifest_sha = hashlib.sha256(serialized).hexdigest()
  if path is None:
    path = PROVENANCE_ROOT / f"go2_safe_action_v2_source_manifest_{manifest_sha}.json"
  resolved = path.expanduser().resolve()
  written_sha = _write_source_manifest(resolved, payload)
  if written_sha != manifest_sha:
    raise RuntimeError("source manifest identity changed during generation")
  return resolved, manifest_sha


def _validate_source_manifest(path: Path) -> str:
  resolved = path.expanduser().resolve()
  try:
    payload = json.loads(resolved.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"cannot read V2 source manifest: {resolved}") from exc
  serialized = _source_manifest_bytes(payload)
  if resolved.read_bytes() != serialized:
    raise RuntimeError("V2 source manifest is not canonical JSON")
  if payload != _build_source_manifest():
    raise RuntimeError("V2 source manifest does not match current training sources")
  return hashlib.sha256(serialized).hexdigest()


def _validate_distillation_checkpoint(
  path: Path, source_manifest_sha256: str
) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  infos = payload.get("infos") or {}
  required = {
    "proprioceptive_stage": "distillation",
    "student_schema_sha256": schema_sha256(),
    "teacher_sha256": TEACHER_SHA256,
    "source_manifest_sha256": source_manifest_sha256,
    "action_interface": ACTION_INTERFACE,
  }
  for name, expected in required.items():
    if infos.get(name) != expected:
      raise RuntimeError(f"V2 distillation checkpoint mismatch: {name}")
  if "student_state_dict" not in payload or "teacher_state_dict" not in payload:
    raise RuntimeError("V2 distillation checkpoint is missing student/teacher state")
  if int(payload.get("iter", -1)) != 299:
    raise RuntimeError("V2 distillation checkpoint iteration must be 299")
  return infos


def _validate_ppo_checkpoint(path: Path, source_manifest_sha256: str) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  infos = payload.get("infos") or {}
  required = {
    "proprioceptive_stage": "ppo",
    "student_schema_sha256": schema_sha256(),
    "source_manifest_sha256": source_manifest_sha256,
    "action_interface": ACTION_INTERFACE,
  }
  for name, expected in required.items():
    if infos.get(name) != expected:
      raise RuntimeError(f"V2 PPO checkpoint mismatch: {name}")
  if "actor_state_dict" not in payload or "critic_state_dict" not in payload:
    raise RuntimeError("V2 PPO checkpoint is missing actor/critic state")
  if int(payload.get("iter", -1)) != 3999:
    raise RuntimeError("V2 PPO checkpoint iteration must be 3999")
  return infos


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dry-run", action="store_true", help="Print the frozen V2 commands only."
  )
  modes = parser.add_mutually_exclusive_group()
  modes.add_argument("--generate-source-manifest", type=Path, metavar="PATH")
  modes.add_argument("--validate-source-manifest", type=Path, metavar="PATH")
  modes.add_argument("--source-manifest", type=Path, metavar="PATH")
  args = parser.parse_args()

  if args.generate_source_manifest is not None:
    path, manifest_sha = _generate_source_manifest(args.generate_source_manifest)
    print(f"source_manifest={path}")
    print(f"source_manifest_sha256={manifest_sha}")
    return
  if args.validate_source_manifest is not None:
    manifest_sha = _validate_source_manifest(args.validate_source_manifest)
    print(f"source_manifest={args.validate_source_manifest.expanduser().resolve()}")
    print(f"source_manifest_sha256={manifest_sha}")
    return

  if _sha256(TEACHER) != TEACHER_SHA256:
    raise RuntimeError("locked V7 teacher SHA256 mismatch")
  safe_schema_sha = schema_sha256()
  if len(safe_schema_sha) != 64:
    raise RuntimeError("V2 safe-action schema SHA256 is malformed")

  with v1_orchestrator._exclusive_training_lock(LOCK_PATH) if not args.dry_run else nullcontext():
    if args.source_manifest is None:
      if args.dry_run:
        payload = _build_source_manifest()
        manifest_sha = hashlib.sha256(_source_manifest_bytes(payload)).hexdigest()
        source_manifest = PROVENANCE_ROOT / (
          f"go2_safe_action_v2_source_manifest_{manifest_sha}.json"
        )
      else:
        source_manifest, manifest_sha = _generate_source_manifest()
    else:
      source_manifest = args.source_manifest.expanduser().resolve()
      manifest_sha = _validate_source_manifest(source_manifest)

    print(f"safe_action_schema_sha256={safe_schema_sha}")
    print(f"source_manifest_sha256={manifest_sha}")
    child_env = os.environ.copy()
    child_env[SOURCE_MANIFEST_PATH_ENV] = str(source_manifest)
    child_env[SOURCE_MANIFEST_SHA256_ENV] = manifest_sha

    known_distill = _runs_with_suffix(DISTILL_RUN_NAME)
    known_ppo = _runs_with_suffix(PPO_RUN_NAME)
    if (known_distill or known_ppo) and not args.dry_run:
      raise RuntimeError("matching safe-action V2 run already exists; refusing duplicate training")

    common = [sys.executable, "scripts/train.py"]
    distill_command = common + [
      DISTILL_TASK,
      "--env.scene.num-envs=2048",
      "--env.seed=42",
      "--agent.seed=42",
      "--agent.max-iterations=300",
      "--agent.save-interval=100",
      "--agent.logger=tensorboard",
      f"--agent.run-name={DISTILL_RUN_NAME}",
      "--agent.resume=True",
      "--agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter",
      "--agent.load-checkpoint=model_13600.pt",
    ]
    v1_orchestrator._run(distill_command, dry_run=args.dry_run, env=child_env)
    if args.dry_run:
      print("PPO load-run resolves to the one newly created V2 distillation run.")
      return

    _validate_source_manifest(source_manifest)
    new_distill = _runs_with_suffix(DISTILL_RUN_NAME) - known_distill
    if len(new_distill) != 1:
      raise RuntimeError("could not resolve exactly one new V2 distillation run")
    distill_run = new_distill.pop()
    distill_manifest = v1_orchestrator._install_source_manifest(
      source_manifest, distill_run
    )
    distill_checkpoint = distill_run / "model_299.pt"
    _validate_distillation_checkpoint(distill_checkpoint, manifest_sha)

    ppo_command = common + [
      PPO_TASK,
      "--env.scene.num-envs=2048",
      "--env.seed=42",
      "--agent.seed=42",
      "--agent.max-iterations=4000",
      "--agent.save-interval=250",
      "--agent.logger=tensorboard",
      f"--agent.run-name={PPO_RUN_NAME}",
      "--agent.resume=True",
      f"--agent.load-run={distill_run.name}",
      "--agent.load-checkpoint=model_299.pt",
    ]
    v1_orchestrator._run(ppo_command, dry_run=False, env=child_env)

    _validate_source_manifest(source_manifest)
    new_ppo = _runs_with_suffix(PPO_RUN_NAME) - known_ppo
    if len(new_ppo) != 1:
      raise RuntimeError("could not resolve exactly one new V2 PPO run")
    ppo_run = new_ppo.pop()
    ppo_manifest = v1_orchestrator._install_source_manifest(source_manifest, ppo_run)
    ppo_checkpoint = ppo_run / "model_3999.pt"
    _validate_ppo_checkpoint(ppo_checkpoint, manifest_sha)

    training_manifest = ppo_run / "safe_action_v2_training_manifest.json"
    document = {
      "schema_version": 1,
      "action_interface": ACTION_INTERFACE,
      "teacher": str(TEACHER),
      "teacher_sha256": TEACHER_SHA256,
      "student_schema_sha256": safe_schema_sha,
      "source_manifest": str(source_manifest),
      "source_manifest_sha256": manifest_sha,
      "distillation_source_manifest": str(distill_manifest),
      "distillation_run": str(distill_run),
      "distillation_checkpoint": str(distill_checkpoint),
      "distillation_checkpoint_sha256": _sha256(distill_checkpoint),
      "ppo_source_manifest": str(ppo_manifest),
      "ppo_run": str(ppo_run),
      "ppo_checkpoint": str(ppo_checkpoint),
      "ppo_checkpoint_sha256": _sha256(ppo_checkpoint),
    }
    with training_manifest.open("x", encoding="utf-8") as stream:
      json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
      stream.write("\n")


if __name__ == "__main__":
  main()
