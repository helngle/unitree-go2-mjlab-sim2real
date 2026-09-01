"""Run the frozen Go2 teacher-rollout BC then proprioceptive PPO arm."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterator

import torch

from src.tasks.velocity.config.go2.sim2real_schema import schema_sha256


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/rsl_rl/go2_velocity"
TEACHER = ROOT / (
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
TEACHER_SHA256 = "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill"
PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1"
DISTILL_RUN_NAME = "go2_sim2real_proprio_v1_v7_teacher_distill_2048env_300iter"
PPO_RUN_NAME = "go2_sim2real_proprio_v1_ppo_2048env_4000iter"
LOCK_PATH = LOG_ROOT / ".go2_proprioceptive_training.lock"
PROVENANCE_ROOT = LOG_ROOT / "provenance"
SOURCE_MANIFEST_PATH_ENV = "GO2_PROPRIO_SOURCE_MANIFEST"
SOURCE_MANIFEST_SHA256_ENV = "GO2_PROPRIO_SOURCE_MANIFEST_SHA256"
TECHNICAL_FAILURE_MARKER = "proprioceptive_technical_failure.json"
REGISTERED_TECHNICAL_FAILURES = {
  "2026-07-27_16-17-11_go2_sim2real_proprio_v1_v7_teacher_distill_2048env_300iter": (
    "837a14f4a6aef0bcab6b4f541b16ec38566624c22dee8d1b5398b0bc7a0b3ba1"
  ),
}

SOURCE_FILES = (
  "setup.py",
  "scripts/train_go2_proprioceptive.py",
  "scripts/train.py",
  "src/assets/robots/__init__.py",
  "src/assets/robots/unitree_go2/__init__.py",
  "src/assets/robots/unitree_go2/go2_constants.py",
  "src/assets/robots/unitree_go2/xmls/go2.xml",
  "src/tasks/velocity/__init__.py",
  "src/tasks/velocity/velocity_env_cfg.py",
  "src/tasks/velocity/config/go2/__init__.py",
  "src/tasks/velocity/config/go2/env_cfgs.py",
  "src/tasks/velocity/config/go2/high_slope_sampling.py",
  "src/tasks/velocity/config/go2/rl_cfg.py",
  "src/tasks/velocity/config/go2/sim2real_schema.py",
  "src/tasks/velocity/mdp/curriculums.py",
  "src/tasks/velocity/mdp/__init__.py",
  "src/tasks/velocity/mdp/mode_velocity_command.py",
  "src/tasks/velocity/mdp/observations.py",
  "src/tasks/velocity/mdp/rewards.py",
  "src/tasks/velocity/mdp/terminations.py",
  "src/tasks/velocity/mdp/velocity_command.py",
  "src/tasks/velocity/rl/runner.py",
  "src/tasks/velocity/rl/__init__.py",
  "src/tasks/velocity/rl/teacher_rollout_distillation.py",
)
PACKAGE_DISTRIBUTIONS = {
  "mjlab": "mjlab",
  "mujoco": "mujoco",
  "numpy": "numpy",
  "onnx": "onnx",
  "rsl_rl": "rsl-rl-lib",
  "torch": "torch",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _runs_with_suffix(suffix: str) -> set[Path]:
  return {path.resolve() for path in LOG_ROOT.glob(f"*_{suffix}") if path.is_dir()}


def _is_registered_prelearning_failure(run: Path) -> bool:
  expected_sha = REGISTERED_TECHNICAL_FAILURES.get(run.name)
  if expected_sha is None:
    return False
  marker = run / TECHNICAL_FAILURE_MARKER
  if not marker.is_file() or _sha256(marker) != expected_sha:
    raise RuntimeError(f"registered technical-failure marker mismatch: {run}")
  payload = json.loads(marker.read_text(encoding="utf-8"))
  if (
    payload.get("status") != "TECHNICAL_FAILURE_PRE_LEARNING"
    or payload.get("stage") != "distillation"
    or int(payload.get("learned_iterations", -1)) != 0
    or int(payload.get("checkpoint_count", -1)) != 0
    or list(run.glob("model_*.pt"))
  ):
    raise RuntimeError(f"technical-failure retry contract mismatch: {run}")
  return True


def _git(*args: str) -> str:
  return subprocess.check_output(
    ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
  ).strip()


def _build_source_manifest() -> dict[str, Any]:
  tracked_files = set(_git("ls-files").splitlines())
  files: dict[str, dict[str, Any]] = {}
  for relative in SOURCE_FILES:
    path = ROOT / relative
    if not path.is_file():
      raise FileNotFoundError(f"required training source is missing: {relative}")
    entry: dict[str, Any] = {
      "git_state": "tracked" if relative in tracked_files else "untracked",
      "sha256": _sha256(path),
      "size_bytes": path.stat().st_size,
    }
    if relative not in tracked_files:
      entry["content_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    files[relative] = entry
  return {
    "schema_version": 1,
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
    path = PROVENANCE_ROOT / f"go2_proprio_source_manifest_{manifest_sha}.json"
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
    raise RuntimeError(f"cannot read source manifest: {resolved}") from exc
  serialized = _source_manifest_bytes(payload)
  if resolved.read_bytes() != serialized:
    raise RuntimeError("source manifest is not canonical JSON")
  current = _build_source_manifest()
  if payload != current:
    raise RuntimeError("source manifest does not match current training sources")
  return hashlib.sha256(serialized).hexdigest()


def _install_source_manifest(source: Path, run_dir: Path) -> Path:
  destination = run_dir / "proprioceptive_source_manifest.json"
  content = source.read_bytes()
  try:
    with destination.open("xb") as stream:
      stream.write(content)
  except FileExistsError:
    if destination.read_bytes() != content:
      raise RuntimeError(f"run contains a different source manifest: {destination}")
  if _sha256(destination) != _sha256(source):
    raise RuntimeError("run source manifest copy verification failed")
  return destination


@contextmanager
def _exclusive_training_lock(path: Path = LOCK_PATH) -> Iterator[None]:
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a+", encoding="ascii") as stream:
    try:
      fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      stream.seek(0)
      owner = stream.read().strip() or "unknown"
      raise RuntimeError(
        f"proprioceptive training lock is held by {owner}; refusing duplicate launch"
      ) from exc
    try:
      stream.seek(0)
      stream.truncate()
      stream.write(f"pid={os.getpid()}\n")
      stream.flush()
      yield
    finally:
      fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
  print(" ".join(command), flush=True)
  if not dry_run:
    subprocess.run(command, cwd=ROOT, check=True, env=env)


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
  }
  for name, expected in required.items():
    if infos.get(name) != expected:
      raise RuntimeError(f"distillation checkpoint provenance mismatch: {name}")
  if "student_state_dict" not in payload or "teacher_state_dict" not in payload:
    raise RuntimeError("distillation checkpoint is missing student/teacher state")
  if int(payload.get("iter", -1)) != 299:
    raise RuntimeError("distillation checkpoint iteration must be 299")
  return infos


def _validate_ppo_checkpoint(path: Path, source_manifest_sha256: str) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  infos = payload.get("infos") or {}
  required = {
    "proprioceptive_stage": "ppo",
    "student_schema_sha256": schema_sha256(),
    "source_manifest_sha256": source_manifest_sha256,
  }
  for name, expected in required.items():
    if infos.get(name) != expected:
      raise RuntimeError(f"PPO checkpoint provenance mismatch: {name}")
  if "actor_state_dict" not in payload or "critic_state_dict" not in payload:
    raise RuntimeError("PPO checkpoint is missing actor/critic state")
  if int(payload.get("iter", -1)) != 3999:
    raise RuntimeError("PPO checkpoint iteration must be 3999")
  return infos


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dry-run", action="store_true", help="Print the frozen subprocess commands only."
  )
  modes = parser.add_mutually_exclusive_group()
  modes.add_argument(
    "--generate-source-manifest",
    type=Path,
    metavar="PATH",
    help="Generate the current canonical source manifest without training.",
  )
  modes.add_argument(
    "--validate-source-manifest",
    type=Path,
    metavar="PATH",
    help="Validate an existing source manifest against the current workspace.",
  )
  modes.add_argument(
    "--source-manifest",
    type=Path,
    metavar="PATH",
    help="Use an explicitly generated and validated manifest for formal training.",
  )
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
  if args.dry_run:
    payload = _build_source_manifest()
    source_manifest_sha = hashlib.sha256(_source_manifest_bytes(payload)).hexdigest()
    source_manifest = args.source_manifest
    if source_manifest is not None:
      source_manifest_sha = _validate_source_manifest(source_manifest)
    print(f"source_manifest_sha256={source_manifest_sha}")

  with _exclusive_training_lock() if not args.dry_run else nullcontext():
    if args.source_manifest is None:
      if args.dry_run:
        source_manifest = PROVENANCE_ROOT / (
          f"go2_proprio_source_manifest_{source_manifest_sha}.json"
        )
      else:
        source_manifest, source_manifest_sha = _generate_source_manifest()
    else:
      source_manifest = args.source_manifest.expanduser().resolve()
      source_manifest_sha = _validate_source_manifest(source_manifest)

    child_env = os.environ.copy()
    child_env[SOURCE_MANIFEST_PATH_ENV] = str(source_manifest)
    child_env[SOURCE_MANIFEST_SHA256_ENV] = source_manifest_sha

    known_distill = _runs_with_suffix(DISTILL_RUN_NAME)
    known_ppo = _runs_with_suffix(PPO_RUN_NAME)
    blocking_distill = {
      run for run in known_distill if not _is_registered_prelearning_failure(run)
    }
    if (blocking_distill or known_ppo) and not args.dry_run:
      raise RuntimeError("matching proprioceptive run already exists; refusing duplicate training")

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
    _run(distill_command, dry_run=args.dry_run, env=child_env)
    if args.dry_run:
      print("PPO command resolves --agent.load-run to the exact new distillation directory.")
      return

    _validate_source_manifest(source_manifest)
    new_distill = _runs_with_suffix(DISTILL_RUN_NAME) - known_distill
    if len(new_distill) != 1:
      raise RuntimeError("could not resolve exactly one new distillation run")
    distill_run = new_distill.pop()
    distill_source_manifest = _install_source_manifest(source_manifest, distill_run)
    distill_checkpoint = distill_run / "model_299.pt"
    _validate_distillation_checkpoint(distill_checkpoint, source_manifest_sha)

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
    _run(ppo_command, dry_run=False, env=child_env)

    _validate_source_manifest(source_manifest)
    new_ppo = _runs_with_suffix(PPO_RUN_NAME) - known_ppo
    if len(new_ppo) != 1:
      raise RuntimeError("could not resolve exactly one new PPO run")
    ppo_run = new_ppo.pop()
    ppo_source_manifest = _install_source_manifest(source_manifest, ppo_run)
    ppo_checkpoint = ppo_run / "model_3999.pt"
    _validate_ppo_checkpoint(ppo_checkpoint, source_manifest_sha)
    manifest = {
      "schema_version": 2,
      "teacher": str(TEACHER),
      "teacher_sha256": TEACHER_SHA256,
      "student_schema_sha256": schema_sha256(),
      "source_manifest": str(source_manifest),
      "source_manifest_sha256": source_manifest_sha,
      "distillation_source_manifest": str(distill_source_manifest),
      "distillation_run": str(distill_run),
      "distillation_checkpoint": str(distill_checkpoint),
      "distillation_checkpoint_sha256": _sha256(distill_checkpoint),
      "ppo_run": str(ppo_run),
      "ppo_source_manifest": str(ppo_source_manifest),
      "ppo_checkpoint": str(ppo_checkpoint),
      "ppo_checkpoint_sha256": _sha256(ppo_checkpoint),
    }
    training_manifest = ppo_run / "proprioceptive_training_manifest.json"
    with training_manifest.open("x", encoding="utf-8") as stream:
      json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
      stream.write("\n")


if __name__ == "__main__":
  main()
