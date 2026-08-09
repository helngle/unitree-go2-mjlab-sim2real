"""Generate provenance and run matched Go2 contact-force Teacher arms."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime
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

from src.tasks.velocity.contact_force_teacher_schema import (
  CANDIDATE_ACTOR_DIM,
  MATCHED_RNG_SEED,
  REFERENCE_COMPONENT,
  REFERENCE_DEVIATIONS,
  REFERENCE_DOI,
  REFERENCE_PAPER,
  REFERENCE_URL,
  SOURCE_ACTOR_DIM,
  TASK_IDS,
  schema_sha256,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (
  ALLOW_ENV_STATE_MISMATCH_ENV,
  SOURCE_MANIFEST_PATH_ENV,
  SOURCE_MANIFEST_SHA256_ENV,
  _validate_checkpoint_infos,
  sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/rsl_rl/go2_velocity"
PROVENANCE_ROOT = LOG_ROOT / "provenance"
LOCK_PATH = LOG_ROOT / ".go2_contact_force_teacher_training.lock"
PREFLIGHT_PATHS = {
  "control_234": ROOT / "docs/reviews/go2_contact_force_teacher_control_preflight_2048env_formal.json",
  "candidate_246": ROOT / "docs/reviews/go2_contact_force_teacher_candidate_preflight_2048env_formal.json",
}
FORMAL_UPDATES = (100, 200, 300, 400)
RUN_NAMES = {
  "control_234": "go2_contact_force_teacher_v1_control_234_2048env_400iter",
  "candidate_246": "go2_contact_force_teacher_v1_candidate_246_2048env_400iter",
}
SOURCE_FILES = (
  "AGENTS.md",
  "setup.py",
  "scripts/train.py",
  "scripts/train_go2_contact_force_teacher.py",
  "scripts/preflight_go2_contact_force_teacher.py",
  "scripts/smoke_go2_contact_force_teacher.py",
  "src/assets/robots/unitree_go2/go2_constants.py",
  "src/assets/robots/unitree_go2/xmls/go2.xml",
  "src/tasks/velocity/velocity_env_cfg.py",
  "src/tasks/velocity/config/go2/__init__.py",
  "src/tasks/velocity/config/go2/env_cfgs.py",
  "src/tasks/velocity/config/go2/rl_cfg.py",
  "src/tasks/velocity/config/go2/high_slope_sampling.py",
  "src/tasks/velocity/mdp/__init__.py",
  "src/tasks/velocity/mdp/curriculums.py",
  "src/tasks/velocity/mdp/mode_velocity_command.py",
  "src/tasks/velocity/mdp/observations.py",
  "src/tasks/velocity/mdp/rewards.py",
  "src/tasks/velocity/mdp/terminations.py",
  "src/tasks/velocity/mdp/velocity_command.py",
  "src/tasks/velocity/contact_force_teacher_schema.py",
  "src/tasks/velocity/privileged_teacher_schema.py",
  "src/tasks/velocity/rl/__init__.py",
  "src/tasks/velocity/rl/runner.py",
  "src/tasks/velocity/rl/privileged_teacher_transfer.py",
  "src/tasks/velocity/rl/contact_force_teacher_transfer.py",
)
PACKAGE_DISTRIBUTIONS = {
  "mjlab": "mjlab",
  "mujoco": "mujoco",
  "numpy": "numpy",
  "rsl_rl": "rsl-rl-lib",
  "torch": "torch",
}


def _git(*args: str) -> str:
  return subprocess.check_output(
    ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
  ).strip()


def _manifest_payload() -> dict[str, Any]:
  tracked = set(_git("ls-files").splitlines())
  files: dict[str, dict[str, Any]] = {}
  for relative in SOURCE_FILES:
    path = ROOT / relative
    if not path.is_file():
      raise FileNotFoundError(f"required training source is missing: {relative}")
    entry: dict[str, Any] = {
      "git_state": "tracked" if relative in tracked else "untracked",
      "sha256": sha256_file(path),
      "size_bytes": path.stat().st_size,
    }
    if relative not in tracked:
      entry["content_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    files[relative] = entry
  return {
    "schema_version": 1,
    "experiment": "go2_contact_force_teacher_v1",
    "contact_force_teacher_schema_sha256": schema_sha256(),
    "approved_method": {
      "user_approved": True,
      "single_variable": "append actor foot contact forces only",
      "transfer": "V7 function-preserving input expansion with zero new columns",
      "reference": {
        "paper": REFERENCE_PAPER,
        "url": REFERENCE_URL,
        "doi": REFERENCE_DOI,
        "component": REFERENCE_COMPONENT,
      },
      "declared_deviations": list(REFERENCE_DEVIATIONS),
    },
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


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
  return (
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    + "\n"
  ).encode("utf-8")


def generate_source_manifest(path: Path | None = None) -> tuple[Path, str]:
  payload = _manifest_payload()
  content = _manifest_bytes(payload)
  digest = hashlib.sha256(content).hexdigest()
  destination = (
    PROVENANCE_ROOT / f"go2_contact_force_teacher_source_manifest_{digest}.json"
    if path is None else path.expanduser().resolve()
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  try:
    with destination.open("xb") as stream:
      stream.write(content)
  except FileExistsError:
    if destination.read_bytes() != content:
      raise RuntimeError(f"refusing to overwrite different manifest: {destination}")
  if sha256_file(destination) != digest:
    raise RuntimeError("source manifest write verification failed")
  return destination.resolve(), digest


def validate_source_manifest(path: Path) -> str:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  content = _manifest_bytes(payload)
  if resolved.read_bytes() != content:
    raise RuntimeError("source manifest is not canonical JSON")
  if payload != _manifest_payload():
    raise RuntimeError("source manifest does not match current training sources")
  return hashlib.sha256(content).hexdigest()


def _all_tensors_finite(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(_all_tensors_finite(item) for item in value.values())
  if isinstance(value, (tuple, list)):
    return all(_all_tensors_finite(item) for item in value)
  return True


def validate_preflight_pair(manifest_sha: str) -> dict[str, Any]:
  payloads: dict[str, dict[str, Any]] = {}
  for arm, path in PREFLIGHT_PATHS.items():
    if not path.is_file():
      raise RuntimeError(f"required 2048-env preflight is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_dim = SOURCE_ACTOR_DIM if arm == "control_234" else CANDIDATE_ACTOR_DIM
    required = (
      payload.get("arm") == arm,
      payload.get("source_manifest_sha256") == manifest_sha,
      payload.get("num_envs") == 2048,
      payload.get("dimensions") == {"actor": expected_dim, "critic": 261, "action": 12},
      payload.get("learn_called") is False,
      payload.get("optimizer_step_called") is False,
      payload.get("fresh_optimizer_state_empty") is True,
      payload.get("actor_prefix_exact") is True,
      payload.get("critic_exact") is True,
      payload.get("candidate_new_columns_zero") is True,
      payload.get("candidate_normalizer_exact") is True,
      payload.get("actor_critic_contact_force_identity") is True,
      payload.get("formal_environment_state_exact") is True,
      payload.get("technical_smoke_override") is False,
      payload.get("finite") == {"observations_actions": True, "rewards": True},
    )
    if not all(required):
      raise RuntimeError(f"2048-env preflight hard gate failed: {path}")
    payloads[arm] = payload
  control = payloads["control_234"]
  candidate = payloads["candidate_246"]
  prefix_sha = control["initial_actor_prefix_sha256"]
  action_sha = control["initial_deterministic_action_sha256"]
  if candidate["initial_actor_prefix_sha256"] != prefix_sha:
    raise RuntimeError("paired preflight actor prefixes differ")
  if candidate["initial_deterministic_action_sha256"] != action_sha:
    raise RuntimeError("paired preflight initial actions differ")
  return {
    "paths": {arm: str(path.resolve()) for arm, path in PREFLIGHT_PATHS.items()},
    "initial_actor_prefix_sha256": prefix_sha,
    "initial_deterministic_action_sha256": action_sha,
    "initial_action_max_abs_difference": 0.0,
    "parity_threshold": 1.0e-6,
    "passed": True,
  }


def validate_formal_run(run: Path, arm: str, manifest_sha: str) -> dict[str, Any]:
  actor_dim = SOURCE_ACTOR_DIM if arm == "control_234" else CANDIDATE_ACTOR_DIM
  observed = sorted(
    int(path.stem.removeprefix("model_"))
    for path in run.glob("model_*.pt")
    if path.stem.removeprefix("model_").isdigit()
  )
  if observed != list(FORMAL_UPDATES):
    raise RuntimeError(f"formal checkpoint schedule differs: {observed}")
  checkpoints = []
  for update in FORMAL_UPDATES:
    path = (run / f"model_{update}.pt").resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iter", -1)) != update:
      raise RuntimeError(f"checkpoint iteration differs: {path}")
    infos = payload.get("infos") or {}
    _validate_checkpoint_infos(infos, actor_dim)
    if infos.get("source_manifest_sha256") != manifest_sha:
      raise RuntimeError(f"checkpoint source manifest differs: {path}")
    for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
      if name not in payload or not _all_tensors_finite(payload[name]):
        raise RuntimeError(f"checkpoint {name} missing or non-finite: {path}")
    checkpoints.append(
      {"path": str(path), "sha256": sha256_file(path), "update": update}
    )
  return {"arm": arm, "run": str(run), "checkpoints": checkpoints}


def _runs(run_name: str) -> set[Path]:
  return {path.resolve() for path in LOG_ROOT.glob(f"*_{run_name}") if path.is_dir()}


@contextmanager
def _training_lock() -> Iterator[None]:
  LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
  with LOCK_PATH.open("a+", encoding="ascii") as stream:
    try:
      fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      raise RuntimeError("contact-force Teacher training lock is already held") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(str(os.getpid()))
    stream.flush()
    yield


def _install_manifest(source: Path, run: Path) -> None:
  destination = run / "contact_force_teacher_source_manifest.json"
  content = source.read_bytes()
  try:
    with destination.open("xb") as stream:
      stream.write(content)
  except FileExistsError:
    if destination.read_bytes() != content:
      raise RuntimeError(f"run contains a different source manifest: {run}")


def run_formal_arms(manifest: Path, arms: tuple[str, ...]) -> dict[str, Any]:
  manifest_sha = validate_source_manifest(manifest)
  preflight = validate_preflight_pair(manifest_sha)
  child_env = os.environ.copy()
  child_env.pop(ALLOW_ENV_STATE_MISMATCH_ENV, None)
  child_env[SOURCE_MANIFEST_PATH_ENV] = str(manifest.resolve())
  child_env[SOURCE_MANIFEST_SHA256_ENV] = manifest_sha
  result: dict[str, Any] = {
    "schema_version": 1,
    "started_at": datetime.now().astimezone().isoformat(),
    "source_manifest": str(manifest.resolve()),
    "source_manifest_sha256": manifest_sha,
    "preflight": preflight,
    "arms": [],
  }
  with _training_lock():
    for arm in arms:
      if validate_source_manifest(manifest) != manifest_sha:
        raise RuntimeError("training sources changed after manifest validation")
      run_name = RUN_NAMES[arm]
      before = _runs(run_name)
      if before:
        raise RuntimeError(f"formal run already exists for {arm}: {sorted(before)}")
      command = (
        sys.executable, str(ROOT / "scripts/train.py"), TASK_IDS[arm],
        "--env.scene.num-envs=2048", f"--env.seed={MATCHED_RNG_SEED}",
        f"--agent.seed={MATCHED_RNG_SEED}", "--agent.max-iterations=400",
        "--agent.save-interval=100", "--agent.logger=tensorboard",
        f"--agent.run-name={run_name}",
      )
      subprocess.run(command, cwd=ROOT, env=child_env, check=True)
      created = _runs(run_name) - before
      if len(created) != 1:
        raise RuntimeError(f"could not resolve unique completed run for {arm}")
      run = created.pop()
      _install_manifest(manifest, run)
      result["arms"].append(validate_formal_run(run, arm, manifest_sha))
  result["completed_at"] = datetime.now().astimezone().isoformat()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--generate-source-manifest", action="store_true")
  parser.add_argument("--validate-source-manifest", type=Path)
  parser.add_argument("--source-manifest", type=Path)
  parser.add_argument("--run-formal", action="store_true")
  parser.add_argument("--arms", nargs="+", choices=tuple(RUN_NAMES), default=tuple(RUN_NAMES))
  parser.add_argument("--output-file", type=Path)
  args = parser.parse_args()
  if args.generate_source_manifest:
    path, digest = generate_source_manifest()
    print(f"source_manifest={path}")
    print(f"source_manifest_sha256={digest}")
    return
  if args.validate_source_manifest is not None:
    print(f"source_manifest_sha256={validate_source_manifest(args.validate_source_manifest)}")
    return
  if not args.run_formal or args.source_manifest is None:
    parser.error("--run-formal requires --source-manifest")
  payload = run_formal_arms(args.source_manifest, tuple(args.arms))
  encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
  print(encoded, end="")
  if args.output_file is not None:
    output = args.output_file.expanduser().resolve()
    if output.exists():
      raise FileExistsError(f"refusing to overwrite formal summary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
  main()
