"""Execute and aggregate the frozen 14-artifact teacher acceptance matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from src.tasks.velocity.evaluation.privileged_teacher_acceptance import (
  EXPECTED_ARTIFACTS,
  RawArtifact,
  aggregate_checkpoint,
  decide,
  validate_artifact_inventory,
)
from src.tasks.velocity.evaluation.proprio_acceptance import sha256_file
from src.tasks.velocity.privileged_teacher_schema import TASK_IDS
from src.tasks.velocity.rl.privileged_teacher_transfer import (
  SOURCE_CHECKPOINT,
  SOURCE_CHECKPOINT_SHA256,
  _validate_probe_checkpoint_infos,
)


ROOT = Path(__file__).resolve().parents[1]
V7_TASK = "Unitree-Go2-Rough-V7"
ROUTES = ("line", "arc", "s_curve")


@dataclass(frozen=True)
class Invocation:
  checkpoint_label: str
  checkpoint_path: str
  checkpoint_sha256: str
  task_id: str
  artifact_id: str
  command: tuple[str, ...]
  output_file: str


def _artifact_command(
  checkpoint: Path, task_id: str, artifact_id: str, output: Path
) -> tuple[str, ...]:
  common = ("--checkpoint", str(checkpoint), "--task-id", task_id)
  if artifact_id == "high_slope_matched":
    return (
      sys.executable, str(ROOT / "scripts/evaluate_go2_high_slope_matched.py"),
      *common, "--profiles", "clean", "randomized", "--radii", "2.5",
      "--output-file", str(output),
    )
  if artifact_id == "flat_matched_routes":
    return (
      sys.executable, str(ROOT / "scripts/evaluate_go2_matched_routes.py"),
      *common, "--profiles", "clean", "full_randomized",
      "--output-file", str(output),
    )
  profile = "randomized" if artifact_id.endswith("_randomized") else "clean"
  profile_common = (*common, "--profile", profile)
  if artifact_id.startswith("continuous_retained_"):
    return (
      sys.executable, str(ROOT / "scripts/evaluate_go2_terrain_boundary.py"),
      *profile_common, "--suite", "continuous_straight", "--route-kind", "straight",
      "--transition-cases", "random_rough", "discrete_obstacle", "stairs_up", "stairs_down",
      "--transition-levels", "3", "5", "7", "--output-file", str(output),
    )
  if artifact_id.startswith("terrain_curves_"):
    route = "s_curve" if "_s_curve_" in artifact_id else "arc"
    return (
      sys.executable, str(ROOT / "scripts/evaluate_go2_terrain_curves.py"),
      *profile_common, "--route-kind", route, "--terrain-kinds", "slope_up", "slope_down",
      "random_rough", "discrete_obstacle", "stairs_up", "stairs_down",
      "--output-file", str(output),
    )
  seed = artifact_id.removeprefix("stairs_level9_seed").split("_")[0]
  return (
    sys.executable, str(ROOT / "scripts/evaluate_go2_terrain_boundary.py"),
    *profile_common, "--suite", "continuous_straight", "--route-kind", "straight",
    "--transition-cases", "stairs_up", "stairs_down", "--transition-levels", "9",
    "--seed", seed, "--output-file", str(output),
  )


def build_matrix(
  checkpoints: tuple[tuple[str, Path, str], ...], output_dir: Path
) -> list[Invocation]:
  invocations: list[Invocation] = []
  for label, checkpoint, task_id in checkpoints:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    for artifact_id in sorted(EXPECTED_ARTIFACTS):
      output = output_dir / label / f"{artifact_id}.json"
      invocations.append(Invocation(
        checkpoint_label=label, checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha, task_id=task_id,
        artifact_id=artifact_id,
        command=_artifact_command(checkpoint, task_id, artifact_id, output),
        output_file=str(output),
      ))
  validate_artifact_inventory([
    RawArtifact(
      checkpoint_label=item.checkpoint_label, checkpoint_path=item.checkpoint_path,
      checkpoint_sha256=item.checkpoint_sha256, task_id=item.task_id,
      artifact_id=item.artifact_id, path=item.output_file,
    ) for item in invocations
  ])
  return invocations


def _validate_checkpoint(checkpoint: Path, task_id: str, *, v7: bool) -> None:
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if v7:
    if checkpoint.resolve() != SOURCE_CHECKPOINT.resolve() or sha256_file(checkpoint) != SOURCE_CHECKPOINT_SHA256:
      raise ValueError("V7 checkpoint identity differs from locked baseline")
    return
  actor_dim = int(payload["actor_state_dict"]["mlp.0.weight"].shape[-1])
  _validate_probe_checkpoint_infos(payload.get("infos") or {}, actor_dim)
  expected = TASK_IDS["control_234"] if actor_dim == 234 else TASK_IDS["candidate_237"]
  if task_id != expected:
    raise ValueError(f"teacher task/checkpoint arm mismatch: {task_id} != {expected}")


def _execute(invocations: list[Invocation], *, resume_existing: bool) -> None:
  for item in invocations:
    output = Path(item.output_file)
    if output.exists() and not resume_existing:
      raise FileExistsError(output)
    if not output.exists():
      output.parent.mkdir(parents=True, exist_ok=True)
      subprocess.run(item.command, cwd=ROOT, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    identity = payload.get("provenance", {}).get("checkpoint", {})
    if identity != {"path": item.checkpoint_path, "sha256": item.checkpoint_sha256}:
      raise ValueError(f"evaluation provenance differs: {output}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--control-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--decision-file", type=Path, required=True)
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--resume-existing", action="store_true")
  args = parser.parse_args()
  control = args.control_checkpoint.expanduser().resolve()
  candidate = args.candidate_checkpoint.expanduser().resolve()
  _validate_checkpoint(SOURCE_CHECKPOINT, V7_TASK, v7=True)
  _validate_checkpoint(control, TASK_IDS["control_234"], v7=False)
  _validate_checkpoint(candidate, TASK_IDS["candidate_237"], v7=False)
  matrix = build_matrix((
    ("v7", SOURCE_CHECKPOINT.resolve(), V7_TASK),
    ("control_234", control, TASK_IDS["control_234"]),
    ("candidate_237", candidate, TASK_IDS["candidate_237"]),
  ), args.output_dir.expanduser().resolve())
  plan = {"schema_version": 1, "evaluation_suite": "go2_privileged_teacher_14_artifact_plan",
          "serial_gpu_only": True, "invocations": [asdict(item) for item in matrix]}
  if args.execute:
    _execute(matrix, resume_existing=args.resume_existing)
    summaries = {}
    for label in ("v7", "control_234", "candidate_237"):
      subset = [item for item in matrix if item.checkpoint_label == label]
      summaries[label] = aggregate_checkpoint([
        RawArtifact(
          checkpoint_label=item.checkpoint_label, checkpoint_path=item.checkpoint_path,
          checkpoint_sha256=item.checkpoint_sha256, task_id=item.task_id,
          artifact_id=item.artifact_id, path=item.output_file,
        ) for item in subset
      ])
    result = decide(v7=summaries["v7"], control=summaries["control_234"], candidate=summaries["candidate_237"])
    result["plan"] = plan
    decision = args.decision_file.expanduser().resolve()
    if decision.exists():
      raise FileExistsError(decision)
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
  else:
    plan_file = args.decision_file.expanduser().resolve()
    if plan_file.exists():
      raise FileExistsError(plan_file)
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
