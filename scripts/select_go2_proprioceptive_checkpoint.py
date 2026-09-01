"""Select a proprioceptive PPO checkpoint after hard-gate bundle validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.evaluation.proprio_acceptance import (
  CheckpointDecision,
  formal_evaluation_provenance,
  load_checkpoint_lineage,
  select_checkpoint_bundles,
  sha256_file,
  validate_formal_checkpoint_schedule,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


WORKSPACE = Path(__file__).resolve().parents[1]

SCREENING_BOOLEAN_GATES = (
  "provenance_valid",
  "lifecycle_valid",
  "recursive_finite",
  "schema_425_valid",
  "onnx_single_input",
  "onnx_no_privileged_inputs",
  "onnx_action_contract_valid",
  "no_reset_storm",
  "placement_valid",
  "action_limits_valid",
)


def discover_expected_checkpoints(run_dir: str | Path) -> list[Path]:
  root = Path(run_dir).expanduser().resolve()
  checkpoints = []
  for path in root.glob("model_*.pt"):
    suffix = path.stem.removeprefix("model_")
    if suffix.isdigit():
      checkpoints.append((int(suffix), path.resolve()))
  if not checkpoints:
    raise ValueError(f"no model_*.pt checkpoints found in {root}")
  return [path for _, path in sorted(checkpoints)]


def load_bundles(
  run_dir: str | Path | None, bundle_files: list[str | Path],
  *, expected_checkpoints: list[Path] | None = None,
) -> list[dict[str, Any]]:
  if expected_checkpoints is None:
    if run_dir is None:
      raise ValueError("run_dir or expected_checkpoints is required")
    expected = discover_expected_checkpoints(run_dir)
  else:
    expected = expected_checkpoints
  validate_formal_checkpoint_schedule(expected)
  bundles = []
  artifact_identities = []
  for name in bundle_files:
    path = Path(name).expanduser().resolve()
    payload = json.loads(path.read_text())
    assert_recursive_json_finite(payload)
    if payload.get("evaluation_suite") != "go2_proprioceptive_checkpoint_acceptance":
      raise ValueError(f"unexpected bundle suite: {path}")
    payload["_bundle_artifact"] = {
      "path": str(path), "sha256": sha256_file(path),
    }
    bundles.append(payload)
    artifact_identities.append(Path(payload["checkpoint"]["path"]).resolve())
  if len(artifact_identities) != len(set(artifact_identities)):
    raise ValueError("duplicate checkpoint bundle")
  if set(artifact_identities) != set(expected):
    missing = sorted(str(path) for path in set(expected) - set(artifact_identities))
    extra = sorted(str(path) for path in set(artifact_identities) - set(expected))
    raise ValueError(f"bundle checkpoint set mismatch: missing={missing}, extra={extra}")
  return bundles


def select(
  run_dir: str | Path | None, bundle_files: list[str | Path],
  *, checkpoint_manifest: str | Path | None = None,
) -> dict[str, Any]:
  lineage = (
    None if checkpoint_manifest is None
    else load_checkpoint_lineage(checkpoint_manifest)
  )
  bundles = load_bundles(
    run_dir, bundle_files,
    expected_checkpoints=None if lineage is None else list(lineage.checkpoints),
  )
  selected, decisions = select_checkpoint_bundles(bundles)
  payload = {
    "schema_version": 1,
    "selection_method": "hard_gates_then_preregistered_lexicographic",
    "weighted_score_used": False,
    "candidate_run_dir": (
      None if run_dir is None else str(Path(run_dir).expanduser().resolve())
    ),
    "checkpoint_lineage": None if lineage is None else {
      "path": str(lineage.manifest_path),
      "sha256": sha256_file(lineage.manifest_path),
    },
    "input_bundles": [bundle["_bundle_artifact"] for bundle in bundles],
    "decisions": [asdict(item) for item in decisions],
    "survivor_count": sum(item.passed for item in decisions),
    "selection_status": "SELECTED" if selected is not None else "NO_SAFE_SURVIVOR",
    "selected_checkpoint": None if selected is None else {
      "path": selected.checkpoint,
      "sha256": selected.checkpoint_sha256,
      "iteration": selected.iteration,
      "lexicographic_key": selected.lexicographic_key,
    },
  }
  primary = selected if selected is not None else decisions[0]
  payload["provenance"] = formal_evaluation_provenance(
    primary.checkpoint, __file__,
    (
      WORKSPACE / "src/tasks/velocity/evaluation/proprio_acceptance.py",
      *(() if lineage is None else (lineage.manifest_path,)),
    ),
    workspace=WORKSPACE,
    source_manifest=None if lineage is None else lineage.source_manifest,
  )
  assert_recursive_json_finite(payload)
  return payload


def select_screening_hard_gates(
  checkpoint_manifest: str | Path, screening_dir: str | Path,
) -> dict[str, Any]:
  """Fail closed before GPU rollout when checkpoint screening has no survivor."""
  lineage = load_checkpoint_lineage(checkpoint_manifest)
  screening_root = Path(screening_dir).expanduser().resolve()
  decisions: list[CheckpointDecision] = []
  screening_artifacts: list[dict[str, str]] = []
  for checkpoint in lineage.checkpoints:
    screening_path = screening_root / f"{checkpoint.stem}.screening.json"
    payload = json.loads(screening_path.read_text())
    assert_recursive_json_finite(payload)
    screening_artifacts.append({
      "path": str(screening_path), "sha256": sha256_file(screening_path),
    })
    expected_sha = sha256_file(checkpoint)
    iteration = int(checkpoint.stem.removeprefix("model_"))
    violations: list[str] = []
    identity = payload.get("checkpoint", {})
    if identity.get("path") != str(checkpoint) or identity.get("sha256") != expected_sha:
      violations.append("screening:checkpoint_identity")
    if payload.get("checkpoint_iteration") != iteration:
      violations.append("screening:checkpoint_iteration")
    source = payload.get("source_manifest", {})
    if (
      source.get("path") != str(lineage.source_manifest)
      or source.get("sha256") != sha256_file(lineage.source_manifest)
    ):
      violations.append("screening:source_manifest_identity")
    onnx = payload.get("screening_onnx", {})
    try:
      if sha256_file(onnx["path"]) != onnx["sha256"]:
        violations.append("screening:onnx_identity")
    except (KeyError, FileNotFoundError, TypeError):
      violations.append("screening:onnx_identity")
    for name in SCREENING_BOOLEAN_GATES:
      if payload.get(name) is not True:
        violations.append(f"contract:{name}")
    parity = payload.get("onnx_max_abs_error")
    if not isinstance(parity, (int, float)) or isinstance(parity, bool):
      violations.append("screening:onnx_max_abs_error")
    elif parity > 1.0e-5:
      violations.append(f"onnx_parity:{parity:.9g}>1e-5")
    fault_count = payload.get("screening_action_fault_count")
    if (
      not isinstance(fault_count, int)
      or isinstance(fault_count, bool)
      or fault_count < 0
    ):
      violations.append("contract:screening_action_fault_count")
    elif fault_count:
      violations.append(f"action_limits:screening_fault_count:{fault_count}")
    if payload.get("action_limits_valid") is not (fault_count == 0):
      violations.append("contract:action_limit_screening_consistency")
    decisions.append(CheckpointDecision(
      str(checkpoint), expected_sha, iteration, not violations,
      tuple(violations), None,
    ))
  survivors = [item for item in decisions if item.passed]
  selection_status = (
    "NO_SAFE_SURVIVOR" if not survivors
    else "SCREENING_SURVIVORS_REQUIRE_ROLLOUT"
  )
  payload = {
    "schema_version": 1,
    "selection_method": "hard_gates_then_preregistered_lexicographic",
    "selection_scope": "checkpoint_screening_hard_gates",
    "weighted_score_used": False,
    "rollout_selection_performed": False,
    "checkpoint_lineage": {
      "path": str(lineage.manifest_path),
      "sha256": sha256_file(lineage.manifest_path),
    },
    "input_screening_artifacts": screening_artifacts,
    "decisions": [asdict(item) for item in decisions],
    "survivor_count": len(survivors),
    "selection_status": selection_status,
    "selected_checkpoint": None,
    "provenance": formal_evaluation_provenance(
      decisions[0].checkpoint, __file__,
      (lineage.manifest_path, *(item["path"] for item in screening_artifacts)),
      workspace=WORKSPACE, source_manifest=lineage.source_manifest,
    ),
  }
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  parser = argparse.ArgumentParser()
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--run-dir", type=Path)
  source.add_argument("--checkpoint-manifest", type=Path)
  evidence = parser.add_mutually_exclusive_group(required=True)
  evidence.add_argument("--bundle", type=Path, action="append")
  evidence.add_argument("--screening-dir", type=Path)
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite formal selection: {output}")
  if args.screening_dir is not None:
    if args.checkpoint_manifest is None:
      raise ValueError("--screening-dir requires --checkpoint-manifest")
    payload = select_screening_hard_gates(
      args.checkpoint_manifest, args.screening_dir
    )
  else:
    payload = select(
      args.run_dir, args.bundle, checkpoint_manifest=args.checkpoint_manifest
    )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
