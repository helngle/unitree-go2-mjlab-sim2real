"""Build the immutable formal checkpoint inventory for a PPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.tasks.velocity.evaluation.proprio_acceptance import (
  EXPECTED_PPO_ITERATIONS,
  load_checkpoint_lineage,
  sha256_file,
)


def _identity(path: Path) -> dict[str, str]:
  resolved = path.expanduser().resolve()
  if not resolved.exists():
    raise FileNotFoundError(resolved)
  return {"path": str(resolved), "sha256": sha256_file(resolved)}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path)
  parser.add_argument("--original-run", type=Path)
  parser.add_argument("--resume-run", type=Path)
  parser.add_argument("--resume-anchor", type=Path)
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--split-iteration", type=int, default=1500)
  parser.add_argument("--exclude-run", type=Path, action="append", required=True)
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()

  monolithic = args.run is not None
  split_values = (args.original_run, args.resume_run, args.resume_anchor)
  if monolithic and any(value is not None for value in split_values):
    parser.error("--run cannot be combined with split-resume arguments")
  if not monolithic and any(value is None for value in split_values):
    parser.error(
      "supply either --run or all of --original-run, --resume-run, "
      "--resume-anchor"
    )
  run = args.run.expanduser().resolve() if monolithic else None
  original = args.original_run.expanduser().resolve() if args.original_run else None
  resumed = args.resume_run.expanduser().resolve() if args.resume_run else None
  anchor = args.resume_anchor.expanduser().resolve() if args.resume_anchor else None
  source_manifest = args.source_manifest.expanduser().resolve()
  checkpoints = []
  for iteration in EXPECTED_PPO_ITERATIONS:
    root = run if monolithic else (
      original if iteration <= args.split_iteration else resumed
    )
    assert root is not None
    checkpoint = root / f"model_{iteration}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("iter", -1)) != iteration:
      raise ValueError(f"checkpoint iteration mismatch: {checkpoint}")
    checkpoints.append({
      "iteration": iteration,
      "path": str(checkpoint.resolve()),
      "sha256": sha256_file(checkpoint),
    })

  resume_anchor = None
  if not monolithic:
    assert anchor is not None and original is not None
    anchor_payload = torch.load(anchor, map_location="cpu", weights_only=False)
    source_checkpoint = original / f"model_{args.split_iteration}.pt"
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    anchor_iteration = args.split_iteration + 1
    if int(anchor_payload.get("iter", -1)) != anchor_iteration:
      raise ValueError(f"resume anchor iteration must be {anchor_iteration}")
    anchor_payload["iter"] = args.split_iteration

    def equal(left, right) -> bool:
      if torch.is_tensor(left):
        return torch.equal(left, right)
      if isinstance(left, dict):
        return left.keys() == right.keys() and all(
          equal(left[key], right[key]) for key in left
        )
      if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
          equal(a, b) for a, b in zip(left, right, strict=True)
        )
      return left == right

    if not equal(anchor_payload, source_payload):
      raise ValueError(
        f"resume anchor differs from model_{args.split_iteration} beyond its cursor"
      )
    resume_anchor = {
      "path": _identity(anchor),
      "derived_from": _identity(source_checkpoint),
      "semantic_change": {"iter": [args.split_iteration, anchor_iteration]},
    }

  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite checkpoint lineage: {output}")
  document = {
    "schema_version": 1,
    "evaluation_suite": "go2_proprioceptive_checkpoint_lineage",
    "execution_mode": "monolithic" if monolithic else "split_resume",
    "source_manifest": _identity(source_manifest),
    **({"run": {"path": str(run)}} if monolithic else {
      "resume_anchor": resume_anchor,
    }),
    "excluded_technical_runs": [
      {
        "path": str(path.expanduser().resolve()),
        "reason": "excluded technical resume branch; not part of formal lineage",
      }
      for path in args.exclude_run
    ],
    "checkpoints": checkpoints,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
  load_checkpoint_lineage(output)
  print(json.dumps(document, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
