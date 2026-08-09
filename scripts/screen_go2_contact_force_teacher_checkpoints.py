"""Fail-closed CPU/TensorBoard screening for contact-force Teacher checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_go2_privileged_teacher_checkpoints import (  # noqa: E402
  _finite,
  audit_tensorboard,
)
from src.tasks.velocity.contact_force_teacher_schema import (  # noqa: E402
  CANDIDATE_ACTOR_DIM,
  CONTACT_FORCE_ACTOR_SLICE,
  CRITIC_DIM,
  SOURCE_ACTOR_DIM,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (  # noqa: E402
  _validate_checkpoint_infos,
  sha256_file,
)


FORMAL_UPDATES = (100, 200, 300, 400)


def _new_column_optimizer_norm(payload: dict[str, Any]) -> float | None:
  actor = payload["actor_state_dict"]
  if actor["mlp.0.weight"].shape[-1] == SOURCE_ACTOR_DIM:
    return None
  start, end = CONTACT_FORCE_ACTOR_SLICE
  weight = actor["mlp.0.weight"][:, start:end]
  norm = float(torch.linalg.vector_norm(weight))
  if not torch.isfinite(torch.as_tensor(norm)) or norm <= 0.0:
    raise RuntimeError("candidate contact-force columns were not learned")
  return norm


def screen_checkpoint(
  path: Path, *, arm: str, manifest: Path, manifest_sha: str,
  tensorboard: dict[str, Any],
) -> dict[str, Any]:
  actor_dim = SOURCE_ACTOR_DIM if arm == "control_234" else CANDIDATE_ACTOR_DIM
  update = int(path.stem.removeprefix("model_"))
  payload = torch.load(path, map_location="cpu", weights_only=False)
  if int(payload.get("iter", -1)) != update:
    raise RuntimeError(f"checkpoint cursor differs from filename: {path}")
  infos = payload.get("infos") or {}
  _validate_checkpoint_infos(infos, actor_dim)
  if (
    infos.get("source_manifest") != str(manifest.resolve())
    or infos.get("source_manifest_sha256") != manifest_sha
  ):
    raise RuntimeError(f"checkpoint manifest identity differs: {path}")
  required = ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")
  if any(name not in payload or not _finite(payload[name]) for name in required):
    raise RuntimeError(f"checkpoint contains missing/non-finite state: {path}")
  actor = payload["actor_state_dict"]
  critic = payload["critic_state_dict"]
  if actor["mlp.0.weight"].shape != (512, actor_dim):
    raise RuntimeError(f"checkpoint actor shape differs: {path}")
  if critic["mlp.0.weight"].shape != (512, CRITIC_DIM):
    raise RuntimeError(f"checkpoint critic shape differs: {path}")
  env_state = infos.get("env_state") or {}
  if any(name not in env_state for name in ("terrain_levels", "terrain_types")):
    raise RuntimeError(f"checkpoint environment state is missing: {path}")
  if any(env_state[name].numel() != 2048 for name in ("terrain_levels", "terrain_types")):
    raise RuntimeError(f"checkpoint environment state is not 2048-D: {path}")
  return {
    "schema_version": 1,
    "checkpoint": {"path": str(path), "sha256": sha256_file(path)},
    "checkpoint_update": update,
    "arm": arm,
    "actor_dim": actor_dim,
    "critic_dim": CRITIC_DIM,
    "action_dim": 12,
    "provenance_valid": True,
    "formal_metadata_valid": True,
    "recursive_finite": True,
    "optimizer_present_finite": True,
    "formal_environment_state_valid": True,
    "tensorboard": tensorboard,
    "tensorboard_finite": True,
    "contact_force_column_weight_norm": _new_column_optimizer_norm(payload),
    "contact_force_columns_learned": arm == "candidate_246",
    "rollout_screening_pending": True,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--arm", choices=("control_234", "candidate_246"), required=True)
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--source-manifest-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  run = args.run_dir.expanduser().resolve()
  manifest = args.source_manifest.expanduser().resolve()
  if sha256_file(manifest) != args.source_manifest_sha256:
    raise RuntimeError("source manifest SHA256 mismatch")
  observed = sorted(
    int(path.stem.removeprefix("model_"))
    for path in run.glob("model_*.pt")
    if path.stem.removeprefix("model_").isdigit()
  )
  if observed != list(FORMAL_UPDATES):
    raise RuntimeError(f"formal checkpoint schedule differs: {observed}")
  tensorboard = audit_tensorboard(run)
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  for update in FORMAL_UPDATES:
    payload = screen_checkpoint(
      (run / f"model_{update}.pt").resolve(), arm=args.arm,
      manifest=manifest, manifest_sha=args.source_manifest_sha256,
      tensorboard=tensorboard,
    )
    output = output_dir / f"{args.arm}_model_{update}.screening.json"
    if output.exists():
      raise FileExistsError(f"refusing to overwrite screening artifact: {output}")
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
  main()
