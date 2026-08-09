"""Fail-closed CPU/TensorBoard screening for formal teacher checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.privileged_teacher_schema import (  # noqa: E402
  CANDIDATE_ACTOR_DIM,
  CRITIC_DIM,
  SOURCE_ACTOR_DIM,
)
from src.tasks.velocity.rl.privileged_teacher_transfer import (  # noqa: E402
  _validate_probe_checkpoint_infos,
  sha256_file,
)


FORMAL_UPDATES = (100, 200, 300, 400)


def _finite(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(_finite(item) for item in value.values())
  if isinstance(value, (tuple, list)):
    return all(_finite(item) for item in value)
  if isinstance(value, float):
    return math.isfinite(value)
  return True


def audit_tensorboard(run_dir: Path) -> dict[str, Any]:
  event_files = sorted(run_dir.glob("events.out.tfevents.*"))
  if len(event_files) != 1:
    raise RuntimeError(f"expected one TensorBoard event file, got {event_files}")
  accumulator = event_accumulator.EventAccumulator(
    str(event_files[0]), size_guidance={
      event_accumulator.SCALARS: 0,
      event_accumulator.HISTOGRAMS: 0,
      event_accumulator.TENSORS: 0,
    },
  )
  accumulator.Reload()
  tags = accumulator.Tags()
  scalar_count = 0
  histogram_count = 0
  tensor_count = 0
  for tag in tags.get("scalars", ()):
    values = accumulator.Scalars(tag)
    if not values or not all(math.isfinite(float(item.value)) for item in values):
      raise RuntimeError(f"TensorBoard scalar is empty or non-finite: {tag}")
    scalar_count += len(values)
  for tag in tags.get("histograms", ()):
    values = accumulator.Histograms(tag)
    for item in values:
      histogram = item.histogram_value
      numbers = (
        histogram.min, histogram.max, histogram.num,
        histogram.sum, histogram.sum_squares,
      )
      if not all(math.isfinite(float(value)) for value in numbers):
        raise RuntimeError(f"TensorBoard histogram is non-finite: {tag}")
    histogram_count += len(values)
  for tag in tags.get("tensors", ()):
    values = accumulator.Tensors(tag)
    for item in values:
      array = tensor_util.make_ndarray(item.tensor_proto)
      if array.dtype.kind in "fc" and not bool(torch.isfinite(torch.as_tensor(array)).all()):
        raise RuntimeError(f"TensorBoard tensor is non-finite: {tag}")
    tensor_count += len(values)
  if scalar_count == 0:
    raise RuntimeError("TensorBoard contains no scalar telemetry")
  return {
    "event_file": str(event_files[0].resolve()),
    "event_file_sha256": sha256_file(event_files[0]),
    "scalar_tags": len(tags.get("scalars", ())),
    "scalar_events": scalar_count,
    "histogram_tags": len(tags.get("histograms", ())),
    "histogram_events": histogram_count,
    "tensor_tags": len(tags.get("tensors", ())),
    "tensor_events": tensor_count,
    "all_finite": True,
  }


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
  _validate_probe_checkpoint_infos(infos, actor_dim)
  if (
    infos.get("source_manifest") != str(manifest.resolve())
    or infos.get("source_manifest_sha256") != manifest_sha
  ):
    raise RuntimeError(f"checkpoint manifest identity differs: {path}")
  required = ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")
  if any(name not in payload or not _finite(payload[name]) for name in required):
    raise RuntimeError(f"checkpoint contains missing/non-finite training state: {path}")
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
    "rollout_screening_pending": True,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--arm", choices=("control_234", "candidate_237"), required=True)
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
    output.write_text(
      json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
  main()
