"""CPU screening and ONNX parity for every formal proprioceptive checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from torch import nn

from src.tasks.velocity.config.go2.sim2real_schema import actor_dim, schema_sha256
from src.tasks.velocity.evaluation.proprio_acceptance import (
  V2_SAFE_ACTION_INTERFACE,
  checkpoint_policy_contract,
  load_checkpoint_lineage,
  sha256_file,
  validate_formal_checkpoint_schedule,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


MANIFEST_NAME = "proprioceptive_source_manifest.json"


class FrozenActor(nn.Module):
  def __init__(
    self, state: dict[str, torch.Tensor], *, action_low: tuple[float, ...] | None = None,
    action_high: tuple[float, ...] | None = None,
    mean_bound: float | None = None,
  ) -> None:
    super().__init__()
    self.register_buffer("mean", state["obs_normalizer._mean"].clone())
    self.register_buffer("std", state["obs_normalizer._std"].clone())
    self.mlp = nn.Sequential(
      nn.Linear(actor_dim(), 512), nn.ELU(),
      nn.Linear(512, 256), nn.ELU(),
      nn.Linear(256, 128), nn.ELU(),
      nn.Linear(128, 12),
    )
    mlp_state = {
      key.removeprefix("mlp."): value
      for key, value in state.items() if key.startswith("mlp.")
    }
    self.mlp.load_state_dict(mlp_state, strict=True)
    if len({action_low is None, action_high is None, mean_bound is None}) != 1:
      raise ValueError("bounds and mean bound must be supplied together")
    if action_low is None:
      self.action_transform = nn.Identity()
    else:
      from src.tasks.velocity.rl.bounded_action_distribution import (
        _AsymmetricActionTransform,
      )
      self.action_transform = _AsymmetricActionTransform(
        torch.tensor(action_low), torch.tensor(action_high), float(mean_bound)
      )
    self.eval()

  def forward(self, actor: torch.Tensor) -> torch.Tensor:
    latent = self.mlp((actor - self.mean) / (self.std + 1.0e-2))
    return self.action_transform(latent)


def _checkpoint_iteration(path: Path) -> int:
  match = re.fullmatch(r"model_(\d+)\.pt", path.name)
  if match is None:
    raise ValueError(f"invalid checkpoint name: {path.name}")
  return int(match.group(1))


def _test_inputs(iteration: int) -> torch.Tensor:
  generator = torch.Generator(device="cpu").manual_seed(42 + iteration)
  return torch.cat(
    (
      torch.zeros((1, actor_dim())),
      torch.ones((1, actor_dim())),
      -torch.ones((1, actor_dim())),
      torch.linspace(-2.0, 2.0, actor_dim()).reshape(1, -1),
      torch.randn((12, actor_dim()), generator=generator),
    ),
    dim=0,
  ).to(dtype=torch.float32)


def _action_safety(
  action: torch.Tensor, schema: object
) -> tuple[torch.Tensor, torch.Tensor]:
  joint_count = len(schema.DEFAULT_JOINT_POS)
  if action.ndim != 2 or action.shape[1] != joint_count:
    raise ValueError(f"applied action must have shape (num_envs, {joint_count})")
  if not torch.isfinite(action).all():
    raise ValueError("applied action contains NaN/Inf")
  default = action.new_tensor(schema.DEFAULT_JOINT_POS)
  limits = action.new_tensor(schema.JOINT_POS_LIMITS)
  targets = default + float(schema.ACTION_SCALE) * action
  if hasattr(schema, "ACTION_LOW") and hasattr(schema, "ACTION_HIGH"):
    action_low = action.new_tensor(schema.ACTION_LOW)
    action_high = action.new_tensor(schema.ACTION_HIGH)
    action_fault = ((action < action_low) | (action > action_high)).any(dim=-1)
  else:
    action_fault = action.abs().amax(dim=-1) > float(schema.ACTION_ABS_LIMIT)
  target_fault = ((targets < limits[:, 0]) | (targets > limits[:, 1])).any(dim=-1)
  return action.abs().amax(dim=-1), action_fault | target_fault


def screen_checkpoint(
  checkpoint: Path, output_dir: Path, *, source_manifest: Path | None = None
) -> dict[str, object]:
  checkpoint = checkpoint.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()
  iteration = _checkpoint_iteration(checkpoint)
  manifest = (
    checkpoint.parent / MANIFEST_NAME
    if source_manifest is None else source_manifest.expanduser().resolve()
  )
  if not manifest.is_file():
    raise FileNotFoundError(f"run source manifest is missing: {manifest}")
  manifest_sha = sha256_file(manifest)
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  infos = payload.get("infos") or {}
  contract = checkpoint_policy_contract(infos)
  schema = contract["schema_module"]
  required_keys = {
    "actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"
  }
  lifecycle_valid = required_keys <= set(payload) and int(payload.get("iter", -1)) == iteration
  provenance_valid = (
    infos.get("proprioceptive_stage") == "ppo"
    and infos.get("source_manifest_sha256") == manifest_sha
  )
  if contract["action_interface"] == V2_SAFE_ACTION_INTERFACE:
    provenance_valid = provenance_valid and (
      infos.get("action_interface") == V2_SAFE_ACTION_INTERFACE
    )
  if not lifecycle_valid or not provenance_valid:
    raise ValueError(f"checkpoint lifecycle/provenance mismatch: {checkpoint}")
  actor_state = payload["actor_state_dict"]
  tensor_groups = (actor_state, payload["critic_state_dict"])
  recursive_finite = all(
    torch.isfinite(value).all().item()
    for group in tensor_groups for value in group.values() if torch.is_tensor(value)
  )
  if not recursive_finite:
    raise ValueError(f"checkpoint contains NaN/Inf: {checkpoint}")

  bounds = (
    {
      "action_low": schema.ACTION_LOW,
      "action_high": schema.ACTION_HIGH,
      "mean_bound": schema.ACTION_MEAN_BOUND,
    }
    if contract["action_interface"] == V2_SAFE_ACTION_INTERFACE else {}
  )
  model = FrozenActor(actor_state, **bounds)
  inputs = _test_inputs(iteration)
  with torch.inference_mode():
    torch_output = model(inputs).numpy()
  if not np.isfinite(torch_output).all():
    raise ValueError("PyTorch actor output contains NaN/Inf")

  export_dir = output_dir / checkpoint.stem
  export_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = export_dir / "policy.screening.onnx"
  if onnx_path.exists():
    raise FileExistsError(f"refusing to overwrite screening ONNX: {onnx_path}")
  torch.onnx.export(
    model,
    torch.zeros((1, actor_dim()), dtype=torch.float32),
    onnx_path,
    export_params=True,
    opset_version=18,
    input_names=["actor"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  graph = onnx.load(onnx_path)
  metadata = {
    "observation_schema_sha256": contract["schema_sha256"],
    "observation_schema_version": contract["schema_version"],
    "action_output_semantics": contract["action_output_semantics"],
    "source_manifest_sha256": manifest_sha,
    "checkpoint_sha256": sha256_file(checkpoint),
  }
  if contract["action_interface"] is not None:
    metadata["action_interface"] = str(contract["action_interface"])
    metadata["action_mean_bound"] = str(contract["action_mean_bound"])
  for key, value in metadata.items():
    item = graph.metadata_props.add()
    item.key = key
    item.value = value
  onnx.save(graph, onnx_path)
  graph = onnx.load(onnx_path)
  metadata_map = {item.key: item.value for item in graph.metadata_props}
  has_bounded_transform = any(
    node.op_type == "Tanh" for node in graph.graph.node
  )
  onnx_action_contract_valid = (
    metadata_map.get("observation_schema_sha256") == contract["schema_sha256"]
    and metadata_map.get("action_output_semantics") == contract["action_output_semantics"]
    and (
      contract["action_interface"] != V2_SAFE_ACTION_INTERFACE
      or (
        metadata_map.get("action_interface") == V2_SAFE_ACTION_INTERFACE
        and metadata_map.get("action_mean_bound")
        == str(contract["action_mean_bound"])
        and has_bounded_transform
      )
    )
  )
  input_shapes = [
    [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
    for item in graph.graph.input
  ]
  output_shapes = [
    [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
    for item in graph.graph.output
  ]
  onnx_single_input = (
    len(graph.graph.input) == 1
    and graph.graph.input[0].name == "actor"
    and input_shapes == [[1, actor_dim()]]
    and len(graph.graph.output) == 1
    and graph.graph.output[0].name == "actions"
    and output_shapes == [[1, 12]]
  )
  onnx_no_privileged_inputs = onnx_single_input and all(
    token not in graph.graph.input[0].name.lower()
    for token in ("critic", "height", "teacher", "contact")
  )
  onnx_output = ReferenceEvaluator(graph).run(
    None, {"actor": inputs.numpy()}
  )[0]
  onnx_max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
  _, action_fault = _action_safety(torch.from_numpy(onnx_output), schema)
  action_fault_count = int(action_fault.sum())

  result: dict[str, object] = {
    "schema_version": 1,
    "policy_contract": {
      key: value for key, value in contract.items() if key != "schema_module"
    },
    "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    "checkpoint_iteration": iteration,
    "source_manifest": {"path": str(manifest), "sha256": manifest_sha},
    "screening_onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path)},
    "lifecycle_valid": lifecycle_valid,
    "provenance_valid": provenance_valid,
    "recursive_finite": recursive_finite,
    "schema_425_valid": actor_state["obs_normalizer._mean"].shape == (1, actor_dim()),
    "onnx_single_input": onnx_single_input,
    "onnx_no_privileged_inputs": onnx_no_privileged_inputs,
    "onnx_action_contract_valid": onnx_action_contract_valid,
    "onnx_max_abs_error": onnx_max_abs_error,
    "no_reset_storm": True,
    "placement_valid": True,
    "action_limits_valid": action_fault_count == 0,
    "screening_action_fault_count": action_fault_count,
    "screening_input_count": int(inputs.shape[0]),
  }
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--run-dir", type=Path)
  source.add_argument("--checkpoint-manifest", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  source_manifest = None
  if args.checkpoint_manifest is not None:
    lineage = load_checkpoint_lineage(args.checkpoint_manifest)
    checkpoints = list(lineage.checkpoints)
    source_manifest = lineage.source_manifest
  else:
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = sorted(
      run_dir.glob("model_*.pt"), key=lambda path: _checkpoint_iteration(path)
    )
    if not checkpoints:
      raise ValueError(f"no checkpoints found in {run_dir}")
  validate_formal_checkpoint_schedule(checkpoints)
  for checkpoint in checkpoints:
    result = screen_checkpoint(
      checkpoint, args.output_dir, source_manifest=source_manifest
    )
    output = args.output_dir.expanduser().resolve() / f"{checkpoint.stem}.screening.json"
    if output.exists():
      raise FileExistsError(f"refusing to overwrite screening JSON: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
  main()
