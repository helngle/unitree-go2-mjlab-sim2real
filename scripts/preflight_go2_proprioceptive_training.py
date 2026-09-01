"""Strict no-learning preflight for the Go2 proprioceptive student arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import warp as wp
from onnx.reference import ReferenceEvaluator

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.sim2real_schema import actor_dim, schema_sha256
from src.tasks.velocity.rl.runner import _proprio_metadata


DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill"
PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1"
V7_CHECKPOINT = Path(
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
EXPECTED_V7_SHA256 = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)
EXPECTED_DIMS = {"actor": 425, "critic": 261, "teacher": 234, "action": 12}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _all_finite(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(_all_finite(item) for item in value.values())
  if isinstance(value, (tuple, list)):
    return all(_all_finite(item) for item in value)
  return True


def _release_cuda_memory(device: str) -> None:
  """Release stage-local Torch and Warp allocations before the next env."""
  gc.collect()
  torch.cuda.empty_cache()
  previous_threshold = wp.get_mempool_release_threshold(device)
  try:
    wp.set_mempool_release_threshold(device, 0)
    wp.synchronize_device(device)
  finally:
    wp.set_mempool_release_threshold(device, previous_threshold)


def run_preflight(
  *, checkpoint: Path, num_envs: int, steps: int, seed: int, device: str,
  variant: str = "v1",
) -> dict[str, Any]:
  """Construct both stages and exercise inference/backprop without updating."""
  checkpoint = checkpoint.expanduser().resolve()
  if checkpoint != V7_CHECKPOINT.resolve():
    raise ValueError(f"teacher must be the locked V7 checkpoint: {V7_CHECKPOINT.resolve()}")
  teacher_sha = _sha256(checkpoint)
  if teacher_sha != EXPECTED_V7_SHA256:
    raise ValueError("locked V7 teacher SHA256 mismatch")
  if num_envs <= 0 or steps < 8:
    raise ValueError("num_envs must be positive and steps must be at least 8")

  if variant not in {"v1", "v2"}:
    raise ValueError("variant must be v1 or v2")
  if variant == "v2":
    distill_task = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill"
    ppo_task = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction"
    schema_module = importlib.import_module(
      "src.tasks.velocity.config.go2.sim2real_safe_action_schema"
    )
  else:
    distill_task = DISTILL_TASK
    ppo_task = PPO_TASK
    schema_module = importlib.import_module(
      "src.tasks.velocity.config.go2.sim2real_schema"
    )
  torch.manual_seed(seed)
  if device.startswith("cuda"):
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)

  distill_env_cfg = load_env_cfg(distill_task)
  distill_env_cfg.scene.num_envs = num_envs
  distill_env_cfg.seed = seed
  for group in distill_env_cfg.observations.values():
    if group is not None:
      group.nan_policy = "error"

  distill_env = ManagerBasedRlEnv(cfg=distill_env_cfg, device=device)
  try:
    distill_cfg = load_rl_cfg(distill_task)
    wrapped = RslRlVecEnvWrapper(distill_env, clip_actions=distill_cfg.clip_actions)
    distill_cls = load_runner_cls(distill_task)
    assert distill_cls is not None
    distill = distill_cls(wrapped, asdict(distill_cfg), device=device)
    distill.load(str(checkpoint), strict=True, map_location=device)

    observations = wrapped.get_observations()
    actual_dims = {
      name: int(observations[name].shape[-1])
      for name in ("actor", "critic", "teacher")
    }
    actual_dims["action"] = int(wrapped.num_actions)
    if actual_dims != EXPECTED_DIMS or actual_dims["actor"] != schema_module.actor_dim():
      raise RuntimeError(f"observation/action dimension mismatch: {actual_dims}")

    if not distill.alg.teacher_loaded:
      raise RuntimeError("teacher strict-load did not set teacher_loaded")
    if any(parameter.requires_grad for parameter in distill.alg.teacher.parameters()):
      raise RuntimeError("teacher parameters are not frozen")
    teacher_state = {
      name: value.detach().cpu().clone()
      for name, value in distill.alg.teacher.state_dict().items()
    }
    if len(teacher_state) != 13:
      raise RuntimeError("teacher state_dict key count differs from locked V7")
    teacher_normalizer_count = float(teacher_state["obs_normalizer.count"])
    if teacher_normalizer_count != 423247872.0:
      raise RuntimeError("teacher observation normalizer count mismatch")
    with torch.no_grad():
      teacher_once = distill.alg.teacher(observations)
      teacher_twice = distill.alg.teacher(observations)
    if not torch.equal(teacher_once, teacher_twice):
      raise RuntimeError("teacher deterministic inference is not bitwise stable")

    # Build and backpropagate the registered behavior loss once, but never step
    # the optimizer. This proves ownership without creating a trained artifact.
    distill.alg.optimizer.zero_grad()
    student_action = distill.alg.student(observations)
    with torch.no_grad():
      teacher_action = distill.alg.teacher(observations)
      if variant == "v2":
        teacher_action = distill.alg.student.distribution.transform(teacher_action)
    behavior_loss = F.huber_loss(student_action, teacher_action)
    behavior_loss.backward()
    student_grads = [
      parameter.grad
      for parameter in distill.alg.student.parameters()
      if parameter.grad is not None
    ]
    if not student_grads or not all(_all_finite(grad) for grad in student_grads):
      raise RuntimeError("student gradient ownership/finite check failed")
    if any(parameter.grad is not None for parameter in distill.alg.teacher.parameters()):
      raise RuntimeError("teacher received a gradient")
    distill.alg.optimizer.zero_grad()

    action_finite = True
    observation_finite = _all_finite(observations)
    reward_finite = True
    distill_reset_count = 0
    teacher_rollout_match = True
    applied_history_match = True
    action_bounds_valid = True
    joint_targets_valid = True
    term = None
    for _ in range(steps):
      action = distill.alg.act(observations)
      with torch.no_grad():
        expected_teacher_action = distill.alg.teacher(observations)
        if variant == "v2":
          expected_teacher_action = distill.alg.student.distribution.transform(
            expected_teacher_action
          )
      teacher_rollout_match &= bool(torch.equal(action, expected_teacher_action))
      observations, reward, dones, extras = wrapped.step(action)
      if variant == "v2":
        term = distill_env.action_manager.get_term("joint_pos")
        applied_history_match &= bool(torch.equal(term.raw_action, action))
        low = torch.tensor(schema_module.ACTION_LOW, device=action.device)
        high = torch.tensor(schema_module.ACTION_HIGH, device=action.device)
        action_bounds_valid &= bool(((action >= low) & (action <= high)).all())
        limits = distill_env.scene["robot"].data.joint_pos_limits[
          :, term.target_ids
        ]
        joint_targets_valid &= bool(
          (
            (term._processed_actions >= limits[..., 0])
            & (term._processed_actions <= limits[..., 1])
          ).all()
        )
      action_finite &= _all_finite(action)
      observation_finite &= _all_finite(observations)
      reward_finite &= _all_finite((reward, extras))
      distill_reset_count += int(dones.sum())
    if not teacher_rollout_match:
      raise RuntimeError("distillation environment did not execute teacher actions")
    for name, expected in teacher_state.items():
      if not torch.equal(distill.alg.teacher.state_dict()[name].detach().cpu(), expected):
        raise RuntimeError(f"teacher state changed during preflight: {name}")
  finally:
    distill_env.close()

  # The action term owns the environment manager. Keeping this loop-local
  # reference alive would retain the complete first 2048-env simulation.
  term = None
  observations = None
  action = None
  expected_teacher_action = None
  reward = None
  dones = None
  extras = None
  student_action = None
  teacher_action = None
  student_grads = None
  del distill, wrapped, distill_env
  if device.startswith("cuda"):
    _release_cuda_memory(device)

  ppo_env_cfg = load_env_cfg(ppo_task)
  if "teacher" in ppo_env_cfg.observations:
    raise RuntimeError("registered PPO environment leaks the teacher group")
  ppo_env_cfg.scene.num_envs = num_envs
  ppo_env_cfg.seed = seed
  for group in ppo_env_cfg.observations.values():
    if group is not None:
      group.nan_policy = "error"
  ppo_env = ManagerBasedRlEnv(cfg=ppo_env_cfg, device=device)
  try:
    ppo_cfg = load_rl_cfg(ppo_task)
    ppo_wrapped = RslRlVecEnvWrapper(ppo_env, clip_actions=ppo_cfg.clip_actions)
    ppo_cls = load_runner_cls(ppo_task)
    assert ppo_cls is not None
    ppo = ppo_cls(ppo_wrapped, asdict(ppo_cfg), device=device)
    ppo_observations = ppo_wrapped.get_observations()
    if set(ppo_observations.keys()) != {"actor", "critic"}:
      raise RuntimeError("PPO TensorDict contains an unexpected observation group")
    if int(ppo_observations["actor"].shape[-1]) != 425:
      raise RuntimeError("registered PPO actor dimension mismatch")
    if int(ppo_observations["critic"].shape[-1]) != 261:
      raise RuntimeError("registered PPO critic dimension mismatch")
    with torch.inference_mode():
      ppo_action = ppo.alg.actor(ppo_observations)
      critic_value = ppo.alg.critic(ppo_observations)
    if not _all_finite((ppo_action, critic_value)):
      raise RuntimeError("PPO actor/critic construction produced non-finite output")

    ppo_reset_count = 0
    for _ in range(steps):
      with torch.inference_mode():
        ppo_action = ppo.alg.actor(ppo_observations)
      ppo_observations, reward, dones, extras = ppo_wrapped.step(ppo_action)
      if variant == "v2":
        term = ppo_env.action_manager.get_term("joint_pos")
        applied_history_match &= bool(torch.equal(term.raw_action, ppo_action))
        low = torch.tensor(schema_module.ACTION_LOW, device=ppo_action.device)
        high = torch.tensor(schema_module.ACTION_HIGH, device=ppo_action.device)
        action_bounds_valid &= bool(
          ((ppo_action >= low) & (ppo_action <= high)).all()
        )
        limits = ppo_env.scene["robot"].data.joint_pos_limits[:, term.target_ids]
        joint_targets_valid &= bool(
          (
            (term._processed_actions >= limits[..., 0])
            & (term._processed_actions <= limits[..., 1])
          ).all()
        )
      action_finite &= _all_finite(ppo_action)
      observation_finite &= _all_finite(ppo_observations)
      reward_finite &= _all_finite((reward, extras))
      ppo_reset_count += int(dones.sum())

    with tempfile.TemporaryDirectory(prefix="go2-proprio-onnx-") as directory:
      ppo.export_policy_to_onnx(directory)
      onnx_path = Path(directory) / "policy.onnx"
      metadata = get_base_metadata(ppo_env, "no-learning-preflight")
      metadata.update(_proprio_metadata(ppo_env, schema_module.__name__))
      attach_metadata_to_onnx(str(onnx_path), metadata)
      model = onnx.load(onnx_path)
      if len(model.graph.input) != 1 or model.graph.input[0].name != "actor":
        raise RuntimeError("ONNX must have exactly one actor input")
      input_dims = [
        dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
      ]
      if input_dims != [1, 425]:
        raise RuntimeError(f"ONNX actor input shape mismatch: {input_dims}")
      metadata_map = {entry.key: entry.value for entry in model.metadata_props}
      if metadata_map.get("observation_schema_sha256") != schema_module.schema_sha256():
        raise RuntimeError("ONNX schema metadata mismatch")
      actor_sample = ppo_observations["actor"][:1].detach().cpu()
      with torch.inference_mode():
        torch_action = ppo.alg.actor(ppo_observations[:1]).cpu().numpy()
      onnx_action = ReferenceEvaluator(model).run(
        None, {"actor": actor_sample.numpy()}
      )[0]
      onnx_max_abs_error = float(np.max(np.abs(torch_action - onnx_action)))
      if onnx_max_abs_error > 1.0e-5:
        raise RuntimeError("PyTorch/ONNX action parity exceeded 1e-5")

    peak_allocated = 0
    peak_reserved = 0
    if device.startswith("cuda"):
      cuda_device = torch.device(device)
      peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
      peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))

    result = {
      "schema_version": 1,
      "task_ids": {"distillation": distill_task, "ppo": ppo_task},
      "teacher_checkpoint": str(checkpoint),
      "teacher_sha256": teacher_sha,
      "student_schema_sha256": schema_module.schema_sha256(),
      "num_envs": num_envs,
      "steps": steps,
      "seed": seed,
      "device": device,
      "dimensions": actual_dims,
      "strict_teacher_load": True,
      "teacher_loaded": True,
      "teacher_frozen": True,
      "teacher_state_key_count": len(teacher_state),
      "teacher_normalizer_count": teacher_normalizer_count,
      "teacher_deterministic_bitwise": True,
      "teacher_state_unchanged": True,
      "teacher_rollout_action_match": teacher_rollout_match,
      "applied_previous_action_match": applied_history_match,
      "action_bounds_valid": action_bounds_valid,
      "joint_targets_valid": joint_targets_valid,
      "behavior_loss": float(behavior_loss.detach()),
      "student_gradient_finite": True,
      "teacher_gradient_absent": True,
      "finite": {
        "observations": observation_finite,
        "actions": action_finite,
        "rewards_and_extras": reward_finite,
        "ppo_actor_critic": True,
        "onnx": True,
      },
      "reset_count": {
        "distillation": distill_reset_count,
        "ppo": ppo_reset_count,
      },
      "ppo_observation_groups": ["actor", "critic"],
      "onnx": {
        "input_count": 1,
        "input_name": "actor",
        "input_shape": [1, 425],
        "max_abs_action_error": onnx_max_abs_error,
        "schema_metadata_match": True,
        "temporary_artifact_removed": True,
      },
      "learn_called": False,
      "optimizer_step_called": False,
      "candidate_checkpoint_written": False,
      "gpu_peak_memory_bytes": {
        "allocated": peak_allocated,
        "reserved": peak_reserved,
      },
    }
    if not all(result["finite"].values()):
      raise RuntimeError("non-finite value observed during no-learning preflight")
    if variant == "v2" and not (
      applied_history_match and action_bounds_valid and joint_targets_valid
    ):
      raise RuntimeError("safe-action V2 applied-action contract failed")
    return result
  finally:
    ppo_env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=V7_CHECKPOINT)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--steps", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--variant", choices=("v1", "v2"), default="v1")
  parser.add_argument("--output-file", type=Path)
  args = parser.parse_args()
  result = run_preflight(
    checkpoint=args.checkpoint,
    num_envs=args.num_envs,
    steps=args.steps,
    seed=args.seed,
    device=args.device,
    variant=args.variant,
  )
  payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
  print(payload)
  if args.output_file is not None:
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
