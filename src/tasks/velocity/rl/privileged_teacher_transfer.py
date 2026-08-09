"""Strict V7-to-V8 iteration-0 transfer for the privileged teacher probe."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.tasks.velocity.privileged_teacher_schema import (
  ACTION_DIM,
  BASE_LIN_VEL_SLICE,
  CANDIDATE_ACTOR_DIM,
  CRITIC_DIM,
  INTERVENTION,
  MATCHED_RNG_SEED,
  NORMALIZER_SOURCE,
  SCHEMA_VERSION,
  SOURCE_ACTOR_DIM,
  SOURCE_CHECKPOINT_RELATIVE,
  SOURCE_CHECKPOINT_SHA256,
  SOURCE_COMMON_STEP_COUNTER,
  SOURCE_ENVIRONMENT_NUM_ENVS,
  SOURCE_ENVIRONMENT_STATE_SHA256,
  TASK_IDS,
  schema_sha256,
)
from .runner import VelocityOnPolicyRunner


WORKSPACE = Path(__file__).resolve().parents[4]
SOURCE_CHECKPOINT = WORKSPACE / SOURCE_CHECKPOINT_RELATIVE
ALLOW_ENV_STATE_MISMATCH_ENV = "GO2_PRIV_TEACHER_ALLOW_ENV_STATE_MISMATCH"
SOURCE_MANIFEST_PATH_ENV = "GO2_PRIV_TEACHER_SOURCE_MANIFEST"
SOURCE_MANIFEST_SHA256_ENV = "GO2_PRIV_TEACHER_SOURCE_MANIFEST_SHA256"

_EXPANDED_NORMALIZER_KEYS = {
  "obs_normalizer._mean",
  "obs_normalizer._var",
  "obs_normalizer._std",
}


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def active_source_manifest() -> tuple[str, str]:
  path_value = os.environ.get(SOURCE_MANIFEST_PATH_ENV)
  sha_value = os.environ.get(SOURCE_MANIFEST_SHA256_ENV)
  if not path_value or not sha_value:
    raise RuntimeError("privileged teacher source manifest is not installed")
  path = Path(path_value).expanduser().resolve()
  if not path.is_file():
    raise RuntimeError(f"privileged teacher source manifest is missing: {path}")
  if len(sha_value) != 64 or sha256_file(path) != sha_value:
    raise RuntimeError("privileged teacher source manifest SHA256 mismatch")
  return str(path), sha_value


def reset_matched_rng(seed: int) -> None:
  """Reset global policy/environment streams after differently sized builds."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def environment_state_sha256(state: Mapping[str, object]) -> str:
  digest = hashlib.sha256()
  digest.update(str(int(state["common_step_counter"])).encode("ascii"))
  for key in ("terrain_levels", "terrain_types"):
    value = state[key]
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"environment state {key} is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def _expanded_actor_tensor(
  key: str,
  source: torch.Tensor,
  target: torch.Tensor,
  source_critic: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
  if key == "mlp.0.weight":
    result = torch.zeros_like(target)
  elif key in _EXPANDED_NORMALIZER_KEYS:
    if source_critic is None or key not in source_critic:
      raise ValueError(f"candidate mapping requires critic statistics for {key}")
    critic_value = source_critic[key]
    if critic_value.shape[-1] != CRITIC_DIM:
      raise ValueError(f"source critic {key} is not {CRITIC_DIM}-D")
    result = torch.empty_like(target)
  else:
    raise ValueError(f"unexpected resized actor tensor: {key}")
  if source.ndim != target.ndim or source.shape[:-1] != target.shape[:-1]:
    raise ValueError(
      f"invalid expansion for {key}: source={tuple(source.shape)}, "
      f"target={tuple(target.shape)}"
    )
  if source.shape[-1] != SOURCE_ACTOR_DIM:
    raise ValueError(f"{key} does not have the locked 234-D source shape")
  if target.shape[-1] != CANDIDATE_ACTOR_DIM:
    raise ValueError(f"{key} does not have the locked 237-D target shape")
  result[..., :SOURCE_ACTOR_DIM].copy_(source.to(result))
  if key in _EXPANDED_NORMALIZER_KEYS:
    result[..., SOURCE_ACTOR_DIM:].copy_(
      source_critic[key][..., SOURCE_ACTOR_DIM:CANDIDATE_ACTOR_DIM].to(result)
    )
  return result


def map_actor_state_dict(
  source: Mapping[str, torch.Tensor],
  target: Mapping[str, torch.Tensor],
  source_critic: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
  """Map V7 actor state into the matched 234-D or appended 237-D actor."""
  if source.keys() != target.keys():
    missing = sorted(set(target).difference(source))
    extra = sorted(set(source).difference(target))
    raise ValueError(f"actor state keys differ: missing={missing}, extra={extra}")
  mapped: dict[str, torch.Tensor] = {}
  for key, target_value in target.items():
    source_value = source[key]
    if source_value.shape == target_value.shape:
      mapped[key] = source_value.detach().clone().to(target_value)
    else:
      mapped[key] = _expanded_actor_tensor(
        key, source_value, target_value, source_critic
      )
  return mapped


def map_critic_state_dict(
  source: Mapping[str, torch.Tensor],
  target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Require and copy the unchanged 261-D critic exactly."""
  if source.keys() != target.keys():
    raise ValueError("critic state keys differ")
  mapped: dict[str, torch.Tensor] = {}
  for key, target_value in target.items():
    source_value = source[key]
    if source_value.shape != target_value.shape:
      raise ValueError(
        f"critic tensor shape differs for {key}: "
        f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
      )
    mapped[key] = source_value.detach().clone().to(target_value)
  return mapped


def _arm_from_actor_dim(actor_dim: int) -> str:
  if actor_dim == SOURCE_ACTOR_DIM:
    return "control_234"
  if actor_dim == CANDIDATE_ACTOR_DIM:
    return "candidate_237"
  raise ValueError(f"unexpected privileged teacher actor dimension: {actor_dim}")


def _validate_probe_checkpoint_infos(infos: dict, actor_dim: int) -> None:
  arm = _arm_from_actor_dim(actor_dim)
  expected = {
    "privileged_teacher_schema_version": SCHEMA_VERSION,
    "privileged_teacher_schema_sha256": schema_sha256(),
    "privileged_teacher_arm": arm,
    "privileged_teacher_task_id": TASK_IDS[arm],
    "transfer_source_sha256": SOURCE_CHECKPOINT_SHA256,
    "transfer_mode": "v7_iteration_0_fresh_optimizer",
    "source_actor_dim": SOURCE_ACTOR_DIM,
    "target_actor_dim": actor_dim,
    "critic_dim": CRITIC_DIM,
    "action_dim": ACTION_DIM,
    "intervention": INTERVENTION,
    "normalizer_source": NORMALIZER_SOURCE,
    "optimizer_restored": False,
    "iteration_restored": False,
    "environment_state_restored_exact": True,
    "technical_smoke_env_state_override": False,
    "source_environment_num_envs": SOURCE_ENVIRONMENT_NUM_ENVS,
    "source_environment_state_sha256": SOURCE_ENVIRONMENT_STATE_SHA256,
    "source_common_step_counter": SOURCE_COMMON_STEP_COUNTER,
    "rng_reset_after_transfer": True,
    "rng_seed": MATCHED_RNG_SEED,
    "checkpoint_labels_are_completed_updates": True,
    "new_actor_columns_zero_initialized": actor_dim == CANDIDATE_ACTOR_DIM,
  }
  mismatches = {
    key: (infos.get(key), value)
    for key, value in expected.items()
    if infos.get(key) != value
  }
  if mismatches:
    raise ValueError(f"privileged teacher checkpoint metadata differs: {mismatches}")
  manifest_path = Path(str(infos.get("source_manifest", ""))).expanduser().resolve()
  manifest_sha = infos.get("source_manifest_sha256")
  if not manifest_path.is_file() or not isinstance(manifest_sha, str):
    raise ValueError("privileged teacher checkpoint source manifest is missing")
  if len(manifest_sha) != 64 or sha256_file(manifest_path) != manifest_sha:
    raise ValueError("privileged teacher checkpoint source manifest differs")


class Go2PrivilegedTeacherTransferRunner(VelocityOnPolicyRunner):
  """Transfer locked V7 models/env state with a fresh iteration-0 optimizer."""

  def _restore_locked_environment_state(self, state: dict) -> dict[str, object]:
    required = {"common_step_counter", "terrain_levels", "terrain_types"}
    missing = sorted(required.difference(state))
    if missing:
      raise ValueError(f"locked V7 environment state is incomplete: {missing}")
    saved_levels = state["terrain_levels"]
    saved_types = state["terrain_types"]
    if saved_levels.shape != saved_types.shape:
      raise ValueError("locked V7 terrain level/type shapes differ")
    saved_num_envs = int(saved_levels.numel())
    if saved_num_envs != SOURCE_ENVIRONMENT_NUM_ENVS:
      raise ValueError(
        f"locked V7 environment state has {saved_num_envs} envs, expected "
        f"{SOURCE_ENVIRONMENT_NUM_ENVS}"
      )
    state_sha = environment_state_sha256(state)
    if state_sha != SOURCE_ENVIRONMENT_STATE_SHA256:
      raise ValueError("locked V7 environment state SHA256 mismatch")
    if int(state["common_step_counter"]) != SOURCE_COMMON_STEP_COUNTER:
      raise ValueError("locked V7 common step counter differs")
    self._restore_environment_state(state)

    env = self.env.unwrapped
    exact = saved_num_envs == int(env.num_envs)
    allow_mismatch = os.environ.get(ALLOW_ENV_STATE_MISMATCH_ENV) == "1"
    if not exact and not allow_mismatch:
      raise ValueError(
        "locked V7 environment state requires exactly "
        f"{saved_num_envs} envs; got {env.num_envs}. Set "
        f"{ALLOW_ENV_STATE_MISMATCH_ENV}=1 only for a technical smoke test."
      )
    if exact:
      terrain = env.scene.terrain
      if terrain is None:
        raise RuntimeError("locked V7 terrain state cannot be restored")
      expected_levels = saved_levels.to(
        terrain.terrain_levels.device, dtype=terrain.terrain_levels.dtype
      )
      expected_types = saved_types.to(
        terrain.terrain_types.device, dtype=terrain.terrain_types.dtype
      )
      if not torch.equal(terrain.terrain_levels, expected_levels):
        raise RuntimeError("locked V7 terrain levels were not restored exactly")
      if not torch.equal(terrain.terrain_types, expected_types):
        raise RuntimeError("locked V7 terrain types were not restored exactly")
      if env.common_step_counter != state["common_step_counter"]:
        raise RuntimeError("locked V7 common step counter was not restored")
    return {
      "source_environment_num_envs": saved_num_envs,
      "environment_state_restored_exact": exact,
      "technical_smoke_env_state_override": not exact and allow_mismatch,
      "source_environment_state_sha256": state_sha,
      "source_common_step_counter": int(state["common_step_counter"]),
      "source_terrain_level_mean": float(saved_levels.float().mean()),
    }

  def _transfer_source(self, path: Path, payload: dict) -> dict:
    if sha256_file(path) != SOURCE_CHECKPOINT_SHA256:
      raise ValueError("locked V7 source checkpoint SHA256 mismatch")
    if int(self.cfg["seed"]) != MATCHED_RNG_SEED:
      raise ValueError(
        f"privileged teacher transfer seed must be {MATCHED_RNG_SEED}"
      )
    source_manifest, source_manifest_sha256 = active_source_manifest()
    actor_count = payload["actor_state_dict"]["obs_normalizer.count"]
    critic_count = payload["critic_state_dict"]["obs_normalizer.count"]
    if not torch.equal(actor_count, critic_count):
      raise ValueError("source actor/critic normalizer counts differ")
    source_infos = payload.get("infos") or {}
    environment_provenance = self._restore_locked_environment_state(
      source_infos.get("env_state") or {}
    )

    actor = self.alg.actor
    critic = self.alg.critic
    actor_state = map_actor_state_dict(
      payload["actor_state_dict"],
      actor.state_dict(),
      payload["critic_state_dict"],
    )
    critic_state = map_critic_state_dict(
      payload["critic_state_dict"], critic.state_dict()
    )
    actor_dim = int(actor_state["mlp.0.weight"].shape[-1])
    critic_dim = int(critic_state["mlp.0.weight"].shape[-1])
    action_dim = int(actor_state["mlp.6.weight"].shape[0])
    if actor_dim not in (SOURCE_ACTOR_DIM, CANDIDATE_ACTOR_DIM):
      raise ValueError(f"unexpected target actor dimension: {actor_dim}")
    if critic_dim != CRITIC_DIM or action_dim != ACTION_DIM:
      raise ValueError(
        f"unexpected critic/action dimensions: {critic_dim}/{action_dim}"
      )

    self.alg.load(
      {
        "actor_state_dict": actor_state,
        "critic_state_dict": critic_state,
      },
      {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
      strict=True,
    )
    if self.current_learning_iteration != 0:
      raise RuntimeError("iteration-0 transfer unexpectedly changed iteration")
    reset_matched_rng(MATCHED_RNG_SEED)
    arm = _arm_from_actor_dim(actor_dim)
    self._source_transfer_ready = True
    self._transfer_provenance = {
      "transfer_mode": "v7_iteration_0_fresh_optimizer",
      "transfer_source": str(path),
      "transfer_source_sha256": SOURCE_CHECKPOINT_SHA256,
      "source_actor_dim": SOURCE_ACTOR_DIM,
      "target_actor_dim": actor_dim,
      "critic_dim": critic_dim,
      "action_dim": action_dim,
      "optimizer_restored": False,
      "iteration_restored": False,
      "rng_reset_after_transfer": True,
      "rng_seed": MATCHED_RNG_SEED,
      "new_actor_columns_zero_initialized": actor_dim == CANDIDATE_ACTOR_DIM,
      "normalizer_source": NORMALIZER_SOURCE,
      "privileged_teacher_schema_version": SCHEMA_VERSION,
      "privileged_teacher_schema_sha256": schema_sha256(),
      "privileged_teacher_arm": arm,
      "privileged_teacher_task_id": TASK_IDS[arm],
      "intervention": INTERVENTION,
      "source_manifest": source_manifest,
      "source_manifest_sha256": source_manifest_sha256,
      "checkpoint_labels_are_completed_updates": True,
      **environment_provenance,
    }
    return dict(self._transfer_provenance)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint == SOURCE_CHECKPOINT.resolve():
      payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
      return self._transfer_source(checkpoint, payload)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_dim = int(self.alg.actor.state_dict()["mlp.0.weight"].shape[-1])
    _validate_probe_checkpoint_infos(payload.get("infos") or {}, actor_dim)
    infos = super().load(path, load_cfg, strict, map_location)
    self._source_transfer_ready = False
    self._transfer_provenance = dict(infos)
    return infos

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if getattr(self, "_source_transfer_ready", False):
      if self.current_learning_iteration != 0:
        raise RuntimeError("source transfer must begin learning at iteration zero")
      # RSL names checkpoints by the zero-based loop index. Starting at one
      # makes model_N mean exactly N completed updates for this probe.
      self.current_learning_iteration = 1
      self._source_transfer_ready = False
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None):
    actor_dim = int(self.alg.actor.state_dict()["mlp.0.weight"].shape[-1])
    arm = _arm_from_actor_dim(actor_dim)
    transfer_infos = {
      "transfer_mode": "v7_iteration_0_fresh_optimizer",
      "transfer_source": str(SOURCE_CHECKPOINT.resolve()),
      "transfer_source_sha256": SOURCE_CHECKPOINT_SHA256,
      "source_actor_dim": SOURCE_ACTOR_DIM,
      "target_actor_dim": actor_dim,
      "critic_dim": CRITIC_DIM,
      "action_dim": ACTION_DIM,
      "optimizer_restored": False,
      "iteration_restored": False,
      "rng_reset_after_transfer": True,
      "rng_seed": MATCHED_RNG_SEED,
      "normalizer_source": NORMALIZER_SOURCE,
      "privileged_teacher_schema_version": SCHEMA_VERSION,
      "privileged_teacher_schema_sha256": schema_sha256(),
      "privileged_teacher_arm": arm,
      "privileged_teacher_task_id": TASK_IDS[arm],
      "intervention": INTERVENTION,
      "checkpoint_labels_are_completed_updates": True,
      **getattr(self, "_transfer_provenance", {}),
    }
    super().save(path, infos={**(infos or {}), **transfer_infos})
