"""Strict V7-to-contact-force-Teacher iteration-0 transfer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import torch

from src.tasks.velocity.contact_force_teacher_schema import (
  ACTION_DIM,
  CANDIDATE_ACTOR_DIM,
  CONTACT_FORCE_CRITIC_SLICE,
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
from .privileged_teacher_transfer import (
  ALLOW_ENV_STATE_MISMATCH_ENV,
  Go2PrivilegedTeacherTransferRunner,
  map_critic_state_dict,
  reset_matched_rng,
  sha256_file,
)


WORKSPACE = Path(__file__).resolve().parents[4]
SOURCE_CHECKPOINT = WORKSPACE / SOURCE_CHECKPOINT_RELATIVE
SOURCE_MANIFEST_PATH_ENV = "GO2_CONTACT_FORCE_TEACHER_SOURCE_MANIFEST"
SOURCE_MANIFEST_SHA256_ENV = "GO2_CONTACT_FORCE_TEACHER_SOURCE_MANIFEST_SHA256"

_EXPANDED_NORMALIZER_KEYS = {
  "obs_normalizer._mean",
  "obs_normalizer._var",
  "obs_normalizer._std",
}


def active_source_manifest() -> tuple[str, str]:
  path_value = os.environ.get(SOURCE_MANIFEST_PATH_ENV)
  sha_value = os.environ.get(SOURCE_MANIFEST_SHA256_ENV)
  if not path_value or not sha_value:
    raise RuntimeError("contact-force Teacher source manifest is not installed")
  path = Path(path_value).expanduser().resolve()
  if not path.is_file():
    raise RuntimeError(f"contact-force Teacher source manifest is missing: {path}")
  if len(sha_value) != 64 or sha256_file(path) != sha_value:
    raise RuntimeError("contact-force Teacher source manifest SHA256 mismatch")
  return str(path), sha_value


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
    if source_critic[key].shape[-1] != CRITIC_DIM:
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
    raise ValueError(f"{key} does not have the locked 246-D target shape")
  result[..., :SOURCE_ACTOR_DIM].copy_(source.to(result))
  if key in _EXPANDED_NORMALIZER_KEYS:
    start, end = CONTACT_FORCE_CRITIC_SLICE
    result[..., SOURCE_ACTOR_DIM:].copy_(
      source_critic[key][..., start:end].to(result)
    )
  return result


def map_actor_state_dict(
  source: Mapping[str, torch.Tensor],
  target: Mapping[str, torch.Tensor],
  source_critic: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
  """Map V7 actor state into the matched 234-D or appended 246-D actor."""
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


def _arm_from_actor_dim(actor_dim: int) -> str:
  if actor_dim == SOURCE_ACTOR_DIM:
    return "control_234"
  if actor_dim == CANDIDATE_ACTOR_DIM:
    return "candidate_246"
  raise ValueError(f"unexpected contact-force Teacher actor dimension: {actor_dim}")


def _validate_checkpoint_infos(infos: dict, actor_dim: int) -> None:
  arm = _arm_from_actor_dim(actor_dim)
  expected = {
    "contact_force_teacher_schema_version": SCHEMA_VERSION,
    "contact_force_teacher_schema_sha256": schema_sha256(),
    "contact_force_teacher_arm": arm,
    "contact_force_teacher_task_id": TASK_IDS[arm],
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
    raise ValueError(f"contact-force Teacher metadata differs: {mismatches}")
  manifest_path = Path(str(infos.get("source_manifest", ""))).expanduser().resolve()
  manifest_sha = infos.get("source_manifest_sha256")
  if not manifest_path.is_file() or not isinstance(manifest_sha, str):
    raise ValueError("contact-force Teacher source manifest is missing")
  if len(manifest_sha) != 64 or sha256_file(manifest_path) != manifest_sha:
    raise ValueError("contact-force Teacher source manifest differs")


class Go2ContactForceTeacherTransferRunner(Go2PrivilegedTeacherTransferRunner):
  """Transfer locked V7 state into the contact-force single-variable arms."""

  def _transfer_source(self, path: Path, payload: dict) -> dict:
    if sha256_file(path) != SOURCE_CHECKPOINT_SHA256:
      raise ValueError("locked V7 source checkpoint SHA256 mismatch")
    if int(self.cfg["seed"]) != MATCHED_RNG_SEED:
      raise ValueError(
        f"contact-force Teacher transfer seed must be {MATCHED_RNG_SEED}"
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

    actor_state = map_actor_state_dict(
      payload["actor_state_dict"],
      self.alg.actor.state_dict(),
      payload["critic_state_dict"],
    )
    critic_state = map_critic_state_dict(
      payload["critic_state_dict"], self.alg.critic.state_dict()
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
      "contact_force_teacher_schema_version": SCHEMA_VERSION,
      "contact_force_teacher_schema_sha256": schema_sha256(),
      "contact_force_teacher_arm": arm,
      "contact_force_teacher_task_id": TASK_IDS[arm],
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
    _validate_checkpoint_infos(payload.get("infos") or {}, actor_dim)
    infos = super(Go2PrivilegedTeacherTransferRunner, self).load(
      path, load_cfg, strict, map_location
    )
    self._source_transfer_ready = False
    self._transfer_provenance = dict(infos)
    return infos

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
      "contact_force_teacher_schema_version": SCHEMA_VERSION,
      "contact_force_teacher_schema_sha256": schema_sha256(),
      "contact_force_teacher_arm": arm,
      "contact_force_teacher_task_id": TASK_IDS[arm],
      "intervention": INTERVENTION,
      "checkpoint_labels_are_completed_updates": True,
      **getattr(self, "_transfer_provenance", {}),
    }
    # Skip the old V8 metadata layer and call the shared runner save directly.
    super(Go2PrivilegedTeacherTransferRunner, self).save(
      path, infos={**(infos or {}), **transfer_infos}
    )
