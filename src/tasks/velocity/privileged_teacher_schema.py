"""Frozen observation and transfer schema for the Go2 V8 teacher probe."""

from __future__ import annotations

import hashlib
import json
from typing import Final


SCHEMA_VERSION: Final = "go2_privileged_lin_vel_teacher_v1"
SOURCE_CHECKPOINT_RELATIVE: Final = (
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_"
  "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt"
)
SOURCE_CHECKPOINT_SHA256: Final = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)
SOURCE_ENVIRONMENT_STATE_SHA256: Final = (
  "3153e9be7fc005a7f90bdaf7daf8ee0df9e86d8adc3840b6069f0400d722175b"
)
SOURCE_ENVIRONMENT_NUM_ENVS: Final = 2048
SOURCE_COMMON_STEP_COUNTER: Final = 326664
MATCHED_RNG_SEED: Final = 42
SOURCE_ACTOR_DIM: Final = 234
CANDIDATE_ACTOR_DIM: Final = 237
CRITIC_DIM: Final = 261
ACTION_DIM: Final = 12
BASE_LIN_VEL_SLICE: Final = (234, 237)
BASE_LIN_VEL_UNIT: Final = "m/s"
BASE_LIN_VEL_FRAME: Final = "imu_site_local_body_aligned"
NORMALIZER_SOURCE: Final = "source_critic[234:237]"
INTERVENTION: Final = "append_actor_base_lin_vel_only"

V7_ACTOR_TERM_DIMS: Final = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("command", 3),
  ("phase", 2),
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("actions", 12),
  ("height_scan", 187),
)
CANDIDATE_ACTOR_TERM_DIMS: Final = (
  *V7_ACTOR_TERM_DIMS,
  ("base_lin_vel", 3),
)
CRITIC_TERM_DIMS: Final = (
  *V7_ACTOR_TERM_DIMS,
  ("base_lin_vel", 3),
  ("foot_height", 4),
  ("foot_air_time", 4),
  ("foot_contact", 4),
  ("foot_contact_forces", 12),
)
TASK_IDS: Final = {
  "control_234": "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher-Control",
  "candidate_237": "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher",
}


def term_slices(terms: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
  result: dict[str, tuple[int, int]] = {}
  offset = 0
  for name, width in terms:
    result[name] = (offset, offset + width)
    offset += width
  return result


V7_ACTOR_TERM_SLICES: Final = term_slices(V7_ACTOR_TERM_DIMS)
CANDIDATE_ACTOR_TERM_SLICES: Final = term_slices(CANDIDATE_ACTOR_TERM_DIMS)
CRITIC_TERM_SLICES: Final = term_slices(CRITIC_TERM_DIMS)


def schema_payload() -> dict[str, object]:
  return {
    "version": SCHEMA_VERSION,
    "source_checkpoint": SOURCE_CHECKPOINT_RELATIVE,
    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    "source_environment_state_sha256": SOURCE_ENVIRONMENT_STATE_SHA256,
    "source_environment_num_envs": SOURCE_ENVIRONMENT_NUM_ENVS,
    "source_common_step_counter": SOURCE_COMMON_STEP_COUNTER,
    "matched_rng_seed": MATCHED_RNG_SEED,
    "source_actor_dim": SOURCE_ACTOR_DIM,
    "candidate_actor_dim": CANDIDATE_ACTOR_DIM,
    "critic_dim": CRITIC_DIM,
    "action_dim": ACTION_DIM,
    "v7_actor_terms": V7_ACTOR_TERM_DIMS,
    "candidate_actor_terms": CANDIDATE_ACTOR_TERM_DIMS,
    "critic_terms": CRITIC_TERM_DIMS,
    "base_lin_vel_slice": BASE_LIN_VEL_SLICE,
    "base_lin_vel_unit": BASE_LIN_VEL_UNIT,
    "base_lin_vel_frame": BASE_LIN_VEL_FRAME,
    "normalizer_source": NORMALIZER_SOURCE,
    "intervention": INTERVENTION,
    "task_ids": TASK_IDS,
  }


def schema_sha256() -> str:
  encoded = json.dumps(
    schema_payload(), sort_keys=True, separators=(",", ":")
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def validate_actor_term_order(names: tuple[str, ...], *, candidate: bool) -> None:
  expected_terms = CANDIDATE_ACTOR_TERM_DIMS if candidate else V7_ACTOR_TERM_DIMS
  expected_names = tuple(name for name, _ in expected_terms)
  if names != expected_names:
    raise ValueError(
      f"privileged teacher actor term order differs: {names} != {expected_names}"
    )


def validate_critic_term_order(names: tuple[str, ...]) -> None:
  expected_names = tuple(name for name, _ in CRITIC_TERM_DIMS)
  if names != expected_names:
    raise ValueError(
      f"privileged teacher critic term order differs: {names} != {expected_names}"
    )


assert sum(width for _, width in V7_ACTOR_TERM_DIMS) == SOURCE_ACTOR_DIM
assert sum(width for _, width in CANDIDATE_ACTOR_TERM_DIMS) == CANDIDATE_ACTOR_DIM
assert sum(width for _, width in CRITIC_TERM_DIMS) == CRITIC_DIM
assert CANDIDATE_ACTOR_TERM_SLICES["base_lin_vel"] == BASE_LIN_VEL_SLICE
assert CRITIC_TERM_SLICES["base_lin_vel"] == BASE_LIN_VEL_SLICE
