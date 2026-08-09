"""Frozen schema for the reference-backed Go2 contact-force Teacher probe."""

from __future__ import annotations

import hashlib
import json
from typing import Final


SCHEMA_VERSION: Final = "go2_privileged_contact_force_teacher_v1"
REFERENCE_PAPER: Final = (
  "Joonho Lee et al., Learning Quadrupedal Locomotion over Challenging "
  "Terrain, Science Robotics 5(47), 2020"
)
REFERENCE_URL: Final = "https://arxiv.org/abs/2010.11251"
REFERENCE_DOI: Final = "10.1126/scirobotics.abc5986"
REFERENCE_COMPONENT: Final = (
  "the privileged Teacher receives foot-ground contact forces"
)
REFERENCE_DEVIATIONS: Final = (
  "Go2 instead of ANYmal",
  "MJLab/MuJoCo instead of RaiSim",
  "PPO instead of TRPO",
  "12 joint-position targets instead of leg-frequency and foot-residual actions",
  "contact-force-only causal ablation instead of the paper's full privileged block",
  "direct MLP append instead of a separate privileged encoder",
  "world-frame net force with signed-log1p preprocessing",
)

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
CANDIDATE_ACTOR_DIM: Final = 246
CRITIC_DIM: Final = 261
ACTION_DIM: Final = 12
CONTACT_FORCE_ACTOR_SLICE: Final = (234, 246)
CONTACT_FORCE_CRITIC_SLICE: Final = (249, 261)
CONTACT_FORCE_RAW_UNIT: Final = "N"
CONTACT_FORCE_FRAME: Final = "world"
CONTACT_FORCE_PREPROCESSING: Final = "sign(force) * log1p(abs(force))"
CONTACT_FORCE_TIMING: Final = "current pre-action observation"
CONTACT_FORCE_SENSOR_REDUCTION: Final = "netforce"
CONTACT_FORCE_NATIVE_ORDER: Final = ("FL", "FR", "RL", "RR")
NORMALIZER_SOURCE: Final = "source_critic[249:261]"
INTERVENTION: Final = "append_actor_foot_contact_forces_only"

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
  ("foot_contact_forces", 12),
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
  "control_234": "Unitree-Go2-Rough-ContactForceTeacher-V1-Control",
  "candidate_246": "Unitree-Go2-Rough-ContactForceTeacher-V1",
}


def term_slices(
  terms: tuple[tuple[str, int], ...],
) -> dict[str, tuple[int, int]]:
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
    "reference": {
      "paper": REFERENCE_PAPER,
      "url": REFERENCE_URL,
      "doi": REFERENCE_DOI,
      "component": REFERENCE_COMPONENT,
      "deviations": REFERENCE_DEVIATIONS,
    },
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
    "contact_force_actor_slice": CONTACT_FORCE_ACTOR_SLICE,
    "contact_force_critic_slice": CONTACT_FORCE_CRITIC_SLICE,
    "contact_force_raw_unit": CONTACT_FORCE_RAW_UNIT,
    "contact_force_frame": CONTACT_FORCE_FRAME,
    "contact_force_preprocessing": CONTACT_FORCE_PREPROCESSING,
    "contact_force_timing": CONTACT_FORCE_TIMING,
    "contact_force_sensor_reduction": CONTACT_FORCE_SENSOR_REDUCTION,
    "contact_force_native_order": CONTACT_FORCE_NATIVE_ORDER,
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
  terms = CANDIDATE_ACTOR_TERM_DIMS if candidate else V7_ACTOR_TERM_DIMS
  expected = tuple(name for name, _ in terms)
  if names != expected:
    raise ValueError(f"contact-force actor term order differs: {names} != {expected}")


def validate_critic_term_order(names: tuple[str, ...]) -> None:
  expected = tuple(name for name, _ in CRITIC_TERM_DIMS)
  if names != expected:
    raise ValueError(f"contact-force critic term order differs: {names} != {expected}")


assert sum(width for _, width in V7_ACTOR_TERM_DIMS) == SOURCE_ACTOR_DIM
assert sum(width for _, width in CANDIDATE_ACTOR_TERM_DIMS) == CANDIDATE_ACTOR_DIM
assert sum(width for _, width in CRITIC_TERM_DIMS) == CRITIC_DIM
assert CANDIDATE_ACTOR_TERM_SLICES["foot_contact_forces"] == CONTACT_FORCE_ACTOR_SLICE
assert CRITIC_TERM_SLICES["foot_contact_forces"] == CONTACT_FORCE_CRITIC_SLICE
