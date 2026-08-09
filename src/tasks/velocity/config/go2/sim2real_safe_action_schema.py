"""Canonical Go2 proprioceptive V2 bounded applied-action schema."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .env_cfgs import PROPRIO_HISTORY_LENGTH
from .sim2real_schema import (
  ACTION_SCALE,
  CONTROL_DT_S,
  DEFAULT_JOINT_POS,
  JOINT_DAMPING,
  JOINT_EFFORT_LIMIT,
  JOINT_NAMES,
  JOINT_POS_LIMITS,
  JOINT_STIFFNESS,
  SDK_JOINT_IDS_MAP,
  STUDENT_TERMS,
  actor_dim,
)


SCHEMA_VERSION = "go2-sim2real-proprio-v2-safe-action"
ACTION_INTERFACE = "bounded_asymmetric_per_joint_v2"
ACTION_OUTPUT_SEMANTICS = "applied_normalized_action"
ACTION_ABS_LIMIT = 4.0
ACTION_MEAN_BOUND = 5.0


ACTION_LOW = tuple(
  max(-ACTION_ABS_LIMIT, (lower - q0) / ACTION_SCALE)
  for q0, (lower, _upper) in zip(DEFAULT_JOINT_POS, JOINT_POS_LIMITS, strict=True)
)
ACTION_HIGH = tuple(
  min(ACTION_ABS_LIMIT, (upper - q0) / ACTION_SCALE)
  for q0, (_lower, upper) in zip(DEFAULT_JOINT_POS, JOINT_POS_LIMITS, strict=True)
)


def _terms() -> list[dict[str, Any]]:
  result = []
  start = 0
  for term in STUDENT_TERMS:
    end = start + term.flattened_dim
    result.append(
      asdict(term)
      | {
        "flattened_dim": term.flattened_dim,
        "index_start": start,
        "index_end_exclusive": end,
        "time_order": "oldest-to-newest" if term.history_length > 1 else "current",
      }
    )
    start = end
  return result


def schema_payload() -> dict[str, Any]:
  return {
    "version": SCHEMA_VERSION,
    "control_dt_s": CONTROL_DT_S,
    "history_order": "term-major, oldest-to-newest",
    "history_reset": "first finite frame backfills all 10 history slots",
    "history_sample_count": PROPRIO_HISTORY_LENGTH,
    "history_endpoint_span_s": (PROPRIO_HISTORY_LENGTH - 1) * CONTROL_DT_S,
    "actor_dim": actor_dim(),
    "terms": _terms(),
    "phase": {
      "period_s": 0.6,
      "stand_command_norm_threshold": 0.1,
      "formula": "phase=(policy_tick*control_dt_s/period_s)%1",
      "reset_policy_tick": 0,
      "getter_has_side_effects": False,
    },
    "previous_action": {
      "timing": "observation t contains applied normalized action from t-1",
      "value": "a_applied=T(z)",
      "reset": "zeros",
    },
    "action": {
      "dim": len(JOINT_NAMES),
      "interface": ACTION_INTERFACE,
      "output_semantics": ACTION_OUTPUT_SEMANTICS,
      "latent": "z",
      "gaussian_mean": "m=5*tanh(m_raw/5)",
      "gaussian_mean_bound": ACTION_MEAN_BOUND,
      "squashed": "u=tanh(z)",
      "mapping": {
        "nonnegative": "a_applied=u*a_high",
        "negative": "a_applied=(-u)*a_low",
      },
      "zero_preservation": "z=0 -> u=0 -> a_applied=0 -> q_target=q0",
      "no_deployment_only_clipping": True,
      "scale": ACTION_SCALE,
      "absolute_normalized_limit": ACTION_ABS_LIMIT,
      "a_low": list(ACTION_LOW),
      "a_high": list(ACTION_HIGH),
      "joint_names": list(JOINT_NAMES),
      "sdk_joint_ids_map": list(SDK_JOINT_IDS_MAP),
      "default_joint_pos": list(DEFAULT_JOINT_POS),
      "stiffness": list(JOINT_STIFFNESS),
      "damping": list(JOINT_DAMPING),
      "effort_limit_nm": list(JOINT_EFFORT_LIMIT),
      "joint_position_limits_rad": [list(limits) for limits in JOINT_POS_LIMITS],
      "target_formula": "q_target=q0+0.25*a_applied",
    },
    "onnx": {
      "input_name": "actor",
      "input_shape": [1, actor_dim()],
      "output_name": "actions",
      "output_shape": [1, len(JOINT_NAMES)],
      "output_semantics": ACTION_OUTPUT_SEMANTICS,
      "dtype": "float32",
      "static_shapes": True,
    },
    "runtime": {
      "low_state_timeout": "SDK host-monotonic isTimeout -> Passive",
      "nonfinite_observation": "Passive",
      "nonfinite_latent": "Passive",
      "nonfinite_action": "Passive",
      "action_limit_violation": "Passive",
      "processed_joint_target_limit_violation": "Passive",
      "hardware_tick_crc_semantics": "HARDWARE_PENDING",
    },
  }


def schema_sha256() -> str:
  canonical = json.dumps(
    schema_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("ascii")
  return hashlib.sha256(canonical).hexdigest()


def latent_to_applied_action(latent: Any) -> tuple[float, ...]:
  values = tuple(float(value) for value in latent)
  if len(values) != 12:
    raise ValueError("latent action must have 12 values")
  if not all(math.isfinite(value) for value in values):
    raise ValueError("latent action contains NaN/Inf")
  result = []
  for value, low, high in zip(values, ACTION_LOW, ACTION_HIGH, strict=True):
    unit = math.tanh(value)
    result.append(unit * high if unit >= 0.0 else (-unit) * low)
  return tuple(result)


def applied_action_to_sdk_targets(action: Any) -> tuple[float, ...]:
  values = tuple(float(value) for value in action)
  if len(values) != 12:
    raise ValueError("applied action must have 12 values")
  if not all(math.isfinite(value) for value in values):
    raise ValueError("applied action contains NaN/Inf")
  for value, low, high in zip(values, ACTION_LOW, ACTION_HIGH, strict=True):
    if value < low or value > high:
      raise ValueError("applied action is outside registered bounds")
  targets = [0.0] * 12
  for index, sdk_id in enumerate(SDK_JOINT_IDS_MAP):
    target = DEFAULT_JOINT_POS[index] + ACTION_SCALE * values[index]
    lower, upper = JOINT_POS_LIMITS[index]
    if not lower <= target <= upper:
      raise ValueError("applied action produces out-of-range target")
    targets[sdk_id] = target
  return tuple(targets)


def validate_deploy_yaml(path: Path) -> None:
  payload = yaml.safe_load(path.read_text(encoding="utf-8"))
  if tuple(payload["joint_ids_map"]) != SDK_JOINT_IDS_MAP:
    raise ValueError("deploy SDK joint mapping differs from V2 schema")
  if float(payload["step_dt"]) != CONTROL_DT_S:
    raise ValueError("deploy control period differs from V2 schema")
  action = payload["actions"]["JointPositionAction"]
  if action.get("clip", "missing") is not None:
    raise ValueError("V2 deploy action clip must be null")
  if tuple(float(value) for value in action["scale"]) != (ACTION_SCALE,) * 12:
    raise ValueError("deploy action scale mismatch")
  if tuple(float(value) for value in action["offset"]) != DEFAULT_JOINT_POS:
    raise ValueError("deploy action offset mismatch")
  observations = payload["observations"]["actor"]
  if list(observations) != [term.deploy_name for term in STUDENT_TERMS]:
    raise ValueError("deploy observation ordering differs from V2 schema")
  for term in STUDENT_TERMS:
    if int(observations[term.deploy_name]["history_length"]) != term.history_length:
      raise ValueError(f"history mismatch for {term.deploy_name}")
  schema = payload["schema"]
  if schema.get("version") != SCHEMA_VERSION:
    raise ValueError("deploy schema version mismatch")
  if schema.get("sha256") != schema_sha256():
    raise ValueError("deploy schema SHA256 mismatch")
  if schema.get("action_interface") != ACTION_INTERFACE:
    raise ValueError("deploy action interface mismatch")
  if schema.get("action_output_semantics") != ACTION_OUTPUT_SEMANTICS:
    raise ValueError("deploy action output semantics mismatch")
  if float(schema.get("action_mean_bound", float("nan"))) != ACTION_MEAN_BOUND:
    raise ValueError("deploy action mean bound mismatch")
  if tuple(float(value) for value in schema["a_low"]) != ACTION_LOW:
    raise ValueError("deploy asymmetric lower bounds mismatch")
  if tuple(float(value) for value in schema["a_high"]) != ACTION_HIGH:
    raise ValueError("deploy asymmetric upper bounds mismatch")
  if payload.get("previous_action", {}).get("semantics") != ACTION_OUTPUT_SEMANTICS:
    raise ValueError("deploy previous-action semantics mismatch")
