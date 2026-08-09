"""Frozen observation/action schema for the Go2 proprioceptive student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import math
from typing import Any

import yaml

from .env_cfgs import PROPRIO_HISTORY_LENGTH


SCHEMA_VERSION = "go2-sim2real-proprio-v1"
CONTROL_DT_S = 0.02
ACTION_SCALE = 0.25
ACTION_ABS_LIMIT = 4.0
PHASE_PERIOD_S = 0.6
STAND_COMMAND_NORM_THRESHOLD = 0.1
DEFAULT_JOINT_POS = (-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8)
JOINT_STIFFNESS = (20.0, 20.0, 40.0) * 4
JOINT_DAMPING = (1.0, 1.0, 2.0) * 4
JOINT_EFFORT_LIMIT = (23.5, 23.5, 45.0) * 4
JOINT_POS_LIMITS = (
  (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),
  (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),
  (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776),
  (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776),
)
JOINT_NAMES = (
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
SDK_JOINT_IDS_MAP = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


@dataclass(frozen=True)
class ObservationTermSchema:
  training_name: str
  deploy_name: str
  dim: int
  history_length: int
  source: str
  unit: str
  frame: str
  sign: str
  scale: tuple[float, ...]

  @property
  def flattened_dim(self) -> int:
    return self.dim * self.history_length


STUDENT_TERMS = (
  ObservationTermSchema("base_ang_vel", "base_ang_vel", 3, PROPRIO_HISTORY_LENGTH, "sdk_imu.gyroscope", "rad/s", "base", "sdk-native xyz", (1.0,) * 3),
  ObservationTermSchema("projected_gravity", "projected_gravity", 3, PROPRIO_HISTORY_LENGTH, "sdk_imu.quaternion", "unitless", "base", "q[wxyz] conjugate rotate world [0,0,-1]", (1.0,) * 3),
  ObservationTermSchema("command", "velocity_commands", 3, 1, "external_command", "m/s,m/s,rad/s", "base-yaw", "vx=ly, vy=-lx, yaw=-rx", (1.0,) * 3),
  ObservationTermSchema("phase", "gait_phase", 2, 1, "policy_tick", "unitless", "control-clock", "sin,cos; zero while command norm<0.1", (1.0,) * 2),
  ObservationTermSchema("joint_pos", "joint_pos_rel", 12, PROPRIO_HISTORY_LENGTH, "sdk_motor_state.q", "rad", "joint-local", "training joint order; q-default_q", (1.0,) * 12),
  ObservationTermSchema("joint_vel", "joint_vel_rel", 12, PROPRIO_HISTORY_LENGTH, "sdk_motor_state.dq", "rad/s", "joint-local", "training joint order", (1.0,) * 12),
  ObservationTermSchema("actions", "last_action", 12, PROPRIO_HISTORY_LENGTH, "runtime_action_cache", "normalized", "training joint order", "previous policy output before scale/offset", (1.0,) * 12),
)


def actor_dim() -> int:
  return sum(term.flattened_dim for term in STUDENT_TERMS)


def schema_payload() -> dict[str, Any]:
  terms = []
  start = 0
  for term in STUDENT_TERMS:
    end = start + term.flattened_dim
    terms.append(
      asdict(term)
      | {
        "flattened_dim": term.flattened_dim,
        "index_start": start,
        "index_end_exclusive": end,
        "time_order": "oldest-to-newest" if term.history_length > 1 else "current",
      }
    )
    start = end
  return {
    "version": SCHEMA_VERSION,
    "control_dt_s": CONTROL_DT_S,
    "history_order": "term-major, oldest-to-newest",
    "history_reset": "first finite frame backfills all 10 history slots",
    "history_sample_count": PROPRIO_HISTORY_LENGTH,
    "history_endpoint_span_s": (PROPRIO_HISTORY_LENGTH - 1) * CONTROL_DT_S,
    "actor_dim": actor_dim(),
    "terms": terms,
    "phase": {
      "period_s": PHASE_PERIOD_S,
      "stand_command_norm_threshold": STAND_COMMAND_NORM_THRESHOLD,
      "formula": "phase=(policy_tick*control_dt_s/period_s)%1",
      "reset_policy_tick": 0,
      "getter_has_side_effects": False,
    },
    "previous_action": {
      "timing": "observation t contains normalized policy action from t-1",
      "reset": "zeros",
    },
    "action": {
      "dim": len(JOINT_NAMES),
      "type": "joint_position_target",
      "scale": ACTION_SCALE,
      "absolute_normalized_limit": ACTION_ABS_LIMIT,
      "joint_names": list(JOINT_NAMES),
      "sdk_joint_ids_map": list(SDK_JOINT_IDS_MAP),
      "default_joint_pos": list(DEFAULT_JOINT_POS),
      "stiffness": list(JOINT_STIFFNESS),
      "damping": list(JOINT_DAMPING),
      "effort_limit_nm": list(JOINT_EFFORT_LIMIT),
      "joint_position_limits_rad": [list(limits) for limits in JOINT_POS_LIMITS],
    },
    "onnx": {
      "input_name": "actor",
      "input_shape": [1, actor_dim()],
      "output_name": "actions",
      "output_shape": [1, len(JOINT_NAMES)],
      "dtype": "float32",
      "static_shapes": True,
    },
    "runtime": {
      "low_state_timeout": "SDK host-monotonic isTimeout -> Passive",
      "nonfinite_observation": "Passive",
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


def _finite_vector(values: Any, expected_dim: int, name: str) -> tuple[float, ...]:
  vector = tuple(float(value) for value in values)
  if len(vector) != expected_dim:
    raise ValueError(f"{name} must have {expected_dim} values")
  if not all(math.isfinite(value) for value in vector):
    raise ValueError(f"{name} contains NaN/Inf")
  return vector


def mock_low_state_to_frame(
  *,
  gyroscope: Any,
  quaternion_wxyz: Any,
  sdk_joint_pos: Any,
  sdk_joint_vel: Any,
  previous_action: Any,
) -> dict[str, tuple[float, ...]]:
  """Convert SDK-order mock LowState fields to one training-order frame."""
  gyro = _finite_vector(gyroscope, 3, "gyroscope")
  quat = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
  quat_norm = math.sqrt(sum(value * value for value in quat))
  if quat_norm < 0.5 or quat_norm > 1.5:
    raise ValueError("quaternion norm outside runtime safety range")
  w, x, y, z = (value / quat_norm for value in quat)
  # q.conjugate() rotates world gravity [0,0,-1] into the base frame.
  gravity = (
    2.0 * (w * y - x * z),
    -2.0 * (w * x + y * z),
    -(1.0 - 2.0 * (x * x + y * y)),
  )
  sdk_q = _finite_vector(sdk_joint_pos, 12, "sdk_joint_pos")
  sdk_dq = _finite_vector(sdk_joint_vel, 12, "sdk_joint_vel")
  action = _finite_vector(previous_action, 12, "previous_action")
  joint_pos = tuple(
    sdk_q[sdk_id] - DEFAULT_JOINT_POS[index]
    for index, sdk_id in enumerate(SDK_JOINT_IDS_MAP)
  )
  joint_vel = tuple(sdk_dq[sdk_id] for sdk_id in SDK_JOINT_IDS_MAP)
  return {
    "base_ang_vel": gyro,
    "projected_gravity": gravity,
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "actions": action,
  }


def assemble_actor_observation(
  history: Any, *, command: Any, phase: Any
) -> tuple[float, ...]:
  """Assemble 10 oldest-to-newest frames into the frozen 425-D layout."""
  frames = tuple(history)
  if len(frames) != PROPRIO_HISTORY_LENGTH:
    raise ValueError(f"history must contain {PROPRIO_HISTORY_LENGTH} frames")
  current = {
    "command": _finite_vector(command, 3, "command"),
    "phase": _finite_vector(phase, 2, "phase"),
  }
  values: list[float] = []
  for term in STUDENT_TERMS:
    if term.training_name in current:
      values.extend(current[term.training_name])
      continue
    for frame in frames:
      values.extend(
        _finite_vector(frame[term.training_name], term.dim, term.training_name)
      )
  if len(values) != actor_dim():
    raise RuntimeError("assembled actor observation has the wrong dimension")
  return tuple(values)


def normalized_action_to_sdk_targets(action: Any) -> tuple[float, ...]:
  """Map normalized training-order actions to SDK-order joint position targets."""
  normalized = _finite_vector(action, 12, "action")
  if any(abs(value) > ACTION_ABS_LIMIT for value in normalized):
    raise ValueError("action exceeds normalized runtime safety limit")
  sdk_targets = [0.0] * 12
  for index, sdk_id in enumerate(SDK_JOINT_IDS_MAP):
    target = DEFAULT_JOINT_POS[index] + ACTION_SCALE * normalized[index]
    lower, upper = JOINT_POS_LIMITS[index]
    if target < lower or target > upper:
      raise ValueError(f"action produces out-of-range target for {JOINT_NAMES[index]}")
    sdk_targets[sdk_id] = target
  return tuple(sdk_targets)


def validate_deploy_yaml(path: Path) -> None:
  payload = yaml.safe_load(path.read_text(encoding="utf-8"))
  if tuple(payload["joint_ids_map"]) != SDK_JOINT_IDS_MAP:
    raise ValueError("deploy SDK joint mapping differs from student schema")
  if float(payload["step_dt"]) != CONTROL_DT_S:
    raise ValueError("deploy control period differs from student schema")
  if list(payload["observations"]) != ["actor"]:
    raise ValueError("deploy must expose exactly one actor observation group")
  observations = payload["observations"]["actor"]
  if list(observations) != [term.deploy_name for term in STUDENT_TERMS]:
    raise ValueError("deploy observation term ordering differs from student schema")
  for term in STUDENT_TERMS:
    configured = observations[term.deploy_name]
    if int(configured["history_length"]) != term.history_length:
      raise ValueError(f"history mismatch for {term.deploy_name}")
    if len(configured["scale"]) != term.dim:
      raise ValueError(f"scale dimension mismatch for {term.deploy_name}")
    if tuple(float(value) for value in configured["scale"]) != term.scale:
      raise ValueError(f"scale value mismatch for {term.deploy_name}")
  action = payload["actions"]["JointPositionAction"]
  if len(action["scale"]) != len(JOINT_NAMES):
    raise ValueError("deploy action scale dimension mismatch")
  if any(float(value) != ACTION_SCALE for value in action["scale"]):
    raise ValueError("deploy action scale value mismatch")
  if tuple(float(value) for value in action["offset"]) != DEFAULT_JOINT_POS:
    raise ValueError("deploy action offset mismatch")
  for key, expected in (
    ("default_joint_pos", DEFAULT_JOINT_POS),
    ("stiffness", JOINT_STIFFNESS),
    ("damping", JOINT_DAMPING),
  ):
    if tuple(float(value) for value in payload[key]) != expected:
      raise ValueError(f"deploy {key} mismatch")
  configured_limits = tuple(
    tuple(float(value) for value in limits)
    for limits in payload["safety"]["joint_pos_limits"]
  )
  if configured_limits != JOINT_POS_LIMITS:
    raise ValueError("deploy joint position safety limits mismatch")
  schema = payload["schema"]
  if schema["version"] != SCHEMA_VERSION or schema["sha256"] != schema_sha256():
    raise ValueError("deploy schema identity mismatch")
  if int(schema["actor_dim"]) != actor_dim() or int(schema["action_dim"]) != len(JOINT_NAMES):
    raise ValueError("deploy schema dimension mismatch")
  if schema.get("history_order") != "term-major, oldest-to-newest":
    raise ValueError("deploy history order mismatch")
