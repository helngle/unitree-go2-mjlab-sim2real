"""Online rollout evidence for evaluation-only terrain routes."""

from __future__ import annotations

import math
from typing import Mapping

import torch


BODY_CONTACT_NAMES = ("base", "upper_leg", "calf")
ACTIVE_SAMPLE_DEFINITION = (
  "one sample per environment and control step while the original route attempt "
  "is active; the terminal step is included and all later/reset-episode steps are frozen"
)
ACTION_ACCELERATION_DEFINITION = (
  "mean(abs(action[t] - 2*action[t-1] + action[t-2])) over action dimensions; "
  "discrete second difference without control-dt scaling"
)
BASE_PITCH_DEFINITION = (
  "absolute base pitch from projected gravity, in radians, sampled on the "
  "active route-attempt denominator"
)
TERRAIN_TANGENT_STANCE_SLIP_DEFINITION = (
  "terrain-tangent foot speed over loaded stance feet only; local normal is "
  "the nearest valid terrain_scan ray within 0.25 m, loaded means contact and "
  "supporting normal force >= 15 N; reward deadband=0.03 m/s, scale=0.10 m/s, "
  "per-foot cost clip=4, with normal-load-normalized step cost"
)
ACTUATOR_EFFORT_DEFINITION = (
  "mean absolute MuJoCo actuator force over the 12 policy joints, in Nm, "
  "sampled on the active route-attempt denominator"
)
MECHANICAL_POWER_DEFINITION = (
  "sum over the 12 policy joints of abs(actuator_force * joint_velocity), "
  "in W, sampled on the active route-attempt denominator"
)


def action_acceleration(
  action: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
) -> torch.Tensor:
  """Return the established per-environment discrete action second difference."""
  if action.ndim != 2:
    raise ValueError("action tensors must have shape (num_envs, action_dim)")
  if previous_action.shape != action.shape or previous_previous_action.shape != action.shape:
    raise ValueError("all action tensors must have identical shapes")
  values = action - 2.0 * previous_action + previous_previous_action
  if not torch.isfinite(values).all():
    raise ValueError("action tensors must be finite")
  return values.abs().mean(dim=-1)


def contact_any(found: torch.Tensor, num_envs: int) -> torch.Tensor:
  """Reduce a sensor's possibly multi-match contact tensor to one bool per env."""
  if found.ndim < 1 or found.shape[0] != num_envs:
    raise ValueError("contact tensor must have leading dimension num_envs")
  return (found > 0).reshape(num_envs, -1).any(dim=-1)


def foot_contact_any(
  found: torch.Tensor, num_envs: int, num_feet: int
) -> torch.Tensor:
  """Reduce contact matches while retaining the individual foot dimension."""
  if num_feet <= 0 or found.ndim < 2 or found.shape[0] != num_envs:
    raise ValueError("foot contact tensor must start with (num_envs, feet)")
  per_env = found[0].numel()
  if per_env % num_feet != 0:
    raise ValueError("foot contact matches cannot be grouped by num_feet")
  return (found > 0).reshape(num_envs, num_feet, -1).any(dim=-1)


def foot_slip_velocity(
  foot_velocity_w_xy: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
  """Mean horizontal speed of contacting feet, or zero when no foot contacts."""
  if foot_velocity_w_xy.ndim != 3 or foot_velocity_w_xy.shape[-1] != 2:
    raise ValueError("foot_velocity_w_xy must have shape (num_envs, feet, 2)")
  if contact.shape != foot_velocity_w_xy.shape[:2] or contact.dtype != torch.bool:
    raise ValueError("contact must be bool with shape (num_envs, feet)")
  if not torch.isfinite(foot_velocity_w_xy).all():
    raise ValueError("foot velocities must be finite")
  speed = torch.linalg.vector_norm(foot_velocity_w_xy, dim=-1)
  count = contact.sum(dim=-1)
  return torch.where(
    count > 0,
    (speed * contact).sum(dim=-1) / count.clamp_min(1),
    torch.zeros_like(speed[:, 0]),
  )


def assert_recursive_json_finite(value: object, path: str = "root") -> None:
  """Reject non-finite values before strict JSON serialization."""
  if isinstance(value, float) and not math.isfinite(value):
    raise ValueError(f"non-finite JSON value at {path}: {value}")
  if isinstance(value, Mapping):
    for key, item in value.items():
      assert_recursive_json_finite(item, f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      assert_recursive_json_finite(item, f"{path}[{index}]")


def _distribution(
  values: torch.Tensor, *, empty_reason: str
) -> dict[str, float | str | None]:
  if values.numel() == 0:
    return {"mean": None, "p95": None, "max": None, "reason": empty_reason}
  if not torch.isfinite(values).all():
    raise ValueError("metric samples must be finite")
  values = values.to(dtype=torch.float64)
  return {
    "mean": float(values.mean()),
    "p95": float(torch.quantile(values, 0.95)),
    "max": float(values.max()),
  }


class OnlineTerrainRolloutMetrics:
  """Retain per-step terrain metrics under the route attempt's sample mask."""

  def __init__(
    self,
    num_envs: int,
    max_steps: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    control_dt_s: float | None = None,
  ) -> None:
    if num_envs <= 0 or max_steps <= 0:
      raise ValueError("num_envs and max_steps must be positive")
    self.num_envs = num_envs
    self.max_steps = max_steps
    if control_dt_s is not None and (
      not math.isfinite(control_dt_s) or control_dt_s <= 0.0
    ):
      raise ValueError("control_dt_s must be finite and positive")
    self.control_dt_s = control_dt_s
    self._next_step = 0
    shape = (num_envs, max_steps)
    self._sample_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    self._action = torch.zeros(shape, dtype=dtype, device=device)
    self._slip = torch.zeros(shape, dtype=dtype, device=device)
    self._slip_available = torch.zeros(shape, dtype=torch.bool, device=device)
    self._pitch = torch.zeros(shape, dtype=dtype, device=device)
    self._pitch_available = torch.zeros(shape, dtype=torch.bool, device=device)
    self._effort = torch.zeros(shape, dtype=dtype, device=device)
    self._effort_available = torch.zeros(shape, dtype=torch.bool, device=device)
    self._power = torch.zeros(shape, dtype=dtype, device=device)
    self._power_available = torch.zeros(shape, dtype=torch.bool, device=device)
    self._normalized_action_abs_max = torch.zeros(
      shape, dtype=dtype, device=device
    )
    self._action_safety_fault = torch.zeros(
      shape, dtype=torch.bool, device=device
    )
    self._action_safety_available = torch.zeros(
      shape, dtype=torch.bool, device=device
    )
    self._joint_target_safety_fault = torch.zeros(
      shape, dtype=torch.bool, device=device
    )
    self._joint_target_safety_available = torch.zeros(
      shape, dtype=torch.bool, device=device
    )
    self._catastrophic = torch.zeros(shape, dtype=torch.bool, device=device)
    self._contacts = {
      name: torch.zeros(shape, dtype=torch.bool, device=device)
      for name in BODY_CONTACT_NAMES
    }
    self._contact_available = {
      name: torch.zeros(shape, dtype=torch.bool, device=device)
      for name in BODY_CONTACT_NAMES
    }

  def update(
    self,
    *,
    sample_mask: torch.Tensor,
    action_acceleration: torch.Tensor,
    foot_slip_velocity: torch.Tensor | None,
    body_contacts: Mapping[str, torch.Tensor | None],
    catastrophic_termination: torch.Tensor,
    base_pitch: torch.Tensor | None = None,
    actuator_effort_abs: torch.Tensor | None = None,
    mechanical_power_abs: torch.Tensor | None = None,
    normalized_action_abs_max: torch.Tensor | None = None,
    action_safety_fault: torch.Tensor | None = None,
    joint_target_safety_fault: torch.Tensor | None = None,
  ) -> None:
    if self._next_step >= self.max_steps:
      raise RuntimeError("metric accumulator exceeds configured max_steps")
    self._validate_bool_vector(sample_mask, "sample_mask")
    self._validate_bool_vector(catastrophic_termination, "catastrophic_termination")
    self._validate_metric_vector(action_acceleration, "action_acceleration")
    unknown = set(body_contacts) - set(BODY_CONTACT_NAMES)
    if unknown:
      raise ValueError(f"unknown body contact names: {sorted(unknown)}")

    column = self._next_step
    self._sample_valid[:, column] = sample_mask
    self._action[:, column] = action_acceleration
    self._catastrophic[:, column] = catastrophic_termination
    if foot_slip_velocity is not None:
      self._validate_metric_vector(foot_slip_velocity, "foot_slip_velocity")
      self._slip[:, column] = foot_slip_velocity
      self._slip_available[:, column] = True
    if base_pitch is not None:
      self._validate_metric_vector(base_pitch, "base_pitch")
      self._pitch[:, column] = base_pitch.abs()
      self._pitch_available[:, column] = True
    if actuator_effort_abs is not None:
      self._validate_metric_vector(actuator_effort_abs, "actuator_effort_abs")
      self._effort[:, column] = actuator_effort_abs
      self._effort_available[:, column] = True
    if mechanical_power_abs is not None:
      self._validate_metric_vector(mechanical_power_abs, "mechanical_power_abs")
      self._power[:, column] = mechanical_power_abs
      self._power_available[:, column] = True
    if (normalized_action_abs_max is None) != (action_safety_fault is None):
      raise ValueError(
        "normalized_action_abs_max and action_safety_fault must be provided together"
      )
    if normalized_action_abs_max is not None and action_safety_fault is not None:
      self._validate_metric_vector(
        normalized_action_abs_max, "normalized_action_abs_max"
      )
      self._validate_bool_vector(action_safety_fault, "action_safety_fault")
      self._normalized_action_abs_max[:, column] = normalized_action_abs_max
      self._action_safety_fault[:, column] = action_safety_fault
      self._action_safety_available[:, column] = True
    if joint_target_safety_fault is not None:
      self._validate_bool_vector(
        joint_target_safety_fault, "joint_target_safety_fault"
      )
      self._joint_target_safety_fault[:, column] = joint_target_safety_fault
      self._joint_target_safety_available[:, column] = True
    for name in BODY_CONTACT_NAMES:
      value = body_contacts.get(name)
      if value is not None:
        self._validate_bool_vector(value, f"{name}_contact")
        self._contacts[name][:, column] = value
        self._contact_available[name][:, column] = True
    self._next_step += 1

  def _validate_bool_vector(self, value: torch.Tensor, name: str) -> None:
    if value.shape != (self.num_envs,) or value.dtype != torch.bool:
      raise ValueError(f"{name} must be bool with shape (num_envs,)")

  def _validate_metric_vector(self, value: torch.Tensor, name: str) -> None:
    if value.shape != (self.num_envs,):
      raise ValueError(f"{name} must have shape (num_envs,)")
    if not torch.isfinite(value).all():
      raise ValueError(f"{name} must be finite")

  def result(self, env_index: int) -> dict[str, object]:
    if env_index < 0 or env_index >= self.num_envs:
      raise IndexError("env_index outside batch")
    valid = self._sample_valid[env_index, :self._next_step]
    sample_count = int(valid.sum())
    action = self._action[env_index, :self._next_step][valid]
    slip_mask = valid & self._slip_available[env_index, :self._next_step]
    slip_reason = (
      "no_active_control_step_samples"
      if sample_count == 0
      else "foot_contact_sensor_unavailable"
    )
    pitch_mask = valid & self._pitch_available[env_index, :self._next_step]
    pitch_reason = (
      "no_active_control_step_samples"
      if sample_count == 0
      else "base_pitch_unavailable"
    )
    effort_mask = valid & self._effort_available[env_index, :self._next_step]
    effort_reason = (
      "no_active_control_step_samples"
      if sample_count == 0
      else "actuator_effort_unavailable"
    )
    power_mask = valid & self._power_available[env_index, :self._next_step]
    power_reason = (
      "no_active_control_step_samples"
      if sample_count == 0
      else "mechanical_power_unavailable"
    )
    catastrophic = self._catastrophic[env_index, :self._next_step] & valid
    action_safety_mask = (
      valid & self._action_safety_available[env_index, :self._next_step]
    )
    action_safety_fault = (
      self._action_safety_fault[env_index, :self._next_step]
      & action_safety_mask
    )
    joint_target_safety_mask = (
      valid
      & self._joint_target_safety_available[env_index, :self._next_step]
    )
    joint_target_safety_fault = (
      self._joint_target_safety_fault[env_index, :self._next_step]
      & joint_target_safety_mask
    )
    power_values = self._power[env_index, :self._next_step][power_mask]
    contacts: dict[str, object] = {}
    for name in BODY_CONTACT_NAMES:
      available = self._contact_available[name][env_index, :self._next_step]
      contact = self._contacts[name][env_index, :self._next_step]
      observed = valid & available
      if not bool(observed.any()):
        reason = (
          "no_active_control_step_samples"
          if sample_count == 0
          else "contact_sensor_unavailable"
        )
        contacts[name] = {
          "non_terminating_count": None,
          "non_terminating_rate": None,
          "all_contact_count": None,
          "all_contact_rate": None,
          "denominator": sample_count,
          "reason": reason,
        }
        continue
      all_count = int((contact & observed).sum())
      non_terminating_count = int((contact & observed & ~catastrophic).sum())
      contacts[name] = {
        "non_terminating_count": non_terminating_count,
        "non_terminating_rate": non_terminating_count / sample_count,
        "all_contact_count": all_count,
        "all_contact_rate": all_count / sample_count,
        "denominator": sample_count,
      }
    return {
      "active_control_step_samples": sample_count,
      "sample_denominator_definition": ACTIVE_SAMPLE_DEFINITION,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
      "action_acceleration": _distribution(
        action, empty_reason="no_active_control_step_samples"
      ),
      "foot_slip_velocity": _distribution(
        self._slip[env_index, :self._next_step][slip_mask],
        empty_reason=slip_reason,
      ),
      "base_pitch_definition": BASE_PITCH_DEFINITION,
      "base_pitch_absolute": _distribution(
        self._pitch[env_index, :self._next_step][pitch_mask],
        empty_reason=pitch_reason,
      ),
      "actuator_effort_definition": ACTUATOR_EFFORT_DEFINITION,
      "actuator_effort_abs": _distribution(
        self._effort[env_index, :self._next_step][effort_mask],
        empty_reason=effort_reason,
      ),
      "mechanical_power_definition": MECHANICAL_POWER_DEFINITION,
      "mechanical_power_abs": _distribution(
        power_values,
        empty_reason=power_reason,
      ),
      "mechanical_energy_abs": (
        None
        if self.control_dt_s is None or power_values.numel() == 0
        else float(power_values.to(dtype=torch.float64).sum() * self.control_dt_s)
      ),
      "mechanical_energy_definition": (
        "sum(active-step absolute mechanical power) * control_dt_s, in J"
      ),
      "action_safety": {
        "normalized_action_abs_max": (
          None
          if not bool(action_safety_mask.any())
          else float(
            self._normalized_action_abs_max[
              env_index, :self._next_step
            ][action_safety_mask].max()
          )
        ),
        "fault_control_step_count": int(action_safety_fault.sum()),
        "fault_occurred": bool(action_safety_fault.any()),
        "available": bool(action_safety_mask.any()),
        "joint_target_fault_control_step_count": int(
          joint_target_safety_fault.sum()
        ),
        "joint_target_fault_occurred": bool(
          joint_target_safety_fault.any()
        ),
        "joint_target_available": bool(joint_target_safety_mask.any()),
      },
      "body_contacts": contacts,
      "catastrophic_termination": {
        "control_step_count": int(catastrophic.sum()),
        "occurred": bool(catastrophic.any()),
      },
    }


class OnlineTerrainTangentSlipMetrics:
  """Retain frozen local-tangent loaded-stance evidence per route attempt."""

  def __init__(
    self,
    num_envs: int,
    max_steps: int,
    num_feet: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    if num_envs <= 0 or max_steps <= 0 or num_feet <= 0:
      raise ValueError("num_envs, max_steps, and num_feet must be positive")
    self.num_envs = num_envs
    self.max_steps = max_steps
    self.num_feet = num_feet
    self._next_step = 0
    step_shape = (num_envs, max_steps)
    foot_shape = (num_envs, max_steps, num_feet)
    self._sample_valid = torch.zeros(step_shape, dtype=torch.bool, device=device)
    self._cost = torch.zeros(step_shape, dtype=dtype, device=device)
    self._slip = torch.zeros(foot_shape, dtype=dtype, device=device)
    self._loaded = torch.zeros(foot_shape, dtype=torch.bool, device=device)
    self._ray_valid = torch.zeros(foot_shape, dtype=torch.bool, device=device)
    self._normal_force = torch.zeros(foot_shape, dtype=dtype, device=device)

  def update(
    self,
    *,
    sample_mask: torch.Tensor,
    cost: torch.Tensor,
    slip_velocity: torch.Tensor,
    loaded: torch.Tensor,
    ray_valid: torch.Tensor,
    normal_force: torch.Tensor,
  ) -> None:
    if self._next_step >= self.max_steps:
      raise RuntimeError("terrain-tangent accumulator exceeds configured max_steps")
    if sample_mask.shape != (self.num_envs,) or sample_mask.dtype != torch.bool:
      raise ValueError("sample_mask must be bool with shape (num_envs,)")
    if cost.shape != (self.num_envs,) or not torch.isfinite(cost).all():
      raise ValueError("cost must be finite with shape (num_envs,)")
    expected = (self.num_envs, self.num_feet)
    for name, value in (("slip_velocity", slip_velocity), ("normal_force", normal_force)):
      if value.shape != expected or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape (num_envs, feet)")
    for name, value in (("loaded", loaded), ("ray_valid", ray_valid)):
      if value.shape != expected or value.dtype != torch.bool:
        raise ValueError(f"{name} must be bool with shape (num_envs, feet)")
    column = self._next_step
    self._sample_valid[:, column] = sample_mask
    self._cost[:, column] = cost
    self._slip[:, column] = slip_velocity
    self._loaded[:, column] = loaded
    self._ray_valid[:, column] = ray_valid
    self._normal_force[:, column] = normal_force
    self._next_step += 1

  def result(self, env_index: int) -> dict[str, object]:
    if env_index < 0 or env_index >= self.num_envs:
      raise IndexError("env_index outside batch")
    valid_steps = self._sample_valid[env_index, :self._next_step]
    active_count = int(valid_steps.sum())
    active_feet = valid_steps[:, None].expand(-1, self.num_feet)
    loaded = self._loaded[env_index, :self._next_step] & active_feet
    ray_valid = self._ray_valid[env_index, :self._next_step] & active_feet
    loaded_count = int(loaded.sum())
    denominator = active_count * self.num_feet
    empty_reason = (
      "no_active_control_step_samples" if active_count == 0
      else "no_loaded_stance_foot_samples"
    )
    return {
      "definition": TERRAIN_TANGENT_STANCE_SLIP_DEFINITION,
      "active_control_step_samples": active_count,
      "loaded_stance_foot_samples": loaded_count,
      "loaded_stance_fraction": loaded_count / denominator if denominator else None,
      "ray_valid_foot_samples": int(ray_valid.sum()),
      "ray_valid_fraction": int(ray_valid.sum()) / denominator if denominator else None,
      "slip_velocity": _distribution(
        self._slip[env_index, :self._next_step][loaded],
        empty_reason=empty_reason,
      ),
      "normal_force": _distribution(
        self._normal_force[env_index, :self._next_step][loaded],
        empty_reason=empty_reason,
      ),
      "load_normalized_slip_cost": _distribution(
        self._cost[env_index, :self._next_step][valid_steps],
        empty_reason="no_active_control_step_samples",
      ),
    }


__all__ = [
  "ACTION_ACCELERATION_DEFINITION",
  "ACTIVE_SAMPLE_DEFINITION",
  "BASE_PITCH_DEFINITION",
  "BODY_CONTACT_NAMES",
  "ACTUATOR_EFFORT_DEFINITION",
  "MECHANICAL_POWER_DEFINITION",
  "OnlineTerrainRolloutMetrics",
  "OnlineTerrainTangentSlipMetrics",
  "TERRAIN_TANGENT_STANCE_SLIP_DEFINITION",
  "action_acceleration",
  "assert_recursive_json_finite",
  "contact_any",
  "foot_contact_any",
  "foot_slip_velocity",
]
