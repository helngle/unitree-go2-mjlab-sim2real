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
  ) -> None:
    if num_envs <= 0 or max_steps <= 0:
      raise ValueError("num_envs and max_steps must be positive")
    self.num_envs = num_envs
    self.max_steps = max_steps
    self._next_step = 0
    shape = (num_envs, max_steps)
    self._sample_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    self._action = torch.zeros(shape, dtype=dtype, device=device)
    self._slip = torch.zeros(shape, dtype=dtype, device=device)
    self._slip_available = torch.zeros(shape, dtype=torch.bool, device=device)
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
    catastrophic = self._catastrophic[env_index, :self._next_step] & valid
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
      "body_contacts": contacts,
      "catastrophic_termination": {
        "control_step_count": int(catastrophic.sum()),
        "occurred": bool(catastrophic.any()),
      },
    }


__all__ = [
  "ACTION_ACCELERATION_DEFINITION",
  "ACTIVE_SAMPLE_DEFINITION",
  "BODY_CONTACT_NAMES",
  "OnlineTerrainRolloutMetrics",
  "action_acceleration",
  "assert_recursive_json_finite",
  "contact_any",
  "foot_contact_any",
  "foot_slip_velocity",
]
