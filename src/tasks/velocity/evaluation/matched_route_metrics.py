"""Contracts and online metrics for matched straight/arc/S evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch


ROUTE_KINDS = ("straight", "arc", "s_curve")
CORE_PROFILE_NAMES = (
  "clean",
  "dynamics_only",
  "observation_only",
  "push_only",
  "full_randomized",
)
FACTOR_PROFILE_NAMES = (
  "foot_friction_only",
  "base_com_only",
  "base_payload_only",
  "motor_strength_only",
  "encoder_bias_only",
  "actor_corruption_only",
)
PROFILE_NAMES = CORE_PROFILE_NAMES + FACTOR_PROFILE_NAMES
DYNAMICS_EVENTS = ("foot_friction", "base_com", "base_payload", "motor_strength")
OBSERVATION_EVENTS = ("encoder_bias",)
RANDOMIZATION_EVENTS = DYNAMICS_EVENTS + OBSERVATION_EVENTS
ACTION_ACCELERATION_DEFINITION = (
  "mean_joint_abs_second_action_difference_per_control_step"
)


@dataclass(frozen=True)
class MatchedProfile:
  name: str
  actor_observation_corruption: bool
  startup_events: tuple[str, ...]
  push_enabled: bool


PROFILES: Mapping[str, MatchedProfile] = {
  "clean": MatchedProfile("clean", False, (), False),
  "dynamics_only": MatchedProfile(
    "dynamics_only", False, DYNAMICS_EVENTS, False
  ),
  "observation_only": MatchedProfile(
    "observation_only", True, OBSERVATION_EVENTS, False
  ),
  "push_only": MatchedProfile("push_only", False, (), True),
  "full_randomized": MatchedProfile(
    "full_randomized", True, RANDOMIZATION_EVENTS, True
  ),
  "foot_friction_only": MatchedProfile(
    "foot_friction_only", False, ("foot_friction",), False
  ),
  "base_com_only": MatchedProfile(
    "base_com_only", False, ("base_com",), False
  ),
  "base_payload_only": MatchedProfile(
    "base_payload_only", False, ("base_payload",), False
  ),
  "motor_strength_only": MatchedProfile(
    "motor_strength_only", False, ("motor_strength",), False
  ),
  "encoder_bias_only": MatchedProfile(
    "encoder_bias_only", False, ("encoder_bias",), False
  ),
  "actor_corruption_only": MatchedProfile(
    "actor_corruption_only", True, (), False
  ),
}


def configure_matched_profile(env_cfg: Any, profile_name: str) -> dict[str, Any]:
  """Apply exactly one evaluation-only randomization profile to ``env_cfg``."""
  try:
    profile = PROFILES[profile_name]
  except KeyError as exc:
    raise ValueError(
      f"profile must be one of {PROFILE_NAMES}, got {profile_name!r}"
    ) from exc
  actor = env_cfg.observations["actor"]
  actor.enable_corruption = profile.actor_observation_corruption
  enabled = set(profile.startup_events)
  for event_name in RANDOMIZATION_EVENTS:
    if event_name not in enabled:
      env_cfg.events.pop(event_name, None)
  if not profile.push_enabled:
    env_cfg.events.pop("push_robot", None)
  missing = sorted(enabled - set(env_cfg.events))
  if missing:
    raise ValueError(
      f"task does not provide randomization events required by {profile_name}: "
      f"{missing}"
    )
  if profile.push_enabled and "push_robot" not in env_cfg.events:
    raise ValueError("task does not provide the push_robot event")

  def jsonable_parameters(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
      return value
    if isinstance(value, Mapping):
      return {
        str(key): converted
        for key, item in value.items()
        if (converted := jsonable_parameters(item)) is not None
      }
    if isinstance(value, Sequence):
      converted = [jsonable_parameters(item) for item in value]
      return [item for item in converted if item is not None]
    return None

  enabled_names = [
    name for name in RANDOMIZATION_EVENTS if name in env_cfg.events
  ]
  if "push_robot" in env_cfg.events:
    enabled_names.append("push_robot")
  event_parameters = {}
  for name in enabled_names:
    event = env_cfg.events[name]
    event_parameters[name] = {
      "mode": getattr(event, "mode", None),
      "interval_range_s": jsonable_parameters(
        getattr(event, "interval_range_s", None)
      ),
      "params": jsonable_parameters(getattr(event, "params", {})),
    }
  return {
    "name": profile.name,
    "actor_observation_corruption": actor.enable_corruption,
    "startup_randomization_events": [
      name for name in RANDOMIZATION_EVENTS if name in env_cfg.events
    ],
    "push_enabled": "push_robot" in env_cfg.events,
    "event_parameters": event_parameters,
  }


@dataclass(frozen=True)
class MatchedRouteContract:
  checkpoint: str
  task_id: str
  seed: int
  profile: str
  num_slots: int
  speeds: tuple[float, ...]
  steps: int
  settle_steps: int
  control_dt: float
  route_kinds: tuple[str, ...] = ROUTE_KINDS

  def __post_init__(self) -> None:
    if not self.checkpoint:
      raise ValueError("checkpoint must not be empty")
    if not self.task_id:
      raise ValueError("task_id must not be empty")
    if self.profile not in PROFILE_NAMES:
      raise ValueError(f"unknown matched profile {self.profile!r}")
    if self.num_slots <= 0 or self.steps <= 0:
      raise ValueError("num_slots and steps must be positive")
    if self.settle_steps < 0 or self.settle_steps >= self.steps:
      raise ValueError("settle_steps must be in [0, steps)")
    if not self.speeds or any(
      not math.isfinite(value) or value <= 0.0 for value in self.speeds
    ):
      raise ValueError("speeds must be finite and positive")
    if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
      raise ValueError("control_dt must be finite and positive")
    if tuple(self.route_kinds) != ROUTE_KINDS:
      raise ValueError(
        f"route_kinds must preserve the matched order {ROUTE_KINDS}"
      )

  def invariant_fields(self) -> dict[str, Any]:
    """Fields that must be identical for every route-kind subprocess."""
    return {
      "checkpoint": self.checkpoint,
      "task_id": self.task_id,
      "seed": self.seed,
      "profile": self.profile,
      "num_envs": self.num_slots,
      "speeds": list(self.speeds),
      "steps": self.steps,
      "settle_steps": self.settle_steps,
      "control_dt": self.control_dt,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
    }


def matched_route_length(radius: float) -> float:
  """Common distance for a straight, a 120-degree arc, and a two-arc S."""
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("radius must be finite and positive")
  return 2.0 * math.pi * radius / 3.0


def matched_route_local_bounds(
  route_kind: str,
  radius: float,
  turn_sign: int,
  *,
  start_xy: tuple[float, float] = (2.0, 8.0),
) -> tuple[float, float, float, float]:
  """Exact XY bounds for the canonical matched route centerline."""
  length = matched_route_length(radius)
  if route_kind not in ROUTE_KINDS:
    raise ValueError(f"route_kind must be one of {ROUTE_KINDS}")
  if turn_sign not in (-1, 1):
    raise ValueError("turn_sign must be -1 or +1")
  start_x, start_y = start_xy
  if not all(math.isfinite(value) for value in start_xy):
    raise ValueError("start_xy must be finite")
  if route_kind == "straight":
    return start_x, start_x + length, start_y, start_y
  x_extent = (
    math.sin(2.0 * math.pi / 3.0) * radius
    if route_kind == "arc"
    else math.sqrt(3.0) * radius
  )
  y_extent = (1.5 * radius if route_kind == "arc" else radius) * turn_sign
  return (
    start_x,
    start_x + x_extent,
    min(start_y, start_y + y_extent),
    max(start_y, start_y + y_extent),
  )


def action_acceleration(
  action: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
) -> torch.Tensor:
  """Per-environment mean absolute second action difference.

  This deliberately matches the existing route evaluators. It is a discrete
  smoothness metric and is not divided by ``control_dt ** 2``.
  """
  if action.ndim != 2:
    raise ValueError("action tensors must have shape (N, action_dim)")
  if previous_action.shape != action.shape or previous_previous_action.shape != action.shape:
    raise ValueError("all action tensors must have identical shapes")
  values = action - 2.0 * previous_action + previous_previous_action
  if not torch.isfinite(values).all():
    raise ValueError("action tensors must be finite")
  return values.abs().mean(dim=-1)


def contact_slip_velocity(
  foot_velocity_w_xy: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
  """Mean horizontal speed of contacting feet, or zero with no contact."""
  if foot_velocity_w_xy.ndim != 3 or foot_velocity_w_xy.shape[-1] != 2:
    raise ValueError("foot_velocity_w_xy must have shape (N, feet, 2)")
  if contact.shape != foot_velocity_w_xy.shape[:2] or contact.dtype != torch.bool:
    raise ValueError("contact must be bool with shape (N, feet)")
  speed = torch.linalg.vector_norm(foot_velocity_w_xy, dim=-1)
  count = contact.sum(dim=-1)
  return torch.where(
    count > 0,
    (speed * contact).sum(dim=-1) / count.clamp_min(1),
    torch.zeros_like(speed[:, 0]),
  )


def _distribution(values: torch.Tensor) -> dict[str, float | None]:
  if values.numel() == 0:
    return {"mean": None, "p95": None, "max": None}
  if not torch.isfinite(values).all():
    raise ValueError("metric samples must be finite")
  values = values.to(dtype=torch.float64)
  return {
    "mean": float(values.mean()),
    "p95": float(torch.quantile(values, 0.95)),
    "max": float(values.max()),
  }


class OnlineMatchedRouteMetrics:
  """Retain compact per-step samples and freeze inactive attempts."""

  METRIC_NAMES = (
    "action_acceleration",
    "slip_velocity",
    "velocity_error",
    "cross_axis_velocity",
  )

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
    self._valid = torch.zeros(
      (num_envs, max_steps), dtype=torch.bool, device=device
    )
    self._values = {
      name: torch.zeros((num_envs, max_steps), dtype=dtype, device=device)
      for name in self.METRIC_NAMES
    }

  def update(
    self,
    *,
    sample_mask: torch.Tensor,
    action_acceleration: torch.Tensor,
    slip_velocity: torch.Tensor,
    velocity_error: torch.Tensor,
    cross_axis_velocity: torch.Tensor,
  ) -> None:
    if self._next_step >= self.max_steps:
      raise RuntimeError("metric accumulator exceeds configured max_steps")
    if sample_mask.shape != (self.num_envs,) or sample_mask.dtype != torch.bool:
      raise ValueError("sample_mask must be bool with shape (num_envs,)")
    inputs = {
      "action_acceleration": action_acceleration,
      "slip_velocity": slip_velocity,
      "velocity_error": velocity_error,
      "cross_axis_velocity": cross_axis_velocity,
    }
    for name, value in inputs.items():
      if value.shape != (self.num_envs,):
        raise ValueError(f"{name} must have shape (num_envs,)")
      if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
      self._values[name][:, self._next_step] = value
    self._valid[:, self._next_step] = sample_mask
    self._next_step += 1

  def result(self, env_index: int) -> dict[str, dict[str, float | None]]:
    if env_index < 0 or env_index >= self.num_envs:
      raise IndexError("env_index outside batch")
    mask = self._valid[env_index, : self._next_step]
    return {
      name: _distribution(values[env_index, : self._next_step][mask])
      for name, values in self._values.items()
    }


def aggregate_distributions(
  scenarios: Iterable[Mapping[str, Any]], metric_name: str
) -> dict[str, float | None]:
  """Aggregate scenario-level raw samples or per-scenario means."""
  values: list[float] = []
  for scenario in scenarios:
    metric = scenario.get("sample_metrics", {}).get(metric_name, {})
    value = metric.get("mean")
    if value is not None:
      values.append(float(value))
  return _distribution(torch.tensor(values, dtype=torch.float64))


def matched_thresholds(
  *, s_action: float, arc_action: float, straight_action: float,
  s_slip: float, reference_slip: float, catastrophic_fraction: float,
) -> dict[str, bool]:
  """Evaluate the provisional matched-reference acceptance thresholds."""
  values = (
    s_action, arc_action, straight_action, s_slip, reference_slip,
    catastrophic_fraction,
  )
  if any(not math.isfinite(value) or value < 0.0 for value in values):
    raise ValueError("threshold inputs must be finite and nonnegative")
  return {
    "s_vs_arc_action_acceleration": s_action <= 1.2 * arc_action,
    "s_vs_straight_action_acceleration": s_action <= 1.3 * straight_action,
    "s_vs_reference_slip": s_slip <= 1.2 * reference_slip,
    "catastrophic_termination": catastrophic_fraction <= 0.05,
  }


def assert_recursive_finite(value: Any, path: str = "root") -> None:
  """Reject NaN/Inf recursively while allowing explicit ``None`` values."""
  if value is None or isinstance(value, (str, bool)):
    return
  if isinstance(value, (int, float)):
    if not math.isfinite(float(value)):
      raise ValueError(f"non-finite value at {path}: {value}")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      assert_recursive_finite(item, f"{path}.{key}")
    return
  if isinstance(value, Sequence):
    for index, item in enumerate(value):
      assert_recursive_finite(item, f"{path}[{index}]")
    return
  raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


__all__ = [
  "ACTION_ACCELERATION_DEFINITION",
  "DYNAMICS_EVENTS",
  "CORE_PROFILE_NAMES",
  "FACTOR_PROFILE_NAMES",
  "MatchedRouteContract",
  "OBSERVATION_EVENTS",
  "OnlineMatchedRouteMetrics",
  "PROFILE_NAMES",
  "PROFILES",
  "RANDOMIZATION_EVENTS",
  "ROUTE_KINDS",
  "action_acceleration",
  "aggregate_distributions",
  "assert_recursive_finite",
  "configure_matched_profile",
  "contact_slip_velocity",
  "matched_route_length",
  "matched_route_local_bounds",
  "matched_thresholds",
]
