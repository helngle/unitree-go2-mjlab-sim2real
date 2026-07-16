"""Online command-response metrics for batched route evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


AXIS_NAMES = ("vx", "vy", "wz")


@dataclass(frozen=True)
class TransientMetricConfig:
  control_dt: float
  num_segments: int
  rise_fraction: float = 0.9
  settling_fraction: float = 0.1
  target_epsilon: float = 1.0e-5

  def __post_init__(self) -> None:
    if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
      raise ValueError("control_dt must be finite and positive")
    if self.num_segments <= 0:
      raise ValueError("num_segments must be positive")
    if not 0.0 < self.rise_fraction <= 1.0:
      raise ValueError("rise_fraction must be in (0, 1]")
    if not 0.0 < self.settling_fraction < 1.0:
      raise ValueError("settling_fraction must be in (0, 1)")
    if not math.isfinite(self.target_epsilon) or self.target_epsilon <= 0.0:
      raise ValueError("target_epsilon must be finite and positive")


class OnlineCommandTransientMetrics:
  """Accumulate compact transient metrics without retaining rollout traces.

  ``segment_index`` is supplied by the evaluator.  A command tape therefore
  keeps its time-indexed segment contract, while a closed-loop controller can
  use its geometric active segment.  Updates with a false sample mask are
  ignored, which freezes reset or completed attempts.
  """

  def __init__(
    self,
    num_envs: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: TransientMetricConfig,
  ) -> None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    self.num_envs = num_envs
    self.config = config
    shape = (num_envs, config.num_segments)
    axis_shape = (*shape, 3)
    self.sample_count = torch.zeros(shape, dtype=torch.long, device=device)
    self.command_sum = torch.zeros(axis_shape, dtype=dtype, device=device)
    self.actual_sum = torch.zeros_like(self.command_sum)
    self.absolute_error_integral = torch.zeros_like(self.command_sum)
    self.target_count = torch.zeros(axis_shape, dtype=torch.long, device=device)
    self.first_target_step = torch.full(axis_shape, -1, dtype=torch.long, device=device)
    self.first_rise_step = torch.full_like(self.first_target_step, -1)
    self.last_outside_band_step = torch.full_like(self.first_target_step, -1)
    self.final_in_band = torch.zeros(axis_shape, dtype=torch.bool, device=device)
    self.overshoot_ratio_max = torch.zeros_like(self.command_sum)
    self.command_delta_sum = torch.zeros((num_envs, 3), dtype=dtype, device=device)
    self.command_delta_max = torch.zeros_like(self.command_delta_sum)
    self.command_delta_linf_sum = torch.zeros(num_envs, dtype=dtype, device=device)
    self.command_delta_linf_max = torch.zeros_like(self.command_delta_linf_sum)
    self.command_delta_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.saturation_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.active_sample_count = torch.zeros_like(self.saturation_count)
    self.previous_command = torch.zeros((num_envs, 3), dtype=dtype, device=device)
    self.has_previous_command = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.previous_segment = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self.transition_step = torch.full_like(self.previous_segment, -1)
    self.switch_target_yaw = torch.zeros(num_envs, dtype=dtype, device=device)
    self.sign_switch_latency_step = torch.full_like(self.previous_segment, -1)

  def _validate_update(
    self,
    command: torch.Tensor,
    actual: torch.Tensor,
    segment_index: torch.Tensor,
    sample_mask: torch.Tensor,
    saturated: torch.Tensor,
  ) -> None:
    if command.shape != (self.num_envs, 3) or actual.shape != command.shape:
      raise ValueError(f"command and actual must have shape ({self.num_envs}, 3)")
    if segment_index.shape != (self.num_envs,) or segment_index.dtype != torch.long:
      raise ValueError(f"segment_index must be long tensor shape ({self.num_envs},)")
    if sample_mask.shape != (self.num_envs,) or sample_mask.dtype != torch.bool:
      raise ValueError(f"sample_mask must be bool tensor shape ({self.num_envs},)")
    if saturated.shape != (self.num_envs,) or saturated.dtype != torch.bool:
      raise ValueError(f"saturated must be bool tensor shape ({self.num_envs},)")
    if torch.any((segment_index < 0) | (segment_index >= self.config.num_segments)):
      raise ValueError("segment_index is outside configured range")
    if not torch.isfinite(command).all() or not torch.isfinite(actual).all():
      raise ValueError("command and actual must be finite")

  def update(
    self,
    *,
    step_index: int,
    command: torch.Tensor,
    actual: torch.Tensor,
    segment_index: torch.Tensor,
    sample_mask: torch.Tensor,
    saturated: torch.Tensor | None = None,
  ) -> None:
    if step_index < 0:
      raise ValueError("step_index must be nonnegative")
    if saturated is None:
      saturated = torch.zeros_like(sample_mask)
    self._validate_update(command, actual, segment_index, sample_mask, saturated)
    active_ids = torch.where(sample_mask)[0]
    if active_ids.numel() == 0:
      return

    segments = segment_index.index_select(0, active_ids)
    command_active = command.index_select(0, active_ids)
    actual_active = actual.index_select(0, active_ids)
    self.active_sample_count[active_ids] += 1
    self.saturation_count[active_ids] += saturated.index_select(0, active_ids).long()

    had_previous = self.has_previous_command.index_select(0, active_ids)
    delta = (command_active - self.previous_command.index_select(0, active_ids)).abs()
    delta_ids = active_ids[had_previous]
    if delta_ids.numel() > 0:
      valid_delta = delta[had_previous]
      self.command_delta_sum[delta_ids] += valid_delta
      self.command_delta_max[delta_ids] = torch.maximum(
        self.command_delta_max.index_select(0, delta_ids), valid_delta
      )
      delta_linf = valid_delta.amax(dim=-1)
      self.command_delta_linf_sum[delta_ids] += delta_linf
      self.command_delta_linf_max[delta_ids] = torch.maximum(
        self.command_delta_linf_max.index_select(0, delta_ids), delta_linf
      )
      self.command_delta_count[delta_ids] += 1

    previous_segment = self.previous_segment.index_select(0, active_ids)
    changed = (previous_segment >= 0) & (segments != previous_segment)
    changed_ids = active_ids[changed]
    if changed_ids.numel() > 0:
      not_recorded = self.transition_step.index_select(0, changed_ids) < 0
      changed_ids = changed_ids[not_recorded]
      if changed_ids.numel() > 0:
        self.transition_step[changed_ids] = step_index
        self.switch_target_yaw[changed_ids] = command[changed_ids, 2]

    transition_seen = self.transition_step.index_select(0, active_ids) >= 0
    latency_missing = self.sign_switch_latency_step.index_select(0, active_ids) < 0
    target_yaw = self.switch_target_yaw.index_select(0, active_ids)
    target_valid = target_yaw.abs() > self.config.target_epsilon
    actual_yaw = actual_active[:, 2]
    sign_matches = actual_yaw * target_yaw > 0.0
    switched = transition_seen & latency_missing & target_valid & sign_matches
    switched_ids = active_ids[switched]
    if switched_ids.numel() > 0:
      self.sign_switch_latency_step[switched_ids] = step_index

    # A command-tape settle window is useful for the route lifecycle and
    # command-delta statistics, but it is not part of either motion segment.
    # Excluding all-zero commands prevents post-stop velocity from biasing the
    # per-segment response gains.
    response_sample = command_active.abs().amax(dim=-1) > self.config.target_epsilon
    response_ids = active_ids[response_sample]
    response_segments = segments[response_sample]
    target = command_active[response_sample]
    response = actual_active[response_sample]
    if response_ids.numel() == 0:
      self.previous_command[active_ids] = command_active
      self.has_previous_command[active_ids] = True
      self.previous_segment[active_ids] = segments
      return
    self.sample_count[response_ids, response_segments] += 1
    self.command_sum[response_ids, response_segments] += target
    self.actual_sum[response_ids, response_segments] += response
    self.absolute_error_integral[response_ids, response_segments] += (
      (response - target).abs() * self.config.control_dt
    )
    valid_target = target.abs() > self.config.target_epsilon
    first_target = self.first_target_step[response_ids, response_segments]
    new_target = valid_target & (first_target < 0)
    if new_target.any():
      rows, axes = torch.where(new_target)
      self.first_target_step[
        response_ids[rows], response_segments[rows], axes
      ] = step_index
    self.target_count[response_ids, response_segments] += valid_target.long()

    signed_response = response * torch.sign(target)
    reached = valid_target & (
      signed_response >= self.config.rise_fraction * target.abs()
    )
    rise_missing = self.first_rise_step[response_ids, response_segments] < 0
    newly_reached = reached & rise_missing
    if newly_reached.any():
      rows, axes = torch.where(newly_reached)
      self.first_rise_step[
        response_ids[rows], response_segments[rows], axes
      ] = step_index

    band = self.config.settling_fraction * target.abs()
    in_band = valid_target & ((response - target).abs() <= band)
    outside = valid_target & ~in_band
    if outside.any():
      rows, axes = torch.where(outside)
      self.last_outside_band_step[
        response_ids[rows], response_segments[rows], axes
      ] = step_index
    previous_in_band = self.final_in_band[response_ids, response_segments]
    self.final_in_band[response_ids, response_segments] = torch.where(
      valid_target, in_band, previous_in_band
    )

    overshoot = torch.where(
      valid_target,
      torch.clamp(signed_response - target.abs(), min=0.0)
      / target.abs().clamp_min(self.config.target_epsilon),
      0.0,
    )
    self.overshoot_ratio_max[response_ids, response_segments] = torch.maximum(
      self.overshoot_ratio_max[response_ids, response_segments], overshoot
    )
    self.previous_command[active_ids] = command_active
    self.has_previous_command[active_ids] = True
    self.previous_segment[active_ids] = segments

  @staticmethod
  def _optional_time(steps: int | None, control_dt: float) -> float | None:
    return None if steps is None else steps * control_dt

  def _axis_transient(self, env_index: int, segment: int, axis: int) -> dict[str, object]:
    target_samples = int(self.target_count[env_index, segment, axis])
    first_target = int(self.first_target_step[env_index, segment, axis])
    first_rise = int(self.first_rise_step[env_index, segment, axis])
    if target_samples == 0 or first_target < 0:
      rise_steps = None
      rise_reason = "no_nonzero_target"
    elif first_rise < 0:
      rise_steps = None
      rise_reason = "target_not_reached"
    else:
      rise_steps = first_rise - first_target
      rise_reason = None

    if target_samples == 0 or first_target < 0:
      settling_steps = None
      settling_reason = "no_nonzero_target"
    elif not bool(self.final_in_band[env_index, segment, axis]):
      settling_steps = None
      settling_reason = "outside_band_at_end"
    else:
      last_outside = int(self.last_outside_band_step[env_index, segment, axis])
      settle_at = max(first_target, last_outside + 1)
      settling_steps = settle_at - first_target
      settling_reason = None
    return {
      "rise_time_90_steps": rise_steps,
      "rise_time_90_s": self._optional_time(rise_steps, self.config.control_dt),
      "rise_time_reason": rise_reason,
      "overshoot_ratio_max": (
        float(self.overshoot_ratio_max[env_index, segment, axis])
        if target_samples > 0 else None
      ),
      "settling_time_10pct_steps": settling_steps,
      "settling_time_10pct_s": self._optional_time(
        settling_steps, self.config.control_dt
      ),
      "settling_time_reason": settling_reason,
    }

  def result(self, env_index: int) -> dict[str, object]:
    if env_index < 0 or env_index >= self.num_envs:
      raise IndexError("env_index outside batch")
    delta_count = max(int(self.command_delta_count[env_index]), 1)
    active_count = max(int(self.active_sample_count[env_index]), 1)
    transition = int(self.transition_step[env_index])
    switch_response = int(self.sign_switch_latency_step[env_index])
    if transition < 0:
      latency_steps = None
      latency_reason = "no_segment_transition"
    elif abs(float(self.switch_target_yaw[env_index])) <= self.config.target_epsilon:
      latency_steps = None
      latency_reason = "no_nonzero_yaw_target_after_transition"
    elif switch_response < 0:
      latency_steps = None
      latency_reason = "yaw_sign_not_reached"
    else:
      latency_steps = switch_response - transition
      latency_reason = None

    segments = []
    for segment in range(self.config.num_segments):
      count = int(self.sample_count[env_index, segment])
      denom = max(count, 1)
      command_mean = self.command_sum[env_index, segment] / denom
      actual_mean = self.actual_sum[env_index, segment] / denom
      gain: list[float | None] = []
      for axis in range(3):
        command_value = float(command_mean[axis])
        gain.append(
          float(actual_mean[axis] / command_mean[axis])
          if abs(command_value) > self.config.target_epsilon else None
        )
      segments.append({
        "segment_index": segment,
        "sample_count": count,
        "commanded_velocity_mean": [float(value) for value in command_mean],
        "actual_velocity_mean": [float(value) for value in actual_mean],
        "response_gain": dict(zip(AXIS_NAMES, gain, strict=True)),
        "integrated_absolute_error": {
          name: float(self.absolute_error_integral[env_index, segment, axis])
          for axis, name in enumerate(AXIS_NAMES)
        },
        "transient": {
          name: self._axis_transient(env_index, segment, axis)
          for axis, name in enumerate(AXIS_NAMES)
        },
      })
    transition_step = transition if transition >= 0 else None
    return {
      "control_dt": self.config.control_dt,
      "transition_step": transition_step,
      "transition_time_s": self._optional_time(transition_step, self.config.control_dt),
      "yaw_sign_switch_latency_steps": latency_steps,
      "yaw_sign_switch_latency_s": self._optional_time(
        latency_steps, self.config.control_dt
      ),
      "yaw_sign_switch_latency_reason": latency_reason,
      "command_delta_abs_mean": [
        float(value) for value in self.command_delta_sum[env_index] / delta_count
      ],
      "command_delta_abs_max": [
        float(value) for value in self.command_delta_max[env_index]
      ],
      "command_delta_linf_mean": float(
        self.command_delta_linf_sum[env_index] / delta_count
      ),
      "command_delta_linf_max": float(self.command_delta_linf_max[env_index]),
      "controller_saturation_fraction": float(
        self.saturation_count[env_index] / active_count
      ),
      "segments": segments,
    }


__all__ = [
  "AXIS_NAMES",
  "OnlineCommandTransientMetrics",
  "TransientMetricConfig",
]
