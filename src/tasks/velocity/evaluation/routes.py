"""Pure tensor geometry and lifecycle helpers for route evaluation.

The route frame uses +x along the route heading and +y to the left.  Inputs
are world-frame positions/headings; output velocity commands use the robot
body frame expected by ``UniformVelocityCommand``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RouteFrameState:
  progress: torch.Tensor
  cross_track: torch.Tensor
  heading_error: torch.Tensor


@dataclass(frozen=True)
class AttemptUpdate:
  """Lifecycle transition; ``sample_mask`` is the pre-transition active mask."""

  sample_mask: torch.Tensor
  completed_now: torch.Tensor
  failed_now: torch.Tensor
  active: torch.Tensor


def straight_route_initial_positions(
  nominal_position_w_xy: torch.Tensor,
  route_heading_w: torch.Tensor | float,
  start_forward_offset: torch.Tensor | float,
  cross_track_offset: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return nominal route starts and offset robot starts.

  Forward offset moves both positions along the route. Cross-track offset
  moves only the robot to the route's left, preserving the nominal centerline.
  """
  _validate_xy(nominal_position_w_xy, name="nominal_position_w_xy")
  batch = nominal_position_w_xy.shape[0]
  heading = _batch_vector(route_heading_w, batch, name="route_heading_w").to(
    nominal_position_w_xy
  )
  forward = _batch_vector(
    start_forward_offset, batch, name="start_forward_offset"
  ).to(nominal_position_w_xy)
  cross = _batch_vector(cross_track_offset, batch, name="cross_track_offset").to(
    nominal_position_w_xy
  )
  tangent = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
  normal = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
  route_start = nominal_position_w_xy + forward.unsqueeze(-1) * tangent
  robot_start = route_start + cross.unsqueeze(-1) * normal
  return route_start, robot_start


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  """Wrap angles to the half-open interval ``[-pi, pi)``."""
  if not isinstance(angle, torch.Tensor):
    raise TypeError("angle must be a torch.Tensor")
  return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _batch_vector(value: torch.Tensor | float, batch: int, *, name: str) -> torch.Tensor:
  tensor = torch.as_tensor(value)
  if tensor.ndim == 0:
    return tensor.expand(batch)
  if tensor.ndim == 1 and tensor.shape[0] == batch:
    return tensor
  raise ValueError(f"{name} must be scalar or shape ({batch},), got {tuple(tensor.shape)}")


def _validate_xy(value: torch.Tensor, *, name: str) -> None:
  if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 2:
    shape = getattr(value, "shape", None)
    raise ValueError(f"{name} must have shape (N, 2), got {shape}")


def route_frame_errors(
  position_w_xy: torch.Tensor,
  heading_w: torch.Tensor,
  route_start_w_xy: torch.Tensor,
  route_heading_w: torch.Tensor | float,
) -> RouteFrameState:
  """Return progress, left-positive cross-track, and heading error."""
  _validate_xy(position_w_xy, name="position_w_xy")
  _validate_xy(route_start_w_xy, name="route_start_w_xy")
  if position_w_xy.shape != route_start_w_xy.shape:
    raise ValueError("position_w_xy and route_start_w_xy must have equal shape")
  batch = position_w_xy.shape[0]
  if not isinstance(heading_w, torch.Tensor) or heading_w.ndim != 1 or heading_w.shape[0] != batch:
    raise ValueError(f"heading_w must have shape ({batch},)")
  route_heading = _batch_vector(route_heading_w, batch, name="route_heading_w").to(
    device=position_w_xy.device, dtype=position_w_xy.dtype
  )
  delta = position_w_xy - route_start_w_xy
  cos_h, sin_h = torch.cos(route_heading), torch.sin(route_heading)
  progress = delta[:, 0] * cos_h + delta[:, 1] * sin_h
  cross_track = -delta[:, 0] * sin_h + delta[:, 1] * cos_h
  return RouteFrameState(progress, cross_track, wrap_to_pi(heading_w - route_heading))


def world_to_body_velocity(linear_velocity_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate world XY velocities into each robot's body frame."""
  _validate_xy(linear_velocity_w, name="linear_velocity_w")
  if not isinstance(heading_w, torch.Tensor) or heading_w.ndim != 1 or heading_w.shape[0] != linear_velocity_w.shape[0]:
    raise ValueError(f"heading_w must have shape ({linear_velocity_w.shape[0]},)")
  cos_h, sin_h = torch.cos(heading_w), torch.sin(heading_w)
  x, y = linear_velocity_w.unbind(dim=-1)
  return torch.stack((cos_h * x + sin_h * y, -sin_h * x + cos_h * y), dim=-1)


def route_normal_velocity(
  linear_velocity_w: torch.Tensor, route_heading_w: torch.Tensor | float
) -> torch.Tensor:
  """Project world XY velocity onto the left-positive route normal."""
  _validate_xy(linear_velocity_w, name="linear_velocity_w")
  heading = _batch_vector(
    route_heading_w, linear_velocity_w.shape[0], name="route_heading_w"
  ).to(linear_velocity_w)
  normal = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
  return torch.sum(linear_velocity_w * normal, dim=-1)


def validate_route_parameters(
  *, route_length: float, control_dt: float | None = None,
  cross_track_tolerance: float | None = None, heading_tolerance: float | None = None,
  target_speed: float | None = None,
) -> None:
  """Validate positive route/control values before constructing a rollout."""
  values = {
    "route_length": route_length,
    "control_dt": control_dt,
    "cross_track_tolerance": cross_track_tolerance,
    "heading_tolerance": heading_tolerance,
    "target_speed": target_speed,
  }
  for name, value in values.items():
    if value is not None and (not math.isfinite(value) or value <= 0.0):
      raise ValueError(f"{name} must be finite and positive, got {value}")


def straight_line_controller(
  position_w_xy: torch.Tensor,
  heading_w: torch.Tensor,
  route_start_w_xy: torch.Tensor,
  route_heading_w: torch.Tensor | float,
  *,
  target_speed: torch.Tensor | float,
  cross_track_gain: torch.Tensor | float,
  heading_gain: torch.Tensor | float,
  max_lateral_speed: torch.Tensor | float,
  max_yaw_rate: torch.Tensor | float,
  route_length: torch.Tensor | float,
) -> torch.Tensor:
  """Generate ``[vx, vy, yaw_rate]`` body commands for a straight route."""
  state = route_frame_errors(position_w_xy, heading_w, route_start_w_xy, route_heading_w)
  batch = position_w_xy.shape[0]
  route_heading = _batch_vector(route_heading_w, batch, name="route_heading_w").to(position_w_xy)
  speed = _batch_vector(target_speed, batch, name="target_speed").to(position_w_xy)
  cross_gain = _batch_vector(cross_track_gain, batch, name="cross_track_gain").to(position_w_xy)
  heading_gain_b = _batch_vector(heading_gain, batch, name="heading_gain").to(position_w_xy)
  lateral_limit = _batch_vector(max_lateral_speed, batch, name="max_lateral_speed").to(position_w_xy)
  yaw_limit = _batch_vector(max_yaw_rate, batch, name="max_yaw_rate").to(position_w_xy)
  length = _batch_vector(route_length, batch, name="route_length").to(position_w_xy)
  for name, values in (("route_length", length), ("target_speed", speed), ("max_lateral_speed", lateral_limit), ("max_yaw_rate", yaw_limit)):
    if not torch.isfinite(values).all() or (values <= 0).any():
      raise ValueError(f"{name} must be finite and positive")
  if (cross_gain < 0).any() or (heading_gain_b < 0).any():
    raise ValueError("controller gains must be nonnegative")
  tangent = torch.stack((torch.cos(route_heading), torch.sin(route_heading)), dim=-1)
  normal = torch.stack((-torch.sin(route_heading), torch.cos(route_heading)), dim=-1)
  lateral = torch.clamp(-cross_gain * state.cross_track, -lateral_limit, lateral_limit)
  world_velocity = speed.unsqueeze(-1) * tangent + lateral.unsqueeze(-1) * normal
  body_velocity = world_to_body_velocity(world_velocity, heading_w)
  yaw = torch.clamp(-heading_gain_b * state.heading_error, -yaw_limit, yaw_limit)
  done = state.progress >= length
  command = torch.cat((body_velocity, yaw.unsqueeze(-1)), dim=-1)
  return torch.where(done.unsqueeze(-1), torch.zeros_like(command), command)


def update_attempt_status(
  active: torch.Tensor,
  progress: torch.Tensor,
  cross_track: torch.Tensor,
  heading_error: torch.Tensor,
  failure_mask: torch.Tensor,
  *, route_length: float | torch.Tensor,
  cross_track_tolerance: float | torch.Tensor,
  heading_tolerance: float | torch.Tensor,
) -> AttemptUpdate:
  """Freeze an attempt after completion or non-timeout failure/reset."""
  tensors = (active, progress, cross_track, heading_error, failure_mask)
  if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in tensors):
    raise ValueError("attempt lifecycle inputs must be 1-D tensors")
  if len({value.shape[0] for value in tensors}) != 1:
    raise ValueError("attempt lifecycle tensors must have equal batch size")
  if failure_mask.dtype != torch.bool or active.dtype != torch.bool:
    raise ValueError("active and failure_mask must be boolean tensors")
  length = _batch_vector(route_length, active.shape[0], name="route_length").to(progress)
  cross_tol = _batch_vector(cross_track_tolerance, active.shape[0], name="cross_track_tolerance").to(progress)
  heading_tol = _batch_vector(heading_tolerance, active.shape[0], name="heading_tolerance").to(progress)
  if (length <= 0).any() or (cross_tol <= 0).any() or (heading_tol <= 0).any():
    raise ValueError("route length and tolerances must be positive")
  completed = active & ~failure_mask & (progress >= length) & (cross_track.abs() <= cross_tol) & (heading_error.abs() <= heading_tol)
  failed = active & failure_mask & ~completed
  return AttemptUpdate(active, completed, failed, active & ~completed & ~failed)
