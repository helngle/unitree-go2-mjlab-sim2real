"""Pure geometry helpers for fixed-radius arcs and continuous S routes.

The route frame is left-handed in the usual robotics sense: positive signed
curvature turns left and negative curvature turns right.  All poses are world
frame; commands returned by the controllers are ``[vx, vy, yaw_rate]`` in the
robot body frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .routes import route_normal_velocity, world_to_body_velocity, wrap_to_pi


def _as_batch(value: torch.Tensor | float, batch: int, ref: torch.Tensor, name: str) -> torch.Tensor:
  out = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
  if out.ndim == 0:
    return out.expand(batch)
  if out.ndim == 1 and out.shape[0] == batch:
    return out
  raise ValueError(f"{name} must be scalar or shape ({batch},), got {tuple(out.shape)}")


def _validate_sign(turn_sign: int | float) -> float:
  sign = float(turn_sign)
  if sign not in (-1.0, 1.0):
    raise ValueError(f"turn_sign must be +1 (left) or -1 (right), got {turn_sign}")
  return sign


@dataclass(frozen=True)
class ArcSpec:
  """A constant-curvature arc, with positive angle/length magnitude."""

  radius: float
  turn_sign: int
  angle: float = math.pi / 2.0

  def __post_init__(self) -> None:
    _validate_sign(self.turn_sign)
    if not math.isfinite(self.radius) or self.radius <= 0.0:
      raise ValueError("radius must be finite and positive")
    if not math.isfinite(self.angle) or self.angle <= 0.0:
      raise ValueError("angle must be finite and positive")

  @property
  def curvature(self) -> float:
    return float(self.turn_sign) / self.radius

  @property
  def length(self) -> float:
    return self.radius * self.angle


@dataclass(frozen=True)
class ArcRoute:
  start_xy: torch.Tensor
  start_heading: torch.Tensor | float
  spec: ArcSpec

  @property
  def length(self) -> float:
    return self.spec.length

  def pose_at(self, progress: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world XY and tangent heading at clamped arc-length progress."""
    if not isinstance(self.start_xy, torch.Tensor) or self.start_xy.shape[-1] != 2:
      raise ValueError("start_xy must have final dimension 2")
    p = torch.as_tensor(progress, device=self.start_xy.device, dtype=self.start_xy.dtype)
    s = p.clamp(0.0, self.length)
    h0 = torch.as_tensor(self.start_heading, device=self.start_xy.device, dtype=self.start_xy.dtype)
    k = self.spec.curvature
    t = torch.stack((torch.cos(h0), torch.sin(h0)), dim=-1)
    n = torch.stack((-torch.sin(h0), torch.cos(h0)), dim=-1)
    # Stable k -> 0 form is unnecessary for the configured finite radii, but
    # this expression is exact for both left and right turns.
    xy = self.start_xy + (torch.sin(k * s) / k).unsqueeze(-1) * t + ((1.0 - torch.cos(k * s)) / k).unsqueeze(-1) * n
    heading = h0 + k * s
    return xy, wrap_to_pi(heading)

  def command_tape(self, speed: float | torch.Tensor) -> torch.Tensor:
    """Return a constant body-frame command for ideal arc tracking."""
    v = torch.as_tensor(speed, device=self.start_xy.device, dtype=self.start_xy.dtype)
    if torch.any(~torch.isfinite(v)) or torch.any(v <= 0):
      raise ValueError("speed must be finite and positive")
    zeros = torch.zeros_like(v)
    yaw = v * self.spec.curvature
    return torch.stack((v, zeros, yaw), dim=-1)


@dataclass(frozen=True)
class SRoute:
  """Two equal-radius opposite arcs with continuous position and tangent."""

  first: ArcRoute
  second: ArcRoute

  @property
  def length(self) -> float:
    return self.first.length + self.second.length

  def pose_at(self, progress: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.as_tensor(progress, device=self.first.start_xy.device, dtype=self.first.start_xy.dtype)
    p_clamped = p.clamp(0.0, self.length)
    first_p = p_clamped.clamp(0.0, self.first.length)
    first_xy, first_h = self.first.pose_at(first_p)
    second_p = (p_clamped - self.first.length).clamp(0.0, self.second.length)
    second_xy, second_h = self.second.pose_at(second_p)
    use_second = (p_clamped > self.first.length).unsqueeze(-1)
    xy = torch.where(use_second, second_xy, first_xy)
    heading = torch.where(p_clamped > self.first.length, second_h, first_h)
    return xy, heading


@dataclass(frozen=True)
class CommandTapeSchedule:
  """Ideal time schedule independent of the robot's realized progress."""

  speed: float
  first_curvature: float
  second_curvature: float
  first_motion_steps: int
  second_motion_steps: int
  settle_steps: int

  @property
  def motion_steps(self) -> int:
    return self.first_motion_steps + self.second_motion_steps

  @property
  def total_steps(self) -> int:
    return self.motion_steps + self.settle_steps

  def command_at(
    self,
    step_index: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
  ) -> torch.Tensor:
    if step_index < 0:
      raise ValueError("step_index must be nonnegative")
    curvature = 0.0
    if step_index < self.first_motion_steps:
      curvature = self.first_curvature
    elif step_index < self.motion_steps:
      curvature = self.second_curvature
    if step_index >= self.motion_steps:
      return torch.zeros(3, device=device, dtype=dtype)
    return torch.tensor(
      [self.speed, 0.0, curvature * self.speed], device=device, dtype=dtype
    )

  def segment_at(self, step_index: int) -> int:
    """Return the time-scheduled route segment, independent of robot state."""
    if step_index < 0:
      raise ValueError("step_index must be nonnegative")
    if self.second_motion_steps == 0:
      return 0
    return int(step_index >= self.first_motion_steps)

  def completion_allowed(self, step_index: int | torch.Tensor) -> bool | torch.Tensor:
    """Whether geometric completion may be accepted after settle ends."""
    if isinstance(step_index, torch.Tensor):
      return step_index >= self.total_steps
    return step_index >= self.total_steps

  def tape_finished(self, step_index: int | torch.Tensor) -> bool | torch.Tensor:
    """Whether the settle window has ended after this step count."""
    if isinstance(step_index, torch.Tensor):
      return step_index >= self.total_steps
    return step_index >= self.total_steps


def make_command_tape_schedule(
  route_kind: str,
  radius: float,
  speed: float,
  turn_sign: int,
  control_dt: float,
  settle_steps: int = 10,
) -> CommandTapeSchedule:
  """Construct an ideal arc/S command tape using ceil time discretization."""
  sign = _validate_sign(turn_sign)
  for name, value in (("radius", radius), ("speed", speed), ("control_dt", control_dt)):
    if not math.isfinite(value) or value <= 0.0:
      raise ValueError(f"{name} must be finite and positive")
  if settle_steps < 0:
    raise ValueError("settle_steps must be nonnegative")
  if route_kind == "arc":
    first_length = radius * math.pi / 2.0
    second_steps = 0
  elif route_kind == "s_curve":
    first_length = radius * math.pi / 3.0
    second_steps = math.ceil(first_length / (speed * control_dt))
  else:
    raise ValueError("route_kind must be 'arc' or 's_curve'")
  first_steps = math.ceil(first_length / (speed * control_dt))
  return CommandTapeSchedule(
    speed=speed,
    first_curvature=sign / radius,
    second_curvature=(-sign / radius if route_kind == "s_curve" else 0.0),
    first_motion_steps=first_steps,
    second_motion_steps=second_steps,
    settle_steps=settle_steps,
  )


def make_arc_route(start_xy: torch.Tensor, start_heading: torch.Tensor | float, radius: float, turn_sign: int, *, angle: float = math.pi / 2.0) -> ArcRoute:
  return ArcRoute(start_xy=start_xy, start_heading=start_heading, spec=ArcSpec(radius, turn_sign, angle))


def make_s_route(start_xy: torch.Tensor, start_heading: torch.Tensor | float, radius: float, first_turn_sign: int, *, angle: float = math.pi / 3.0) -> SRoute:
  first = make_arc_route(start_xy, start_heading, radius, first_turn_sign, angle=angle)
  end_xy, end_heading = first.pose_at(first.length)
  second = make_arc_route(end_xy, end_heading, radius, -int(first_turn_sign), angle=angle)
  return SRoute(first, second)


def _arc_projection(route: ArcRoute, position_w_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if position_w_xy.ndim != 2 or position_w_xy.shape[-1] != 2:
    raise ValueError("position_w_xy must have shape (N, 2)")
  batch = position_w_xy.shape[0]
  start = route.start_xy
  if start.ndim == 1:
    start = start.expand(batch, -1)
  if start.shape != position_w_xy.shape:
    raise ValueError("start_xy must be shape (2,) or match position batch")
  h0 = _as_batch(route.start_heading, batch, position_w_xy, "start_heading")
  n = torch.stack((-torch.sin(h0), torch.cos(h0)), dim=-1)
  center = start + n / route.spec.curvature
  radial = position_w_xy - center
  phi = torch.atan2(radial[:, 1], radial[:, 0])
  radial0 = -n / route.spec.curvature
  phi0 = torch.atan2(radial0[:, 1], radial0[:, 0])
  signed_delta = wrap_to_pi(phi - phi0) * float(route.spec.turn_sign)
  progress = signed_delta.clamp(0.0, route.spec.angle) * route.spec.radius
  expected_xy, expected_h = route.pose_at(progress)
  tangent = torch.stack((torch.cos(expected_h), torch.sin(expected_h)), dim=-1)
  normal = torch.stack((-torch.sin(expected_h), torch.cos(expected_h)), dim=-1)
  cross = torch.sum((position_w_xy - expected_xy) * normal, dim=-1)
  return progress, cross, expected_h


def arc_route_errors(route: ArcRoute, position_w_xy: torch.Tensor, heading_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return progress, signed cross-track, and heading error for an arc."""
  if heading_w.ndim != 1 or heading_w.shape[0] != position_w_xy.shape[0]:
    raise ValueError("heading_w must have shape (N,)")
  progress, cross, expected_h = _arc_projection(route, position_w_xy)
  return progress, cross, wrap_to_pi(heading_w - expected_h)


def s_route_errors(route: SRoute, position_w_xy: torch.Tensor, heading_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return global progress and local errors for the active S-route arc."""
  p1, c1, e1 = arc_route_errors(route.first, position_w_xy, heading_w)
  p2, c2, e2 = arc_route_errors(route.second, position_w_xy, heading_w)
  use_second = p1 >= route.first.length - 1.0e-5
  return (
    torch.where(use_second, route.first.length + p2, p1),
    torch.where(use_second, c2, c1),
    torch.where(use_second, e2, e1),
  )


def arc_command_controller(route: ArcRoute, position_w_xy: torch.Tensor, heading_w: torch.Tensor, *, target_speed: float, cross_track_gain: float = 1.2, heading_gain: float = 1.0, max_lateral_speed: float = 0.3, max_yaw_rate: float = 0.7) -> torch.Tensor:
  """Closed-loop body command for an arc; zero after its endpoint."""
  progress, cross, heading_error = arc_route_errors(route, position_w_xy, heading_w)
  _, expected_h = route.pose_at(progress)
  tangent = torch.stack((torch.cos(expected_h), torch.sin(expected_h)), dim=-1)
  normal = torch.stack((-torch.sin(expected_h), torch.cos(expected_h)), dim=-1)
  lateral = torch.clamp(-cross_track_gain * cross, -max_lateral_speed, max_lateral_speed)
  world_v = target_speed * tangent + lateral.unsqueeze(-1) * normal
  body_v = world_to_body_velocity(world_v, heading_w)
  yaw = torch.clamp(route.spec.curvature * target_speed - heading_gain * heading_error, -max_yaw_rate, max_yaw_rate)
  command = torch.cat((body_v, yaw.unsqueeze(-1)), dim=-1)
  done = progress >= route.length - 1.0e-5
  return torch.where(done.unsqueeze(-1), torch.zeros_like(command), command)


def s_command_controller(route: SRoute, position_w_xy: torch.Tensor, heading_w: torch.Tensor, *, target_speed: float, cross_track_gain: float = 1.2, heading_gain: float = 1.0, max_lateral_speed: float = 0.3, max_yaw_rate: float = 0.7) -> torch.Tensor:
  """Closed-loop controller for the two-arc S route."""
  # Select the active arc by geometric progress from the first endpoint.
  p1, _, _ = arc_route_errors(route.first, position_w_xy, heading_w)
  progress, cross, heading_error = s_route_errors(route, position_w_xy, heading_w)
  use_second = p1 >= route.first.length - 1.0e-5
  _, expected_h = route.pose_at(progress)
  tangent = torch.stack((torch.cos(expected_h), torch.sin(expected_h)), dim=-1)
  normal = torch.stack((-torch.sin(expected_h), torch.cos(expected_h)), dim=-1)
  lateral = torch.clamp(-cross_track_gain * cross, -max_lateral_speed, max_lateral_speed)
  world_v = target_speed * tangent + lateral.unsqueeze(-1) * normal
  body_v = world_to_body_velocity(world_v, heading_w)
  curvature = torch.where(use_second, torch.full_like(progress, route.second.spec.curvature), torch.full_like(progress, route.first.spec.curvature))
  yaw = torch.clamp(curvature * target_speed - heading_gain * heading_error, -max_yaw_rate, max_yaw_rate)
  command = torch.cat((body_v, yaw.unsqueeze(-1)), dim=-1)
  done = progress >= route.length - 1.0e-5
  return torch.where(done.unsqueeze(-1), torch.zeros_like(command), command)


__all__ = ["ArcSpec", "ArcRoute", "SRoute", "CommandTapeSchedule", "make_arc_route", "make_s_route", "make_command_tape_schedule", "arc_route_errors", "s_route_errors", "arc_command_controller", "s_command_controller"]
