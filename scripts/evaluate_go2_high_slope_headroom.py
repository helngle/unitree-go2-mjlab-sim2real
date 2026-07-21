"""Evaluation-only controller-headroom A/B on matched high-slope routes.

This harness reuses the strict high-slope matched evaluator.  It reconstructs
a fresh environment for every ``profile x scale x route_kind`` arm and changes
only ``max_lateral_speed`` and ``max_yaw_rate`` by the requested scale.  A
small scoped trace captures the already-generated command tensor immediately
before policy inference so per-axis clamp counts use the exact same active
sample lifecycle as the underlying evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
import math
from pathlib import Path
import sys
from types import FrameType
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tyro

from mjlab.utils.torch import configure_torch_backends

import scripts.evaluate_go2_high_slope_matched as matched_evaluator
from scripts.evaluate_go2_high_slope_matched import HighSlopeMatchedConfig
from src.tasks.velocity.evaluation.high_slope_headroom import (
  HEADROOM_SCALES,
  ONLY_CHANGED_CONTROLLER_FIELDS,
  SATURATION_AXES,
  TARGET_STRATA,
  build_headroom_scenarios,
  per_axis_saturation,
  scale_controller_limits,
  validate_headroom_pair,
  validate_headroom_result,
)
from src.tasks.velocity.evaluation.high_slope_matched import (
  PROFILE_NAMES,
  ROUTE_KINDS,
  ROUTE_LENGTH_DEFINITION,
  effective_slope_parameters,
  geometry_preflight,
  validate_matched_result_invariants,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  ACTION_ACCELERATION_DEFINITION,
  ACTIVE_SAMPLE_DEFINITION,
  assert_recursive_json_finite,
)


@dataclass(frozen=True)
class HighSlopeHeadroomConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  profiles: tuple[str, ...] = ("clean",)
  controller_scales: tuple[float, ...] = HEADROOM_SCALES
  radii: tuple[float, ...] = (2.5,)
  speeds: tuple[float, ...] = (0.5,)
  turn_signs: tuple[int, ...] = (1, -1)
  repeats: int = 1
  steps: int = 2400
  settle_steps: int = 10
  seed: int = 42
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  base_max_lateral_speed: float = 0.3
  base_max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.30
  heading_tolerance: float = math.radians(20.0)
  corridor_half_width: float = 0.4
  output_file: str = "go2_high_slope_controller_headroom_ab.json"


def _matched_config(
  cfg: HighSlopeHeadroomConfig, scale: float
) -> HighSlopeMatchedConfig:
  limits = scale_controller_limits(
    cfg.base_max_lateral_speed, cfg.base_max_yaw_rate, scale
  )
  return HighSlopeMatchedConfig(
    checkpoint=cfg.checkpoint,
    task_id=cfg.task_id,
    profiles=cfg.profiles,
    slope_directions=tuple(item[0] for item in TARGET_STRATA),
    levels=tuple(item[1] for item in TARGET_STRATA),
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    repeats=cfg.repeats,
    steps=cfg.steps,
    settle_steps=cfg.settle_steps,
    seed=cfg.seed,
    cross_track_gain=cfg.cross_track_gain,
    heading_gain=cfg.heading_gain,
    max_lateral_speed=limits.max_lateral_speed,
    max_yaw_rate=limits.max_yaw_rate,
    cross_track_tolerance=cfg.cross_track_tolerance,
    heading_tolerance=cfg.heading_tolerance,
    corridor_half_width=cfg.corridor_half_width,
    output_file=cfg.output_file,
  )


def _validate_config(cfg: HighSlopeHeadroomConfig) -> dict[str, object]:
  if tuple(float(x) for x in cfg.controller_scales) != HEADROOM_SCALES:
    raise ValueError(
      f"controller_scales must be the strict A/B pair {HEADROOM_SCALES}"
    )
  if not cfg.profiles or len(set(cfg.profiles)) != len(cfg.profiles):
    raise ValueError("profiles must be nonempty and unique")
  if any(profile not in PROFILE_NAMES for profile in cfg.profiles):
    raise ValueError(f"profiles must contain only {PROFILE_NAMES}")
  scenarios = build_headroom_scenarios(
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    repeats=cfg.repeats,
  )
  if not scenarios:
    raise ValueError("headroom scenario matrix must not be empty")
  # Reuse all strict numeric, horizon, and geometry checks.  The base config's
  # direction/level Cartesian product is validation-only; the rollout receives
  # the explicitly filtered TARGET_STRATA scenarios above.
  matched_evaluator._validate_config(_matched_config(cfg, 1.0))
  return geometry_preflight(cfg.radii, cfg.turn_signs)


def _command_injection_line() -> int:
  """Locate the first exact command write in the reused evaluator."""
  source, start = inspect.getsourcelines(matched_evaluator._evaluate_route_kind)
  marker = "command_term.vel_command_b[:] = command"
  matches = [start + index for index, line in enumerate(source) if line.strip() == marker]
  if len(matches) != 2:
    raise RuntimeError(
      "matched evaluator command injection contract changed; expected two writes"
    )
  return matches[0]


class _AxisSaturationCapture:
  """Capture exact per-axis saturation at the matched command injection site."""

  def __init__(self) -> None:
    self.target_code = matched_evaluator._evaluate_route_kind.__code__
    self.target_line = _command_injection_line()
    self.previous_trace: Callable[..., Any] | None = None
    self.counts: torch.Tensor | None = None
    self.any_counts: torch.Tensor | None = None
    self.denominators: torch.Tensor | None = None
    self.error: BaseException | None = None

  def _local_trace(
    self, frame: FrameType, event: str, arg: object
  ) -> Callable[..., Any]:
    del arg
    if event == "line" and frame.f_lineno == self.target_line:
      try:
        command = frame.f_locals["command"]
        active = frame.f_locals["active"]
        motion_active = frame.f_locals["motion_active"]
        cfg = frame.f_locals["cfg"]
        flags = per_axis_saturation(
          command,
          max_lateral_speed=cfg.max_lateral_speed,
          max_yaw_rate=cfg.max_yaw_rate,
        ) & motion_active.unsqueeze(-1)
        if self.counts is None:
          self.counts = torch.zeros_like(flags, dtype=torch.long)
          self.any_counts = torch.zeros_like(active, dtype=torch.long)
          self.denominators = torch.zeros_like(active, dtype=torch.long)
        self.counts += flags.long()
        if self.any_counts is None or self.denominators is None:
          raise RuntimeError("capture accumulators were not initialized")
        self.any_counts += flags.any(dim=-1).long()
        self.denominators += active.long()
      except BaseException as exc:  # surfaced immediately after the rollout
        self.error = exc
    return self._local_trace

  def _global_trace(
    self, frame: FrameType, event: str, arg: object
  ) -> Callable[..., Any] | None:
    del arg
    if event == "call" and frame.f_code is self.target_code:
      return self._local_trace
    return None

  def __enter__(self) -> _AxisSaturationCapture:
    self.previous_trace = sys.gettrace()
    if self.previous_trace is not None:
      raise RuntimeError("cannot nest headroom capture inside another trace")
    sys.settrace(self._global_trace)
    return self

  def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
    del exc_type, exc, traceback
    sys.settrace(self.previous_trace)

  def summaries(
    self, expected_envs: int
  ) -> tuple[list[dict[str, dict[str, int | float]]], list[int]]:
    if self.error is not None:
      raise RuntimeError("per-axis saturation capture failed") from self.error
    if self.counts is None or self.any_counts is None or self.denominators is None:
      raise RuntimeError("per-axis saturation capture observed no command steps")
    if self.counts.shape != (expected_envs, 3):
      raise RuntimeError("per-axis saturation capture shape mismatch")
    counts = self.counts.detach().cpu()
    any_counts = self.any_counts.detach().cpu()
    denominators = self.denominators.detach().cpu()
    outputs = []
    for index in range(expected_envs):
      denominator = int(denominators[index])
      outputs.append({
        name: {
          "count": int(counts[index, axis]),
          "rate": int(counts[index, axis]) / max(denominator, 1),
          "denominator": denominator,
        }
        for axis, name in enumerate(SATURATION_AXES)
      })
    return outputs, [int(value) for value in any_counts]


def _evaluate_route_kind_with_axis_saturation(
  cfg: HighSlopeMatchedConfig,
  profile: str,
  route_kind: str,
  scenarios: list[dict[str, Any]],
  *,
  fresh_environment_identity: str,
) -> dict[str, Any]:
  with _AxisSaturationCapture() as capture:
    result = matched_evaluator._evaluate_route_kind(
      cfg, profile, route_kind, scenarios
    )
  summaries, any_counts = capture.summaries(len(scenarios))
  for row, summary, any_count in zip(
    result["scenarios"], summaries, any_counts, strict=True
  ):
    if summary["vx"]["count"] != 0:
      raise RuntimeError("vx was unexpectedly reported as controller-clamped")
    existing_any_count = int(row["controller_saturation_count"])
    if existing_any_count != any_count:
      raise RuntimeError("per-axis capture disagrees with aggregate saturation")
    if int(summary["vy"]["denominator"]) != int(row["steps_sampled"]):
      raise RuntimeError("per-axis saturation denominator diverged")
    row["controller_saturation_by_axis"] = summary
  result["fresh_environment_identity"] = fresh_environment_identity
  result["per_axis_saturation_definition"] = {
    "vx": "not_clamped_in_this_experiment",
    "vy": "abs(commanded_vy) >= max_lateral_speed - 1e-6",
    "wz": "abs(commanded_wz) >= max_yaw_rate - 1e-6",
    "denominator": ACTIVE_SAMPLE_DEFINITION,
  }
  return result


def evaluate(cfg: HighSlopeHeadroomConfig) -> dict[str, Any]:
  preflight = _validate_config(cfg)
  scenarios = [item.as_dict() for item in build_headroom_scenarios(
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    repeats=cfg.repeats,
  )]
  checkpoint = str(Path(cfg.checkpoint).expanduser().resolve())
  profiles: dict[str, Any] = {}
  invocation = 0
  for profile in cfg.profiles:
    scale_results: dict[str, Any] = {}
    for scale in cfg.controller_scales:
      matched_cfg = _matched_config(cfg, scale)
      limits = scale_controller_limits(
        cfg.base_max_lateral_speed, cfg.base_max_yaw_rate, scale
      )
      route_results: dict[str, Any] = {}
      for route_kind in ROUTE_KINDS:
        invocation += 1
        route_results[route_kind] = _evaluate_route_kind_with_axis_saturation(
          matched_cfg,
          profile,
          route_kind,
          scenarios,
          fresh_environment_identity=(
            f"fresh-env-{invocation}:{profile}:scale={scale:.1f}:{route_kind}"
          ),
        )
      validate_matched_result_invariants(route_results)
      profile_settings = [
        route_results[kind]["profile_settings"] for kind in ROUTE_KINDS
      ]
      if any(item != profile_settings[0] for item in profile_settings[1:]):
        raise RuntimeError("profile settings differ across fresh environments")
      scale_results[f"{scale:.1f}"] = {
        "controller_scale": float(scale),
        "base_controller_limits": {
          "max_lateral_speed": cfg.base_max_lateral_speed,
          "max_yaw_rate": cfg.base_max_yaw_rate,
        },
        "effective_controller_limits": limits.as_dict(),
        "matched_slot_order": [row["matched_slot"] for row in scenarios],
        "profile_settings": profile_settings[0],
        "fresh_environment_per_route_kind": True,
        "route_results": route_results,
      }
    validate_headroom_pair(
      scale_results,
      base_lateral_speed=cfg.base_max_lateral_speed,
      base_yaw_rate=cfg.base_max_yaw_rate,
      scales=cfg.controller_scales,
    )
    profiles[profile] = {
      "profile": profile,
      "scale_results": scale_results,
    }
  result = {
    "schema_version": 1,
    "evaluation_suite": "high_slope_controller_headroom_ab",
    "git_head": matched_evaluator._git_head(),
    "checkpoint": checkpoint,
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "config": asdict(cfg),
    "geometry_preflight": preflight,
    "ab_invariants": {
      "controller_scales": list(cfg.controller_scales),
      "only_changed_controller_fields": list(ONLY_CHANGED_CONTROLLER_FIELDS),
      "base_controller_limits": {
        "max_lateral_speed": cfg.base_max_lateral_speed,
        "max_yaw_rate": cfg.base_max_yaw_rate,
      },
      "checkpoint": checkpoint,
      "task_id": cfg.task_id,
      "seed": cfg.seed,
      "target_strata": [
        effective_slope_parameters(direction, level)
        for direction, level in TARGET_STRATA
      ],
      "route_kinds": list(ROUTE_KINDS),
      "route_length_definition": ROUTE_LENGTH_DEFINITION,
      "same_seed_environment_reconstruction": True,
      "fresh_environment_per_profile_scale_route_kind": True,
    },
    "metric_invariants": {
      "sample_denominator": ACTIVE_SAMPLE_DEFINITION,
      "action_acceleration_definition": ACTION_ACCELERATION_DEFINITION,
      "attempt_freeze": (
        "inherited strict matched lifecycle: terminal step included and all "
        "samples after reset or fixed completion settle are excluded"
      ),
    },
    "coverage": {
      "evaluation_only": True,
      "training_changed": False,
      "policy_changed": False,
      "terrain_changed_between_ab_arms": False,
      "randomization_changed_between_ab_arms": False,
      "matched_straight_arc_s_curve": True,
      "slope_up_high": True,
      "slope_down_extreme": True,
    },
    "profiles": profiles,
  }
  validate_headroom_result(result)
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(HighSlopeHeadroomConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print(json.dumps(result, indent=2, allow_nan=False))
  print(f"[INFO] Wrote high-slope headroom A/B evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
