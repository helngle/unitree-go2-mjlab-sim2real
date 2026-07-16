"""Evaluate V7 on high terrain curves or straight continuous transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro
import torch

from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  CONTINUOUS_TRANSITION_KEYS,
  CONTINUOUS_TRANSITION_KINDS,
  HIGH_DIFFICULTIES,
  boundary_transition_difficulty_matrix,
  boundary_transition_metadata,
  continuous_boundary_coverage,
  continuous_feature_heightfield,
  difficulty_for_high_level,
  effective_high_terrain_parameters,
  make_boundary_transition_generator,
  make_high_difficulty_curve_generator,
  reject_curved_transition,
  validate_straight_transition_footprint,
)
from src.tasks.velocity.evaluation.terrain_curved_routes import (
  DEFAULT_CORRIDOR_HALF_WIDTH,
  TERRAIN_CURVE_KINDS,
  validate_route_footprint,
)


@dataclass(frozen=True)
class TerrainBoundaryConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  suite: str = "high_curves"
  route_kind: str = "arc"
  terrain_kinds: tuple[str, ...] = TERRAIN_CURVE_KINDS
  transition_cases: tuple[str, ...] = CONTINUOUS_TRANSITION_KINDS
  transition_levels: tuple[int, ...] = (7, 9)
  radii: tuple[float, ...] = (2.5, 4.0)
  speeds: tuple[float, ...] = (0.3, 0.5)
  turn_signs: tuple[int, ...] = (1, -1)
  cross_track_offsets: tuple[float, ...] = (0.0,)
  yaw_offsets: tuple[float, ...] = (0.0,)
  repeats: int = 1
  settle_steps: int = 10
  cross_track_gain: float = 1.2
  heading_gain: float = 1.0
  max_lateral_speed: float = 0.3
  max_yaw_rate: float = 0.7
  cross_track_tolerance: float = 0.30
  heading_tolerance: float = math.radians(20.0)
  corridor_half_width: float = DEFAULT_CORRIDOR_HALF_WIDTH
  steps: int = 2400
  seed: int = 42
  profile: str = "clean"
  output_file: str = "go2_terrain_boundary_evaluation.json"


class _OnlineRouteErrorMetrics:
  """Retain absolute route errors under the evaluator's lifecycle mask."""

  def __init__(
    self,
    num_envs: int,
    max_steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
  ) -> None:
    if num_envs <= 0 or max_steps <= 0:
      raise ValueError("num_envs and max_steps must be positive")
    self.num_envs = num_envs
    self.max_steps = max_steps
    self._next_step = 0
    shape = (num_envs, max_steps)
    self._valid = torch.zeros(shape, dtype=torch.bool, device=device)
    self._cross = torch.zeros(shape, dtype=dtype, device=device)
    self._heading = torch.zeros(shape, dtype=dtype, device=device)

  def update(
    self,
    cross_track: torch.Tensor,
    heading: torch.Tensor,
    sample_mask: torch.Tensor,
  ) -> None:
    if self._next_step >= self.max_steps:
      raise RuntimeError("route error accumulator exceeds configured max_steps")
    for name, value in (
      ("cross_track", cross_track),
      ("heading", heading),
    ):
      if value.shape != (self.num_envs,) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape (num_envs,)")
    if sample_mask.shape != (self.num_envs,) or sample_mask.dtype != torch.bool:
      raise ValueError("sample_mask must be bool with shape (num_envs,)")
    column = self._next_step
    self._valid[:, column] = sample_mask
    self._cross[:, column] = cross_track.abs()
    self._heading[:, column] = heading.abs()
    self._next_step += 1

  def result(self, env_index: int) -> dict[str, object]:
    if not 0 <= env_index < self.num_envs:
      raise IndexError("env_index outside batch")
    valid = self._valid[env_index, : self._next_step]

    def distribution(values: torch.Tensor) -> dict[str, float | str | None]:
      values = values[valid].to(dtype=torch.float64)
      if values.numel() == 0:
        return {
          "mean": None,
          "p95": None,
          "max": None,
          "reason": "no_active_control_step_samples",
        }
      return {
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
      }

    return {
      "active_control_step_samples": int(valid.sum()),
      "cross_track_absolute": distribution(
        self._cross[env_index, : self._next_step]
      ),
      "heading_absolute": distribution(
        self._heading[env_index, : self._next_step]
      ),
    }


def _git_head() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1],
    text=True,
  ).strip()


def _strict_json_finite(value: object, path: str = "root") -> None:
  if isinstance(value, float) and not math.isfinite(value):
    raise ValueError(f"non-finite JSON value at {path}: {value}")
  if isinstance(value, dict):
    for key, item in value.items():
      _strict_json_finite(item, f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _strict_json_finite(item, f"{path}[{index}]")


def _validate_config(cfg: TerrainBoundaryConfig) -> None:
  if cfg.suite not in ("high_curves", "continuous_straight"):
    raise ValueError("suite must be 'high_curves' or 'continuous_straight'")
  if cfg.repeats <= 0 or cfg.steps <= 0:
    raise ValueError("repeats and steps must be positive")
  if not cfg.speeds or any(
    not math.isfinite(speed) or speed <= 0.0 for speed in cfg.speeds
  ):
    raise ValueError("speeds must be finite and positive")
  if cfg.suite == "high_curves":
    if cfg.route_kind not in ("arc", "s_curve"):
      raise ValueError("high_curves route_kind must be 'arc' or 's_curve'")
    unknown = set(cfg.terrain_kinds) - set(TERRAIN_CURVE_KINDS)
    if unknown or not cfg.terrain_kinds:
      raise ValueError(f"unsupported high curve terrain kinds: {sorted(unknown)}")
    if not cfg.radii or any(
      not math.isfinite(radius) or radius <= 0.0 for radius in cfg.radii
    ):
      raise ValueError("radii must be finite and positive")
    if not cfg.turn_signs or any(sign not in (-1, 1) for sign in cfg.turn_signs):
      raise ValueError("turn_signs must contain only -1 and +1")
    for radius in cfg.radii:
      for turn_sign in cfg.turn_signs:
        validate_route_footprint(
          cfg.route_kind,
          radius,
          turn_sign,
          corridor_half_width=cfg.corridor_half_width,
        )
  else:
    if cfg.route_kind != "straight":
      for kind in cfg.transition_cases:
        reject_curved_transition(kind, cfg.route_kind)
    if not cfg.transition_cases:
      raise ValueError("transition_cases must not be empty")
    unknown = set(cfg.transition_cases) - set(CONTINUOUS_TRANSITION_KINDS)
    if unknown:
      raise ValueError(f"unsupported transition cases: {sorted(unknown)}")
    if not cfg.transition_levels or any(
      not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 9
      for level in cfg.transition_levels
    ):
      raise ValueError("transition_levels must contain integer rows in [0, 9]")
    validate_straight_transition_footprint(
      corridor_half_width=cfg.corridor_half_width
    )


def _evaluate_high_curves(cfg: TerrainBoundaryConfig) -> dict[str, Any]:
  import scripts.evaluate_go2_terrain_curves as terrain_curves

  base_cfg = terrain_curves.TerrainCurvedRouteConfig(
    checkpoint=cfg.checkpoint,
    task_id=cfg.task_id,
    route_kind=cfg.route_kind,
    terrain_kinds=cfg.terrain_kinds,
    terrain_levels=(0, 1),
    radii=cfg.radii,
    speeds=cfg.speeds,
    turn_signs=cfg.turn_signs,
    cross_track_offsets=cfg.cross_track_offsets,
    yaw_offsets=cfg.yaw_offsets,
    repeats=cfg.repeats,
    settle_steps=cfg.settle_steps,
    cross_track_gain=cfg.cross_track_gain,
    heading_gain=cfg.heading_gain,
    max_lateral_speed=cfg.max_lateral_speed,
    max_yaw_rate=cfg.max_yaw_rate,
    cross_track_tolerance=cfg.cross_track_tolerance,
    heading_tolerance=cfg.heading_tolerance,
    corridor_half_width=cfg.corridor_half_width,
    steps=cfg.steps,
    seed=cfg.seed,
    profile=cfg.profile,
    output_file=cfg.output_file,
  )
  original = (
    terrain_curves.make_terrain_curve_generator,
    terrain_curves.difficulty_for_level,
    terrain_curves.effective_terrain_parameters,
  )
  terrain_curves.make_terrain_curve_generator = make_high_difficulty_curve_generator
  terrain_curves.difficulty_for_level = difficulty_for_high_level
  terrain_curves.effective_terrain_parameters = effective_high_terrain_parameters
  try:
    result = terrain_curves.evaluate(base_cfg)
  finally:
    (
      terrain_curves.make_terrain_curve_generator,
      terrain_curves.difficulty_for_level,
      terrain_curves.effective_terrain_parameters,
    ) = original
  result.update(
    {
      "schema_version": 2,
      "evaluation_suite": "high_curves",
      "requested_high_difficulties": list(HIGH_DIFFICULTIES),
      "coverage_boundary": {
        "arc_or_s_curve_on_whole_primitive_patch": True,
        "continuous_transition_curves": False,
        "stairs_curves": False,
        "random_rough_high_claim": (
          "difficulty-invariant V7 distribution; high label does not imply harder geometry"
        ),
      },
    }
  )
  return result


def _evaluate_continuous_straight(cfg: TerrainBoundaryConfig) -> dict[str, Any]:
  import scripts.evaluate_go2_routes as routes
  import src.tasks.velocity.evaluation.route_terrains as route_terrains
  try:
    from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
      OnlineTerrainRolloutMetrics,
      action_acceleration,
      contact_any,
      foot_contact_any,
      foot_slip_velocity,
    )
  except ImportError as exc:
    raise RuntimeError(
      "continuous formal metrics require the terrain_rollout_metrics integration"
    ) from exc

  route_cfg = routes.RouteConfig(
    checkpoints=(cfg.checkpoint,),
    task_id=cfg.task_id,
    mode="line_follow",
    terrain_suite="continuous",
    transition_cases=cfg.transition_cases,
    levels=cfg.transition_levels,
    repeats=cfg.repeats,
    cross_track_offsets=cfg.cross_track_offsets,
    yaw_offsets=cfg.yaw_offsets,
    route_heading=0.0,
    route_length=None,
    target_speed=cfg.speeds[0],
    cross_track_gain=cfg.cross_track_gain,
    heading_gain=cfg.heading_gain,
    max_lateral_speed=cfg.max_lateral_speed,
    max_yaw_rate=cfg.max_yaw_rate,
    cross_track_tolerance=cfg.cross_track_tolerance,
    heading_tolerance=cfg.heading_tolerance,
    steps=cfg.steps,
    seed=cfg.seed,
    profile=cfg.profile,
    output_file=cfg.output_file,
  )
  original = (
    route_terrains.TERRAIN_KIND_TO_KEY,
    route_terrains.continuous_route_difficulty_matrix,
    route_terrains.route_terrain_metadata,
    route_terrains.make_continuous_route_terrain_generator,
    routes.ManagerBasedRlEnv,
    routes.update_attempt_status,
  )
  capture: dict[str, Any] = {}

  def env_constructor(*args: Any, **kwargs: Any) -> Any:
    env = original[4](*args, **kwargs)
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"))
    capture["env"] = env
    capture["foot_ids"] = foot_ids
    capture["terrain_metrics"] = OnlineTerrainRolloutMetrics(
      env.num_envs,
      cfg.steps,
      device=env.device,
      dtype=robot.data.root_link_pos_w.dtype,
    )
    capture["route_metrics"] = _OnlineRouteErrorMetrics(
      env.num_envs,
      cfg.steps,
      device=env.device,
      dtype=robot.data.root_link_pos_w.dtype,
    )
    return env

  def lifecycle_hook(*args: Any, **kwargs: Any) -> Any:
    lifecycle = original[5](*args, **kwargs)
    env = capture.get("env")
    if env is None:
      raise RuntimeError("continuous rollout environment was not captured")
    cross_track = args[2]
    heading = args[3]
    route_metrics = capture["route_metrics"]
    route_metrics.update(cross_track, heading, lifecycle.sample_mask)
    robot = env.scene["robot"]
    foot_ids = capture["foot_ids"]
    try:
      feet_found = env.scene["feet_ground_contact"].data.found
    except KeyError:
      slip = None
    else:
      feet = foot_contact_any(feet_found, env.num_envs, len(foot_ids))
      slip = foot_slip_velocity(
        robot.data.site_lin_vel_w[:, foot_ids, :2], feet
      )

    def sensor_contact(name: str) -> torch.Tensor | None:
      try:
        return contact_any(env.scene[name].data.found, env.num_envs)
      except KeyError:
        return None

    catastrophic = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    for name in env.termination_manager.active_terms:
      if not env.termination_manager.get_term_cfg(name).time_out:
        catastrophic |= env.termination_manager.get_term(name).bool()
    capture["terrain_metrics"].update(
      sample_mask=lifecycle.sample_mask,
      action_acceleration=action_acceleration(
        env.action_manager.action,
        env.action_manager.prev_action,
        env.action_manager.prev_prev_action,
      ),
      foot_slip_velocity=slip,
      body_contacts={
        "base": sensor_contact("base_ground_contact"),
        "upper_leg": sensor_contact("upper_leg_ground_contact"),
        "calf": sensor_contact("calf_ground_contact"),
      },
      catastrophic_termination=catastrophic,
    )
    return lifecycle

  route_terrains.TERRAIN_KIND_TO_KEY = CONTINUOUS_TRANSITION_KEYS
  route_terrains.continuous_route_difficulty_matrix = (
    boundary_transition_difficulty_matrix
  )
  route_terrains.route_terrain_metadata = boundary_transition_metadata
  route_terrains.make_continuous_route_terrain_generator = (
    make_boundary_transition_generator
  )
  routes.ManagerBasedRlEnv = env_constructor
  routes.update_attempt_status = lifecycle_hook
  try:
    result = routes._evaluate_checkpoint(
      Path(cfg.checkpoint).expanduser().resolve(), route_cfg
    )
  finally:
    (
      route_terrains.TERRAIN_KIND_TO_KEY,
      route_terrains.continuous_route_difficulty_matrix,
      route_terrains.route_terrain_metadata,
      route_terrains.make_continuous_route_terrain_generator,
      routes.ManagerBasedRlEnv,
      routes.update_attempt_status,
    ) = original

  footprint = validate_straight_transition_footprint(
    corridor_half_width=cfg.corridor_half_width
  )
  for index, scenario in enumerate(result["scenarios"]):
    route_metrics = capture["route_metrics"].result(index)
    terrain_metrics = capture["terrain_metrics"].result(index)
    if (
      route_metrics["active_control_step_samples"] != scenario["steps_sampled"]
      or terrain_metrics["active_control_step_samples"] != scenario["steps_sampled"]
    ):
      raise RuntimeError("continuous metric sample mask diverged from route rollout")
    cross_distribution = route_metrics["cross_track_absolute"]
    heading_distribution = route_metrics["heading_absolute"]
    action_distribution = terrain_metrics["action_acceleration"]
    slip_distribution = terrain_metrics["foot_slip_velocity"]
    contacts = terrain_metrics["body_contacts"]
    scenario.update(
      {
        "route_error_metrics": route_metrics,
        "cross_track_p95": cross_distribution["p95"],
        "heading_p95": heading_distribution["p95"],
        "terrain_rollout_metrics": terrain_metrics,
        "action_acceleration_p95": action_distribution["p95"],
        "action_acceleration_max": action_distribution["max"],
        "slip_velocity_p95": slip_distribution["p95"],
        "slip_velocity_max": slip_distribution["max"],
        "base_contact_count": contacts["base"]["non_terminating_count"],
        "base_contact_rate": contacts["base"]["non_terminating_rate"],
        "upper_leg_contact_count": contacts["upper_leg"]["non_terminating_count"],
        "upper_leg_contact_rate": contacts["upper_leg"]["non_terminating_rate"],
        "calf_contact_count": contacts["calf"]["non_terminating_count"],
        "calf_contact_rate": contacts["calf"]["non_terminating_rate"],
        "catastrophic_termination": terrain_metrics["catastrophic_termination"][
          "occurred"
        ],
      }
    )
    metadata = boundary_transition_metadata(
      scenario["transition_case"], scenario["difficulty"]
    )
    scenario["effective_terrain_parameters"] = metadata.effective_parameters
    scenario["difficulty_affects_geometry"] = metadata.difficulty_affects_geometry
    scenario["geometry_contract"] = metadata.geometry_contract
    scenario["route_bounds_local"] = footprint.route_bounds
    scenario["corridor_bounds_local"] = footprint.corridor_bounds
    scenario["scan_footprint_bounds_local"] = footprint.scan_footprint_bounds
    scenario["corridor_boundary_margin"] = footprint.corridor_boundary_margin
    scenario["scan_boundary_margin"] = footprint.scan_boundary_margin
    if scenario["transition_case"] in ("random_rough", "discrete_obstacle"):
      _, _, heights = continuous_feature_heightfield(
        scenario["transition_case"], scenario["difficulty"], cfg.seed
      )
      scenario["surface_height_min"] = float(heights.min())
      scenario["surface_height_max"] = float(heights.max())
      scenario["max_adjacent_height_delta"] = float(
        max(np_abs_diff_max(heights, axis=0), np_abs_diff_max(heights, axis=1))
      )
  result["schema_version"] = 2
  result["evaluation_suite"] = "continuous_straight"
  result["git_head"] = _git_head()
  result["coverage"] = continuous_boundary_coverage()
  result["straight_transition_footprint"] = asdict(footprint)
  result["metric_coverage"] = {
    "completion_progress_reset_failure": True,
    "cross_track_rms_max_final": True,
    "heading_rms_max_final": True,
    "commanded_actual_velocity_mean": True,
    "termination_counts": True,
    "cross_track_p95": True,
    "heading_p95": True,
    "slip_action_acceleration_p95_max": True,
    "nonterminating_body_contact_rates": True,
    "formal_metric_gate_complete": True,
  }
  return result


def np_abs_diff_max(values: Any, axis: int) -> float:
  """Small NumPy boundary kept local so CLI import stays lightweight."""
  import numpy as np

  differences = np.abs(np.diff(values, axis=axis))
  return float(differences.max()) if differences.size else 0.0


def evaluate(cfg: TerrainBoundaryConfig) -> dict[str, Any]:
  _validate_config(cfg)
  result = (
    _evaluate_high_curves(cfg)
    if cfg.suite == "high_curves"
    else _evaluate_continuous_straight(cfg)
  )
  result["boundary_config"] = asdict(cfg)
  _strict_json_finite(result)
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(TerrainBoundaryConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  print(f"[INFO] Wrote terrain boundary evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
