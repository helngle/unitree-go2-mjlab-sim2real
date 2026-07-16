"""Evaluate curved Go2 routes on large evaluation-only terrain patches.

This harness reuses the validated flat curved-route rollout and replaces only
its terrain generator and post-reset placement.  It does not register or alter
any production training task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from mjlab.utils.torch import configure_torch_backends

import scripts.evaluate_go2_curved_routes as flat_curves
from scripts.evaluate_go2_curved_routes import CurvedRouteConfig
from src.tasks.velocity.evaluation.terrain_curved_routes import (
  DEFAULT_CORRIDOR_HALF_WIDTH,
  PATCH_SIZE,
  ROUTE_START_LOCAL,
  TERRAIN_CURVE_KINDS,
  TERRAIN_KIND_TO_TYPE,
  TerrainCurveKind,
  continuous_transition_coverage,
  difficulty_for_level,
  effective_terrain_parameters,
  make_terrain_curve_generator,
  relocate_root_pose,
  slope_direction_is_compatible,
  validate_route_footprint,
)


TERRAIN_KEYS: dict[TerrainCurveKind, str] = {
  "slope_up": "hf_pyramid_slope_inv",
  "slope_down": "hf_pyramid_slope",
  "random_rough": "random_rough",
  "discrete_obstacle": "discrete_obstacles",
}


@dataclass(frozen=True)
class TerrainCurvedRouteConfig:
  checkpoint: str
  task_id: str = "Unitree-Go2-Rough-V7"
  route_kind: str = "arc"
  mode: str = "closed_loop"
  terrain_kinds: tuple[str, ...] = TERRAIN_CURVE_KINDS
  terrain_levels: tuple[int, ...] = (0, 1)
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
  steps: int = 2000
  seed: int = 42
  profile: str = "clean"
  output_file: str = "go2_terrain_curved_route_evaluation.json"


def _base_config(cfg: TerrainCurvedRouteConfig) -> CurvedRouteConfig:
  base_names = {field.name for field in fields(CurvedRouteConfig)}
  values = {
    name: value
    for name, value in asdict(cfg).items()
    if name in base_names
  }
  return CurvedRouteConfig(**values)


def _validate_config(cfg: TerrainCurvedRouteConfig) -> None:
  flat_curves._validate_config(_base_config(cfg))
  if not cfg.terrain_kinds:
    raise ValueError("terrain_kinds must not be empty")
  unknown = set(cfg.terrain_kinds) - set(TERRAIN_CURVE_KINDS)
  if unknown:
    raise ValueError(f"unsupported terrain curve kinds: {sorted(unknown)}")
  if not cfg.terrain_levels or any(level not in (0, 1) for level in cfg.terrain_levels):
    raise ValueError("terrain_levels must contain only 0 (low) or 1 (medium)")
  if not math.isfinite(cfg.corridor_half_width) or cfg.corridor_half_width <= 0.0:
    raise ValueError("corridor_half_width must be finite and positive")
  for radius in cfg.radii:
    for turn_sign in cfg.turn_signs:
      validate_route_footprint(
        cfg.route_kind,
        radius,
        turn_sign,
        corridor_half_width=cfg.corridor_half_width,
      )
      if any(kind.startswith("slope") for kind in cfg.terrain_kinds) and not (
        slope_direction_is_compatible(cfg.route_kind, radius, turn_sign)
      ):
        raise ValueError("route direction is incompatible with pyramid slope")


def _scenarios(cfg: TerrainCurvedRouteConfig) -> list[dict[str, Any]]:
  return [
    {
      "terrain_kind": terrain_kind,
      "level": level,
      "radius": radius,
      "speed": speed,
      "turn_sign": turn_sign,
      "cross_track_offset": cross,
      "yaw_offset": yaw,
      "repeat": repeat,
    }
    for terrain_kind in cfg.terrain_kinds
    for level in cfg.terrain_levels
    for radius in cfg.radii
    for speed in cfg.speeds
    for turn_sign in cfg.turn_signs
    for cross in cfg.cross_track_offsets
    for yaw in cfg.yaw_offsets
    for repeat in range(cfg.repeats)
  ]


def _strict_json_finite(value: object, path: str = "root") -> None:
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError(f"non-finite JSON value at {path}: {value}")
  elif isinstance(value, dict):
    for key, item in value.items():
      _strict_json_finite(item, f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _strict_json_finite(item, f"{path}[{index}]")


def _git_head() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1],
    text=True,
  ).strip()


def _terrain_assignment(
  env: ManagerBasedRlEnv,
  scenarios: list[dict[str, Any]],
  cross_offsets: torch.Tensor,
  yaw_offsets: torch.Tensor,
  capture: dict[str, Any],
) -> tuple[torch.Tensor, float, torch.Tensor]:
  """Relocate to requested patches, validate invariance, then place routes."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  device = robot.data.root_link_pos_w.device
  levels = torch.tensor(
    [scenario["level"] for scenario in scenarios],
    dtype=torch.long,
    device=device,
  )
  types = torch.tensor(
    [TERRAIN_KIND_TO_TYPE[scenario["terrain_kind"]] for scenario in scenarios],
    dtype=torch.long,
    device=device,
  )
  old_origins = terrain.env_origins.clone()
  old_root = robot.data.root_link_pose_w.clone()
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  new_origins = terrain.env_origins.clone()

  relative_before = old_root[:, :3] - old_origins
  relocated, arithmetic_error = relocate_root_pose(
    old_root, old_origins, new_origins
  )
  robot.write_root_link_pose_to_sim(relocated)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  relative_after = robot.data.root_link_pos_w - new_origins
  assignment_error = max(
    arithmetic_error,
    float(torch.max(torch.abs(relative_after - relative_before))),
  )
  if assignment_error > 1.0e-4:
    raise RuntimeError(
      f"terrain assignment relocation error: {assignment_error:.6f}"
    )

  clearance = relocated[:, 2] - new_origins[:, 2]
  route_start = new_origins[:, :2].clone()
  root = robot.data.root_link_pose_w.clone()
  root[:, :2] = route_start + torch.stack(
    (torch.zeros_like(cross_offsets), cross_offsets), dim=-1
  )
  root[:, 2] = new_origins[:, 2] + clearance
  old_heading = robot.data.heading_w.clone()
  root[:, 3:7] = quat_mul(
    quat_from_euler_xyz(
      torch.zeros_like(yaw_offsets),
      torch.zeros_like(yaw_offsets),
      yaw_offsets - old_heading,
    ),
    root[:, 3:7],
  )
  robot.write_root_link_pose_to_sim(root)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  expected = torch.cat(
    (route_start, (new_origins[:, 2] + clearance).unsqueeze(-1)), dim=-1
  )
  placement_error = float(
    torch.max(torch.abs(robot.data.root_link_pos_w - expected))
  )
  if placement_error > 1.0e-4:
    raise RuntimeError(f"terrain curve placement error: {placement_error:.6f}")

  capture["origins"] = new_origins.detach().cpu().tolist()
  capture["patch_origins"] = (
    new_origins
    - torch.tensor(
      [ROUTE_START_LOCAL[0], ROUTE_START_LOCAL[1], 0.0],
      device=device,
      dtype=new_origins.dtype,
    )
  ).detach().cpu().tolist()
  capture["route_placement_error_max"] = placement_error
  capture["levels"] = levels.detach().cpu().tolist()
  capture["types"] = types.detach().cpu().tolist()
  return route_start, assignment_error, clearance


def _contact_termination_summary(
  termination_counts: dict[str, float],
) -> dict[str, dict[str, object]]:
  mapping = {
    "fell": "fell_over",
    "base": "illegal_base_contact",
    "upper_leg": "illegal_upper_leg_contact",
    "calf": "illegal_calf_contact",
  }
  return {
    label: {
      "termination_count": termination_counts.get(term),
      "available": term in termination_counts,
      "scope": "termination events; non-terminating contact rate is not retained",
    }
    for label, term in mapping.items()
  }


def evaluate(cfg: TerrainCurvedRouteConfig) -> dict[str, Any]:
  _validate_config(cfg)
  scenarios = _scenarios(cfg)
  capture: dict[str, Any] = {}
  original_generator = flat_curves.make_curved_flat_generator
  original_placement = flat_curves._place_routes

  def generator(seed: int):
    return make_terrain_curve_generator(seed)

  def placement(
    env: ManagerBasedRlEnv,
    cross_offsets: torch.Tensor,
    yaw_offsets: torch.Tensor,
  ) -> tuple[torch.Tensor, float, torch.Tensor]:
    return _terrain_assignment(
      env, scenarios, cross_offsets, yaw_offsets, capture
    )

  flat_curves.make_curved_flat_generator = generator
  flat_curves._place_routes = placement
  try:
    raw = flat_curves._evaluate_scenarios(
      _base_config(cfg), scenarios
    )
  finally:
    flat_curves.make_curved_flat_generator = original_generator
    flat_curves._place_routes = original_placement

  outputs = raw["scenarios"]
  for index, (scenario, output) in enumerate(zip(scenarios, outputs, strict=True)):
    kind: TerrainCurveKind = scenario["terrain_kind"]
    label, difficulty = difficulty_for_level(scenario["level"])
    footprint = validate_route_footprint(
      cfg.route_kind,
      scenario["radius"],
      scenario["turn_sign"],
      corridor_half_width=cfg.corridor_half_width,
    )
    output.update({
      "terrain_type": TERRAIN_KEYS[kind],
      "terrain_curve_kind": kind,
      "terrain_level": scenario["level"],
      "terrain_type_index": capture["types"][index],
      "difficulty_label": label,
      "difficulty": difficulty,
      "effective_terrain_parameters": effective_terrain_parameters(
        kind, scenario["level"]
      ),
      "terrain_origin_xyz": capture["origins"][index],
      "terrain_patch_origin_xyz": capture["patch_origins"][index],
      "terrain_patch_size": list(PATCH_SIZE),
      "route_start_local": list(ROUTE_START_LOCAL),
      "corridor_bounds_local": footprint.corridor_bounds,
      "scan_footprint_bounds_local": footprint.scan_footprint_bounds,
      "corridor_inside_patch": footprint.corridor_inside_patch,
      "scan_footprint_inside_patch": footprint.scan_footprint_inside_patch,
      "corridor_boundary_margin": footprint.corridor_boundary_margin,
      "scan_boundary_margin": footprint.scan_boundary_margin,
      "slope_direction_compatible": (
        slope_direction_is_compatible(
          cfg.route_kind, scenario["radius"], scenario["turn_sign"]
        ) if kind.startswith("slope") else None
      ),
      "contact_termination_summary": _contact_termination_summary(
        output["termination_counts"]
      ),
    })

  completion = sum(item["completed"] for item in outputs) / max(len(outputs), 1)
  result = {
    "schema_version": 1,
    "git_head": _git_head(),
    "config": asdict(cfg),
    "checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
    "task_id": cfg.task_id,
    "profile_settings": raw["profile_settings"],
    "num_envs": raw["num_envs"],
    "completion_rate": completion,
    "terrain_assignment_position_error_max": raw[
      "terrain_assignment_position_error_max"
    ],
    "route_placement_position_error_max": capture[
      "route_placement_error_max"
    ],
    "coverage": {
      "slope_curves": any(kind.startswith("slope") for kind in cfg.terrain_kinds),
      "random_rough_curves": "random_rough" in cfg.terrain_kinds,
      "discrete_obstacle_curves": "discrete_obstacle" in cfg.terrain_kinds,
      "continuous_approach_feature_exit": continuous_transition_coverage(),
      "stairs_curves": False,
    },
    "limitations": [
      "Continuous approach-feature-exit curves are not implemented because the "
      "validated 8x4 m transition patch cannot contain the requested curves.",
      "Contact output currently reports catastrophic termination events; "
      "non-terminating body-part contact rates are not retained by the reused rollout.",
      "Per-step P95/max slip and action-acceleration are not retained by the reused "
      "rollout; mean values remain available per scenario.",
    ],
    "scenarios": outputs,
  }
  _strict_json_finite(result)
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(TerrainCurvedRouteConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  print(f"[INFO] Wrote terrain curved-route evaluation to {output}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
