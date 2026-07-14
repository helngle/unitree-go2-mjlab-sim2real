"""Evaluate Go2 checkpoints on clean or randomized rough-terrain matrices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


COMMAND_CASES = {
  "forward_0.3": (0.3, 0.0, 0.0),
  "forward_0.6": (0.6, 0.0, 0.0),
  "forward_0.9": (0.9, 0.0, 0.0),
  "lateral_left": (0.0, 0.3, 0.0),
  "lateral_right": (0.0, -0.3, 0.0),
  "yaw_left": (0.0, 0.0, 0.5),
  "yaw_right": (0.0, 0.0, -0.5),
}

RANDOMIZATION_EVENTS = (
  "foot_friction",
  "encoder_bias",
  "base_com",
  "base_payload",
  "motor_strength",
)


@dataclass(frozen=True)
class EvaluateConfig:
  checkpoints: tuple[str, ...]
  task_id: str = "Unitree-Go2-Rough-V6"
  levels: tuple[int, ...] = (3, 5, 7, 9)
  repeats: int = 4
  steps: int = 1000
  seed: int = 42
  command_x: float = 0.6
  command_y: float = 0.0
  command_yaw: float = 0.0
  command_cases: tuple[str, ...] = ()
  profile: str = "clean"
  output_file: str | None = None


def _resolve_command_cases(
  cfg: EvaluateConfig,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
  if not cfg.command_cases:
    return (("custom", (cfg.command_x, cfg.command_y, cfg.command_yaw)),)

  unknown = sorted(set(cfg.command_cases) - COMMAND_CASES.keys())
  if unknown:
    available = ", ".join(COMMAND_CASES)
    raise ValueError(
      f"Unknown command cases: {', '.join(unknown)}. Available cases: {available}."
    )
  return tuple((name, COMMAND_CASES[name]) for name in cfg.command_cases)


def _configure_profile(env_cfg, profile: str) -> None:
  if profile not in {"clean", "dynamics", "randomized"}:
    raise ValueError(
      f"Unknown profile {profile!r}; expected clean, dynamics, or randomized."
    )

  if profile == "clean":
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    for event_name in RANDOMIZATION_EVENTS:
      env_cfg.events.pop(event_name, None)
  elif profile == "dynamics":
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    env_cfg.events.pop("encoder_bias", None)


def _evaluate(checkpoint: Path, cfg: EvaluateConfig) -> dict:
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  command_cases = _resolve_command_cases(cfg)
  terrain_cfg = env_cfg.scene.terrain
  assert terrain_cfg is not None and terrain_cfg.terrain_generator is not None
  generator_cfg = terrain_cfg.terrain_generator
  num_types = generator_cfg.num_cols
  terrain_names = list(generator_cfg.sub_terrains)
  proportions = np.array(
    [sub_terrain.proportion for sub_terrain in generator_cfg.sub_terrains.values()]
  )
  proportions /= proportions.sum()
  column_terrain_names = [
    terrain_names[
      int(np.where(column / num_types + 0.001 < np.cumsum(proportions))[0][0])
    ]
    for column in range(num_types)
  ]
  terrain_matrix_size = len(cfg.levels) * num_types * cfg.repeats
  num_envs = len(command_cases) * terrain_matrix_size
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  _configure_profile(env_cfg, cfg.profile)
  profile_settings = {
    "actor_observation_corruption": env_cfg.observations[
      "actor"
    ].enable_corruption,
    "startup_randomization_events": [
      name for name in RANDOMIZATION_EVENTS if name in env_cfg.events
    ],
    "push_enabled": "push_robot" in env_cfg.events,
  }

  command_cfg = env_cfg.commands["twist"]
  assert isinstance(command_cfg, UniformVelocityCommandCfg)
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  first_command = command_cases[0][1]
  command_cfg.ranges.lin_vel_x = (first_command[0], first_command[0])
  command_cfg.ranges.lin_vel_y = (first_command[1], first_command[1])
  command_cfg.ranges.ang_vel_z = (first_command[2], first_command[2])
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  old_env_origins = terrain.env_origins.clone()
  old_root_pos = robot.data.root_link_pos_w.clone()
  level_values = torch.tensor(cfg.levels, device=env.device, dtype=torch.long)
  level_values = level_values.clamp(0, terrain.max_terrain_level - 1)
  terrain_levels = level_values.repeat_interleave(num_types * cfg.repeats)
  terrain_types = torch.arange(num_types, device=env.device).repeat(
    len(cfg.levels) * cfg.repeats
  )
  terrain.terrain_levels[:] = terrain_levels.repeat(len(command_cases))
  terrain.terrain_types[:] = terrain_types.repeat(len(command_cases))
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]
  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :3] += terrain.env_origins - old_env_origins
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  placement_error = torch.max(
    torch.abs(
      (robot.data.root_link_pos_w - terrain.env_origins)
      - (old_root_pos - old_env_origins)
    )
  )
  if placement_error > 1.0e-4:
    raise RuntimeError(
      f"Terrain assignment did not preserve root offset: {placement_error.item():.6f}."
    )
  command_case_ids = torch.arange(
    len(command_cases), device=env.device
  ).repeat_interleave(terrain_matrix_size)
  command_values = torch.tensor(
    [command for _, command in command_cases],
    device=env.device,
    dtype=torch.float32,
  )[command_case_ids]

  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")
  command_term = env.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)

  command_term.vel_command_b[:] = command_values
  observation = wrapped_env.get_observations()
  termination_counts = {
    name: torch.zeros(num_envs, device=env.device)
    for name in env.termination_manager.active_terms
  }
  linear_error_sum = torch.zeros(num_envs, device=env.device)
  linear_abs_error_sum = torch.zeros(num_envs, 2, device=env.device)
  actual_linear_velocity_sum = torch.zeros(num_envs, 2, device=env.device)
  linear_response_gain_sum = torch.zeros(num_envs, device=env.device)
  linear_cross_axis_sum = torch.zeros(num_envs, device=env.device)
  linear_command_sample_count = torch.zeros(num_envs, device=env.device)
  yaw_error_sum = torch.zeros(num_envs, device=env.device)
  slip_sum = torch.zeros(num_envs, device=env.device)
  action_acc_sum = torch.zeros(num_envs, device=env.device)
  sample_count = torch.zeros(num_envs, device=env.device)
  feet_sensor = env.scene["feet_ground_contact"]
  foot_site_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"))

  for _ in range(cfg.steps):
    command_term.vel_command_b[:] = command_values
    with torch.inference_mode():
      action = policy(observation)
    wrapped_env.step(action)
    command_term.vel_command_b[:] = command_values
    observation = wrapped_env.get_observations()

    actual_lin = robot.data.root_link_lin_vel_b[:, :2]
    actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
    command_lin = command_values[:, :2]
    linear_error_sum += torch.norm(command_lin - actual_lin, dim=-1)
    linear_abs_error_sum += torch.abs(command_lin - actual_lin)
    actual_linear_velocity_sum += actual_lin
    command_norm = torch.norm(command_lin, dim=-1)
    linear_command_mask = command_norm > 1.0e-6
    command_norm_sq = torch.sum(command_lin.square(), dim=-1).clamp_min(1.0e-6)
    linear_response_gain_sum += torch.where(
      linear_command_mask,
      torch.sum(actual_lin * command_lin, dim=-1) / command_norm_sq,
      0.0,
    )
    linear_cross_axis_sum += torch.where(
      linear_command_mask,
      torch.abs(
        command_lin[:, 0] * actual_lin[:, 1]
        - command_lin[:, 1] * actual_lin[:, 0]
      )
      / command_norm.clamp_min(1.0e-6),
      0.0,
    )
    linear_command_sample_count += linear_command_mask
    yaw_error_sum += torch.abs(command_values[:, 2] - actual_yaw)
    in_contact = feet_sensor.data.found > 0
    foot_speed = torch.norm(
      robot.data.site_lin_vel_w[:, foot_site_ids, :2], dim=-1
    )
    slip_sum += (foot_speed * in_contact).sum(dim=-1) / in_contact.sum(dim=-1).clamp_min(1)
    action_acc = (
      env.action_manager.action
      - 2 * env.action_manager.prev_action
      + env.action_manager.prev_prev_action
    )
    action_acc_sum += torch.mean(torch.abs(action_acc), dim=-1)
    sample_count += 1
    for name in termination_counts:
      termination_counts[name] += env.termination_manager.get_term(name).float()

  def summarize(mask: torch.Tensor) -> dict:
    denominator = sample_count[mask].clamp_min(1)
    linear_denominator = linear_command_sample_count[mask].sum().clamp_min(1)
    linear_abs_error_mean = (
      linear_abs_error_sum[mask] / denominator.unsqueeze(1)
    ).mean(dim=0)
    actual_linear_velocity_mean = (
      actual_linear_velocity_sum[mask] / denominator.unsqueeze(1)
    ).mean(dim=0)
    return {
      "num_envs": int(mask.sum().item()),
      "linear_velocity_error_mean": float((linear_error_sum[mask] / denominator).mean()),
      "linear_velocity_abs_error_xy_mean": [
        float(value) for value in linear_abs_error_mean
      ],
      "actual_linear_velocity_xy_mean": [
        float(value) for value in actual_linear_velocity_mean
      ],
      "linear_command_response_gain_mean": float(
        linear_response_gain_sum[mask].sum() / linear_denominator
      ),
      "linear_cross_axis_velocity_mean": float(
        linear_cross_axis_sum[mask].sum() / linear_denominator
      ),
      "yaw_velocity_error_mean": float((yaw_error_sum[mask] / denominator).mean()),
      "slip_velocity_mean": float((slip_sum[mask] / denominator).mean()),
      "action_acceleration_mean": float((action_acc_sum[mask] / denominator).mean()),
      "terminations_per_env": {
        name: float(values[mask].mean()) for name, values in termination_counts.items()
      },
    }

  result = {
    "checkpoint": str(checkpoint),
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "steps": cfg.steps,
    "profile": cfg.profile,
    "profile_settings": profile_settings,
    "terrain_assignment_position_error_max": float(placement_error),
    "command": list(command_cases[0][1]) if len(command_cases) == 1 else None,
    "commands": {
      name: list(command) for name, command in command_cases
    },
    "overall": summarize(torch.ones(num_envs, dtype=torch.bool, device=env.device)),
    "by_command": {},
    "by_command_and_level": {},
    "by_command_and_terrain_type": {},
    "by_level": {},
    "by_terrain_column": {},
    "by_terrain_type": {},
  }
  for command_id, (name, command) in enumerate(command_cases):
    command_mask = command_case_ids == command_id
    result["by_command"][name] = {
      "command": list(command),
      **summarize(command_mask),
    }
    result["by_command_and_level"][name] = {
      str(int(level)): summarize(
        command_mask & (terrain.terrain_levels == level)
      )
      for level in level_values.unique(sorted=True)
    }
    result["by_command_and_terrain_type"][name] = {}
    for terrain_name in terrain_names:
      columns = [
        column
        for column, column_name in enumerate(column_terrain_names)
        if column_name == terrain_name
      ]
      terrain_mask = torch.zeros(
        num_envs, dtype=torch.bool, device=env.device
      )
      for column in columns:
        terrain_mask |= terrain.terrain_types == column
      result["by_command_and_terrain_type"][name][terrain_name] = summarize(
        command_mask & terrain_mask
      )
  for level in level_values.unique(sorted=True):
    result["by_level"][str(int(level))] = summarize(terrain.terrain_levels == level)
  for column in range(num_types):
    result["by_terrain_column"][str(column)] = {
      "terrain_type": column_terrain_names[column],
      **summarize(terrain.terrain_types == column),
    }
  for terrain_name in terrain_names:
    columns = [
      column
      for column, column_name in enumerate(column_terrain_names)
      if column_name == terrain_name
    ]
    mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    for column in columns:
      mask |= terrain.terrain_types == column
    result["by_terrain_type"][terrain_name] = summarize(mask)
  env.close()
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(EvaluateConfig)
  results = [_evaluate(Path(path).expanduser().resolve(), cfg) for path in cfg.checkpoints]
  output = {"config": asdict(cfg), "results": results}
  output_path = Path(cfg.output_file) if cfg.output_file else Path("go2_rough_evaluation.json")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(output, indent=2) + "\n")
  print(json.dumps(output, indent=2))
  print(f"[INFO] Wrote evaluation to {output_path}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
