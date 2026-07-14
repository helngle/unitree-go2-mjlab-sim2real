"""Diagnose lateral velocity response, gait timing, and joint usage."""

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
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


COMMANDS = {
  "forward_0.6": (0.6, 0.0, 0.0),
  "lateral_left": (0.0, 0.3, 0.0),
  "lateral_right": (0.0, -0.3, 0.0),
}


@dataclass(frozen=True)
class DiagnoseConfig:
  checkpoints: tuple[str, ...]
  task_id: str = "Unitree-Go2-Rough-V7"
  levels: tuple[int, ...] = (5, 9)
  terrain_types: tuple[str, ...] = (
    "flat",
    "pyramid_stairs",
    "hf_pyramid_slope_inv",
  )
  command_cases: tuple[str, ...] = (
    "forward_0.6",
    "lateral_left",
    "lateral_right",
  )
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 800
  phase_bins: int = 12
  seed: int = 42
  output_file: str = "go2_lateral_diagnostics.json"


def _column_terrain_names(generator_cfg) -> list[str]:
  terrain_names = list(generator_cfg.sub_terrains)
  proportions = np.asarray(
    [terrain.proportion for terrain in generator_cfg.sub_terrains.values()],
    dtype=np.float64,
  )
  proportions /= proportions.sum()
  cumulative = np.cumsum(proportions)
  return [
    terrain_names[
      int(np.where(column / generator_cfg.num_cols + 0.001 < cumulative)[0][0])
    ]
    for column in range(generator_cfg.num_cols)
  ]


def _evaluate(checkpoint: Path, cfg: DiagnoseConfig) -> dict:
  unknown_commands = sorted(set(cfg.command_cases) - COMMANDS.keys())
  if unknown_commands:
    raise ValueError(f"Unknown command cases: {unknown_commands}.")

  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  terrain_cfg = env_cfg.scene.terrain
  assert terrain_cfg is not None and terrain_cfg.terrain_generator is not None
  generator_cfg = terrain_cfg.terrain_generator
  column_names = _column_terrain_names(generator_cfg)
  terrain_columns = {
    name: [index for index, column_name in enumerate(column_names) if column_name == name]
    for name in cfg.terrain_types
  }
  missing_terrains = [name for name, columns in terrain_columns.items() if not columns]
  if missing_terrains:
    raise ValueError(f"Terrain types have no columns: {missing_terrains}.")

  scenarios = [
    (command_name, level, terrain_name, repeat)
    for command_name in cfg.command_cases
    for level in cfg.levels
    for terrain_name in cfg.terrain_types
    for repeat in range(cfg.repeats)
  ]
  num_envs = len(scenarios)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  env_cfg.curriculum = {}
  env_cfg.observations["actor"].enable_corruption = False
  for event_name in (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "base_payload",
    "motor_strength",
  ):
    env_cfg.events.pop(event_name, None)

  first_command = COMMANDS[cfg.command_cases[0]]
  command_cfg = env_cfg.commands["twist"]
  assert isinstance(command_cfg, UniformVelocityCommandCfg)
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (first_command[0], first_command[0])
  command_cfg.ranges.lin_vel_y = (first_command[1], first_command[1])
  command_cfg.ranges.ang_vel_z = (first_command[2], first_command[2])
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  robot = env.scene["robot"]
  old_origins = terrain.env_origins.clone()
  old_root_pos = robot.data.root_link_pos_w.clone()

  command_values = torch.tensor(
    [COMMANDS[scenario[0]] for scenario in scenarios],
    device=env.device,
    dtype=torch.float32,
  )
  level_values = torch.tensor(
    [scenario[1] for scenario in scenarios], device=env.device, dtype=torch.long
  ).clamp(0, terrain.max_terrain_level - 1)
  type_values = torch.tensor(
    [
      terrain_columns[scenario[2]][scenario[3] % len(terrain_columns[scenario[2]])]
      for scenario in scenarios
    ],
    device=env.device,
    dtype=torch.long,
  )
  terrain.terrain_levels[:] = level_values
  terrain.terrain_types[:] = type_values
  terrain.env_origins[:] = terrain.terrain_origins[level_values, type_values]
  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :3] += terrain.env_origins - old_origins
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  placement_error = torch.max(
    torch.abs(
      (robot.data.root_link_pos_w - terrain.env_origins)
      - (old_root_pos - old_origins)
    )
  )
  if placement_error > 1.0e-4:
    raise RuntimeError(f"Terrain relocation error: {placement_error.item():.6f}.")

  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")
  command_term = env.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)

  foot_site_ids, foot_names = robot.find_sites(("FR", "FL", "RR", "RL"))
  joint_ids, joint_names = robot.find_joints((".*",), preserve_order=True)
  action_dim = env.action_manager.action.shape[1]
  if len(joint_names) != action_dim:
    raise RuntimeError(
      f"Expected one action per joint, got {action_dim} actions and {len(joint_names)} joints."
    )

  feet_sensor = env.scene["feet_ground_contact"]
  num_feet = len(foot_site_ids)
  phase_offsets = torch.tensor(
    [0.0, 0.5, 0.5, 0.0], device=env.device
  ).view(1, num_feet)
  sample_count = torch.zeros(num_envs, device=env.device)
  actual_velocity_sum = torch.zeros(num_envs, 2, device=env.device)
  absolute_error_sum = torch.zeros(num_envs, 2, device=env.device)
  direction_correct_sum = torch.zeros(num_envs, device=env.device)
  cross_axis_sum = torch.zeros(num_envs, device=env.device)
  yaw_abs_sum = torch.zeros(num_envs, device=env.device)
  gravity_xy_sum = torch.zeros(num_envs, device=env.device)
  contact_sum = torch.zeros(num_envs, num_feet, device=env.device)
  gait_match_sum = torch.zeros_like(contact_sum)
  contact_pattern_sum = torch.zeros(num_envs, 16, device=env.device)
  foot_pos_sum = torch.zeros(num_envs, num_feet, 3, device=env.device)
  foot_pos_sq_sum = torch.zeros_like(foot_pos_sum)
  foot_pos_min = torch.full_like(foot_pos_sum, torch.inf)
  foot_pos_max = torch.full_like(foot_pos_sum, -torch.inf)
  action_sum = torch.zeros(num_envs, action_dim, device=env.device)
  action_sq_sum = torch.zeros_like(action_sum)
  action_min = torch.full_like(action_sum, torch.inf)
  action_max = torch.full_like(action_sum, -torch.inf)
  joint_rel_sq_sum = torch.zeros_like(action_sum)
  phase_count = torch.zeros(num_envs, cfg.phase_bins, device=env.device)
  phase_foot_pos_sum = torch.zeros(
    num_envs, cfg.phase_bins, num_feet, 3, device=env.device
  )
  phase_contact_sum = torch.zeros(
    num_envs, cfg.phase_bins, num_feet, device=env.device
  )
  termination_counts = {
    name: torch.zeros(num_envs, device=env.device)
    for name in env.termination_manager.active_terms
  }

  command_term.vel_command_b[:] = command_values
  observation = wrapped_env.get_observations()
  env_indices = torch.arange(num_envs, device=env.device)
  total_steps = cfg.warmup_steps + cfg.sample_steps
  for step in range(total_steps):
    command_term.vel_command_b[:] = command_values
    with torch.inference_mode():
      action = policy(observation)
    wrapped_env.step(action)
    command_term.vel_command_b[:] = command_values
    observation = wrapped_env.get_observations()
    if step < cfg.warmup_steps:
      continue

    actual_velocity = robot.data.root_link_lin_vel_b[:, :2]
    actual_velocity_sum += actual_velocity
    absolute_error_sum += torch.abs(command_values[:, :2] - actual_velocity)
    primary_is_x = torch.abs(command_values[:, 0]) > 1.0e-6
    primary_command = torch.where(
      primary_is_x, command_values[:, 0], command_values[:, 1]
    )
    primary_actual = torch.where(
      primary_is_x, actual_velocity[:, 0], actual_velocity[:, 1]
    )
    cross_actual = torch.where(
      primary_is_x, actual_velocity[:, 1], actual_velocity[:, 0]
    )
    direction_correct_sum += (primary_command * primary_actual > 0.0).float()
    cross_axis_sum += torch.abs(cross_actual)
    yaw_abs_sum += torch.abs(robot.data.root_link_ang_vel_b[:, 2])
    gravity_xy_sum += torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)

    contact = feet_sensor.data.found > 0
    contact_sum += contact.float()
    global_phase = (env.episode_length_buf * env.step_dt / 0.6) % 1.0
    leg_phase = (global_phase.unsqueeze(1) + phase_offsets) % 1.0
    expected_contact = leg_phase < 0.56
    gait_match_sum += (contact == expected_contact).float()
    pattern = torch.sum(
      contact.long() * (2 ** torch.arange(num_feet, device=env.device)), dim=1
    )
    contact_pattern_sum[env_indices, pattern] += 1

    root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1)
    foot_delta_w = (
      robot.data.site_pos_w[:, foot_site_ids, :]
      - robot.data.root_link_pos_w[:, None, :]
    )
    foot_pos_b = quat_apply_inverse(
      root_quat.reshape(-1, 4), foot_delta_w.reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)
    foot_pos_sum += foot_pos_b
    foot_pos_sq_sum += foot_pos_b.square()
    foot_pos_min = torch.minimum(foot_pos_min, foot_pos_b)
    foot_pos_max = torch.maximum(foot_pos_max, foot_pos_b)

    current_action = env.action_manager.action
    action_sum += current_action
    action_sq_sum += current_action.square()
    action_min = torch.minimum(action_min, current_action)
    action_max = torch.maximum(action_max, current_action)
    joint_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    joint_rel_sq_sum += joint_rel.square()

    phase_bin = torch.clamp((global_phase * cfg.phase_bins).long(), max=cfg.phase_bins - 1)
    phase_count[env_indices, phase_bin] += 1
    phase_foot_pos_sum[env_indices, phase_bin] += foot_pos_b
    phase_contact_sum[env_indices, phase_bin] += contact.float()
    sample_count += 1
    for name in termination_counts:
      termination_counts[name] += env.termination_manager.get_term(name).float()

  def _vector(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values]

  def summarize(mask: torch.Tensor, command: tuple[float, float, float]) -> dict:
    total_samples = sample_count[mask].sum().clamp_min(1)
    actual_mean = actual_velocity_sum[mask].sum(dim=0) / total_samples
    abs_error_mean = absolute_error_sum[mask].sum(dim=0) / total_samples
    primary_axis = 0 if abs(command[0]) > 1.0e-6 else 1
    response_gain = actual_mean[primary_axis] / command[primary_axis]
    foot_mean = foot_pos_sum[mask].sum(dim=0) / total_samples
    foot_variance = foot_pos_sq_sum[mask].sum(dim=0) / total_samples - foot_mean.square()
    action_mean = action_sum[mask].sum(dim=0) / total_samples
    action_variance = action_sq_sum[mask].sum(dim=0) / total_samples - action_mean.square()
    phase_trajectory = []
    for phase_index in range(cfg.phase_bins):
      count = phase_count[mask, phase_index].sum().clamp_min(1)
      phase_trajectory.append(
        {
          "phase_center": (phase_index + 0.5) / cfg.phase_bins,
          "foot_position_body_mean": [
            _vector(values)
            for values in phase_foot_pos_sum[mask, phase_index].sum(dim=0) / count
          ],
          "foot_contact_fraction": _vector(
            phase_contact_sum[mask, phase_index].sum(dim=0) / count
          ),
        }
      )
    pattern_counts = contact_pattern_sum[mask].sum(dim=0)
    return {
      "num_envs": int(mask.sum()),
      "command": list(command),
      "velocity": {
        "actual_xy_mean": _vector(actual_mean),
        "absolute_error_xy_mean": _vector(abs_error_mean),
        "primary_response_gain": float(response_gain),
        "direction_correct_fraction": float(
          direction_correct_sum[mask].sum() / total_samples
        ),
        "cross_axis_abs_mean": float(cross_axis_sum[mask].sum() / total_samples),
        "yaw_abs_mean": float(yaw_abs_sum[mask].sum() / total_samples),
        "projected_gravity_xy_norm_mean": float(
          gravity_xy_sum[mask].sum() / total_samples
        ),
      },
      "gait": {
        "foot_names": list(foot_names),
        "contact_duty_factor": _vector(contact_sum[mask].sum(dim=0) / total_samples),
        "fixed_trot_match_fraction": _vector(
          gait_match_sum[mask].sum(dim=0) / total_samples
        ),
        "contact_pattern_fraction": {
          str(index): float(value / total_samples)
          for index, value in enumerate(pattern_counts)
          if value > 0
        },
        "foot_position_body_mean": [_vector(values) for values in foot_mean],
        "foot_position_body_std": [
          _vector(values) for values in foot_variance.clamp_min(0).sqrt()
        ],
        "foot_position_body_range": [
          _vector(values)
          for values in foot_pos_max[mask].amax(dim=0) - foot_pos_min[mask].amin(dim=0)
        ],
        "phase_trajectory": phase_trajectory,
      },
      "actions": {
        "joint_names": list(joint_names),
        "mean": _vector(action_mean),
        "std": _vector(action_variance.clamp_min(0).sqrt()),
        "range": _vector(action_max[mask].amax(dim=0) - action_min[mask].amin(dim=0)),
        "joint_position_relative_rms": _vector(
          (joint_rel_sq_sum[mask].sum(dim=0) / total_samples).sqrt()
        ),
      },
      "terminations_per_env": {
        name: float(values[mask].mean()) for name, values in termination_counts.items()
      },
    }

  result = {
    "checkpoint": str(checkpoint),
    "task_id": cfg.task_id,
    "seed": cfg.seed,
    "warmup_steps": cfg.warmup_steps,
    "sample_steps": cfg.sample_steps,
    "terrain_assignment_position_error_max": float(placement_error),
    "scenarios": {},
  }
  for command_name in cfg.command_cases:
    command = COMMANDS[command_name]
    for level in cfg.levels:
      for terrain_name in cfg.terrain_types:
        mask = torch.tensor(
          [
            scenario[0] == command_name
            and scenario[1] == level
            and scenario[2] == terrain_name
            for scenario in scenarios
          ],
          device=env.device,
          dtype=torch.bool,
        )
        key = f"{command_name}|level_{level}|{terrain_name}"
        result["scenarios"][key] = summarize(mask, command)
  env.close()
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(DiagnoseConfig)
  results = [_evaluate(Path(path).expanduser().resolve(), cfg) for path in cfg.checkpoints]
  output = {"config": asdict(cfg), "results": results}
  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(output, indent=2) + "\n")
  print(json.dumps(output, indent=2))
  print(f"[INFO] Wrote lateral diagnostics to {output_path}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
