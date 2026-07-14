"""Evaluate calf-contact recovery for trained Go2 velocity policies."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class DiagnoseConfig:
  checkpoints: tuple[str, ...]
  num_envs: int = 512
  steps: int = 1500
  seed: int = 42
  command_x: float = 0.6
  command_y: float = 0.0
  command_yaw: float = 0.0
  force_threshold: float = 10.0
  termination_force_threshold: float = 60.0
  termination_min_substeps: int = 3
  recovery_time_s: float = 0.5
  fall_angle_deg: float = 70.0
  destabilizing_angle_deg: float = 35.0
  destabilizing_velocity_error_increase: float = 0.5
  output_file: str | None = None


def _mean(values: torch.Tensor) -> float:
  return values.float().mean().item() if values.numel() else 0.0


def _percentile(values: torch.Tensor, quantile: float) -> float:
  return torch.quantile(values.float(), quantile).item() if values.numel() else 0.0


def _evaluate_checkpoint(checkpoint: Path, cfg: DiagnoseConfig) -> dict:
  task_id = "Unitree-Go2-Rough-V5"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = int(1e9)
  env_cfg.terminations = {}
  env_cfg.events.pop("push_robot", None)
  command_cfg = env_cfg.commands["twist"]
  assert isinstance(command_cfg, UniformVelocityCommandCfg)
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (cfg.command_x, cfg.command_x)
  command_cfg.ranges.lin_vel_y = (cfg.command_y, cfg.command_y)
  command_cfg.ranges.ang_vel_z = (cfg.command_yaw, cfg.command_yaw)
  command_cfg.ranges.heading = None
  terrain = env_cfg.scene.terrain
  if terrain is not None and terrain.terrain_generator is not None:
    terrain.max_init_terrain_level = 5

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")

  command_term = env.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)
  command = torch.tensor(
    [cfg.command_x, cfg.command_y, cfg.command_yaw],
    device=env.device,
    dtype=torch.float32,
  )

  calf_sensor: ContactSensor = env.scene["calf_ground_contact"]
  robot = env.scene["robot"]
  observation = wrapped_env.get_observations()
  num_calves = 8
  recovery_steps = math.ceil(cfg.recovery_time_s / env.step_dt)
  if cfg.termination_min_substeps > 4:
    raise ValueError("termination_min_substeps exceeds calf sensor history length")
  fall_angle = math.radians(cfg.fall_angle_deg)
  destabilizing_angle = math.radians(cfg.destabilizing_angle_deg)

  event_started = torch.zeros(cfg.num_envs, dtype=torch.bool, device=env.device)
  event_complete = torch.zeros_like(event_started)
  event_leg = torch.full(
    (cfg.num_envs,), -1, dtype=torch.long, device=env.device
  )
  event_peak_force = torch.zeros(cfg.num_envs, device=env.device)
  event_duration_substeps = torch.zeros(
    cfg.num_envs, dtype=torch.long, device=env.device
  )
  event_age_steps = torch.zeros_like(event_duration_substeps)
  would_trigger_termination = torch.zeros_like(event_started)
  trigger_angle = torch.zeros(cfg.num_envs, device=env.device)
  trigger_velocity_error = torch.zeros(cfg.num_envs, device=env.device)
  pre_angle = torch.zeros(cfg.num_envs, device=env.device)
  pre_velocity_error = torch.zeros(cfg.num_envs, device=env.device)
  max_angle = torch.zeros(cfg.num_envs, device=env.device)
  max_velocity_error = torch.zeros(cfg.num_envs, device=env.device)
  fell = torch.zeros_like(event_started)
  count_by_calf = torch.zeros(num_calves, dtype=torch.long, device=env.device)
  calf_geom_names = list(
    dict.fromkeys(slot.primary_name for slot in calf_sensor._slots)  # noqa: SLF001
  )
  assert len(calf_geom_names) == num_calves

  for _ in range(cfg.steps):
    command_term.vel_command_b[:] = command
    with torch.inference_mode():
      action = policy(observation)
    observation, _, _, _ = wrapped_env.step(action)
    command_term.vel_command_b[:] = command

    force_history = calf_sensor.data.force_history
    assert force_history is not None
    force_mag = torch.norm(force_history, dim=-1)
    peak_by_calf, _ = force_mag.max(dim=-1)
    step_peak, step_leg = peak_by_calf.max(dim=-1)
    duration_by_calf = (force_mag > cfg.force_threshold).sum(dim=-1)
    termination_hits = force_mag > cfg.termination_force_threshold
    trigger_by_calf = termination_hits.unfold(
      -1, cfg.termination_min_substeps, 1
    ).all(dim=-1).any(dim=-1)
    step_duration = torch.gather(
      duration_by_calf, dim=1, index=step_leg.unsqueeze(-1)
    ).squeeze(-1)

    projected_gravity = robot.data.projected_gravity_b
    angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs()
    actual_velocity = robot.data.root_link_lin_vel_b[:, :2]
    velocity_error = torch.norm(command[:2] - actual_velocity, dim=-1)

    new_event = (~event_started) & (step_peak > cfg.force_threshold)
    event_started |= new_event
    event_leg[new_event] = step_leg[new_event]
    event_peak_force[new_event] = step_peak[new_event]
    event_duration_substeps[new_event] = 0
    pre_angle[new_event] = angle[new_event]
    pre_velocity_error[new_event] = velocity_error[new_event]
    max_angle[new_event] = angle[new_event]
    max_velocity_error[new_event] = velocity_error[new_event]
    if new_event.any():
      count_by_calf += torch.bincount(step_leg[new_event], minlength=num_calves)

    active = event_started & ~event_complete
    event_age_steps[active] += 1
    same_leg_peak = torch.gather(
      peak_by_calf, dim=1, index=event_leg.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    same_leg_duration = torch.gather(
      duration_by_calf, dim=1, index=event_leg.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    same_leg_trigger = torch.gather(
      trigger_by_calf, dim=1, index=event_leg.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    event_peak_force[active] = torch.maximum(
      event_peak_force[active], same_leg_peak[active]
    )
    event_duration_substeps[active] += same_leg_duration[active]
    newly_triggering = active & same_leg_trigger & ~would_trigger_termination
    trigger_angle[newly_triggering] = angle[newly_triggering]
    trigger_velocity_error[newly_triggering] = velocity_error[newly_triggering]
    would_trigger_termination[active] |= same_leg_trigger[active]
    max_angle[active] = torch.maximum(max_angle[active], angle[active])
    max_velocity_error[active] = torch.maximum(
      max_velocity_error[active], velocity_error[active]
    )
    fell[active] |= angle[active] > fall_angle
    event_complete |= active & (event_age_steps >= recovery_steps)

    if event_complete.all():
      break

  completed = event_complete
  destabilizing = completed & ~fell & (
    (max_angle > destabilizing_angle)
    | (
      max_velocity_error - pre_velocity_error
      > cfg.destabilizing_velocity_error_increase
    )
  )
  recoverable = completed & ~fell & ~destabilizing
  triggered = completed & would_trigger_termination
  untriggered = completed & ~would_trigger_termination
  stable_at_onset = completed & (pre_angle <= destabilizing_angle)

  def angle_gate_metrics(threshold_deg: float) -> dict:
    threshold = math.radians(threshold_deg)
    eligible = stable_at_onset
    predicted = eligible & would_trigger_termination & (trigger_angle > threshold)
    bad = eligible & (destabilizing | fell)
    predicted_count = predicted.sum().item()
    bad_count = bad.sum().item()
    return {
      "angle_threshold_deg": threshold_deg,
      "termination_count": predicted_count,
      "bad_event_recall": (predicted & bad).sum().item() / max(bad_count, 1),
      "terminated_event_bad_fraction": (predicted & bad).sum().item()
      / max(predicted_count, 1),
      "recoverable_terminated_count": (predicted & recoverable).sum().item(),
    }

  def classified_subset(mask: torch.Tensor) -> dict:
    count = mask.sum().item()
    return {
      "count": count,
      "recoverable_fraction": (recoverable & mask).sum().item() / max(count, 1),
      "destabilizing_fraction": (destabilizing & mask).sum().item() / max(count, 1),
      "fall_fraction": (fell & mask).sum().item() / max(count, 1),
      "peak_force_mean": _mean(event_peak_force[mask]),
      "duration_substeps_mean": _mean(event_duration_substeps[mask]),
    }

  completed_count = completed.sum().item()
  result = {
    "checkpoint": str(checkpoint),
    "num_envs": cfg.num_envs,
    "steps_requested": cfg.steps,
    "events_started": event_started.sum().item(),
    "events_completed": completed_count,
    "event_coverage": completed_count / cfg.num_envs,
    "recoverable_count": recoverable.sum().item(),
    "destabilizing_count": destabilizing.sum().item(),
    "fall_count": (completed & fell).sum().item(),
    "recoverable_fraction": recoverable.sum().item() / max(completed_count, 1),
    "destabilizing_fraction": destabilizing.sum().item() / max(completed_count, 1),
    "fall_fraction": (completed & fell).sum().item() / max(completed_count, 1),
    "would_trigger_termination": classified_subset(triggered),
    "would_not_trigger_termination": classified_subset(untriggered),
    "stable_at_onset": classified_subset(stable_at_onset),
    "stable_at_onset_and_would_trigger": classified_subset(
      stable_at_onset & would_trigger_termination
    ),
    "stable_at_onset_and_would_not_trigger": classified_subset(
      stable_at_onset & ~would_trigger_termination
    ),
    "trigger_angle_deg_mean": math.degrees(_mean(trigger_angle[triggered])),
    "trigger_velocity_error_mean": _mean(trigger_velocity_error[triggered]),
    "stable_onset_angle_gate_candidates": [
      angle_gate_metrics(threshold) for threshold in (15.0, 25.0, 35.0, 45.0)
    ],
    "peak_force_mean": _mean(event_peak_force[completed]),
    "peak_force_p50": _percentile(event_peak_force[completed], 0.5),
    "peak_force_p90": _percentile(event_peak_force[completed], 0.9),
    "duration_substeps_mean": _mean(event_duration_substeps[completed]),
    "pre_angle_deg_mean": math.degrees(_mean(pre_angle[completed])),
    "max_angle_deg_mean": math.degrees(_mean(max_angle[completed])),
    "pre_velocity_error_mean": _mean(pre_velocity_error[completed]),
    "max_velocity_error_mean": _mean(max_velocity_error[completed]),
    "velocity_error_increase_mean": _mean(
      max_velocity_error[completed] - pre_velocity_error[completed]
    ),
    "count_by_calf_geom": dict(
      zip(calf_geom_names, count_by_calf.cpu().tolist(), strict=True)
    ),
  }
  wrapped_env.close()
  return result


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(DiagnoseConfig)
  results = []
  for checkpoint_str in cfg.checkpoints:
    checkpoint = Path(checkpoint_str).expanduser().resolve()
    if not checkpoint.exists():
      raise FileNotFoundError(checkpoint)
    result = _evaluate_checkpoint(checkpoint, cfg)
    results.append(result)
    print(json.dumps(result, indent=2))

  output = {"config": asdict(cfg), "results": results}
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[INFO] Wrote diagnostics to {output_path}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
