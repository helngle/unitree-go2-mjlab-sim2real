"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  fixed_vx: float | None = None
  """Optional fixed body-frame forward velocity in m/s."""
  fixed_vy: float = 0.0
  """Body-frame lateral velocity used with ``fixed_vx``."""
  fixed_yaw_rate: float = 0.0
  """Yaw rate used with ``fixed_vx``."""
  terrain_demo: Literal[
    "default",
    "stairs_up",
    "stairs_down",
    "stairs_up_down",
    "slope_up",
    "slope_down",
  ] = "default"
  """Place one robot at the entrance of a deterministic terrain route."""
  terrain_level: int = 5
  """Continuous terrain difficulty row, from 0 to 9."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _configure_fixed_velocity_command(env_cfg, command: tuple[float, float, float]) -> None:
  """Lock a velocity task to one body-frame command for interactive playback."""
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
  from src.tasks.velocity.mdp.mode_velocity_command import ModeVelocityCommandCfg

  if len(command) != 3 or not all(math.isfinite(value) for value in command):
    raise ValueError("fixed_command must contain three finite values: vx vy yaw_rate")
  command_cfg = env_cfg.commands.get("twist")
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("fixed_command requires a velocity task with a twist command")

  vx, vy, yaw_rate = command
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (vx, vx)
  command_cfg.ranges.lin_vel_y = (vy, vy)
  command_cfg.ranges.ang_vel_z = (yaw_rate, yaw_rate)
  command_cfg.ranges.heading = None

  if isinstance(command_cfg, ModeVelocityCommandCfg):
    command_cfg.general_probability = 1.0
    command_cfg.lateral_probability = 0.0
    command_cfg.yaw_probability = 0.0
    command_cfg.high_speed_probability = 0.0
    command_cfg.focus_high_speed_probability = 0.0
    command_cfg.general_lin_vel_x = (vx, vx)
    command_cfg.general_lin_vel_y = (vy, vy)
    command_cfg.general_ang_vel_z = (yaw_rate, yaw_rate)
    # Viser builds joystick sliders from the superclass ranges and requires
    # every positive maximum to be at least 0.1. Mode sampling still uses the
    # fixed general_* ranges above while the joystick remains disabled.
    command_cfg.ranges.lin_vel_x = (-max(abs(vx), 0.1), max(abs(vx), 0.1))
    command_cfg.ranges.lin_vel_y = (-max(abs(vy), 0.1), max(abs(vy), 0.1))
    command_cfg.ranges.ang_vel_z = (
      -max(abs(yaw_rate), 0.1),
      max(abs(yaw_rate), 0.1),
    )


def _configure_terrain_demo(env_cfg, terrain_demo: str, terrain_level: int) -> int:
  """Install the evaluation-only route terrain and return its column index."""
  from src.tasks.velocity.evaluation.route_terrains import (
    GENERATOR_NUM_ROWS,
    TERRAIN_KIND_TO_KEY,
    make_continuous_route_terrain_generator,
    make_stairs_up_down_demo_terrain_generator,
  )

  if terrain_demo not in (*TERRAIN_KIND_TO_KEY, "stairs_up_down"):
    raise ValueError(f"unknown terrain demo: {terrain_demo!r}")
  if not 0 <= terrain_level < GENERATOR_NUM_ROWS:
    raise ValueError(
      f"terrain_level must be in [0, {GENERATOR_NUM_ROWS - 1}], got {terrain_level}"
    )

  terrain_cfg = env_cfg.scene.terrain
  if terrain_cfg is None:
    raise ValueError("terrain_demo requires a task with generated terrain")
  if terrain_demo == "stairs_up_down":
    terrain_cfg.terrain_generator = make_stairs_up_down_demo_terrain_generator(
      seed=42
    )
    terrain_key = "pyramid_stairs"
  else:
    terrain_cfg.terrain_generator = make_continuous_route_terrain_generator(
      seed=42
    )
    terrain_key = TERRAIN_KIND_TO_KEY[terrain_demo]
  env_cfg.scene.num_envs = 1
  env_cfg.curriculum = {}
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, 128)

  # Playback should isolate the terrain and command, not domain randomization.
  env_cfg.observations["actor"].enable_corruption = False
  for sensor_cfg in env_cfg.scene.sensors or ():
    if sensor_cfg.name == "terrain_scan":
      sensor_cfg.debug_vis = False
  for name in (
    "foot_friction",
    "encoder_bias",
    "base_com",
    "base_payload",
    "motor_strength",
    "push_robot",
    "randomize_terrain",
  ):
    env_cfg.events.pop(name, None)
  reset_cfg = env_cfg.events["reset_base"]
  reset_cfg.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  reset_cfg.params["velocity_range"] = {}

  return list(terrain_cfg.terrain_generator.sub_terrains).index(terrain_key)


def _place_terrain_demo(env: ManagerBasedRlEnv, terrain_level: int, terrain_column: int) -> float:
  """Relocate the robot after the wrapper reset to the selected route entrance."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise ValueError("terrain demo requires generated terrain origins")
  robot = env.scene["robot"]
  old_origin = terrain.env_origins.clone()
  old_root = robot.data.root_link_pos_w.clone()
  level = min(terrain_level, terrain.max_terrain_level - 1)
  terrain.terrain_levels[:] = level
  terrain.terrain_types[:] = terrain_column
  terrain.env_origins[:] = terrain.terrain_origins[level, terrain_column]

  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :3] += terrain.env_origins - old_origin
  robot.write_root_link_pose_to_sim(root_pose)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  error = torch.max(
    torch.abs(
      (robot.data.root_link_pos_w - terrain.env_origins)
      - (old_root - old_origin)
    )
  )
  if error > 1.0e-4:
    raise RuntimeError(f"terrain demo placement error: {error.item():.6f}")
  return float(error)


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  terrain_column: int | None = None
  fixed_command = (
    None
    if cfg.fixed_vx is None
    else (cfg.fixed_vx, cfg.fixed_vy, cfg.fixed_yaw_rate)
  )
  if cfg.terrain_demo != "default":
    if cfg.num_envs not in (None, 1):
      raise ValueError("terrain_demo supports exactly one environment")
    terrain_column = _configure_terrain_demo(
      env_cfg, cfg.terrain_demo, cfg.terrain_level
    )
    if fixed_command is None:
      fixed_command = (0.4, 0.0, 0.0)
  if fixed_command is not None:
    _configure_fixed_velocity_command(env_cfg, fixed_command)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if terrain_column is not None:
    placement_error = _place_terrain_demo(
      env.unwrapped, cfg.terrain_level, terrain_column
    )
    command_term = env.unwrapped.command_manager.get_term("twist")
    assert fixed_command is not None
    command_term.vel_command_b[:] = torch.tensor(
      fixed_command, device=env.unwrapped.device
    )
    print(
      f"[INFO]: Terrain demo={cfg.terrain_demo}, level={cfg.terrain_level}, "
      f"command={fixed_command}, placement_error={placement_error:.2e}"
    )
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
