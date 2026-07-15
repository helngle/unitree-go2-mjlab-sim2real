"""Configuration tests for fixed-command web playback demos."""

from __future__ import annotations

import unittest

import mjlab
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import tyro
from mjlab.tasks.registry import load_env_cfg

from scripts.play import (
  PlayConfig,
  _configure_fixed_velocity_command,
  _configure_terrain_demo,
)
from src.tasks.velocity.mdp.mode_velocity_command import ModeVelocityCommandCfg


class FixedVelocityDemoTest(unittest.TestCase):
  def test_v7_command_is_locked_to_general_forward_mode(self) -> None:
    env_cfg = load_env_cfg("Unitree-Go2-Rough-V7", play=True)
    _configure_fixed_velocity_command(env_cfg, (0.4, 0.0, 0.0))
    command = env_cfg.commands["twist"]
    self.assertIsInstance(command, ModeVelocityCommandCfg)
    self.assertEqual(command.resampling_time_range, (1.0e9, 1.0e9))
    self.assertEqual(command.general_probability, 1.0)
    self.assertEqual(command.lateral_probability, 0.0)
    self.assertEqual(command.yaw_probability, 0.0)
    self.assertEqual(command.high_speed_probability, 0.0)
    self.assertEqual(command.focus_high_speed_probability, 0.0)
    self.assertEqual(command.general_lin_vel_x, (0.4, 0.4))
    self.assertEqual(command.general_lin_vel_y, (0.0, 0.0))
    self.assertEqual(command.general_ang_vel_z, (0.0, 0.0))
    self.assertEqual(command.ranges.lin_vel_x, (-0.4, 0.4))
    self.assertEqual(command.ranges.lin_vel_y, (-0.1, 0.1))
    self.assertEqual(command.ranges.ang_vel_z, (-0.1, 0.1))

  def test_stairs_demo_uses_clean_single_env_and_fixed_spawn(self) -> None:
    env_cfg = load_env_cfg("Unitree-Go2-Rough-V7", play=True)
    column = _configure_terrain_demo(env_cfg, "stairs_up", 5)
    self.assertEqual(column, 0)
    self.assertEqual(env_cfg.scene.num_envs, 1)
    self.assertEqual(env_cfg.sim.nconmax, 128)
    self.assertEqual(env_cfg.curriculum, {})
    self.assertFalse(env_cfg.observations["actor"].enable_corruption)
    self.assertNotIn("push_robot", env_cfg.events)
    self.assertNotIn("randomize_terrain", env_cfg.events)
    self.assertEqual(
      env_cfg.events["reset_base"].params["pose_range"]["yaw"], (0.0, 0.0)
    )

  def test_invalid_command_and_level_are_rejected(self) -> None:
    env_cfg = load_env_cfg("Unitree-Go2-Rough-V7", play=True)
    with self.assertRaises(ValueError):
      _configure_fixed_velocity_command(env_cfg, (float("nan"), 0.0, 0.0))
    with self.assertRaises(ValueError):
      _configure_terrain_demo(env_cfg, "stairs_up", 10)

  def test_main_cli_flags_parse_fixed_velocity_components(self) -> None:
    cfg = tyro.cli(
      PlayConfig,
      args=[
        "--terrain-demo", "stairs_up",
        "--terrain-level", "5",
        "--fixed-vx", "0.4",
        "--fixed-vy", "0.0",
        "--fixed-yaw-rate", "0.0",
        "--viewer", "viser",
      ],
      default=PlayConfig(),
      config=mjlab.TYRO_FLAGS,
    )
    self.assertEqual(cfg.fixed_vx, 0.4)
    self.assertEqual(cfg.fixed_vy, 0.0)
    self.assertEqual(cfg.fixed_yaw_rate, 0.0)


if __name__ == "__main__":
  unittest.main()
