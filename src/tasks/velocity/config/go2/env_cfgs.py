"""Unitree Go2 velocity environment configurations."""

import math
from copy import deepcopy
from dataclasses import replace
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfDiscreteObstaclesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
)

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 2

  # Go2 rough v3: keep the forward terrain traversal bias from V2, but add
  # contact/clearance/smoothness pressure so higher terrain is learned cleanly.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.02
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.15, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (0.15, 0.8),
      "lin_vel_y": (-0.1, 0.1),
      "ang_vel_z": (-0.3, 0.3),
    },
    {
      "step": 10000 * 24,
      "lin_vel_x": (-0.2, 1.0),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 14000 * 24,
      "lin_vel_x": (-0.5, 1.2),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.7, 0.7),
    },
  ]

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["body_orientation_l2"].weight = -1.2
  cfg.rewards["action_rate_l2"].weight = -0.07
  cfg.rewards["foot_clearance"].weight = -1.2
  cfg.rewards["foot_clearance"].params["target_height"] = 0.12
  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 5.0},
  )

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_rough_v4_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Go2 rough v4 with terrain-relative clearance and graded contacts."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      # Go2 visual meshes use group 2 and otherwise contaminate the height map.
      sensor.include_geom_groups = (0,)

  cfg.rewards["foot_clearance"].func = mdp.feet_clearance_terrain_relative
  cfg.rewards["foot_clearance"].params["terrain_sensor_name"] = "terrain_scan"
  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=mdp.contact_force_cost,
    weight=-1.5,
    params={
      "sensor_name": "nonfoot_ground_touch",
      "soft_threshold": 5.0,
      "force_scale": 20.0,
      "max_cost_per_substep": 2.0,
    },
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.sustained_illegal_contact,
    params={
      "sensor_name": "nonfoot_ground_touch",
      "force_threshold": 35.0,
      "min_substeps": 2,
    },
  )
  return cfg


def unitree_go2_flat_rough_obs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat Go2 prior with rough-terrain observation shape.

  This keeps the rough actor/critic observation terms, including height scan,
  so checkpoints can warmstart ``Unitree-Go2-Rough`` without network shape
  changes.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Keep terrain_scan and height_scan so the observation dimensions match rough.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  cfg.events.pop("randomize_terrain", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.0, 0.9)
  twist_cmd.ranges.lin_vel_y = (-0.25, 0.25)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.9),
        "lin_vel_y": (-0.25, 0.25),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (-0.3, 1.2),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  if play:
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  return cfg


def unitree_go2_rough_v5_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Go2 rough v5 with body-part contact attribution."""
  cfg = unitree_go2_rough_v4_env_cfg(play=play)

  def ground_contact_sensor(name: str, pattern: str) -> ContactSensorCfg:
    return ContactSensorCfg(
      name=name,
      primary=ContactMatch(mode="geom", entity="robot", pattern=pattern),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )

  body_sensor = ground_contact_sensor("base_ground_contact", r"base[123]_collision")
  upper_leg_sensor = ground_contact_sensor(
    "upper_leg_ground_contact", r".*_(hip|thigh)_collision"
  )
  calf_sensor = ground_contact_sensor(
    "calf_ground_contact", r".*_calf[12]_collision"
  )
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != "nonfoot_ground_touch"
  ) + (body_sensor, upper_leg_sensor, calf_sensor)

  cfg.rewards["foot_clearance"].params["contact_sensor_name"] = "feet_ground_contact"
  cfg.rewards.pop("nonfoot_contact", None)
  cfg.rewards["base_contact"] = RewardTermCfg(
    func=mdp.contact_force_cost,
    weight=-3.0,
    params={
      "sensor_name": body_sensor.name,
      "soft_threshold": 3.0,
      "force_scale": 15.0,
      "max_cost_per_substep": 3.0,
      "metric_prefix": "base_contact",
    },
  )
  cfg.rewards["upper_leg_contact"] = RewardTermCfg(
    func=mdp.contact_force_cost,
    weight=-2.0,
    params={
      "sensor_name": upper_leg_sensor.name,
      "soft_threshold": 5.0,
      "force_scale": 20.0,
      "max_cost_per_substep": 2.5,
      "metric_prefix": "upper_leg_contact",
    },
  )
  cfg.rewards["calf_contact"] = RewardTermCfg(
    func=mdp.contact_force_cost,
    weight=-0.75,
    params={
      "sensor_name": calf_sensor.name,
      "soft_threshold": 10.0,
      "force_scale": 30.0,
      "max_cost_per_substep": 1.5,
      "metric_prefix": "calf_contact",
    },
  )

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations["illegal_base_contact"] = TerminationTermCfg(
    func=mdp.sustained_illegal_contact,
    params={
      "sensor_name": body_sensor.name,
      "force_threshold": 20.0,
      "min_substeps": 2,
    },
  )
  cfg.terminations["illegal_upper_leg_contact"] = TerminationTermCfg(
    func=mdp.sustained_illegal_contact,
    params={
      "sensor_name": upper_leg_sensor.name,
      "force_threshold": 35.0,
      "min_substeps": 2,
    },
  )
  cfg.terminations["illegal_calf_contact"] = TerminationTermCfg(
    func=mdp.sustained_illegal_contact,
    params={
      "sensor_name": calf_sensor.name,
      "force_threshold": 60.0,
      "min_substeps": 3,
    },
  )
  return cfg


def unitree_go2_rough_v5_1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Go2 rough v5.1 with orientation-gated calf termination."""
  cfg = unitree_go2_rough_v5_env_cfg(play=play)
  cfg.terminations["illegal_calf_contact"].params["min_orientation_angle"] = (
    math.radians(15.0)
  )
  return cfg


def unitree_go2_rough_v6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the frozen V5.1 task with deployment-oriented training diversity."""
  cfg = unitree_go2_rough_v5_1_env_cfg(play=play)

  terrain = cfg.scene.terrain
  assert terrain is not None and terrain.terrain_generator is not None
  terrain.terrain_generator = deepcopy(terrain.terrain_generator)
  if play:
    terrain.terrain_generator.num_cols = 8
  else:
    terrain.max_init_terrain_level = 7
  terrain.terrain_generator.sub_terrains = {
    "flat": BoxFlatTerrainCfg(proportion=0.15),
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.02, 0.12),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.02, 0.12),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
      proportion=0.10,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
      proportion=0.10,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=0.15,
      noise_range=(0.01, 0.06),
      noise_step=0.01,
      border_width=0.25,
    ),
    "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
      proportion=0.20,
      obstacle_width_range=(0.30, 0.80),
      obstacle_height_range=(0.02, 0.10),
      num_obstacles=32,
      platform_width=2.0,
      border_width=0.25,
      origin_z_offset=0.02,
    ),
  }

  if play:
    return cfg

  cfg.events["push_robot"].interval_range_s = (10.0, 15.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
  }
  cfg.events["base_payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "ranges": (-1.0, 3.0),
      "operation": "add",
    },
  )
  cfg.events["motor_strength"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2]),
      "effort_limit_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  return cfg


def unitree_go2_rough_v7_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create V7 with explicit lateral, yaw, and focused high-speed commands."""
  cfg = unitree_go2_rough_v6_env_cfg(play=play)

  old_twist_cmd = cfg.commands["twist"]
  assert isinstance(old_twist_cmd, UniformVelocityCommandCfg)
  cfg.commands["twist"] = mdp.ModeVelocityCommandCfg(
    resampling_time_range=old_twist_cmd.resampling_time_range,
    debug_vis=old_twist_cmd.debug_vis,
    entity_name=old_twist_cmd.entity_name,
    heading_command=False,
    rel_standing_envs=old_twist_cmd.rel_standing_envs,
    rel_heading_envs=0.0,
    init_velocity_prob=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 1.0),
      lin_vel_y=(-0.3, 0.3),
      ang_vel_z=(-0.7, 0.7),
      heading=None,
    ),
    viz=deepcopy(old_twist_cmd.viz),
  )
  cfg.curriculum.pop("command_vel", None)
  return cfg


def unitree_go2_rough_v7_1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create V7.1 with a lateral-heavy two-stage command curriculum."""
  cfg = unitree_go2_rough_v7_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, mdp.ModeVelocityCommandCfg)
  cfg.commands["twist"] = replace(
    twist_cmd,
    general_probability=0.30,
    lateral_probability=0.45,
    yaw_probability=0.10,
    high_speed_probability=0.15,
    lateral_speed=(0.1, 0.2),
    lateral_speed_stages=((250 * 24, (0.1, 0.3)),),
    # model_13600.pt stores this counter; stages are relative to its resume point.
    stage_origin_step=326664,
  )
  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)
  cfg.rewards.pop("nonfoot_contact", None)
  cfg.rewards["body_orientation_l2"].weight = -1.0
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.10

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.rel_heading_envs = 1.0
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (-1.0, 2.0),
        "lin_vel_y": (-1.0, 1.0),
      },
    ]

  if play:
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
