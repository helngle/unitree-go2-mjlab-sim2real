from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_rough_obs_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_rough_v4_env_cfg,
  unitree_go2_rough_v5_1_env_cfg,
  unitree_go2_rough_v5_env_cfg,
  unitree_go2_rough_v6_env_cfg,
  unitree_go2_rough_v7_1_env_cfg,
  unitree_go2_rough_v7_env_cfg,
  unitree_go2_rough_v7_lateral_pose_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-RoughObs",
  env_cfg=unitree_go2_flat_rough_obs_env_cfg(),
  play_env_cfg=unitree_go2_flat_rough_obs_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V4",
  env_cfg=unitree_go2_rough_v4_env_cfg(),
  play_env_cfg=unitree_go2_rough_v4_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V5",
  env_cfg=unitree_go2_rough_v5_env_cfg(),
  play_env_cfg=unitree_go2_rough_v5_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V5.1",
  env_cfg=unitree_go2_rough_v5_1_env_cfg(),
  play_env_cfg=unitree_go2_rough_v5_1_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V6",
  env_cfg=unitree_go2_rough_v6_env_cfg(),
  play_env_cfg=unitree_go2_rough_v6_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V7",
  env_cfg=unitree_go2_rough_v7_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V7.1",
  env_cfg=unitree_go2_rough_v7_1_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_1_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V7-LateralPose",
  env_cfg=unitree_go2_rough_v7_lateral_pose_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_lateral_pose_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
