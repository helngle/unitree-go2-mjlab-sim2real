from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import (
  VelocitySafeActionDistillationRunner,
  VelocitySafeActionOnPolicyRunner,
  VelocityDistillationRunner,
  VelocityOnPolicyRunner,
)
from src.tasks.velocity.rl.privileged_teacher_transfer import (
  Go2PrivilegedTeacherTransferRunner,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (
  Go2ContactForceTeacherTransferRunner,
)

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_rough_obs_env_cfg,
  unitree_go2_rough_contact_force_teacher_control_env_cfg,
  unitree_go2_rough_contact_force_teacher_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_rough_v4_env_cfg,
  unitree_go2_rough_v5_1_env_cfg,
  unitree_go2_rough_v5_env_cfg,
  unitree_go2_rough_v6_env_cfg,
  unitree_go2_rough_v7_1_env_cfg,
  unitree_go2_rough_v7_env_cfg,
  unitree_go2_rough_v7_high_slope_probe_env_cfg,
  unitree_go2_rough_v7_lateral_pose_env_cfg,
  unitree_go2_rough_v7_sim2real_proprio_distill_env_cfg,
  unitree_go2_rough_v7_sim2real_proprio_env_cfg,
  unitree_go2_rough_v7_sim2real_proprio_safe_action_distill_env_cfg,
  unitree_go2_rough_v7_sim2real_proprio_safe_action_env_cfg,
  unitree_go2_rough_v7_stance_slip_env_cfg,
  unitree_go2_rough_v8_privileged_lin_vel_teacher_control_env_cfg,
  unitree_go2_rough_v8_privileged_lin_vel_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_go2_contact_force_teacher_runner_cfg,
  unitree_go2_privileged_teacher_runner_cfg,
  unitree_go2_ppo_runner_cfg,
  unitree_go2_proprio_distillation_runner_cfg,
  unitree_go2_proprio_ppo_runner_cfg,
  unitree_go2_proprio_safe_action_distillation_runner_cfg,
  unitree_go2_proprio_safe_action_ppo_runner_cfg,
)

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
  task_id="Unitree-Go2-Rough-V7-HighSlopeProbe",
  env_cfg=unitree_go2_rough_v7_high_slope_probe_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_high_slope_probe_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V7-StanceSlip",
  env_cfg=unitree_go2_rough_v7_stance_slip_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_stance_slip_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher",
  env_cfg=unitree_go2_rough_v8_privileged_lin_vel_teacher_env_cfg(),
  play_env_cfg=unitree_go2_rough_v8_privileged_lin_vel_teacher_env_cfg(play=True),
  rl_cfg=unitree_go2_privileged_teacher_runner_cfg(),
  runner_cls=Go2PrivilegedTeacherTransferRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher-Control",
  env_cfg=unitree_go2_rough_v8_privileged_lin_vel_teacher_control_env_cfg(),
  play_env_cfg=unitree_go2_rough_v8_privileged_lin_vel_teacher_control_env_cfg(
    play=True
  ),
  rl_cfg=unitree_go2_privileged_teacher_runner_cfg(control=True),
  runner_cls=Go2PrivilegedTeacherTransferRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-ContactForceTeacher-V1",
  env_cfg=unitree_go2_rough_contact_force_teacher_env_cfg(),
  play_env_cfg=unitree_go2_rough_contact_force_teacher_env_cfg(play=True),
  rl_cfg=unitree_go2_contact_force_teacher_runner_cfg(),
  runner_cls=Go2ContactForceTeacherTransferRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-ContactForceTeacher-V1-Control",
  env_cfg=unitree_go2_rough_contact_force_teacher_control_env_cfg(),
  play_env_cfg=unitree_go2_rough_contact_force_teacher_control_env_cfg(
    play=True
  ),
  rl_cfg=unitree_go2_contact_force_teacher_runner_cfg(control=True),
  runner_cls=Go2ContactForceTeacherTransferRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill",
  env_cfg=unitree_go2_rough_v7_sim2real_proprio_distill_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_sim2real_proprio_distill_env_cfg(play=True),
  rl_cfg=unitree_go2_proprio_distillation_runner_cfg(),
  runner_cls=VelocityDistillationRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-Sim2Real-Proprio-V1",
  env_cfg=unitree_go2_rough_v7_sim2real_proprio_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_sim2real_proprio_env_cfg(play=True),
  rl_cfg=unitree_go2_proprio_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill",
  env_cfg=unitree_go2_rough_v7_sim2real_proprio_safe_action_distill_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_sim2real_proprio_safe_action_distill_env_cfg(play=True),
  rl_cfg=unitree_go2_proprio_safe_action_distillation_runner_cfg(),
  runner_cls=VelocitySafeActionDistillationRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction",
  env_cfg=unitree_go2_rough_v7_sim2real_proprio_safe_action_env_cfg(),
  play_env_cfg=unitree_go2_rough_v7_sim2real_proprio_safe_action_env_cfg(play=True),
  rl_cfg=unitree_go2_proprio_safe_action_ppo_runner_cfg(),
  runner_cls=VelocitySafeActionOnPolicyRunner,
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
