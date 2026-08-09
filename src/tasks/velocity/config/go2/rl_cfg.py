"""RL configuration for Unitree Go2 velocity task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_go2_privileged_teacher_runner_cfg(
  *, control: bool = False
) -> RslRlOnPolicyRunnerCfg:
  """Create the matched iteration-0 transfer probe configuration."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.seed = 42
  cfg.max_iterations = 400
  cfg.save_interval = 100
  cfg.logger = "tensorboard"
  cfg.resume = True
  cfg.load_run = (
    "2026-07-14_11-29-13_"
    "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter"
  )
  cfg.load_checkpoint = "model_13600.pt"
  arm = "control_234" if control else "candidate_237"
  cfg.run_name = f"go2_v8_privileged_lin_vel_teacher_{arm}_2048env_400iter"
  return cfg


def unitree_go2_contact_force_teacher_runner_cfg(
  *, control: bool = False
) -> RslRlOnPolicyRunnerCfg:
  """Create the matched contact-force Teacher transfer-probe configuration."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.seed = 42
  cfg.max_iterations = 400
  cfg.save_interval = 100
  cfg.logger = "tensorboard"
  cfg.resume = True
  cfg.load_run = (
    "2026-07-14_11-29-13_"
    "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter"
  )
  cfg.load_checkpoint = "model_13600.pt"
  arm = "control_234" if control else "candidate_246"
  cfg.run_name = f"go2_contact_force_teacher_v1_{arm}_2048env_400iter"
  return cfg


def _go2_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _go2_bounded_policy_model_cfg() -> RslRlModelCfg:
  from .sim2real_safe_action_schema import (
    ACTION_HIGH,
    ACTION_LOW,
    ACTION_MEAN_BOUND,
  )

  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": (
        "src.tasks.velocity.rl.bounded_action_distribution:"
        "AsymmetricBoundedGaussianDistribution"
      ),
      "action_low": ACTION_LOW,
      "action_high": ACTION_HIGH,
      "mean_bound": ACTION_MEAN_BOUND,
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


@dataclass
class Go2ProprioDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Fixed V7-teacher distillation stage for the deployable student."""

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=_go2_policy_model_cfg)
  teacher: RslRlModelCfg = field(default_factory=_go2_policy_model_cfg)
  algorithm: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": (
        "src.tasks.velocity.rl.teacher_rollout_distillation:"
        "TeacherRolloutDistillation"
      ),
      "num_learning_epochs": 1,
      "gradient_length": 24,
      "learning_rate": 1.0e-3,
      "max_grad_norm": 1.0,
      "loss_type": "huber",
      "optimizer": "adam",
    }
  )


def unitree_go2_proprio_distillation_runner_cfg(
) -> Go2ProprioDistillationRunnerCfg:
  """Create the frozen first stage of the proprioceptive training arm."""
  return Go2ProprioDistillationRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=300,
    obs_groups={"student": ("actor",), "teacher": ("teacher",)},
    save_interval=100,
    experiment_name="go2_velocity",
    run_name="go2_sim2real_proprio_v1_v7_teacher_distill_2048env_300iter",
    logger="tensorboard",
    resume=True,
    load_run=(
      "2026-07-14_11-29-13_"
      "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter"
    ),
    load_checkpoint="model_13600.pt",
  )


def unitree_go2_proprio_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the frozen PPO stage initialized from distillation output."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.seed = 42
  cfg.max_iterations = 4000
  cfg.save_interval = 250
  cfg.experiment_name = "go2_velocity"
  cfg.run_name = "go2_sim2real_proprio_v1_ppo_2048env_4000iter"
  cfg.logger = "tensorboard"
  cfg.resume = True
  cfg.load_run = "DISTILLATION_RUN_RESOLVED_BY_ORCHESTRATOR"
  cfg.load_checkpoint = "model_299.pt"
  return cfg


@dataclass
class Go2ProprioSafeActionDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """V2 distillation with bounded student actions and an unchanged V7 teacher."""

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=_go2_bounded_policy_model_cfg)
  teacher: RslRlModelCfg = field(default_factory=_go2_policy_model_cfg)
  algorithm: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": (
        "src.tasks.velocity.rl.teacher_rollout_distillation:"
        "BoundedTeacherRolloutDistillation"
      ),
      "num_learning_epochs": 1,
      "gradient_length": 24,
      "learning_rate": 1.0e-3,
      "max_grad_norm": 1.0,
      "loss_type": "huber",
      "optimizer": "adam",
    }
  )


def unitree_go2_proprio_safe_action_distillation_runner_cfg(
) -> Go2ProprioSafeActionDistillationRunnerCfg:
  return Go2ProprioSafeActionDistillationRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=300,
    obs_groups={"student": ("actor",), "teacher": ("teacher",)},
    save_interval=100,
    experiment_name="go2_velocity",
    run_name=(
      "go2_sim2real_proprio_v2_safe_action_meanbound5_"
      "v7_teacher_distill_2048env_300iter"
    ),
    logger="tensorboard",
    resume=True,
    load_run=(
      "2026-07-14_11-29-13_"
      "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter"
    ),
    load_checkpoint="model_13600.pt",
  )


def unitree_go2_proprio_safe_action_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_go2_proprio_ppo_runner_cfg()
  cfg.actor = _go2_bounded_policy_model_cfg()
  cfg.run_name = (
    "go2_sim2real_proprio_v2_safe_action_meanbound5_ppo_2048env_4000iter"
  )
  cfg.load_run = "SAFE_ACTION_DISTILLATION_RUN_RESOLVED_BY_ORCHESTRATOR"
  return cfg
