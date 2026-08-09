from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from rsl_rl.runners import DistillationRunner

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.env_cfgs import PROPRIO_HISTORY_LENGTH
from src.tasks.velocity.config.go2.env_cfgs import (
  PROPRIO_DISTILL_MAX_TERRAIN_LEVEL,
  PROPRIO_EFFORT_SCALE_RANGE,
  PROPRIO_FRICTION_RANGE,
  PROPRIO_KD_SCALE_RANGE,
  PROPRIO_KP_SCALE_RANGE,
  PROPRIO_LIMB_MASS_SCALE_RANGE,
)
from src.tasks.velocity.config.go2.sim2real_schema import (
  ACTION_SCALE,
  ACTION_ABS_LIMIT,
  CONTROL_DT_S,
  DEFAULT_JOINT_POS,
  JOINT_POS_LIMITS,
  SDK_JOINT_IDS_MAP,
  STUDENT_TERMS,
  actor_dim,
  assemble_actor_observation,
  mock_low_state_to_frame,
  normalized_action_to_sdk_targets,
  schema_sha256,
  schema_payload,
  validate_deploy_yaml,
)
from src.tasks.velocity.rl.runner import (
  V7_TEACHER_SHA256,
  VelocityDistillationRunner,
  VelocityOnPolicyRunner,
)
from src.tasks.velocity.rl.teacher_rollout_distillation import (
  TeacherRolloutDistillation,
)


DISTILL_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill"
PPO_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1"
DEPLOY_YAML = Path(
  "deploy/robots/go2/config/policy/velocity/"
  "v1_proprio_candidate/params/deploy.yaml"
)
SCHEMA_JSON = DEPLOY_YAML.parent / "observation_schema.json"
REJECTED_RUN = (
  "2026-07-27_09-58-02_"
  "go2_rough_v7_local_tangent_stance_slip_2048env_400iter"
)


class TestObservationContract(unittest.TestCase):
  def setUp(self) -> None:
    self.cfg = load_env_cfg(PPO_TASK)

  def test_student_is_deployable_and_has_frozen_history(self) -> None:
    actor = self.cfg.observations["actor"]
    self.assertIsNone(actor.terms["height_scan"])
    self.assertEqual(
      [name for name, term in actor.terms.items() if term is not None],
      [term.training_name for term in STUDENT_TERMS],
    )
    for name in (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      self.assertEqual(actor.terms[name].history_length, PROPRIO_HISTORY_LENGTH)
      self.assertTrue(actor.terms[name].flatten_history_dim)
    for name in ("command", "phase"):
      self.assertEqual(actor.terms[name].history_length, 0)
    self.assertTrue(actor.terms["joint_pos"].params["biased"])
    self.assertEqual(actor_dim(), 425)

  def test_teacher_and_critic_are_isolated_from_student_history(self) -> None:
    distill_cfg = load_env_cfg(DISTILL_TASK)
    teacher = distill_cfg.observations["teacher"]
    critic = self.cfg.observations["critic"]
    self.assertIsNotNone(teacher.terms["height_scan"])
    self.assertIsNotNone(critic.terms["height_scan"])
    self.assertFalse(teacher.enable_corruption)
    self.assertEqual(teacher.history_length, 1)
    self.assertEqual(critic.history_length, 1)
    self.assertEqual(critic.terms["joint_pos"].params.get("biased", False), False)
    self.assertIsNot(
      self.cfg.observations["actor"].terms["joint_pos"],
      critic.terms["joint_pos"],
    )
    self.assertNotIn("teacher", self.cfg.observations)

  def test_registered_runners_and_observation_groups_are_exact(self) -> None:
    self.assertIs(load_runner_cls(DISTILL_TASK), VelocityDistillationRunner)
    self.assertIs(load_runner_cls(PPO_TASK), VelocityOnPolicyRunner)
    self.assertTrue(issubclass(VelocityDistillationRunner, DistillationRunner))
    distill = load_rl_cfg(DISTILL_TASK)
    ppo = load_rl_cfg(PPO_TASK)
    self.assertEqual(distill.obs_groups, {"student": ("actor",), "teacher": ("teacher",)})
    self.assertEqual(ppo.obs_groups, {"actor": ("actor",), "critic": ("critic",)})

  def test_distillation_excludes_teacher_failure_levels(self) -> None:
    cfg = load_env_cfg(DISTILL_TASK)
    self.assertEqual(
      cfg.scene.terrain.max_init_terrain_level,
      PROPRIO_DISTILL_MAX_TERRAIN_LEVEL,
    )
    self.assertNotIn("terrain_levels", cfg.curriculum)

  def test_sim2real_randomization_is_scoped_and_covers_all_groups(self) -> None:
    cfg = self.cfg
    self.assertEqual(cfg.events["foot_friction"].params["ranges"], PROPRIO_FRICTION_RANGE)
    motor = cfg.events["motor_strength"].params
    self.assertEqual(motor["asset_cfg"].actuator_ids, [0, 1, 2])
    self.assertEqual(motor["effort_limit_range"], PROPRIO_EFFORT_SCALE_RANGE)
    gains = cfg.events["pd_gains"].params
    self.assertEqual(gains["asset_cfg"].actuator_ids, [0, 1, 2])
    self.assertEqual(gains["kp_range"], PROPRIO_KP_SCALE_RANGE)
    self.assertEqual(gains["kd_range"], PROPRIO_KD_SCALE_RANGE)
    inertia = cfg.events["limb_pseudo_inertia"].params
    alpha = inertia["alpha_range"]
    self.assertAlmostEqual(torch.exp(torch.tensor(2 * alpha[0])).item(), PROPRIO_LIMB_MASS_SCALE_RANGE[0])
    self.assertAlmostEqual(torch.exp(torch.tensor(2 * alpha[1])).item(), PROPRIO_LIMB_MASS_SCALE_RANGE[1])


class TestDeploySchema(unittest.TestCase):
  def test_schema_and_yaml_match(self) -> None:
    validate_deploy_yaml(DEPLOY_YAML)
    payload = yaml.safe_load(DEPLOY_YAML.read_text(encoding="utf-8"))
    self.assertEqual(tuple(payload["joint_ids_map"]), SDK_JOINT_IDS_MAP)
    self.assertEqual(payload["step_dt"], CONTROL_DT_S)
    self.assertEqual(payload["actions"]["JointPositionAction"]["scale"], [ACTION_SCALE] * 12)
    self.assertEqual(len(schema_sha256()), 64)
    artifact = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    self.assertEqual(artifact.pop("schema_sha256"), schema_sha256())
    canonical_payload = json.loads(json.dumps(schema_payload(), allow_nan=False))
    self.assertEqual(artifact, canonical_payload)

  def test_wrong_history_or_mapping_is_rejected(self) -> None:
    payload = yaml.safe_load(DEPLOY_YAML.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "deploy.yaml"
      payload["observations"]["actor"]["joint_pos_rel"]["history_length"] = 9
      path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "history mismatch"):
        validate_deploy_yaml(path)
      payload["observations"]["actor"]["joint_pos_rel"]["history_length"] = 10
      payload["joint_ids_map"][0] = 0
      path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "SDK joint mapping"):
        validate_deploy_yaml(path)

  def test_wrong_joint_position_safety_limits_are_rejected(self) -> None:
    payload = yaml.safe_load(DEPLOY_YAML.read_text(encoding="utf-8"))
    payload["safety"]["joint_pos_limits"][0][0] -= 0.01
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "deploy.yaml"
      path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "joint position safety limits"):
        validate_deploy_yaml(path)

  def test_cpp_phase_and_runtime_fail_closed_contract_is_present(self) -> None:
    env_source = Path(
      "deploy/include/isaaclab/envs/manager_based_rl_env.h"
    ).read_text(encoding="utf-8")
    phase_source = Path(
      "deploy/include/isaaclab/envs/mdp/observations/observations.h"
    ).read_text(encoding="utf-8")
    state_source = Path("deploy/robots/go2/src/State_RLBase.cpp").read_text(
      encoding="utf-8"
    )
    action_source = Path(
      "deploy/include/isaaclab/manager/action_manager.h"
    ).read_text(encoding="utf-8")
    state_header = Path("deploy/include/FSM/State_RLBase.h").read_text(
      encoding="utf-8"
    )
    hold_source = Path(
      "deploy/include/isaaclab/utils/joint_command_safety.h"
    ).read_text(encoding="utf-8")
    articulation_source = Path("deploy/include/unitree_articulation.h").read_text(
      encoding="utf-8"
    )
    algorithm_source = Path(
      "deploy/include/isaaclab/algorithms/algorithms.h"
    ).read_text(encoding="utf-8")
    self.assertIn("policy_tick += 1", env_source)
    self.assertIn("runtime_fault.store(true)", env_source)
    self.assertIn("joint_pos_limits", env_source)
    self.assertIn("processed[index] < joint_limits[index][0]", env_source)
    self.assertIn("env->policy_tick * env->step_dt", phase_source)
    self.assertNotIn("global_phase +=", phase_source)
    self.assertIn("env->runtime_fault.load()", state_source)
    self.assertIn("!env->action_ready.load()", state_source)
    self.assertIn("std::mutex action_mtx_", action_source)
    self.assertIn("std::atomic<bool> policy_thread_running", state_header)
    self.assertIn("env->reset()", state_header)
    self.assertLess(
      state_header.index("env->reset()"),
      state_header.index("policy_thread_running = true"),
    )
    self.assertIn("initialize_measured_position_hold", state_header)
    self.assertIn("mapped_motors[sdk_id]", hold_source)
    self.assertIn("motor.q() = data.joint_pos[index]", hold_source)
    self.assertIn("motor.kp() = data.joint_stiffness[index]", hold_source)
    self.assertIn("motor.kd() = data.joint_damping[index]", hold_source)
    self.assertIn("data.root_quat_w.normalize()", articulation_source)
    self.assertIn("session->GetOutputCount() != 1", algorithm_source)
    self.assertIn("input_shapes[0][0] != 1", algorithm_source)

  def test_mock_lowstate_history_action_lowcmd_round_trip(self) -> None:
    sdk_q = [float(index) / 10.0 for index in range(12)]
    sdk_dq = [float(index) for index in range(12)]
    frames = []
    for tick in range(10):
      frames.append(
        mock_low_state_to_frame(
          gyroscope=(tick, tick + 0.1, tick + 0.2),
          quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
          sdk_joint_pos=sdk_q,
          sdk_joint_vel=sdk_dq,
          previous_action=(float(tick),) * 12,
        )
      )
    actor = assemble_actor_observation(
      frames, command=(0.5, -0.2, 0.3), phase=(0.0, 1.0)
    )
    self.assertEqual(len(actor), 425)
    self.assertEqual(actor[:3], (0.0, 0.1, 0.2))
    self.assertEqual(actor[27:30], (9.0, 9.1, 9.2))
    self.assertEqual(actor[60:65], (0.5, -0.2, 0.3, 0.0, 1.0))
    self.assertEqual(actor[305:317], (0.0,) * 12)
    self.assertEqual(actor[-12:], (9.0,) * 12)

    sdk_targets = normalized_action_to_sdk_targets([1.0] * 12)
    for training_index, sdk_id in enumerate(SDK_JOINT_IDS_MAP):
      self.assertAlmostEqual(
        sdk_targets[sdk_id], DEFAULT_JOINT_POS[training_index] + ACTION_SCALE
      )

  def test_mock_sdk_chain_rejects_nonfinite_and_action_limit(self) -> None:
    with self.assertRaisesRegex(ValueError, "quaternion norm"):
      mock_low_state_to_frame(
        gyroscope=(0, 0, 0),
        quaternion_wxyz=(0, 0, 0, 0),
        sdk_joint_pos=[0] * 12,
        sdk_joint_vel=[0] * 12,
        previous_action=[0] * 12,
      )
    with self.assertRaisesRegex(ValueError, "NaN/Inf"):
      normalized_action_to_sdk_targets([float("nan")] + [0] * 11)
    with self.assertRaisesRegex(ValueError, "safety limit"):
      normalized_action_to_sdk_targets([ACTION_ABS_LIMIT + 0.01] + [0] * 11)
    # The normalized action is inside the global abs-4 bound, but the front
    # hip target would be -1.1 rad and violate the MJCF mechanical limit.
    with self.assertRaisesRegex(ValueError, "out-of-range target"):
      normalized_action_to_sdk_targets([-4.0] + [0] * 11)
    self.assertEqual(len(JOINT_POS_LIMITS), 12)

  def test_rejected_stance_slip_run_is_not_referenced(self) -> None:
    paths = (
      Path("src/tasks/velocity/config/go2/rl_cfg.py"),
      Path("src/tasks/velocity/config/go2/sim2real_schema.py"),
      DEPLOY_YAML,
    )
    for path in paths:
      with self.subTest(path=path):
        self.assertNotIn(REJECTED_RUN, path.read_text(encoding="utf-8"))


class TestDistillationInitialization(unittest.TestCase):
  def test_distillation_runner_initializes_logging_distribution_without_sampling(self) -> None:
    class Observations(dict):
      def to(self, device):
        self.device = device
        return self

    class Distribution:
      def __init__(self) -> None:
        self.updated = None

      def update(self, value) -> None:
        self.updated = value

    class Student:
      def __init__(self) -> None:
        self.distribution = Distribution()
        self.mlp = lambda latent: latent + 1.0

      def get_latent(self, observations):
        return observations["actor"]

      @property
      def output_std(self):
        return torch.ones(12)

    runner = object.__new__(VelocityDistillationRunner)
    runner.env = SimpleNamespace(
      get_observations=lambda: Observations(actor=torch.zeros(1, 425))
    )
    runner.device = "cpu"
    runner.alg = SimpleNamespace(student=Student())
    with patch.object(DistillationRunner, "learn") as parent_learn:
      runner.learn(3, True)
    self.assertIsNotNone(runner.alg.student.distribution.updated)
    parent_learn.assert_called_once_with(3, True)

  def test_distillation_checkpoint_initializes_only_ppo_actor(self) -> None:
    calls: list[tuple[dict, dict, bool]] = []

    class FakeAlgorithm:
      def load(self, payload, load_cfg, strict):
        calls.append((payload, load_cfg, strict))
        return False

    runner = object.__new__(VelocityOnPolicyRunner)
    runner.alg = FakeAlgorithm()
    checkpoint = {
      "student_state_dict": {"mlp.0.weight": torch.ones(1)},
      "teacher_state_dict": {"mlp.0.weight": torch.zeros(1)},
      "optimizer_state_dict": {"state": {1: {}}},
      "iter": 299,
      "infos": {
        "stage": "distillation",
        "proprioceptive_stage": "distillation",
        "student_schema_sha256": schema_sha256(),
        "teacher_sha256": V7_TEACHER_SHA256,
      },
    }
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model_299.pt"
      torch.save(checkpoint, path)
      infos = runner.load(str(path), strict=True, map_location="cpu")

    self.assertEqual(infos["stage"], "distillation")
    self.assertEqual(len(calls), 1)
    payload, load_cfg, strict = calls[0]
    self.assertIs(payload["actor_state_dict"], payload["student_state_dict"])
    self.assertEqual(load_cfg, {"actor": True, "iteration": False})
    self.assertTrue(strict)

  def test_distillation_config_does_not_reference_rejected_checkpoint(self) -> None:
    cfg = asdict(load_rl_cfg(DISTILL_TASK))
    self.assertNotIn(REJECTED_RUN, repr(cfg))
    self.assertEqual(cfg["load_checkpoint"], "model_13600.pt")
    self.assertEqual(cfg["algorithm"]["gradient_length"], cfg["num_steps_per_env"])
    self.assertEqual(
      cfg["algorithm"]["class_name"],
      "src.tasks.velocity.rl.teacher_rollout_distillation:TeacherRolloutDistillation",
    )

  def test_teacher_rollout_returns_teacher_action(self) -> None:
    algorithm = object.__new__(TeacherRolloutDistillation)
    algorithm.teacher = lambda obs: obs["teacher"] + 2.0
    algorithm.transition = SimpleNamespace()
    observations = {"teacher": torch.tensor([[1.0, 3.0]])}
    action = algorithm.act(observations)
    torch.testing.assert_close(action, torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(algorithm.transition.actions, action)
    torch.testing.assert_close(algorithm.transition.privileged_actions, action)

  def test_update_consumes_all_batches_and_uses_elementwise_mean(self) -> None:
    student = torch.nn.Linear(1, 1, bias=False)
    student.weight.data.zero_()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    batches = [
      SimpleNamespace(
        observations=torch.tensor([[float(index)]]),
        privileged_actions=torch.tensor([[1.0]]),
        dones=torch.zeros(1),
      )
      for index in range(24)
    ]

    class Model(torch.nn.Module):
      def __init__(self) -> None:
        super().__init__()
        self.linear = student

      def forward(self, obs):
        return self.linear(obs)

      def reset(self, *args, **kwargs):
        return None

      def detach_hidden_state(self, *args, **kwargs):
        return None

      def get_hidden_state(self):
        return None

    algorithm = object.__new__(TeacherRolloutDistillation)
    algorithm.num_updates = 0
    algorithm.student = Model()
    algorithm.teacher = Model()
    algorithm.optimizer = optimizer
    algorithm.storage = SimpleNamespace(
      num_transitions_per_env=24,
      generator=lambda: iter(batches),
      clear=lambda: None,
    )
    algorithm.loss_fn = torch.nn.functional.huber_loss
    algorithm.device = "cpu"
    algorithm.last_hidden_states = (None, None)
    algorithm.is_multi_gpu = False
    algorithm.max_grad_norm = None
    result = algorithm.update()
    self.assertEqual(algorithm.last_update_batch_count, 24)
    self.assertEqual(algorithm.last_optimizer_step_count, 1)
    self.assertAlmostEqual(result["behavior"], 0.5)
    self.assertNotEqual(float(student.weight.detach()), 0.0)


if __name__ == "__main__":
  unittest.main()
