from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.env_cfgs import (
  unitree_go2_rough_v7_env_cfg,
  unitree_go2_rough_v8_privileged_lin_vel_teacher_control_env_cfg,
  unitree_go2_rough_v8_privileged_lin_vel_teacher_env_cfg,
)
from src.tasks.velocity.rl.privileged_teacher_transfer import (
  ALLOW_ENV_STATE_MISMATCH_ENV,
  CANDIDATE_ACTOR_DIM,
  SOURCE_ACTOR_DIM,
  SOURCE_CHECKPOINT,
  SOURCE_CHECKPOINT_SHA256,
  Go2PrivilegedTeacherTransferRunner,
  _validate_probe_checkpoint_infos,
  environment_state_sha256,
  map_actor_state_dict,
  map_critic_state_dict,
  reset_matched_rng,
  sha256_file,
)
from src.tasks.velocity.privileged_teacher_schema import (
  BASE_LIN_VEL_FRAME,
  BASE_LIN_VEL_SLICE,
  BASE_LIN_VEL_UNIT,
  CANDIDATE_ACTOR_TERM_SLICES,
  CRITIC_TERM_SLICES,
  INTERVENTION,
  MATCHED_RNG_SEED,
  NORMALIZER_SOURCE,
  SCHEMA_VERSION,
  SOURCE_COMMON_STEP_COUNTER,
  SOURCE_ENVIRONMENT_NUM_ENVS,
  SOURCE_ENVIRONMENT_STATE_SHA256,
  TASK_IDS,
  schema_sha256,
)
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner


CANDIDATE_TASK = "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher"
CONTROL_TASK = "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher-Control"


def _asdict_mapping(values: dict[str, object]) -> dict[str, object]:
  return {name: asdict(value) for name, value in values.items()}


def _actor_state(dim: int) -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.randn(1, dim),
    "obs_normalizer._var": torch.rand(1, dim) + 0.1,
    "obs_normalizer._std": torch.rand(1, dim) + 0.1,
    "obs_normalizer.count": torch.tensor(123.0),
    "distribution.std_param": torch.randn(12),
    "mlp.0.weight": torch.randn(512, dim),
    "mlp.0.bias": torch.randn(512),
    "mlp.2.weight": torch.randn(256, 512),
    "mlp.2.bias": torch.randn(256),
    "mlp.4.weight": torch.randn(128, 256),
    "mlp.4.bias": torch.randn(128),
    "mlp.6.weight": torch.randn(12, 128),
    "mlp.6.bias": torch.randn(12),
  }


def _critic_normalizer_state() -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.randn(1, 261),
    "obs_normalizer._var": torch.rand(1, 261) + 0.1,
    "obs_normalizer._std": torch.rand(1, 261) + 0.1,
    "obs_normalizer.count": torch.tensor(123.0),
  }


class TestPrivilegedTeacherConfig(unittest.TestCase):
  def test_frozen_schema_locks_offsets_semantics_and_identity(self) -> None:
    self.assertEqual(BASE_LIN_VEL_SLICE, (234, 237))
    self.assertEqual(CANDIDATE_ACTOR_TERM_SLICES["base_lin_vel"], (234, 237))
    self.assertEqual(CRITIC_TERM_SLICES["base_lin_vel"], (234, 237))
    self.assertEqual(BASE_LIN_VEL_UNIT, "m/s")
    self.assertEqual(BASE_LIN_VEL_FRAME, "imu_site_local_body_aligned")
    self.assertEqual(NORMALIZER_SOURCE, "source_critic[234:237]")
    self.assertEqual(INTERVENTION, "append_actor_base_lin_vel_only")
    self.assertEqual(TASK_IDS["candidate_237"], CANDIDATE_TASK)
    self.assertEqual(TASK_IDS["control_234"], CONTROL_TASK)
    self.assertEqual(len(schema_sha256()), 64)

  def test_registered_tasks_use_transfer_runner_and_frozen_training_cfg(self) -> None:
    for task_id, dim, arm in (
      (CANDIDATE_TASK, 237, "candidate_237"),
      (CONTROL_TASK, 234, "control_234"),
    ):
      self.assertIs(load_runner_cls(task_id), Go2PrivilegedTeacherTransferRunner)
      cfg = load_rl_cfg(task_id)
      self.assertEqual(cfg.seed, 42)
      self.assertEqual(cfg.max_iterations, 400)
      self.assertEqual(cfg.save_interval, 100)
      self.assertTrue(cfg.resume)
      self.assertEqual(cfg.load_checkpoint, "model_13600.pt")
      self.assertIn(arm, cfg.run_name)
      env_cfg = load_env_cfg(task_id)
      terms = env_cfg.observations["actor"].terms
      expected_dim = SOURCE_ACTOR_DIM + (3 if "base_lin_vel" in terms else 0)
      self.assertEqual(expected_dim, dim)

  def test_candidate_changes_only_appended_actor_term(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    candidate = unitree_go2_rough_v8_privileged_lin_vel_teacher_env_cfg()
    baseline_actor = baseline.observations["actor"]
    candidate_actor = candidate.observations["actor"]
    self.assertEqual(
      list(candidate_actor.terms), [*baseline_actor.terms, "base_lin_vel"]
    )
    lin_vel = candidate_actor.terms["base_lin_vel"]
    self.assertEqual(lin_vel.params, {"sensor_name": "robot/imu_lin_vel"})
    self.assertIsNone(lin_vel.noise)
    candidate_actor.terms.pop("base_lin_vel")
    self.assertEqual(asdict(baseline_actor), asdict(candidate_actor))
    self.assertEqual(
      _asdict_mapping(baseline.observations["critic"].terms),
      _asdict_mapping(candidate.observations["critic"].terms),
    )
    for field in (
      "rewards", "commands", "terminations", "actions", "events", "curriculum"
    ):
      self.assertEqual(
        _asdict_mapping(getattr(baseline, field)),
        _asdict_mapping(getattr(candidate, field)),
      )

  def test_control_is_exact_v7_environment(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    control = unitree_go2_rough_v8_privileged_lin_vel_teacher_control_env_cfg()
    for field in (
      "observations",
      "rewards",
      "commands",
      "terminations",
      "actions",
      "events",
      "metrics",
      "curriculum",
    ):
      self.assertEqual(
        _asdict_mapping(getattr(baseline, field)),
        _asdict_mapping(getattr(control, field)),
      )
    baseline_terrain = asdict(baseline.scene.terrain)
    control_terrain = asdict(control.scene.terrain)
    baseline_terrain.pop("spec_fn")
    control_terrain.pop("spec_fn")
    self.assertEqual(baseline_terrain, control_terrain)


class TestPrivilegedTeacherTransfer(unittest.TestCase):
  def test_candidate_mapping_preserves_old_state_and_initializes_new_dims(self) -> None:
    torch.manual_seed(7)
    source = _actor_state(SOURCE_ACTOR_DIM)
    source_critic = _critic_normalizer_state()
    target = _actor_state(CANDIDATE_ACTOR_DIM)
    mapped = map_actor_state_dict(source, target, source_critic)
    for key, source_value in source.items():
      mapped_value = mapped[key]
      if source_value.shape == mapped_value.shape:
        torch.testing.assert_close(mapped_value, source_value)
      else:
        torch.testing.assert_close(
          mapped_value[..., :SOURCE_ACTOR_DIM], source_value
        )
    self.assertTrue(
      torch.equal(
        mapped["mlp.0.weight"][..., SOURCE_ACTOR_DIM:],
        torch.zeros_like(mapped["mlp.0.weight"][..., SOURCE_ACTOR_DIM:]),
      )
    )
    for key in (
      "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"
    ):
      torch.testing.assert_close(
        mapped[key][..., SOURCE_ACTOR_DIM:],
        source_critic[key][..., SOURCE_ACTOR_DIM:CANDIDATE_ACTOR_DIM],
      )

  def test_iteration_zero_candidate_has_exact_first_layer_parity(self) -> None:
    torch.manual_seed(11)
    source = _actor_state(SOURCE_ACTOR_DIM)
    target = _actor_state(CANDIDATE_ACTOR_DIM)
    mapped = map_actor_state_dict(source, target, _critic_normalizer_state())
    source_obs = torch.randn(8, SOURCE_ACTOR_DIM)
    candidate_obs = torch.cat((source_obs, torch.randn(8, 3)), dim=-1)
    source_normalized = (
      source_obs - source["obs_normalizer._mean"]
    ) / source["obs_normalizer._std"]
    candidate_normalized = (
      candidate_obs - mapped["obs_normalizer._mean"]
    ) / mapped["obs_normalizer._std"]
    source_out = torch.nn.functional.linear(
      source_normalized, source["mlp.0.weight"], source["mlp.0.bias"]
    )
    candidate_out = torch.nn.functional.linear(
      candidate_normalized, mapped["mlp.0.weight"], mapped["mlp.0.bias"]
    )
    torch.testing.assert_close(candidate_out, source_out)

  def test_control_mapping_is_exact_clone(self) -> None:
    source = _actor_state(SOURCE_ACTOR_DIM)
    mapped = map_actor_state_dict(source, _actor_state(SOURCE_ACTOR_DIM))
    for key in source:
      torch.testing.assert_close(mapped[key], source[key])
      self.assertIsNot(mapped[key], source[key])

  def test_critic_rejects_any_shape_change(self) -> None:
    source = {"mlp.0.weight": torch.randn(512, 261)}
    mapped = map_critic_state_dict(source, {"mlp.0.weight": torch.empty(512, 261)})
    torch.testing.assert_close(mapped["mlp.0.weight"], source["mlp.0.weight"])
    with self.assertRaisesRegex(ValueError, "shape differs"):
      map_critic_state_dict(source, {"mlp.0.weight": torch.empty(512, 260)})

  def test_locked_checkpoint_provenance_when_present(self) -> None:
    if not SOURCE_CHECKPOINT.exists():
      self.skipTest(f"workspace checkpoint absent: {SOURCE_CHECKPOINT}")
    self.assertEqual(Path(SOURCE_CHECKPOINT).name, "model_13600.pt")
    self.assertEqual(sha256_file(SOURCE_CHECKPOINT), SOURCE_CHECKPOINT_SHA256)

  def test_locked_checkpoint_contains_exact_normalizer_and_env_sources(self) -> None:
    if not SOURCE_CHECKPOINT.exists():
      self.skipTest(f"workspace checkpoint absent: {SOURCE_CHECKPOINT}")
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    self.assertEqual(actor["mlp.0.weight"].shape, (512, 234))
    self.assertEqual(critic["mlp.0.weight"].shape, (512, 261))
    self.assertTrue(
      torch.equal(
        actor["obs_normalizer.count"], critic["obs_normalizer.count"]
      )
    )
    for key in (
      "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"
    ):
      self.assertEqual(critic[key][..., 234:237].shape, (1, 3))
      self.assertTrue(torch.isfinite(critic[key][..., 234:237]).all())
    env_state = payload["infos"]["env_state"]
    self.assertEqual(env_state["terrain_levels"].numel(), 2048)
    self.assertEqual(env_state["terrain_types"].numel(), 2048)
    self.assertEqual(
      environment_state_sha256(env_state), SOURCE_ENVIRONMENT_STATE_SHA256
    )
    self.assertEqual(SOURCE_ENVIRONMENT_NUM_ENVS, 2048)
    self.assertEqual(env_state["common_step_counter"], SOURCE_COMMON_STEP_COUNTER)

  def test_rng_reset_removes_network_construction_draw_difference(self) -> None:
    reset_matched_rng(42)
    control_draws = torch.randn(4, 12)
    _ = torch.randn(1536)
    reset_matched_rng(42)
    candidate_draws = torch.randn(4, 12)
    torch.testing.assert_close(candidate_draws, control_draws)

  def test_source_learning_uses_completed_update_checkpoint_labels(self) -> None:
    runner = object.__new__(Go2PrivilegedTeacherTransferRunner)
    runner.current_learning_iteration = 0
    runner._source_transfer_ready = True
    with patch.object(VelocityOnPolicyRunner, "learn") as parent_learn:
      runner.learn(400, init_at_random_ep_len=True)
    self.assertEqual(runner.current_learning_iteration, 1)
    self.assertFalse(runner._source_transfer_ready)
    parent_learn.assert_called_once_with(400, True)
    self.assertEqual(list(range(1, 1 + 400))[-1], 400)
    save_iterations = [it for it in range(1, 401) if it % 100 == 0]
    self.assertEqual(save_iterations, [100, 200, 300, 400])

  def test_probe_checkpoint_metadata_is_fail_closed(self) -> None:
    infos = {
      "privileged_teacher_schema_version": SCHEMA_VERSION,
      "privileged_teacher_schema_sha256": schema_sha256(),
      "privileged_teacher_arm": "candidate_237",
      "privileged_teacher_task_id": CANDIDATE_TASK,
      "transfer_source_sha256": SOURCE_CHECKPOINT_SHA256,
      "transfer_mode": "v7_iteration_0_fresh_optimizer",
      "source_actor_dim": 234,
      "target_actor_dim": 237,
      "critic_dim": 261,
      "action_dim": 12,
      "intervention": INTERVENTION,
      "normalizer_source": NORMALIZER_SOURCE,
      "optimizer_restored": False,
      "iteration_restored": False,
      "environment_state_restored_exact": True,
      "technical_smoke_env_state_override": False,
      "source_environment_num_envs": SOURCE_ENVIRONMENT_NUM_ENVS,
      "source_environment_state_sha256": SOURCE_ENVIRONMENT_STATE_SHA256,
      "source_common_step_counter": SOURCE_COMMON_STEP_COUNTER,
      "rng_reset_after_transfer": True,
      "rng_seed": MATCHED_RNG_SEED,
      "checkpoint_labels_are_completed_updates": True,
      "new_actor_columns_zero_initialized": True,
    }
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "source_manifest.json"
      manifest.write_text("{}\n", encoding="ascii")
      infos["source_manifest"] = str(manifest.resolve())
      infos["source_manifest_sha256"] = sha256_file(manifest)
      _validate_probe_checkpoint_infos(infos, 237)
      infos["privileged_teacher_schema_sha256"] = "bad"
      with self.assertRaisesRegex(ValueError, "metadata differs"):
        _validate_probe_checkpoint_infos(infos, 237)

  def test_environment_state_mismatch_override_name_is_explicit(self) -> None:
    self.assertEqual(
      ALLOW_ENV_STATE_MISMATCH_ENV,
      "GO2_PRIV_TEACHER_ALLOW_ENV_STATE_MISMATCH",
    )


if __name__ == "__main__":
  unittest.main()
