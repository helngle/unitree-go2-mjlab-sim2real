from __future__ import annotations

from dataclasses import asdict
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.env_cfgs import (
  unitree_go2_rough_contact_force_teacher_control_env_cfg,
  unitree_go2_rough_contact_force_teacher_env_cfg,
  unitree_go2_rough_v7_env_cfg,
)
from src.tasks.velocity.contact_force_teacher_schema import (
  CANDIDATE_ACTOR_DIM,
  CANDIDATE_ACTOR_TERM_SLICES,
  CONTACT_FORCE_ACTOR_SLICE,
  CONTACT_FORCE_CRITIC_SLICE,
  CONTACT_FORCE_FRAME,
  CONTACT_FORCE_PREPROCESSING,
  CRITIC_TERM_SLICES,
  INTERVENTION,
  NORMALIZER_SOURCE,
  REFERENCE_DOI,
  REFERENCE_URL,
  SOURCE_ACTOR_DIM,
  TASK_IDS,
  schema_sha256,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (
  Go2ContactForceTeacherTransferRunner,
  map_actor_state_dict,
)


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


class TestContactForceTeacherConfig(unittest.TestCase):
  def test_schema_records_reference_and_exact_slices(self) -> None:
    self.assertEqual(REFERENCE_DOI, "10.1126/scirobotics.abc5986")
    self.assertEqual(REFERENCE_URL, "https://arxiv.org/abs/2010.11251")
    self.assertEqual(CONTACT_FORCE_ACTOR_SLICE, (234, 246))
    self.assertEqual(CONTACT_FORCE_CRITIC_SLICE, (249, 261))
    self.assertEqual(
      CANDIDATE_ACTOR_TERM_SLICES["foot_contact_forces"], (234, 246)
    )
    self.assertEqual(CRITIC_TERM_SLICES["foot_contact_forces"], (249, 261))
    self.assertEqual(CONTACT_FORCE_FRAME, "world")
    self.assertEqual(
      CONTACT_FORCE_PREPROCESSING, "sign(force) * log1p(abs(force))"
    )
    self.assertEqual(NORMALIZER_SOURCE, "source_critic[249:261]")
    self.assertEqual(INTERVENTION, "append_actor_foot_contact_forces_only")
    self.assertEqual(len(schema_sha256()), 64)

  def test_candidate_changes_only_appended_actor_term(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    candidate = unitree_go2_rough_contact_force_teacher_env_cfg()
    baseline_actor = baseline.observations["actor"]
    candidate_actor = candidate.observations["actor"]
    self.assertEqual(
      list(candidate_actor.terms),
      [*baseline_actor.terms, "foot_contact_forces"],
    )
    self.assertEqual(
      asdict(candidate_actor.terms["foot_contact_forces"]),
      asdict(
        candidate.observations["critic"].terms["foot_contact_forces"]
      ),
    )
    candidate_actor.terms.pop("foot_contact_forces")
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

  def test_control_is_exact_v7_and_tasks_are_registered(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    control = unitree_go2_rough_contact_force_teacher_control_env_cfg()
    for field in (
      "observations", "rewards", "commands", "terminations", "actions",
      "events", "metrics", "curriculum",
    ):
      self.assertEqual(
        _asdict_mapping(getattr(baseline, field)),
        _asdict_mapping(getattr(control, field)),
      )
    for arm, dim in (("control_234", 234), ("candidate_246", 246)):
      task_id = TASK_IDS[arm]
      self.assertIs(
        load_runner_cls(task_id), Go2ContactForceTeacherTransferRunner
      )
      cfg = load_rl_cfg(task_id)
      self.assertEqual(cfg.seed, 42)
      self.assertEqual(cfg.max_iterations, 400)
      self.assertEqual(cfg.save_interval, 100)
      self.assertIn(arm, cfg.run_name)
      actor_terms = load_env_cfg(task_id).observations["actor"].terms
      observed_dim = SOURCE_ACTOR_DIM + (
        12 if "foot_contact_forces" in actor_terms else 0
      )
      self.assertEqual(observed_dim, dim)


class TestContactForceTeacherTransfer(unittest.TestCase):
  def test_candidate_mapping_uses_force_normalizer_and_zero_actor_columns(self) -> None:
    torch.manual_seed(7)
    source = _actor_state(SOURCE_ACTOR_DIM)
    critic = _critic_normalizer_state()
    target = _actor_state(CANDIDATE_ACTOR_DIM)
    mapped = map_actor_state_dict(source, target, critic)
    torch.testing.assert_close(
      mapped["mlp.0.weight"][:, :SOURCE_ACTOR_DIM],
      source["mlp.0.weight"],
    )
    self.assertTrue(
      torch.equal(
        mapped["mlp.0.weight"][:, SOURCE_ACTOR_DIM:],
        torch.zeros_like(mapped["mlp.0.weight"][:, SOURCE_ACTOR_DIM:]),
      )
    )
    start, end = CONTACT_FORCE_CRITIC_SLICE
    for key in (
      "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"
    ):
      torch.testing.assert_close(
        mapped[key][..., SOURCE_ACTOR_DIM:], critic[key][..., start:end]
      )

  def test_iteration_zero_actions_are_exactly_equal(self) -> None:
    torch.manual_seed(11)
    source = _actor_state(SOURCE_ACTOR_DIM)
    critic = _critic_normalizer_state()
    mapped = map_actor_state_dict(
      source, _actor_state(CANDIDATE_ACTOR_DIM), critic
    )
    source_obs = torch.randn(8, SOURCE_ACTOR_DIM)
    force_obs = torch.randn(8, 12)
    candidate_obs = torch.cat((source_obs, force_obs), dim=-1)
    source_norm = (
      source_obs - source["obs_normalizer._mean"]
    ) / source["obs_normalizer._std"]
    candidate_norm = (
      candidate_obs - mapped["obs_normalizer._mean"]
    ) / mapped["obs_normalizer._std"]
    source_out = torch.nn.functional.linear(
      source_norm, source["mlp.0.weight"], source["mlp.0.bias"]
    )
    candidate_out = torch.nn.functional.linear(
      candidate_norm, mapped["mlp.0.weight"], mapped["mlp.0.bias"]
    )
    torch.testing.assert_close(candidate_out, source_out, rtol=0.0, atol=0.0)

  def test_wrong_critic_or_target_dimensions_fail_closed(self) -> None:
    with self.assertRaisesRegex(ValueError, "not 261-D"):
      map_actor_state_dict(
        _actor_state(SOURCE_ACTOR_DIM),
        _actor_state(CANDIDATE_ACTOR_DIM),
        {
          **_critic_normalizer_state(),
          "obs_normalizer._mean": torch.randn(1, 260),
        },
      )
    with self.assertRaisesRegex(ValueError, "locked 246-D"):
      map_actor_state_dict(
        _actor_state(SOURCE_ACTOR_DIM),
        _actor_state(245),
        _critic_normalizer_state(),
      )


if __name__ == "__main__":
  unittest.main()
