from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.config.go2.env_cfgs import (
  unitree_go2_rough_v7_env_cfg,
  unitree_go2_rough_v7_high_slope_probe_env_cfg,
)
from src.tasks.velocity.config.go2.high_slope_sampling import (
  HighSlopeHardCaseSampler,
  TARGET_HARD_CASE_RATIO,
  hard_case_slot_mask,
  high_slope_sampling_metric,
  reweighted_slot_probabilities,
  sample_reweighted_slots,
  terrain_column_names,
)
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner


TASK_ID = "Unitree-Go2-Rough-V7-HighSlopeProbe"
TELEMETRY_METRICS = (
  "candidate_hard_ratio",
  "changed_slot_ratio",
  "hard_case_batch_ratio",
  "hard_case_reset_ratio",
  "hard_case_population_ratio",
  "total_reset_count",
  "total_hard_count",
)


def _sub_terrains() -> dict[str, SimpleNamespace]:
  return {
    "flat": SimpleNamespace(proportion=0.15),
    "pyramid_stairs": SimpleNamespace(proportion=0.15),
    "pyramid_stairs_inv": SimpleNamespace(proportion=0.15),
    "hf_pyramid_slope": SimpleNamespace(proportion=0.10),
    "hf_pyramid_slope_inv": SimpleNamespace(proportion=0.10),
    "random_rough": SimpleNamespace(proportion=0.15),
    "discrete_obstacles": SimpleNamespace(proportion=0.20),
  }


def _sampler_params() -> dict[str, object]:
  return {
    "target_hard_case_ratio": TARGET_HARD_CASE_RATIO,
    "slope_up_levels": (8, 9),
    "slope_down_levels": (9,),
    "seed_offset": 700,
  }


def _asdict_mapping(values: dict[str, object]) -> dict[str, object]:
  return {name: asdict(value) for name, value in values.items()}


def _fake_env(num_envs: int = 2048, seed: int = 42) -> SimpleNamespace:
  num_rows, num_cols = 10, 20
  slot_ids = torch.arange(num_envs) % (num_rows * num_cols)
  levels = torch.div(slot_ids, num_cols, rounding_mode="floor")
  types = slot_ids.remainder(num_cols)
  terrain_origins = torch.zeros(num_rows, num_cols, 3)
  for level in range(num_rows):
    for terrain_type in range(num_cols):
      terrain_origins[level, terrain_type] = torch.tensor(
        (float(level), float(terrain_type), float(level * num_cols + terrain_type))
      )
  generator_cfg = SimpleNamespace(
    curriculum=True,
    sub_terrains=_sub_terrains(),
  )
  terrain = SimpleNamespace(
    terrain_levels=levels.clone(),
    terrain_types=types.clone(),
    terrain_origins=terrain_origins,
    env_origins=terrain_origins[levels, types].clone(),
    max_terrain_level=num_rows,
    cfg=SimpleNamespace(terrain_generator=generator_cfg),
  )
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    cfg=SimpleNamespace(seed=seed),
    scene=SimpleNamespace(terrain=terrain),
  )
  return env


def _make_sampler(env: SimpleNamespace) -> HighSlopeHardCaseSampler:
  cfg = SimpleNamespace(params=_sampler_params())
  return HighSlopeHardCaseSampler(cfg=cfg, env=env)  # type: ignore[arg-type]


class TestHighSlopeSamplingMath(unittest.TestCase):
  def setUp(self) -> None:
    self.column_names = terrain_column_names(_sub_terrains(), 20)
    self.hard_mask = hard_case_slot_mask(
      num_rows=10,
      column_names=self.column_names,
    )

  def test_v7_columns_and_nominal_hard_case_ratio(self) -> None:
    self.assertEqual(self.column_names.count("hf_pyramid_slope"), 2)
    self.assertEqual(self.column_names.count("hf_pyramid_slope_inv"), 2)
    self.assertEqual(int(self.hard_mask.sum()), 6)
    self.assertAlmostEqual(float(self.hard_mask.float().mean()), 0.03)

  def test_reweight_preserves_both_conditional_distributions(self) -> None:
    base = torch.arange(1, 201, dtype=torch.float64)
    result = reweighted_slot_probabilities(base, self.hard_mask, 0.10)
    self.assertAlmostEqual(float(result[self.hard_mask].sum()), 0.10)
    self.assertAlmostEqual(float(result[~self.hard_mask].sum()), 0.90)
    for mask in (self.hard_mask, ~self.hard_mask):
      expected = base[mask] / base[mask].sum()
      actual = result[mask] / result[mask].sum()
      torch.testing.assert_close(actual, expected)

  def test_membership_target_changes_only_required_candidates(self) -> None:
    candidate_slots = torch.arange(200, dtype=torch.long)
    donor = reweighted_slot_probabilities(
      torch.ones(200), self.hard_mask, TARGET_HARD_CASE_RATIO
    )
    generator = torch.Generator().manual_seed(17)
    sampled, hard_count, changed_count, residual = sample_reweighted_slots(
      candidate_slots,
      donor,
      self.hard_mask,
      target_hard_case_ratio=TARGET_HARD_CASE_RATIO,
      quota_residual=0.0,
      generator=generator,
    )
    candidate_hard_count = int(self.hard_mask[candidate_slots].sum())
    self.assertEqual(candidate_hard_count, 6)
    self.assertEqual(hard_count, 20)
    self.assertEqual(changed_count, 14)
    self.assertAlmostEqual(residual, 0.0)
    changed = sampled != candidate_slots
    self.assertEqual(int(changed.sum()), changed_count)
    self.assertTrue(torch.equal(sampled[~changed], candidate_slots[~changed]))
    self.assertEqual(int(self.hard_mask[sampled].sum()), hard_count)

  def test_cumulative_quota_handles_single_environment_batches(self) -> None:
    donor = reweighted_slot_probabilities(
      torch.ones(200), self.hard_mask, TARGET_HARD_CASE_RATIO
    )
    generator = torch.Generator().manual_seed(3)
    residual = 0.0
    total_hard = 0
    for _ in range(10):
      candidate = torch.tensor([0])
      sampled, hard_count, _, residual = sample_reweighted_slots(
        candidate,
        donor,
        self.hard_mask,
        target_hard_case_ratio=TARGET_HARD_CASE_RATIO,
        quota_residual=residual,
        generator=generator,
      )
      total_hard += hard_count
      self.assertEqual(int(self.hard_mask[sampled].sum()), hard_count)
    self.assertEqual(total_hard, 1)
    self.assertAlmostEqual(residual, 0.0, places=12)


class TestHighSlopeSamplerIntegration(unittest.TestCase):
  def test_2048_reset_ratio_minimal_change_and_origin_relocation(self) -> None:
    env = _fake_env()
    sampler = _make_sampler(env)
    terrain = env.scene.terrain
    old_slots = terrain.terrain_levels * 20 + terrain.terrain_types
    old_origins = terrain.env_origins.clone()
    old_hard_count = int(sampler.hard_mask[old_slots].sum())

    sampler(env, slice(None))

    new_slots = terrain.terrain_levels * 20 + terrain.terrain_types
    hard_count = int(sampler.hard_mask[new_slots].sum())
    expected_hard_count = int(env.num_envs * TARGET_HARD_CASE_RATIO)
    expected_changed = abs(expected_hard_count - old_hard_count)
    changed = new_slots != old_slots
    self.assertEqual(hard_count, expected_hard_count)
    self.assertEqual(int(changed.sum()), expected_changed)
    self.assertAlmostEqual(sampler.hard_case_batch_ratio, 204 / 2048)
    self.assertAlmostEqual(sampler.changed_slot_ratio, expected_changed / 2048)
    torch.testing.assert_close(terrain.env_origins[~changed], old_origins[~changed])
    torch.testing.assert_close(
      terrain.env_origins,
      terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )

  def test_deterministic_seed_and_tensor_batch(self) -> None:
    env_a, env_b = _fake_env(), _fake_env()
    sampler_a, sampler_b = _make_sampler(env_a), _make_sampler(env_b)
    env_ids = torch.arange(512)
    sampler_a(env_a, env_ids)
    sampler_b(env_b, env_ids)
    torch.testing.assert_close(
      env_a.scene.terrain.terrain_levels, env_b.scene.terrain.terrain_levels
    )
    torch.testing.assert_close(
      env_a.scene.terrain.terrain_types, env_b.scene.terrain.terrain_types
    )
    self.assertAlmostEqual(sampler_a.hard_case_reset_ratio, sampler_b.hard_case_reset_ratio)

  def test_metric_exposes_recorded_reset_ratio(self) -> None:
    env = _fake_env()
    sampler = _make_sampler(env)
    sampler(env, slice(None))
    env.event_manager = SimpleNamespace(
      get_term_cfg=lambda name: SimpleNamespace(func=sampler)
    )
    values = high_slope_sampling_metric(
      env, metric_name="hard_case_reset_ratio"
    )
    self.assertEqual(values.shape, (env.num_envs,))
    torch.testing.assert_close(
      values, torch.full_like(values, sampler.hard_case_reset_ratio)
    )

  def test_sampled_stream_preserves_group_conditionals(self) -> None:
    env = _fake_env()
    sampler = _make_sampler(env)
    terrain = env.scene.terrain
    baseline_levels = terrain.terrain_levels.clone()
    baseline_types = terrain.terrain_types.clone()
    baseline_slots = baseline_levels * 20 + baseline_types
    baseline_histogram = torch.bincount(baseline_slots, minlength=200).float()

    for _ in range(50):
      terrain.terrain_levels.copy_(baseline_levels)
      terrain.terrain_types.copy_(baseline_types)
      terrain.env_origins.copy_(
        terrain.terrain_origins[baseline_levels, baseline_types]
      )
      sampler(env, slice(None))

    audit = sampler.sampling_audit()
    histogram = audit["sampled_slot_histogram"].float()
    hard_mask = audit["hard_slot_mask"]
    self.assertEqual(audit["total_reset_count"], 50 * 2048)
    self.assertEqual(audit["total_hard_count"], 10_240)
    self.assertAlmostEqual(audit["hard_case_reset_ratio"], 0.10)
    self.assertEqual(int(histogram.sum()), audit["total_reset_count"])
    for mask in (hard_mask, ~hard_mask):
      actual = histogram[mask] / histogram[mask].sum()
      expected = baseline_histogram[mask] / baseline_histogram[mask].sum()
      self.assertLess(float(torch.max(torch.abs(actual - expected))), 0.015)

  def test_rebase_and_state_round_trip(self) -> None:
    env_a = _fake_env()
    sampler_a = _make_sampler(env_a)
    sampler_a(env_a, slice(None))
    state = sampler_a.state_dict()

    env_b = _fake_env()
    env_b.scene.terrain.terrain_levels.copy_(env_a.scene.terrain.terrain_levels)
    env_b.scene.terrain.terrain_types.copy_(env_a.scene.terrain.terrain_types)
    sampler_b = _make_sampler(env_b)
    sampler_b.load_state_dict(state)
    self.assertEqual(sampler_b.total_reset_count, 2048)
    self.assertEqual(sampler_b.total_hard_count, 204)
    torch.testing.assert_close(
      sampler_b.sampled_slot_histogram, sampler_a.sampled_slot_histogram
    )

    baseline = _fake_env()
    for env, sampler in ((env_a, sampler_a), (env_b, sampler_b)):
      env.scene.terrain.terrain_levels.copy_(baseline.scene.terrain.terrain_levels)
      env.scene.terrain.terrain_types.copy_(baseline.scene.terrain.terrain_types)
      sampler(env, torch.arange(512))
    torch.testing.assert_close(
      env_a.scene.terrain.terrain_levels, env_b.scene.terrain.terrain_levels
    )
    torch.testing.assert_close(
      env_a.scene.terrain.terrain_types, env_b.scene.terrain.terrain_types
    )
    torch.testing.assert_close(
      sampler_a.sampled_slot_histogram, sampler_b.sampled_slot_histogram
    )

    hard_slots = sampler_a.hard_mask.nonzero().flatten()
    terrain = env_a.scene.terrain
    restored_slots = torch.zeros(env_a.num_envs, dtype=torch.long)
    restored_slots[:64] = hard_slots.repeat(11)[:64]
    terrain.terrain_levels.copy_(torch.div(restored_slots, 20, rounding_mode="floor"))
    terrain.terrain_types.copy_(restored_slots.remainder(20))
    sampler_a.rebase()
    self.assertEqual(sampler_a.total_reset_count, 0)
    self.assertEqual(sampler_a.total_hard_count, 0)
    self.assertEqual(int(sampler_a.sampled_slot_histogram.sum()), 0)
    self.assertAlmostEqual(sampler_a.hard_case_population_ratio, 64 / 2048)


class _FakeScene(dict):
  def __init__(self, terrain: SimpleNamespace, robot: SimpleNamespace):
    super().__init__(robot=robot)
    self.terrain = terrain

  def write_data_to_sim(self) -> None:
    pass


class TestSamplerRunnerPersistence(unittest.TestCase):
  def _runner(self) -> tuple[VelocityOnPolicyRunner, SimpleNamespace, HighSlopeHardCaseSampler]:
    env = _fake_env()
    sampler = _make_sampler(env)
    sampler(env, slice(None))
    robot = SimpleNamespace()
    robot.data = SimpleNamespace(
      root_link_pose_w=torch.cat(
        (
          env.scene.terrain.env_origins.clone(),
          torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(env.num_envs, 1),
        ),
        dim=1,
      )
    )

    def write_root_link_pose_to_sim(pose: torch.Tensor) -> None:
      robot.data.root_link_pose_w.copy_(pose)

    robot.write_root_link_pose_to_sim = write_root_link_pose_to_sim
    env.scene = _FakeScene(env.scene.terrain, robot)
    env.event_manager = SimpleNamespace(
      get_term_cfg=lambda name: SimpleNamespace(func=sampler)
    )
    env.common_step_counter = 326664
    env.sim = SimpleNamespace(forward=lambda: None, sense=lambda: None)
    runner = object.__new__(VelocityOnPolicyRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    return runner, env, sampler

  def test_probe_checkpoint_restores_exact_sampler_stream(self) -> None:
    runner, env, sampler = self._runner()
    saved = runner._environment_state()
    saved_histogram = saved["high_slope_sampling"]["sampled_slot_histogram"].clone()
    sampler.rebase()
    env.scene.terrain.terrain_levels.zero_()
    env.scene.terrain.terrain_types.zero_()
    runner._restore_environment_state(saved)
    self.assertEqual(sampler.total_reset_count, 2048)
    self.assertEqual(sampler.total_hard_count, 204)
    torch.testing.assert_close(sampler.sampled_slot_histogram.cpu(), saved_histogram)

  def test_v7_checkpoint_rebases_after_terrain_restore(self) -> None:
    runner, env, sampler = self._runner()
    hard_slots = sampler.hard_mask.nonzero().flatten()
    restored_slots = torch.zeros(env.num_envs, dtype=torch.long)
    restored_slots[:64] = hard_slots.repeat(11)[:64]
    old_v7_state = {
      "common_step_counter": 326664,
      "terrain_levels": torch.div(restored_slots, 20, rounding_mode="floor"),
      "terrain_types": restored_slots.remainder(20),
    }
    runner._restore_environment_state(old_v7_state)
    self.assertEqual(sampler.total_reset_count, 0)
    self.assertEqual(sampler.total_hard_count, 0)
    self.assertAlmostEqual(sampler.hard_case_population_ratio, 64 / 2048)
    self.assertEqual(int(sampler.sampled_slot_histogram.sum()), 0)


class TestHighSlopeProbeConfig(unittest.TestCase):
  def test_task_registration_and_event_order(self) -> None:
    cfg = load_env_cfg(TASK_ID)
    reset_events = [name for name, term in cfg.events.items() if term.mode == "reset"]
    self.assertLess(
      reset_events.index("high_slope_sampling"), reset_events.index("reset_base")
    )
    self.assertEqual(
      cfg.events["high_slope_sampling"].params["target_hard_case_ratio"], 0.10
    )
    self.assertIs(load_runner_cls(TASK_ID), load_runner_cls("Unitree-Go2-Rough-V7"))

  def test_probe_diff_is_sampler_and_telemetry_only(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg()
    probe = unitree_go2_rough_v7_high_slope_probe_env_cfg()
    probe_events = dict(probe.events)
    probe_events.pop("high_slope_sampling")
    probe_metrics = dict(probe.metrics)
    for name in TELEMETRY_METRICS:
      probe_metrics.pop(name)

    self.assertEqual(_asdict_mapping(baseline.rewards), _asdict_mapping(probe.rewards))
    self.assertEqual(_asdict_mapping(baseline.commands), _asdict_mapping(probe.commands))
    self.assertEqual(
      _asdict_mapping(baseline.terminations), _asdict_mapping(probe.terminations)
    )
    self.assertEqual(
      _asdict_mapping(baseline.observations), _asdict_mapping(probe.observations)
    )
    self.assertEqual(_asdict_mapping(baseline.actions), _asdict_mapping(probe.actions))
    self.assertEqual(_asdict_mapping(baseline.events), _asdict_mapping(probe_events))
    self.assertEqual(_asdict_mapping(baseline.metrics), _asdict_mapping(probe_metrics))
    self.assertEqual(
      _asdict_mapping(baseline.curriculum), _asdict_mapping(probe.curriculum)
    )
    baseline_terrain = asdict(baseline.scene.terrain)
    probe_terrain = asdict(probe.scene.terrain)
    baseline_terrain.pop("spec_fn")
    probe_terrain.pop("spec_fn")
    self.assertEqual(baseline_terrain, probe_terrain)

  def test_play_config_is_unchanged_v7(self) -> None:
    baseline = unitree_go2_rough_v7_env_cfg(play=True)
    probe = unitree_go2_rough_v7_high_slope_probe_env_cfg(play=True)
    self.assertEqual(_asdict_mapping(baseline.rewards), _asdict_mapping(probe.rewards))
    self.assertNotIn("high_slope_sampling", probe.events)
    for name in TELEMETRY_METRICS:
      self.assertNotIn(name, probe.metrics)


if __name__ == "__main__":
  unittest.main()
