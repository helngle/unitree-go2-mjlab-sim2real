from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from torch.distributions import Normal

from rsl_rl.modules.distribution import GaussianDistribution

from src.tasks.velocity.config.go2.rl_cfg import (
  unitree_go2_proprio_distillation_runner_cfg,
  unitree_go2_proprio_ppo_runner_cfg,
  unitree_go2_proprio_safe_action_distillation_runner_cfg,
  unitree_go2_proprio_safe_action_ppo_runner_cfg,
)
from src.tasks.velocity.config.go2.sim2real_schema import (
  ACTION_ABS_LIMIT,
  ACTION_SCALE,
  DEFAULT_JOINT_POS,
  JOINT_POS_LIMITS,
)
from src.tasks.velocity.rl.bounded_action_distribution import (
  AsymmetricBoundedGaussianDistribution,
)
from src.tasks.velocity.rl.teacher_rollout_distillation import (
  BoundedTeacherRolloutDistillation,
  TeacherRolloutDistillation,
)


def _bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
  low = tuple(
    max(-ACTION_ABS_LIMIT, (lower - default) / ACTION_SCALE)
    for default, (lower, _) in zip(DEFAULT_JOINT_POS, JOINT_POS_LIMITS, strict=True)
  )
  high = tuple(
    min(ACTION_ABS_LIMIT, (upper - default) / ACTION_SCALE)
    for default, (_, upper) in zip(DEFAULT_JOINT_POS, JOINT_POS_LIMITS, strict=True)
  )
  return low, high


def _distribution() -> AsymmetricBoundedGaussianDistribution:
  low, high = _bounds()
  return AsymmetricBoundedGaussianDistribution(
    12, action_low=low, action_high=high
  )


class TestAsymmetricBoundedGaussian(unittest.TestCase):
  def test_bounds_match_go2_joint_target_contract(self) -> None:
    low, high = _bounds()
    self.assertAlmostEqual(low[0], -3.7888)
    self.assertAlmostEqual(high[3], 3.7888)
    self.assertAlmostEqual(low[2], -3.6908)
    self.assertAlmostEqual(high[2], 3.84896)
    for index, (lower, upper) in enumerate(zip(low, high, strict=True)):
      for action in (lower, 0.0, upper):
        target = DEFAULT_JOINT_POS[index] + ACTION_SCALE * action
        self.assertGreaterEqual(target + 1.0e-7, JOINT_POS_LIMITS[index][0])
        self.assertLessEqual(target - 1.0e-7, JOINT_POS_LIMITS[index][1])

  def test_zero_and_finite_extremes_map_to_expected_sides(self) -> None:
    distribution = _distribution()
    latent = torch.stack(
      (torch.full((12,), -100.0), torch.zeros(12), torch.full((12,), 100.0))
    )
    action = distribution.transform(latent)
    low, high = _bounds()
    torch.testing.assert_close(action[0], torch.tensor(low))
    torch.testing.assert_close(action[1], torch.zeros(12))
    torch.testing.assert_close(action[2], torch.tensor(high))

  def test_one_hundred_thousand_finite_inputs_are_safe(self) -> None:
    distribution = _distribution()
    generator = torch.Generator().manual_seed(42)
    latent = torch.randn((100_000, 12), generator=generator) * 20.0
    action = distribution.transform(latent)
    low, high = _bounds()
    self.assertTrue(torch.isfinite(action).all())
    self.assertTrue((action >= torch.tensor(low)).all())
    self.assertTrue((action <= torch.tensor(high)).all())
    targets = torch.tensor(DEFAULT_JOINT_POS) + ACTION_SCALE * action
    limits = torch.tensor(JOINT_POS_LIMITS)
    self.assertTrue((targets >= limits[:, 0]).all())
    self.assertTrue((targets <= limits[:, 1]).all())

  def test_nonfinite_latent_and_applied_actions_fail_explicitly(self) -> None:
    distribution = _distribution()
    with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
      distribution.transform(torch.tensor([[float("nan")] + [0.0] * 11]))
    distribution.update(torch.zeros((1, 12)))
    with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
      distribution.log_prob(torch.tensor([[float("inf")] + [0.0] * 11]))
    low, _ = _bounds()
    invalid = torch.zeros((1, 12))
    invalid[0, 0] = low[0] - 0.01
    with self.assertRaisesRegex(RuntimeError, "outside"):
      distribution.log_prob(invalid)

  def test_inverse_log_prob_includes_asymmetric_tanh_jacobian(self) -> None:
    distribution = _distribution()
    mean = torch.linspace(-0.2, 0.2, 12).reshape(1, -1)
    distribution.update(mean)
    latent = torch.linspace(-1.0, 1.0, 24).reshape(2, 12)
    action = distribution.transform(latent)
    actual = distribution.log_prob(action)
    low, high = _bounds()
    unit = torch.tanh(latent)
    scale = torch.where(
      unit >= 0.0, torch.tensor(high), -torch.tensor(low)
    )
    expected = (
      Normal(mean.expand_as(latent), torch.ones_like(latent)).log_prob(latent)
      - torch.log(scale * (1.0 - unit.square()))
    ).sum(dim=-1)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)

  def test_boundary_log_prob_and_sample_entropy_are_finite(self) -> None:
    distribution = _distribution()
    distribution.update(torch.zeros((2, 12)))
    low, high = _bounds()
    boundary = torch.stack((torch.tensor(low), torch.tensor(high)))
    self.assertTrue(torch.isfinite(distribution.log_prob(boundary)).all())
    sample = distribution.sample()
    self.assertTrue(torch.isfinite(sample).all())
    self.assertTrue(torch.isfinite(distribution.entropy).all())

  def test_extreme_raw_means_are_finite_and_bounded_before_sampling(self) -> None:
    distribution = _distribution()
    raw = torch.tensor([[1.0e20, -1.0e20] + [0.0] * 10])
    distribution.update(raw)
    mean, std = distribution.params
    self.assertTrue(torch.isfinite(mean).all())
    self.assertTrue(torch.isfinite(std).all())
    self.assertLessEqual(float(mean.abs().max()), distribution.mean_bound)
    self.assertGreater(float(std.min()), 0.0)
    sample = distribution.sample()
    self.assertTrue(torch.isfinite(sample).all())
    self.assertTrue(torch.isfinite(distribution.log_prob(sample)).all())

  def test_deterministic_output_uses_the_registered_mean_bound(self) -> None:
    distribution = _distribution()
    raw = torch.full((2, 12), 1.0e6)
    bounded_mean = distribution.mean_bound * torch.tanh(
      raw / distribution.mean_bound
    )
    expected = distribution.transform(bounded_mean)
    torch.testing.assert_close(distribution.deterministic_output(raw), expected)
    torch.testing.assert_close(
      distribution.as_deterministic_output_module()(raw), expected
    )

  def test_transformed_entropy_has_finite_pathwise_gradients(self) -> None:
    distribution = _distribution()
    mean = torch.full((4, 12), 0.3, requires_grad=True)
    distribution.update(mean)
    distribution.sample()
    (-distribution.entropy.mean()).backward()
    self.assertIsNotNone(mean.grad)
    self.assertTrue(torch.isfinite(mean.grad).all())
    self.assertGreater(float(mean.grad.abs().sum()), 0.0)
    self.assertIsNotNone(distribution.std_param.grad)
    self.assertTrue(torch.isfinite(distribution.std_param.grad).all())

  def test_exact_kl_is_computed_in_latent_space(self) -> None:
    distribution = _distribution()
    old = (torch.zeros((3, 12)), torch.ones((3, 12)))
    new = (torch.full((3, 12), 0.2), torch.full((3, 12), 0.8))
    actual = distribution.kl_divergence(old, new)
    expected = torch.distributions.kl_divergence(
      Normal(*old), Normal(*new)
    ).sum(dim=-1)
    torch.testing.assert_close(actual, expected)

  def test_old_gaussian_distribution_state_loads_strictly(self) -> None:
    old = GaussianDistribution(12, init_std=0.7, std_type="scalar")
    bounded = _distribution()
    bounded.load_state_dict(old.state_dict(), strict=True)
    self.assertEqual(set(bounded.state_dict()), {"std_param"})
    torch.testing.assert_close(bounded.std_param, old.std_param)

  def test_deterministic_module_exports_applied_actions_to_onnx(self) -> None:
    distribution = _distribution()
    module = distribution.as_deterministic_output_module().eval()
    inputs = torch.linspace(-3.0, 3.0, 36).reshape(3, 12)
    expected = module(inputs).detach().numpy()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "bounded.onnx"
      torch.onnx.export(
        module,
        torch.zeros((3, 12)),
        path,
        opset_version=18,
        input_names=["latent"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
      )
      graph = onnx.load(path)
      actual = ReferenceEvaluator(graph).run(
        None, {"latent": inputs.numpy()}
      )[0]
    np.testing.assert_allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


class TestBoundedDistillationContract(unittest.TestCase):
  def test_v1_configs_retain_the_unbounded_gaussian_contract(self) -> None:
    distill = asdict(unitree_go2_proprio_distillation_runner_cfg())
    ppo = asdict(unitree_go2_proprio_ppo_runner_cfg())
    self.assertEqual(
      distill["student"]["distribution_cfg"]["class_name"],
      "GaussianDistribution",
    )
    self.assertEqual(
      distill["algorithm"]["class_name"],
      "src.tasks.velocity.rl.teacher_rollout_distillation:"
      "TeacherRolloutDistillation",
    )
    self.assertEqual(
      ppo["actor"]["distribution_cfg"]["class_name"],
      "GaussianDistribution",
    )

  def test_student_is_bounded_and_teacher_remains_original_gaussian(self) -> None:
    distill = asdict(unitree_go2_proprio_safe_action_distillation_runner_cfg())
    ppo = asdict(unitree_go2_proprio_safe_action_ppo_runner_cfg())
    self.assertEqual(
      distill["student"]["distribution_cfg"]["class_name"],
      "src.tasks.velocity.rl.bounded_action_distribution:"
      "AsymmetricBoundedGaussianDistribution",
    )
    self.assertEqual(
      distill["teacher"]["distribution_cfg"]["class_name"],
      "GaussianDistribution",
    )
    self.assertEqual(
      distill["algorithm"]["class_name"],
      "src.tasks.velocity.rl.teacher_rollout_distillation:"
      "BoundedTeacherRolloutDistillation",
    )
    self.assertEqual(
      ppo["actor"]["distribution_cfg"]["class_name"],
      distill["student"]["distribution_cfg"]["class_name"],
    )

  def test_new_bc_maps_teacher_raw_for_environment_and_label(self) -> None:
    distribution = _distribution()
    algorithm = object.__new__(BoundedTeacherRolloutDistillation)
    raw = torch.tensor([[0.5] * 12, [-0.5] * 12])
    algorithm.teacher = lambda obs: raw
    algorithm.student = SimpleNamespace(distribution=distribution)
    algorithm.transition = SimpleNamespace()
    obs = {"teacher": torch.zeros(2, 1)}
    applied = algorithm.act(obs)
    expected = distribution.transform(raw)
    torch.testing.assert_close(applied, expected)
    torch.testing.assert_close(algorithm.last_teacher_raw_actions, raw)
    torch.testing.assert_close(algorithm.transition.actions, expected)
    torch.testing.assert_close(algorithm.transition.privileged_actions, expected)
    self.assertIs(algorithm.transition.observations, obs)

  def test_legacy_bc_class_still_returns_unmodified_teacher_action(self) -> None:
    algorithm = object.__new__(TeacherRolloutDistillation)
    raw = torch.tensor([[0.5] * 12])
    algorithm.teacher = lambda obs: raw
    algorithm.transition = SimpleNamespace()
    obs = {"teacher": torch.zeros(1, 1)}
    actual = algorithm.act(obs)
    torch.testing.assert_close(actual, raw)
    torch.testing.assert_close(algorithm.transition.privileged_actions, raw)


if __name__ == "__main__":
  unittest.main()
