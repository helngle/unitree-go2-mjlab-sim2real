import unittest

import torch

from src.tasks.velocity.evaluation.foot_contact_observability import (
  BootstrapInterval,
  CANONICAL_FOOT_ORDER,
  CoverageThresholds,
  NATIVE_FOOT_GEOMS,
  NATIVE_FOOT_ORDER,
  TrajectoryTimeline,
  assert_no_group_leakage,
  assign_group_folds,
  balanced_binary_weights,
  balanced_log_loss,
  binary_coverage,
  build_future_labels,
  contact_chatter_metrics,
  coverage_pass,
  debounce_binary,
  downsample_anchor_mask,
  fail_closed_observability_gate,
  fit_weighted_logistic,
  fit_weighted_ridge,
  group_balanced_weights,
  paired_cluster_bootstrap,
  reorder_feet,
  validate_pre_action_snapshot,
  validate_native_foot_order,
  weighted_pr_auc,
)


def _timeline(states: int = 9) -> TrajectoryTimeline:
  contact = torch.zeros(states, 4, dtype=torch.bool)
  loaded = torch.zeros_like(contact)
  slip = torch.zeros(states, 4)
  scheduled = torch.zeros_like(contact)
  progress = torch.arange(states, dtype=torch.float64) * 0.01
  state_valid = torch.ones(states, dtype=torch.bool)
  attempt = torch.zeros(states, dtype=torch.long)
  action_valid = torch.ones(states - 1, dtype=torch.bool)
  done = torch.zeros(states - 1, dtype=torch.bool)
  catastrophic = torch.zeros_like(done)
  complete = torch.zeros_like(done)
  return TrajectoryTimeline(
    contact, loaded, slip, scheduled, progress, state_valid, attempt,
    action_valid, done, catastrophic, complete,
  )


class FootOrderContractTest(unittest.TestCase):
  def test_native_order_and_canonical_permutation_are_frozen(self) -> None:
    contract = validate_native_foot_order(NATIVE_FOOT_GEOMS)
    self.assertEqual(contract.runtime_order, NATIVE_FOOT_ORDER)
    self.assertEqual(contract.canonical_order, CANONICAL_FOOT_ORDER)
    self.assertEqual(contract.native_to_canonical, (1, 0, 3, 2))
    native = torch.tensor([[10, 20, 30, 40]])
    self.assertEqual(
      reorder_feet(native, contract.native_to_canonical).tolist(),
      [[20, 10, 40, 30]],
    )

  def test_wrong_runtime_order_fails_closed(self) -> None:
    with self.assertRaises(ValueError):
      validate_native_foot_order(
        ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")
      )

  def test_pre_action_snapshot_matches_native_critic_contact(self) -> None:
    actor = torch.zeros(2, 234)
    critic = torch.zeros(2, 261)
    contact = torch.tensor(
      [[True, False, True, False], [False, True, False, True]]
    )
    critic[:, 245:249] = contact
    validate_pre_action_snapshot(
      actor_observation=actor,
      critic_observation=critic,
      contact_before_observation=contact,
      contact_after_observation=contact.clone(),
      episode_tick_before=torch.tensor([10, 11]),
      episode_tick_after=torch.tensor([10, 11]),
    )
    with self.assertRaises(ValueError):
      validate_pre_action_snapshot(
        actor_observation=actor,
        critic_observation=critic,
        contact_before_observation=contact,
        contact_after_observation=~contact,
        episode_tick_before=torch.tensor([10, 11]),
        episode_tick_after=torch.tensor([10, 11]),
      )


class TimelineContractTest(unittest.TestCase):
  def test_two_tick_debounce_rejects_chatter_and_dates_confirmed_event(self) -> None:
    raw = torch.tensor(
      [[0], [1], [0], [0], [1], [1], [1]], dtype=torch.bool
    ).repeat(1, 4)
    valid = torch.ones(7, dtype=torch.bool)
    attempt = torch.zeros(7, dtype=torch.long)
    state, rise, fall = debounce_binary(
      raw, state_valid=valid, attempt_id=attempt, confirmation_steps=2
    )
    self.assertFalse(rise[1].any())
    self.assertTrue(rise[4].all())
    self.assertTrue(state[4:].all())
    self.assertFalse(fall.any())

  def test_debounce_does_not_cross_attempt_boundary(self) -> None:
    raw = torch.tensor([[0], [1], [1]], dtype=torch.bool).repeat(1, 4)
    valid = torch.ones(3, dtype=torch.bool)
    attempt = torch.tensor([0, 0, 1])
    _, rise, _ = debounce_binary(
      raw, state_valid=valid, attempt_id=attempt, confirmation_steps=2
    )
    self.assertFalse(rise.any())

  def test_future_labels_capture_events_and_censor_terminal_windows(self) -> None:
    base = _timeline(9)
    contact = base.contact.clone()
    loaded = base.loaded.clone()
    slip = base.slip_speed.clone()
    contact[2:, 0] = True
    loaded[2:, 0] = True
    slip[3:, 0] = 0.2
    done = base.done_after.clone()
    catastrophic = base.catastrophic_after.clone()
    done[5] = True
    catastrophic[5] = True
    state_valid = base.state_valid.clone()
    state_valid[7:] = False
    action_valid = base.action_valid.clone()
    action_valid[6:] = False
    timeline = TrajectoryTimeline(
      contact, loaded, slip, base.scheduled_stance, base.progress,
      state_valid, base.attempt_id, action_valid, done, catastrophic,
      base.route_complete_after,
    )
    labels = build_future_labels(
      timeline, horizons=(2,), scenario_speed=0.5,
      boundary_grace_steps=0,
    )[2]
    self.assertTrue(labels.unexpected_transition[0])
    self.assertTrue(labels.unexpected_transition_valid[0])
    self.assertTrue(labels.slip_onset[1])
    self.assertTrue(labels.catastrophic_failure[4])
    self.assertTrue(labels.catastrophic_failure_valid[4])
    self.assertFalse(labels.future_progress_valid[4])
    self.assertFalse(labels.catastrophic_failure_valid[6])

  def test_progress_is_normalized_and_never_crosses_route_completion(self) -> None:
    base = _timeline(7)
    complete = base.route_complete_after.clone()
    complete[3] = True
    action_valid = base.action_valid.clone()
    action_valid[4:] = False
    state_valid = base.state_valid.clone()
    state_valid[5:] = False
    timeline = TrajectoryTimeline(
      base.contact, base.loaded, base.slip_speed, base.scheduled_stance,
      base.progress, state_valid, base.attempt_id, action_valid,
      base.done_after, base.catastrophic_after, complete,
    )
    labels = build_future_labels(
      timeline, horizons=(2,), scenario_speed=0.5, control_dt_s=0.02
    )[2]
    self.assertAlmostEqual(float(labels.future_progress[0]), 1.0)
    self.assertTrue(labels.future_progress_valid[0])
    self.assertFalse(labels.future_progress_valid[2])

  def test_label_builder_rejects_reset_episode_states(self) -> None:
    base = _timeline(7)
    attempt = base.attempt_id.clone()
    attempt[5:] = 1
    timeline = TrajectoryTimeline(
      base.contact, base.loaded, base.slip_speed, base.scheduled_stance,
      base.progress, base.state_valid, attempt, base.action_valid,
      base.done_after, base.catastrophic_after, base.route_complete_after,
    )
    with self.assertRaises(ValueError):
      build_future_labels(timeline, horizons=(2,), scenario_speed=0.5)

  def test_chatter_does_not_use_reset_edges(self) -> None:
    contact = torch.tensor(
      [[0], [1], [0], [1], [1]], dtype=torch.bool
    ).repeat(1, 4)
    valid = torch.ones(5, dtype=torch.bool)
    attempt = torch.tensor([0, 0, 0, 1, 1])
    result = contact_chatter_metrics(
      contact, state_valid=valid, attempt_id=attempt
    )
    self.assertEqual(result["raw_transition_edges"], [2, 2, 2, 2])
    self.assertEqual(result["isolated_excursion_edges"], [2, 2, 2, 2])

  def test_downsample_is_fixed_at_full_rate_indices(self) -> None:
    self.assertEqual(
      torch.where(downsample_anchor_mask(27, every=5, start=10))[0].tolist(),
      [10, 15, 20, 25],
    )


class GroupAndStatisticsContractTest(unittest.TestCase):
  def test_same_seed_slot_never_changes_fold(self) -> None:
    seeds = [42, 42, 42, 43, 43, 44]
    slots = [0, 0, 1, 0, 0, 15]
    folds = assign_group_folds(seeds, slots)
    self.assertEqual(int(folds[0]), int(folds[1]))
    self.assertEqual(int(folds[3]), int(folds[4]))
    self.assertEqual(int(folds[0]), 0)
    self.assertEqual(int(folds[2]), 0)
    self.assertEqual(int(folds[3]), 1)
    self.assertEqual(int(folds[5]), 2)
    for held_out in range(3):
      test = folds == held_out
      train = ~test
      assert_no_group_leakage(seeds, slots, train, test)
    with self.assertRaises(ValueError):
      assign_group_folds([45], [0])

  def test_explicit_group_leakage_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      assert_no_group_leakage(
        [42, 42], [0, 0], torch.tensor([True, False]),
        torch.tensor([False, True]),
      )

  def test_ridge_standardization_is_train_data_only(self) -> None:
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    weights = group_balanced_weights(["a", "a", "b", "b"])
    model = fit_weighted_ridge(x, y, sample_weight=weights, l2=1.0e-6)
    self.assertAlmostEqual(float(model.mean[0]), 1.5)
    self.assertLess(float((model.predict(x) - y).abs().max()), 1.0e-4)

  def test_weighted_logistic_is_deterministic_and_separates_classes(self) -> None:
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y = torch.tensor([False, False, True, True])
    weight = balanced_binary_weights(y, ["a", "b", "c", "d"])
    first = fit_weighted_logistic(x, y, sample_weight=weight, l2=1.0e-4)
    second = fit_weighted_logistic(x, y, sample_weight=weight, l2=1.0e-4)
    probability = first.predict_proba(x)
    self.assertTrue(torch.equal(first.coefficient, second.coefficient))
    self.assertTrue(torch.all(probability[2:] > probability[:2].max()))
    self.assertLess(balanced_log_loss(y, probability), 0.1)
    self.assertEqual(weighted_pr_auc(y, probability), 1.0)

  def test_cluster_bootstrap_pairs_rows_and_is_deterministic(self) -> None:
    baseline = torch.tensor([2.0, 4.0, 8.0, 10.0])
    candidate = torch.tensor([1.0, 3.0, 7.0, 9.0])
    clusters = ["a", "a", "b", "b"]
    first = paired_cluster_bootstrap(
      baseline, candidate, clusters, resamples=500, seed=7
    )
    second = paired_cluster_bootstrap(
      baseline, candidate, clusters, resamples=500, seed=7
    )
    self.assertEqual(first, second)
    self.assertAlmostEqual(first.estimate, 1.0)
    self.assertGreater(first.ci_low, 0.0)
    self.assertEqual(first.cluster_count, 2)

  def test_bootstrap_rejects_cluster_in_multiple_strata(self) -> None:
    with self.assertRaises(ValueError):
      paired_cluster_bootstrap(
        torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]),
        ["same", "same"], strata=["clean", "random"], resamples=10,
      )


class CoverageAndGateContractTest(unittest.TestCase):
  def test_coverage_counts_independent_clusters(self) -> None:
    target = torch.tensor([True, True, False, False])
    valid = torch.ones(4, dtype=torch.bool)
    result = binary_coverage(target, valid, ["a", "a", "b", "c"])
    self.assertEqual(result["clusters"], 3)
    self.assertEqual(result["positive_clusters"], 1)
    self.assertEqual(result["negative_clusters"], 2)

  def test_coverage_fails_closed_on_chatter_or_missing_events(self) -> None:
    thresholds = CoverageThresholds(
      min_clusters=2, min_positive_clusters=1, min_negative_clusters=1,
      min_positive_anchors=1, min_negative_anchors=1,
      min_progress_clusters=2, min_progress_anchors=2,
      min_ray_valid_fraction=0.9,
      max_chatter_fraction=0.1,
    )
    ok, reasons = coverage_pass(
      binary={"failure": {
        "clusters": 2, "positive_clusters": 0, "negative_clusters": 2,
        "positive_anchors": 0, "negative_anchors": 4,
      }},
      progress_clusters=2, progress_valid_anchors=4, ray_valid_fraction=1.0,
      chatter_fraction=0.2, thresholds=thresholds,
    )
    self.assertFalse(ok)
    self.assertTrue(any("positive_clusters" in reason for reason in reasons))
    self.assertIn("contact_chatter_above_maximum", reasons)

  def test_observability_gate_requires_every_stratum_ci(self) -> None:
    positive = BootstrapInterval(0.1, 0.01, 0.2, 0.95, 8, 10_000, 1)
    zero_crossing = BootstrapInterval(0.1, -0.01, 0.2, 0.95, 8, 10_000, 1)
    failed = fail_closed_observability_gate(
      integrity_checks={"foot_order": True, "timing": True},
      coverage_ok=True,
      primary_by_stratum={"clean_vx0.3": positive, "random_vx0.5": zero_crossing},
      secondary_macro={"h10": positive, "h50": positive},
      direction_estimates={"failure": 0.0, "progress": 0.01},
    )
    self.assertFalse(failed["observability_diagnostic_passed"])
    self.assertEqual(failed["decision"], "INCONCLUSIVE_DO_NOT_TRAIN")
    passed = fail_closed_observability_gate(
      integrity_checks={"foot_order": True, "timing": True},
      coverage_ok=True,
      primary_by_stratum={"clean_vx0.3": positive, "random_vx0.5": positive},
      secondary_macro={"h10": positive, "h50": positive},
      direction_estimates={"failure": 0.0, "progress": 0.01},
    )
    self.assertTrue(passed["observability_diagnostic_passed"])

  def test_clearance_dominance_rejects_contact(self) -> None:
    positive = BootstrapInterval(0.1, 0.01, 0.2, 0.95, 8, 10_000, 1)
    dominated = BootstrapInterval(-0.2, -0.3, -0.1, 0.95, 8, 10_000, 1)
    result = fail_closed_observability_gate(
      integrity_checks={"all": True}, coverage_ok=True,
      primary_by_stratum={"all": positive}, secondary_macro={"h10": positive},
      direction_estimates={"all": 0.1}, contact_minus_clearance=dominated,
    )
    self.assertFalse(result["observability_diagnostic_passed"])
    self.assertIn("clearance_significantly_dominates_contact", result["reasons"])


if __name__ == "__main__":
  unittest.main()
