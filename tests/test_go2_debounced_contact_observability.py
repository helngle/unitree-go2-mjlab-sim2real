import unittest

import torch

from src.tasks.velocity.evaluation.debounced_contact_observability import (
  CausalConfirmedContactFilter,
  recompute_causal_confirmed_contact,
)


def _four(values: list[int]) -> torch.Tensor:
  return torch.tensor(values, dtype=torch.bool)[:, None].repeat(1, 4)


class OnlineCausalConfirmedContactTest(unittest.TestCase):
  def test_transition_occurs_only_at_confirmation_tick_without_backdating(self) -> None:
    filter_state = CausalConfirmedContactFilter(1)
    output = []
    changed = []
    for tick, value in enumerate((0, 1, 1)):
      step = filter_state.step(
        torch.full((1, 4), bool(value)),
        attempt_id=torch.tensor([7]),
        episode_start=torch.tensor([tick == 0]),
      )
      output.append(bool(step.contact[0, 0]))
      changed.append(bool(step.changed[0, 0]))
    self.assertEqual(output, [False, False, True])
    self.assertEqual(changed, [False, False, True])

  def test_one_tick_excursion_is_rejected_per_foot(self) -> None:
    raw = torch.tensor(
      [
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
      ],
      dtype=torch.bool,
    )
    result = recompute_causal_confirmed_contact(
      raw,
      attempt_id=torch.zeros(5, dtype=torch.long),
      episode_start=torch.tensor([True, False, False, False, False]),
    )
    self.assertEqual(
      result.contact.to(torch.int).tolist(),
      [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
      ],
    )
    self.assertTrue(result.changed[3, 1])
    self.assertTrue(result.changed[4, 3])
    self.assertEqual(int(result.changed.sum()), 2)

  def test_reset_initializes_baseline_without_cross_episode_transition(self) -> None:
    raw = _four([0, 1, 1])
    result = recompute_causal_confirmed_contact(
      raw,
      attempt_id=torch.tensor([10, 10, 11]),
      episode_start=torch.tensor([True, False, True]),
    )
    self.assertEqual(result.contact[:, 0].tolist(), [False, False, True])
    self.assertFalse(result.changed.any())

  def test_missing_or_implicit_reset_fails_closed(self) -> None:
    filter_state = CausalConfirmedContactFilter(1)
    common = {
      "raw_contact": torch.zeros(1, 4, dtype=torch.bool),
      "attempt_id": torch.tensor([3]),
      "episode_start": torch.tensor([False]),
    }
    with self.assertRaisesRegex(ValueError, "missing episode_start"):
      filter_state.step(**common)

    filter_state.step(
      common["raw_contact"],
      attempt_id=common["attempt_id"],
      episode_start=torch.tensor([True]),
    )
    with self.assertRaisesRegex(ValueError, "changed without episode_start"):
      filter_state.step(
        common["raw_contact"],
        attempt_id=torch.tensor([4]),
        episode_start=torch.tensor([False]),
      )

  def test_invalid_gap_requires_fresh_episode_initialization(self) -> None:
    raw = _four([0, 1, 1])
    with self.assertRaisesRegex(ValueError, "missing episode_start"):
      recompute_causal_confirmed_contact(
        raw,
        attempt_id=torch.zeros(3, dtype=torch.long),
        episode_start=torch.tensor([True, False, False]),
        state_valid=torch.tensor([True, False, True]),
      )


class OfflineOnlineParityTest(unittest.TestCase):
  def test_batched_offline_recompute_equals_online_steps(self) -> None:
    raw = torch.tensor(
      [
        [[0, 0, 1, 1], [1, 1, 0, 0]],
        [[1, 0, 1, 1], [0, 1, 0, 0]],
        [[1, 1, 1, 1], [0, 1, 1, 0]],
        [[1, 1, 0, 1], [0, 1, 1, 1]],
      ],
      dtype=torch.bool,
    )
    attempts = torch.tensor([[1, 8], [1, 8], [1, 8], [1, 9]])
    starts = torch.tensor(
      [[True, True], [False, False], [False, False], [False, True]]
    )
    offline = recompute_causal_confirmed_contact(
      raw, attempt_id=attempts, episode_start=starts
    )

    online_filter = CausalConfirmedContactFilter(2)
    online_contact = []
    online_changed = []
    for tick in range(raw.shape[0]):
      step = online_filter.step(
        raw[tick], attempt_id=attempts[tick], episode_start=starts[tick]
      )
      online_contact.append(step.contact)
      online_changed.append(step.changed)
    self.assertTrue(torch.equal(offline.contact, torch.stack(online_contact)))
    self.assertTrue(torch.equal(offline.changed, torch.stack(online_changed)))

  def test_future_mutation_cannot_change_any_prefix_output(self) -> None:
    raw = _four([0, 1, 1, 0, 0])
    attempt = torch.zeros(5, dtype=torch.long)
    starts = torch.tensor([True, False, False, False, False])
    reference = recompute_causal_confirmed_contact(
      raw, attempt_id=attempt, episode_start=starts
    )
    mutated = raw.clone()
    mutated[3:] = ~mutated[3:]
    alternate = recompute_causal_confirmed_contact(
      mutated, attempt_id=attempt, episode_start=starts
    )
    self.assertTrue(torch.equal(reference.contact[:3], alternate.contact[:3]))
    self.assertTrue(torch.equal(reference.changed[:3], alternate.changed[:3]))

  def test_input_contract_rejects_non_boolean_or_wrong_foot_count(self) -> None:
    with self.assertRaises(ValueError):
      recompute_causal_confirmed_contact(
        torch.zeros(2, 4),
        attempt_id=torch.zeros(2, dtype=torch.long),
        episode_start=torch.tensor([True, False]),
      )
    with self.assertRaises(ValueError):
      recompute_causal_confirmed_contact(
        torch.zeros(2, 3, dtype=torch.bool),
        attempt_id=torch.zeros(2, dtype=torch.long),
        episode_start=torch.tensor([True, False]),
      )


if __name__ == "__main__":
  unittest.main()
