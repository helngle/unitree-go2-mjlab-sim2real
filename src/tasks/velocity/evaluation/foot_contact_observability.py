"""Pure contracts for the Go2 foot-contact observability diagnostic.

This module is evaluation-only.  It deliberately contains no environment,
policy, training, or GPU code.  The collector is expected to provide one
continuous, pre-action/post-action timeline per original route attempt.  A
post-action terminal state must be captured before automatic reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


NATIVE_FOOT_ORDER = ("FL", "FR", "RL", "RR")
CANONICAL_FOOT_ORDER = ("FR", "FL", "RR", "RL")
NATIVE_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in NATIVE_FOOT_ORDER)
CANONICAL_FOOT_GEOMS = tuple(
  f"{name}_foot_collision" for name in CANONICAL_FOOT_ORDER
)
NATIVE_TO_CANONICAL = (1, 0, 3, 2)
CANONICAL_TO_NATIVE = (1, 0, 3, 2)

DEFAULT_HORIZONS = (10, 25, 50)
DEFAULT_DOWNSAMPLE = 5
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_809
FORMAL_SEEDS = (42, 43, 44)
DEFAULT_RIDGE_L2_GRID = (
  1.0e-4,
  1.0e-3,
  1.0e-2,
  1.0e-1,
  1.0,
  10.0,
)
DEFAULT_LOGISTIC_L2_GRID = DEFAULT_RIDGE_L2_GRID


@dataclass(frozen=True)
class FootOrderContract:
  runtime_order: tuple[str, ...]
  canonical_order: tuple[str, ...]
  native_to_canonical: tuple[int, ...]
  canonical_to_native: tuple[int, ...]


def validate_native_foot_order(primary_names: Sequence[str]) -> FootOrderContract:
  """Fail unless the runtime slots have the frozen FL/FR/RL/RR order."""
  names = tuple(primary_names)
  if names != NATIVE_FOOT_GEOMS:
    raise ValueError(
      "feet_ground_contact runtime order must be exactly "
      f"{NATIVE_FOOT_GEOMS}, got {names}"
    )
  native_to_canonical = tuple(names.index(name) for name in CANONICAL_FOOT_GEOMS)
  canonical_to_native = tuple(
    CANONICAL_FOOT_GEOMS.index(name) for name in names
  )
  if native_to_canonical != NATIVE_TO_CANONICAL:
    raise RuntimeError("unexpected native-to-canonical foot permutation")
  return FootOrderContract(
    runtime_order=NATIVE_FOOT_ORDER,
    canonical_order=CANONICAL_FOOT_ORDER,
    native_to_canonical=native_to_canonical,
    canonical_to_native=canonical_to_native,
  )


def reorder_feet(
  values: torch.Tensor, permutation: Sequence[int], *, dim: int = -1
) -> torch.Tensor:
  """Reorder a four-foot tensor without changing its values."""
  if values.ndim == 0:
    raise ValueError("foot tensor must have at least one dimension")
  dim = dim % values.ndim
  if values.shape[dim] != 4:
    raise ValueError("selected foot dimension must have length four")
  perm = tuple(int(index) for index in permutation)
  if sorted(perm) != [0, 1, 2, 3]:
    raise ValueError("foot permutation must contain 0,1,2,3 exactly once")
  ids = torch.tensor(perm, dtype=torch.long, device=values.device)
  return values.index_select(dim, ids)


def validate_pre_action_snapshot(
  *,
  actor_observation: torch.Tensor,
  critic_observation: torch.Tensor,
  contact_before_observation: torch.Tensor,
  contact_after_observation: torch.Tensor,
  episode_tick_before: torch.Tensor,
  episode_tick_after: torch.Tensor,
  critic_contact_slice: tuple[int, int] = (245, 249),
) -> None:
  """Verify that obs234 and native contact4 describe one pre-action state."""
  if actor_observation.ndim != 2 or actor_observation.shape[1] != 234:
    raise ValueError("actor observation must have shape (envs, 234)")
  if critic_observation.ndim != 2 or critic_observation.shape[1] != 261:
    raise ValueError("critic observation must have shape (envs, 261)")
  envs = actor_observation.shape[0]
  if critic_observation.shape[0] != envs:
    raise ValueError("actor and critic batch sizes differ")
  for name, value in (
    ("contact_before_observation", contact_before_observation),
    ("contact_after_observation", contact_after_observation),
  ):
    if value.shape != (envs, 4) or value.dtype != torch.bool:
      raise ValueError(f"{name} must be native-order bool shape (envs, 4)")
  if episode_tick_before.shape != (envs,) or episode_tick_after.shape != (envs,):
    raise ValueError("episode ticks must have shape (envs,)")
  if not torch.isfinite(actor_observation).all() or not torch.isfinite(
    critic_observation
  ).all():
    raise ValueError("pre-action observations must be finite")
  if not torch.equal(contact_before_observation, contact_after_observation):
    raise ValueError("contact changed while computing the pre-action observation")
  if not torch.equal(episode_tick_before, episode_tick_after):
    raise ValueError("episode tick changed before policy action")
  start, end = critic_contact_slice
  if (start, end) != (245, 249):
    raise ValueError("critic foot_contact slice must remain [245:249]")
  critic_contact = critic_observation[:, start:end]
  if not torch.equal(critic_contact, contact_after_observation.to(critic_contact.dtype)):
    raise ValueError("critic foot_contact does not equal the native sensor bits")


def _validate_timeline_shapes(
  contact: torch.Tensor,
  state_valid: torch.Tensor,
  attempt_id: torch.Tensor,
) -> tuple[int, int]:
  if contact.ndim != 2 or contact.shape[1] != 4 or contact.dtype != torch.bool:
    raise ValueError("contact must be bool with shape (states, 4)")
  states = contact.shape[0]
  if state_valid.shape != (states,) or state_valid.dtype != torch.bool:
    raise ValueError("state_valid must be bool with shape (states,)")
  if attempt_id.shape != (states,) or attempt_id.dtype == torch.bool:
    raise ValueError("attempt_id must have shape (states,) and integer values")
  if attempt_id.dtype.is_floating_point:
    raise ValueError("attempt_id must be an integer tensor")
  return states, contact.shape[1]


def debounce_binary(
  signal: torch.Tensor,
  *,
  state_valid: torch.Tensor,
  attempt_id: torch.Tensor,
  confirmation_steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Debounce a per-foot bool timeline without crossing invalid/reset states.

  The returned event is assigned to the first tick of the candidate run after
  enough future ticks confirm it.  ``rise`` and ``fall`` therefore retain the
  physical transition time while never using an unconfirmed one-tick pulse.
  """
  states, feet = _validate_timeline_shapes(signal, state_valid, attempt_id)
  if confirmation_steps < 1:
    raise ValueError("confirmation_steps must be positive")
  debounced = torch.zeros_like(signal)
  rise = torch.zeros_like(signal)
  fall = torch.zeros_like(signal)
  for foot in range(feet):
    current: bool | None = None
    candidate: bool | None = None
    candidate_start = -1
    candidate_count = 0
    previous_attempt: int | None = None
    for step in range(states):
      if not bool(state_valid[step]):
        current = None
        candidate = None
        candidate_count = 0
        previous_attempt = None
        continue
      this_attempt = int(attempt_id[step])
      value = bool(signal[step, foot])
      if current is None or previous_attempt != this_attempt:
        current = value
        candidate = None
        candidate_count = 0
        debounced[step, foot] = current
        previous_attempt = this_attempt
        continue
      previous_attempt = this_attempt
      if value == current:
        candidate = None
        candidate_count = 0
      elif candidate is value:
        candidate_count += 1
      else:
        candidate = value
        candidate_start = step
        candidate_count = 1
      if candidate is not None and candidate_count >= confirmation_steps:
        old = current
        current = candidate
        debounced[candidate_start : step + 1, foot] = current
        (rise if current and not old else fall)[candidate_start, foot] = True
        candidate = None
        candidate_count = 0
      debounced[step, foot] = current
  return debounced, rise, fall


def hysteretic_slip_onset(
  slip_speed: torch.Tensor,
  loaded: torch.Tensor,
  *,
  state_valid: torch.Tensor,
  attempt_id: torch.Tensor,
  on_threshold: float = 0.10,
  off_threshold: float = 0.05,
  confirmation_steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return hysteretic slip state and confirmed per-foot onset events."""
  if slip_speed.shape != loaded.shape or loaded.dtype != torch.bool:
    raise ValueError("slip_speed and loaded must share shape; loaded must be bool")
  if slip_speed.ndim != 2 or slip_speed.shape[1] != 4:
    raise ValueError("slip tensors must have shape (states, 4)")
  if not torch.isfinite(slip_speed[state_valid]).all():
    raise ValueError("valid slip speeds must be finite")
  if not (math.isfinite(on_threshold) and math.isfinite(off_threshold)):
    raise ValueError("slip thresholds must be finite")
  if off_threshold < 0.0 or on_threshold <= off_threshold:
    raise ValueError("slip thresholds must satisfy 0 <= off < on")

  raw_state = torch.zeros_like(loaded)
  active = torch.zeros(4, dtype=torch.bool, device=loaded.device)
  previous_attempt: int | None = None
  for step in range(slip_speed.shape[0]):
    if not bool(state_valid[step]):
      active.zero_()
      previous_attempt = None
      continue
    this_attempt = int(attempt_id[step])
    if previous_attempt != this_attempt:
      active.zero_()
    previous_attempt = this_attempt
    active = torch.where(
      ~loaded[step],
      torch.zeros_like(active),
      torch.where(
        active,
        slip_speed[step] > off_threshold,
        slip_speed[step] >= on_threshold,
      ),
    )
    raw_state[step] = active
  debounced, onset, _ = debounce_binary(
    raw_state,
    state_valid=state_valid,
    attempt_id=attempt_id,
    confirmation_steps=confirmation_steps,
  )
  return debounced, onset


def unexpected_contact_events(
  contact: torch.Tensor,
  scheduled_stance: torch.Tensor,
  *,
  state_valid: torch.Tensor,
  attempt_id: torch.Tensor,
  confirmation_steps: int = 2,
  boundary_grace_steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return unexpected transition, touchdown, and liftoff events.

  A touchdown in scheduled swing or liftoff in scheduled stance is unexpected.
  Events within ``boundary_grace_steps`` of a scheduled gait transition are
  excluded.  Both tensors must already use the same (native or canonical) order.
  """
  if scheduled_stance.shape != contact.shape or scheduled_stance.dtype != torch.bool:
    raise ValueError("scheduled_stance must be bool and match contact")
  if boundary_grace_steps < 0:
    raise ValueError("boundary_grace_steps must be non-negative")
  _, touchdown, liftoff = debounce_binary(
    contact,
    state_valid=state_valid,
    attempt_id=attempt_id,
    confirmation_steps=confirmation_steps,
  )
  boundary = torch.zeros_like(scheduled_stance)
  changes = torch.zeros_like(scheduled_stance)
  same_attempt = attempt_id[1:] == attempt_id[:-1]
  changes[1:] = (
    scheduled_stance[1:] != scheduled_stance[:-1]
  ) & state_valid[1:, None] & state_valid[:-1, None] & same_attempt[:, None]
  for offset in range(-boundary_grace_steps, boundary_grace_steps + 1):
    if offset < 0:
      boundary[:offset] |= changes[-offset:]
    elif offset > 0:
      boundary[offset:] |= changes[:-offset]
    else:
      boundary |= changes
  unexpected_touchdown = touchdown & ~scheduled_stance & ~boundary
  unexpected_liftoff = liftoff & scheduled_stance & ~boundary
  return (
    unexpected_touchdown | unexpected_liftoff,
    unexpected_touchdown,
    unexpected_liftoff,
  )


@dataclass(frozen=True)
class TrajectoryTimeline:
  """One original route attempt represented by states and intervening actions."""

  contact: torch.Tensor
  loaded: torch.Tensor
  slip_speed: torch.Tensor
  scheduled_stance: torch.Tensor
  progress: torch.Tensor
  state_valid: torch.Tensor
  attempt_id: torch.Tensor
  action_valid: torch.Tensor
  done_after: torch.Tensor
  catastrophic_after: torch.Tensor
  route_complete_after: torch.Tensor


@dataclass(frozen=True)
class HorizonLabels:
  slip_onset: torch.Tensor
  slip_onset_valid: torch.Tensor
  unexpected_transition: torch.Tensor
  unexpected_transition_valid: torch.Tensor
  catastrophic_failure: torch.Tensor
  catastrophic_failure_valid: torch.Tensor
  future_progress: torch.Tensor
  future_progress_valid: torch.Tensor


def _validate_trajectory(timeline: TrajectoryTimeline) -> int:
  states, _ = _validate_timeline_shapes(
    timeline.contact, timeline.state_valid, timeline.attempt_id
  )
  actions = states - 1
  for name in ("loaded", "scheduled_stance"):
    value = getattr(timeline, name)
    if value.shape != timeline.contact.shape or value.dtype != torch.bool:
      raise ValueError(f"{name} must be bool and match contact")
  if timeline.slip_speed.shape != timeline.contact.shape:
    raise ValueError("slip_speed must match contact")
  if timeline.progress.shape != (states,):
    raise ValueError("progress must have one value per state")
  if not torch.isfinite(timeline.progress[timeline.state_valid]).all():
    raise ValueError("valid progress must be finite")
  for name in (
    "action_valid",
    "done_after",
    "catastrophic_after",
    "route_complete_after",
  ):
    value = getattr(timeline, name)
    if value.shape != (actions,) or value.dtype != torch.bool:
      raise ValueError(f"{name} must be bool with shape (states-1,)")
  if torch.any(timeline.catastrophic_after & ~timeline.done_after):
    raise ValueError("catastrophic action must also be done")
  if torch.any(timeline.route_complete_after & timeline.catastrophic_after):
    raise ValueError("route completion and catastrophic failure cannot coincide")
  valid_attempts = torch.unique(timeline.attempt_id[timeline.state_valid])
  if valid_attempts.numel() > 1:
    raise ValueError("one trajectory artifact must contain only one original attempt")
  endings = torch.where(timeline.done_after | timeline.route_complete_after)[0]
  if endings.numel() > 1:
    raise ValueError("one route attempt may have at most one terminal action")
  if endings.numel() == 1:
    terminal_action = int(endings[0])
    if timeline.action_valid[terminal_action + 1 :].any():
      raise ValueError("actions after the terminal action must be frozen")
    if timeline.state_valid[terminal_action + 2 :].any():
      raise ValueError("automatic-reset states must not enter the trajectory")
  return actions


def build_future_labels(
  timeline: TrajectoryTimeline,
  *,
  horizons: Sequence[int] = DEFAULT_HORIZONS,
  control_dt_s: float = 0.02,
  scenario_speed: float,
  confirmation_steps: int = 2,
  boundary_grace_steps: int = 2,
) -> dict[int, HorizonLabels]:
  """Build survival-aware labels without crossing a reset or attempt boundary."""
  actions = _validate_trajectory(timeline)
  horizons = tuple(int(value) for value in horizons)
  if not horizons or any(value <= 0 for value in horizons):
    raise ValueError("horizons must be nonempty positive integers")
  if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
    raise ValueError("control_dt_s must be finite and positive")
  if not math.isfinite(scenario_speed) or scenario_speed <= 0.0:
    raise ValueError("scenario_speed must be finite and positive")

  _, slip_event = hysteretic_slip_onset(
    timeline.slip_speed,
    timeline.loaded,
    state_valid=timeline.state_valid,
    attempt_id=timeline.attempt_id,
    confirmation_steps=confirmation_steps,
  )
  unexpected, _, _ = unexpected_contact_events(
    timeline.contact,
    timeline.scheduled_stance,
    state_valid=timeline.state_valid,
    attempt_id=timeline.attempt_id,
    confirmation_steps=confirmation_steps,
    boundary_grace_steps=boundary_grace_steps,
  )
  slip_any = slip_event.any(dim=1)
  unexpected_any = unexpected.any(dim=1)
  output: dict[int, HorizonLabels] = {}
  for horizon in horizons:
    slip_target = torch.zeros(actions, dtype=torch.bool)
    slip_valid = torch.zeros(actions, dtype=torch.bool)
    unexpected_target = torch.zeros(actions, dtype=torch.bool)
    unexpected_valid = torch.zeros(actions, dtype=torch.bool)
    failure_target = torch.zeros(actions, dtype=torch.bool)
    failure_valid = torch.zeros(actions, dtype=torch.bool)
    progress_target = torch.zeros(actions, dtype=torch.float64)
    progress_valid = torch.zeros(actions, dtype=torch.bool)
    for anchor in range(actions):
      if not bool(timeline.state_valid[anchor]):
        continue
      stop = min(anchor + horizon, actions)
      event_end = stop + 1
      same_attempt = bool(
        timeline.state_valid[anchor:event_end].all()
        and (timeline.attempt_id[anchor:event_end] == timeline.attempt_id[anchor]).all()
      )
      valid_actions = bool(timeline.action_valid[anchor:stop].all())
      if stop <= anchor or not valid_actions:
        continue
      slip_seen = bool(slip_any[anchor + 1 : event_end].any())
      unexpected_seen = bool(unexpected_any[anchor + 1 : event_end].any())
      failure_seen = bool(timeline.catastrophic_after[anchor:stop].any())
      ended = bool(
        timeline.done_after[anchor:stop].any()
        or timeline.route_complete_after[anchor:stop].any()
      )
      full_horizon = stop == anchor + horizon and same_attempt and not ended

      if slip_seen:
        slip_target[anchor] = True
        slip_valid[anchor] = True
      elif full_horizon:
        slip_valid[anchor] = True
      if unexpected_seen:
        unexpected_target[anchor] = True
        unexpected_valid[anchor] = True
      elif full_horizon:
        unexpected_valid[anchor] = True
      if failure_seen:
        failure_target[anchor] = True
        failure_valid[anchor] = True
      elif full_horizon:
        failure_valid[anchor] = True
      if full_horizon:
        denominator = scenario_speed * horizon * control_dt_s
        progress_target[anchor] = float(
          (timeline.progress[anchor + horizon] - timeline.progress[anchor])
          / denominator
        )
        progress_valid[anchor] = True
    output[horizon] = HorizonLabels(
      slip_onset=slip_target,
      slip_onset_valid=slip_valid,
      unexpected_transition=unexpected_target,
      unexpected_transition_valid=unexpected_valid,
      catastrophic_failure=failure_target,
      catastrophic_failure_valid=failure_valid,
      future_progress=progress_target,
      future_progress_valid=progress_valid,
    )
  return output


def downsample_anchor_mask(
  length: int, *, every: int = DEFAULT_DOWNSAMPLE, start: int = 10
) -> torch.Tensor:
  if length < 0 or every <= 0 or start < 0:
    raise ValueError("length, every, and start must be non-negative; every > 0")
  mask = torch.zeros(length, dtype=torch.bool)
  if start < length:
    mask[start::every] = True
  return mask


def contact_chatter_metrics(
  contact: torch.Tensor,
  *,
  state_valid: torch.Tensor,
  attempt_id: torch.Tensor,
) -> dict[str, object]:
  """Measure raw toggles and isolated 010/101 one-tick excursions."""
  states, feet = _validate_timeline_shapes(contact, state_valid, attempt_id)
  toggles = torch.zeros(feet, dtype=torch.long)
  isolated = torch.zeros(feet, dtype=torch.long)
  for step in range(1, states):
    same = bool(
      state_valid[step]
      and state_valid[step - 1]
      and attempt_id[step] == attempt_id[step - 1]
    )
    if same:
      toggles += (contact[step] != contact[step - 1]).long()
  for step in range(1, states - 1):
    same = bool(
      state_valid[step - 1 : step + 2].all()
      and (attempt_id[step - 1 : step + 2] == attempt_id[step]).all()
    )
    if same:
      pulse = (contact[step - 1] == contact[step + 1]) & (
        contact[step] != contact[step - 1]
      )
      # Each isolated excursion contributes two raw transition edges.
      isolated += 2 * pulse.long()
  rates = torch.where(
    toggles > 0,
    isolated.to(torch.float64) / toggles,
    torch.zeros(feet, dtype=torch.float64),
  )
  return {
    "foot_order": list(NATIVE_FOOT_ORDER),
    "raw_transition_edges": toggles.tolist(),
    "isolated_excursion_edges": isolated.tolist(),
    "isolated_excursion_fraction": rates.tolist(),
    "max_isolated_excursion_fraction": float(rates.max()) if feet else 0.0,
  }


def stable_group_fold(seed: int, matched_slot: int, *, num_folds: int = 3) -> int:
  """Assign a formal group to the exact leave-one-seed-out fold."""
  if isinstance(seed, bool) or isinstance(matched_slot, bool):
    raise TypeError("seed and matched_slot must be integers")
  if not isinstance(seed, int) or not isinstance(matched_slot, int):
    raise TypeError("seed and matched_slot must be integers")
  if matched_slot < 0:
    raise ValueError("matched_slot must be non-negative")
  if num_folds != len(FORMAL_SEEDS):
    raise ValueError("formal diagnostic uses exactly three leave-one-seed-out folds")
  if seed not in FORMAL_SEEDS:
    raise ValueError(f"formal diagnostic seed must be one of {FORMAL_SEEDS}")
  return FORMAL_SEEDS.index(seed)


def assign_group_folds(
  seeds: Sequence[int], matched_slots: Sequence[int], *, num_folds: int = 3
) -> torch.Tensor:
  if len(seeds) != len(matched_slots):
    raise ValueError("seeds and matched_slots must have equal length")
  group_fold: dict[tuple[int, int], int] = {}
  values: list[int] = []
  for seed, slot in zip(seeds, matched_slots, strict=True):
    key = (int(seed), int(slot))
    fold = stable_group_fold(*key, num_folds=num_folds)
    previous = group_fold.setdefault(key, fold)
    if previous != fold:
      raise RuntimeError("one group was assigned to multiple folds")
    values.append(fold)
  return torch.tensor(values, dtype=torch.long)


def assert_no_group_leakage(
  seeds: Sequence[int],
  matched_slots: Sequence[int],
  train_mask: torch.Tensor,
  test_mask: torch.Tensor,
) -> None:
  if train_mask.dtype != torch.bool or test_mask.dtype != torch.bool:
    raise ValueError("train/test masks must be bool")
  if train_mask.shape != test_mask.shape or train_mask.numel() != len(seeds):
    raise ValueError("train/test masks must align with group arrays")
  if torch.any(train_mask & test_mask):
    raise ValueError("train and test rows overlap")
  train = {
    (int(seeds[i]), int(matched_slots[i]))
    for i in torch.where(train_mask)[0].tolist()
  }
  test = {
    (int(seeds[i]), int(matched_slots[i]))
    for i in torch.where(test_mask)[0].tolist()
  }
  leaked = train & test
  if leaked:
    raise ValueError(f"(seed, matched_slot) group leakage: {sorted(leaked)}")


@dataclass(frozen=True)
class RidgeModel:
  mean: torch.Tensor
  scale: torch.Tensor
  coefficient: torch.Tensor
  intercept: torch.Tensor
  l2: float

  def predict(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != self.mean.numel():
      raise ValueError("ridge features have the wrong shape")
    normalized = (features.to(torch.float64) - self.mean) / self.scale
    return normalized @ self.coefficient + self.intercept


def group_balanced_weights(group_ids: Sequence[object]) -> torch.Tensor:
  if not group_ids:
    raise ValueError("group_ids must be nonempty")
  counts: dict[object, int] = {}
  for group in group_ids:
    counts[group] = counts.get(group, 0) + 1
  weight = torch.tensor(
    [1.0 / counts[group] for group in group_ids], dtype=torch.float64
  )
  return weight / weight.sum()


def fit_weighted_ridge(
  features: torch.Tensor,
  target: torch.Tensor,
  *,
  sample_weight: torch.Tensor,
  l2: float,
) -> RidgeModel:
  """Fit a deterministic train-fold-only standardized ridge model."""
  if features.ndim != 2 or target.shape != (features.shape[0],):
    raise ValueError("features must be (rows, dims) and target must be (rows,)")
  if sample_weight.shape != target.shape:
    raise ValueError("sample_weight must match target")
  if not math.isfinite(l2) or l2 <= 0.0:
    raise ValueError("l2 must be finite and positive")
  x = features.to(torch.float64)
  y = target.to(torch.float64)
  w = sample_weight.to(torch.float64)
  if not torch.isfinite(x).all() or not torch.isfinite(y).all():
    raise ValueError("ridge inputs must be finite")
  if not torch.isfinite(w).all() or torch.any(w < 0.0) or float(w.sum()) <= 0.0:
    raise ValueError("ridge weights must be finite, non-negative, and nonempty")
  w = w / w.sum()
  mean = (x * w[:, None]).sum(dim=0)
  variance = ((x - mean).square() * w[:, None]).sum(dim=0)
  scale = variance.sqrt().clamp_min(1.0e-12)
  xn = (x - mean) / scale
  y_mean = (y * w).sum()
  yc = y - y_mean
  gram = xn.T @ (xn * w[:, None])
  rhs = xn.T @ (yc * w)
  coefficient = torch.linalg.solve(
    gram + l2 * torch.eye(x.shape[1], dtype=torch.float64), rhs
  )
  return RidgeModel(mean, scale, coefficient, y_mean, float(l2))


@dataclass(frozen=True)
class LogisticModel:
  mean: torch.Tensor
  scale: torch.Tensor
  coefficient: torch.Tensor
  intercept: torch.Tensor
  l2: float
  iterations: int

  def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != self.mean.numel():
      raise ValueError("logistic features have the wrong shape")
    normalized = (features.to(torch.float64) - self.mean) / self.scale
    return normalized @ self.coefficient + self.intercept

  def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.predict_logits(features))


def balanced_binary_weights(
  target: torch.Tensor, group_ids: Sequence[object]
) -> torch.Tensor:
  """Combine equal-group and equal-class weighting for a binary endpoint."""
  if target.ndim != 1 or target.dtype != torch.bool:
    raise ValueError("binary target must be a bool vector")
  if target.numel() != len(group_ids):
    raise ValueError("group_ids must align with target")
  if not target.any() or target.all():
    raise ValueError("balanced binary fitting requires both classes")
  weight = group_balanced_weights(group_ids)
  positive = target
  negative = ~target
  weight[positive] *= 0.5 / weight[positive].sum()
  weight[negative] *= 0.5 / weight[negative].sum()
  return weight / weight.sum()


def _logistic_objective(
  design: torch.Tensor,
  target: torch.Tensor,
  weight: torch.Tensor,
  parameter: torch.Tensor,
  l2: float,
) -> torch.Tensor:
  logits = design @ parameter
  data = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, target, weight=weight, reduction="sum"
  )
  return data + 0.5 * l2 * parameter[1:].square().sum()


def fit_weighted_logistic(
  features: torch.Tensor,
  target: torch.Tensor,
  *,
  sample_weight: torch.Tensor,
  l2: float,
  max_iterations: int = 100,
  tolerance: float = 1.0e-8,
) -> LogisticModel:
  """Fit deterministic L2 logistic regression with damped Newton updates."""
  if features.ndim != 2 or target.shape != (features.shape[0],):
    raise ValueError("features must be (rows, dims) and target must be (rows,)")
  if target.dtype != torch.bool:
    raise ValueError("logistic target must be bool")
  if sample_weight.shape != target.shape:
    raise ValueError("sample_weight must match target")
  if not math.isfinite(l2) or l2 <= 0.0:
    raise ValueError("l2 must be finite and positive")
  if max_iterations <= 0 or tolerance <= 0.0:
    raise ValueError("logistic iteration controls must be positive")
  if not target.any() or target.all():
    raise ValueError("logistic fitting requires both classes")
  x = features.to(torch.float64)
  y = target.to(torch.float64)
  w = sample_weight.to(torch.float64)
  if not torch.isfinite(x).all() or not torch.isfinite(w).all():
    raise ValueError("logistic inputs and weights must be finite")
  if torch.any(w < 0.0) or float(w.sum()) <= 0.0:
    raise ValueError("logistic weights must be non-negative and nonempty")
  w = w / w.sum()
  mean = (x * w[:, None]).sum(dim=0)
  variance = ((x - mean).square() * w[:, None]).sum(dim=0)
  scale = variance.sqrt().clamp_min(1.0e-12)
  normalized = (x - mean) / scale
  design = torch.cat(
    (torch.ones(x.shape[0], 1, dtype=torch.float64), normalized), dim=1
  )
  positive_rate = (y * w).sum().clamp(1.0e-8, 1.0 - 1.0e-8)
  parameter = torch.zeros(design.shape[1], dtype=torch.float64)
  parameter[0] = torch.logit(positive_rate)
  penalty = torch.eye(design.shape[1], dtype=torch.float64) * l2
  penalty[0, 0] = 0.0
  for iteration in range(1, max_iterations + 1):
    probability = torch.sigmoid(design @ parameter)
    gradient = design.T @ ((probability - y) * w) + penalty @ parameter
    curvature = w * probability * (1.0 - probability)
    hessian = design.T @ (design * curvature[:, None]) + penalty
    step = torch.linalg.solve(hessian, gradient)
    current = _logistic_objective(design, y, w, parameter, l2)
    step_scale = 1.0
    accepted = False
    for _ in range(30):
      proposal = parameter - step_scale * step
      proposed = _logistic_objective(design, y, w, proposal, l2)
      if bool(proposed <= current):
        parameter = proposal
        accepted = True
        break
      step_scale *= 0.5
    if not accepted:
      raise RuntimeError("logistic line search failed to reduce the objective")
    if float((step_scale * step).abs().max()) <= tolerance:
      return LogisticModel(
        mean=mean,
        scale=scale,
        coefficient=parameter[1:],
        intercept=parameter[0],
        l2=float(l2),
        iterations=iteration,
      )
  raise RuntimeError("logistic regression did not converge")


def balanced_log_loss(
  target: torch.Tensor,
  probability: torch.Tensor,
  *,
  sample_weight: torch.Tensor | None = None,
) -> float:
  """Return the equally weighted positive/negative binary log loss."""
  if target.ndim != 1 or target.dtype != torch.bool or probability.shape != target.shape:
    raise ValueError("target must be bool and probability must have matching shape")
  if not target.any() or target.all():
    raise ValueError("balanced log loss requires both classes")
  if not torch.isfinite(probability).all() or torch.any(
    (probability < 0.0) | (probability > 1.0)
  ):
    raise ValueError("probabilities must be finite and in [0,1]")
  weight = (
    torch.ones(target.numel(), dtype=torch.float64)
    if sample_weight is None
    else sample_weight.to(torch.float64)
  )
  if weight.shape != target.shape or torch.any(weight < 0.0):
    raise ValueError("sample_weight must be non-negative and match target")
  probability = probability.to(torch.float64).clamp(1.0e-12, 1.0 - 1.0e-12)
  loss = -(target * torch.log(probability) + (~target) * torch.log1p(-probability))
  positive = (loss[target] * weight[target]).sum() / weight[target].sum()
  negative = (loss[~target] * weight[~target]).sum() / weight[~target].sum()
  return float(0.5 * (positive + negative))


def weighted_pr_auc(
  target: torch.Tensor,
  probability: torch.Tensor,
  *,
  sample_weight: torch.Tensor | None = None,
) -> float:
  """Return deterministic weighted average precision (stepwise PR area)."""
  if target.ndim != 1 or target.dtype != torch.bool or probability.shape != target.shape:
    raise ValueError("target must be bool and probability must have matching shape")
  if not target.any():
    raise ValueError("PR-AUC requires at least one positive")
  if not torch.isfinite(probability).all():
    raise ValueError("probabilities must be finite")
  weight = (
    torch.ones(target.numel(), dtype=torch.float64)
    if sample_weight is None
    else sample_weight.to(torch.float64)
  )
  if weight.shape != target.shape or torch.any(weight < 0.0):
    raise ValueError("sample_weight must be non-negative and match target")
  order = torch.argsort(probability, descending=True, stable=True)
  ordered_target = target[order]
  ordered_weight = weight[order]
  true_positive = torch.cumsum(ordered_weight * ordered_target, dim=0)
  predicted_positive = torch.cumsum(ordered_weight, dim=0)
  precision = true_positive / predicted_positive.clamp_min(1.0e-12)
  positive_increment = ordered_weight * ordered_target
  return float(
    (precision * positive_increment).sum() / positive_increment.sum()
  )


@dataclass(frozen=True)
class BootstrapInterval:
  estimate: float
  ci_low: float
  ci_high: float
  confidence: float
  cluster_count: int
  resamples: int
  seed: int


def paired_cluster_bootstrap(
  baseline_error: torch.Tensor,
  candidate_error: torch.Tensor,
  cluster_ids: Sequence[object],
  *,
  strata: Sequence[object] | None = None,
  resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
  seed: int = DEFAULT_BOOTSTRAP_SEED,
  confidence: float = 0.95,
) -> BootstrapInterval:
  """Bootstrap paired error reduction by cluster, never by individual step."""
  if baseline_error.shape != candidate_error.shape or baseline_error.ndim != 1:
    raise ValueError("paired errors must be one-dimensional with equal shape")
  if baseline_error.numel() != len(cluster_ids):
    raise ValueError("cluster_ids must align with errors")
  if strata is not None and len(strata) != len(cluster_ids):
    raise ValueError("strata must align with errors")
  if resamples <= 0 or not (0.0 < confidence < 1.0):
    raise ValueError("resamples and confidence are invalid")
  difference = baseline_error.to(torch.float64) - candidate_error.to(torch.float64)
  if not torch.isfinite(difference).all():
    raise ValueError("bootstrap errors must be finite")
  grouped: dict[tuple[object, object], list[float]] = {}
  stratum_values = strata if strata is not None else ("all",) * len(cluster_ids)
  cluster_to_stratum: dict[object, object] = {}
  for value, cluster, stratum in zip(
    difference.tolist(), cluster_ids, stratum_values, strict=True
  ):
    old = cluster_to_stratum.setdefault(cluster, stratum)
    if old != stratum:
      raise ValueError("one bootstrap cluster appears in multiple strata")
    grouped.setdefault((stratum, cluster), []).append(float(value))
  by_stratum: dict[object, list[float]] = {}
  for (stratum, _), values in grouped.items():
    by_stratum.setdefault(stratum, []).append(sum(values) / len(values))
  if not by_stratum:
    raise ValueError("bootstrap requires at least one cluster")
  estimate = sum(
    sum(values) / len(values) for values in by_stratum.values()
  ) / len(by_stratum)
  generator = torch.Generator(device="cpu").manual_seed(seed)
  draws = torch.empty(resamples, dtype=torch.float64)
  for draw in range(resamples):
    stratum_means = []
    for values in by_stratum.values():
      tensor = torch.tensor(values, dtype=torch.float64)
      ids = torch.randint(
        tensor.numel(), (tensor.numel(),), generator=generator
      )
      stratum_means.append(float(tensor[ids].mean()))
    draws[draw] = sum(stratum_means) / len(stratum_means)
  alpha = (1.0 - confidence) / 2.0
  return BootstrapInterval(
    estimate=float(estimate),
    ci_low=float(torch.quantile(draws, alpha)),
    ci_high=float(torch.quantile(draws, 1.0 - alpha)),
    confidence=confidence,
    cluster_count=len(cluster_to_stratum),
    resamples=resamples,
    seed=seed,
  )


@dataclass(frozen=True)
class CoverageThresholds:
  min_clusters: int = 16
  min_positive_clusters: int = 8
  min_negative_clusters: int = 8
  min_positive_anchors: int = 200
  min_negative_anchors: int = 200
  min_progress_clusters: int = 16
  min_progress_anchors: int = 5_000
  min_ray_valid_fraction: float = 0.99
  max_chatter_fraction: float = 0.10


def binary_coverage(
  target: torch.Tensor,
  valid: torch.Tensor,
  cluster_ids: Sequence[object],
) -> dict[str, int]:
  if target.shape != valid.shape or target.ndim != 1:
    raise ValueError("binary target/valid must be aligned vectors")
  if target.dtype != torch.bool or valid.dtype != torch.bool:
    raise ValueError("binary target/valid must be bool")
  if target.numel() != len(cluster_ids):
    raise ValueError("cluster_ids must align with binary target")
  used_clusters = set()
  positive_clusters = set()
  negative_clusters = set()
  for value, available, cluster in zip(
    target.tolist(), valid.tolist(), cluster_ids, strict=True
  ):
    if not available:
      continue
    used_clusters.add(cluster)
    (positive_clusters if value else negative_clusters).add(cluster)
  return {
    "clusters": len(used_clusters),
    "positive_clusters": len(positive_clusters),
    "negative_clusters": len(negative_clusters),
    "positive_anchors": int((target & valid).sum()),
    "negative_anchors": int((~target & valid).sum()),
  }


def coverage_pass(
  *,
  binary: Mapping[str, Mapping[str, int]],
  progress_clusters: int,
  progress_valid_anchors: int,
  ray_valid_fraction: float,
  chatter_fraction: float,
  thresholds: CoverageThresholds = CoverageThresholds(),
) -> tuple[bool, tuple[str, ...]]:
  reasons: list[str] = []
  for name, value in binary.items():
    checks = {
      "clusters": thresholds.min_clusters,
      "positive_clusters": thresholds.min_positive_clusters,
      "negative_clusters": thresholds.min_negative_clusters,
      "positive_anchors": thresholds.min_positive_anchors,
      "negative_anchors": thresholds.min_negative_anchors,
    }
    for field, minimum in checks.items():
      if int(value.get(field, -1)) < minimum:
        reasons.append(f"{name}:{field}_below_{minimum}")
  if progress_clusters < thresholds.min_progress_clusters:
    reasons.append("progress_clusters_below_minimum")
  if progress_valid_anchors < thresholds.min_progress_anchors:
    reasons.append("progress_valid_anchors_below_minimum")
  if not math.isfinite(ray_valid_fraction) or (
    ray_valid_fraction < thresholds.min_ray_valid_fraction
  ):
    reasons.append("ray_valid_fraction_below_minimum")
  if not math.isfinite(chatter_fraction) or (
    chatter_fraction > thresholds.max_chatter_fraction
  ):
    reasons.append("contact_chatter_above_maximum")
  return not reasons, tuple(reasons)


def fail_closed_observability_gate(
  *,
  integrity_checks: Mapping[str, bool],
  coverage_ok: bool,
  primary_by_stratum: Mapping[str, BootstrapInterval],
  secondary_macro: Mapping[str, BootstrapInterval],
  direction_estimates: Mapping[str, float],
  contact_minus_clearance: BootstrapInterval | None = None,
) -> dict[str, object]:
  """Apply the preregistered positive-direction/CI fail-closed decision."""
  reasons = [name for name, passed in integrity_checks.items() if not passed]
  if not coverage_ok:
    reasons.append("coverage_failed")
  if not primary_by_stratum:
    reasons.append("primary_strata_missing")
  for name, interval in primary_by_stratum.items():
    if interval.estimate <= 0.0 or interval.ci_low <= 0.0:
      reasons.append(f"primary_no_positive_ci:{name}")
  for name, interval in secondary_macro.items():
    if interval.estimate <= 0.0 or interval.ci_low <= 0.0:
      reasons.append(f"secondary_no_positive_ci:{name}")
  for name, estimate in direction_estimates.items():
    if not math.isfinite(estimate) or estimate < 0.0:
      reasons.append(f"direction_not_consistent:{name}")
  if contact_minus_clearance is not None and contact_minus_clearance.ci_high < 0.0:
    reasons.append("clearance_significantly_dominates_contact")
  passed = not reasons
  return {
    "observability_diagnostic_passed": passed,
    "decision": (
      "PASS_CONTACT_INCREMENTAL_VALUE"
      if passed
      else "INCONCLUSIVE_DO_NOT_TRAIN"
    ),
    "reasons": reasons,
  }
