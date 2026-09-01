"""Strictly causal confirmed-contact filtering for Go2 diagnostics.

This module is deliberately CPU-only and has no simulator or training
dependencies.  A committed per-foot contact bit changes only on the second
consecutive actor-visible observation of the new raw value.  The change is
reported at that second observation; earlier outputs are never rewritten.

Episode starts are explicit.  The first visible observation initializes the
episode baseline without emitting a transition, and pending confirmations are
never allowed to cross a reset or an invalid gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


NATIVE_FOOT_ORDER = ("FL", "FR", "RL", "RR")
FOOT_COUNT = len(NATIVE_FOOT_ORDER)
DEFAULT_CONFIRMATION_TICKS = 2


@dataclass(frozen=True)
class ConfirmedContactStep:
  """One actor-visible output from :class:`CausalConfirmedContactFilter`."""

  contact: torch.Tensor
  changed: torch.Tensor
  valid: torch.Tensor


@dataclass(frozen=True)
class ConfirmedContactTimeline:
  """Offline reconstruction with the same timing as the online filter."""

  contact: torch.Tensor
  changed: torch.Tensor
  valid: torch.Tensor


def _require_cpu(name: str, value: torch.Tensor) -> None:
  if value.device.type != "cpu":
    raise ValueError(f"{name} must be a CPU tensor")


def _validate_integer(name: str, value: torch.Tensor) -> None:
  if value.dtype == torch.bool or value.dtype.is_floating_point:
    raise ValueError(f"{name} must be an integer tensor")


class CausalConfirmedContactFilter:
  """Batched online contact filter with explicit, fail-closed resets.

  ``episode_start`` must be true on the first visible tick for an environment
  and whenever its ``attempt_id`` changes.  A visible tick after an invalid
  tick is also a new initialization and therefore requires
  ``episode_start=True``.  Violations raise instead of silently carrying state
  between episodes.
  """

  def __init__(
    self,
    batch_size: int,
    *,
    feet: int = FOOT_COUNT,
    confirmation_ticks: int = DEFAULT_CONFIRMATION_TICKS,
  ) -> None:
    if batch_size < 1:
      raise ValueError("batch_size must be positive")
    if feet != FOOT_COUNT:
      raise ValueError("Go2 confirmed contact must use four native-order feet")
    if confirmation_ticks < 1:
      raise ValueError("confirmation_ticks must be positive")
    self.batch_size = int(batch_size)
    self.feet = int(feet)
    self.confirmation_ticks = int(confirmation_ticks)
    self._committed = torch.zeros(self.batch_size, self.feet, dtype=torch.bool)
    self._candidate = torch.zeros_like(self._committed)
    self._candidate_count = torch.zeros(
      self.batch_size, self.feet, dtype=torch.long
    )
    self._attempt_id = torch.zeros(self.batch_size, dtype=torch.long)
    self._initialized = torch.zeros(self.batch_size, dtype=torch.bool)

  def step(
    self,
    raw_contact: torch.Tensor,
    *,
    attempt_id: torch.Tensor,
    episode_start: torch.Tensor,
    visible: torch.Tensor | None = None,
  ) -> ConfirmedContactStep:
    """Advance exactly one actor-visible tick without backdating output."""
    expected_contact = (self.batch_size, self.feet)
    if raw_contact.shape != expected_contact or raw_contact.dtype != torch.bool:
      raise ValueError(
        f"raw_contact must be native-order bool shape {expected_contact}"
      )
    if attempt_id.shape != (self.batch_size,):
      raise ValueError("attempt_id must have shape (batch_size,)")
    _validate_integer("attempt_id", attempt_id)
    if episode_start.shape != (self.batch_size,) or episode_start.dtype != torch.bool:
      raise ValueError("episode_start must be bool shape (batch_size,)")
    if visible is None:
      visible = torch.ones(self.batch_size, dtype=torch.bool)
    if visible.shape != (self.batch_size,) or visible.dtype != torch.bool:
      raise ValueError("visible must be bool shape (batch_size,)")
    for name, value in (
      ("raw_contact", raw_contact),
      ("attempt_id", attempt_id),
      ("episode_start", episode_start),
      ("visible", visible),
    ):
      _require_cpu(name, value)
    if bool((episode_start & ~visible).any()):
      raise ValueError("an invisible tick cannot initialize an episode")

    # Invalid observations terminate all filter state for those environments.
    invalid = ~visible
    self._initialized[invalid] = False
    self._candidate_count[invalid] = 0

    continuing = visible & ~episode_start
    missing_start = continuing & ~self._initialized
    if bool(missing_start.any()):
      ids = torch.where(missing_start)[0].tolist()
      raise ValueError(f"visible episode is missing episode_start for envs {ids}")
    unexpected_attempt = continuing & (attempt_id != self._attempt_id)
    if bool(unexpected_attempt.any()):
      ids = torch.where(unexpected_attempt)[0].tolist()
      raise ValueError(f"attempt_id changed without episode_start for envs {ids}")
    duplicate_start = episode_start & self._initialized & (
      attempt_id == self._attempt_id
    )
    if bool(duplicate_start.any()):
      ids = torch.where(duplicate_start)[0].tolist()
      raise ValueError(f"duplicate episode_start for unchanged attempt_id in envs {ids}")

    changed = torch.zeros_like(self._committed)

    # A reset observation establishes a baseline.  It is not a transition and
    # requires no historical sample from the preceding episode.
    if bool(episode_start.any()):
      self._committed[episode_start] = raw_contact[episode_start]
      self._candidate[episode_start] = raw_contact[episode_start]
      self._candidate_count[episode_start] = 0
      self._attempt_id[episode_start] = attempt_id[episode_start]
      self._initialized[episode_start] = True

    active = continuing
    if bool(active.any()):
      equals_committed = raw_contact == self._committed
      cancel = active[:, None] & equals_committed
      self._candidate_count[cancel] = 0

      differs = active[:, None] & ~equals_committed
      same_candidate = differs & (raw_contact == self._candidate) & (
        self._candidate_count > 0
      )
      new_candidate = differs & ~same_candidate
      self._candidate[new_candidate] = raw_contact[new_candidate]
      self._candidate_count[new_candidate] = 1
      self._candidate_count[same_candidate] += 1

      confirmed = differs & (
        self._candidate_count >= self.confirmation_ticks
      )
      self._committed[confirmed] = self._candidate[confirmed]
      changed[confirmed] = True
      self._candidate_count[confirmed] = 0

    output = torch.zeros_like(self._committed)
    output[visible] = self._committed[visible]
    return ConfirmedContactStep(
      contact=output.clone(),
      changed=changed,
      valid=visible.clone(),
    )


def recompute_causal_confirmed_contact(
  raw_contact: torch.Tensor,
  *,
  attempt_id: torch.Tensor,
  episode_start: torch.Tensor,
  state_valid: torch.Tensor | None = None,
  confirmation_ticks: int = DEFAULT_CONFIRMATION_TICKS,
) -> ConfirmedContactTimeline:
  """Recompute a trajectory batch using only each tick's available prefix.

  Accepted contact shapes are ``(ticks, batch, 4)`` and ``(ticks, 4)``.  The
  latter is treated as a single trajectory and returned with the same rank.
  ``attempt_id``, ``episode_start`` and ``state_valid`` use the corresponding
  ``(ticks, batch)`` or ``(ticks,)`` shape.
  """
  _require_cpu("raw_contact", raw_contact)
  if raw_contact.dtype != torch.bool or raw_contact.ndim not in (2, 3):
    raise ValueError("raw_contact must be bool shape (ticks, 4) or (ticks, batch, 4)")
  single = raw_contact.ndim == 2
  contact = raw_contact[:, None, :] if single else raw_contact
  if contact.shape[2] != FOOT_COUNT:
    raise ValueError("raw_contact must use four native-order feet")
  ticks, batch, _ = contact.shape
  if ticks < 1 or batch < 1:
    raise ValueError("trajectory batch must contain at least one tick and trajectory")
  expected = (ticks,) if single else (ticks, batch)
  if attempt_id.shape != expected:
    raise ValueError(f"attempt_id must have shape {expected}")
  _validate_integer("attempt_id", attempt_id)
  if episode_start.shape != expected or episode_start.dtype != torch.bool:
    raise ValueError(f"episode_start must be bool shape {expected}")
  _require_cpu("attempt_id", attempt_id)
  _require_cpu("episode_start", episode_start)
  if state_valid is None:
    state_valid = torch.ones(expected, dtype=torch.bool)
  if state_valid.shape != expected or state_valid.dtype != torch.bool:
    raise ValueError(f"state_valid must be bool shape {expected}")
  _require_cpu("state_valid", state_valid)

  attempts = attempt_id[:, None] if single else attempt_id
  starts = episode_start[:, None] if single else episode_start
  valid = state_valid[:, None] if single else state_valid
  filter_state = CausalConfirmedContactFilter(
    batch, confirmation_ticks=confirmation_ticks
  )
  outputs = torch.zeros_like(contact)
  changes = torch.zeros_like(contact)
  output_valid = torch.zeros(ticks, batch, dtype=torch.bool)
  for tick in range(ticks):
    step = filter_state.step(
      contact[tick],
      attempt_id=attempts[tick],
      episode_start=starts[tick],
      visible=valid[tick],
    )
    outputs[tick] = step.contact
    changes[tick] = step.changed
    output_valid[tick] = step.valid

  if single:
    return ConfirmedContactTimeline(
      outputs[:, 0], changes[:, 0], output_valid[:, 0]
    )
  return ConfirmedContactTimeline(outputs, changes, output_valid)
