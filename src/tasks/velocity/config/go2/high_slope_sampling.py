"""Level-aware hard-case terrain sampling for the Go2 V7 slope probe."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


SLOPE_DOWN_TERRAIN = "hf_pyramid_slope"
SLOPE_UP_TERRAIN = "hf_pyramid_slope_inv"
SLOPE_UP_LEVELS = (8, 9)
SLOPE_DOWN_LEVELS = (9,)
TARGET_HARD_CASE_RATIO = 0.10


def terrain_column_names(
  sub_terrains: Mapping[str, Any], num_cols: int
) -> tuple[str, ...]:
  """Reproduce mjlab's curriculum terrain-family allocation per column."""
  if num_cols <= 0:
    raise ValueError("num_cols must be positive")
  if not sub_terrains:
    raise ValueError("sub_terrains must not be empty")
  names = tuple(sub_terrains)
  proportions = np.asarray(
    [float(sub_terrains[name].proportion) for name in names], dtype=np.float64
  )
  if not np.isfinite(proportions).all() or np.any(proportions < 0.0):
    raise ValueError("terrain proportions must be finite and nonnegative")
  total = float(proportions.sum())
  if total <= 0.0:
    raise ValueError("terrain proportions must have positive mass")
  cumulative = np.cumsum(proportions / total)
  return tuple(
    names[int(np.where(column / num_cols + 0.001 < cumulative)[0][0])]
    for column in range(num_cols)
  )


def hard_case_slot_mask(
  *,
  num_rows: int,
  column_names: Sequence[str],
  slope_up_levels: Sequence[int] = SLOPE_UP_LEVELS,
  slope_down_levels: Sequence[int] = SLOPE_DOWN_LEVELS,
) -> torch.Tensor:
  """Return a flattened ``[level, terrain-column]`` hard-case mask."""
  if num_rows <= 0 or not column_names:
    raise ValueError("terrain grid dimensions must be positive")
  for level in (*slope_up_levels, *slope_down_levels):
    if isinstance(level, bool) or not isinstance(level, int):
      raise TypeError("hard-case levels must be integers")
    if not 0 <= level < num_rows:
      raise ValueError(f"hard-case level {level} is outside [0, {num_rows})")
  mask = torch.zeros((num_rows, len(column_names)), dtype=torch.bool)
  for column, name in enumerate(column_names):
    if name == SLOPE_UP_TERRAIN:
      mask[list(slope_up_levels), column] = True
    elif name == SLOPE_DOWN_TERRAIN:
      mask[list(slope_down_levels), column] = True
  if not mask.any():
    raise ValueError("configured terrain grid contains no requested hard-case slots")
  return mask.flatten()


def reweighted_slot_probabilities(
  base_probabilities: torch.Tensor,
  hard_mask: torch.Tensor,
  target_hard_case_ratio: float,
  *,
  fallback_probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
  """Set hard-case mass while preserving conditional distributions.

  The transformation is equivalent to multiplying hard and non-hard slots by
  separate constants.  A fallback is used only when the observed population has
  zero mass in one group, as happens before V7 reaches levels 8 and 9.
  """
  if base_probabilities.ndim != 1 or hard_mask.shape != base_probabilities.shape:
    raise ValueError("base_probabilities and hard_mask must be matching vectors")
  if hard_mask.dtype != torch.bool:
    raise TypeError("hard_mask must have boolean dtype")
  if not 0.0 < target_hard_case_ratio < 1.0:
    raise ValueError("target_hard_case_ratio must be in (0, 1)")
  base = base_probabilities.to(dtype=torch.float64)
  if not torch.isfinite(base).all() or torch.any(base < 0.0) or base.sum() <= 0.0:
    raise ValueError("base probabilities must be finite, nonnegative, and nonempty")
  base = base / base.sum()
  fallback = (
    torch.ones_like(base) if fallback_probabilities is None
    else fallback_probabilities.to(device=base.device, dtype=base.dtype)
  )
  if fallback.shape != base.shape:
    raise ValueError("fallback_probabilities must match base_probabilities")
  if (
    not torch.isfinite(fallback).all()
    or torch.any(fallback < 0.0)
    or fallback.sum() <= 0.0
  ):
    raise ValueError("fallback probabilities must be finite and nonnegative")

  result = torch.zeros_like(base)
  for group_mask, mass in (
    (hard_mask, target_hard_case_ratio),
    (~hard_mask, 1.0 - target_hard_case_ratio),
  ):
    conditional = base * group_mask
    if conditional.sum() <= 0.0:
      conditional = fallback * group_mask
    if conditional.sum() <= 0.0:
      raise ValueError("hard and non-hard groups must both contain probability mass")
    result += float(mass) * conditional / conditional.sum()
  return result


def sample_reweighted_slots(
  candidate_slots: torch.Tensor,
  donor_probabilities: torch.Tensor,
  hard_mask: torch.Tensor,
  *,
  target_hard_case_ratio: float,
  quota_residual: float,
  generator: torch.Generator,
) -> tuple[torch.Tensor, int, int, float]:
  """Change only candidate slots whose hard/non-hard membership must change."""
  if candidate_slots.ndim != 1:
    raise ValueError("candidate_slots must be a vector")
  if donor_probabilities.ndim != 1 or donor_probabilities.shape != hard_mask.shape:
    raise ValueError("donor probabilities and hard_mask must be matching vectors")
  if not 0.0 < target_hard_case_ratio < 1.0:
    raise ValueError("target_hard_case_ratio must be in (0, 1)")
  if (
    not torch.isfinite(donor_probabilities).all()
    or torch.any(donor_probabilities < 0.0)
    or donor_probabilities.sum() <= 0.0
  ):
    raise ValueError("donor probabilities must be finite and nonnegative")
  if candidate_slots.numel() and (
    candidate_slots.min() < 0 or candidate_slots.max() >= len(hard_mask)
  ):
    raise ValueError("candidate slot index is out of range")
  if not 0.0 <= quota_residual < 1.0:
    raise ValueError("quota_residual must be in [0, 1)")
  count = len(candidate_slots)
  if count == 0:
    return candidate_slots.clone(), 0, 0, quota_residual

  desired_hard = count * target_hard_case_ratio + quota_residual
  hard_count = int(desired_hard + 1.0e-12)
  next_residual = desired_hard - hard_count
  hard_count = min(max(hard_count, 0), count)

  output = candidate_slots.clone()
  candidate_hard = hard_mask[candidate_slots]
  current_hard_count = int(candidate_hard.sum())
  delta = hard_count - current_hard_count
  if delta == 0:
    return output, hard_count, 0, next_residual

  source_mask = ~candidate_hard if delta > 0 else candidate_hard
  source_indices = source_mask.nonzero().flatten()
  changed_count = abs(delta)
  selected = source_indices[
    torch.randperm(len(source_indices), device=output.device, generator=generator)[
      :changed_count
    ]
  ]
  target_group = hard_mask if delta > 0 else ~hard_mask
  group_probabilities = donor_probabilities * target_group
  if group_probabilities.sum() <= 0.0:
    raise ValueError("donor distribution has no mass in the requested group")
  output[selected] = torch.multinomial(
    group_probabilities,
    changed_count,
    replacement=True,
    generator=generator,
  )
  return output, hard_count, changed_count, next_residual


class HighSlopeHardCaseSampler:
  """Reset event that targets existing high-slope terrain slots."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self.target_ratio = float(params["target_hard_case_ratio"])
    if not 0.0 < self.target_ratio < 1.0:
      raise ValueError("target_hard_case_ratio must be in (0, 1)")
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("high-slope sampling requires generated curriculum terrain")
    generator_cfg = terrain.cfg.terrain_generator
    if generator_cfg is None or not generator_cfg.curriculum:
      raise ValueError("high-slope sampling requires curriculum=True")

    self.num_rows, self.num_cols = terrain.terrain_origins.shape[:2]
    column_names = terrain_column_names(generator_cfg.sub_terrains, self.num_cols)
    self.hard_mask = hard_case_slot_mask(
      num_rows=self.num_rows,
      column_names=column_names,
      slope_up_levels=tuple(params["slope_up_levels"]),
      slope_down_levels=tuple(params["slope_down_levels"]),
    ).to(env.device)
    self.nominal_probabilities = torch.full(
      (self.num_rows * self.num_cols,),
      1.0 / (self.num_rows * self.num_cols),
      dtype=torch.float64,
      device=env.device,
    )
    seed = int(env.cfg.seed if env.cfg.seed is not None else 0)
    seed += int(params.get("seed_offset", 0))
    self.generator = torch.Generator(device=env.device)
    self.seed = seed
    self.generator.manual_seed(self.seed)
    self.sampled_slot_histogram = torch.zeros(
      self.num_rows * self.num_cols,
      dtype=torch.long,
      device=env.device,
    )
    self.rebase()

  def _population_ratio(self) -> float:
    terrain = self._env.scene.terrain
    assert terrain is not None
    slots = terrain.terrain_levels * self.num_cols + terrain.terrain_types
    return float(self.hard_mask[slots].float().mean())

  def rebase(self) -> None:
    """Start a new sampling stream from the currently restored terrain state."""
    self.generator.manual_seed(self.seed)
    self.quota_residual = 0.0
    self.total_reset_count = 0
    self.total_hard_count = 0
    self.sampled_slot_histogram.zero_()
    self.candidate_hard_ratio = 0.0
    self.changed_slot_ratio = 0.0
    self.hard_case_batch_ratio = 0.0
    self.hard_case_reset_ratio = 0.0
    self.hard_case_population_ratio = self._population_ratio()

  def state_dict(self) -> dict[str, Any]:
    """Return exact state needed for deterministic probe continuation."""
    return {
      "schema_version": 1,
      "target_hard_case_ratio": self.target_ratio,
      "quota_residual": self.quota_residual,
      "total_reset_count": self.total_reset_count,
      "total_hard_count": self.total_hard_count,
      "sampled_slot_histogram": self.sampled_slot_histogram.detach().cpu().clone(),
      "generator_state": self.generator.get_state().cpu().clone(),
      "candidate_hard_ratio": self.candidate_hard_ratio,
      "changed_slot_ratio": self.changed_slot_ratio,
      "hard_case_batch_ratio": self.hard_case_batch_ratio,
      "hard_case_reset_ratio": self.hard_case_reset_ratio,
      "hard_case_population_ratio": self.hard_case_population_ratio,
    }

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Restore a probe sampling stream after terrain state restoration."""
    if state.get("schema_version") != 1:
      raise ValueError("unsupported high-slope sampler state schema")
    if float(state.get("target_hard_case_ratio", -1.0)) != self.target_ratio:
      raise ValueError("sampler target ratio differs from checkpoint state")
    histogram = torch.as_tensor(
      state["sampled_slot_histogram"], device=self._env.device, dtype=torch.long
    )
    if histogram.shape != self.sampled_slot_histogram.shape or torch.any(histogram < 0):
      raise ValueError("invalid sampled slot histogram in checkpoint")
    total_reset_count = int(state["total_reset_count"])
    total_hard_count = int(state["total_hard_count"])
    if total_reset_count < 0 or not 0 <= total_hard_count <= total_reset_count:
      raise ValueError("invalid sampler counters in checkpoint")
    if int(histogram.sum()) != total_reset_count:
      raise ValueError("sampled slot histogram disagrees with reset count")
    if int(histogram[self.hard_mask].sum()) != total_hard_count:
      raise ValueError("hard slot histogram disagrees with hard count")
    quota_residual = float(state["quota_residual"])
    if not 0.0 <= quota_residual < 1.0:
      raise ValueError("invalid sampler quota residual in checkpoint")

    self.generator.set_state(torch.as_tensor(state["generator_state"]).cpu())
    self.quota_residual = quota_residual
    self.total_reset_count = total_reset_count
    self.total_hard_count = total_hard_count
    self.sampled_slot_histogram.copy_(histogram)
    for name in (
      "candidate_hard_ratio",
      "changed_slot_ratio",
      "hard_case_batch_ratio",
      "hard_case_reset_ratio",
    ):
      setattr(self, name, float(state[name]))
    expected_ratio = total_hard_count / max(total_reset_count, 1)
    if abs(self.hard_case_reset_ratio - expected_ratio) > 1.0e-12:
      raise ValueError("saved hard-case reset ratio disagrees with counters")
    self.hard_case_population_ratio = self._population_ratio()

  def sampling_audit(self) -> dict[str, Any]:
    """Expose integer reset counts and sampled slot membership for acceptance."""
    return {
      "total_reset_count": self.total_reset_count,
      "total_hard_count": self.total_hard_count,
      "sampled_slot_histogram": self.sampled_slot_histogram.detach().cpu().clone(),
      "hard_slot_mask": self.hard_mask.detach().cpu().clone(),
      "target_hard_case_ratio": self.target_ratio,
      "hard_case_reset_ratio": self.hard_case_reset_ratio,
    }

  def _env_ids_tensor(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
      return torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
    return env_ids.to(device=self._env.device, dtype=torch.long).flatten()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    **_: Any,
  ) -> None:
    ids = self._env_ids_tensor(env_ids)
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    all_slot_ids = terrain.terrain_levels * self.num_cols + terrain.terrain_types
    counts = torch.bincount(
      all_slot_ids, minlength=self.num_rows * self.num_cols
    ).to(torch.float64)
    probabilities = reweighted_slot_probabilities(
      counts,
      self.hard_mask,
      self.target_ratio,
      fallback_probabilities=self.nominal_probabilities,
    )
    candidate_slots = all_slot_ids[ids]
    candidate_hard_count = int(self.hard_mask[candidate_slots].sum())
    sampled, hard_count, changed_count, self.quota_residual = sample_reweighted_slots(
      candidate_slots,
      probabilities,
      self.hard_mask,
      target_hard_case_ratio=self.target_ratio,
      quota_residual=self.quota_residual,
      generator=self.generator,
    )
    levels = torch.div(sampled, self.num_cols, rounding_mode="floor")
    types = sampled.remainder(self.num_cols)
    changed = sampled != candidate_slots
    changed_ids = ids[changed]
    terrain.terrain_levels[changed_ids] = levels[changed]
    terrain.terrain_types[changed_ids] = types[changed]
    terrain.env_origins[changed_ids] = terrain.terrain_origins[
      levels[changed], types[changed]
    ]

    self.total_reset_count += len(ids)
    self.total_hard_count += hard_count
    self.sampled_slot_histogram += torch.bincount(
      sampled, minlength=self.num_rows * self.num_cols
    )
    population_slots = terrain.terrain_levels * self.num_cols + terrain.terrain_types
    denominator = max(len(ids), 1)
    self.candidate_hard_ratio = candidate_hard_count / denominator
    self.changed_slot_ratio = changed_count / denominator
    self.hard_case_batch_ratio = hard_count / denominator
    self.hard_case_reset_ratio = self.total_hard_count / max(self.total_reset_count, 1)
    self.hard_case_population_ratio = float(
      self.hard_mask[population_slots].float().mean()
    )


def high_slope_sampling_metric(
  env: ManagerBasedRlEnv,
  metric_name: str,
  event_name: str = "high_slope_sampling",
) -> torch.Tensor:
  """Expose reset-sampler telemetry through the normal metrics logger."""
  term = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(term, HighSlopeHardCaseSampler):
    raise TypeError(f"event {event_name!r} is not a HighSlopeHardCaseSampler")
  value = getattr(term, metric_name)
  return torch.full((env.num_envs,), float(value), device=env.device)


__all__ = [
  "HighSlopeHardCaseSampler",
  "SLOPE_DOWN_LEVELS",
  "SLOPE_DOWN_TERRAIN",
  "SLOPE_UP_LEVELS",
  "SLOPE_UP_TERRAIN",
  "TARGET_HARD_CASE_RATIO",
  "hard_case_slot_mask",
  "high_slope_sampling_metric",
  "reweighted_slot_probabilities",
  "sample_reweighted_slots",
  "terrain_column_names",
]
