"""Analyze the registered Go2 foot-contact observability raw suite on CPU.

The analyzer is deliberately evaluation-only.  It accepts only the exact
18-chunk formal manifest, reconstructs one pre0+post state timeline per route
attempt, and writes a fail-closed JSON/Markdown decision without overwriting an
existing result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks.velocity.evaluation.foot_contact_observability import (  # noqa: E402
  FORMAL_SEEDS,
  NATIVE_FOOT_GEOMS,
  NATIVE_FOOT_ORDER,
  NATIVE_TO_CANONICAL,
  TrajectoryTimeline,
  assert_no_group_leakage,
  balanced_binary_weights,
  build_future_labels,
  contact_chatter_metrics,
  downsample_anchor_mask,
  fit_weighted_ridge,
  group_balanced_weights,
  validate_native_foot_order,
  weighted_pr_auc,
)


CONTRACT = ROOT / "docs/reviews/go2_foot_contact_observability_contract_20260809.json"
DEFAULT_RAW_DIR = ROOT / "docs/reviews/go2_foot_contact_observability_raw_20260809"
DEFAULT_OUTPUT_PREFIX = ROOT / "docs/reviews/go2_foot_contact_observability_summary_20260809"
COLLECTOR = ROOT / "scripts/collect_go2_foot_contact_observability.py"
SEEDS = (42, 43, 44)
PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")
SPEEDS = (0.3, 0.5)
HORIZONS = (10, 25, 50)
TARGETS = ("slip_onset", "unexpected_transition", "catastrophic_failure", "future_progress")
BINARY_TARGETS = TARGETS[:3]
FEATURES = ("baseline", "contact", "clearance")
FIXED_L2 = 0.01
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_809
CONTROL_DT = 0.02
ANCHOR_START = 10
ANCHOR_STRIDE = 5


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, torch.Tensor):
    return value.tolist()
  if isinstance(value, tuple):
    return [_jsonable(item) for item in value]
  if isinstance(value, list):
    return [_jsonable(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def _contract_value(contract: dict[str, Any], path: str) -> Any:
  value: Any = contract
  for key in path.split("."):
    value = value[key]
  return value


def _validate_contract(contract: dict[str, Any]) -> None:
  expected = {
    "evaluation_suite": "go2_foot_contact_incremental_observability_v1",
    "source.actor_dim": 234,
    "source.critic_dim": 261,
    "source.critic_contact_slice": [245, 249],
    "matrix.seeds": [42, 43, 44],
    "matrix.profiles": ["clean", "randomized"],
    "matrix.route_kinds": ["straight", "arc", "s_curve"],
    "matrix.matched_slots_per_chunk": 16,
    "matrix.steps_per_chunk": 2400,
    "matrix.chunk_count": 18,
    "matrix.trajectory_count": 288,
    "timing.control_dt_s": CONTROL_DT,
    "timing.anchor_stride_ticks": ANCHOR_STRIDE,
    "timing.horizons_ticks": list(HORIZONS),
    "foot_semantics.runtime_sensor_order": list(NATIVE_FOOT_ORDER),
    "foot_semantics.runtime_geom_names": list(NATIVE_FOOT_GEOMS),
    "foot_semantics.native_to_canonical_permutation": list(NATIVE_TO_CANONICAL),
    "foot_semantics.gait_offsets_native": [0.0, 0.5, 0.5, 0.0],
    "foot_semantics.gait_period_s": 0.6,
    "foot_semantics.gait_stance_threshold": 0.56,
    "cross_validation.outer_folds": 3,
    "cross_validation.outer_split": "leave_one_seed_out",
    "cross_validation.fixed_l2": FIXED_L2,
    "cross_validation.lambda_grid": None,
    "bootstrap.unit": ["seed", "matched_slot"],
    "bootstrap.stratify_by_speed": True,
    "bootstrap.resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap.seed": BOOTSTRAP_SEED,
    "coverage_gates_per_profile_speed_horizon.usable_clusters_min": 16,
    "coverage_gates_per_profile_speed_horizon.binary_positive_event_clusters_min": 8,
    "coverage_gates_per_profile_speed_horizon.binary_positive_anchors_min": 200,
    "coverage_gates_per_profile_speed_horizon.binary_negative_anchors_min": 200,
    "coverage_gates_per_profile_speed_horizon.progress_clusters_min": 16,
    "coverage_gates_per_profile_speed_horizon.progress_anchors_min": 5000,
  }
  mismatches = {
    path: {"actual": _contract_value(contract, path), "expected": wanted}
    for path, wanted in expected.items()
    if _contract_value(contract, path) != wanted
  }
  if mismatches:
    raise ValueError(f"machine contract differs from analyzer freeze: {mismatches}")


def _resolve_inventory_path(value: str, manifest_path: Path) -> Path:
  path = Path(value).expanduser()
  if path.is_absolute():
    return path.resolve()
  candidate = (manifest_path.parent / path).resolve()
  return candidate if candidate.exists() else (ROOT / path).resolve()


def _load_manifest(raw_dir: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
  manifest_path = raw_dir / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"formal manifest is required: {manifest_path}")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  contract_sha = _sha256(CONTRACT)
  contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
  _validate_contract(contract)
  expected_manifest = {
    "evaluation_suite": "go2_foot_contact_observability_raw_manifest_v1",
    "contract_sha256": contract_sha,
    "chunk_count": 18,
    "trajectory_count": 288,
    "training_changed": False,
    "learn_called": False,
  }
  mismatch = {
    key: {"actual": manifest.get(key), "expected": value}
    for key, value in expected_manifest.items() if manifest.get(key) != value
  }
  if mismatch:
    raise ValueError(f"formal manifest identity mismatch: {mismatch}")
  if manifest.get("collector_sha256") != _sha256(COLLECTOR):
    raise ValueError("manifest collector SHA differs from the frozen collector")
  inventory = manifest.get("chunks")
  if not isinstance(inventory, list) or len(inventory) != 18:
    raise ValueError("manifest must inventory exactly 18 chunks")
  expected_ids = {(seed, profile, route) for seed in SEEDS for profile in PROFILES for route in ROUTES}
  actual_ids = {(item.get("seed"), item.get("profile"), item.get("route_kind")) for item in inventory}
  if actual_ids != expected_ids:
    raise ValueError(f"manifest chunk identities differ: {actual_ids ^ expected_ids}")
  loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
  observed_rows = 0
  for item in sorted(inventory, key=lambda x: (x["seed"], x["profile"], x["route_kind"])):
    path = _resolve_inventory_path(str(item["path"]), manifest_path)
    expected_path = (
      raw_dir / f"seed_{item['seed']}" / item["profile"] /
      f"{item['route_kind']}.pt"
    ).resolve()
    if path != expected_path:
      raise ValueError(f"inventoried chunk path differs from registered layout: {path}")
    if not path.is_file():
      raise FileNotFoundError(f"inventoried chunk is missing: {path}")
    if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
      raise ValueError(f"inventoried chunk size/SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    identity = {
      "evaluation_suite": "go2_foot_contact_observability_raw_chunk_v1",
      "mode": "formal", "seed": item["seed"], "profile": item["profile"],
      "route_kind": item["route_kind"], "num_envs": 16,
      "steps_requested": 2400, "contract_sha256": contract_sha,
      "collector_sha256": manifest["collector_sha256"],
      "checkpoint_sha256": contract["source"]["checkpoint_sha256"],
    }
    bad = {key: (payload.get(key), value) for key, value in identity.items() if payload.get(key) != value}
    if bad:
      raise ValueError(f"chunk identity mismatch in {path}: {bad}")
    arrays = payload.get("arrays") or {}
    active = arrays.get("anchor_active")
    resets = payload.get("reset_count")
    if not isinstance(active, torch.Tensor) or not isinstance(resets, torch.Tensor):
      raise ValueError(f"chunk inventory counters unavailable in {path}")
    raw_rows = int(active.sum())
    if (
      raw_rows != int(item["raw_rows"])
      or int(resets.sum()) != int(item["resets"])
      or int(payload["steps_executed"]) != int(item["steps_executed"])
    ):
      raise ValueError(f"chunk counters differ from manifest inventory: {path}")
    observed_rows += raw_rows
    loaded.append((item, payload))
  if observed_rows != int(manifest.get("observed_raw_rows", -1)):
    raise ValueError("manifest observed_raw_rows differs from exact chunk sum")
  manifest["manifest_path"] = str(manifest_path)
  manifest["manifest_sha256"] = _sha256(manifest_path)
  manifest["validated_contract_sha256"] = contract_sha
  return manifest, loaded


@dataclass
class RowDataset:
  actor: torch.Tensor
  contact: torch.Tensor
  clearance: torch.Tensor
  clearance_valid: torch.Tensor
  seed: torch.Tensor
  slot: torch.Tensor
  speed: torch.Tensor
  profile: list[str]
  route: list[str]
  targets: dict[int, dict[str, torch.Tensor]]
  valid: dict[int, dict[str, torch.Tensor]]

  @property
  def rows(self) -> int:
    return int(self.actor.shape[0])

  def groups(self, mask: torch.Tensor) -> list[tuple[int, int]]:
    ids = torch.where(mask)[0].tolist()
    return [(int(self.seed[i]), int(self.slot[i])) for i in ids]


@dataclass(frozen=True)
class _LogisticLBFGSModel:
  mean: torch.Tensor
  scale: torch.Tensor
  coefficient: torch.Tensor
  intercept: torch.Tensor
  iterations: int

  def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
    normalized = (features.double() - self.mean) / self.scale
    return torch.sigmoid(normalized @ self.coefficient + self.intercept)


def _fit_logistic_lbfgs(
  features: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor,
) -> _LogisticLBFGSModel:
  """Frozen deterministic float64 LBFGS logistic solver from the contract."""
  x = features.double()
  y = target.double()
  weight = sample_weight.double()
  if x.ndim != 2 or y.shape != (x.shape[0],) or weight.shape != y.shape:
    raise ValueError("logistic inputs are misaligned")
  if not target.bool().any() or target.bool().all():
    raise ValueError("logistic training fold requires both classes")
  if not torch.isfinite(x).all() or not torch.isfinite(weight).all():
    raise ValueError("logistic inputs must be finite")
  weight = weight / weight.sum()
  mean = (x * weight[:, None]).sum(dim=0)
  variance = ((x - mean).square() * weight[:, None]).sum(dim=0)
  scale = variance.sqrt().clamp_min(1.0e-12)
  normalized = (x - mean) / scale
  positive_rate = (y * weight).sum().clamp(1.0e-8, 1.0 - 1.0e-8)
  parameter = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
  with torch.no_grad():
    parameter[-1] = torch.logit(positive_rate)
  optimizer = torch.optim.LBFGS(
    [parameter], lr=1.0, max_iter=200, tolerance_grad=1.0e-8,
    tolerance_change=1.0e-8, history_size=100, line_search_fn="strong_wolfe",
  )

  def closure() -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    logits = normalized @ parameter[:-1] + parameter[-1]
    data = torch.nn.functional.binary_cross_entropy_with_logits(
      logits, y, weight=weight, reduction="sum"
    )
    loss = data + 0.5 * FIXED_L2 * parameter[:-1].square().sum()
    loss.backward()
    return loss

  loss = optimizer.step(closure)
  state = optimizer.state[parameter]
  iterations = int(state.get("n_iter", 0))
  if not torch.isfinite(parameter).all() or not math.isfinite(float(loss.detach())):
    raise RuntimeError("LBFGS logistic solver returned nonfinite state")
  if iterations >= 200:
    closure()
    gradient = float(parameter.grad.abs().max()) if parameter.grad is not None else math.inf
    if gradient > 1.0e-8:
      raise RuntimeError(
        f"LBFGS logistic solver did not converge: iterations={iterations}, grad={gradient}"
      )
  return _LogisticLBFGSModel(
    mean=mean, scale=scale, coefficient=parameter[:-1].detach().clone(),
    intercept=parameter[-1].detach().clone(), iterations=iterations,
  )


def _require_array(arrays: dict[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
  value = arrays.get(name)
  if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
    raise ValueError(f"raw array {name} must be {dtype} shape {shape}")
  return value


def _scheduled_stance(ticks: torch.Tensor) -> torch.Tensor:
  offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), dtype=torch.float64)
  phase = (ticks.to(torch.float64)[:, None] * CONTROL_DT / 0.6 + offsets) % 1.0
  return phase < 0.56


def _trajectory_rows(
  payload: dict[str, Any], env_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
  arrays = payload["arrays"]
  active = arrays["anchor_active"][env_index]
  n = int(active.sum())
  if n <= ANCHOR_START or not bool(active[:n].all()) or bool(active[n:].any()):
    raise ValueError("anchor_active must be one nonempty frozen prefix")
  scenario = payload["scenarios"][env_index]
  if int(scenario["matched_slot"]) != env_index:
    raise ValueError("scenario/native matched-slot order differs")

  pre_contact = arrays["sensor_contact"][env_index, :n]
  critic_contact = arrays["critic_contact"][env_index, :n]
  if not torch.equal(pre_contact, critic_contact):
    raise ValueError("critic contact and native sensor contact differ")
  contact = torch.cat((pre_contact[:1], arrays["post_contact"][env_index, :n]), dim=0).bool()
  progress = torch.cat((arrays["pre_progress"][env_index, :1], arrays["post_progress"][env_index, :n])).double()
  pre_ticks = arrays["pre_episode_tick"][env_index, :n]
  post_ticks = arrays["post_episode_tick"][env_index, :n]
  if n > 1 and not torch.equal(pre_ticks[1:], post_ticks[:-1]):
    raise ValueError("pre0+post timeline ticks do not join exactly")
  ticks = torch.cat((pre_ticks[:1], post_ticks))
  if bool((ticks < 0).any()):
    raise ValueError("valid state timeline contains a negative episode tick")

  force = arrays["post_force_w"][env_index, :n].double()
  velocity = arrays["post_foot_velocity_w"][env_index, :n].double()
  normal = arrays["post_terrain_normal_w"][env_index, :n].double()
  if not all(torch.isfinite(value).all() for value in (progress, force, velocity, normal)):
    raise ValueError("valid post-state label arrays must be finite")
  normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-12)
  normal = normal / normal_norm
  normal_force = (force * normal).sum(dim=-1).abs()
  loaded_post = arrays["post_contact"][env_index, :n].bool() & (normal_force >= 15.0)
  normal_velocity = (velocity * normal).sum(dim=-1, keepdim=True) * normal
  slip_post = torch.linalg.vector_norm(velocity - normal_velocity, dim=-1)
  loaded = torch.cat((torch.zeros(1, 4, dtype=torch.bool), loaded_post), dim=0)
  slip = torch.cat((torch.zeros(1, 4, dtype=torch.float64), slip_post), dim=0)

  done = arrays["done"][env_index, :n].bool()
  catastrophic = arrays["catastrophic"][env_index, :n].bool()
  complete = arrays["route_completed"][env_index, :n].bool()
  code = arrays["termination_code"][env_index, :n]
  if bool((catastrophic & ~done).any()) or bool((complete & catastrophic).any()):
    raise ValueError("terminal/catastrophic lifecycle is inconsistent")
  if int((done | complete).sum()) > 1 or bool((done[:-1] | complete[:-1]).any()):
    raise ValueError("anchor exists after terminal state")
  if bool(done.any()) and int(code[done][0]) == 0:
    raise ValueError("done state lacks a terminal reason code")

  timeline = TrajectoryTimeline(
    contact=contact, loaded=loaded, slip_speed=slip,
    scheduled_stance=_scheduled_stance(ticks), progress=progress,
    state_valid=torch.ones(n + 1, dtype=torch.bool),
    attempt_id=torch.zeros(n + 1, dtype=torch.long),
    action_valid=torch.ones(n, dtype=torch.bool), done_after=done,
    catastrophic_after=catastrophic, route_complete_after=complete,
  )
  labels = build_future_labels(
    timeline, horizons=HORIZONS, control_dt_s=CONTROL_DT,
    scenario_speed=float(scenario["speed"]), confirmation_steps=2,
    boundary_grace_steps=2,
  )
  anchors = downsample_anchor_mask(n, every=ANCHOR_STRIDE, start=ANCHOR_START)
  chatter = contact_chatter_metrics(
    contact, state_valid=timeline.state_valid, attempt_id=timeline.attempt_id
  )
  row = {
    "actor": arrays["actor_observation"][env_index, :n][anchors].float(),
    "contact": pre_contact[anchors].float(),
    "clearance": arrays["clearance"][env_index, :n][anchors].float(),
    "clearance_valid": arrays["clearance_valid"][env_index, :n][anchors].all(dim=1),
    "count": int(anchors.sum()), "labels": {}, "valid": {},
  }
  for horizon, value in labels.items():
    row["labels"][horizon] = {
      "slip_onset": value.slip_onset[anchors],
      "unexpected_transition": value.unexpected_transition[anchors],
      "catastrophic_failure": value.catastrophic_failure[anchors],
      "future_progress": value.future_progress[anchors],
    }
    row["valid"][horizon] = {
      "slip_onset": value.slip_onset_valid[anchors],
      "unexpected_transition": value.unexpected_transition_valid[anchors],
      "catastrophic_failure": value.catastrophic_failure_valid[anchors],
      "future_progress": value.future_progress_valid[anchors],
    }
  audit = {
    "profile": payload["profile"], "speed": float(scenario["speed"]),
    "raw_transition_edges": chatter["raw_transition_edges"],
    "isolated_excursion_edges": chatter["isolated_excursion_edges"],
  }
  return row, audit


def _build_dataset(chunks: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[RowDataset, dict[str, Any]]:
  actor: list[torch.Tensor] = []
  contact: list[torch.Tensor] = []
  clearance: list[torch.Tensor] = []
  clearance_valid: list[torch.Tensor] = []
  seeds: list[torch.Tensor] = []
  slots: list[torch.Tensor] = []
  speeds: list[torch.Tensor] = []
  profiles: list[str] = []
  routes: list[str] = []
  targets = {h: {t: [] for t in TARGETS} for h in HORIZONS}
  valid = {h: {t: [] for t in TARGETS} for h in HORIZONS}
  transition = {(p, s): torch.zeros(4, dtype=torch.long) for p in PROFILES for s in SPEEDS}
  isolated = {(p, s): torch.zeros(4, dtype=torch.long) for p in PROFILES for s in SPEEDS}
  trajectory_count = 0
  raw_rows = 0
  reset_total = 0

  validate_native_foot_order(NATIVE_FOOT_GEOMS)
  for item, payload in chunks:
    invariants = payload.get("invariants") or {}
    expected_invariants = {
      "training_changed": False,
      "learn_called": False,
      "cached_observation_not_recomputed_inside_loop": True,
      "command_cache_identity": True,
      "initial_post_placement_observation_refreshes": 1,
      "critic_contact_equals_native_sensor": True,
      "terminal_state_captured_inside_reset_hook": True,
      "finite_actor_observation": True,
      "finite_policy_action": True,
    }
    if any(invariants.get(key) != value for key, value in expected_invariants.items()):
      raise ValueError(f"collector invariant failed in {item['path']}: {invariants}")
    if tuple(payload.get("native_foot_names", ())) != NATIVE_FOOT_ORDER:
      raise ValueError("chunk native foot order differs")
    if tuple(payload.get("runtime_sensor_names", ())) != NATIVE_FOOT_GEOMS:
      raise ValueError("chunk runtime sensor order differs")
    if tuple(payload.get("native_to_canonical_permutation", ())) != NATIVE_TO_CANONICAL:
      raise ValueError("chunk canonical permutation record differs")
    critic_slices = payload.get("critic_term_slices") or {}
    if tuple(critic_slices.get("foot_contact", ())) != (245, 249):
      raise ValueError("chunk critic foot_contact slice differs from [245:249]")
    scenarios = payload.get("scenarios") or []
    if len(scenarios) != 16 or [int(x["matched_slot"]) for x in scenarios] != list(range(16)):
      raise ValueError("chunk must contain native slots 0..15 exactly")
    arrays = payload.get("arrays") or {}
    shapes = {
      "actor_observation": ((16, 2400, 234), torch.float32),
      "critic_contact": ((16, 2400, 4), torch.bool),
      "sensor_contact": ((16, 2400, 4), torch.bool),
      "clearance": ((16, 2400, 4), torch.float32),
      "clearance_valid": ((16, 2400, 4), torch.bool),
      "pre_progress": ((16, 2400), torch.float32),
      "pre_episode_tick": ((16, 2400), torch.int32),
      "anchor_active": ((16, 2400), torch.bool),
      "post_contact": ((16, 2400, 4), torch.bool),
      "post_force_w": ((16, 2400, 4, 3), torch.float32),
      "post_foot_velocity_w": ((16, 2400, 4, 3), torch.float32),
      "post_terrain_normal_w": ((16, 2400, 4, 3), torch.float32),
      "post_ray_valid": ((16, 2400, 4), torch.bool),
      "post_progress": ((16, 2400), torch.float32),
      "post_episode_tick": ((16, 2400), torch.int32),
      "done": ((16, 2400), torch.bool),
      "catastrophic": ((16, 2400), torch.bool),
      "termination_code": ((16, 2400), torch.int16),
      "route_completed": ((16, 2400), torch.bool),
    }
    for name, (shape, dtype) in shapes.items():
      _require_array(arrays, name, shape, dtype)
    reset_count = payload.get("reset_count")
    if not isinstance(reset_count, torch.Tensor) or reset_count.shape != (16,) or bool((reset_count > 1).any()):
      raise ValueError("reset_count must be shape 16 and <=1")
    reset_total += int(reset_count.sum())
    raw_rows += int(arrays["anchor_active"].sum())
    for env_index in range(16):
      row, audit = _trajectory_rows(payload, env_index)
      count = row["count"]
      actor.append(row["actor"]); contact.append(row["contact"])
      clearance.append(row["clearance"]); clearance_valid.append(row["clearance_valid"])
      seeds.append(torch.full((count,), int(payload["seed"]), dtype=torch.long))
      slots.append(torch.full((count,), env_index, dtype=torch.long))
      speed = float(payload["scenarios"][env_index]["speed"])
      speeds.append(torch.full((count,), speed, dtype=torch.float64))
      profiles.extend([str(payload["profile"])] * count)
      routes.extend([str(payload["route_kind"])] * count)
      for horizon in HORIZONS:
        for target in TARGETS:
          targets[horizon][target].append(row["labels"][horizon][target])
          valid[horizon][target].append(row["valid"][horizon][target])
      key = (audit["profile"], audit["speed"])
      transition[key] += torch.tensor(audit["raw_transition_edges"])
      isolated[key] += torch.tensor(audit["isolated_excursion_edges"])
      trajectory_count += 1
  dataset = RowDataset(
    actor=torch.cat(actor), contact=torch.cat(contact), clearance=torch.cat(clearance),
    clearance_valid=torch.cat(clearance_valid), seed=torch.cat(seeds),
    slot=torch.cat(slots), speed=torch.cat(speeds), profile=profiles, route=routes,
    targets={h: {t: torch.cat(v) for t, v in values.items()} for h, values in targets.items()},
    valid={h: {t: torch.cat(v) for t, v in values.items()} for h, values in valid.items()},
  )
  if dataset.actor.shape[1] != 234 or not torch.isfinite(dataset.actor).all():
    raise ValueError("paired actor rows are nonfinite or not 234D")
  if not torch.isfinite(dataset.contact).all() or not bool(((dataset.contact == 0) | (dataset.contact == 1)).all()):
    raise ValueError("contact feature is not finite binary")
  if not torch.isfinite(dataset.clearance[dataset.clearance_valid]).all():
    raise ValueError("all-four-ray-valid clearance feature contains nonfinite values")
  audit = {
    "trajectory_count": trajectory_count, "raw_active_rows": raw_rows,
    "downsampled_anchor_rows": dataset.rows, "reset_total": reset_total,
    "foot_order_native": list(NATIVE_FOOT_ORDER),
    "canonical_order_record_only": ["FR", "FL", "RR", "RL"],
    "native_to_canonical_permutation_record_only": list(NATIVE_TO_CANONICAL),
    "transition_by_profile_speed": {},
  }
  for profile in PROFILES:
    for speed in SPEEDS:
      key = (profile, speed)
      fractions = torch.where(
        transition[key] > 0,
        isolated[key].double() / transition[key].double(),
        torch.zeros(4, dtype=torch.float64),
      )
      audit["transition_by_profile_speed"][f"{profile}|vx={speed:.1f}"] = {
        "raw_transition_edges_native": transition[key].tolist(),
        "isolated_excursion_edges_native": isolated[key].tolist(),
        "isolated_excursion_fraction_native": fractions.tolist(),
        "max_chatter_fraction": float(fractions.max()),
      }
  return dataset, audit


def _profile_mask(dataset: RowDataset, profile: str) -> torch.Tensor:
  return torch.tensor([value == profile for value in dataset.profile], dtype=torch.bool)


def _coverage(dataset: RowDataset, audit: dict[str, Any]) -> tuple[dict[str, Any], bool, list[str]]:
  result: dict[str, Any] = {}
  reasons: list[str] = []
  profile_masks = {profile: _profile_mask(dataset, profile) for profile in PROFILES}
  for horizon in HORIZONS:
    for profile in PROFILES:
      for speed in SPEEDS:
        name = f"{profile}|vx={speed:.1f}|H={horizon}"
        stratum = profile_masks[profile] & torch.isclose(dataset.speed, torch.tensor(speed, dtype=torch.float64))
        common = stratum & dataset.clearance_valid
        groups_all = dataset.groups(common)
        entry: dict[str, Any] = {
          "usable_clusters": len(set(groups_all)),
          "paired_anchor_rows": int(common.sum()),
          "all_four_ray_valid_fraction": float(dataset.clearance_valid[stratum].double().mean()) if bool(stratum.any()) else 0.0,
          "binary": {},
        }
        for target in BINARY_TARGETS:
          mask = common & dataset.valid[horizon][target]
          values = dataset.targets[horizon][target][mask].bool()
          groups = dataset.groups(mask)
          positives = {group for group, value in zip(groups, values.tolist(), strict=True) if value}
          negatives = {group for group, value in zip(groups, values.tolist(), strict=True) if not value}
          entry["binary"][target] = {
            "clusters": len(set(groups)), "positive_event_clusters": len(positives),
            "negative_event_clusters": len(negatives),
            "positive_anchors": int(values.sum()), "negative_anchors": int((~values).sum()),
          }
        progress_mask = common & dataset.valid[horizon]["future_progress"]
        entry["progress_clusters"] = len(set(dataset.groups(progress_mask)))
        entry["progress_anchors"] = int(progress_mask.sum())
        transition = audit["transition_by_profile_speed"][f"{profile}|vx={speed:.1f}"]
        entry["raw_transition_edges_native"] = transition["raw_transition_edges_native"]
        entry["max_chatter_fraction"] = transition["max_chatter_fraction"]
        local: list[str] = []
        if entry["usable_clusters"] < 16: local.append("usable_clusters_below_16")
        if entry["all_four_ray_valid_fraction"] < 0.99: local.append("ray_valid_fraction_below_0.99")
        if any(int(x) < 1 for x in entry["raw_transition_edges_native"]): local.append("raw_transition_missing_for_a_foot")
        if entry["max_chatter_fraction"] > 0.10: local.append("contact_chatter_above_0.10")
        for target, value in entry["binary"].items():
          for field, minimum in (("clusters", 16), ("positive_event_clusters", 8),
                                 ("positive_anchors", 200), ("negative_anchors", 200)):
            if int(value[field]) < minimum: local.append(f"{target}:{field}_below_{minimum}")
        if entry["progress_clusters"] < 16: local.append("progress_clusters_below_16")
        if entry["progress_anchors"] < 5000: local.append("progress_anchors_below_5000")
        entry["passed"] = not local
        entry["reasons"] = local
        reasons.extend(f"{name}:{reason}" for reason in local)
        result[name] = entry
  return result, not reasons, reasons


def _feature_matrix(dataset: RowDataset, feature: str, mask: torch.Tensor) -> torch.Tensor:
  base = dataset.actor[mask]
  if feature == "baseline": return base
  if feature == "contact": return torch.cat((base, dataset.contact[mask]), dim=1)
  if feature == "clearance": return torch.cat((base, dataset.clearance[mask]), dim=1)
  raise ValueError(feature)


def _fit_oof(dataset: RowDataset, horizon: int, target: str) -> dict[str, torch.Tensor]:
  eligible = dataset.clearance_valid & dataset.valid[horizon][target]
  if not bool(eligible.any()):
    raise ValueError(f"no eligible rows for {target} H{horizon}")
  indices = torch.where(eligible)[0]
  seeds = dataset.seed[eligible]
  slots = dataset.slot[eligible]
  y = dataset.targets[horizon][target][eligible]
  matrices = {
    feature: _feature_matrix(dataset, feature, eligible) for feature in FEATURES
  }
  predictions = {feature: torch.full((indices.numel(),), torch.nan, dtype=torch.float64) for feature in FEATURES}
  for held_seed in FORMAL_SEEDS:
    train = seeds != held_seed
    test = seeds == held_seed
    assert_no_group_leakage(seeds.tolist(), slots.tolist(), train, test)
    if not bool(train.any()) or not bool(test.any()):
      raise ValueError(f"leave-one-seed-out fold {held_seed} is empty")
    groups = [(int(seed), int(slot)) for seed, slot in zip(seeds[train].tolist(), slots[train].tolist(), strict=True)]
    for feature in FEATURES:
      x_train = matrices[feature][train]
      x_test = matrices[feature][test]
      if target in BINARY_TARGETS:
        weights = balanced_binary_weights(y[train].bool(), groups)
        model = _fit_logistic_lbfgs(x_train, y[train].bool(), weights)
        predictions[feature][test] = model.predict_proba(x_test)
      else:
        weights = group_balanced_weights(groups)
        model = fit_weighted_ridge(
          x_train, y[train].double(), sample_weight=weights, l2=FIXED_L2
        )
        predictions[feature][test] = model.predict(x_test)
  if any(not torch.isfinite(value).all() for value in predictions.values()):
    raise RuntimeError(f"OOF prediction is incomplete for {target} H{horizon}")
  predictions["eligible_indices"] = indices
  return predictions


@dataclass(frozen=True)
class EffectInterval:
  estimate: float
  ci_low: float
  ci_high: float
  baseline_loss: float
  candidate_loss: float
  clusters: int
  resamples: int
  seed: int


def _loss_stats(
  baseline_error: torch.Tensor, candidate_error: torch.Tensor,
  target: torch.Tensor | None,
) -> torch.Tensor:
  if target is None:
    return torch.stack((baseline_error, candidate_error, torch.ones_like(baseline_error)), dim=1)
  positive = target.bool()
  return torch.stack((
    baseline_error * positive, candidate_error * positive, positive.double(),
    baseline_error * ~positive, candidate_error * ~positive, (~positive).double(),
  ), dim=1)


def _metric_from_stats(stats: torch.Tensor, binary: bool) -> tuple[torch.Tensor, torch.Tensor]:
  if binary:
    if bool((stats[..., 2] <= 0).any()) or bool((stats[..., 5] <= 0).any()):
      raise ValueError("balanced log-loss bootstrap sample lacks one class")
    return (
      0.5 * (stats[..., 0] / stats[..., 2] + stats[..., 3] / stats[..., 5]),
      0.5 * (stats[..., 1] / stats[..., 2] + stats[..., 4] / stats[..., 5]),
    )
  if bool((stats[..., 2] <= 0).any()):
    raise ValueError("progress bootstrap sample is empty")
  return stats[..., 0] / stats[..., 2], stats[..., 1] / stats[..., 2]


def _bootstrap_cluster_sums(
  matrix: torch.Tensor, *, generator: torch.Generator, binary: bool,
) -> torch.Tensor:
  """Draw cluster sums, conditionally redrawing class-degenerate replicates."""
  ids = torch.randint(
    matrix.shape[0], (BOOTSTRAP_RESAMPLES, matrix.shape[0]), generator=generator
  )
  sampled = matrix[ids].sum(dim=1)
  if not binary:
    return sampled
  invalid = (sampled[:, 2] <= 0) | (sampled[:, 5] <= 0)
  attempts = 0
  while bool(invalid.any()):
    attempts += 1
    if attempts > 1000:
      raise RuntimeError("could not obtain class-complete cluster bootstrap draws")
    count = int(invalid.sum())
    replacement_ids = torch.randint(
      matrix.shape[0], (count, matrix.shape[0]), generator=generator
    )
    sampled[invalid] = matrix[replacement_ids].sum(dim=1)
    invalid = (sampled[:, 2] <= 0) | (sampled[:, 5] <= 0)
  return sampled


def _bootstrap_effect(
  baseline_error: torch.Tensor, candidate_error: torch.Tensor,
  clusters: Sequence[tuple[int, int]], speeds: Sequence[float],
  *, target: torch.Tensor | None,
) -> tuple[EffectInterval, torch.Tensor]:
  if baseline_error.numel() == 0 or baseline_error.shape != candidate_error.shape:
    raise ValueError("paired bootstrap errors are empty or misaligned")
  if len(clusters) != baseline_error.numel() or len(speeds) != baseline_error.numel():
    raise ValueError("bootstrap metadata is misaligned")
  row_stats = _loss_stats(baseline_error.double(), candidate_error.double(), target)
  grouped: dict[tuple[float, tuple[int, int]], torch.Tensor] = {}
  cluster_speed: dict[tuple[int, int], float] = {}
  for index, (cluster, speed) in enumerate(zip(clusters, speeds, strict=True)):
    previous_speed = cluster_speed.setdefault(cluster, float(speed))
    if previous_speed != float(speed):
      raise ValueError("one (seed, slot) bootstrap cluster appears in multiple speeds")
    key = (float(speed), cluster)
    grouped[key] = grouped.get(key, torch.zeros(row_stats.shape[1], dtype=torch.float64)) + row_stats[index]
  by_speed: dict[float, list[torch.Tensor]] = {}
  for (speed, _), stats in sorted(grouped.items()):
    by_speed.setdefault(speed, []).append(stats)
  generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
  original_base: list[torch.Tensor] = []
  original_candidate: list[torch.Tensor] = []
  draw_base: list[torch.Tensor] = []
  draw_candidate: list[torch.Tensor] = []
  for speed in sorted(by_speed):
    matrix = torch.stack(by_speed[speed])
    base, candidate = _metric_from_stats(matrix.sum(dim=0), target is not None)
    original_base.append(base); original_candidate.append(candidate)
    sampled = _bootstrap_cluster_sums(
      matrix, generator=generator, binary=target is not None
    )
    base_draw, candidate_draw = _metric_from_stats(sampled, target is not None)
    draw_base.append(base_draw); draw_candidate.append(candidate_draw)
  base = torch.stack(original_base).mean()
  candidate = torch.stack(original_candidate).mean()
  if float(base) <= 0.0:
    raise ValueError("baseline loss must be positive for relative effect")
  draws = (torch.stack(draw_base).mean(dim=0) - torch.stack(draw_candidate).mean(dim=0)) / torch.stack(draw_base).mean(dim=0)
  estimate = float((base - candidate) / base)
  interval = EffectInterval(
    estimate=estimate, ci_low=float(torch.quantile(draws, 0.025)),
    ci_high=float(torch.quantile(draws, 0.975)), baseline_loss=float(base),
    candidate_loss=float(candidate), clusters=len({cluster for cluster in clusters}),
    resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
  )
  return interval, draws


def _score_scope(
  dataset: RowDataset, horizon: int, target_name: str,
  predictions: dict[str, torch.Tensor], scope: torch.Tensor, feature: str,
) -> tuple[dict[str, Any], torch.Tensor]:
  original = predictions["eligible_indices"]
  keep = scope[original]
  y = dataset.targets[horizon][target_name][original][keep]
  base_pred = predictions["baseline"][keep]
  candidate_pred = predictions[feature][keep]
  if target_name in BINARY_TARGETS:
    y_bool = y.bool()
    base_error = -(y_bool * torch.log(base_pred.clamp(1e-12, 1 - 1e-12)) + (~y_bool) * torch.log1p(-base_pred.clamp(1e-12, 1 - 1e-12)))
    candidate_error = -(y_bool * torch.log(candidate_pred.clamp(1e-12, 1 - 1e-12)) + (~y_bool) * torch.log1p(-candidate_pred.clamp(1e-12, 1 - 1e-12)))
    binary_target: torch.Tensor | None = y_bool
  else:
    base_error = (y.double() - base_pred).abs()
    candidate_error = (y.double() - candidate_pred).abs()
    binary_target = None
  selected = original[keep]
  clusters = [(int(dataset.seed[i]), int(dataset.slot[i])) for i in selected.tolist()]
  speeds = dataset.speed[selected].tolist()
  interval, draws = _bootstrap_effect(base_error, candidate_error, clusters, speeds, target=binary_target)
  result: dict[str, Any] = asdict(interval)
  result["loss"] = "balanced_log_loss" if binary_target is not None else "normalized_mae"
  if binary_target is not None:
    score_weight = group_balanced_weights(clusters)
    result["baseline_pr_auc"] = weighted_pr_auc(y_bool, base_pred, sample_weight=score_weight)
    result["candidate_pr_auc"] = weighted_pr_auc(y_bool, candidate_pred, sample_weight=score_weight)
    result["pr_auc_delta"] = result["candidate_pr_auc"] - result["baseline_pr_auc"]
  return result, draws


def _models_and_metrics(dataset: RowDataset) -> tuple[dict[str, Any], dict[str, Any]]:
  metrics: dict[str, Any] = {feature: {} for feature in ("contact", "clearance")}
  draws: dict[str, Any] = {feature: {} for feature in ("contact", "clearance")}
  profile_masks = {profile: _profile_mask(dataset, profile) for profile in PROFILES}
  all_rows = torch.ones(dataset.rows, dtype=torch.bool)
  for horizon in HORIZONS:
    for target in TARGETS:
      prediction = _fit_oof(dataset, horizon, target)
      for feature in ("contact", "clearance"):
        endpoint = metrics[feature].setdefault(target, {}).setdefault(str(horizon), {"by_stratum": {}})
        endpoint_draws = draws[feature].setdefault(target, {}).setdefault(str(horizon), {"by_stratum": {}})
        macro, macro_draw = _score_scope(dataset, horizon, target, prediction, all_rows, feature)
        endpoint["macro"] = macro; endpoint_draws["macro"] = macro_draw
        for profile in PROFILES:
          for speed in SPEEDS:
            key = f"{profile}|vx={speed:.1f}"
            scope = profile_masks[profile] & torch.isclose(dataset.speed, torch.tensor(speed, dtype=torch.float64))
            value, value_draws = _score_scope(dataset, horizon, target, prediction, scope, feature)
            endpoint["by_stratum"][key] = value
            endpoint_draws["by_stratum"][key] = value_draws
  return metrics, draws


def _mean_interval(values: Sequence[dict[str, Any]], draw_values: Sequence[torch.Tensor]) -> dict[str, Any]:
  draws = torch.stack(tuple(draw_values)).mean(dim=0)
  return {
    "estimate": sum(float(value["estimate"]) for value in values) / len(values),
    "ci_low": float(torch.quantile(draws, 0.025)),
    "ci_high": float(torch.quantile(draws, 0.975)),
    "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
  }


def _apply_pass_gates(metrics: dict[str, Any], draws: dict[str, Any]) -> dict[str, Any]:
  reasons: list[str] = []
  checks: dict[str, Any] = {}
  contact = metrics["contact"]
  contact_draws = draws["contact"]
  for profile in PROFILES:
    for speed in SPEEDS:
      stratum = f"{profile}|vx={speed:.1f}"
      values = [contact[target]["25"]["by_stratum"][stratum] for target in TARGETS]
      samples = [contact_draws[target]["25"]["by_stratum"][stratum] for target in TARGETS]
      composite = _mean_interval(values, samples)
      checks[f"primary_H25_composite:{stratum}"] = composite
      if composite["estimate"] <= 0 or composite["ci_low"] <= 0:
        reasons.append(f"primary_H25_composite_no_positive_ci:{stratum}")
  for horizon in (10, 50):
    stratum_values = []
    stratum_draws = []
    for profile in PROFILES:
      for speed in SPEEDS:
        stratum = f"{profile}|vx={speed:.1f}"
        values = [contact[target][str(horizon)]["by_stratum"][stratum] for target in TARGETS]
        samples = [contact_draws[target][str(horizon)]["by_stratum"][stratum] for target in TARGETS]
        composite = _mean_interval(values, samples)
        checks[f"secondary_H{horizon}_composite:{stratum}"] = composite
        stratum_values.append(composite); stratum_draws.append(torch.stack(samples).mean(dim=0))
        if composite["estimate"] <= 0:
          reasons.append(f"H{horizon}_composite_point_not_positive:{stratum}")
    macro = _mean_interval(stratum_values, stratum_draws)
    checks[f"secondary_H{horizon}_composite_macro"] = macro
    if macro["ci_low"] <= 0:
      reasons.append(f"H{horizon}_composite_macro_ci_not_positive")
  for target in TARGETS:
    value = contact[target]["25"]["macro"]
    checks[f"H25_target_macro:{target}"] = value
    if value["ci_low"] <= 0:
      reasons.append(f"H25_target_macro_ci_not_positive:{target}")
  for target in ("slip_onset", "unexpected_transition"):
    for horizon in HORIZONS:
      value = contact[target][str(horizon)]["macro"]
      checks[f"event_macro:{target}:H{horizon}"] = value
      if value["ci_low"] <= 0:
        reasons.append(f"event_macro_ci_not_positive:{target}:H{horizon}")
  for target in TARGETS:
    for horizon in HORIZONS:
      for stratum, value in contact[target][str(horizon)]["by_stratum"].items():
        if value["estimate"] < 0:
          reasons.append(f"negative_point_gain:{target}:H{horizon}:{stratum}")
        if target in BINARY_TARGETS and value["pr_auc_delta"] < 0:
          reasons.append(f"binary_pr_auc_decrease:{target}:H{horizon}:{stratum}")
        cdraw = contact_draws[target][str(horizon)]["by_stratum"][stratum]
        qdraw = draws["clearance"][target][str(horizon)]["by_stratum"][stratum]
        difference = cdraw - qdraw
        comparison = {
          "contact_minus_clearance_estimate": value["estimate"] - metrics["clearance"][target][str(horizon)]["by_stratum"][stratum]["estimate"],
          "ci_low": float(torch.quantile(difference, 0.025)),
          "ci_high": float(torch.quantile(difference, 0.975)),
        }
        checks[f"contact_vs_clearance:{target}:H{horizon}:{stratum}"] = comparison
        if comparison["ci_high"] < 0:
          reasons.append(f"clearance_significantly_dominates_contact:{target}:H{horizon}:{stratum}")
  return {"passed": not reasons, "reasons": reasons, "checks": checks}


def analyze(raw_dir: Path) -> dict[str, Any]:
  manifest, chunks = _load_manifest(raw_dir)
  dataset, audit = _build_dataset(chunks)
  if audit["trajectory_count"] != 288:
    raise ValueError("trajectory inventory is not exactly 288")
  coverage, coverage_ok, coverage_reasons = _coverage(dataset, audit)
  summary: dict[str, Any] = {
    "schema_version": 1,
    "evaluation_suite": "go2_foot_contact_observability_analysis_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "decision_scope": "evaluation_only_gate_before_any_238d_teacher_training",
    "manifest": {
      "path": manifest["manifest_path"], "sha256": manifest["manifest_sha256"],
      "contract_sha256": manifest["validated_contract_sha256"],
      "collector_sha256": manifest["collector_sha256"], "chunk_count": 18,
      "trajectory_count": 288, "observed_raw_rows": manifest["observed_raw_rows"],
    },
    "analysis_contract": {
      "device": "cpu", "state_timeline": "pre0_plus_post",
      "native_foot_order": list(NATIVE_FOOT_ORDER),
      "canonical_permutation_record_only": list(NATIVE_TO_CANONICAL),
      "anchor_start_tick": ANCHOR_START, "anchor_stride_ticks": ANCHOR_STRIDE,
      "horizons_ticks": list(HORIZONS), "outer_split": "exact_leave_one_seed_out",
      "group": ["seed", "matched_slot"], "fixed_l2": FIXED_L2,
      "binary_model": "l2_logistic_torch_lbfgs_deterministic_float64_cpu",
      "progress_model": "ridge_torch_linalg_solve_float64_cpu",
      "bootstrap_unit": ["seed", "matched_slot"], "bootstrap_stratify_by": "speed",
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
      "paired_complete_case": "baseline/contact/clearance share all-four-ray-valid rows",
    },
    "raw_audit": audit, "coverage": coverage,
    "coverage_passed": coverage_ok, "coverage_reasons": coverage_reasons,
    "eligible_rows_by_target_horizon": {
      str(h): {t: int((dataset.clearance_valid & dataset.valid[h][t]).sum()) for t in TARGETS}
      for h in HORIZONS
    },
    "training_changed": False, "learn_called": False,
    "technical_failures": [],
  }
  if not coverage_ok:
    summary.update({
      "analysis_status": "coverage_inconclusive", "metrics": {}, "pass_gates": {},
      "observability_diagnostic_passed": False,
      "decision": "INCONCLUSIVE_DO_NOT_TRAIN",
      "decision_reasons": ["coverage_failed", *coverage_reasons],
    })
    return summary
  try:
    metrics, bootstrap_draws = _models_and_metrics(dataset)
    gates = _apply_pass_gates(metrics, bootstrap_draws)
  except Exception as error:
    failure = {
      "type": type(error).__name__, "message": str(error),
      "traceback": traceback.format_exc(),
    }
    summary.update({
      "analysis_status": "technical_failure", "metrics": {}, "pass_gates": {},
      "technical_failures": [failure],
      "observability_diagnostic_passed": False,
      "decision": "INCONCLUSIVE_DO_NOT_TRAIN",
      "decision_reasons": ["model_or_metric_technical_failure"],
    })
    return summary
  summary.update({
    "analysis_status": "completed", "metrics": metrics, "pass_gates": gates,
    "observability_diagnostic_passed": gates["passed"],
    "decision": "OBSERVABILITY_DIAGNOSTIC_PASSED" if gates["passed"] else "INCONCLUSIVE_DO_NOT_TRAIN",
    "decision_reasons": gates["reasons"],
  })
  return summary


def _markdown(summary: dict[str, Any]) -> str:
  lines = [
    "# Go2 foot-contact observability diagnostic", "",
    f"- Decision: `{summary['decision']}`",
    f"- Analysis status: `{summary['analysis_status']}`",
    f"- Manifest SHA256: `{summary.get('manifest', {}).get('sha256', 'unavailable')}`",
    f"- Contract SHA256: `{summary.get('manifest', {}).get('contract_sha256', 'unavailable')}`",
    "- Training changed: `false`; `learn()` called: `false`", "",
  ]
  failures = summary.get("technical_failures", [])
  if failures:
    lines.extend(("## Technical failures", ""))
    lines.extend(f"- `{item['type']}`: {item['message']}" for item in failures)
    lines.append("")
  reasons = summary.get("decision_reasons", [])
  lines.extend(("## Decision reasons", ""))
  lines.extend(["- None."] if not reasons else [f"- {reason}" for reason in reasons])
  lines.extend(("", "## Coverage", "", "| profile / speed / H | pass | clusters | slip + | unexpected + | failure + | progress rows | rays |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
  for name, value in summary.get("coverage", {}).items():
    binary = value["binary"]
    lines.append(
      f"| {name} | {value['passed']} | {value['usable_clusters']} | "
      f"{binary['slip_onset']['positive_anchors']} | "
      f"{binary['unexpected_transition']['positive_anchors']} | "
      f"{binary['catastrophic_failure']['positive_anchors']} | "
      f"{value['progress_anchors']} | {value['all_four_ray_valid_fraction']:.4f} |"
    )
  lines.extend(("", "## Interpretation", ""))
  if summary["decision"] == "OBSERVABILITY_DIAGNOSTIC_PASSED":
    lines.append("The preregistered incremental-observability gates passed; this artifact only authorizes considering the 238D teacher probe, not training by itself.")
  else:
    lines.append("The evidence is inconclusive under the frozen gate. Do not start the 238D teacher training from this result.")
  return "\n".join(lines) + "\n"


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
  if args.output_json is not None or args.output_md is not None:
    if args.output_json is None or args.output_md is None:
      raise ValueError("--output-json and --output-md must be supplied together")
    return args.output_json.expanduser().resolve(), args.output_md.expanduser().resolve()
  prefix = args.output_prefix.expanduser().resolve()
  return Path(f"{prefix}.json"), Path(f"{prefix}.md")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
  parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument("--torch-threads", type=int, default=4)
  args = parser.parse_args()
  if args.torch_threads <= 0:
    parser.error("--torch-threads must be positive")
  output_json, output_md = _output_paths(args)
  if output_json.exists() or output_md.exists():
    raise FileExistsError(f"refusing to overwrite summary: {output_json} / {output_md}")
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_md.parent.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(args.torch_threads)
  torch.set_num_interop_threads(1)
  technical = False
  try:
    summary = analyze(args.raw_dir.expanduser().resolve())
    technical = summary.get("analysis_status") == "technical_failure"
  except Exception as error:  # Fail closed, but always leave an auditable summary.
    technical = True
    summary = {
      "schema_version": 1,
      "evaluation_suite": "go2_foot_contact_observability_analysis_v1",
      "created_at": datetime.now().astimezone().isoformat(),
      "analysis_status": "technical_failure", "training_changed": False,
      "learn_called": False, "coverage": {}, "coverage_passed": False,
      "observability_diagnostic_passed": False,
      "decision": "INCONCLUSIVE_DO_NOT_TRAIN",
      "decision_reasons": ["technical_failure"],
      "technical_failures": [{
        "type": type(error).__name__, "message": str(error),
        "traceback": traceback.format_exc(),
      }],
    }
  content = json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n"
  with output_json.open("x", encoding="utf-8") as stream:
    stream.write(content)
  with output_md.open("x", encoding="utf-8") as stream:
    stream.write(_markdown(summary))
  print(f"WROTE {output_json}")
  print(f"WROTE {output_md}")
  print(f"DECISION {summary['decision']}")
  if technical:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
