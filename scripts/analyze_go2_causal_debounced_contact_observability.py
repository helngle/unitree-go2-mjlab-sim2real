"""Analyze the preregistered causal debounced-contact observability suite.

This CPU-only analyzer accepts exactly the V2 24-chunk formal inventory.  It
independently replays the causal two-tick filter, constructs one paired row set
for the 234D baseline and 238D candidate, evaluates exact four-fold
leave-one-seed-out models, and fails closed on any integrity, coverage, model,
or pass-gate violation.
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

from src.tasks.velocity.evaluation.debounced_contact_observability import (  # noqa: E402
  recompute_causal_confirmed_contact,
)
from src.tasks.velocity.evaluation.foot_contact_observability import (  # noqa: E402
  NATIVE_FOOT_GEOMS,
  NATIVE_FOOT_ORDER,
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


CONTRACT = ROOT / "docs/reviews/go2_causal_debounced_contact_observability_contract_20260809.json"
COLLECTOR = ROOT / "scripts/collect_go2_causal_debounced_contact_observability.py"
FILTER = ROOT / "src/tasks/velocity/evaluation/debounced_contact_observability.py"
DEFAULT_RAW_DIR = ROOT / "docs/reviews/go2_causal_debounced_contact_observability_raw_20260809_v2"
DEFAULT_OUTPUT_PREFIX = ROOT / "docs/reviews/go2_causal_debounced_contact_observability_summary_20260809_v2"

SEEDS = (1042, 1043, 1044, 1045)
PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")
SPEEDS = (0.3, 0.5)
HORIZONS = (10, 25, 50)
TARGETS = ("slip_onset", "unexpected_transition", "catastrophic_failure", "future_progress")
BINARY_TARGETS = TARGETS[:3]
FEATURES = ("baseline", "candidate")
CONTROL_DT = 0.02
ANCHOR_START = 10
ANCHOR_STRIDE = 5
FIXED_L2 = 0.01
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_810
RAW_SUITE = "go2_causal_debounced_contact_observability_raw_chunk_v2"
MANIFEST_SUITE = "go2_causal_debounced_contact_observability_raw_manifest_v2"
ANALYSIS_SUITE = "go2_causal_debounced_contact_observability_analysis_v2"
EXPECTED_CONTRACT_SHA256 = "033ed364510dd6b77b7b60160287374bfb2b7e3b49be52516a483bc10ca54240"
EXPECTED_FILTER_SHA256 = "ddbea37832ada56ecac7dd153b4c35eb604609f5222a0f05e9c5e555814d1ab2"

EXPECTED_INVARIANTS = {
  "training_changed": False,
  "learn_called": False,
  "checkpoint_sha256_verified": True,
  "actor_critic_action_dims_verified": True,
  "runtime_contact_order_exact": True,
  "critic_contact_equals_native_sensor": True,
  "command_cache_patch_exact": True,
  "initial_post_placement_observation_refreshes": 1,
  "external_observation_recompute_inside_loop": False,
  "terminal_state_captured_inside_reset_hook": True,
  "post_reset_rows_excluded": True,
  "confirmed_online_replay_bitwise_equal": True,
  "confirmed_early_or_backfill_count": 0,
  "confirmed_update_before_policy": True,
  "confirmed_reset_isolated": True,
  "recursive_finite": True,
}


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
    "schema_version": 2,
    "evaluation_suite": "go2_causal_debounced_contact_incremental_observability_v2",
    "source.actor_dim": 234,
    "source.critic_dim": 261,
    "source.critic_raw_contact_slice": [245, 249],
    "single_variable.candidate_dim": 238,
    "matrix.seeds": list(SEEDS),
    "matrix.profiles": list(PROFILES),
    "matrix.route_kinds": list(ROUTES),
    "matrix.matched_slots_per_chunk": 16,
    "matrix.steps_per_chunk": 2400,
    "matrix.chunk_count": 24,
    "matrix.trajectory_count": 384,
    "timing.anchor_start_tick": ANCHOR_START,
    "timing.anchor_stride_ticks": ANCHOR_STRIDE,
    "timing.horizons_ticks": list(HORIZONS),
    "foot_semantics.runtime_sensor_order": list(NATIVE_FOOT_ORDER),
    "foot_semantics.runtime_geom_names": list(NATIVE_FOOT_GEOMS),
    "foot_semantics.confirmation_ticks": 2,
    "foot_semantics.future_backfill_forbidden": True,
    "features.common_scoring_rows": "anchor_active_and_label_valid_and_confirmed_contact_valid",
    "cross_validation.outer_folds": 4,
    "cross_validation.outer_split": "leave_one_seed_out",
    "cross_validation.fixed_l2": FIXED_L2,
    "bootstrap.resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap.seed": BOOTSTRAP_SEED,
    "artifact_inventory.raw_chunks_exact": 24,
    "artifact_inventory.trajectories_exact": 384,
    "artifact_inventory.unique_cv_groups_exact": 64,
  }
  mismatches = {
    path: {"actual": _contract_value(contract, path), "expected": wanted}
    for path, wanted in expected.items()
    if _contract_value(contract, path) != wanted
  }
  if mismatches:
    raise ValueError(f"machine contract differs from analyzer freeze: {mismatches}")
  forbidden = set(contract["single_variable"]["forbidden_candidate_features"])
  if (
    "clearance" not in forbidden
    or contract["single_variable"]["clearance_collection"]
    != "integrity_only_not_a_model_feature_or_row_mask"
  ):
    raise ValueError("clearance must remain integrity-only and excluded from models/masks")


def _resolve_inventory_path(value: str, manifest_path: Path) -> Path:
  path = Path(value).expanduser()
  if path.is_absolute():
    return path.resolve()
  local = (manifest_path.parent / path).resolve()
  return local if local.exists() else (ROOT / path).resolve()


def _filter_sha(payload: dict[str, Any]) -> Any:
  if "causal_filter_sha256" not in payload:
    raise ValueError("payload lacks exact causal_filter_sha256 field")
  return payload["causal_filter_sha256"]


def _load_manifest(raw_dir: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
  manifest_path = raw_dir / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"formal V2 manifest is required: {manifest_path}")
  if raw_dir.resolve() == (ROOT / "docs/reviews/go2_foot_contact_observability_raw_20260809").resolve():
    raise ValueError("V1 raw directory is forbidden as V2 formal evidence")
  contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
  _validate_contract(contract)
  contract_sha = _sha256(CONTRACT)
  collector_sha = _sha256(COLLECTOR)
  filter_sha = _sha256(FILTER)
  if contract_sha != EXPECTED_CONTRACT_SHA256:
    raise ValueError("V2 contract SHA differs from the preregistered analyzer freeze")
  if filter_sha != EXPECTED_FILTER_SHA256:
    raise ValueError("causal-filter SHA differs from the preregistered analyzer freeze")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  expected = {
    "schema_version": 2,
    "evaluation_suite": MANIFEST_SUITE,
    "contract_sha256": contract_sha,
    "collector_sha256": collector_sha,
    "chunk_count": 24,
    "trajectory_count": 384,
    "training_changed": False,
    "learn_called": False,
  }
  mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
  if mismatch:
    raise ValueError(f"formal V2 manifest identity mismatch: {mismatch}")
  if _filter_sha(manifest) != filter_sha:
    raise ValueError("manifest causal-filter SHA differs from frozen filter")
  inventory = manifest.get("chunks")
  if not isinstance(inventory, list) or len(inventory) != 24:
    raise ValueError("manifest must inventory exactly 24 chunks")
  expected_ids = {(seed, profile, route) for seed in SEEDS for profile in PROFILES for route in ROUTES}
  actual_ids = {(item.get("seed"), item.get("profile"), item.get("route_kind")) for item in inventory}
  if actual_ids != expected_ids or len(actual_ids) != len(inventory):
    raise ValueError(f"manifest chunk identities differ: {actual_ids ^ expected_ids}")
  loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
  observed_rows = 0
  for item in sorted(inventory, key=lambda value: (value["seed"], value["profile"], value["route_kind"])):
    path = _resolve_inventory_path(str(item["path"]), manifest_path)
    registered = (raw_dir / f"seed_{item['seed']}" / item["profile"] / f"{item['route_kind']}.pt").resolve()
    if path != registered or not path.is_file():
      raise ValueError(f"inventoried chunk path is missing or outside V2 layout: {path}")
    if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
      raise ValueError(f"inventoried chunk size/SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    identity = {
      "schema_version": 2,
      "evaluation_suite": RAW_SUITE,
      "mode": "formal",
      "seed": item["seed"],
      "profile": item["profile"],
      "route_kind": item["route_kind"],
      "task_id": contract["source"]["task_id"],
      "num_envs": 16,
      "steps_requested": 2400,
      "contract_sha256": contract_sha,
      "collector_sha256": collector_sha,
      "checkpoint_sha256": contract["source"]["checkpoint_sha256"],
    }
    bad = {key: (payload.get(key), value) for key, value in identity.items() if payload.get(key) != value}
    if bad or _filter_sha(payload) != filter_sha:
      raise ValueError(f"chunk frozen identity mismatch in {path}: {bad}")
    checkpoint = Path(str(payload.get("checkpoint", ""))).expanduser().resolve()
    expected_checkpoint = (ROOT / contract["source"]["checkpoint"]).resolve()
    if checkpoint != expected_checkpoint:
      raise ValueError(f"chunk checkpoint path differs in {path}")
    if payload.get("invariants") != EXPECTED_INVARIANTS:
      raise ValueError(f"chunk invariant dictionary differs in {path}")
    active = (payload.get("arrays") or {}).get("anchor_active")
    resets = payload.get("reset_count")
    if not isinstance(active, torch.Tensor) or active.shape != (16, 2400):
      raise ValueError(f"chunk active mask shape differs in {path}")
    if not isinstance(resets, torch.Tensor) or resets.shape != (16,) or bool((resets > 1).any()):
      raise ValueError(f"chunk reset count differs in {path}")
    raw_rows = int(active.sum())
    counters = {
      "raw_rows": raw_rows,
      "resets": int(resets.sum()),
      "steps_executed": int(payload["steps_executed"]),
    }
    if any(int(item[key]) != value for key, value in counters.items()):
      raise ValueError(f"chunk counters differ from manifest: {path}")
    observed_rows += raw_rows
    loaded.append((item, payload))
  if observed_rows != int(manifest.get("observed_raw_rows", -1)) or observed_rows > 921_600:
    raise ValueError("manifest observed row count differs or exceeds registered maximum")
  manifest["manifest_path"] = str(manifest_path)
  manifest["manifest_sha256"] = _sha256(manifest_path)
  manifest["validated_filter_sha256"] = filter_sha
  return manifest, loaded


@dataclass
class RowDataset:
  actor: torch.Tensor
  confirmed: torch.Tensor
  confirmed_valid: torch.Tensor
  seed: torch.Tensor
  slot: torch.Tensor
  tick: torch.Tensor
  speed: torch.Tensor
  profile: list[str]
  route: list[str]
  trajectory: list[tuple[int, str, str, int]]
  targets: dict[int, dict[str, torch.Tensor]]
  valid: dict[int, dict[str, torch.Tensor]]

  @property
  def rows(self) -> int:
    return int(self.actor.shape[0])

  def groups(self, mask: torch.Tensor) -> list[tuple[int, int]]:
    ids = torch.where(mask)[0].tolist()
    return [(int(self.seed[index]), int(self.slot[index])) for index in ids]

  def trajectories(self, mask: torch.Tensor) -> list[tuple[int, str, str, int]]:
    return [self.trajectory[index] for index in torch.where(mask)[0].tolist()]


def _require_array(arrays: dict[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
  value = arrays.get(name)
  if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
    raise ValueError(f"raw array {name} must be {dtype} shape {shape}")
  return value


def _scheduled_stance(ticks: torch.Tensor) -> torch.Tensor:
  offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), dtype=torch.float64)
  phase = (ticks.double()[:, None] * CONTROL_DT / 0.6 + offsets) % 1.0
  return phase < 0.56


def _transition_counts(signal: torch.Tensor) -> torch.Tensor:
  if signal.shape[0] < 2:
    return torch.zeros(4, dtype=torch.long)
  return (signal[1:] != signal[:-1]).sum(dim=0).long()


def _near_flip_fraction(changed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  flips = changed.sum(dim=0).long()
  near = torch.zeros(4, dtype=torch.long)
  for tick in range(changed.shape[0]):
    if tick + 1 < changed.shape[0]:
      near += (changed[tick] & changed[tick + 1]).long()
    if tick + 2 < changed.shape[0]:
      near += (changed[tick] & ~changed[tick + 1] & changed[tick + 2]).long()
  return flips, near


def _trajectory_rows(payload: dict[str, Any], env_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
  arrays = payload["arrays"]
  active = arrays["anchor_active"][env_index]
  n = int(active.sum())
  if n <= ANCHOR_START or not bool(active[:n].all()) or bool(active[n:].any()):
    raise ValueError("anchor_active must be one frozen nonempty prefix")
  scenario = payload["scenarios"][env_index]
  if int(scenario["matched_slot"]) != env_index:
    raise ValueError("scenario order differs from native matched-slot order")

  raw = arrays["sensor_contact"][env_index, :n].bool()
  critic = arrays["critic_contact"][env_index, :n].bool()
  saved = arrays["confirmed_contact"][env_index, :n].bool()
  saved_valid4 = arrays["confirmed_contact_valid"][env_index, :n].bool()
  saved_changed = arrays["confirmed_contact_changed"][env_index, :n].bool()
  if not torch.equal(raw, critic):
    raise ValueError("critic raw contact differs from native sensor contact")
  replay = recompute_causal_confirmed_contact(
    raw,
    attempt_id=torch.zeros(n, dtype=torch.long),
    episode_start=torch.arange(n) == 0,
  )
  replay_valid4 = replay.valid[:, None].expand(-1, 4)
  if (
    not torch.equal(saved, replay.contact)
    or not torch.equal(saved_changed, replay.changed)
    or not torch.equal(saved_valid4, replay_valid4)
  ):
    raise ValueError("saved confirmed contact differs from independent causal replay")
  if not bool(saved_valid4.all()):
    raise ValueError("active causal confirmed-contact row is invalid")
  expected_changed = torch.zeros_like(saved_changed)
  expected_changed[1:] = (raw[1:] == raw[:-1]) & (raw[1:] != saved[:-1])
  early_or_backfill = int((saved_changed != expected_changed).sum())
  if early_or_backfill != 0:
    raise ValueError("confirmed contact contains an early flip or future backfill")

  pre_ticks = arrays["pre_episode_tick"][env_index, :n]
  post_ticks = arrays["post_episode_tick"][env_index, :n]
  if n > 1 and not torch.equal(pre_ticks[1:], post_ticks[:-1]):
    raise ValueError("pre0+post state ticks do not join exactly")
  ticks = torch.cat((pre_ticks[:1], post_ticks))
  if bool((ticks < 0).any()):
    raise ValueError("valid state timeline contains negative episode tick")

  post_raw = arrays["post_contact"][env_index, :n].bool()
  contact_timeline = torch.cat((raw[:1], post_raw), dim=0)
  progress = torch.cat((arrays["pre_progress"][env_index, :1], arrays["post_progress"][env_index, :n])).double()
  force = arrays["post_force_w"][env_index, :n].double()
  velocity = arrays["post_foot_velocity_w"][env_index, :n].double()
  normal = arrays["post_terrain_normal_w"][env_index, :n].double()
  if not all(torch.isfinite(value).all() for value in (progress, force, velocity, normal)):
    raise ValueError("post-state label arrays are nonfinite")
  normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-12)
  normal_force = (force * normal).sum(dim=-1).abs()
  loaded_post = post_raw & (normal_force >= 15.0)
  tangent_velocity = velocity - (velocity * normal).sum(dim=-1, keepdim=True) * normal
  slip_post = torch.linalg.vector_norm(tangent_velocity, dim=-1)
  loaded = torch.cat((torch.zeros(1, 4, dtype=torch.bool), loaded_post), dim=0)
  slip = torch.cat((torch.zeros(1, 4, dtype=torch.float64), slip_post), dim=0)

  done = arrays["done"][env_index, :n].bool()
  catastrophic = arrays["catastrophic"][env_index, :n].bool()
  complete = arrays["route_completed"][env_index, :n].bool()
  code = arrays["termination_code"][env_index, :n]
  if bool((catastrophic & ~done).any()) or bool((complete & catastrophic).any()):
    raise ValueError("terminal lifecycle is inconsistent")
  if int((done | complete).sum()) > 1 or bool((done[:-1] | complete[:-1]).any()):
    raise ValueError("trajectory contains an anchor after its terminal action")
  if bool(done.any()) and int(code[done][0]) == 0:
    raise ValueError("done state lacks a terminal reason")
  termination_names = payload.get("termination_code_names")
  if not isinstance(termination_names, list) or not termination_names or termination_names[0] != "none":
    raise ValueError("termination code-name table is missing or malformed")
  if bool(done.any()) and int(code[done][0]) >= len(termination_names):
    raise ValueError("done state termination code is outside the registered name table")
  if int(payload["reset_count"][env_index]) != int(done.sum()):
    raise ValueError("trajectory reset counter differs from recorded done action")
  timeline = TrajectoryTimeline(
    contact=contact_timeline,
    loaded=loaded,
    slip_speed=slip,
    scheduled_stance=_scheduled_stance(ticks),
    progress=progress,
    state_valid=torch.ones(n + 1, dtype=torch.bool),
    attempt_id=torch.zeros(n + 1, dtype=torch.long),
    action_valid=torch.ones(n, dtype=torch.bool),
    done_after=done,
    catastrophic_after=catastrophic,
    route_complete_after=complete,
  )
  labels = build_future_labels(
    timeline,
    horizons=HORIZONS,
    control_dt_s=CONTROL_DT,
    scenario_speed=float(scenario["speed"]),
    confirmation_steps=2,
    boundary_grace_steps=2,
  )
  anchors = downsample_anchor_mask(n, every=ANCHOR_STRIDE, start=ANCHOR_START)
  row = {
    "actor": arrays["actor_observation"][env_index, :n][anchors].float(),
    "confirmed": saved[anchors].float(),
    "confirmed_valid": saved_valid4[anchors].all(dim=1),
    "tick": pre_ticks[anchors].long(),
    "count": int(anchors.sum()),
    "labels": {},
    "valid": {},
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
  raw_chatter = contact_chatter_metrics(
    contact_timeline,
    state_valid=timeline.state_valid,
    attempt_id=timeline.attempt_id,
  )
  confirmed_chatter = contact_chatter_metrics(
    saved,
    state_valid=torch.ones(n, dtype=torch.bool),
    attempt_id=torch.zeros(n, dtype=torch.long),
  )
  flips, near = _near_flip_fraction(saved_changed)
  clearance = arrays["clearance"][env_index, :n]
  clearance_valid = arrays["clearance_valid"][env_index, :n]
  clearance_finite = bool(torch.isfinite(clearance[clearance_valid]).all())
  audit = {
    "profile": str(payload["profile"]),
    "speed": float(scenario["speed"]),
    "raw_transitions": _transition_counts(raw),
    "confirmed_transitions": _transition_counts(saved),
    "confirmed_isolated_edges": torch.tensor(confirmed_chatter["isolated_excursion_edges"]),
    "raw_isolated_edges": torch.tensor(raw_chatter["isolated_excursion_edges"]),
    "confirmed_flips": flips,
    "confirmed_near_flips": near,
    "early_or_backfill": early_or_backfill,
    "clearance_valid": int(clearance_valid.sum()),
    "clearance_total": int(clearance_valid.numel()),
    "clearance_finite": clearance_finite,
  }
  return row, audit


def _build_dataset(chunks: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[RowDataset, dict[str, Any]]:
  actor: list[torch.Tensor] = []
  confirmed: list[torch.Tensor] = []
  confirmed_valid: list[torch.Tensor] = []
  seeds: list[torch.Tensor] = []
  slots: list[torch.Tensor] = []
  ticks: list[torch.Tensor] = []
  speeds: list[torch.Tensor] = []
  profiles: list[str] = []
  routes: list[str] = []
  trajectories: list[tuple[int, str, str, int]] = []
  targets = {horizon: {target: [] for target in TARGETS} for horizon in HORIZONS}
  valid = {horizon: {target: [] for target in TARGETS} for horizon in HORIZONS}
  transition = {
    (profile, speed): {
      key: torch.zeros(4, dtype=torch.long)
      for key in ("raw", "confirmed", "confirmed_isolated", "raw_isolated", "flips", "near")
    }
    for profile in PROFILES for speed in SPEEDS
  }
  trajectory_count = raw_rows = reset_total = early_total = 0
  clearance_valid = clearance_total = 0
  clearance_finite = True
  validate_native_foot_order(NATIVE_FOOT_GEOMS)
  for item, payload in chunks:
    if tuple(payload.get("native_foot_names", ())) != NATIVE_FOOT_ORDER:
      raise ValueError("chunk native foot order differs")
    if tuple(payload.get("runtime_sensor_names", ())) != NATIVE_FOOT_GEOMS:
      raise ValueError("chunk runtime sensor order differs")
    scenarios = payload.get("scenarios") or []
    if len(scenarios) != 16 or [int(value["matched_slot"]) for value in scenarios] != list(range(16)):
      raise ValueError("chunk must contain matched slots 0..15 exactly")
    expected_scenarios = {
      (direction, level, speed, sign, 2.5, 0)
      for direction in ("slope_up", "slope_down")
      for level in (0, 1)
      for speed in SPEEDS
      for sign in (1, -1)
    }
    actual_scenarios = {
      (
        str(value["slope_direction"]),
        int(value["level"]),
        float(value["speed"]),
        int(value["turn_sign"]),
        float(value["radius"]),
        int(value["repeat"]),
      )
      for value in scenarios
    }
    if actual_scenarios != expected_scenarios:
      raise ValueError("chunk scenario matrix differs from the registered 16 cells")
    arrays = payload.get("arrays") or {}
    shapes = {
      "actor_observation": ((16, 2400, 234), torch.float32),
      "critic_contact": ((16, 2400, 4), torch.bool),
      "sensor_contact": ((16, 2400, 4), torch.bool),
      "confirmed_contact": ((16, 2400, 4), torch.bool),
      "confirmed_contact_valid": ((16, 2400, 4), torch.bool),
      "confirmed_contact_changed": ((16, 2400, 4), torch.bool),
      "clearance": ((16, 2400, 4), torch.float32),
      "clearance_valid": ((16, 2400, 4), torch.bool),
      "pre_progress": ((16, 2400), torch.float32),
      "command": ((16, 2400, 3), torch.float32),
      "pre_episode_tick": ((16, 2400), torch.int32),
      "policy_action": ((16, 2400, 12), torch.float32),
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
    if set(arrays) != set(shapes):
      raise ValueError(
        f"chunk raw-array inventory differs: missing={sorted(set(shapes) - set(arrays))}, "
        f"extra={sorted(set(arrays) - set(shapes))}"
      )
    for name, (shape, dtype) in shapes.items():
      _require_array(arrays, name, shape, dtype)
    critic_slices = payload.get("critic_term_slices") or {}
    if tuple(critic_slices.get("foot_contact", ())) != (245, 249):
      raise ValueError("critic raw-contact slice differs from [245:249]")
    expected_actor_slices = {
      "base_ang_vel": (0, 3),
      "projected_gravity": (3, 6),
      "command": (6, 9),
      "phase": (9, 11),
      "joint_pos": (11, 23),
      "joint_vel": (23, 35),
      "actions": (35, 47),
      "height_scan": (47, 234),
    }
    actor_slices = {
      key: tuple(value) for key, value in (payload.get("actor_term_slices") or {}).items()
    }
    if actor_slices != expected_actor_slices:
      raise ValueError("actor observation-term slices differ from frozen V7 234D")
    if bool(payload.get("actor_observation_corruption")) is not (
      payload["profile"] == "randomized"
    ) or bool(payload.get("critic_observation_corruption")):
      raise ValueError("observation corruption profile differs from contract")
    reset = payload["reset_count"]
    reset_total += int(reset.sum())
    raw_rows += int(arrays["anchor_active"].sum())
    for env_index in range(16):
      row, audit = _trajectory_rows(payload, env_index)
      count = row["count"]
      seed = int(payload["seed"])
      profile = str(payload["profile"])
      route = str(payload["route_kind"])
      speed = float(scenarios[env_index]["speed"])
      actor.append(row["actor"])
      confirmed.append(row["confirmed"])
      confirmed_valid.append(row["confirmed_valid"])
      seeds.append(torch.full((count,), seed, dtype=torch.long))
      slots.append(torch.full((count,), env_index, dtype=torch.long))
      ticks.append(row["tick"])
      speeds.append(torch.full((count,), speed, dtype=torch.float64))
      profiles.extend([profile] * count)
      routes.extend([route] * count)
      trajectories.extend([(seed, profile, route, env_index)] * count)
      for horizon in HORIZONS:
        for target in TARGETS:
          targets[horizon][target].append(row["labels"][horizon][target])
          valid[horizon][target].append(row["valid"][horizon][target])
      totals = transition[(profile, speed)]
      totals["raw"] += audit["raw_transitions"]
      totals["confirmed"] += audit["confirmed_transitions"]
      totals["confirmed_isolated"] += audit["confirmed_isolated_edges"]
      totals["raw_isolated"] += audit["raw_isolated_edges"]
      totals["flips"] += audit["confirmed_flips"]
      totals["near"] += audit["confirmed_near_flips"]
      early_total += audit["early_or_backfill"]
      clearance_valid += audit["clearance_valid"]
      clearance_total += audit["clearance_total"]
      clearance_finite &= audit["clearance_finite"]
      trajectory_count += 1
  dataset = RowDataset(
    actor=torch.cat(actor),
    confirmed=torch.cat(confirmed),
    confirmed_valid=torch.cat(confirmed_valid),
    seed=torch.cat(seeds),
    slot=torch.cat(slots),
    tick=torch.cat(ticks),
    speed=torch.cat(speeds),
    profile=profiles,
    route=routes,
    trajectory=trajectories,
    targets={h: {t: torch.cat(values) for t, values in targets[h].items()} for h in HORIZONS},
    valid={h: {t: torch.cat(values) for t, values in valid[h].items()} for h in HORIZONS},
  )
  if dataset.actor.shape[1] != 234 or not torch.isfinite(dataset.actor).all():
    raise ValueError("actor rows are nonfinite or not 234D")
  if dataset.confirmed.shape[1] != 4 or not bool(((dataset.confirmed == 0) | (dataset.confirmed == 1)).all()):
    raise ValueError("confirmed feature is not binary native contact4")
  by_stratum: dict[str, Any] = {}
  integrity_reasons: list[str] = []
  for profile in PROFILES:
    for speed in SPEEDS:
      totals = transition[(profile, speed)]
      retention = torch.where(
        totals["raw"] > 0,
        totals["confirmed"].double() / totals["raw"].double(),
        torch.full((4,), float("inf")),
      )
      near_fraction = torch.where(
        totals["flips"] > 0,
        totals["near"].double() / totals["flips"].double(),
        torch.full((4,), float("inf")),
      )
      name = f"{profile}|vx={speed:.1f}"
      by_stratum[name] = {
        "raw_transition_edges_native": totals["raw"].tolist(),
        "confirmed_transition_edges_native": totals["confirmed"].tolist(),
        "confirmed_to_raw_retention_native": retention.tolist(),
        "confirmed_isolated_excursion_edges_native": totals["confirmed_isolated"].tolist(),
        "raw_isolated_excursion_edges_native": totals["raw_isolated"].tolist(),
        "confirmed_near_flip_fraction_native": near_fraction.tolist(),
      }
      if bool((totals["confirmed"] < 100).any()):
        integrity_reasons.append(f"{name}:confirmed_transition_each_foot_below_100")
      if bool(((retention < 0.5) | (retention > 1.0)).any()):
        integrity_reasons.append(f"{name}:confirmed_to_raw_retention_outside_0.5_1.0")
      if bool((totals["confirmed_isolated"] != 0).any()):
        integrity_reasons.append(f"{name}:confirmed_one_tick_excursion_present")
      if bool((near_fraction > 0.1).any()):
        integrity_reasons.append(f"{name}:confirmed_near_flip_fraction_above_0.1")
  if early_total != 0:
    integrity_reasons.append("confirmed_early_or_backfill_count_nonzero")
  audit = {
    "trajectory_count": trajectory_count,
    "raw_active_rows": raw_rows,
    "downsampled_anchor_rows": dataset.rows,
    "reset_total": reset_total,
    "independent_causal_replay_bitwise_equal": True,
    "confirmed_early_or_backfill_count": early_total,
    "contact_by_profile_speed": by_stratum,
    "raw_chatter_report_only": True,
    "clearance_integrity_only": {
      "finite_on_valid_rays": clearance_finite,
      "valid_ray_fraction": clearance_valid / max(clearance_total, 1),
      "used_as_model_feature": False,
      "used_as_row_mask": False,
    },
    "integrity_passed": not integrity_reasons,
    "integrity_reasons": integrity_reasons,
  }
  return dataset, audit


def _profile_mask(dataset: RowDataset, profile: str) -> torch.Tensor:
  return torch.tensor([value == profile for value in dataset.profile], dtype=torch.bool)


def _row_id_hash(dataset: RowDataset, indices: torch.Tensor) -> str:
  digest = hashlib.sha256()
  for index in indices.tolist():
    identity = (
      f"{int(dataset.seed[index])}|{dataset.profile[index]}|{dataset.route[index]}|"
      f"{int(dataset.slot[index])}|{int(dataset.tick[index])}\n"
    )
    digest.update(identity.encode("utf-8"))
  return digest.hexdigest()


def _target_coverage(
  dataset: RowDataset, mask: torch.Tensor, target: str, horizon: int
) -> dict[str, Any]:
  values = dataset.targets[horizon][target][mask].bool()
  groups = dataset.groups(mask)
  trajectories = dataset.trajectories(mask)
  positive_groups = {group for group, value in zip(groups, values.tolist(), strict=True) if value}
  negative_groups = {group for group, value in zip(groups, values.tolist(), strict=True) if not value}
  positive_trajectories = {
    trajectory for trajectory, value in zip(trajectories, values.tolist(), strict=True) if value
  }
  return {
    "groups": len(set(groups)),
    "trajectories": len(set(trajectories)),
    "anchors": int(values.numel()),
    "positive_event_groups": len(positive_groups),
    "negative_event_groups": len(negative_groups),
    "positive_anchors": int(values.sum()),
    "negative_anchors": int((~values).sum()),
    "positive_trajectories": len(positive_trajectories),
    "positive_trajectory_route_kinds": sorted({value[2] for value in positive_trajectories}),
  }


def _coverage_entry(
  dataset: RowDataset,
  *,
  horizon: int,
  scope: torch.Tensor,
  held_out: bool,
) -> tuple[dict[str, Any], list[str]]:
  # Usable inventory is feature availability only; each endpoint applies its
  # own label-valid mask below so catastrophic positives are not erased by the
  # full-window progress censor.
  usable = scope & dataset.confirmed_valid
  entry: dict[str, Any] = {
    "usable_groups": len(set(dataset.groups(usable))),
    "usable_trajectories": len(set(dataset.trajectories(usable))),
    "targets": {},
  }
  reasons: list[str] = []
  usable_group_min = 6 if held_out else 24
  if entry["usable_groups"] < usable_group_min:
    reasons.append(f"usable_groups_below_{usable_group_min}")
  if not held_out and entry["usable_trajectories"] < 72:
    reasons.append("usable_trajectories_below_72")
  for target in TARGETS:
    mask = usable & dataset.valid[horizon][target]
    value = _target_coverage(dataset, mask, target, horizon)
    entry["targets"][target] = value
    prefix = target
    if target in ("slip_onset", "unexpected_transition"):
      positive_groups = 4 if held_out else 24
      negative_groups = 4 if held_out else 24
      positive_anchors = 80 if held_out else 400
      negative_anchors = 80 if held_out else 400
      checks = (
        ("positive_event_groups", positive_groups),
        ("negative_event_groups", negative_groups),
        ("positive_anchors", positive_anchors),
        ("negative_anchors", negative_anchors),
      )
      for field, minimum in checks:
        if int(value[field]) < minimum:
          reasons.append(f"{prefix}:{field}_below_{minimum}")
      if not held_out and len(value["positive_trajectory_route_kinds"]) < 2:
        reasons.append(f"{prefix}:positive_trajectory_route_kinds_below_2")
    elif target == "catastrophic_failure":
      positive_groups = 2 if held_out else 12
      negative_groups = 4 if held_out else 24
      positive_anchor_min = (
        {10: 4, 25: 10, 50: 20}[horizon]
        if held_out else {10: 48, 25: 120, 50: 240}[horizon]
      )
      checks = (
        ("positive_event_groups", positive_groups),
        ("negative_event_groups", negative_groups),
        ("positive_anchors", positive_anchor_min),
        ("negative_anchors", 100 if held_out else 400),
      )
      for field, minimum in checks:
        if int(value[field]) < minimum:
          reasons.append(f"{prefix}:{field}_below_{minimum}")
      if not held_out and len(value["positive_trajectory_route_kinds"]) < 2:
        reasons.append(f"{prefix}:positive_trajectory_route_kinds_below_2")
    else:
      group_min = 6 if held_out else 24
      anchor_min = 1200 if held_out else 6500
      trajectory_min = None if held_out else 72
      if value["groups"] < group_min:
        reasons.append(f"future_progress:groups_below_{group_min}")
      if value["anchors"] < anchor_min:
        reasons.append(f"future_progress:anchors_below_{anchor_min}")
      if trajectory_min is not None and value["trajectories"] < trajectory_min:
        reasons.append(f"future_progress:trajectories_below_{trajectory_min}")
  entry["passed"] = not reasons
  entry["reasons"] = reasons
  return entry, reasons


def _coverage(dataset: RowDataset) -> tuple[dict[str, Any], bool, list[str]]:
  if len(set(dataset.trajectory)) != 384:
    raise ValueError("formal dataset must contain exactly 384 trajectories")
  cv_groups = {(int(seed), int(slot)) for seed, slot in zip(dataset.seed.tolist(), dataset.slot.tolist(), strict=True)}
  if len(cv_groups) != 64:
    raise ValueError("formal dataset must contain exactly 64 unique CV groups")
  profiles = {profile: _profile_mask(dataset, profile) for profile in PROFILES}
  overall: dict[str, Any] = {}
  held_out: dict[str, Any] = {}
  reasons: list[str] = []
  for horizon in HORIZONS:
    for profile in PROFILES:
      for speed in SPEEDS:
        stratum = profiles[profile] & torch.isclose(
          dataset.speed, torch.tensor(speed, dtype=torch.float64)
        )
        name = f"{profile}|vx={speed:.1f}|H={horizon}"
        entry, local = _coverage_entry(
          dataset, horizon=horizon, scope=stratum, held_out=False
        )
        overall[name] = entry
        reasons.extend(f"overall:{name}:{reason}" for reason in local)
        for seed in SEEDS:
          seed_scope = stratum & (dataset.seed == seed)
          seed_name = f"seed={seed}|{name}"
          seed_entry, seed_local = _coverage_entry(
            dataset, horizon=horizon, scope=seed_scope, held_out=True
          )
          held_out[seed_name] = seed_entry
          reasons.extend(f"held_out:{seed_name}:{reason}" for reason in seed_local)
  return {"overall": overall, "held_out_seed": held_out}, not reasons, reasons


@dataclass(frozen=True)
class _LogisticModel:
  mean: torch.Tensor
  scale: torch.Tensor
  coefficient: torch.Tensor
  intercept: torch.Tensor
  iterations: int

  def predict(self, features: torch.Tensor) -> torch.Tensor:
    normalized = (features.double() - self.mean) / self.scale
    return torch.sigmoid(normalized @ self.coefficient + self.intercept)


def _fit_logistic(
  features: torch.Tensor,
  target: torch.Tensor,
  sample_weight: torch.Tensor,
) -> _LogisticModel:
  x = features.double()
  y = target.double()
  weight = sample_weight.double()
  if x.ndim != 2 or y.shape != (x.shape[0],) or weight.shape != y.shape:
    raise ValueError("logistic inputs are misaligned")
  if not target.bool().any() or target.bool().all():
    raise ValueError("logistic training fold requires both classes")
  if not torch.isfinite(x).all() or not torch.isfinite(weight).all():
    raise ValueError("logistic inputs are nonfinite")
  weight = weight / weight.sum()
  mean = (x * weight[:, None]).sum(dim=0)
  variance = ((x - mean).square() * weight[:, None]).sum(dim=0)
  scale = variance.sqrt().clamp_min(1e-12)
  normalized = (x - mean) / scale
  positive_rate = (y * weight).sum().clamp(1e-8, 1 - 1e-8)
  parameter = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
  with torch.no_grad():
    parameter[-1] = torch.logit(positive_rate)
  optimizer = torch.optim.LBFGS(
    [parameter], lr=1.0, max_iter=200, tolerance_grad=1e-8,
    tolerance_change=1e-8, history_size=100, line_search_fn="strong_wolfe",
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
  iterations = int(optimizer.state[parameter].get("n_iter", 0))
  if not torch.isfinite(parameter).all() or not math.isfinite(float(loss.detach())):
    raise RuntimeError("LBFGS returned nonfinite state")
  if iterations >= 200:
    closure()
    gradient = float(parameter.grad.abs().max()) if parameter.grad is not None else math.inf
    if gradient > 1e-8:
      raise RuntimeError(f"LBFGS did not converge: iterations={iterations}, grad={gradient}")
  return _LogisticModel(
    mean=mean,
    scale=scale,
    coefficient=parameter[:-1].detach().clone(),
    intercept=parameter[-1].detach().clone(),
    iterations=iterations,
  )


def _feature_matrix(dataset: RowDataset, feature: str, mask: torch.Tensor) -> torch.Tensor:
  base = dataset.actor[mask]
  if feature == "baseline":
    return base
  if feature == "candidate":
    return torch.cat((base, dataset.confirmed[mask]), dim=1)
  raise ValueError(f"forbidden feature requested: {feature}")


def _fit_oof(dataset: RowDataset, horizon: int, target: str) -> dict[str, Any]:
  eligible = dataset.confirmed_valid & dataset.valid[horizon][target]
  indices = torch.where(eligible)[0]
  if indices.numel() == 0:
    raise ValueError(f"no eligible rows for {target} H{horizon}")
  row_hash = _row_id_hash(dataset, indices)
  seeds = dataset.seed[eligible]
  slots = dataset.slot[eligible]
  y = dataset.targets[horizon][target][eligible]
  matrices = {feature: _feature_matrix(dataset, feature, eligible) for feature in FEATURES}
  if matrices["baseline"].shape[1] != 234 or matrices["candidate"].shape[1] != 238:
    raise ValueError("paired feature dimensions differ from 234/238")
  predictions = {
    feature: torch.full((indices.numel(),), torch.nan, dtype=torch.float64)
    for feature in FEATURES
  }
  fold_rows: dict[str, int] = {}
  for held_seed in SEEDS:
    train = seeds != held_seed
    test = seeds == held_seed
    assert_no_group_leakage(seeds.tolist(), slots.tolist(), train, test)
    if not bool(train.any()) or not bool(test.any()):
      raise ValueError(f"exact LOSO fold {held_seed} is empty")
    if set(seeds[test].tolist()) != {held_seed} or held_seed in set(seeds[train].tolist()):
      raise ValueError("outer split is not exact leave-one-seed-out")
    groups = [
      (int(seed), int(slot))
      for seed, slot in zip(seeds[train].tolist(), slots[train].tolist(), strict=True)
    ]
    if target in BINARY_TARGETS:
      weights = balanced_binary_weights(y[train].bool(), groups)
    else:
      weights = group_balanced_weights(groups)
    for feature in FEATURES:
      if target in BINARY_TARGETS:
        model = _fit_logistic(matrices[feature][train], y[train].bool(), weights)
        predictions[feature][test] = model.predict(matrices[feature][test])
      else:
        model = fit_weighted_ridge(
          matrices[feature][train], y[train].double(),
          sample_weight=weights, l2=FIXED_L2,
        )
        predictions[feature][test] = model.predict(matrices[feature][test])
    fold_rows[str(held_seed)] = int(test.sum())
  if any(not torch.isfinite(value).all() for value in predictions.values()):
    raise RuntimeError(f"OOF prediction is incomplete for {target} H{horizon}")
  return {
    **predictions,
    "eligible_indices": indices,
    "row_id_sha256": row_hash,
    "baseline_row_id_sha256": row_hash,
    "candidate_row_id_sha256": row_hash,
    "paired_mask_bitwise_identical": True,
    "fold_rows": fold_rows,
  }


@dataclass(frozen=True)
class EffectInterval:
  estimate: float
  ci_low: float
  ci_high: float
  baseline_loss: float
  candidate_loss: float
  cluster_count: int
  resamples: int
  seed: int


def _loss_stats(
  baseline_error: torch.Tensor,
  candidate_error: torch.Tensor,
  target: torch.Tensor | None,
) -> torch.Tensor:
  if target is None:
    return torch.stack(
      (baseline_error, candidate_error, torch.ones_like(baseline_error)), dim=1
    )
  positive = target.bool()
  return torch.stack(
    (
      baseline_error * positive,
      candidate_error * positive,
      positive.double(),
      baseline_error * ~positive,
      candidate_error * ~positive,
      (~positive).double(),
    ),
    dim=1,
  )


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
  matrix: torch.Tensor,
  *,
  generator: torch.Generator,
  binary: bool,
) -> torch.Tensor:
  ids = torch.randint(
    matrix.shape[0],
    (BOOTSTRAP_RESAMPLES, matrix.shape[0]),
    generator=generator,
  )
  sampled = matrix[ids].sum(dim=1)
  if not binary:
    return sampled
  invalid = (sampled[:, 2] <= 0) | (sampled[:, 5] <= 0)
  attempts = 0
  while bool(invalid.any()):
    attempts += 1
    if attempts > 1000:
      raise RuntimeError("could not generate class-complete bootstrap samples")
    count = int(invalid.sum())
    replacements = torch.randint(
      matrix.shape[0],
      (count, matrix.shape[0]),
      generator=generator,
    )
    sampled[invalid] = matrix[replacements].sum(dim=1)
    invalid = (sampled[:, 2] <= 0) | (sampled[:, 5] <= 0)
  return sampled


def _bootstrap_effect(
  baseline_error: torch.Tensor,
  candidate_error: torch.Tensor,
  clusters: Sequence[tuple[int, int]],
  speeds: Sequence[float],
  *,
  target: torch.Tensor | None,
) -> tuple[EffectInterval, torch.Tensor]:
  if baseline_error.numel() == 0 or baseline_error.shape != candidate_error.shape:
    raise ValueError("paired bootstrap errors are empty or misaligned")
  if len(clusters) != baseline_error.numel() or len(speeds) != baseline_error.numel():
    raise ValueError("paired bootstrap metadata is misaligned")
  stats = _loss_stats(baseline_error.double(), candidate_error.double(), target)
  grouped: dict[tuple[float, tuple[int, int]], torch.Tensor] = {}
  cluster_speed: dict[tuple[int, int], float] = {}
  for index, (cluster, speed) in enumerate(zip(clusters, speeds, strict=True)):
    previous = cluster_speed.setdefault(cluster, float(speed))
    if previous != float(speed):
      raise ValueError("one bootstrap cluster appears in multiple speed strata")
    key = (float(speed), cluster)
    grouped[key] = grouped.get(
      key, torch.zeros(stats.shape[1], dtype=torch.float64)
    ) + stats[index]
  by_speed: dict[float, list[torch.Tensor]] = {}
  for (speed, _cluster), value in sorted(grouped.items()):
    by_speed.setdefault(speed, []).append(value)
  generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
  original_base: list[torch.Tensor] = []
  original_candidate: list[torch.Tensor] = []
  draw_base: list[torch.Tensor] = []
  draw_candidate: list[torch.Tensor] = []
  for speed in sorted(by_speed):
    matrix = torch.stack(by_speed[speed])
    base, candidate = _metric_from_stats(matrix.sum(dim=0), target is not None)
    sampled = _bootstrap_cluster_sums(
      matrix, generator=generator, binary=target is not None
    )
    base_draw, candidate_draw = _metric_from_stats(sampled, target is not None)
    original_base.append(base)
    original_candidate.append(candidate)
    draw_base.append(base_draw)
    draw_candidate.append(candidate_draw)
  base = torch.stack(original_base).mean()
  candidate = torch.stack(original_candidate).mean()
  base_draws = torch.stack(draw_base).mean(dim=0)
  candidate_draws = torch.stack(draw_candidate).mean(dim=0)
  if float(base) <= 0 or bool((base_draws <= 0).any()):
    raise ValueError("baseline loss must remain positive")
  effect_draws = (base_draws - candidate_draws) / base_draws
  interval = EffectInterval(
    estimate=float((base - candidate) / base),
    ci_low=float(torch.quantile(effect_draws, 0.025)),
    ci_high=float(torch.quantile(effect_draws, 0.975)),
    baseline_loss=float(base),
    candidate_loss=float(candidate),
    cluster_count=len(set(clusters)),
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
  )
  return interval, effect_draws


def _score_scope(
  dataset: RowDataset,
  horizon: int,
  target_name: str,
  prediction: dict[str, Any],
  scope: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
  original = prediction["eligible_indices"]
  keep = scope[original]
  selected = original[keep]
  y = dataset.targets[horizon][target_name][selected]
  baseline = prediction["baseline"][keep]
  candidate = prediction["candidate"][keep]
  if target_name in BINARY_TARGETS:
    y_bool = y.bool()
    baseline_probability = baseline.clamp(1e-12, 1 - 1e-12)
    candidate_probability = candidate.clamp(1e-12, 1 - 1e-12)
    baseline_error = -(
      y_bool * torch.log(baseline_probability)
      + (~y_bool) * torch.log1p(-baseline_probability)
    )
    candidate_error = -(
      y_bool * torch.log(candidate_probability)
      + (~y_bool) * torch.log1p(-candidate_probability)
    )
    binary_target: torch.Tensor | None = y_bool
  else:
    baseline_error = (y.double() - baseline).abs()
    candidate_error = (y.double() - candidate).abs()
    binary_target = None
  clusters = [
    (int(dataset.seed[index]), int(dataset.slot[index]))
    for index in selected.tolist()
  ]
  speeds = dataset.speed[selected].tolist()
  interval, draws = _bootstrap_effect(
    baseline_error,
    candidate_error,
    clusters,
    speeds,
    target=binary_target,
  )
  result: dict[str, Any] = asdict(interval)
  result["loss"] = "balanced_log_loss" if binary_target is not None else "normalized_mae"
  result["paired_rows"] = int(selected.numel())
  result["row_id_sha256"] = _row_id_hash(dataset, selected)
  if binary_target is not None:
    score_weight = group_balanced_weights(clusters)
    result["baseline_pr_auc"] = weighted_pr_auc(
      y_bool, baseline, sample_weight=score_weight
    )
    result["candidate_pr_auc"] = weighted_pr_auc(
      y_bool, candidate, sample_weight=score_weight
    )
    result["pr_auc_delta"] = result["candidate_pr_auc"] - result["baseline_pr_auc"]
  return result, draws


def _models_and_metrics(dataset: RowDataset) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  metrics: dict[str, Any] = {}
  draws: dict[str, Any] = {}
  pairing: dict[str, Any] = {}
  profile_masks = {profile: _profile_mask(dataset, profile) for profile in PROFILES}
  all_rows = torch.ones(dataset.rows, dtype=torch.bool)
  for horizon in HORIZONS:
    for target in TARGETS:
      prediction = _fit_oof(dataset, horizon, target)
      endpoint = metrics.setdefault(target, {}).setdefault(str(horizon), {"by_stratum": {}})
      endpoint_draws = draws.setdefault(target, {}).setdefault(str(horizon), {"by_stratum": {}})
      pairing[f"{target}|H={horizon}"] = {
        "eligible_rows": int(prediction["eligible_indices"].numel()),
        "row_id_sha256": prediction["row_id_sha256"],
        "baseline_row_id_sha256": prediction["baseline_row_id_sha256"],
        "candidate_row_id_sha256": prediction["candidate_row_id_sha256"],
        "paired_mask_bitwise_identical": prediction["paired_mask_bitwise_identical"],
        "fold_rows": prediction["fold_rows"],
      }
      macro, macro_draw = _score_scope(
        dataset, horizon, target, prediction, all_rows
      )
      endpoint["macro"] = macro
      endpoint_draws["macro"] = macro_draw
      for profile in PROFILES:
        for speed in SPEEDS:
          name = f"{profile}|vx={speed:.1f}"
          scope = profile_masks[profile] & torch.isclose(
            dataset.speed, torch.tensor(speed, dtype=torch.float64)
          )
          value, value_draws = _score_scope(
            dataset, horizon, target, prediction, scope
          )
          endpoint["by_stratum"][name] = value
          endpoint_draws["by_stratum"][name] = value_draws
  return metrics, draws, pairing


def _mean_interval(
  values: Sequence[dict[str, Any]],
  samples: Sequence[torch.Tensor],
) -> dict[str, Any]:
  draws = torch.stack(tuple(samples)).mean(dim=0)
  return {
    "estimate": sum(float(value["estimate"]) for value in values) / len(values),
    "ci_low": float(torch.quantile(draws, 0.025)),
    "ci_high": float(torch.quantile(draws, 0.975)),
    "resamples": BOOTSTRAP_RESAMPLES,
    "seed": BOOTSTRAP_SEED,
  }


def _apply_pass_gates(metrics: dict[str, Any], draws: dict[str, Any]) -> dict[str, Any]:
  reasons: list[str] = []
  checks: dict[str, Any] = {}
  for profile in PROFILES:
    for speed in SPEEDS:
      stratum = f"{profile}|vx={speed:.1f}"
      values = [metrics[target]["25"]["by_stratum"][stratum] for target in TARGETS]
      samples = [draws[target]["25"]["by_stratum"][stratum] for target in TARGETS]
      composite = _mean_interval(values, samples)
      checks[f"primary_H25_composite:{stratum}"] = composite
      if composite["estimate"] <= 0 or composite["ci_low"] <= 0:
        reasons.append(f"primary_H25_composite_no_positive_ci:{stratum}")
  for target in TARGETS:
    value = metrics[target]["25"]["macro"]
    checks[f"H25_target_macro:{target}"] = value
    if value["ci_low"] <= 0:
      reasons.append(f"H25_target_macro_ci_not_positive:{target}")
  for target in ("slip_onset", "unexpected_transition"):
    for horizon in HORIZONS:
      value = metrics[target][str(horizon)]["macro"]
      checks[f"event_macro:{target}:H{horizon}"] = value
      if value["ci_low"] <= 0:
        reasons.append(f"event_macro_ci_not_positive:{target}:H{horizon}")
  for horizon in (10, 50):
    stratum_values: list[dict[str, Any]] = []
    stratum_draws: list[torch.Tensor] = []
    for profile in PROFILES:
      for speed in SPEEDS:
        stratum = f"{profile}|vx={speed:.1f}"
        values = [metrics[target][str(horizon)]["by_stratum"][stratum] for target in TARGETS]
        samples = [draws[target][str(horizon)]["by_stratum"][stratum] for target in TARGETS]
        composite = _mean_interval(values, samples)
        checks[f"secondary_H{horizon}_composite:{stratum}"] = composite
        stratum_values.append(composite)
        stratum_draws.append(torch.stack(samples).mean(dim=0))
        if composite["estimate"] <= 0:
          reasons.append(f"H{horizon}_composite_point_not_positive:{stratum}")
    macro = _mean_interval(stratum_values, stratum_draws)
    checks[f"secondary_H{horizon}_composite_macro"] = macro
    if macro["ci_low"] <= 0:
      reasons.append(f"H{horizon}_composite_macro_ci_not_positive")
  for target in TARGETS:
    for horizon in HORIZONS:
      for stratum, value in metrics[target][str(horizon)]["by_stratum"].items():
        if value["estimate"] < 0:
          reasons.append(f"negative_point_gain:{target}:H{horizon}:{stratum}")
        if target in BINARY_TARGETS and value["pr_auc_delta"] < 0:
          reasons.append(f"binary_pr_auc_decrease:{target}:H{horizon}:{stratum}")
  return {"passed": not reasons, "reasons": reasons, "checks": checks}


def _decision(
  *,
  integrity_ok: bool,
  coverage_ok: bool,
  model_ok: bool | None,
) -> tuple[str, str]:
  if not integrity_ok:
    return "technical_failure", "TECHNICAL_FAILURE_DO_NOT_TRAIN"
  if not coverage_ok:
    return "coverage_inconclusive", "INCONCLUSIVE_DO_NOT_TRAIN"
  if model_ok is None:
    return "technical_failure", "TECHNICAL_FAILURE_DO_NOT_TRAIN"
  if not model_ok:
    return "completed", "DEBOUNCED_CONTACT_REJECTED_DO_NOT_TRAIN"
  return "completed", "OBSERVABILITY_DIAGNOSTIC_PASSED"


def analyze(raw_dir: Path) -> dict[str, Any]:
  manifest, chunks = _load_manifest(raw_dir)
  dataset, audit = _build_dataset(chunks)
  if audit["trajectory_count"] != 384:
    raise ValueError("trajectory inventory is not exactly 384")
  coverage, coverage_ok, coverage_reasons = _coverage(dataset)
  base_summary: dict[str, Any] = {
    "schema_version": 2,
    "evaluation_suite": ANALYSIS_SUITE,
    "created_at": datetime.now().astimezone().isoformat(),
    "decision_scope": "evaluation_only_gate_before_any_238d_teacher_training",
    "manifest": {
      "path": manifest["manifest_path"],
      "sha256": manifest["manifest_sha256"],
      "contract_sha256": manifest["contract_sha256"],
      "collector_sha256": manifest["collector_sha256"],
      "causal_filter_sha256": manifest["validated_filter_sha256"],
      "chunk_count": 24,
      "trajectory_count": 384,
      "unique_cv_groups": 64,
      "observed_raw_rows": manifest["observed_raw_rows"],
    },
    "analysis_contract": {
      "device": "cpu",
      "features": {"baseline": 234, "candidate": 238},
      "forbidden_model_features": [
        "raw_contact", "clearance", "base_lin_vel", "history_stack",
        "action", "post_action_state",
      ],
      "clearance_used_as_model_feature": False,
      "clearance_used_as_row_mask": False,
      "confirmed_contact_replay": "independent_causal_two_tick_no_backfill",
      "anchor_start_tick": ANCHOR_START,
      "anchor_stride_ticks": ANCHOR_STRIDE,
      "horizons_ticks": list(HORIZONS),
      "outer_split": "exact_four_fold_leave_one_seed_out",
      "outer_seeds": list(SEEDS),
      "group": ["seed", "matched_slot"],
      "fixed_l2": FIXED_L2,
      "bootstrap_unit": ["seed", "matched_slot"],
      "bootstrap_stratify_by": "speed",
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": BOOTSTRAP_SEED,
    },
    "raw_audit": audit,
    "coverage": coverage,
    "coverage_passed": coverage_ok,
    "coverage_reasons": coverage_reasons,
    "eligible_rows_by_target_horizon": {
      str(horizon): {
        target: int(
          (dataset.confirmed_valid & dataset.valid[horizon][target]).sum()
        )
        for target in TARGETS
      }
      for horizon in HORIZONS
    },
    "training_changed": False,
    "learn_called": False,
    "technical_failures": [],
  }
  if not audit["integrity_passed"]:
    status, decision = _decision(
      integrity_ok=False, coverage_ok=coverage_ok, model_ok=None
    )
    base_summary.update({
      "analysis_status": status,
      "metrics": {},
      "paired_row_audit": {},
      "pass_gates": {},
      "observability_diagnostic_passed": False,
      "decision": decision,
      "decision_reasons": ["integrity_failed", *audit["integrity_reasons"]],
    })
    return base_summary
  if not coverage_ok:
    status, decision = _decision(
      integrity_ok=True, coverage_ok=False, model_ok=None
    )
    base_summary.update({
      "analysis_status": status,
      "metrics": {},
      "paired_row_audit": {},
      "pass_gates": {},
      "observability_diagnostic_passed": False,
      "decision": decision,
      "decision_reasons": ["coverage_failed", *coverage_reasons],
    })
    return base_summary
  try:
    metrics, bootstrap_draws, pairing = _models_and_metrics(dataset)
    gates = _apply_pass_gates(metrics, bootstrap_draws)
  except Exception as error:
    status, decision = _decision(
      integrity_ok=True, coverage_ok=True, model_ok=None
    )
    base_summary.update({
      "analysis_status": status,
      "metrics": {},
      "paired_row_audit": {},
      "pass_gates": {},
      "technical_failures": [{
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      }],
      "observability_diagnostic_passed": False,
      "decision": decision,
      "decision_reasons": ["model_or_metric_technical_failure"],
    })
    return base_summary
  status, decision = _decision(
    integrity_ok=True, coverage_ok=True, model_ok=bool(gates["passed"])
  )
  base_summary.update({
    "analysis_status": status,
    "metrics": metrics,
    "paired_row_audit": pairing,
    "pass_gates": gates,
    "observability_diagnostic_passed": bool(gates["passed"]),
    "decision": decision,
    "decision_reasons": gates["reasons"],
  })
  return base_summary


def _markdown(summary: dict[str, Any]) -> str:
  lines = [
    "# Go2 causal debounced-contact observability diagnostic",
    "",
    f"- Decision: `{summary['decision']}`",
    f"- Analysis status: `{summary['analysis_status']}`",
    f"- Coverage passed: `{summary.get('coverage_passed', False)}`",
    f"- Manifest SHA256: `{summary.get('manifest', {}).get('sha256', 'unavailable')}`",
    f"- Contract SHA256: `{summary.get('manifest', {}).get('contract_sha256', 'unavailable')}`",
    f"- Causal filter SHA256: `{summary.get('manifest', {}).get('causal_filter_sha256', 'unavailable')}`",
    "- Compared features: baseline `234D`; candidate `234D + causal confirmed contact4 = 238D`",
    "- Clearance: integrity-only; excluded from model features and row masks",
    "- Training changed: `false`; `learn()` called: `false`",
    "",
    "## Decision reasons",
    "",
  ]
  reasons = summary.get("decision_reasons", [])
  lines.extend(["- None."] if not reasons else [f"- {reason}" for reason in reasons])
  failures = summary.get("technical_failures", [])
  if failures:
    lines.extend(("", "## Technical failures", ""))
    lines.extend(
      f"- `{item['type']}`: {item['message']}" for item in failures
    )
  lines.extend(("", "## Interpretation", ""))
  if summary["decision"] == "OBSERVABILITY_DIAGNOSTIC_PASSED":
    lines.append(
      "The preregistered V2 observability gate passed. This unlocks schema and "
      "preflight work only; it does not authorize formal PPO training by itself."
    )
  elif summary["decision"] == "DEBOUNCED_CONTACT_REJECTED_DO_NOT_TRAIN":
    lines.append(
      "Coverage was adequate, but the registered incremental model gates failed. "
      "Reject this debounced-contact candidate and do not train it."
    )
  elif summary["decision"] == "INCONCLUSIVE_DO_NOT_TRAIN":
    lines.append(
      "Registered coverage was insufficient. The diagnostic is inconclusive and "
      "does not authorize candidate training."
    )
  else:
    lines.append(
      "The formal evidence or analysis failed an integrity/technical requirement. "
      "Do not use this result for training decisions."
    )
  return "\n".join(lines) + "\n"


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
  if args.output_json is not None or args.output_md is not None:
    if args.output_json is None or args.output_md is None:
      raise ValueError("--output-json and --output-md must be supplied together")
    return args.output_json.expanduser().resolve(), args.output_md.expanduser().resolve()
  prefix = args.output_prefix.expanduser().resolve()
  return Path(f"{prefix}.json"), Path(f"{prefix}.md")


def _self_test() -> None:
  raw = torch.tensor(
    [[0], [1], [1], [0], [1], [1]], dtype=torch.bool
  ).repeat(1, 4)
  replay = recompute_causal_confirmed_contact(
    raw,
    attempt_id=torch.zeros(6, dtype=torch.long),
    episode_start=torch.tensor([True, False, False, False, False, False]),
  )
  expected = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)
  if not torch.equal(replay.contact[:, 0], expected):
    raise AssertionError("causal replay backfilled or switched early")
  mutated = raw.clone()
  mutated[4:] = ~mutated[4:]
  alternate = recompute_causal_confirmed_contact(
    mutated,
    attempt_id=torch.zeros(6, dtype=torch.long),
    episode_start=torch.tensor([True, False, False, False, False, False]),
  )
  if not torch.equal(replay.contact[:4], alternate.contact[:4]):
    raise AssertionError("future mutation changed a causal replay prefix")

  rows = 4
  zeros_bool = torch.zeros(rows, dtype=torch.bool)
  dataset = RowDataset(
    actor=torch.zeros(rows, 234),
    confirmed=torch.tensor(
      [[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
      dtype=torch.float32,
    ),
    confirmed_valid=torch.ones(rows, dtype=torch.bool),
    seed=torch.tensor(SEEDS),
    slot=torch.tensor([0, 1, 2, 3]),
    tick=torch.tensor([10, 15, 20, 25]),
    speed=torch.tensor([0.3, 0.3, 0.5, 0.5], dtype=torch.float64),
    profile=["clean"] * rows,
    route=["straight"] * rows,
    trajectory=[(seed, "clean", "straight", index) for index, seed in enumerate(SEEDS)],
    targets={
      horizon: {
        "slip_onset": torch.tensor([False, True, False, True]),
        "unexpected_transition": torch.tensor([False, True, False, True]),
        "catastrophic_failure": torch.tensor([False, True, False, True]),
        "future_progress": torch.ones(rows, dtype=torch.float64),
      }
      for horizon in HORIZONS
    },
    valid={
      horizon: {target: ~zeros_bool for target in TARGETS}
      for horizon in HORIZONS
    },
  )
  mask = torch.ones(rows, dtype=torch.bool)
  if _feature_matrix(dataset, "baseline", mask).shape != (rows, 234):
    raise AssertionError("baseline feature dimension changed")
  if _feature_matrix(dataset, "candidate", mask).shape != (rows, 238):
    raise AssertionError("candidate feature dimension changed")
  try:
    _feature_matrix(dataset, "clearance", mask)
  except ValueError:
    pass
  else:
    raise AssertionError("clearance was admitted as a model feature")
  indices = torch.arange(rows)
  if _row_id_hash(dataset, indices) != _row_id_hash(dataset, indices.clone()):
    raise AssertionError("paired row hash is nondeterministic")
  interval, _draws = _bootstrap_effect(
    torch.tensor([2.0, 2.0, 2.0, 2.0]),
    torch.tensor([1.0, 1.0, 1.0, 1.0]),
    [(seed, index) for index, seed in enumerate(SEEDS)],
    [0.3, 0.3, 0.5, 0.5],
    target=None,
  )
  if not math.isclose(interval.estimate, 0.5) or interval.ci_low <= 0:
    raise AssertionError("paired bootstrap did not preserve positive effect")
  if _decision(integrity_ok=True, coverage_ok=False, model_ok=None)[1] != "INCONCLUSIVE_DO_NOT_TRAIN":
    raise AssertionError("coverage failure did not fail closed")
  if _decision(integrity_ok=True, coverage_ok=True, model_ok=False)[1] != "DEBOUNCED_CONTACT_REJECTED_DO_NOT_TRAIN":
    raise AssertionError("model-gate failure did not reject candidate")
  try:
    _filter_sha({"filter_sha256": "forbidden-alias"})
  except ValueError:
    pass
  else:
    raise AssertionError("noncanonical filter SHA alias was accepted")
  if _filter_sha({"causal_filter_sha256": "ok"}) != "ok":
    raise AssertionError("canonical causal filter SHA was rejected")
  if FEATURES != ("baseline", "candidate"):
    raise AssertionError("a forbidden third model feature was registered")
  try:
    _coverage(dataset)
  except ValueError as error:
    if "384 trajectories" not in str(error):
      raise
  else:
    raise AssertionError("partial synthetic inventory did not fail closed")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
  parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument("--torch-threads", type=int, default=4)
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.torch_threads <= 0:
    parser.error("--torch-threads must be positive")
  torch.set_num_threads(args.torch_threads)
  torch.set_num_interop_threads(1)
  if args.self_test:
    if _sha256(CONTRACT) != EXPECTED_CONTRACT_SHA256:
      raise RuntimeError("V2 contract SHA differs from analyzer freeze")
    if _sha256(FILTER) != EXPECTED_FILTER_SHA256:
      raise RuntimeError("causal-filter SHA differs from analyzer freeze")
    _validate_contract(json.loads(CONTRACT.read_text(encoding="utf-8")))
    _self_test()
    print("SELF_TEST_OK")
    return
  output_json, output_md = _output_paths(args)
  if output_json.exists() or output_md.exists():
    raise FileExistsError(f"refusing to overwrite summary: {output_json} / {output_md}")
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_md.parent.mkdir(parents=True, exist_ok=True)
  technical = False
  try:
    summary = analyze(args.raw_dir.expanduser().resolve())
    technical = summary.get("analysis_status") == "technical_failure"
  except Exception as error:
    technical = True
    summary = {
      "schema_version": 2,
      "evaluation_suite": ANALYSIS_SUITE,
      "created_at": datetime.now().astimezone().isoformat(),
      "analysis_status": "technical_failure",
      "training_changed": False,
      "learn_called": False,
      "coverage": {},
      "coverage_passed": False,
      "observability_diagnostic_passed": False,
      "decision": "TECHNICAL_FAILURE_DO_NOT_TRAIN",
      "decision_reasons": ["technical_failure"],
      "technical_failures": [{
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      }],
    }
  with output_json.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n")
  with output_md.open("x", encoding="utf-8") as stream:
    stream.write(_markdown(summary))
  print(f"WROTE {output_json}")
  print(f"WROTE {output_md}")
  print(f"DECISION {summary['decision']}")
  if technical:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
