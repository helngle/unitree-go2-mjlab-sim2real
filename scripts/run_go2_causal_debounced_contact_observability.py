"""Run and inventory the 24 registered causal-contact chunks serially."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.evaluation.debounced_contact_observability import (
  recompute_causal_confirmed_contact,
)


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts/collect_go2_causal_debounced_contact_observability.py"
CONTRACT = ROOT / (
  "docs/reviews/go2_causal_debounced_contact_observability_contract_20260809.json"
)
FILTER_MODULE = ROOT / (
  "src/tasks/velocity/evaluation/debounced_contact_observability.py"
)
DEFAULT_OUTPUT_DIR = ROOT / (
  "docs/reviews/go2_causal_debounced_contact_observability_raw_20260809_v2"
)
SEEDS = (1042, 1043, 1044, 1045)
PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")
CHUNK_COUNT = 24
TRAJECTORY_COUNT = 384
RAW_ROW_UPPER_BOUND = 921_600
CHECKPOINT_SHA256 = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)
ARRAY_SPECS = {
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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_array(
  arrays: dict[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype,
) -> torch.Tensor:
  value = arrays.get(name)
  if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
    actual = None if not isinstance(value, torch.Tensor) else (value.shape, value.dtype)
    raise ValueError(f"raw array {name} differs: {actual} != {(shape, dtype)}")
  return value


def _validate_chunk(
  path: Path,
  *,
  seed: int,
  profile: str,
  route: str,
  contract_sha: str,
  collector_sha: str,
  filter_sha: str,
) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=True)
  expected = {
    "schema_version": 2,
    "evaluation_suite": (
      "go2_causal_debounced_contact_observability_raw_chunk_v2"
    ),
    "mode": "formal",
    "seed": seed,
    "profile": profile,
    "route_kind": route,
    "num_envs": 16,
    "steps_requested": 2400,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "contract_sha256": contract_sha,
    "collector_sha256": collector_sha,
    "causal_filter_sha256": filter_sha,
  }
  mismatch = {
    key: (payload.get(key), value)
    for key, value in expected.items()
    if payload.get(key) != value
  }
  if mismatch:
    raise ValueError(f"raw chunk identity mismatch in {path}: {mismatch}")
  expected_invariants = {
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
  invariants = payload.get("invariants") or {}
  if invariants != expected_invariants:
    raise ValueError(f"raw chunk invariant failed in {path}: {invariants}")
  audit = payload.get("causal_filter_audit") or {}
  expected_audit = {
    "filter_module": str(FILTER_MODULE),
    "filter_module_sha256": filter_sha,
    "confirmation_ticks": 2,
    "saved_confirmed_equals_cpu_oracle_bitwise": True,
    "saved_changed_equals_cpu_oracle_bitwise": True,
    "saved_valid_equals_cpu_oracle_bitwise": True,
    "contact_mismatch_count": 0,
    "changed_mismatch_count": 0,
    "valid_mismatch_count": 0,
    "early_flip_count": 0,
    "missing_or_future_backfill_count": 0,
    "early_flip_or_future_backfill_count": 0,
  }
  if audit != expected_audit:
    raise ValueError(f"causal filter audit failed in {path}: {audit}")
  scenarios = payload.get("scenarios") or []
  if len(scenarios) != 16 or [
    scenario["matched_slot"] for scenario in scenarios
  ] != list(range(16)):
    raise ValueError(f"raw chunk matched-slot coverage differs in {path}")
  if tuple(payload.get("runtime_sensor_names") or ()) != (
    "FL_foot_collision",
    "FR_foot_collision",
    "RL_foot_collision",
    "RR_foot_collision",
  ):
    raise ValueError(f"runtime sensor order differs in {path}")

  arrays = payload.get("arrays") or {}
  if set(arrays) != set(ARRAY_SPECS):
    raise ValueError(
      f"raw array inventory differs in {path}: "
      f"missing={sorted(set(ARRAY_SPECS) - set(arrays))}, "
      f"extra={sorted(set(arrays) - set(ARRAY_SPECS))}"
    )
  checked = {
    name: _validate_array(arrays, name, shape, dtype)
    for name, (shape, dtype) in ARRAY_SPECS.items()
  }
  active = checked["anchor_active"]
  critic = checked["critic_contact"]
  sensor = checked["sensor_contact"]
  confirmed = checked["confirmed_contact"]
  confirmed_valid = checked["confirmed_contact_valid"]
  confirmed_changed = checked["confirmed_contact_changed"]
  if not torch.equal(critic, sensor):
    raise ValueError(f"critic/sensor raw contact differs in {path}")
  if not torch.equal(confirmed_valid, active[:, :, None].expand(-1, -1, 4)):
    raise ValueError(f"confirmed validity differs from active rows in {path}")
  if bool((confirmed & ~confirmed_valid).any()):
    raise ValueError(f"confirmed contact is set on invalid rows in {path}")
  if bool((active[:, 1:] & ~active[:, :-1]).any()):
    raise ValueError(f"active rows restart after a frozen trajectory in {path}")
  episode_start = active.transpose(0, 1).clone()
  episode_start[1:] &= ~active.transpose(0, 1)[:-1]
  slots = torch.tensor(
    [int(scenario["matched_slot"]) for scenario in scenarios], dtype=torch.long
  )
  oracle = recompute_causal_confirmed_contact(
    sensor.transpose(0, 1),
    attempt_id=slots[None, :].expand(2400, 16),
    episode_start=episode_start,
    state_valid=active.transpose(0, 1),
    confirmation_ticks=2,
  )
  if not torch.equal(confirmed.transpose(0, 1), oracle.contact):
    raise ValueError(f"confirmed contact differs from CPU replay in {path}")
  if not torch.equal(confirmed_changed.transpose(0, 1), oracle.changed):
    raise ValueError(f"confirmed changes differ from CPU replay in {path}")
  reset_count = payload.get("reset_count")
  if (
    not isinstance(reset_count, torch.Tensor)
    or reset_count.shape != (16,)
    or bool((reset_count > 1).any())
  ):
    raise ValueError(f"reset count differs in {path}")
  return {
    "seed": seed,
    "profile": profile,
    "route_kind": route,
    "path": str(path),
    "sha256": sha256_file(path),
    "size_bytes": path.stat().st_size,
    "raw_rows": int(active.sum()),
    "resets": int(reset_count.sum()),
    "steps_executed": int(payload["steps_executed"]),
    "early_flip_or_future_backfill_count": 0,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--resume-existing", action="store_true")
  args = parser.parse_args()
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "manifest.json"
  if manifest_path.exists():
    raise FileExistsError(f"formal manifest already exists: {manifest_path}")
  contract_sha = sha256_file(CONTRACT)
  collector_sha = sha256_file(COLLECTOR)
  filter_sha = sha256_file(FILTER_MODULE)
  inventory: list[dict[str, Any]] = []
  started = datetime.now().astimezone().isoformat()
  for seed in SEEDS:
    for profile in PROFILES:
      for route in ROUTES:
        output = output_dir / f"seed_{seed}" / profile / f"{route}.pt"
        if output.exists():
          if not args.resume_existing:
            raise FileExistsError(output)
          print(f"VALIDATE {seed} {profile} {route}", flush=True)
        else:
          output.parent.mkdir(parents=True, exist_ok=True)
          command = (
            sys.executable,
            str(COLLECTOR),
            "--seed",
            str(seed),
            "--profile",
            profile,
            "--route-kind",
            route,
            "--steps",
            "2400",
            "--mode",
            "formal",
            "--output-file",
            str(output),
          )
          print(f"START {seed} {profile} {route}", flush=True)
          subprocess.run(command, cwd=ROOT, check=True)
          print(f"DONE {seed} {profile} {route}", flush=True)
        inventory.append(
          _validate_chunk(
            output,
            seed=seed,
            profile=profile,
            route=route,
            contract_sha=contract_sha,
            collector_sha=collector_sha,
            filter_sha=filter_sha,
          )
        )
  identities = {
    (item["seed"], item["profile"], item["route_kind"])
    for item in inventory
  }
  if len(inventory) != CHUNK_COUNT or len(identities) != CHUNK_COUNT:
    raise RuntimeError("formal raw inventory is not exact 24/24")
  raw_rows = sum(int(item["raw_rows"]) for item in inventory)
  if raw_rows > RAW_ROW_UPPER_BOUND:
    raise RuntimeError("raw row count exceeds registered upper bound")
  payload = {
    "schema_version": 2,
    "evaluation_suite": (
      "go2_causal_debounced_contact_observability_raw_manifest_v2"
    ),
    "started_at": started,
    "completed_at": datetime.now().astimezone().isoformat(),
    "contract": str(CONTRACT),
    "contract_sha256": contract_sha,
    "collector": str(COLLECTOR),
    "collector_sha256": collector_sha,
    "filter_module": str(FILTER_MODULE),
    "causal_filter_sha256": filter_sha,
    "chunk_count": len(inventory),
    "trajectory_count": TRAJECTORY_COUNT,
    "raw_row_upper_bound": RAW_ROW_UPPER_BOUND,
    "observed_raw_rows": raw_rows,
    "training_changed": False,
    "learn_called": False,
    "early_flip_or_future_backfill_count": 0,
    "chunks": inventory,
  }
  content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
  with manifest_path.open("x", encoding="utf-8") as stream:
    stream.write(content)
  print(f"WROTE {manifest_path}", flush=True)
  print(f"SHA256 {sha256_file(manifest_path)}", flush=True)


if __name__ == "__main__":
  main()
