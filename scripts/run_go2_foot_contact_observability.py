"""Run and inventory the 18 registered observability raw chunks serially."""

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


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts/collect_go2_foot_contact_observability.py"
CONTRACT = ROOT / "docs/reviews/go2_foot_contact_observability_contract_20260809.json"
SEEDS = (42, 43, 44)
PROFILES = ("clean", "randomized")
ROUTES = ("straight", "arc", "s_curve")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_chunk(
  path: Path, *, seed: int, profile: str, route: str, contract_sha: str,
) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=True)
  expected = {
    "evaluation_suite": "go2_foot_contact_observability_raw_chunk_v1",
    "mode": "formal", "seed": seed, "profile": profile,
    "route_kind": route, "num_envs": 16, "steps_requested": 2400,
    "contract_sha256": contract_sha,
  }
  mismatch = {
    key: (payload.get(key), value)
    for key, value in expected.items() if payload.get(key) != value
  }
  if mismatch:
    raise ValueError(f"raw chunk identity mismatch in {path}: {mismatch}")
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
  if invariants != expected_invariants:
    raise ValueError(f"raw chunk invariant failed in {path}: {invariants}")
  scenarios = payload.get("scenarios") or []
  if len(scenarios) != 16 or [x["matched_slot"] for x in scenarios] != list(range(16)):
    raise ValueError(f"raw chunk matched-slot coverage differs in {path}")
  arrays = payload.get("arrays") or {}
  active = arrays.get("anchor_active")
  if not isinstance(active, torch.Tensor) or active.shape != (16, 2400):
    raise ValueError(f"raw chunk anchor shape differs in {path}")
  return {
    "seed": seed, "profile": profile, "route_kind": route,
    "path": str(path), "sha256": sha256_file(path),
    "size_bytes": path.stat().st_size,
    "raw_rows": int(active.sum()),
    "resets": int(payload["reset_count"].sum()),
    "steps_executed": int(payload["steps_executed"]),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--resume-existing", action="store_true")
  args = parser.parse_args()
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "manifest.json"
  if manifest_path.exists():
    raise FileExistsError(f"formal manifest already exists: {manifest_path}")
  contract_sha = sha256_file(CONTRACT)
  collector_sha = sha256_file(COLLECTOR)
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
            sys.executable, str(COLLECTOR), "--seed", str(seed),
            "--profile", profile, "--route-kind", route,
            "--steps", "2400", "--mode", "formal",
            "--output-file", str(output),
          )
          print(f"START {seed} {profile} {route}", flush=True)
          subprocess.run(command, cwd=ROOT, check=True)
          print(f"DONE {seed} {profile} {route}", flush=True)
        inventory.append(_validate_chunk(
          output, seed=seed, profile=profile, route=route,
          contract_sha=contract_sha,
        ))
  identities = {
    (x["seed"], x["profile"], x["route_kind"]) for x in inventory
  }
  if len(inventory) != 18 or len(identities) != 18:
    raise RuntimeError("formal raw inventory is not exact 18/18")
  raw_rows = sum(int(x["raw_rows"]) for x in inventory)
  if raw_rows > 691_200:
    raise RuntimeError("raw row count exceeds registered upper bound")
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_foot_contact_observability_raw_manifest_v1",
    "started_at": started,
    "completed_at": datetime.now().astimezone().isoformat(),
    "contract": str(CONTRACT), "contract_sha256": contract_sha,
    "collector": str(COLLECTOR), "collector_sha256": collector_sha,
    "chunk_count": len(inventory), "trajectory_count": 18 * 16,
    "raw_row_upper_bound": 691_200, "observed_raw_rows": raw_rows,
    "training_changed": False, "learn_called": False,
    "chunks": inventory,
  }
  content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
  with manifest_path.open("x", encoding="utf-8") as stream:
    stream.write(content)
  print(f"WROTE {manifest_path}", flush=True)
  print(f"SHA256 {sha256_file(manifest_path)}", flush=True)


if __name__ == "__main__":
  main()
