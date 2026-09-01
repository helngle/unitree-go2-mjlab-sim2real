"""Strict causal-coverage wrapper for the V7 actuator triplet evaluator.

This file deliberately keeps the existing triplet evaluator unchanged.  It runs
its private condition runner with a longer, common horizon, then applies the
per-cell identity, coverage, sham-noise and provenance gates that the original
diagnostic did not have.  It is an evaluation-only tool; no task or asset is
modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_go2_actuator_headroom_triplet as triplet


EXPECTED_CHECKPOINT_SHA256 = "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
REQUIRED_REPEATS = (16, 32, 64)
CRITICAL_CELLS = ("slope_up_high|vx_0.3", "slope_up_high|vx_0.5")


@dataclass(frozen=True)
class CausalCoverageConfig:
  checkpoint: str = triplet.base.V7_CHECKPOINT
  repeats: int = 16
  warmup_steps: int = 100
  sample_steps: int = 1600
  post_windows: tuple[int, ...] = (50, 100, 300)
  seed: int = 42
  device: str = "cuda:0"
  output_file: str = "go2_actuator_headroom_causal_coverage.json"
  complex_artifact: str = (
    "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_"
    "focus_probe_2048env_500iter/complex_terrain_causal_diagnostic_v2.json"
  )


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _finite(value: Any) -> bool:
  if isinstance(value, float):
    return math.isfinite(value)
  if isinstance(value, list):
    return all(_finite(item) for item in value)
  if isinstance(value, dict):
    return all(_finite(item) for item in value.values())
  return True


def _validate_config(cfg: CausalCoverageConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  expected = Path(triplet.base.V7_CHECKPOINT).resolve()
  if checkpoint != expected or _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("checkpoint path or SHA256 is not the locked V7 model_13600.pt")
  if cfg.repeats not in REQUIRED_REPEATS:
    raise ValueError(f"repeats must be one of {REQUIRED_REPEATS}; do not pool invocations")
  if cfg.warmup_steps != 100 or cfg.sample_steps < 1300:
    raise ValueError("warmup=100 and a horizon leaving a complete post-300 window are required")
  if cfg.post_windows != (50, 100, 300):
    raise ValueError("post windows must be 50, 100 and 300")
  if cfg.seed != 42:
    raise ValueError("seed is locked to 42")


def _run(cfg: CausalCoverageConfig) -> list[dict[str, Any]]:
  base_cfg = triplet.TripletConfig(
    checkpoint=cfg.checkpoint,
    repeats=cfg.repeats,
    warmup_steps=cfg.warmup_steps,
    sample_steps=cfg.sample_steps,
    post_windows=cfg.post_windows,
    seed=cfg.seed,
    device=cfg.device,
    output_file="unused.json",
  )
  # _run_condition contains the existing matched source/sham/probe rollout.
  # Calling it directly lets this new evaluator add post-300 without changing
  # the old formal artifact or its source SHA.
  return [
    triplet._run_condition(base_cfg, condition, terrain_kind, level)
    for condition, terrain_kind, level in triplet.TRIGGER_CONDITIONS
  ]


def _scenario_key(pair: dict[str, Any]) -> str:
  return "|".join((
    str(pair.get("terrain_condition")),
    f"vx_{float(pair.get('speed')):.1f}",
  ))


def _cell_seed(cell: str) -> int:
  return int.from_bytes(hashlib.sha256(cell.encode("ascii")).digest()[:4], "little") % 10000


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
  if not values:
    return {"n_valid": 0, "mean": None, "p05": None, "p95": None, "ci95": None, "status": "no_samples"}
  array = np.asarray(values, dtype=np.float64)
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(array), size=(10000, len(array)))
  means = array[indices].mean(axis=1)
  return {
    "n_valid": int(len(array)),
    "mean": float(array.mean()),
    "median": float(np.median(array)),
    "p05": float(np.quantile(array, 0.05)),
    "p95": float(np.quantile(array, 0.95)),
    "max": float(array.max()),
    "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    "status": "ok",
  }


def _cell_stats(pairs: list[dict[str, Any]], cell: str) -> dict[str, Any]:
  selected = [
    pair for pair in pairs
    if _scenario_key(pair) == cell
    and pair.get("trigger", {}).get("status") == "applied"
    and pair.get("branch_identity", {}).get("branch_pass")
    and pair.get("post_100", {}).get("status") == "complete"
  ]
  post300_count = sum(pair.get("post_300", {}).get("status") == "complete" for pair in selected)
  source = [float(pair["post_100"]["source_saturation"]["fraction"]) for pair in selected]
  sham = [float(pair["post_100"]["sham_saturation"]["fraction"]) for pair in selected]
  probe = [float(pair["post_100"]["probe_saturation"]["fraction"]) for pair in selected]
  effect = [s - p for s, p in zip(sham, probe, strict=True)]
  noise = [s - c for s, c in zip(source, sham, strict=True)]
  abs_noise = [abs(item) for item in noise]
  excess = [e - n for e, n in zip(effect, abs_noise, strict=True)]
  cell_seed = _cell_seed(cell)
  effect_boot = _bootstrap(effect, 42042 + cell_seed)
  excess_boot = _bootstrap(excess, 52042 + cell_seed)
  sham_total = float(sum(sham))
  probe_total = float(sum(probe))
  reduction = None if sham_total <= 1.0e-12 else (sham_total - probe_total) / sham_total
  consistency = None if not effect else sum(item > 0 for item in effect) / len(effect)
  coverage_pass = len(selected) >= 8
  sham_effect_pass = bool(
    coverage_pass and reduction is not None and reduction >= 0.50
    and excess_boot["ci95"] is not None and excess_boot["ci95"][0] > 0
    and consistency is not None and consistency >= 0.75
  )
  return {
    "cell": cell,
    "valid_post100_triplets": len(selected),
    "valid_post300_triplets": post300_count,
    "required_triplets": 8,
    "coverage_pass": coverage_pass,
    "matched_slots": [pair.get("matched_slot") for pair in selected],
    "source_saturation_fraction": _bootstrap(source, 62042 + cell_seed),
    "sham_saturation_fraction": _bootstrap(sham, 72042 + cell_seed),
    "probe_saturation_fraction": _bootstrap(probe, 82042 + cell_seed),
    "effect_probe_minus_sham_improvement": effect_boot,
    "noise_source_minus_sham": _bootstrap(noise, 92042 + cell_seed),
    "p95_abs_sham_noise": None if not abs_noise else float(np.quantile(abs_noise, 0.95)),
    "effect_minus_abs_noise": excess_boot,
    "probe_reduction_relative_to_sham": reduction,
    "improvement_consistency": consistency,
    "sham_effect_pass": sham_effect_pass,
    "time_order_status": "unmeasured_by_triplet_source; one control-step terminal uncertainty",
    "status": "ok" if coverage_pass else "insufficient_coverage",
  }


def _validate_identity(conditions: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
  expected: set[tuple[str, float, int]] = set()
  seen: set[tuple[str, float, int]] = set()
  errors: list[str] = []
  for condition in conditions:
    for pair in condition["pairs"]:
      key = (str(pair.get("terrain_condition")), float(pair.get("speed")), int(pair.get("repeat")))
      expected.add(key)
      if key in seen:
        errors.append(f"duplicate scenario key {key}")
      seen.add(key)
      expected_slot = int(pair.get("repeat")) + (repeats if math.isclose(float(pair.get("speed")), 0.5) else 0)
      if pair.get("matched_slot") != expected_slot:
        errors.append(f"matched_slot/repeat mismatch {key}")
      if pair.get("terrain_condition") not in ("flat", "slope_up_high"):
        errors.append(f"unexpected terrain {key}")
    if len(condition.get("pairs", [])) != 2 * repeats:
      errors.append(f"wrong pair count for {condition.get('terrain_condition')}")
  return {"pass": not errors, "expected_pair_keys": len(expected), "errors": errors}


def evaluate(cfg: CausalCoverageConfig) -> dict[str, Any]:
  _validate_config(cfg)
  conditions = _run(cfg)
  identity = _validate_identity(conditions, cfg.repeats)
  all_pairs = [pair for condition in conditions for pair in condition["pairs"]]
  cells = {cell: _cell_stats(all_pairs, cell) for cell in CRITICAL_CELLS}
  strict = identity["pass"] and all(
    condition["runtime_identity"]["runtime_pass"]
    and condition["terrain_assignment_position_error_max"] < 1.0e-4
    and condition["terrain_placement_position_error_max"] < 1.0e-4
    for condition in conditions
  )
  complex_path = Path(cfg.complex_artifact).expanduser().resolve()
  complex_payload = json.loads(complex_path.read_text())
  complex_provenance_pass = (
    complex_payload.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
    and _finite(complex_payload)
  )
  coverage_pass = all(cell["coverage_pass"] for cell in cells.values())
  sham_pass = all(cell["sham_effect_pass"] for cell in cells.values())
  payload: dict[str, Any] = {
    "schema_version": 2,
    "evaluation_suite": "go2_actuator_headroom_causal_coverage",
    "checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
    "checkpoint_sha256": _sha256(cfg.checkpoint),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(__file__),
    "config": asdict(cfg),
    "arms": [{"name": name, "multiplier": value} for name, value in zip(triplet.ARMS, triplet.MULTIPLIERS, strict=True)],
    "single_variable_contract": {"trigger": "same source joint at 0.98 limit for 3 valid rows", "application": "probe at detect+1", "unchanged": ["checkpoint", "terrain", "command", "termination", "gait", "observation", "reward", "network", "PPO"]},
    "conditions": conditions,
    "identity_gate": identity,
    "runtime_gate": strict,
    "complex_factor_provenance": {"path": str(complex_path), "sha256": _sha256(complex_path), "checkpoint_sha256": complex_payload.get("checkpoint_sha256"), "pass": complex_provenance_pass},
    "cell_statistics": cells,
    "causal_decision": {
      "coverage_pass": coverage_pass,
      "sham_pass": sham_pass,
      "strict_gate_pass": strict and complex_provenance_pass,
      "verdict": "INCONCLUSIVE",
      "reason": "time-order metrics are not captured by the legacy triplet; CONTACT_CAUSAL requires a simultaneous friction triplet",
      "primary_factor_evidence": "C",
      "training_ready": False,
      "next_single_variable": "terrain-relative stance/contact-stability or local-tangent slip shaping after a simultaneous friction sham/probe gate",
    },
    "metric_sources": {
      "actuator_triplet": "this artifact: saturation/lifecycle/runtime identity",
      "gait_contact": "high_slope_actuator_audit_clean_seed42_48slots_1200steps_v1.json",
      "factor_diagnostic": str(complex_path),
      "limitations": ["triplet legacy runner does not retain raw per-step gait/contact traces", "time-order onset metrics remain unmeasured", "CONTACT_CAUSAL requires source=0.6/sham=0.6/probe=1.2 simultaneous replay", "repeats are same-seed GPU world-index replications, not guaranteed iid"],
    },
  }
  if not _finite(payload):
    raise ValueError("non-finite value in causal coverage payload")
  return payload


def main() -> None:
  import tyro
  cfg = tyro.cli(CausalCoverageConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  artifact_sha = _sha256(output)
  summary = output.with_name(output.stem + "_summary.md")
  summary.write_text(
    "# V7 causal coverage summary\n\n"
    f"- artifact SHA256: `{artifact_sha}`\n"
    f"- checkpoint SHA256: `{payload['checkpoint_sha256']}`\n"
    f"- repeats: `{cfg.repeats}`, horizon: `{cfg.sample_steps}`, seed: `42`\n"
    f"- strict/runtime gate: `{payload['causal_decision']['strict_gate_pass']}`\n"
    f"- coverage gate: `{payload['causal_decision']['coverage_pass']}`\n"
    f"- sham gate: `{payload['causal_decision']['sham_pass']}`\n"
    f"- verdict: **{payload['causal_decision']['verdict']}**\n"
    "- training_ready: **false**; no training started.\n"
  )
  manifest = output.with_name(output.stem + "_manifest.json")
  manifest.write_text(json.dumps({
    "artifact_sha256": artifact_sha,
    "summary_sha256": _sha256(summary),
    "evaluator_source_sha256": payload["evaluator_source_sha256"],
    "checkpoint_sha256": payload["checkpoint_sha256"],
    "complex_artifact_sha256": payload["complex_factor_provenance"]["sha256"],
  }, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"output": str(output), "artifact_sha256": artifact_sha, "decision": payload["causal_decision"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
