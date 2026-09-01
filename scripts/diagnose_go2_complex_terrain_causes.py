"""Evaluation-only synthesis and counterfactuals for Go2 V7 terrain failures.

The established gait and actuator evaluators remain immutable provenance.  This
tool reuses their artifacts and adds the missing fixed-friction, height-scan,
and 1.0x sham sentinels before producing an A/B/C/D/E evidence summary.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_go2_actuator_headroom_triggered as triggered
import scripts.diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


RUN_DIR = Path(
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter"
)
GAIT_BASELINE = RUN_DIR / "high_slope_gait_diagnostics_clean_seed42_48slots_1200steps_v2.json"
ACTUATOR_BASELINE = RUN_DIR / "high_slope_actuator_audit_clean_seed42_48slots_1200steps_v1.json"
TRIGGERED_BASELINE = RUN_DIR / "actuator_headroom_triggered_clean_seed42_64worlds_1200steps_v4.json"
FACTOR_NAMES = (
  "friction_low_0p3",
  "friction_nominal_0p6",
  "friction_high_1p2",
  "height_scan_masked",
)
FRICTION_VALUES = {
  "friction_low_0p3": 0.3,
  "friction_nominal_0p6": 0.6,
  "friction_high_1p2": 1.2,
}


@dataclass(frozen=True)
class ComplexTerrainConfig:
  checkpoint: str = gait.V7_CHECKPOINT
  factors: tuple[str, ...] = FACTOR_NAMES
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  run_factor_rollouts: bool = True
  run_sham_sentinel: bool = True
  reuse_factor_artifact: str | None = None
  forced_sham_trigger_step: int | None = None
  gait_baseline: str = str(GAIT_BASELINE)
  actuator_baseline: str = str(ACTUATOR_BASELINE)
  triggered_baseline: str = str(TRIGGERED_BASELINE)
  output_file: str = str(RUN_DIR / "complex_terrain_causal_diagnostic.json")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
  resolved = Path(path).expanduser().resolve()
  value = json.loads(resolved.read_text())
  assert_recursive_json_finite(value)
  return value


def _validate_config(cfg: ComplexTerrainConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(gait.V7_CHECKPOINT).resolve():
    raise ValueError("this diagnostic is locked to V7 model_13600.pt")
  if checkpoint.name in gait.FORBIDDEN_CHECKPOINTS:
    raise ValueError("forbidden checkpoint")
  if not cfg.factors or any(name not in FACTOR_NAMES for name in cfg.factors):
    raise ValueError(f"factors must be selected from {FACTOR_NAMES}")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("invalid repeats/warmup/sample_steps")
  if cfg.forced_sham_trigger_step is not None and not (
    2 <= cfg.forced_sham_trigger_step < cfg.sample_steps - 1
  ):
    raise ValueError("forced sham trigger must leave a following step")


def _set_height_scan_scale(env_cfg: Any, value: float) -> None:
  actor = env_cfg.observations["actor"]
  try:
    actor.terms["height_scan"].scale = value
  except KeyError as exc:
    raise RuntimeError("V7 actor has no height_scan observation") from exc


def _configure_factor_profile(
  env_cfg: Any, factor: str, original_profile: Any,
) -> dict[str, Any]:
  friction_event = deepcopy(env_cfg.events.get("foot_friction"))
  settings = original_profile(env_cfg, "clean")
  if factor in FRICTION_VALUES:
    if friction_event is None:
      raise RuntimeError("V7 task has no foot_friction event")
    value = FRICTION_VALUES[factor]
    friction_event.params["ranges"] = (value, value)
    friction_event.params["operation"] = "abs"
    friction_event.params["shared_random"] = True
    env_cfg.events["foot_friction"] = friction_event
    settings["startup_randomization_events"] = ["foot_friction"]
    settings["fixed_foot_friction"] = value
  elif factor == "height_scan_masked":
    _set_height_scan_scale(env_cfg, 0.0)
    settings["actor_height_scan_scale"] = 0.0
    settings["height_scan_counterfactual"] = "all actor height-scan samples multiplied by zero"
  else:
    raise ValueError(f"unknown factor {factor}")
  settings["factor"] = factor
  settings["single_variable"] = True
  return settings


def _refresh_after_assignment(env: Any) -> None:
  for sensor in env.scene._sensors.values():
    sensor._invalidate_cache()
  env.observation_manager._obs_buffer = None
  env.obs_buf = env.observation_manager.compute(update_history=True)


@contextmanager
def _patched_gait_factor(factor: str) -> Iterator[None]:
  original_profile = gait._configure_profile
  original_assign = gait._assign_terrain

  def configure(env_cfg: Any, profile: str) -> dict[str, Any]:
    del profile
    return _configure_factor_profile(env_cfg, factor, original_profile)

  def assign(env: Any, scenarios: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    placement = original_assign(env, scenarios, device)
    _refresh_after_assignment(env)
    placement["post_assignment_observation_refresh"] = True
    return placement

  gait._configure_profile = configure
  gait._assign_terrain = assign
  try:
    yield
  finally:
    gait._configure_profile = original_profile
    gait._assign_terrain = original_assign


def _run_factor(cfg: ComplexTerrainConfig, factor: str) -> dict[str, Any]:
  gait_cfg = gait.GaitConfig(
    checkpoint=cfg.checkpoint,
    profiles=("clean",),
    speeds=(0.3, 0.5),
    repeats=cfg.repeats,
    warmup_steps=cfg.warmup_steps,
    sample_steps=cfg.sample_steps,
    seed=cfg.seed,
    device=cfg.device,
    output_file="unused.json",
  )
  with _patched_gait_factor(factor):
    payload = gait.evaluate(gait_cfg)
  payload["evaluation_suite"] = "go2_complex_terrain_factor_counterfactual"
  payload["factor"] = factor
  payload["source_evaluator"] = str(Path(gait.__file__).resolve())
  payload["source_evaluator_sha256"] = _sha256(Path(gait.__file__).resolve())
  payload["post_assignment_observation_refresh"] = True
  assert_recursive_json_finite(payload)
  return payload


def _run_sham(cfg: ComplexTerrainConfig) -> dict[str, Any]:
  old_multipliers = triggered.MULTIPLIERS
  triggered.MULTIPLIERS = (1.0, 1.0)
  try:
    payload = triggered.evaluate(triggered.TriggeredConfig(
      checkpoint=cfg.checkpoint,
      repeats=cfg.repeats,
      warmup_steps=cfg.warmup_steps,
      sample_steps=cfg.sample_steps,
      seed=cfg.seed,
      device=cfg.device,
      forced_trigger_step=cfg.forced_sham_trigger_step,
      output_file="unused.json",
    ))
  finally:
    triggered.MULTIPLIERS = old_multipliers
  payload["evaluation_suite"] = "go2_actuator_1p0x_sham_branch_sentinel"
  payload["sham_contract"] = {
    "source": "1.00x",
    "sham": "1.00x before and after branch",
    "branch_copy": "same full-copy path as triggered evaluator",
    "limitation": (
      "separate invocation from the 1.25x artifact; this estimates branch noise "
      "but is not a simultaneous three-world source/sham/probe experiment"
    ),
  }
  payload["causal_decision"] = {
    "verdict": "SHAM_SENTINEL_ONLY",
    "training_ready": False,
  }
  assert_recursive_json_finite(payload)
  return payload


def _scenario_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for profile in payload.get("profiles", {}).values():
    for condition in profile.get("conditions", {}).values():
      rows.extend(condition.get("scenarios", []))
  return rows


def _metric_mean(row: dict[str, Any], name: str) -> float | None:
  value = row.get(name)
  if isinstance(value, dict):
    mean = value.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None
  if isinstance(value, list):
    means = [item.get("mean") for item in value if isinstance(item, dict)]
    finite = [float(item) for item in means if isinstance(item, (int, float)) and math.isfinite(item)]
    return sum(finite) / len(finite) if finite else None
  return None


def _aggregate_rows(payload: dict[str, Any]) -> dict[str, Any]:
  groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
  for row in _scenario_rows(payload):
    groups.setdefault((row["terrain_condition"], float(row["speed"])), []).append(row)
  output: dict[str, Any] = {}
  for (condition, speed), rows in groups.items():
    key = f"{condition}|vx_{speed:g}"
    result: dict[str, Any] = {
      "attempts": len(rows),
      "completed": sum(not bool(row.get("failed")) for row in rows),
      "failure_reasons": {},
    }
    for row in rows:
      reason = row.get("first_failure_reason")
      if reason is not None:
        result["failure_reasons"][reason] = result["failure_reasons"].get(reason, 0) + 1
    for name in (
      "base_pitch", "terrain_relative_clearance", "foot_slip_tangent",
      "step_length", "forward_swing_displacement", "action_acceleration",
    ):
      values = [value for row in rows if (value := _metric_mean(row, name)) is not None]
      result[name] = {
        "mean": sum(values) / len(values) if values else None,
        "count": len(values),
        "status": "ok" if values else "no_valid_samples",
      }
    gains = [
      float(row["response_gain"]["vx"])
      for row in rows
      if isinstance(row.get("response_gain", {}).get("vx"), (int, float))
    ]
    result["response_gain_vx"] = {
      "mean": sum(gains) / len(gains) if gains else None,
      "count": len(gains),
      "status": "ok" if gains else "no_valid_samples",
    }
    output[key] = result
  return output


def _paired_delta(control: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
  output: dict[str, Any] = {}
  keys = sorted(set(control) & set(probe))
  for key in keys:
    c, p = control[key], probe[key]
    comparable = c["attempts"] == p["attempts"]
    row: dict[str, Any] = {
      "comparable": comparable,
      "comparison_status": "ok" if comparable else "attempt_count_mismatch",
      "completion_delta": int(p["completed"] - c["completed"]) if comparable else None,
      "attempts": min(c["attempts"], p["attempts"]),
      "control_attempts": c["attempts"],
      "probe_attempts": p["attempts"],
    }
    for metric in (
      "response_gain_vx", "step_length", "terrain_relative_clearance",
      "foot_slip_tangent", "base_pitch", "action_acceleration",
    ):
      cv, pv = c[metric]["mean"], p[metric]["mean"]
      row[f"{metric}_delta"] = None if cv is None or pv is None else pv - cv
      row[f"{metric}_relative_change"] = (
        None if cv is None or pv is None or abs(cv) < 1.0e-12 else (pv - cv) / abs(cv)
      )
    output[key] = row
  return output


def _classify(
  baseline: dict[str, Any], factors: dict[str, dict[str, Any]],
  actuator: dict[str, Any], triggered_result: dict[str, Any],
  sham: dict[str, Any] | None,
) -> dict[str, Any]:
  del actuator
  evidence: dict[str, list[str]] = {name: [] for name in "ABCDE"}
  baseline_agg = _aggregate_rows(baseline)
  factor_agg = {name: _aggregate_rows(value) for name, value in factors.items()}
  deltas = {
    name: _paired_delta(baseline_agg, rows)
    for name, rows in factor_agg.items()
  }
  contrasts: dict[str, dict[str, Any]] = {}
  if "friction_nominal_0p6" in factor_agg:
    for name in ("friction_low_0p3", "friction_high_1p2"):
      if name in factor_agg:
        contrasts[f"{name}_minus_friction_nominal_0p6"] = _paired_delta(
          factor_agg["friction_nominal_0p6"], factor_agg[name]
        )

  evidence["A"].append("formal gait baseline shows high-slope local-tangent step shortening")
  evidence["B"].append("formal gait baseline does not show a systematic high-slope clearance decrease")
  evidence["C"].append("formal gait baseline shows higher stance slip/tangential load on slopes")
  verdict = triggered_result.get("causal_decision", {}).get("verdict")
  evidence["D"].append(f"triggered 1.25x actuator verdict is {verdict}")
  if sham is None:
    evidence["E"].append("1.0x sham branch sentinel was not run")
  else:
    evidence["E"].append("1.0x sham was measured in a separate invocation, not a simultaneous triplet")

  scan = deltas.get("height_scan_masked", {})
  scan_hits = [
    row for key, row in scan.items()
    if key.startswith("slope_") and row["comparable"] and (
      row["completion_delta"] <= -2
      or (row["response_gain_vx_relative_change"] is not None
          and row["response_gain_vx_relative_change"] <= -0.2)
    )
  ]
  if scan_hits:
    flat_scan_failure = any(
      key.startswith("flat|") and row["comparable"]
      and row["completion_delta"] is not None and row["completion_delta"] <= -2
      for key, row in scan.items()
    )
    if flat_scan_failure:
      evidence["E"].append(
        "zeroing height scan destroys flat as well as slope behavior; the channel is globally used but this is not a slope-specific perception diagnosis"
      )
    else:
      evidence["A"].append("masking height scan materially degrades at least one slope cell")

  friction = contrasts.get(
    "friction_high_1p2_minus_friction_nominal_0p6", {}
  )
  friction_hits = [
    row for key, row in friction.items()
    if key.startswith("slope_") and row["comparable"] and (
      row["completion_delta"] >= 2
      or (
        row["response_gain_vx_relative_change"] is not None
        and row["step_length_relative_change"] is not None
        and row["response_gain_vx_relative_change"] >= 0.2
        and row["step_length_relative_change"] >= 0.2
      )
    )
  ]
  if friction_hits:
    evidence["C"].append(
      f"fixed friction 1.2 materially improves {len(friction_hits)} slope cells relative to the same refreshed 0.6 sentinel"
    )

  hard_limitations = [
    "current sham is not simultaneous with the 1.25x branch",
    "CPU-MJWarp is not the native deterministic MuJoCo C backend",
    "full touchdown-to-touchdown stride and support-polygon margin are not in the formal baseline",
    "real Go2 torque-speed, power, voltage, thermal and latency limits are unmeasured",
  ]
  primary = "C" if len(friction_hits) >= 2 else "INCONCLUSIVE"
  training_ready = False
  if primary == "INCONCLUSIVE" and scan_hits and sham is not None and verdict == "SATURATION_DOWNSTREAM":
    primary = "A"
  return {
    "primary_class": primary,
    "training_ready": training_ready,
    "evidence": evidence,
    "factor_deltas": deltas,
    "factor_contrasts": contrasts,
    "hard_limitations": hard_limitations,
    "next_single_variable": (
      "none until simultaneous source/sham/probe branch consistency passes"
      if primary == "INCONCLUSIVE" else
      "terrain-conditioned foot-placement/step-length shaping" if primary == "A" else
      "friction-robust stance/contact-stability shaping"
    ),
  }


def evaluate(cfg: ComplexTerrainConfig) -> dict[str, Any]:
  _validate_config(cfg)
  gait_baseline = _load_json(cfg.gait_baseline)
  actuator_baseline = _load_json(cfg.actuator_baseline)
  triggered_baseline = _load_json(cfg.triggered_baseline)
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  expected_sha = _sha256(checkpoint)
  for name, artifact in (
    ("gait", gait_baseline), ("actuator", actuator_baseline),
    ("triggered", triggered_baseline),
  ):
    if artifact.get("checkpoint_sha256") != expected_sha:
      raise RuntimeError(f"{name} artifact/checkpoint identity mismatch")

  cached = _load_json(cfg.reuse_factor_artifact) if cfg.reuse_factor_artifact else None
  if cached is not None and cfg.run_factor_rollouts:
    raise ValueError("reuse_factor_artifact and run_factor_rollouts are mutually exclusive")
  if cached is not None:
    factors = cached.get("factor_rollouts", {})
    sham = cached.get("sham_sentinel")
  else:
    factors = {
      factor: _run_factor(cfg, factor)
      for factor in cfg.factors
    } if cfg.run_factor_rollouts else {}
    sham = _run_sham(cfg) if cfg.run_sham_sentinel else None
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_complex_terrain_causal_diagnostic",
    "config": asdict(cfg),
    "checkpoint_sha256": expected_sha,
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
    "artifact_provenance": {
      "gait": {"path": str(Path(cfg.gait_baseline).resolve()), "sha256": _sha256(Path(cfg.gait_baseline).resolve())},
      "actuator": {"path": str(Path(cfg.actuator_baseline).resolve()), "sha256": _sha256(Path(cfg.actuator_baseline).resolve())},
      "triggered": {"path": str(Path(cfg.triggered_baseline).resolve()), "sha256": _sha256(Path(cfg.triggered_baseline).resolve())},
    },
    "single_variable_contract": {
      "factor_rollouts": list(cfg.factors),
      "unchanged": [
        "checkpoint", "policy", "terrain geometry", "commands", "termination",
        "gait", "reward", "network", "PPO",
      ],
      "training_started": False,
    },
    "known_metric_limitations": {
      "step_length": "existing formal field is liftoff-to-touchdown swing displacement, not touchdown-to-touchdown stride",
      "first_interval": "existing formal swing/stance duration can include a left-censored initial interval",
      "support_stability": "no support polygon/COP margin is available",
      "actuator": "actuator-space force only; no real torque-speed/power/thermal/latency envelope",
      "terminal_pd": "terminal reset-hook force is stale by one physics substep",
    },
    "factor_rollouts": factors,
    "factor_aggregates": {name: _aggregate_rows(value) for name, value in factors.items()},
    "sham_sentinel": sham,
  }
  payload["causal_classification"] = _classify(
    gait_baseline, factors, actuator_baseline, triggered_baseline, sham
  )
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  gait.configure_torch_backends()
  cfg = tyro.cli(ComplexTerrainConfig)
  payload = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps({
    "output": str(output),
    "checkpoint_sha256": payload["checkpoint_sha256"],
    "causal_classification": payload["causal_classification"],
  }, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
