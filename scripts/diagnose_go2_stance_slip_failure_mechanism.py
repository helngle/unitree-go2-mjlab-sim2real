"""Evaluation-only diagnosis for the rejected Go2 stance-slip training run.

The matrix is deliberately fixed to the registered V7 reference and the three
diagnostic checkpoints. It reuses the existing matched terrain placement,
gait telemetry, and pre-reset actuator capture without changing a training
task or contact parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_go2_high_slope_actuators as actuator
from scripts import diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE / (
  "logs/rsl_rl/go2_velocity/"
  "2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter"
)
CHECKPOINTS: dict[str, dict[str, str]] = {
  "v7": {
    "path": str(WORKSPACE / gait.V7_CHECKPOINT),
    "sha256": "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff",
    "task_id": "Unitree-Go2-Rough-V7",
  },
  "model_13700": {
    "path": str(RUN_DIR / "model_13700.pt"),
    "sha256": "02f9e821739babb844598b735da5aaac4d42d8a88f203431d28689d06519f2fc",
    "task_id": "Unitree-Go2-Rough-V7-StanceSlip",
  },
  "model_13900": {
    "path": str(RUN_DIR / "model_13900.pt"),
    "sha256": "4ab8740c7170b25923d4130b850fc77407f365923ad1634bb96a92ebf2eb8dea",
    "task_id": "Unitree-Go2-Rough-V7-StanceSlip",
  },
  "model_13999": {
    "path": str(RUN_DIR / "model_13999.pt"),
    "sha256": "db46dcc1272cb0a722b695568c8cdf4d086af1075cd4d0b53da7a75a643563e3",
    "task_id": "Unitree-Go2-Rough-V7-StanceSlip",
  },
}
CONDITIONS = (
  ("slope_up_high", "slope_up", 0),
  ("slope_up_extreme", "slope_up", 1),
)


@dataclass(frozen=True)
class MechanismConfig:
  checkpoint_labels: tuple[str, ...] = tuple(CHECKPOINTS)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 4
  warmup_steps: int = 100
  sample_steps: int = 2400
  stable_tail_steps: int = 300
  failure_windows: tuple[int, ...] = (50, 100)
  seed: int = 42
  device: str = "cuda:0"
  formal: bool = True
  chunk_mode: bool = False
  merge_inputs: tuple[str, ...] = ()
  output_file: str = str(
    RUN_DIR / "diagnostics/stance_slip_failure_mechanism_clean_seed42_r4_2400steps_v1.json"
  )


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git(*args: str) -> str:
  return subprocess.check_output(["git", *args], cwd=WORKSPACE, text=True).strip()


def _dirty_fingerprint() -> dict[str, Any]:
  status = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "-z"], cwd=WORKSPACE,
  )
  diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=WORKSPACE)
  return {
    "dirty": bool(status),
    "status_sha256": hashlib.sha256(status).hexdigest(),
    "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "status_porcelain": status.decode("utf-8").rstrip("\0").replace("\0", "\n"),
  }


def _validate_config(cfg: MechanismConfig, *, verify_files: bool = True) -> None:
  if cfg.merge_inputs:
    if cfg.chunk_mode:
      raise ValueError("merge_inputs and chunk_mode are mutually exclusive")
    return
  if not cfg.checkpoint_labels or len(set(cfg.checkpoint_labels)) != len(cfg.checkpoint_labels):
    raise ValueError("checkpoint_labels must be nonempty and unique")
  if any(label not in CHECKPOINTS for label in cfg.checkpoint_labels):
    raise ValueError(f"checkpoint_labels must be drawn from {tuple(CHECKPOINTS)}")
  if cfg.chunk_mode and len(cfg.checkpoint_labels) != 1:
    raise ValueError("chunk_mode requires exactly one registered checkpoint")
  if cfg.formal and not cfg.chunk_mode and cfg.checkpoint_labels != tuple(CHECKPOINTS):
    raise ValueError("formal matrix requires V7 and all three registered checkpoints")
  if cfg.speeds != (0.3, 0.5) or cfg.seed != 42:
    raise ValueError("speeds and seed are frozen at (0.3, 0.5) and 42")
  if cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps <= 0:
    raise ValueError("repeats/sample_steps must be positive and warmup nonnegative")
  if cfg.stable_tail_steps <= 0 or cfg.stable_tail_steps > cfg.sample_steps:
    raise ValueError("stable_tail_steps must fit within sample_steps")
  if any(window <= 0 or window > cfg.sample_steps for window in cfg.failure_windows):
    raise ValueError("failure windows must fit within sample_steps")
  if cfg.formal and (cfg.repeats < 4 or cfg.warmup_steps != 100 or cfg.sample_steps != 2400):
    raise ValueError("formal matrix requires >=4 repeats, 100 warmup, and 2400 sample steps")
  if verify_files:
    for label in cfg.checkpoint_labels:
      identity = CHECKPOINTS[label]
      path = Path(identity["path"])
      if not path.is_file() or _sha256(path) != identity["sha256"]:
        raise ValueError(f"checkpoint identity mismatch: {label} {path}")


def _weighted_mean(stats: list[dict[str, Any]]) -> float | None:
  available = [item for item in stats if item.get("mean") is not None and int(item.get("count", 0)) > 0]
  if not available:
    return None
  count = sum(int(item["count"]) for item in available)
  return sum(float(item["mean"]) * int(item["count"]) for item in available) / count


def _mean(values: list[float | None]) -> float | None:
  finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
  return None if not finite else sum(finite) / len(finite)


def _gait_row_summary(row: dict[str, Any], control_dt: float) -> dict[str, Any]:
  loaded = _mean([float(value) for value in row["loaded_contact_fraction"]])
  stance = _weighted_mean(row["stance_duration"])
  swing = _weighted_mean(row["swing_duration"])
  duty = _mean(row["duty_factor"])
  actual_vx = float(row["actual_velocity_mean"][0])
  return {
    "matched_slot": int(row["matched_slot"]),
    "speed": float(row["speed"]),
    "repeat": int(row["repeat"]),
    "sample_count": int(row["sample_count"]),
    "failed": bool(row["failed"]),
    "first_failure_reason": row["first_failure_reason"],
    "response_gain_vx": row["response_gain"]["vx"],
    "actual_vx": actual_vx,
    "forward_displacement_proxy_m": actual_vx * int(row["sample_count"]) * control_dt,
    "loaded_contact_fraction_20_10_hysteresis": loaded,
    "tangent_slip_loaded_20_10": _weighted_mean(row["foot_slip_tangent"]),
    "normal_force_loaded_20_10": _weighted_mean(row["foot_force_normal"]),
    "tangent_force_loaded_20_10": _weighted_mean(row["foot_force_tangent"]),
    "step_length_absolute": _weighted_mean(row["step_length_absolute"]),
    "stance_duration_s": stance,
    "swing_duration_s": swing,
    "duty_factor_completed_intervals": duty,
    "swing_clearance": _weighted_mean(row["terrain_relative_clearance"]),
    "action_acceleration": row["action_acceleration"]["mean"],
    "base_pitch": row["base_pitch"]["mean"],
  }


def _pair_rows(
  reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  ref = {(row["matched_slot"], row["speed"], row["repeat"]): row for row in reference}
  cand = {(row["matched_slot"], row["speed"], row["repeat"]): row for row in candidate}
  if ref.keys() != cand.keys():
    raise RuntimeError("gait matched identities differ between checkpoints")
  output = []
  for key in sorted(ref):
    a, b = ref[key], cand[key]
    common = min(int(a["sample_count"]), int(b["sample_count"]))
    slip_down = (
      a["tangent_slip_loaded_20_10"] is not None
      and b["tangent_slip_loaded_20_10"] is not None
      and b["tangent_slip_loaded_20_10"] < a["tangent_slip_loaded_20_10"]
    )
    gain_down = (
      a["response_gain_vx"] is not None and b["response_gain_vx"] is not None
      and b["response_gain_vx"] < a["response_gain_vx"]
    )
    gait_conservative = any((
      a["step_length_absolute"] is not None and b["step_length_absolute"] is not None
      and b["step_length_absolute"] < a["step_length_absolute"],
      a["duty_factor_completed_intervals"] is not None
      and b["duty_factor_completed_intervals"] is not None
      and abs(b["duty_factor_completed_intervals"] - a["duty_factor_completed_intervals"]) >= 0.02,
      a["stance_duration_s"] is not None and b["stance_duration_s"] is not None
      and abs(b["stance_duration_s"] - a["stance_duration_s"]) >= 0.1 * max(a["stance_duration_s"], 1.0e-8),
    ))
    output.append({
      "identity": {"matched_slot": key[0], "speed": key[1], "repeat": key[2]},
      "common_active_prefix_steps": common,
      "equal_full_horizon": common == int(a["sample_count"]) == int(b["sample_count"]),
      "reference": a,
      "candidate": b,
      "direction_flags": {
        "slip_down": slip_down,
        "gain_down": gain_down,
        "gait_conservative": gait_conservative,
        "reward_avoidance_pattern": slip_down and gain_down and gait_conservative,
      },
    })
  return output


def _direction_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
  full = [pair for pair in pairs if pair["equal_full_horizon"]]
  return {
    "pair_count": len(pairs),
    "equal_full_horizon_pair_count": len(full),
    "slip_and_gain_down_count": sum(
      pair["direction_flags"]["slip_down"] and pair["direction_flags"]["gain_down"]
      for pair in full
    ),
    "reward_avoidance_pattern_count": sum(
      pair["direction_flags"]["reward_avoidance_pattern"] for pair in full
    ),
    "time_order_available": False,
    "time_order_limitation": (
      "gait intervals are aggregate over each active attempt; the actuator audit provides "
      "true pre-reset terminal windows, but no registered onset for gait conservatism"
    ),
  }


def _comparisons(models: dict[str, Any]) -> dict[str, Any]:
  if "v7" not in models:
    return {}
  output: dict[str, Any] = {}
  for label in CHECKPOINTS:
    if label == "v7" or label not in models:
      continue
    by_condition: dict[str, Any] = {}
    for condition, _, _ in CONDITIONS:
      pairs = _pair_rows(
        models["v7"]["gait"][condition]["rows"],
        models[label]["gait"][condition]["rows"],
      )
      by_condition[condition] = {
        "direction_summary": _direction_summary(pairs),
        "pairs": pairs,
      }
    output[label] = by_condition
  return output


def evaluate(cfg: MechanismConfig) -> dict[str, Any]:
  _validate_config(cfg)
  models: dict[str, Any] = {}
  for label in cfg.checkpoint_labels:
    identity = CHECKPOINTS[label]
    gait_conditions: dict[str, Any] = {}
    for condition, terrain_kind, level in CONDITIONS:
      gait_cfg = gait.GaitConfig(
        checkpoint=identity["path"], task_id=identity["task_id"],
        profiles=("clean",), speeds=cfg.speeds, repeats=cfg.repeats,
        warmup_steps=cfg.warmup_steps, sample_steps=cfg.sample_steps,
        seed=cfg.seed, device=cfg.device,
      )
      result = gait._evaluate_condition(
        # The legacy gait helper validates this label against its original
        # three-condition CLI matrix; terrain geometry is selected separately
        # by terrain_kind and level.
        gait_cfg, "clean", "slope_up_high", terrain_kind, level
      )
      control_dt = float(result["episode_settings"]["control_dt"])
      gait_conditions[condition] = {
        "terrain_assignment_position_error_max": result["terrain_assignment_position_error_max"],
        "terrain_placement_position_error_max": result["terrain_placement_position_error_max"],
        "control_dt": control_dt,
        "rows": [_gait_row_summary(row, control_dt) for row in result["scenarios"]],
      }
    audit_cfg = actuator.AuditConfig(
      checkpoint=identity["path"], task_id=identity["task_id"], profile="clean",
      speeds=cfg.speeds, repeats=cfg.repeats, warmup_steps=cfg.warmup_steps,
      sample_steps=cfg.sample_steps, stable_tail_steps=cfg.stable_tail_steps,
      failure_windows=cfg.failure_windows, seed=cfg.seed, device=cfg.device,
    )
    actuator_result = actuator._audit_condition(
      audit_cfg, "slope_up_high", "slope_up", 0
    )
    models[label] = {
      "identity": identity,
      "gait": gait_conditions,
      "actuator": actuator_result,
    }

  source_files = [
    Path(__file__).resolve(), Path(gait.__file__).resolve(), Path(actuator.__file__).resolve(),
    WORKSPACE / "src/tasks/velocity/mdp/rewards.py",
  ]
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_stance_slip_failure_mechanism_diagnosis",
    "config": asdict(cfg),
    "provenance": {
      "git_branch": _git("branch", "--show-current"),
      "git_head": _git("rev-parse", "HEAD"),
      "dirty_state": _dirty_fingerprint(),
      "source_sha256": {str(path): _sha256(path) for path in source_files},
      "checkpoint_identities": {label: CHECKPOINTS[label] for label in cfg.checkpoint_labels},
    },
    "metric_contract": {
      "training_changed": False,
      "friction_changed": False,
      "route_scope": "straight commands on clean slope-up high and extreme",
      "horizon": "100 warmup plus 2400 active-attempt sample steps in formal mode",
      "reward_metric_source": (
        "the existing formal straight/arc/S acceptance JSON remains authoritative for the exact "
        "signed 15 N loaded mask and frozen slip cost"
      ),
      "gait_state": "separate 20 N on / 10 N off hysteretic state; never substituted for reward loaded15",
      "terminal_actuator_capture": (
        "post-step pre-reset snapshot captured by intercepting env._reset_idx; reset episode excluded"
      ),
      "gait_terminal_limitation": (
        "gait diagnostic uses explicit pre-step fallback for a reset row; suitable for attempt-level "
        "gait distributions, not registered failure-onset ordering"
      ),
    },
    "models": models,
    "comparisons_to_v7": _comparisons(models),
  }
  assert_recursive_json_finite(payload)
  return payload


def merge_artifacts(cfg: MechanismConfig) -> dict[str, Any]:
  _validate_config(cfg)
  if len(cfg.merge_inputs) != len(CHECKPOINTS):
    raise ValueError("merge requires exactly four one-checkpoint chunk artifacts")
  inputs: list[tuple[Path, dict[str, Any]]] = []
  models: dict[str, Any] = {}
  contract: dict[str, Any] | None = None
  expected_fields = (
    "speeds", "repeats", "warmup_steps", "sample_steps", "stable_tail_steps",
    "failure_windows", "seed", "device", "formal",
  )
  expected_config: dict[str, Any] | None = None
  for name in cfg.merge_inputs:
    path = Path(name).expanduser().resolve()
    payload = json.loads(path.read_text())
    assert_recursive_json_finite(payload)
    if payload.get("evaluation_suite") != "go2_stance_slip_failure_mechanism_diagnosis":
      raise ValueError(f"unexpected chunk suite: {path}")
    chunk_cfg = payload["config"]
    selected = {field: chunk_cfg[field] for field in expected_fields}
    if expected_config is None:
      expected_config = selected
    elif selected != expected_config:
      raise ValueError(f"chunk evaluation config mismatch: {path}")
    if not chunk_cfg.get("chunk_mode") or len(payload["models"]) != 1:
      raise ValueError(f"input is not a one-checkpoint chunk: {path}")
    for label, model in payload["models"].items():
      if label in models or label not in CHECKPOINTS:
        raise ValueError(f"duplicate or unknown chunk checkpoint: {label}")
      if model["identity"] != CHECKPOINTS[label]:
        raise ValueError(f"chunk checkpoint identity mismatch: {label}")
      models[label] = model
    if contract is None:
      contract = payload["metric_contract"]
    elif payload["metric_contract"] != contract:
      raise ValueError(f"chunk metric contract mismatch: {path}")
    inputs.append((path, payload))
  if set(models) != set(CHECKPOINTS):
    raise ValueError(f"merged checkpoint set mismatch: {tuple(models)}")
  merged_cfg = asdict(cfg)
  merged_cfg["checkpoint_labels"] = list(CHECKPOINTS)
  payload = {
    "schema_version": 1,
    "evaluation_suite": "go2_stance_slip_failure_mechanism_diagnosis",
    "config": merged_cfg,
    "provenance": {
      "git_branch": _git("branch", "--show-current"),
      "git_head": _git("rev-parse", "HEAD"),
      "dirty_state": _dirty_fingerprint(),
      "checkpoint_identities": CHECKPOINTS,
      "chunk_artifact_sha256": {str(path): _sha256(path) for path, _ in inputs},
      "chunk_provenance": {next(iter(item["models"])): item["provenance"] for _, item in inputs},
    },
    "metric_contract": contract,
    "models": models,
    "comparisons_to_v7": _comparisons(models),
    "technical_execution_note": (
      "The identical matrix was executed as one checkpoint per process after repeated "
      "MuJoCo-Warp environment construction in one process exhausted CUDA graph resources."
    ),
  }
  assert_recursive_json_finite(payload)
  return payload


def main() -> None:
  cfg = tyro.cli(MechanismConfig)
  output = Path(cfg.output_file).expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
  payload = merge_artifacts(cfg) if cfg.merge_inputs else evaluate(cfg)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  print(json.dumps({
    "output": str(output),
    "models": list(payload["models"]),
    "comparisons": {
      model: {
        condition: value["direction_summary"]
        for condition, value in conditions.items()
      }
      for model, conditions in payload["comparisons_to_v7"].items()
    },
  }, indent=2, allow_nan=False))


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
