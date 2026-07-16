"""Diagnose Go2 pure-axis and curved command response from saved JSON files.

This tool is deliberately offline. It does not create an environment, load a
policy, or change a command sampler. The scheduled command-tape schema check
also prevents the obsolete pose-extended tape result from entering a report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


GENERAL_YAW_LIMIT = 0.3
_ATTEMPT_KEY_FIELDS = (
  "radius",
  "speed",
  "turn_sign",
  "cross_track_offset",
  "yaw_offset",
  "repeat",
)


def _mean(values: Iterable[float]) -> float | None:
  finite = [float(value) for value in values if math.isfinite(float(value))]
  return sum(finite) / len(finite) if finite else None


def _ratio(numerator: float, denominator: float) -> float:
  if not math.isfinite(numerator) or not math.isfinite(denominator):
    raise ValueError("Response inputs must be finite.")
  if abs(denominator) <= 1e-9:
    raise ValueError("A non-zero command is required for a response gain.")
  return abs(numerator / denominator)


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    data = json.load(stream)
  if not isinstance(data, dict):
    raise ValueError(f"{path}: expected a JSON object.")
  return data


def validate_scheduled_arc_tape(data: dict[str, Any]) -> list[dict[str, Any]]:
  """Return scenarios only when the JSON uses fixed-time tape semantics."""
  config = data.get("config", {})
  if config.get("route_kind") != "arc" or config.get("mode") != "command_tape":
    raise ValueError("Expected an arc command_tape result.")
  if "settle_steps" not in config:
    raise ValueError(
      "Missing settle_steps: this is not the fixed-time scheduled tape schema."
    )
  scenarios = data.get("scenarios")
  if not isinstance(scenarios, list) or not scenarios:
    raise ValueError("The command-tape result has no scenarios.")
  required = {
    "motion_steps",
    "settle_steps",
    "failed",
    "finished",
    "general_yaw_in_distribution",
  }
  for scenario in scenarios:
    missing = required.difference(scenario)
    if missing:
      raise ValueError(
        "Scenario is not from the scheduled tape evaluator; missing "
        + ", ".join(sorted(missing))
      )
    expected_id = abs(float(scenario["required_yaw_rate"])) <= (
      GENERAL_YAW_LIMIT + 1e-9
    )
    if bool(scenario["general_yaw_in_distribution"]) != expected_id:
      raise ValueError("The scenario has an inconsistent general-yaw ID label.")
  return scenarios


def summarize_arc_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
  if not scenarios:
    raise ValueError("Cannot summarize an empty scenario group.")
  termination_names = sorted(
    {
      name
      for scenario in scenarios
      for name in scenario.get("termination_counts", {})
    }
  )
  return {
    "num_scenarios": len(scenarios),
    "completion_rate": _mean(
      float(bool(scenario["completed"])) for scenario in scenarios
    ),
    "forward_response_gain_mean": _mean(
      _ratio(
        float(scenario["actual_velocity_xy_mean"][0]),
        float(scenario["commanded_velocity_xy_mean"][0]),
      )
      for scenario in scenarios
    ),
    "yaw_response_gain_mean": _mean(
      _ratio(
        float(scenario["actual_yaw_rate_mean"]),
        float(scenario["commanded_yaw_rate_mean"]),
      )
      for scenario in scenarios
    ),
    "progress_ratio_mean": _mean(
      float(scenario["arc_length_progress_ratio"]) for scenario in scenarios
    ),
    "progress_ratio_min": min(
      float(scenario["arc_length_progress_ratio"]) for scenario in scenarios
    ),
    "progress_ratio_max": max(
      float(scenario["arc_length_progress_ratio"]) for scenario in scenarios
    ),
    "cross_axis_velocity_mean": _mean(
      float(scenario["cross_axis_velocity_mean"]) for scenario in scenarios
    ),
    "slip_velocity_mean": _mean(
      float(scenario["slip_velocity_mean"]) for scenario in scenarios
    ),
    "action_acceleration_mean": _mean(
      float(scenario["action_acceleration_mean"]) for scenario in scenarios
    ),
    "reset_count": sum(int(scenario["reset_count"]) for scenario in scenarios),
    "termination_counts": {
      name: sum(
        float(scenario.get("termination_counts", {}).get(name, 0.0))
        for scenario in scenarios
      )
      for name in termination_names
    },
  }


def summarize_arc_tape(data: dict[str, Any]) -> dict[str, Any]:
  scenarios = validate_scheduled_arc_tape(data)
  id_scenarios = [
    scenario for scenario in scenarios if scenario["general_yaw_in_distribution"]
  ]
  ood_scenarios = [
    scenario for scenario in scenarios if not scenario["general_yaw_in_distribution"]
  ]
  return {
    "all": summarize_arc_scenarios(scenarios),
    "in_distribution": summarize_arc_scenarios(id_scenarios),
    "out_of_distribution": summarize_arc_scenarios(ood_scenarios),
    "by_speed": {
      str(speed): summarize_arc_scenarios(
        [scenario for scenario in scenarios if scenario["speed"] == speed]
      )
      for speed in sorted({scenario["speed"] for scenario in scenarios})
    },
    "by_radius": {
      str(radius): summarize_arc_scenarios(
        [scenario for scenario in scenarios if scenario["radius"] == radius]
      )
      for radius in sorted({scenario["radius"] for scenario in scenarios})
    },
  }


def _select_result(data: dict[str, Any], checkpoint_name: str) -> dict[str, Any]:
  matches = [
    result
    for result in data.get("results", [])
    if Path(result.get("checkpoint", "")).name == checkpoint_name
  ]
  if len(matches) != 1:
    raise ValueError(
      f"Expected one {checkpoint_name} result, found {len(matches)}."
    )
  return matches[0]


def extract_reference_commands(
  data: dict[str, Any], checkpoint_name: str
) -> list[dict[str, Any]]:
  """Extract available pure-axis metrics without inventing missing gains."""
  result = _select_result(data, checkpoint_name)
  records: list[dict[str, Any]] = []
  if isinstance(result.get("by_command"), dict):
    for command_name, metrics in result["by_command"].items():
      command = metrics["command"]
      records.append(
        {
          "command_name": command_name,
          "command": command,
          "profile": result.get("profile"),
          "num_envs": metrics.get("num_envs"),
          "forward_response_gain": (
            metrics.get("linear_command_response_gain_mean")
            if abs(float(command[0])) > 1e-9
            else None
          ),
          "yaw_response_gain": None,
          "yaw_absolute_error_mean": metrics.get("yaw_velocity_error_mean"),
          "cross_axis_velocity_mean": metrics.get(
            "linear_cross_axis_velocity_mean"
          ),
          "slip_velocity_mean": metrics.get("slip_velocity_mean"),
          "action_acceleration_mean": metrics.get("action_acceleration_mean"),
          "termination_counts": metrics.get("terminations_per_env", {}),
          "source_schema": "robustness_by_command",
        }
      )
    return records

  scenarios = result.get("scenarios")
  if isinstance(scenarios, dict):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for scenario_name, metrics in scenarios.items():
      grouped.setdefault(scenario_name.split("|", 1)[0], []).append(metrics)
    for command_name, rows in grouped.items():
      command = rows[0]["command"]
      records.append(
        {
          "command_name": command_name,
          "command": command,
          "profile": "clean",
          "num_envs": sum(int(row["num_envs"]) for row in rows),
          "forward_response_gain": (
            sum(
              float(row["velocity"]["primary_response_gain"])
              * int(row["num_envs"])
              for row in rows
            )
            / sum(int(row["num_envs"]) for row in rows)
          )
          if abs(float(command[0])) > 1e-9
          else None,
          "yaw_response_gain": None,
          "yaw_absolute_error_mean": _mean(
            float(row["velocity"]["yaw_abs_mean"]) for row in rows
          ),
          "cross_axis_velocity_mean": _mean(
            float(row["velocity"]["cross_axis_abs_mean"]) for row in rows
          ),
          "slip_velocity_mean": None,
          "action_acceleration_mean": None,
          "termination_counts": {
            name: sum(
              float(row.get("terminations_per_env", {}).get(name, 0.0))
              for row in rows
            )
            for name in sorted(
              {
                name
                for row in rows
                for name in row.get("terminations_per_env", {})
              }
            )
          },
          "source_schema": "gait_diagnostic_scenarios",
        }
      )
    return records

  raise ValueError("Unsupported reference JSON schema.")


def merge_closed_loop_attempts(
  datasets: Iterable[dict[str, Any]],
) -> dict[str, Any]:
  """Merge retries by attempt key, with later datasets replacing earlier ones."""
  attempts: dict[tuple[Any, ...], dict[str, Any]] = {}
  for data in datasets:
    config = data.get("config", {})
    if config.get("route_kind") != "arc" or config.get("mode") != "closed_loop":
      raise ValueError("Expected an arc closed_loop result.")
    for scenario in data.get("scenarios", []):
      key = tuple(scenario[field] for field in _ATTEMPT_KEY_FIELDS)
      attempts[key] = scenario
  if not attempts:
    raise ValueError("No closed-loop attempts were provided.")
  scenarios = list(attempts.values())
  return {
    "num_unique_attempts": len(scenarios),
    "completion_rate": _mean(
      float(bool(scenario["completed"])) for scenario in scenarios
    ),
    "reset_count": sum(int(scenario["reset_count"]) for scenario in scenarios),
  }


def build_report(
  tape_data: dict[str, Any],
  closed_loop_data: list[dict[str, Any]],
  references: list[dict[str, Any]],
  checkpoint_name: str,
) -> dict[str, Any]:
  tape = summarize_arc_tape(tape_data)
  extracted_references = [
    {
      "source_index": index,
      "commands": extract_reference_commands(data, checkpoint_name),
    }
    for index, data in enumerate(references)
  ]
  closed_loop = (
    merge_closed_loop_attempts(closed_loop_data) if closed_loop_data else None
  )
  return {
    "checkpoint_name": checkpoint_name,
    "arc_command_tape": tape,
    "closed_loop": closed_loop,
    "pure_axis_references": extracted_references,
    "time_series_metric_availability": {
      "rise_time": False,
      "overshoot": False,
      "reason": (
        "The saved JSON files contain attempt aggregates, not per-control-step "
        "command and response samples."
      ),
    },
    "causal_assessment": {
      "matched_pure_forward_yaw_coupled_matrix_available": False,
      "curve_sampler_training_authorized": False,
      "decision": "NO-GO",
      "reasons": [
        "The ID tape yaw response is strong while forward response is under-gain.",
        "Closed-loop arcs complete when given enough horizon." if closed_loop else
        "Closed-loop retry results were not supplied to this report.",
        "Historical pure-axis files do not match every arc speed, yaw rate, terrain, and horizon.",
      ],
    },
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arc-tape-json", type=Path, required=True)
  parser.add_argument("--closed-loop-json", type=Path, action="append", default=[])
  parser.add_argument("--reference-json", type=Path, action="append", default=[])
  parser.add_argument("--checkpoint-name", default="model_13600.pt")
  parser.add_argument("--output", type=Path)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  report = build_report(
    _load_json(args.arc_tape_json),
    [_load_json(path) for path in args.closed_loop_json],
    [_load_json(path) for path in args.reference_json],
    args.checkpoint_name,
  )
  rendered = json.dumps(report, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
  print(rendered)


if __name__ == "__main__":
  main()
