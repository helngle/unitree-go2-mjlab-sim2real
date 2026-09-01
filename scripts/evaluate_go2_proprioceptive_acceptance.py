"""Build or execute the frozen all-checkpoint proprio acceptance matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.velocity.evaluation.proprio_acceptance import (
  SAFETY_METRICS,
  SIM2REAL_RANDOMIZATION_EVENTS,
  canonical_profile_name,
  checkpoint_task_id,
  formal_evaluation_provenance,
  load_checkpoint_lineage,
  select_checkpoint_bundles,
  sha256_file,
  validate_formal_checkpoint_schedule,
)
from src.tasks.velocity.evaluation.terrain_rollout_metrics import (
  assert_recursive_json_finite,
)


WORKSPACE = Path(__file__).resolve().parents[1]
V7_CHECKPOINT = WORKSPACE / (
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
V7_SHA256 = "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
V7_TASK = "Unitree-Go2-Rough-V7"
EXPECTED_ACCEPTANCE_ARTIFACTS = frozenset({
  "high_slope_matched",
  "flat_matched_routes",
  *(
    artifact
    for profile in ("clean", "randomized")
    for artifact in (
      f"continuous_retained_{profile}",
      f"terrain_curves_arc_{profile}",
      f"terrain_curves_s_curve_{profile}",
      f"stairs_level9_seed42_{profile}",
      f"stairs_level9_seed43_{profile}",
      f"stairs_level9_seed44_{profile}",
    )
  ),
})


@dataclass(frozen=True)
class Invocation:
  checkpoint_label: str
  checkpoint_path: str
  checkpoint_sha256: str
  artifact_id: str
  canonical_profiles: tuple[str, ...]
  command: tuple[str, ...]
  output_file: str


def validate_invocation_inventory(invocations: list[Invocation]) -> None:
  """Require the exact preregistered 14 artifacts for every checkpoint."""
  by_checkpoint: dict[str, list[Invocation]] = defaultdict(list)
  for invocation in invocations:
    by_checkpoint[invocation.checkpoint_path].append(invocation)
  if not by_checkpoint:
    raise ValueError("acceptance matrix is empty")
  for checkpoint, items in by_checkpoint.items():
    artifact_ids = [item.artifact_id for item in items]
    if len(artifact_ids) != len(set(artifact_ids)):
      raise ValueError(f"duplicate acceptance invocation for {checkpoint}")
    observed = set(artifact_ids)
    if len(items) != 14 or observed != EXPECTED_ACCEPTANCE_ARTIFACTS:
      missing = sorted(EXPECTED_ACCEPTANCE_ARTIFACTS - observed)
      extra = sorted(observed - EXPECTED_ACCEPTANCE_ARTIFACTS)
      raise ValueError(
        f"acceptance invocation inventory mismatch for {checkpoint}: "
        f"count={len(items)}, missing={missing}, extra={extra}"
      )
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path == V7_CHECKPOINT.resolve():
      expected_task = V7_TASK
      expected_sha = V7_SHA256
    else:
      expected_task = checkpoint_task_id(checkpoint_path)
      expected_sha = sha256_file(checkpoint_path)
    for item in items:
      expected_profiles = (
        ("clean", "randomized")
        if item.artifact_id in {"high_slope_matched", "flat_matched_routes"}
        else (
          ("randomized",)
          if item.artifact_id.endswith("_randomized") else ("clean",)
        )
      )
      if item.canonical_profiles != expected_profiles:
        raise ValueError(
          f"acceptance invocation profile mismatch for {checkpoint}: "
          f"{item.artifact_id}"
        )
      if item.checkpoint_sha256 != expected_sha:
        raise ValueError(
          f"acceptance invocation checkpoint SHA mismatch for {checkpoint}"
        )
      common = ("--checkpoint", checkpoint, "--task-id", expected_task)
      if item.artifact_id == "high_slope_matched":
        script = "evaluate_go2_high_slope_matched.py"
        arguments = common + (
          "--profiles", "clean", "randomized", "--radii", "2.5",
        )
      elif item.artifact_id == "flat_matched_routes":
        script = "evaluate_go2_matched_routes.py"
        arguments = common + ("--profiles", "clean", "full_randomized")
      else:
        profile = item.canonical_profiles[0]
        profile_common = common + ("--profile", profile)
        if item.artifact_id.startswith("continuous_retained_"):
          script = "evaluate_go2_terrain_boundary.py"
          arguments = profile_common + (
            "--suite", "continuous_straight", "--route-kind", "straight",
            "--transition-cases", "random_rough", "discrete_obstacle",
            "stairs_up", "stairs_down", "--transition-levels", "3", "5", "7",
          )
        elif item.artifact_id.startswith("terrain_curves_"):
          route = "s_curve" if "_s_curve_" in item.artifact_id else "arc"
          script = "evaluate_go2_terrain_curves.py"
          arguments = profile_common + (
            "--route-kind", route, "--terrain-kinds", "slope_up", "slope_down",
            "random_rough", "discrete_obstacle", "stairs_up", "stairs_down",
          )
        else:
          seed = item.artifact_id.removeprefix("stairs_level9_seed").split("_")[0]
          script = "evaluate_go2_terrain_boundary.py"
          arguments = profile_common + (
            "--suite", "continuous_straight", "--route-kind", "straight",
            "--transition-cases", "stairs_up", "stairs_down",
            "--transition-levels", "9", "--seed", seed,
          )
      expected_output = (
        Path(item.output_file).expanduser().resolve().parent
        / f"{item.artifact_id}.json"
      )
      if (
        Path(item.output_file).expanduser().resolve() != expected_output
        or expected_output.parent.name != checkpoint_path.stem
      ):
        raise ValueError(
          f"acceptance invocation output mismatch for {checkpoint}: "
          f"{item.artifact_id}"
        )
      expected_command = (
        sys.executable, str(WORKSPACE / "scripts" / script), *arguments,
        "--output-file", item.output_file,
      )
      if item.command != expected_command:
        raise ValueError(
          f"acceptance invocation command mismatch for {checkpoint}: "
          f"{item.artifact_id}"
        )


def discover_checkpoints(run_dir: str | Path) -> list[Path]:
  root = Path(run_dir).expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(root)
  paths = []
  for path in root.glob("model_*.pt"):
    suffix = path.stem.removeprefix("model_")
    if suffix.isdigit():
      paths.append((int(suffix), path.resolve()))
  if not paths:
    raise ValueError(f"no model_*.pt checkpoints found in {root}")
  return [path for _, path in sorted(paths)]


def _invocation(
  checkpoint: Path,
  task_id: str,
  output_dir: Path,
  artifact_id: str,
  script: str,
  arguments: tuple[str, ...],
  profiles: tuple[str, ...],
) -> Invocation:
  output = output_dir / checkpoint.stem / f"{artifact_id}.json"
  command = (
    sys.executable, str(WORKSPACE / "scripts" / script),
    *arguments, "--output-file", str(output),
  )
  return Invocation(
    checkpoint_label=checkpoint.stem,
    checkpoint_path=str(checkpoint),
    checkpoint_sha256=sha256_file(checkpoint),
    artifact_id=artifact_id,
    canonical_profiles=profiles,
    command=command,
    output_file=str(output),
  )


def build_matrix(
  run_dir: str | Path | None, output_dir: str | Path, *, include_v7: bool = True,
  checkpoints: list[Path] | None = None,
) -> list[Invocation]:
  output_root = Path(output_dir).expanduser().resolve()
  if checkpoints is None:
    if run_dir is None:
      raise ValueError("run_dir or checkpoints is required")
    checkpoints = discover_checkpoints(run_dir)
  checkpoint_tasks = [(path, checkpoint_task_id(path)) for path in checkpoints]
  if include_v7:
    if sha256_file(V7_CHECKPOINT) != V7_SHA256:
      raise RuntimeError("locked V7 checkpoint SHA256 mismatch")
    checkpoint_tasks.insert(0, (V7_CHECKPOINT.resolve(), V7_TASK))
  invocations: list[Invocation] = []
  for checkpoint, task_id in checkpoint_tasks:
    common = ("--checkpoint", str(checkpoint), "--task-id", task_id)
    invocations.append(_invocation(
      checkpoint, task_id, output_root, "high_slope_matched",
      "evaluate_go2_high_slope_matched.py",
      common + ("--profiles", "clean", "randomized", "--radii", "2.5"),
      ("clean", "randomized"),
    ))
    invocations.append(_invocation(
      checkpoint, task_id, output_root, "flat_matched_routes",
      "evaluate_go2_matched_routes.py",
      common + ("--profiles", "clean", "full_randomized"),
      ("clean", "randomized"),
    ))
    for profile in ("clean", "randomized"):
      profile_common = common + ("--profile", profile)
      invocations.append(_invocation(
        checkpoint, task_id, output_root, f"continuous_retained_{profile}",
        "evaluate_go2_terrain_boundary.py",
        profile_common + (
          "--suite", "continuous_straight", "--route-kind", "straight",
          "--transition-cases",
          "random_rough", "discrete_obstacle", "stairs_up", "stairs_down",
          "--transition-levels", "3", "5", "7",
        ), (profile,),
      ))
      for route in ("arc", "s_curve"):
        invocations.append(_invocation(
          checkpoint, task_id, output_root, f"terrain_curves_{route}_{profile}",
          "evaluate_go2_terrain_curves.py",
          profile_common + (
            "--route-kind", route, "--terrain-kinds", "slope_up", "slope_down",
            "random_rough", "discrete_obstacle", "stairs_up", "stairs_down",
          ), (profile,),
        ))
      for seed in (42, 43, 44):
        invocations.append(_invocation(
          checkpoint, task_id, output_root, f"stairs_level9_seed{seed}_{profile}",
          "evaluate_go2_terrain_boundary.py",
          profile_common + (
            "--suite", "continuous_straight", "--route-kind", "straight",
            "--transition-cases",
            "stairs_up", "stairs_down", "--transition-levels", "9",
            "--seed", str(seed),
          ), (profile,),
        ))
  validate_invocation_inventory(invocations)
  return invocations


def _write_strict(path: Path, payload: Any) -> None:
  if path.exists():
    raise FileExistsError(f"refusing to overwrite formal artifact: {path}")
  assert_recursive_json_finite(payload)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _execute(
  invocations: list[Invocation], *, resume_existing: bool = False
) -> list[dict[str, str]]:
  artifacts = []
  for invocation in invocations:
    output = Path(invocation.output_file)
    if output.exists():
      if not resume_existing:
        raise FileExistsError(f"refusing to overwrite evaluator artifact: {output}")
    else:
      output.parent.mkdir(parents=True, exist_ok=True)
      subprocess.run(invocation.command, cwd=WORKSPACE, check=True)
    payload = json.loads(output.read_text())
    assert_recursive_json_finite(payload)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
      raise ValueError(f"evaluator artifact provenance is missing: {output}")
    identity = provenance.get("checkpoint")
    if not isinstance(identity, dict) or (
      identity.get("path") != invocation.checkpoint_path
      or identity.get("sha256") != invocation.checkpoint_sha256
    ):
      raise ValueError(f"evaluator artifact checkpoint mismatch: {output}")
    artifacts.append({"path": str(output), "sha256": sha256_file(output)})
  return artifacts


def _mean(values: list[float]) -> float:
  if not values:
    raise ValueError("cannot aggregate an empty metric group")
  return sum(values) / len(values)


def _scenario_metric(scenario: dict[str, Any], name: str) -> float:
  terrain = scenario.get("terrain_rollout_metrics", {})
  contacts = terrain.get("body_contacts", {})
  if name == "terrain_tangent_slip":
    value = scenario.get("terrain_tangent_stance_slip_mean")
  elif name == "base_pitch_absolute":
    value = terrain.get("base_pitch_absolute", {}).get("mean")
  elif name == "action_acceleration":
    value = terrain.get("action_acceleration", {}).get("mean")
  elif name == "actuator_effort":
    value = terrain.get("actuator_effort_abs", {}).get("mean")
  elif name == "mechanical_power":
    value = terrain.get("mechanical_power_abs", {}).get("mean")
  elif name == "mechanical_energy":
    value = terrain.get("mechanical_energy_abs")
  elif name in {"base_contact", "upper_leg_contact", "calf_contact"}:
    key = name.removesuffix("_contact")
    value = contacts.get(key, {}).get("non_terminating_rate")
  elif name == "failure_risk":
    value = float(bool(scenario.get("failed") or scenario.get("catastrophic_termination")))
  else:
    raise KeyError(name)
  if value is None:
    raise ValueError(f"scenario is missing unified safety metric {name}")
  return float(value)


def _forward_metrics(scenario: dict[str, Any]) -> tuple[bool, float, float]:
  command = scenario.get("commanded_velocity_mean")
  actual = scenario.get("actual_velocity_mean")
  if not isinstance(command, list) or not isinstance(actual, list):
    return False, 0.0, 0.0
  moving = abs(float(command[0])) > 1.0e-6
  if not moving:
    return False, 0.0, 0.0
  gain = float(actual[0]) / float(command[0])
  error = sum(abs(float(a) - float(b)) for a, b in zip(actual, command, strict=True))
  return True, gain, error


def _iter_scenarios(
  invocation: Invocation, payload: dict[str, Any]
) -> list[tuple[str, str, str, str, dict[str, Any]]]:
  rows = []
  def validate_profile(settings: dict[str, Any], profile: str) -> None:
    expected = (
      list(SIM2REAL_RANDOMIZATION_EVENTS) if profile == "randomized" else []
    )
    if settings.get("startup_randomization_events") != expected:
      raise ValueError(
        f"{invocation.artifact_id} {profile} profile event mismatch"
      )
    if bool(settings.get("actor_observation_corruption")) != (profile == "randomized"):
      raise ValueError(
        f"{invocation.artifact_id} {profile} actor corruption mismatch"
      )
    if bool(settings.get("push_enabled")) != (profile == "randomized"):
      raise ValueError(f"{invocation.artifact_id} {profile} push mismatch")

  if invocation.artifact_id in {"high_slope_matched", "flat_matched_routes"}:
    category = "complex" if invocation.artifact_id == "high_slope_matched" else "retained"
    scene = "high_slope" if category == "complex" else "flat"
    for raw_profile, profile_data in payload["profiles"].items():
      profile = canonical_profile_name(raw_profile)
      validate_profile(profile_data["profile_settings"], profile)
      for raw_route, result in profile_data["route_results"].items():
        route = "line" if raw_route == "straight" else raw_route
        for scenario in result["scenarios"]:
          scenario_scene = scene
          if category == "complex":
            scenario_scene = (
              f"{scenario.get('slope_direction', 'slope')}_"
              f"level{scenario.get('level', 'unknown')}"
            )
          rows.append((profile, category, scenario_scene, route, scenario))
    return rows
  profile = canonical_profile_name(invocation.canonical_profiles[0])
  validate_profile(payload["profile_settings"], profile)
  for scenario in payload["scenarios"]:
    if invocation.artifact_id.startswith("terrain_curves_"):
      kind = str(scenario["terrain_curve_kind"])
      category = "complex" if kind.startswith("slope") else "retained"
      scene = f"{kind}_level{scenario['terrain_level']}"
      route = "line" if scenario.get("route_kind") == "straight" else str(scenario.get("route_kind", invocation.artifact_id.split("_")[2]))
    else:
      transition = str(scenario["transition_case"])
      level = int(scenario.get("level", scenario.get("transition_level", 0)))
      category = "complex" if level >= 9 or transition.startswith("slope") else "retained"
      scene = f"{transition}_level{level}"
      route = "line"
    rows.append((profile, category, scene, route, scenario))
  return rows


def _profile_settings(
  invocation: Invocation, payload: dict[str, Any]
) -> dict[str, dict[str, Any]]:
  if invocation.artifact_id in {"high_slope_matched", "flat_matched_routes"}:
    return {
      canonical_profile_name(profile): value["profile_settings"]
      for profile, value in payload["profiles"].items()
    }
  return {
    canonical_profile_name(invocation.canonical_profiles[0]): payload["profile_settings"]
  }


def _group_artifacts(
  invocations: list[Invocation], payloads: dict[str, dict[str, Any]]
) -> dict[str, dict[tuple[str, str, str, str], dict[str, Any]]]:
  by_checkpoint: dict[str, dict[tuple[str, str, str, str], list[tuple[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
  for invocation in invocations:
    payload = payloads[invocation.output_file]
    for profile, category, scene, route, scenario in _iter_scenarios(invocation, payload):
      identity_fields = {
        key: scenario[key] for key in (
          "matched_slot", "terrain_type", "terrain_curve_kind", "terrain_level",
          "transition_case", "level", "difficulty", "radius", "speed",
          "turn_sign", "repeat", "seed",
          "terrain_origin_xyz", "terrain_patch_origin_xyz", "route_start_xy",
          "route_endpoint_xy", "effective_terrain_parameters", "geometry",
        ) if key in scenario
      }
      scenario_id = hashlib.sha256(json.dumps(
        [invocation.artifact_id, identity_fields], sort_keys=True,
        separators=(",", ":"),
      ).encode()).hexdigest()
      by_checkpoint[invocation.checkpoint_path][(profile, category, scene, route)].append((scenario_id, scenario))
  grouped: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {}
  for checkpoint, groups in by_checkpoint.items():
    grouped[checkpoint] = {}
    for key, items in groups.items():
      items.sort(key=lambda item: item[0])
      moving_values = [_forward_metrics(item)[1:] for _, item in items if _forward_metrics(item)[0]]
      metrics: dict[str, float] = {}
      metric_availability: dict[str, dict[str, int | bool]] = {}
      for metric in SAFETY_METRICS:
        values = []
        for _, item in items:
          try:
            values.append(_scenario_metric(item, metric))
          except ValueError as error:
            if str(error) != f"scenario is missing unified safety metric {metric}":
              raise
        available = len(values) == len(items)
        metric_availability[metric] = {
          "available": available,
          "available_scenarios": len(values),
          "expected_scenarios": len(items),
        }
        # A partial mean would not be matched evidence. Omit the metric and
        # let the formal contract reject the checkpoint as unavailable.
        if available:
          metrics[metric] = _mean(values)
      grouped[checkpoint][key] = {
        "scenario_ids": [identity for identity, _ in items],
        "completion": _mean([float(bool(item.get("completed"))) for _, item in items]),
        "moving_forward": bool(moving_values),
        "forward_gain": _mean([value[0] for value in moving_values]) if moving_values else 0.0,
        "command_tracking_error": _mean([value[1] for value in moving_values]) if moving_values else 0.0,
        "metrics": metrics,
        "metric_availability": metric_availability,
        "action_limits_valid": all(
          item.get("terrain_rollout_metrics", {})
          .get("action_safety", {})
          .get("available") is True
          and item.get("terrain_rollout_metrics", {})
          .get("action_safety", {})
          .get("fault_occurred") is False
          for _, item in items
        ),
        "no_reset_storm": all(
          isinstance(item.get("reset_count"), int)
          and 0 <= item["reset_count"] <= 1
          for _, item in items
        ),
        "lifecycle_valid": all(
          isinstance(item.get("completed"), bool)
          and isinstance(item.get("reset_count"), int)
          and item.get("terrain_rollout_metrics", {}).get(
            "active_control_step_samples", 0
          ) > 0
          for _, item in items
        ),
      }
  return grouped


def assemble_bundles(
  invocations: list[Invocation], screening_dir: str | Path,
  bundle_dir: str | Path,
  *, source_manifest: str | Path | None = None,
  lineage_manifest: str | Path | None = None,
) -> list[Path]:
  validate_invocation_inventory(invocations)
  payloads = {}
  for invocation in invocations:
    path = Path(invocation.output_file)
    payload = json.loads(path.read_text())
    assert_recursive_json_finite(payload)
    provenance = payload.get("provenance", {})
    raw_identity = provenance.get("checkpoint", {})
    if (
      raw_identity.get("path") != invocation.checkpoint_path
      or raw_identity.get("sha256") != invocation.checkpoint_sha256
    ):
      raise ValueError(f"raw evaluator checkpoint provenance mismatch: {path}")
    payloads[invocation.output_file] = payload
  grouped = _group_artifacts(invocations, payloads)
  v7_path = str(V7_CHECKPOINT.resolve())
  reference = grouped[v7_path]
  reference_profiles = {
    item.artifact_id: _profile_settings(item, payloads[item.output_file])
    for item in invocations if item.checkpoint_path == v7_path
  }
  for item in invocations:
    if item.checkpoint_path == v7_path:
      continue
    if _profile_settings(item, payloads[item.output_file]) != reference_profiles[item.artifact_id]:
      raise ValueError(
        f"same-scene profile parameters differ from V7: "
        f"{item.checkpoint_path} {item.artifact_id}"
      )
  candidate_invocations = {
    item.checkpoint_path: item for item in invocations if item.checkpoint_path != v7_path
  }
  outputs = []
  for checkpoint_path, identity in sorted(candidate_invocations.items()):
    screening_path = Path(screening_dir).expanduser().resolve() / f"{Path(checkpoint_path).stem}.screening.json"
    screening = json.loads(screening_path.read_text())
    assert_recursive_json_finite(screening)
    screening_fault_count = screening.get("screening_action_fault_count")
    if (
      not isinstance(screening_fault_count, int)
      or isinstance(screening_fault_count, bool)
      or screening_fault_count < 0
      or screening.get("action_limits_valid") is not (screening_fault_count == 0)
    ):
      raise ValueError(
        f"inconsistent screening action-limit result: {screening_path}"
      )
    candidate = grouped[checkpoint_path]
    if set(candidate) != set(reference):
      raise ValueError(f"same-scene V7 group mismatch for {checkpoint_path}")
    groups = []
    for key in sorted(candidate):
      profile, category, scene, route = key
      value = candidate[key]
      v7 = reference[key]
      if value["scenario_ids"] != v7["scenario_ids"]:
        raise ValueError(f"same-scene scenario identity mismatch for {checkpoint_path}: {key}")
      scene_identity = hashlib.sha256(json.dumps(
        [key, value["scenario_ids"]], sort_keys=True,
        separators=(",", ":"),
      ).encode()).hexdigest()
      groups.append({
        "profile": profile, "category": category, "scene": scene,
        "route_kind": route, "completion": value["completion"],
        "moving_forward": value["moving_forward"],
        "forward_gain": value["forward_gain"],
        "command_tracking_error": value["command_tracking_error"],
        "metrics": value["metrics"], "v7_reference": v7["metrics"],
        "metric_availability": value["metric_availability"],
        "v7_metric_availability": v7["metric_availability"],
        "scene_identity": scene_identity,
        "matched_reference_identity": scene_identity,
      })
    raw_dependencies = [
      item.output_file for item in invocations
      if item.checkpoint_path in {checkpoint_path, v7_path}
    ]
    raw_dependencies.append(str(screening_path))
    if lineage_manifest is not None:
      raw_dependencies.append(str(Path(lineage_manifest).expanduser().resolve()))
    provenance = formal_evaluation_provenance(
      checkpoint_path, __file__, raw_dependencies, workspace=WORKSPACE,
      source_manifest=source_manifest,
    )
    contract = dict(screening)
    contract.update({
      "provenance_valid": True,
      "recursive_finite": True,
      "profile_contract_valid": True,
      "same_scene_reference_valid": True,
      "unified_metrics_valid": all(
        availability["available"] is True
        for grouped_value in (*candidate.values(), *reference.values())
        for availability in grouped_value["metric_availability"].values()
      ),
      "coverage_complete": EXPECTED_ACCEPTANCE_ARTIFACTS == {
        item.artifact_id for item in invocations
        if item.checkpoint_path == checkpoint_path
      },
      "action_limits_valid": bool(screening.get("action_limits_valid"))
      and all(item["action_limits_valid"] for item in candidate.values()),
      "onnx_action_contract_valid": bool(screening.get("onnx_action_contract_valid")),
      "no_reset_storm": bool(screening.get("no_reset_storm"))
      and all(item["no_reset_storm"] for item in candidate.values()),
      "lifecycle_valid": bool(screening.get("lifecycle_valid"))
      and all(item["lifecycle_valid"] for item in candidate.values()),
    })
    bundle = {
      "schema_version": 1,
      "evaluation_suite": "go2_proprioceptive_checkpoint_acceptance",
      "checkpoint": {
        "path": checkpoint_path, "sha256": identity.checkpoint_sha256,
      },
      "screening_artifact": {
        "path": str(screening_path), "sha256": sha256_file(screening_path),
      },
      "provenance": provenance,
      "contract": contract,
      "groups": groups,
    }
    output = Path(bundle_dir).expanduser().resolve() / f"{Path(checkpoint_path).stem}.acceptance.json"
    _write_strict(output, bundle)
    outputs.append(output)
  return outputs


def main() -> None:
  parser = argparse.ArgumentParser()
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--run-dir", type=Path)
  source.add_argument("--checkpoint-manifest", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--plan-file", type=Path, required=True)
  parser.add_argument("--screening-dir", type=Path)
  parser.add_argument("--bundle-dir", type=Path)
  parser.add_argument("--selection-file", type=Path)
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--resume-existing", action="store_true")
  args = parser.parse_args()
  lineage = None
  if args.checkpoint_manifest is not None:
    lineage = load_checkpoint_lineage(args.checkpoint_manifest)
    checkpoints = list(lineage.checkpoints)
  else:
    checkpoints = discover_checkpoints(args.run_dir)
  validate_formal_checkpoint_schedule(checkpoints)
  matrix = build_matrix(args.run_dir, args.output_dir, checkpoints=checkpoints)
  plan = {
    "schema_version": 1,
    "evaluation_suite": "go2_proprioceptive_all_checkpoint_acceptance_plan",
    "gpu_execution_requested": bool(args.execute),
    "serial_gpu_only": True,
    "candidate_run_dir": (
      None if args.run_dir is None else str(args.run_dir.expanduser().resolve())
    ),
    "checkpoint_lineage": None if lineage is None else {
      "path": str(lineage.manifest_path),
      "sha256": sha256_file(lineage.manifest_path),
    },
    "checkpoint_count_including_v7": len({item.checkpoint_path for item in matrix}),
    "invocations": [asdict(item) for item in matrix],
  }
  _write_strict(args.plan_file.expanduser().resolve(), plan)
  if args.execute:
    if args.screening_dir is None or args.bundle_dir is None or args.selection_file is None:
      raise ValueError(
        "--execute requires --screening-dir, --bundle-dir, and --selection-file"
      )
    inventory = _execute(matrix, resume_existing=args.resume_existing)
    inventory_path = args.plan_file.expanduser().resolve().with_suffix(".inventory.json")
    _write_strict(inventory_path, {
      "schema_version": 1,
      "plan": {"path": str(args.plan_file.resolve()), "sha256": sha256_file(args.plan_file)},
      "artifacts": inventory,
    })
    bundle_paths = assemble_bundles(
      matrix, args.screening_dir, args.bundle_dir,
      source_manifest=None if lineage is None else lineage.source_manifest,
      lineage_manifest=None if lineage is None else lineage.manifest_path,
    )
    bundles = [json.loads(path.read_text()) for path in bundle_paths]
    selected, decisions = select_checkpoint_bundles(bundles)
    primary = selected if selected is not None else decisions[0]
    _write_strict(args.selection_file.expanduser().resolve(), {
      "schema_version": 1,
      "selection_method": "hard_gates_then_preregistered_lexicographic",
      "weighted_score_used": False,
      "input_bundles": [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in bundle_paths
      ],
      "decisions": [asdict(item) for item in decisions],
      "survivor_count": sum(item.passed for item in decisions),
      "selection_status": "SELECTED" if selected is not None else "NO_SAFE_SURVIVOR",
      "selected_checkpoint": None if selected is None else {
        "path": selected.checkpoint,
        "sha256": selected.checkpoint_sha256,
        "iteration": selected.iteration,
        "lexicographic_key": selected.lexicographic_key,
      },
      "provenance": formal_evaluation_provenance(
        primary.checkpoint, __file__,
        (
          WORKSPACE / "src/tasks/velocity/evaluation/proprio_acceptance.py",
          WORKSPACE / "scripts/select_go2_proprioceptive_checkpoint.py",
        ), workspace=WORKSPACE,
        source_manifest=None if lineage is None else lineage.source_manifest,
      ),
    })
  print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
