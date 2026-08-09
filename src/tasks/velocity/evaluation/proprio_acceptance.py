"""Shared evaluation-only contracts for proprioceptive checkpoint acceptance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import torch


SIM2REAL_RANDOMIZATION_EVENTS = (
  "foot_friction",
  "encoder_bias",
  "base_com",
  "base_payload",
  "motor_strength",
  "pd_gains",
  "limb_pseudo_inertia",
)
CANONICAL_PROFILES = ("clean", "randomized")
EXPECTED_PPO_ITERATIONS = (0, *range(250, 4000, 250), 3999)
V1_STUDENT_SCHEMA = "go2-sim2real-proprio-v1"
V2_SAFE_ACTION_INTERFACE = "bounded_asymmetric_per_joint_v2"
V1_STUDENT_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V1"
V2_SAFE_ACTION_TASK = "Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction"
PRIMARY_SAFETY_METRICS = (
  "failure_risk",
  "terrain_tangent_slip",
  "action_acceleration",
  "actuator_effort",
  "mechanical_power",
  "mechanical_energy",
  "base_pitch_absolute",
)
BODY_CONTACT_METRICS = (
  "base_contact",
  "upper_leg_contact",
  "calf_contact",
)
SAFETY_METRICS = PRIMARY_SAFETY_METRICS + BODY_CONTACT_METRICS


def _jsonable(value: Any) -> Any:
  if is_dataclass(value):
    return _jsonable(asdict(value))
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


def sha256_file(path: str | Path) -> str:
  resolved = Path(path).expanduser().resolve()
  digest = hashlib.sha256()
  with resolved.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_policy_contract(infos: Mapping[str, Any]) -> dict[str, Any]:
  """Resolve the registered task/schema from checkpoint-owned provenance."""
  if "action_interface" not in infos:
    action_interface = None
    from src.tasks.velocity.config.go2 import sim2real_schema as schema
    task_id = V1_STUDENT_TASK
  elif infos["action_interface"] == V2_SAFE_ACTION_INTERFACE:
    action_interface = V2_SAFE_ACTION_INTERFACE
    from src.tasks.velocity.config.go2 import sim2real_safe_action_schema as schema
    task_id = V2_SAFE_ACTION_TASK
  else:
    raise ValueError(
      f"unsupported checkpoint action interface: {infos['action_interface']!r}"
    )
  expected_sha = schema.schema_sha256()
  if infos.get("student_schema_sha256") != expected_sha:
    raise ValueError("checkpoint student schema SHA256 mismatch")
  action_mean_bound = getattr(schema, "ACTION_MEAN_BOUND", None)
  if action_mean_bound is not None and infos.get("action_mean_bound") != action_mean_bound:
    raise ValueError("checkpoint action mean bound mismatch")
  return {
    "task_id": task_id,
    "schema_module": schema,
    "schema_version": schema.SCHEMA_VERSION,
    "schema_sha256": expected_sha,
    "action_interface": action_interface,
    "action_output_semantics": getattr(
      schema, "ACTION_OUTPUT_SEMANTICS", "legacy_normalized_action"
    ),
    "action_mean_bound": action_mean_bound,
  }


def checkpoint_task_id(path: str | Path) -> str:
  payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
  infos = payload.get("infos") or {}
  if infos.get("proprioceptive_stage") != "ppo":
    raise ValueError("student checkpoint must be a registered PPO output")
  return str(checkpoint_policy_contract(infos)["task_id"])


def canonical_profile_name(profile: str) -> str:
  if profile == "full_randomized":
    return "randomized"
  if profile not in CANONICAL_PROFILES:
    raise ValueError(
      f"formal profile must be one of {CANONICAL_PROFILES}, got {profile!r}"
    )
  return profile


def install_sim2real_randomization_contract(env_cfg: Any) -> None:
  """Install the frozen student dynamics events on an evaluation config copy."""
  from src.tasks.velocity.config.go2.env_cfgs import (
    unitree_go2_rough_v7_sim2real_proprio_env_cfg,
  )

  reference = unitree_go2_rough_v7_sim2real_proprio_env_cfg()
  # Reinsert every event in the frozen training order. Updating existing dict
  # entries would preserve task-specific insertion order and could change the
  # shared RNG stream even when the event parameters match.
  for name in SIM2REAL_RANDOMIZATION_EVENTS:
    env_cfg.events.pop(name, None)
  for name in SIM2REAL_RANDOMIZATION_EVENTS:
    if name not in reference.events:
      raise RuntimeError(f"frozen sim2real task is missing event {name!r}")
    env_cfg.events[name] = deepcopy(reference.events[name])


def configure_sim2real_profile(env_cfg: Any, profile: str) -> dict[str, Any]:
  """Apply and verify the canonical clean/randomized acceptance profile."""
  profile = canonical_profile_name(profile)
  install_sim2real_randomization_contract(env_cfg)
  actor = env_cfg.observations["actor"]
  actor.enable_corruption = profile == "randomized"
  if profile == "clean":
    for name in SIM2REAL_RANDOMIZATION_EVENTS + ("push_robot",):
      env_cfg.events.pop(name, None)
  expected = set(SIM2REAL_RANDOMIZATION_EVENTS) if profile == "randomized" else set()
  actual = {name for name in SIM2REAL_RANDOMIZATION_EVENTS if name in env_cfg.events}
  if actual != expected:
    raise RuntimeError(
      f"{profile} randomization contract mismatch: expected={sorted(expected)} "
      f"actual={sorted(actual)}"
    )
  if profile == "clean" and "push_robot" in env_cfg.events:
    raise RuntimeError("clean profile must disable push_robot")
  if profile == "randomized" and "push_robot" not in env_cfg.events:
    raise RuntimeError("randomized profile requires push_robot")
  return {
    "canonical_profile": profile,
    "actor_observation_corruption": bool(actor.enable_corruption),
    "startup_randomization_events": [
      name for name in SIM2REAL_RANDOMIZATION_EVENTS if name in env_cfg.events
    ],
    "push_enabled": "push_robot" in env_cfg.events,
    "event_parameters": {
      name: {
        "mode": getattr(env_cfg.events[name], "mode", None),
        "interval_range_s": _jsonable(
          getattr(env_cfg.events[name], "interval_range_s", None)
        ),
        "params": _jsonable(getattr(env_cfg.events[name], "params", {})),
      }
      for name in SIM2REAL_RANDOMIZATION_EVENTS
      if name in env_cfg.events
    },
  }


def formal_evaluation_provenance(
  checkpoint: str | Path,
  evaluator: str | Path,
  dependencies: Sequence[str | Path] = (),
  *,
  workspace: str | Path | None = None,
  source_manifest: str | Path | None = None,
) -> dict[str, Any]:
  """Bind a formal artifact to checkpoint, evaluator, source, and dirty state."""
  evaluator_path = Path(evaluator).expanduser().resolve()
  root = (
    Path(workspace).expanduser().resolve()
    if workspace is not None else evaluator_path.parents[1]
  )
  checkpoint_path = Path(checkpoint).expanduser().resolve()
  paths = [evaluator_path, *(Path(item).expanduser().resolve() for item in dependencies)]
  for path in (checkpoint_path, *paths):
    if not path.is_file():
      raise FileNotFoundError(path)

  def git(*args: str) -> str:
    return subprocess.run(
      ("git", *args), cwd=root, check=True, capture_output=True, text=True
    ).stdout.rstrip("\n")

  status = git("status", "--porcelain=v1", "--untracked-files=all")
  tracked_diff = subprocess.run(
    ("git", "diff", "--binary", "HEAD"), cwd=root, check=True,
    capture_output=True,
  ).stdout
  if source_manifest is None:
    sibling_manifests = (
      checkpoint_path.parent / "proprioceptive_source_manifest.json",
      checkpoint_path.parent / "privileged_teacher_source_manifest.json",
    )
    present = [path for path in sibling_manifests if path.is_file()]
    if len(present) > 1:
      raise ValueError("checkpoint directory contains multiple source manifests")
    source_manifest = present[0] if present else None
  manifest_identity = None
  embedded_manifest_sha = None
  if source_manifest is not None:
    manifest_path = Path(source_manifest).expanduser().resolve()
    if not manifest_path.is_file():
      raise FileNotFoundError(manifest_path)
    manifest_identity = {
      "path": str(manifest_path),
      "sha256": sha256_file(manifest_path),
    }
    checkpoint_payload = torch.load(
      checkpoint_path, map_location="cpu", weights_only=False
    )
    infos = checkpoint_payload.get("infos", {})
    embedded_manifest_sha = infos.get("source_manifest_sha256")
    if embedded_manifest_sha != manifest_identity["sha256"]:
      raise ValueError("checkpoint embedded source manifest SHA256 mismatch")
  return {
    "git_branch": git("branch", "--show-current"),
    "git_head": git("rev-parse", "HEAD"),
    "dirty_state": {
      "is_dirty": bool(status),
      "status_porcelain": status,
      "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
      "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    },
    "checkpoint": {
      "path": str(checkpoint_path),
      "sha256": sha256_file(checkpoint_path),
    },
    "evaluator": {
      "path": str(evaluator_path),
      "sha256": sha256_file(evaluator_path),
    },
    "dependencies": {
      str(path): sha256_file(path) for path in paths[1:]
    },
    "source_manifest": manifest_identity,
    "checkpoint_embedded_source_manifest_sha256": embedded_manifest_sha,
  }


def base_pitch_absolute(robot: Any) -> torch.Tensor:
  gravity = robot.data.projected_gravity_b
  pitch = torch.atan2(
    -gravity[:, 0],
    torch.linalg.vector_norm(gravity[:, 1:], dim=-1).clamp_min(1.0e-6),
  )
  if not torch.isfinite(pitch).all():
    raise ValueError("base pitch contains NaN/Inf")
  return pitch.abs()


def environment_control_dt(env: Any) -> float:
  physics_dt = (
    env.cfg.sim.dt if hasattr(env.cfg.sim, "dt")
    else env.cfg.sim.mujoco.timestep
  )
  control_dt = float(physics_dt * env.cfg.decimation)
  if not math.isfinite(control_dt) or control_dt <= 0.0:
    raise ValueError("environment control dt must be finite and positive")
  return control_dt


def actuator_effort_and_power(robot: Any) -> tuple[torch.Tensor, torch.Tensor]:
  force = robot.data.actuator_force
  velocity = robot.data.joint_vel
  if force.ndim != 2 or velocity.shape != force.shape:
    raise ValueError("actuator force and joint velocity must have identical (N,J) shape")
  if not torch.isfinite(force).all() or not torch.isfinite(velocity).all():
    raise ValueError("actuator effort/power input contains NaN/Inf")
  return force.abs().mean(dim=-1), (force * velocity).abs().sum(dim=-1)


def normalized_action_safety(
  action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return per-env max magnitude and deploy-contract action fault."""
  from src.tasks.velocity.config.go2.sim2real_schema import (
    ACTION_ABS_LIMIT,
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    JOINT_POS_LIMITS,
  )

  if action.ndim != 2 or action.shape[1] != len(DEFAULT_JOINT_POS):
    raise ValueError("normalized action must have shape (num_envs, 12)")
  if not torch.isfinite(action).all():
    raise ValueError("normalized action contains NaN/Inf")
  default = action.new_tensor(DEFAULT_JOINT_POS)
  limits = action.new_tensor(JOINT_POS_LIMITS)
  targets = default + ACTION_SCALE * action
  max_abs = action.abs().amax(dim=-1)
  fault = (max_abs > ACTION_ABS_LIMIT) | (
    (targets < limits[:, 0]) | (targets > limits[:, 1])
  ).any(dim=-1)
  return max_abs, fault


def processed_joint_target_safety(
  env: Any, *, tolerance: float = 1.0e-6
) -> torch.Tensor:
  """Return per-env faults from the action term's processed joint targets."""
  term = env.action_manager.get_term("joint_pos")
  targets = term._processed_actions
  limits = env.scene["robot"].data.joint_pos_limits[:, term.target_ids]
  if not torch.isfinite(targets).all() or not torch.isfinite(limits).all():
    raise ValueError("processed joint target safety input contains NaN/Inf")
  return (
    (targets < limits[..., 0] - tolerance)
    | (targets > limits[..., 1] + tolerance)
  ).any(dim=-1)


class TerrainTangentTelemetry:
  """Evaluation-only adapter for the frozen loaded local-tangent slip metric."""

  def __init__(
    self, env: Any, *, site_names: tuple[str, ...] = ("FR", "FL", "RR", "RL")
  ) -> None:
    self.env = env
    self.robot = env.scene["robot"]
    self.foot_ids, found = self.robot.find_sites(site_names, preserve_order=True)
    if tuple(found) != site_names:
      raise RuntimeError(f"foot site order mismatch: {found}")
    self.num_feet = len(site_names)
    self.contact_sensor = env.scene["feet_ground_contact"]
    self.terrain_sensor = env.scene["terrain_scan"]
    geom_names = tuple(f"{name}_foot_collision" for name in site_names)
    sensor_names = [
      slot.primary_name for slot in self.contact_sensor._slots
      if slot.field_name == "found"
    ]
    missing = [name for name in geom_names if name not in sensor_names]
    if missing:
      raise RuntimeError(f"feet_ground_contact is missing geoms: {missing}")
    self.permutation = torch.tensor(
      [sensor_names.index(name) for name in geom_names],
      dtype=torch.long, device=env.device,
    )

  def sample(self) -> tuple[torch.Tensor, ...]:
    from src.tasks.velocity.mdp.rewards import (
      terrain_relative_loaded_stance_slip_cost,
    )

    found = self.contact_sensor.data.found
    force = self.contact_sensor.data.force
    if found is None or force is None:
      raise RuntimeError("loaded tangent telemetry requires contact found/force")
    contact = (found > 0).reshape(
      self.env.num_envs, self.num_feet, -1
    ).any(dim=-1).index_select(1, self.permutation)
    contact_force = force.reshape(
      self.env.num_envs, self.num_feet, 3
    ).index_select(1, self.permutation)
    return terrain_relative_loaded_stance_slip_cost(
      self.robot.data.site_pos_w[:, self.foot_ids, :],
      self.robot.data.site_lin_vel_w[:, self.foot_ids, :],
      contact_force,
      contact,
      self.terrain_sensor.data.hit_pos_w,
      self.terrain_sensor.data.normals_w,
      self.terrain_sensor.data.distances,
      normal_force_threshold=15.0,
      max_horizontal_distance=0.25,
      slip_deadband=0.03,
      slip_scale=0.10,
      max_cost_per_foot=4.0,
    )


@dataclass(frozen=True)
class CheckpointDecision:
  checkpoint: str
  checkpoint_sha256: str
  iteration: int
  passed: bool
  violations: tuple[str, ...]
  lexicographic_key: tuple[float, ...] | None


@dataclass(frozen=True)
class CheckpointLineage:
  manifest_path: Path
  source_manifest: Path
  checkpoints: tuple[Path, ...]
  payload: Mapping[str, Any]


def load_checkpoint_lineage(path: str | Path) -> CheckpointLineage:
  """Load and strictly validate a formal checkpoint inventory."""
  manifest_path = Path(path).expanduser().resolve()
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  if payload.get("schema_version") != 1:
    raise ValueError("checkpoint lineage schema_version must be 1")
  if payload.get("evaluation_suite") != "go2_proprioceptive_checkpoint_lineage":
    raise ValueError("unexpected checkpoint lineage suite")

  source = payload.get("source_manifest")
  if not isinstance(source, Mapping):
    raise ValueError("checkpoint lineage source_manifest is missing")
  source_manifest = Path(str(source.get("path", ""))).expanduser()
  if not source_manifest.is_absolute():
    raise ValueError("checkpoint lineage source_manifest path must be absolute")
  source_manifest = source_manifest.resolve()
  source_sha = str(source.get("sha256", ""))
  if len(source_sha) != 64 or sha256_file(source_manifest) != source_sha:
    raise ValueError("checkpoint lineage source_manifest SHA256 mismatch")

  entries = payload.get("checkpoints")
  if not isinstance(entries, list):
    raise ValueError("checkpoint lineage checkpoints must be a list")
  checkpoints: list[Path] = []
  seen_paths: set[Path] = set()
  seen_iterations: set[int] = set()
  lineage_policy_contract: dict[str, Any] | None = None
  for entry in entries:
    if not isinstance(entry, Mapping):
      raise ValueError("checkpoint lineage entry must be an object")
    iteration = int(entry.get("iteration", -1))
    checkpoint = Path(str(entry.get("path", ""))).expanduser()
    if not checkpoint.is_absolute():
      raise ValueError("checkpoint lineage paths must be absolute")
    checkpoint = checkpoint.resolve()
    if checkpoint.name != f"model_{iteration}.pt":
      raise ValueError(f"checkpoint lineage filename/iteration mismatch: {checkpoint}")
    expected_sha = str(entry.get("sha256", ""))
    if len(expected_sha) != 64 or sha256_file(checkpoint) != expected_sha:
      raise ValueError(f"checkpoint lineage SHA256 mismatch: {checkpoint}")
    if checkpoint in seen_paths or iteration in seen_iterations:
      raise ValueError("checkpoint lineage contains duplicate path or iteration")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = checkpoint_payload.get("infos") or {}
    if int(checkpoint_payload.get("iter", -1)) != iteration:
      raise ValueError(f"checkpoint embedded iteration mismatch: {checkpoint}")
    if infos.get("proprioceptive_stage") != "ppo":
      raise ValueError(f"checkpoint stage mismatch: {checkpoint}")
    if infos.get("source_manifest_sha256") != source_sha:
      raise ValueError(f"checkpoint source-manifest mismatch: {checkpoint}")
    policy_contract = checkpoint_policy_contract(infos)
    policy_identity = {
      name: policy_contract[name]
      for name in ("task_id", "schema_version", "schema_sha256", "action_interface")
    }
    if lineage_policy_contract is None:
      lineage_policy_contract = policy_identity
    elif policy_identity != lineage_policy_contract:
      raise ValueError("checkpoint lineage mixes incompatible policy contracts")
    checkpoints.append(checkpoint)
    seen_paths.add(checkpoint)
    seen_iterations.add(iteration)
  validate_formal_checkpoint_schedule(checkpoints)

  def equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
      return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
      return left.keys() == right.keys() and all(
        equal(left[key], right[key]) for key in left
      )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
      return len(left) == len(right) and all(
        equal(a, b) for a, b in zip(left, right, strict=True)
      )
    return left == right

  execution_mode = payload.get("execution_mode", "split_resume")
  if execution_mode == "monolithic":
    run_identity = payload.get("run")
    if not isinstance(run_identity, Mapping):
      raise ValueError("monolithic checkpoint lineage run is missing")
    run_path = Path(str(run_identity.get("path", ""))).expanduser()
    if not run_path.is_absolute() or not run_path.resolve().is_dir():
      raise ValueError("monolithic checkpoint lineage run must be an existing absolute dir")
    run_path = run_path.resolve()
    if any(checkpoint.parent != run_path for checkpoint in checkpoints):
      raise ValueError("monolithic checkpoint lineage mixes run directories")
    if "resume_anchor" in payload:
      raise ValueError("monolithic checkpoint lineage cannot contain a resume anchor")
  elif execution_mode == "split_resume":
    anchor = payload.get("resume_anchor")
    if not isinstance(anchor, Mapping):
      raise ValueError("checkpoint lineage resume_anchor is missing")
    for label in ("path", "derived_from"):
      identity = anchor.get(label)
      if not isinstance(identity, Mapping):
        raise ValueError(f"checkpoint lineage resume_anchor.{label} is missing")
      identity_path = Path(str(identity.get("path", ""))).expanduser()
      if not identity_path.is_absolute():
        raise ValueError(f"resume_anchor.{label} path must be absolute")
      expected_sha = str(identity.get("sha256", ""))
      if len(expected_sha) != 64 or sha256_file(identity_path) != expected_sha:
        raise ValueError(f"resume_anchor.{label} SHA256 mismatch")
    semantic_change = anchor.get("semantic_change")
    if (
      not isinstance(semantic_change, Mapping)
      or set(semantic_change) != {"iter"}
      or not isinstance(semantic_change["iter"], list)
      or len(semantic_change["iter"]) != 2
    ):
      raise ValueError("resume anchor must declare exactly one iter cursor change")
    source_iteration, anchor_iteration = map(int, semantic_change["iter"])
    if anchor_iteration != source_iteration + 1:
      raise ValueError("resume anchor cursor must advance by exactly one iteration")
    derived_path = Path(str(anchor["derived_from"]["path"])).expanduser().resolve()
    if derived_path.name != f"model_{source_iteration}.pt":
      raise ValueError("resume anchor source filename/cursor mismatch")
    if derived_path not in seen_paths:
      raise ValueError("resume anchor source is not in the formal checkpoint schedule")
    anchor_path = Path(str(anchor["path"]["path"])).expanduser().resolve()
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    derived_payload = torch.load(derived_path, map_location="cpu", weights_only=False)
    if int(anchor_payload.get("iter", -1)) != anchor_iteration:
      raise ValueError("resume anchor embedded cursor mismatch")
    normalized_anchor = deepcopy(anchor_payload)
    normalized_anchor["iter"] = source_iteration
    if not equal(normalized_anchor, derived_payload):
      raise ValueError("resume anchor differs from its source beyond the iter cursor")
  else:
    raise ValueError(f"unsupported checkpoint lineage execution mode: {execution_mode!r}")
  excluded = payload.get("excluded_technical_runs")
  if not isinstance(excluded, list) or not excluded:
    raise ValueError("checkpoint lineage must register excluded technical runs")
  for item in excluded:
    if not isinstance(item, Mapping) or not item.get("reason"):
      raise ValueError("excluded technical run requires a reason")
    excluded_path = Path(str(item.get("path", ""))).expanduser()
    if not excluded_path.is_absolute() or not excluded_path.resolve().is_dir():
      raise ValueError("excluded technical run path must be an existing absolute dir")

  return CheckpointLineage(
    manifest_path=manifest_path,
    source_manifest=source_manifest,
    checkpoints=tuple(checkpoints),
    payload=payload,
  )


def checkpoint_iteration(path: str | Path) -> int:
  match = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
  if match is None:
    raise ValueError(f"checkpoint name must match model_<iteration>.pt: {path}")
  return int(match.group(1))


def validate_formal_checkpoint_schedule(paths: Sequence[str | Path]) -> None:
  observed = tuple(sorted(checkpoint_iteration(path) for path in paths))
  if observed != EXPECTED_PPO_ITERATIONS:
    raise ValueError(
      "formal checkpoint schedule mismatch: "
      f"expected={EXPECTED_PPO_ITERATIONS}, observed={observed}"
    )


def _finite_number(value: Any, path: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{path} must be numeric")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{path} must be finite")
  return result


def _validate_bundle_provenance(
  provenance: Mapping[str, Any], checkpoint_path: str, checkpoint_sha: str
) -> list[str]:
  violations: list[str] = []
  identity = provenance.get("checkpoint", {})
  if identity.get("path") != checkpoint_path or identity.get("sha256") != checkpoint_sha:
    violations.append("provenance:checkpoint_identity")
  for label in ("evaluator", "source_manifest"):
    value = provenance.get(label)
    if not isinstance(value, Mapping):
      violations.append(f"provenance:{label}_missing")
      continue
    try:
      if sha256_file(value["path"]) != value["sha256"]:
        violations.append(f"provenance:{label}_sha256")
    except (KeyError, FileNotFoundError, TypeError):
      violations.append(f"provenance:{label}_identity")
  source = provenance.get("source_manifest")
  if isinstance(source, Mapping) and (
    provenance.get("checkpoint_embedded_source_manifest_sha256")
    != source.get("sha256")
  ):
    violations.append("provenance:checkpoint_source_manifest")
  dependencies = provenance.get("dependencies")
  if not isinstance(dependencies, Mapping) or not dependencies:
    violations.append("provenance:dependencies_missing")
  else:
    for path, expected in dependencies.items():
      try:
        if sha256_file(path) != expected:
          violations.append(f"provenance:dependency_sha256:{path}")
      except (FileNotFoundError, TypeError):
        violations.append(f"provenance:dependency_identity:{path}")
  dirty = provenance.get("dirty_state", {})
  if not isinstance(dirty, Mapping) or any(
    not isinstance(dirty.get(name), str) or len(dirty[name]) != 64
    for name in ("status_sha256", "tracked_diff_sha256")
  ):
    violations.append("provenance:dirty_fingerprint")
  if not provenance.get("git_branch") or not provenance.get("git_head"):
    violations.append("provenance:git_identity")
  return violations


def evaluate_checkpoint_bundle(bundle: Mapping[str, Any]) -> CheckpointDecision:
  """Apply frozen hard gates, then construct the preregistered ranking key."""
  checkpoint = bundle.get("checkpoint", {})
  path = str(Path(checkpoint["path"]).expanduser().resolve())
  sha = str(checkpoint["sha256"])
  if len(sha) != 64 or sha256_file(path) != sha:
    raise ValueError("checkpoint SHA256 mismatch")
  iteration = checkpoint_iteration(path)
  required_contract = (
    "provenance_valid", "lifecycle_valid", "recursive_finite",
    "schema_425_valid", "onnx_single_input", "onnx_no_privileged_inputs",
    "onnx_action_contract_valid",
    "no_reset_storm", "placement_valid", "action_limits_valid",
    "profile_contract_valid", "same_scene_reference_valid",
    "unified_metrics_valid", "coverage_complete",
  )
  contract = bundle.get("contract", {})
  violations = [f"contract:{name}" for name in required_contract if contract.get(name) is not True]
  screening_action_fault_count = contract.get("screening_action_fault_count")
  if (
    not isinstance(screening_action_fault_count, int)
    or isinstance(screening_action_fault_count, bool)
    or screening_action_fault_count < 0
  ):
    violations.append("contract:screening_action_fault_count")
  elif screening_action_fault_count != 0:
    violations.append(
      f"action_limits:screening_fault_count:{screening_action_fault_count}"
    )
  if contract.get("action_limits_valid") is not (
    isinstance(screening_action_fault_count, int)
    and not isinstance(screening_action_fault_count, bool)
    and screening_action_fault_count == 0
  ):
    violations.append("contract:action_limit_screening_consistency")
  provenance = bundle.get("provenance")
  if not isinstance(provenance, Mapping):
    violations.append("provenance:missing")
  else:
    violations.extend(_validate_bundle_provenance(provenance, path, sha))
  parity = _finite_number(contract.get("onnx_max_abs_error"), "onnx_max_abs_error")
  if parity > 1.0e-5:
    violations.append(f"onnx_parity:{parity:.9g}>1e-5")

  rows = list(bundle.get("groups", ()))
  if not rows:
    violations.append("coverage:no_groups")
  identities: set[tuple[str, str, str, str]] = set()
  retained: list[float] = []
  route_completion: list[float] = []
  forward_gains: list[float] = []
  tracking: list[float] = []
  safety_values: dict[str, list[float]] = {name: [] for name in SAFETY_METRICS}
  complex_by_profile: dict[str, list[float]] = {name: [] for name in CANONICAL_PROFILES}
  retained_by_profile: dict[str, list[float]] = {name: [] for name in CANONICAL_PROFILES}
  for index, row in enumerate(rows):
    profile = canonical_profile_name(str(row.get("profile")))
    category = str(row.get("category"))
    scene = str(row.get("scene"))
    route = str(row.get("route_kind"))
    identity = (profile, category, scene, route)
    if identity in identities:
      violations.append(f"coverage:duplicate:{identity}")
    identities.add(identity)
    if category not in {"retained", "complex"}:
      violations.append(f"coverage:unknown_category:{category}")
      continue
    completion = _finite_number(row.get("completion"), f"groups[{index}].completion")
    threshold = {
      ("retained", "clean"): 0.80,
      ("retained", "randomized"): 0.70,
      ("complex", "clean"): 0.65,
      ("complex", "randomized"): 0.55,
    }[(category, profile)]
    if completion < threshold:
      violations.append(
        f"completion:{profile}:{category}:{scene}:{route}:{completion:.9g}<{threshold}"
      )
    (retained_by_profile if category == "retained" else complex_by_profile)[profile].append(completion)
    if category == "retained":
      retained.append(completion)
    if route in {"line", "straight", "arc", "s_curve"}:
      route_completion.append(completion)
    if row.get("moving_forward") is True:
      forward_gains.append(_finite_number(row.get("forward_gain"), f"groups[{index}].forward_gain"))
      tracking.append(_finite_number(row.get("command_tracking_error"), f"groups[{index}].command_tracking_error"))
    metrics = row.get("metrics", {})
    reference = row.get("v7_reference", {})
    availability = row.get("metric_availability", {})
    reference_availability = row.get("v7_metric_availability", {})
    if row.get("matched_reference_identity") != row.get("scene_identity"):
      violations.append(f"matched_reference:{profile}:{scene}:{route}")
    for metric in SAFETY_METRICS:
      candidate_available = (
        isinstance(availability.get(metric), Mapping)
        and availability[metric].get("available") is True
        and metric in metrics
      )
      reference_available = (
        isinstance(reference_availability.get(metric), Mapping)
        and reference_availability[metric].get("available") is True
        and metric in reference
      )
      if not candidate_available:
        violations.append(
          f"unified_metric:candidate_unavailable:{metric}:{profile}:{scene}:{route}"
        )
      if not reference_available:
        violations.append(
          f"unified_metric:reference_unavailable:{metric}:{profile}:{scene}:{route}"
        )
      if not candidate_available or not reference_available:
        continue
      candidate_value = _finite_number(metrics.get(metric), f"groups[{index}].metrics.{metric}")
      reference_value = _finite_number(reference.get(metric), f"groups[{index}].v7_reference.{metric}")
      safety_values[metric].append(candidate_value)
      if candidate_value > 1.2 * reference_value:
        violations.append(
          f"safety:{metric}:{profile}:{scene}:{route}:{candidate_value:.9g}>1.2x{reference_value:.9g}"
        )

  for profile in CANONICAL_PROFILES:
    if not retained_by_profile[profile]:
      violations.append(f"coverage:no_{profile}_retained")
    if not complex_by_profile[profile]:
      violations.append(f"coverage:no_{profile}_complex")
  required_routes = {"line", "arc", "s_curve"}
  observed_routes = {"line" if row.get("route_kind") == "straight" else row.get("route_kind") for row in rows}
  for route in sorted(required_routes - observed_routes):
    violations.append(f"coverage:no_route:{route}")
  mean_gain = sum(forward_gains) / len(forward_gains) if forward_gains else -math.inf
  if not forward_gains:
    violations.append("coverage:no_moving_forward_groups")
  elif mean_gain < 0.75:
    violations.append(f"forward_gain_mean:{mean_gain:.9g}<0.75")

  if violations:
    key = None
  else:
    key = (
      min(complex_by_profile["randomized"]),
      min(complex_by_profile["clean"]),
      min(retained),
      min(route_completion),
      min(forward_gains),
      -(sum(tracking) / len(tracking)),
      *(
        -sum(safety_values[name]) / len(safety_values[name])
        for name in PRIMARY_SAFETY_METRICS
      ),
      *(
        -sum(safety_values[name]) / len(safety_values[name])
        for name in BODY_CONTACT_METRICS
      ),
      -float(iteration),
    )
  return CheckpointDecision(path, sha, iteration, not violations, tuple(violations), key)


def select_checkpoint_bundles(
  bundles: Sequence[Mapping[str, Any]],
) -> tuple[CheckpointDecision | None, list[CheckpointDecision]]:
  decisions = [evaluate_checkpoint_bundle(bundle) for bundle in bundles]
  paths = [item.checkpoint for item in decisions]
  if len(paths) != len(set(paths)):
    raise ValueError("duplicate checkpoint bundle")
  survivors = [item for item in decisions if item.passed]
  selected = max(survivors, key=lambda item: item.lexicographic_key) if survivors else None
  return selected, decisions


__all__ = [
  "CANONICAL_PROFILES",
  "BODY_CONTACT_METRICS",
  "CheckpointDecision",
  "CheckpointLineage",
  "EXPECTED_PPO_ITERATIONS",
  "PRIMARY_SAFETY_METRICS",
  "SAFETY_METRICS",
  "SIM2REAL_RANDOMIZATION_EVENTS",
  "TerrainTangentTelemetry",
  "actuator_effort_and_power",
  "base_pitch_absolute",
  "canonical_profile_name",
  "checkpoint_iteration",
  "checkpoint_policy_contract",
  "checkpoint_task_id",
  "configure_sim2real_profile",
  "evaluate_checkpoint_bundle",
  "environment_control_dt",
  "formal_evaluation_provenance",
  "install_sim2real_randomization_contract",
  "load_checkpoint_lineage",
  "normalized_action_safety",
  "processed_joint_target_safety",
  "select_checkpoint_bundles",
  "sha256_file",
  "validate_formal_checkpoint_schedule",
]
