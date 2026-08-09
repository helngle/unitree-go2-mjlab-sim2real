import os
import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import wandb
from rsl_rl.runners import DistillationRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


V7_TEACHER_CHECKPOINT = Path(
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
V7_TEACHER_SHA256 = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)
SOURCE_MANIFEST_PATH_ENV = "GO2_PROPRIO_SOURCE_MANIFEST"
SOURCE_MANIFEST_SHA256_ENV = "GO2_PROPRIO_SOURCE_MANIFEST_SHA256"
V1_SCHEMA_MODULE = "src.tasks.velocity.config.go2.sim2real_schema"
V2_SAFE_ACTION_SCHEMA_MODULE = (
  "src.tasks.velocity.config.go2.sim2real_safe_action_schema"
)
SAFE_ACTION_INTERFACE = "bounded_asymmetric_per_joint_v2"
SAFE_ACTION_TELEMETRY_TOLERANCE = 1.0e-6


class _SafeActionTelemetry:
  """Collect and validate the complete V2 action chain during each rollout."""

  def __init__(self, runner) -> None:
    self.runner = runner
    self.reset()

  def reset(self) -> None:
    self.step_count = 0
    self.action_fault_rows = 0
    self.target_fault_rows = 0
    self.latent_abs_max = 0.0
    self.unit_abs_max = 0.0
    self.applied_abs_max = 0.0
    self.target_margin_min = float("inf")
    self.latest: dict[str, torch.Tensor] = {}

  def _latent(self) -> torch.Tensor:
    distillation_raw = getattr(
      self.runner.alg, "last_teacher_raw_actions", None
    )
    if distillation_raw is not None:
      return distillation_raw
    distribution = self.runner.alg.get_policy().distribution
    latent = getattr(distribution, "_last_latent", None)
    if latent is None:
      raise RuntimeError("safe-action latent/raw telemetry is missing")
    return latent

  def record_action(self, actions: torch.Tensor) -> None:
    distribution = self.runner.alg.get_policy().distribution
    if distribution is None or not hasattr(distribution, "transform"):
      raise RuntimeError("safe-action telemetry cannot resolve action transform")
    latent = self._latent().detach()
    actions = actions.detach()
    unit = torch.tanh(latent)
    expected = distribution.transform(latent)
    if not torch.allclose(
      expected, actions, atol=SAFE_ACTION_TELEMETRY_TOLERANCE, rtol=0.0
    ):
      raise RuntimeError("latent/raw to applied-action telemetry mismatch")
    if not all(torch.isfinite(value).all() for value in (latent, unit, actions)):
      raise RuntimeError("safe-action telemetry observed NaN/Inf")

    low = distribution.action_low.to(device=actions.device, dtype=actions.dtype)
    high = distribution.action_high.to(device=actions.device, dtype=actions.dtype)
    faults = ((actions < low - SAFE_ACTION_TELEMETRY_TOLERANCE) | (
      actions > high + SAFE_ACTION_TELEMETRY_TOLERANCE
    )).any(dim=-1)
    fault_rows = int(faults.sum().item())
    self.action_fault_rows += fault_rows
    if fault_rows:
      raise RuntimeError(
        f"safe-action telemetry observed {fault_rows} action-bound fault rows"
      )

    self.step_count += 1
    self.latent_abs_max = max(self.latent_abs_max, float(latent.abs().max()))
    self.unit_abs_max = max(self.unit_abs_max, float(unit.abs().max()))
    self.applied_abs_max = max(self.applied_abs_max, float(actions.abs().max()))
    self.latest.update(
      {
        "latent_raw": latent.float().cpu(),
        "u": unit.float().cpu(),
        "a_applied": actions.float().cpu(),
      }
    )

  def record_target(self) -> None:
    env = self.runner.env.unwrapped
    term = env.action_manager.get_term("joint_pos")
    targets = term._processed_actions.detach()
    limits = env.scene["robot"].data.joint_pos_limits[:, term.target_ids]
    lower_margin = targets - limits[..., 0]
    upper_margin = limits[..., 1] - targets
    margin = torch.minimum(lower_margin, upper_margin)
    if not all(torch.isfinite(value).all() for value in (targets, margin)):
      raise RuntimeError("safe-action q_target telemetry observed NaN/Inf")
    faults = (margin < -SAFE_ACTION_TELEMETRY_TOLERANCE).any(dim=-1)
    fault_rows = int(faults.sum().item())
    self.target_fault_rows += fault_rows
    if fault_rows:
      raise RuntimeError(
        f"safe-action telemetry observed {fault_rows} joint-target fault rows"
      )
    self.target_margin_min = min(self.target_margin_min, float(margin.min()))
    self.latest.update(
      {
        "q_target": targets.float().cpu(),
        "q_target_limit_margin": margin.float().cpu(),
      }
    )

  def write(self, iteration: int) -> None:
    expected_steps = int(self.runner.cfg["num_steps_per_env"])
    required = {
      "latent_raw", "u", "a_applied", "q_target", "q_target_limit_margin"
    }
    if self.step_count != expected_steps or set(self.latest) != required:
      raise RuntimeError(
        "safe-action telemetry is incomplete: "
        f"steps={self.step_count}/{expected_steps}, fields={sorted(self.latest)}"
      )
    writer = self.runner.logger.writer
    if writer is None:
      raise RuntimeError("safe-action TensorBoard telemetry writer is missing")
    scalars = {
      "latent_raw_abs_max": self.latent_abs_max,
      "u_abs_max": self.unit_abs_max,
      "a_applied_abs_max": self.applied_abs_max,
      "q_target_limit_margin_min": self.target_margin_min,
      "action_fault_rows": self.action_fault_rows,
      "joint_target_fault_rows": self.target_fault_rows,
      "finite": 1.0,
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in scalars.values()):
      raise RuntimeError("safe-action scalar telemetry observed NaN/Inf")
    for name, value in scalars.items():
      writer.add_scalar(f"ActionTelemetry/{name}", value, iteration)
    for name, value in self.latest.items():
      writer.add_histogram(f"ActionTelemetry/{name}", value, iteration)
    self.reset()


class _SafeActionTelemetryMixin:
  """Add fail-closed V2 action telemetry without changing training math."""

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    telemetry = _SafeActionTelemetry(self)
    original_act = self.alg.act
    original_step = self.env.step
    original_log = self.logger.log

    def act_with_telemetry(observations):
      actions = original_act(observations)
      telemetry.record_action(actions)
      return actions

    def step_with_telemetry(actions):
      result = original_step(actions)
      telemetry.record_target()
      return result

    def log_with_telemetry(*args: Any, **kwargs: Any) -> None:
      original_log(*args, **kwargs)
      iteration = kwargs.get("it", args[0] if args else None)
      if iteration is None:
        raise RuntimeError("safe-action telemetry cannot resolve iteration")
      telemetry.write(int(iteration))

    self.alg.act = act_with_telemetry
    self.env.step = step_with_telemetry
    self.logger.log = log_with_telemetry
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.alg.act = original_act
      self.env.step = original_step
      self.logger.log = original_log


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _active_source_manifest(
  *, required: bool
) -> tuple[str, str] | None:
  manifest_path = os.environ.get(SOURCE_MANIFEST_PATH_ENV)
  manifest_sha = os.environ.get(SOURCE_MANIFEST_SHA256_ENV)
  if manifest_path is None and manifest_sha is None:
    if required:
      raise RuntimeError("formal proprioceptive training source manifest is missing")
    return None
  if not manifest_path or not manifest_sha:
    raise RuntimeError("incomplete proprioceptive source manifest environment")
  resolved = Path(manifest_path).expanduser().resolve()
  if not resolved.is_file():
    raise RuntimeError(f"proprioceptive source manifest is missing: {resolved}")
  actual_sha = _sha256(resolved)
  if actual_sha != manifest_sha:
    raise RuntimeError("proprioceptive source manifest SHA256 mismatch")
  return str(resolved), manifest_sha


def _validate_checkpoint_source_manifest(
  infos: dict, *, required: bool
) -> tuple[str, str] | None:
  active = _active_source_manifest(required=required)
  if active is None:
    return None
  manifest_path, manifest_sha = active
  if infos.get("source_manifest_sha256") != manifest_sha:
    raise ValueError("checkpoint source manifest SHA256 mismatch")
  checkpoint_path = infos.get("source_manifest")
  if checkpoint_path is not None and Path(checkpoint_path).expanduser().resolve() != Path(
    manifest_path
  ):
    raise ValueError("checkpoint source manifest path mismatch")
  return active


def _load_schema_module(module_path: str) -> ModuleType:
  module = importlib.import_module(module_path)
  for name in ("SCHEMA_VERSION", "actor_dim", "schema_sha256"):
    if not hasattr(module, name):
      raise TypeError(f"proprioceptive schema {module_path!r} is missing {name}")
  return module


def _validate_checkpoint_contract(
  infos: dict,
  *,
  schema_module_path: str,
  expected_stage: str,
  action_interface: str | None,
) -> None:
  schema = _load_schema_module(schema_module_path)
  if infos.get("proprioceptive_stage") != expected_stage:
    raise ValueError(f"checkpoint is not a registered {expected_stage} output")
  if infos.get("student_schema_sha256") != schema.schema_sha256():
    raise ValueError("checkpoint student schema SHA256 mismatch")
  if action_interface is not None and infos.get("action_interface") != action_interface:
    raise ValueError("checkpoint action interface mismatch")
  action_mean_bound = getattr(schema, "ACTION_MEAN_BOUND", None)
  if action_mean_bound is not None and infos.get("action_mean_bound") != action_mean_bound:
    raise ValueError("checkpoint action mean bound mismatch")


def _proprio_metadata(
  env, schema_module_path: str = V1_SCHEMA_MODULE
) -> dict[str, str | int]:
  if tuple(env.observation_manager.group_obs_dim.get("actor", ())) != (425,):
    return {}
  schema = _load_schema_module(schema_module_path)

  metadata = {
    "observation_schema_version": schema.SCHEMA_VERSION,
    "observation_schema_sha256": schema.schema_sha256(),
    "actor_observation_dim": schema.actor_dim(),
    "history_order": "term-major, oldest-to-newest",
  }
  action_interface = getattr(schema, "ACTION_INTERFACE", None)
  if action_interface is not None:
    metadata["action_interface"] = action_interface
  action_output = getattr(schema, "ACTION_OUTPUT_SEMANTICS", None)
  if action_output is not None:
    metadata["action_output_semantics"] = action_output
  action_mean_bound = getattr(schema, "ACTION_MEAN_BOUND", None)
  if action_mean_bound is not None:
    metadata["action_mean_bound"] = str(action_mean_bound)
  active_manifest = _active_source_manifest(required=False)
  if active_manifest is not None:
    metadata["source_manifest_sha256"] = active_manifest[1]
  return metadata


def _export_runner_policy(runner, path: str, filename: str, verbose: bool) -> None:
  onnx_model = runner.alg.get_policy().as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()
  os.makedirs(path, exist_ok=True)
  input_names = onnx_model.input_names
  if getattr(onnx_model, "input_size", None) == 425:
    input_names = ["actor"]
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    os.path.join(path, filename),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=input_names,
    output_names=onnx_model.output_names,
    dynamic_axes={},
    dynamo=False,
  )


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper
  schema_module_path = V1_SCHEMA_MODULE
  action_interface: str | None = None

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    _export_runner_policy(self, path, filename, verbose)

  def _high_slope_sampler(self):
    event_manager = self.env.unwrapped.event_manager
    try:
      term_cfg = event_manager.get_term_cfg("high_slope_sampling")
    except ValueError:
      return None
    sampler = term_cfg.func
    if not all(hasattr(sampler, name) for name in ("state_dict", "load_state_dict", "rebase")):
      raise TypeError("high_slope_sampling event does not implement persistence")
    return sampler

  def _environment_state(self) -> dict:
    env = self.env.unwrapped
    state = {"common_step_counter": env.common_step_counter}
    terrain = env.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
      state["terrain_levels"] = terrain.terrain_levels.detach().cpu().clone()
      state["terrain_types"] = terrain.terrain_types.detach().cpu().clone()
    sampler = self._high_slope_sampler()
    if sampler is not None:
      state["high_slope_sampling"] = sampler.state_dict()
    return state

  def _restore_environment_state(self, state: dict) -> None:
    env = self.env.unwrapped
    env.common_step_counter = state.get(
      "common_step_counter", env.common_step_counter
    )
    terrain = env.scene.terrain
    saved_levels = state.get("terrain_levels")
    saved_types = state.get("terrain_types")
    sampler = self._high_slope_sampler()
    saved_sampler_state = state.get("high_slope_sampling")
    if terrain is None or terrain.terrain_origins is None:
      return
    if saved_levels is None:
      print(
        "[WARN] Checkpoint predates terrain curriculum persistence; "
        "using configured start levels."
      )
      if sampler is not None:
        sampler.rebase()
      return
    if saved_levels.numel() != env.num_envs:
      print(
        "[WARN] Skipping terrain curriculum restore: checkpoint has "
        f"{saved_levels.numel()} environments, current run has {env.num_envs}."
      )
      if sampler is not None:
        sampler.rebase()
      return

    old_origins = terrain.env_origins.clone()
    terrain.terrain_levels.copy_(
      saved_levels.to(env.device, dtype=terrain.terrain_levels.dtype)
    )
    if saved_types is not None:
      terrain.terrain_types.copy_(
        saved_types.to(env.device, dtype=terrain.terrain_types.dtype)
      )
    terrain.terrain_levels.clamp_(0, terrain.max_terrain_level - 1)
    terrain.terrain_types.clamp_(0, terrain.terrain_origins.shape[1] - 1)
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]

    robot = env.scene["robot"]
    root_pose = robot.data.root_link_pose_w.clone()
    root_pose[:, :3] += terrain.env_origins - old_origins
    robot.write_root_link_pose_to_sim(root_pose)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    if sampler is not None:
      if saved_sampler_state is None:
        sampler.rebase()
      else:
        sampler.load_state_dict(saved_sampler_state)
    print(
      "[INFO] Restored terrain curriculum: "
      f"mean level {terrain.terrain_levels.float().mean().item():.3f}."
    )

  def save(self, path: str, infos=None):
    infos = {**(infos or {}), "env_state": self._environment_state()}
    if tuple(self.env.unwrapped.observation_manager.group_obs_dim.get("actor", ())) == (
      425,
    ):
      schema = _load_schema_module(self.schema_module_path)
      manifest_path, manifest_sha = _active_source_manifest(required=True)
      infos.update(
        {
          "proprioceptive_stage": "ppo",
          "student_schema_sha256": schema.schema_sha256(),
          "source_manifest": manifest_path,
          "source_manifest_sha256": manifest_sha,
        }
      )
      if self.action_interface is not None:
        infos["action_interface"] = self.action_interface
        infos["action_mean_bound"] = getattr(schema, "ACTION_MEAN_BOUND")
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    metadata.update(_proprio_metadata(self.env.unwrapped, self.schema_module_path))
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if "student_state_dict" in loaded and "actor_state_dict" not in loaded:
      # The second stage starts from the distilled deployable actor only.  The
      # PPO critic and optimizer are intentionally initialized from scratch.
      infos = loaded.get("infos") or {}
      _validate_checkpoint_contract(
        infos,
        schema_module_path=self.schema_module_path,
        expected_stage="distillation",
        action_interface=self.action_interface,
      )
      if infos.get("teacher_sha256") != V7_TEACHER_SHA256:
        raise ValueError("distillation checkpoint teacher SHA256 mismatch")
      _validate_checkpoint_source_manifest(infos, required=False)
      loaded["actor_state_dict"] = loaded["student_state_dict"]
      self.alg.load(
        loaded, {"actor": True, "iteration": False}, strict
      )
      return loaded.get("infos") or {}
    loaded_infos = loaded.get("infos") or {}
    if loaded_infos.get("proprioceptive_stage") == "ppo":
      _validate_checkpoint_contract(
        loaded_infos,
        schema_module_path=self.schema_module_path,
        expected_stage="ppo",
        action_interface=self.action_interface,
      )
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and infos.get("proprioceptive_stage") == "ppo":
      _validate_checkpoint_source_manifest(infos, required=False)
    restore_training_state = load_cfg is None or load_cfg.get("iteration", False)
    if restore_training_state and infos and "env_state" in infos:
      self._restore_environment_state(infos["env_state"])
    elif restore_training_state:
      print(
        "[WARN] Checkpoint has no terrain curriculum state; "
        "using configured start levels."
      )
    return infos


class VelocityDistillationRunner(DistillationRunner):
  """RSL-RL distillation runner with mjlab-compatible config handling."""

  schema_module_path = V1_SCHEMA_MODULE
  action_interface: str | None = None

  def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    _export_runner_policy(self, path, filename, verbose)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    # RSL-RL's shared logger reads student.output_std even though deterministic
    # teacher rollouts never initialize the student's output distribution.
    observations = self.env.get_observations().to(self.device)
    student = self.alg.student
    if student.distribution is None:
      raise RuntimeError("distillation student must have an output distribution")
    with torch.inference_mode():
      latent = student.get_latent(observations)
      student.distribution.update(student.mlp(latent))
    if not torch.isfinite(student.output_std).all():
      raise RuntimeError("distillation student logging std is non-finite")
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _active_source_manifest(required=False)
    if "actor_state_dict" in payload:
      expected = V7_TEACHER_CHECKPOINT.resolve()
      if checkpoint != expected:
        raise ValueError(f"teacher checkpoint must be locked to {expected}")
      if _sha256(checkpoint) != V7_TEACHER_SHA256:
        raise ValueError("locked V7 teacher SHA256 mismatch")
    elif "student_state_dict" in payload:
      infos = payload.get("infos") or {}
      _validate_checkpoint_contract(
        infos,
        schema_module_path=self.schema_module_path,
        expected_stage="distillation",
        action_interface=self.action_interface,
      )
      if infos.get("teacher_sha256") != V7_TEACHER_SHA256:
        raise ValueError("distillation resume teacher SHA256 mismatch")
      _validate_checkpoint_source_manifest(infos, required=False)
    else:
      raise ValueError("unsupported distillation checkpoint schema")
    infos = super().load(path, load_cfg, strict, map_location)
    if self.alg.teacher_loaded:
      self.alg.teacher.requires_grad_(False)
      self.alg.teacher.eval()
    return infos

  def save(self, path: str, infos=None) -> None:
    schema = _load_schema_module(self.schema_module_path)
    manifest_path, manifest_sha = _active_source_manifest(required=True)

    infos = {
      **(infos or {}),
      "proprioceptive_stage": "distillation",
      "teacher_checkpoint": str(V7_TEACHER_CHECKPOINT.resolve()),
      "teacher_sha256": V7_TEACHER_SHA256,
      "student_schema_sha256": schema.schema_sha256(),
      "source_manifest": manifest_path,
      "source_manifest_sha256": manifest_sha,
      "env_state": {
        "common_step_counter": self.env.unwrapped.common_step_counter,
      },
    }
    if self.action_interface is not None:
      infos["action_interface"] = self.action_interface
      infos["action_mean_bound"] = getattr(schema, "ACTION_MEAN_BOUND")
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name
      if self.logger.logger_type == "wandb" and wandb.run
      else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    metadata.update(_proprio_metadata(self.env.unwrapped, self.schema_module_path))
    attach_metadata_to_onnx(onnx_path, metadata)


class VelocitySafeActionOnPolicyRunner(
  _SafeActionTelemetryMixin, VelocityOnPolicyRunner
):
  """V2 PPO runner whose exported and executed actions are bounded/applied."""

  schema_module_path = V2_SAFE_ACTION_SCHEMA_MODULE
  action_interface = SAFE_ACTION_INTERFACE


class VelocitySafeActionDistillationRunner(
  _SafeActionTelemetryMixin, VelocityDistillationRunner
):
  """V2 distillation runner with safe-action schema provenance."""

  schema_module_path = V2_SAFE_ACTION_SCHEMA_MODULE
  action_interface = SAFE_ACTION_INTERFACE
