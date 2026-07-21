import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

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
    infos = super().load(path, load_cfg, strict, map_location)
    restore_training_state = load_cfg is None or load_cfg.get("iteration", False)
    if restore_training_state and infos and "env_state" in infos:
      self._restore_environment_state(infos["env_state"])
    elif restore_training_state:
      print(
        "[WARN] Checkpoint has no terrain curriculum state; "
        "using configured start levels."
      )
    return infos
