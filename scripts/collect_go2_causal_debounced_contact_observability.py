"""Collect the registered causal confirmed-contact observability chunks.

This is an evaluation-only V7 rollout.  It never registers a new task,
restores an optimizer, or calls ``learn``.  The rollout timing intentionally
matches the V1 raw-contact collector, while adding the causal two-tick contact
feature seen at each pre-action state.  At chunk completion, the saved online
feature is checked bitwise against an independent CPU trajectory replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends

from scripts.collect_go2_foot_contact_observability import (
  _catastrophic_and_code,
  _commands,
  _contact,
  _force,
  _git_head,
  _make_routes,
  _route_state_fn,
  _runtime_term_slices,
  _sha256,
)
from scripts.diagnose_go2_high_slope_gait import _normal_and_clearance
from scripts.evaluate_go2_curved_routes import (
  _configure_episode_length,
  _configure_profile,
)
from scripts.evaluate_go2_high_slope_matched import (
  HighSlopeMatchedConfig,
  _scenarios,
  _terrain_assignment,
)
from src.tasks.velocity.evaluation.debounced_contact_observability import (
  recompute_causal_confirmed_contact,
)
from src.tasks.velocity.evaluation.routes import update_attempt_status
from src.tasks.velocity.evaluation.terrain_boundary_scenarios import (
  make_high_difficulty_curve_generator,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
  "docs/reviews/go2_causal_debounced_contact_observability_contract_20260809.json"
)
FILTER_MODULE = ROOT / (
  "src/tasks/velocity/evaluation/debounced_contact_observability.py"
)
TIMING_REFERENCE = ROOT / "scripts/collect_go2_foot_contact_observability.py"
CHECKPOINT = ROOT / (
  "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_"
  "go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt"
)
CHECKPOINT_SHA256 = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)
TASK_ID = "Unitree-Go2-Rough-V7"
NATIVE_FOOT_NAMES = ("FL", "FR", "RL", "RR")
NATIVE_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in NATIVE_FOOT_NAMES)
CANONICAL_FOOT_NAMES = ("FR", "FL", "RR", "RL")
NATIVE_TO_CANONICAL = (1, 0, 3, 2)
ACTOR_DIM = 234
CRITIC_DIM = 261
CRITIC_CONTACT_SLICE = slice(245, 249)
ROUTE_KINDS = ("straight", "arc", "s_curve")
PROFILES = ("clean", "randomized")
FORMAL_SEEDS = (1042, 1043, 1044, 1045)
CONFIRMATION_TICKS = 2


class _DeviceCausalConfirmedContactFilter:
  """Device-resident online equivalent of the frozen CPU filter.

  Contract violations are accumulated on-device and checked once after the
  rollout.  ``step`` therefore performs no host synchronization or device
  transfer in the policy loop.
  """

  def __init__(
    self,
    batch_size: int,
    *,
    device: torch.device | str,
    confirmation_ticks: int = CONFIRMATION_TICKS,
  ) -> None:
    if batch_size < 1:
      raise ValueError("batch_size must be positive")
    if confirmation_ticks < 1:
      raise ValueError("confirmation_ticks must be positive")
    self.batch_size = int(batch_size)
    self.confirmation_ticks = int(confirmation_ticks)
    self.device = torch.device(device)
    shape = (self.batch_size, 4)
    self._committed = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._candidate = torch.zeros_like(self._committed)
    self._candidate_count = torch.zeros(
      shape, dtype=torch.long, device=self.device
    )
    self._attempt_id = torch.zeros(
      self.batch_size, dtype=torch.long, device=self.device
    )
    self._initialized = torch.zeros(
      self.batch_size, dtype=torch.bool, device=self.device
    )
    self._contract_violation = torch.zeros(
      (), dtype=torch.bool, device=self.device
    )

  def step(
    self,
    raw_contact: torch.Tensor,
    *,
    attempt_id: torch.Tensor,
    episode_start: torch.Tensor,
    visible: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_contact = (self.batch_size, 4)
    if raw_contact.shape != expected_contact or raw_contact.dtype != torch.bool:
      raise ValueError(
        f"raw_contact must be native-order bool shape {expected_contact}"
      )
    for name, value, shape, dtype in (
      ("attempt_id", attempt_id, (self.batch_size,), torch.long),
      ("episode_start", episode_start, (self.batch_size,), torch.bool),
      ("visible", visible, (self.batch_size,), torch.bool),
    ):
      if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    for name, value in (
      ("raw_contact", raw_contact),
      ("attempt_id", attempt_id),
      ("episode_start", episode_start),
      ("visible", visible),
    ):
      if value.device != self.device:
        raise ValueError(f"{name} must remain on {self.device}")

    continuing = visible & ~episode_start
    violations = (
      (episode_start & ~visible)
      | (continuing & ~self._initialized)
      | (continuing & (attempt_id != self._attempt_id))
      | (episode_start & self._initialized & (attempt_id == self._attempt_id))
    )
    self._contract_violation |= violations.any()

    invalid = ~visible
    self._initialized[invalid] = False
    self._candidate_count[invalid] = 0
    changed = torch.zeros_like(self._committed)

    self._committed[episode_start] = raw_contact[episode_start]
    self._candidate[episode_start] = raw_contact[episode_start]
    self._candidate_count[episode_start] = 0
    self._attempt_id[episode_start] = attempt_id[episode_start]
    self._initialized[episode_start] = True

    equals_committed = raw_contact == self._committed
    cancel = continuing[:, None] & equals_committed
    self._candidate_count[cancel] = 0
    differs = continuing[:, None] & ~equals_committed
    same_candidate = differs & (raw_contact == self._candidate) & (
      self._candidate_count > 0
    )
    new_candidate = differs & ~same_candidate
    self._candidate[new_candidate] = raw_contact[new_candidate]
    self._candidate_count[new_candidate] = 1
    self._candidate_count[same_candidate] += 1
    confirmed = differs & (
      self._candidate_count >= self.confirmation_ticks
    )
    self._committed[confirmed] = self._candidate[confirmed]
    changed[confirmed] = True
    self._candidate_count[confirmed] = 0

    output = torch.where(
      visible[:, None], self._committed, torch.zeros_like(self._committed)
    )
    valid = visible[:, None].expand(-1, 4)
    return output, changed, valid

  def assert_contract(self) -> None:
    if bool(self._contract_violation):
      raise RuntimeError("device causal filter observed a reset/attempt violation")


def _empty_arrays(
  num_envs: int, steps: int, device: torch.device,
) -> dict[str, torch.Tensor]:
  f = lambda *shape: torch.full(shape, torch.nan, device=device)  # noqa: E731
  return {
    "actor_observation": f(num_envs, steps, ACTOR_DIM),
    "critic_contact": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "sensor_contact": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "confirmed_contact": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "confirmed_contact_valid": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "confirmed_contact_changed": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "clearance": f(num_envs, steps, 4),
    "clearance_valid": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "pre_progress": f(num_envs, steps),
    "command": f(num_envs, steps, 3),
    "pre_episode_tick": torch.full(
      (num_envs, steps), -1, dtype=torch.int32, device=device
    ),
    "policy_action": f(num_envs, steps, 12),
    "anchor_active": torch.zeros(
      num_envs, steps, dtype=torch.bool, device=device
    ),
    "post_contact": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "post_force_w": f(num_envs, steps, 4, 3),
    "post_foot_velocity_w": f(num_envs, steps, 4, 3),
    "post_terrain_normal_w": f(num_envs, steps, 4, 3),
    "post_ray_valid": torch.zeros(
      num_envs, steps, 4, dtype=torch.bool, device=device
    ),
    "post_progress": f(num_envs, steps),
    "post_episode_tick": torch.full(
      (num_envs, steps), -1, dtype=torch.int32, device=device
    ),
    "done": torch.zeros(num_envs, steps, dtype=torch.bool, device=device),
    "catastrophic": torch.zeros(
      num_envs, steps, dtype=torch.bool, device=device
    ),
    "termination_code": torch.zeros(
      num_envs, steps, dtype=torch.int16, device=device
    ),
    "route_completed": torch.zeros(
      num_envs, steps, dtype=torch.bool, device=device
    ),
  }


def _cpu_oracle_audit(
  arrays: dict[str, torch.Tensor], scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
  raw = arrays["sensor_contact"].detach().cpu().transpose(0, 1)
  saved = arrays["confirmed_contact"].detach().cpu().transpose(0, 1)
  saved_changed = (
    arrays["confirmed_contact_changed"].detach().cpu().transpose(0, 1)
  )
  saved_valid = (
    arrays["confirmed_contact_valid"].detach().cpu().transpose(0, 1)
  )
  state_valid = arrays["anchor_active"].detach().cpu().transpose(0, 1)
  ticks, batch = state_valid.shape
  episode_start = state_valid.clone()
  if ticks > 1:
    episode_start[1:] &= ~state_valid[:-1]
  slot_ids = torch.tensor(
    [int(scenario["matched_slot"]) for scenario in scenarios], dtype=torch.long
  )
  attempt_id = slot_ids[None, :].expand(ticks, batch).clone()
  oracle = recompute_causal_confirmed_contact(
    raw,
    attempt_id=attempt_id,
    episode_start=episode_start,
    state_valid=state_valid,
    confirmation_ticks=CONFIRMATION_TICKS,
  )
  oracle_valid = oracle.valid[:, :, None].expand(-1, -1, 4)
  contact_mismatch = saved ^ oracle.contact
  changed_mismatch = saved_changed ^ oracle.changed
  valid_mismatch = saved_valid ^ oracle_valid
  early_flip = saved_changed & ~oracle.changed
  missing_or_backfilled_flip = oracle.changed & ~saved_changed
  any_mismatch = contact_mismatch | changed_mismatch | valid_mismatch
  return {
    "filter_module": str(FILTER_MODULE),
    "filter_module_sha256": _sha256(FILTER_MODULE),
    "confirmation_ticks": CONFIRMATION_TICKS,
    "saved_confirmed_equals_cpu_oracle_bitwise": not bool(
      contact_mismatch.any()
    ),
    "saved_changed_equals_cpu_oracle_bitwise": not bool(changed_mismatch.any()),
    "saved_valid_equals_cpu_oracle_bitwise": not bool(valid_mismatch.any()),
    "contact_mismatch_count": int(contact_mismatch.sum()),
    "changed_mismatch_count": int(changed_mismatch.sum()),
    "valid_mismatch_count": int(valid_mismatch.sum()),
    "early_flip_count": int(early_flip.sum()),
    "missing_or_future_backfill_count": int(missing_or_backfilled_flip.sum()),
    "early_flip_or_future_backfill_count": int(any_mismatch.sum()),
  }


def collect(
  *,
  seed: int,
  profile: str,
  route_kind: str,
  steps: int,
  matched_slots: tuple[int, ...] | None,
  mode: str,
  device: str,
  smoke_timeout_steps: int | None = None,
) -> dict[str, Any]:
  if profile not in PROFILES or route_kind not in ROUTE_KINDS:
    raise ValueError("invalid profile or route kind")
  if mode not in {"smoke", "formal"}:
    raise ValueError("mode must be smoke or formal")
  if mode == "formal" and (
    seed not in FORMAL_SEEDS
    or steps != 2400
    or matched_slots is not None
    or smoke_timeout_steps is not None
  ):
    raise ValueError("formal collection requires registered seed/all slots/2400 steps")
  if mode == "smoke" and (steps <= 0 or steps > 100):
    raise ValueError("smoke collection is limited to 100 steps")
  if smoke_timeout_steps is not None and (
    mode != "smoke" or smoke_timeout_steps <= 0 or smoke_timeout_steps >= steps
  ):
    raise ValueError("smoke timeout must be positive, shorter than steps, and smoke-only")
  if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
    raise RuntimeError("V7 checkpoint SHA256 mismatch")

  cfg = HighSlopeMatchedConfig(
    checkpoint=str(CHECKPOINT),
    task_id=TASK_ID,
    profiles=(profile,),
    radii=(2.5,),
    steps=steps,
    seed=seed,
  )
  scenarios = _scenarios(cfg)
  if matched_slots is not None:
    requested = set(matched_slots)
    scenarios = [
      scenario
      for scenario in scenarios
      if int(scenario["matched_slot"]) in requested
    ]
    if len(scenarios) != len(requested):
      raise ValueError("matched slot request is incomplete or invalid")
  num_envs = len(scenarios)
  if mode == "formal" and num_envs != 16:
    raise RuntimeError("formal chunk must contain 16 matched slots")

  torch.manual_seed(seed)
  np.random.seed(seed)
  env_cfg = load_env_cfg(TASK_ID)
  agent_cfg = load_rl_cfg(TASK_ID)
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_generator = make_high_difficulty_curve_generator(seed)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env_cfg.curriculum = {}
  profile_settings = _configure_profile(env_cfg, profile)
  episode_settings = _configure_episode_length(env_cfg, steps + 20)
  if smoke_timeout_steps is not None:
    env_cfg.episode_length_s = (
      smoke_timeout_steps * episode_settings["control_dt"]
    )
    episode_settings["smoke_forced_timeout_steps"] = smoke_timeout_steps
    episode_settings["effective_episode_length_s"] = float(
      env_cfg.episode_length_s
    )
  command_cfg = env_cfg.commands["twist"]
  if not isinstance(command_cfg, UniformVelocityCommandCfg):
    raise TypeError("V7 twist command is not UniformVelocityCommand-compatible")
  if hasattr(command_cfg, "focus_terrain_names"):
    command_cfg.focus_terrain_names = ()
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.init_velocity_prob = 0.0
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (0.3, 0.5)
  command_cfg.ranges.lin_vel_y = (-0.3, 0.3)
  command_cfg.ranges.ang_vel_z = (-0.7, 0.7)
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(CHECKPOINT),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    placement = _terrain_assignment(env, scenarios)
    env.observation_manager.compute(update_history=True)
    observation = wrapped.get_observations()
    route_start = placement["route_start"]
    routes, route_lengths = _make_routes(route_kind, scenarios, route_start)
    route_state = _route_state_fn(routes, route_start, num_envs)
    robot = env.scene["robot"]
    command_term = env.command_manager.get_term("twist")
    if not isinstance(command_term, UniformVelocityCommand):
      raise TypeError("twist command term is incompatible")
    foot_ids, foot_names = robot.find_sites(
      NATIVE_FOOT_NAMES, preserve_order=True
    )
    if tuple(foot_names) != NATIVE_FOOT_NAMES:
      raise RuntimeError(f"native foot site order mismatch: {foot_names}")
    feet_sensor = env.scene["feet_ground_contact"]
    terrain_sensor = env.scene["terrain_scan"]
    sensor_names = tuple(
      slot.primary_name
      for slot in feet_sensor._slots
      if slot.field_name == "found"
    )
    if sensor_names != NATIVE_GEOM_NAMES:
      raise RuntimeError(
        f"runtime contact order differs: {sensor_names} != {NATIVE_GEOM_NAMES}"
      )
    actor_slices = _runtime_term_slices(env, "actor")
    critic_slices = _runtime_term_slices(env, "critic")
    expected_actor = {
      "base_ang_vel": (0, 3),
      "projected_gravity": (3, 6),
      "command": (6, 9),
      "phase": (9, 11),
      "joint_pos": (11, 23),
      "joint_vel": (23, 35),
      "actions": (35, 47),
      "height_scan": (47, 234),
    }
    expected_critic = {
      **expected_actor,
      "base_lin_vel": (234, 237),
      "foot_height": (237, 241),
      "foot_air_time": (241, 245),
      "foot_contact": (245, 249),
      "foot_contact_forces": (249, 261),
    }
    if actor_slices != expected_actor or critic_slices != expected_critic:
      raise RuntimeError(
        f"runtime observation term slices differ: {actor_slices}/{critic_slices}"
      )
    actor_corruption = bool(env.observation_manager.cfg["actor"].enable_corruption)
    critic_corruption = bool(
      env.observation_manager.cfg["critic"].enable_corruption
    )
    if actor_corruption is not (profile == "randomized") or critic_corruption:
      raise RuntimeError("runtime observation corruption profile differs")

    arrays = _empty_arrays(num_envs, steps, env.device)
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    reached_endpoint = torch.zeros_like(active)
    settle_remaining = torch.zeros(
      num_envs, dtype=torch.long, device=env.device
    )
    reset_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    cached_observation_only = True
    command_cache_identity = True
    critic_sensor_contact_equal = True
    capture: dict[str, torch.Tensor] = {}
    capture_ids = torch.empty(0, dtype=torch.long, device=env.device)
    termination_names = ["none"]
    confirmed_filter = _DeviceCausalConfirmedContactFilter(
      num_envs, device=env.device, confirmation_ticks=CONFIRMATION_TICKS
    )
    filter_visible_previous = torch.zeros(
      num_envs, dtype=torch.bool, device=env.device
    )
    filter_attempt_id = torch.tensor(
      [int(scenario["matched_slot"]) for scenario in scenarios],
      dtype=torch.long,
      device=env.device,
    )

    def snapshot() -> dict[str, torch.Tensor]:
      contact = _contact(feet_sensor, num_envs)
      force = _force(feet_sensor, num_envs).clone()
      foot_pos = robot.data.site_pos_w[:, foot_ids, :].clone()
      foot_vel = robot.data.site_lin_vel_w[:, foot_ids, :].clone()
      fallback = torch.zeros(num_envs, 4, 3, device=env.device)
      fallback[..., 2] = 1.0
      _, normal, valid = _normal_and_clearance(
        terrain_sensor, foot_pos, fallback
      )
      progress, _, _ = route_state(
        robot.data.root_link_pos_w[:, :2], robot.data.heading_w
      )
      catastrophic, code, names = _catastrophic_and_code(env, num_envs)
      nonlocal termination_names
      termination_names = names
      return {
        "post_contact": contact.clone(),
        "post_force_w": force,
        "post_foot_velocity_w": foot_vel,
        "post_terrain_normal_w": normal.clone(),
        "post_ray_valid": valid.clone(),
        "post_progress": progress.clone(),
        "post_episode_tick": env.episode_length_buf.to(torch.int32).clone(),
        "catastrophic": catastrophic,
        "termination_code": code,
      }

    original_reset_idx = env._reset_idx

    def capture_reset(env_ids: torch.Tensor | None = None) -> None:
      nonlocal capture, capture_ids
      if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)
      full = snapshot()
      capture_ids = env_ids.clone()
      capture = {
        key: value.index_select(0, env_ids).clone()
        for key, value in full.items()
      }
      original_reset_idx(env_ids)

    env._reset_idx = capture_reset  # type: ignore[method-assign]
    executed_steps = 0
    for step_index in range(steps):
      if not bool(active.any()):
        break
      executed_steps = step_index + 1
      pre_pos = robot.data.root_link_pos_w[:, :2].clone()
      pre_heading = robot.data.heading_w.clone()
      pre_progress, pre_cross, pre_heading_error = route_state(
        pre_pos, pre_heading
      )
      motion_active = active & ~reached_endpoint
      command = _commands(
        routes, route_start, pre_pos, pre_heading, motion_active
      )
      command_term.vel_command_b[:] = command
      observation["actor"][:, 6:9].copy_(command)
      observation["critic"][:, 6:9].copy_(command)
      command_cache_identity &= bool(
        torch.equal(observation["actor"][:, 6:9], command)
        and torch.equal(observation["critic"][:, 6:9], command)
      )
      before_contact = _contact(feet_sensor, num_envs).clone()
      before_tick = env.episode_length_buf.clone()
      actor_observation = observation["actor"]
      critic_observation = observation["critic"]
      if actor_observation.shape != (num_envs, ACTOR_DIM):
        raise RuntimeError(
          f"actor observation shape differs: {actor_observation.shape}"
        )
      if critic_observation.shape != (num_envs, CRITIC_DIM):
        raise RuntimeError(
          f"critic observation shape differs: {critic_observation.shape}"
        )
      critic_contact = critic_observation[:, CRITIC_CONTACT_SLICE]
      equal = torch.equal(critic_contact, before_contact.float())
      critic_sensor_contact_equal &= equal
      if not equal:
        raise RuntimeError("critic contact slice differs from native sensor contact")

      filter_visible = active
      filter_episode_start = filter_visible & ~filter_visible_previous
      confirmed_contact, confirmed_changed, confirmed_valid = confirmed_filter.step(
        before_contact,
        attempt_id=filter_attempt_id,
        episode_start=filter_episode_start,
        visible=filter_visible,
      )
      filter_visible_previous.copy_(filter_visible)

      foot_pos = robot.data.site_pos_w[:, foot_ids, :].clone()
      fallback = torch.zeros(num_envs, 4, 3, device=env.device)
      fallback[..., 2] = 1.0
      clearance, _, clearance_valid = _normal_and_clearance(
        terrain_sensor, foot_pos, fallback
      )

      write = active.clone()
      arrays["actor_observation"][:, step_index] = torch.where(
        write[:, None], actor_observation, torch.nan
      )
      arrays["critic_contact"][:, step_index] = (
        write[:, None] & critic_contact.bool()
      )
      arrays["sensor_contact"][:, step_index] = (
        write[:, None] & before_contact
      )
      arrays["confirmed_contact"][:, step_index] = (
        confirmed_contact
      )
      arrays["confirmed_contact_valid"][:, step_index] = (
        confirmed_valid
      )
      arrays["confirmed_contact_changed"][:, step_index] = (
        confirmed_changed
      )
      arrays["clearance"][:, step_index] = torch.where(
        write[:, None], clearance, torch.nan
      )
      arrays["clearance_valid"][:, step_index] = write[:, None] & clearance_valid
      arrays["pre_progress"][:, step_index] = torch.where(
        write, pre_progress, torch.nan
      )
      arrays["command"][:, step_index] = torch.where(
        write[:, None], command, torch.nan
      )
      arrays["pre_episode_tick"][:, step_index] = torch.where(
        write,
        before_tick.to(torch.int32),
        torch.full_like(before_tick, -1, dtype=torch.int32),
      )
      arrays["anchor_active"][:, step_index] = write

      with torch.inference_mode():
        action = policy(observation)
      if action.shape != (num_envs, 12):
        raise RuntimeError(f"policy action shape differs: {action.shape}")
      arrays["policy_action"][:, step_index] = torch.where(
        write[:, None], action, torch.nan
      )

      capture = {}
      capture_ids = torch.empty(0, dtype=torch.long, device=env.device)
      next_observation, _, dones, _ = wrapped.step(action)
      command_term.vel_command_b[:] = command
      reset = dones.bool()
      reset_count += (reset & active).long()
      post = snapshot()
      if capture:
        for key, value in capture.items():
          post[key][capture_ids] = value
      post_progress = post["post_progress"]
      post_cross = route_state(
        torch.where(reset[:, None], pre_pos, robot.data.root_link_pos_w[:, :2]),
        torch.where(reset, pre_heading, robot.data.heading_w),
      )[1]
      post_heading_error = route_state(
        torch.where(reset[:, None], pre_pos, robot.data.root_link_pos_w[:, :2]),
        torch.where(reset, pre_heading, robot.data.heading_w),
      )[2]
      lifecycle = update_attempt_status(
        motion_active,
        post_progress,
        post_cross,
        post_heading_error,
        reset,
        route_length=route_lengths,
        cross_track_tolerance=0.30,
        heading_tolerance=float(np.deg2rad(20.0)),
      )
      reached_endpoint |= lifecycle.completed_now
      settle_remaining = torch.where(
        lifecycle.completed_now,
        torch.full_like(settle_remaining, 10),
        settle_remaining,
      )
      settling = reached_endpoint & active
      decrement = settling & ~lifecycle.completed_now & ~reset
      settle_remaining = torch.where(
        decrement, (settle_remaining - 1).clamp_min(0), settle_remaining
      )
      settle_done = settling & (settle_remaining == 0) & ~reset
      finished = reset | lifecycle.failed_now | settle_done

      for key in (
        "post_contact",
        "post_force_w",
        "post_foot_velocity_w",
        "post_terrain_normal_w",
        "post_ray_valid",
        "post_progress",
        "post_episode_tick",
        "catastrophic",
        "termination_code",
      ):
        target = arrays[key][:, step_index]
        value = post[key]
        if target.dtype.is_floating_point:
          arrays[key][:, step_index] = torch.where(
            write.reshape((num_envs,) + (1,) * (value.ndim - 1)),
            value,
            torch.full_like(value, torch.nan),
          )
        else:
          arrays[key][:, step_index] = torch.where(
            write.reshape((num_envs,) + (1,) * (value.ndim - 1)),
            value,
            torch.zeros_like(value),
          )
      arrays["done"][:, step_index] = write & reset
      arrays["route_completed"][:, step_index] = write & settle_done
      active &= ~finished
      observation = next_observation

    finite_actor = bool(
      torch.isfinite(arrays["actor_observation"][arrays["anchor_active"]]).all()
    )
    finite_action = bool(
      torch.isfinite(arrays["policy_action"][arrays["anchor_active"]]).all()
    )
    confirmed_filter.assert_contract()
    oracle_audit = _cpu_oracle_audit(arrays, scenarios)
    oracle_equal = all(
      bool(oracle_audit[key])
      for key in (
        "saved_confirmed_equals_cpu_oracle_bitwise",
        "saved_changed_equals_cpu_oracle_bitwise",
        "saved_valid_equals_cpu_oracle_bitwise",
      )
    )
    recursive_finite = True
    for value in arrays.values():
      if value.dtype.is_floating_point:
        recursive_finite &= bool(
          torch.isfinite(value[arrays["anchor_active"]]).all()
        )
    if not cached_observation_only or not command_cache_identity:
      raise RuntimeError("cached observation/command patch invariant failed")
    if not critic_sensor_contact_equal:
      raise RuntimeError("pre-action observation timing invariant failed")
    if not finite_actor or not finite_action:
      raise RuntimeError("non-finite actor observation or policy action")
    if (reset_count > 1).any():
      raise RuntimeError("trajectory contains more than one recorded reset")
    if not oracle_equal or oracle_audit["early_flip_or_future_backfill_count"] != 0:
      raise RuntimeError("saved causal contact differs from independent CPU replay")
    if not recursive_finite:
      raise RuntimeError("an active raw row contains a non-finite value")

    return {
      "schema_version": 2,
      "evaluation_suite": (
        "go2_causal_debounced_contact_observability_raw_chunk_v2"
      ),
      "mode": mode,
      "seed": seed,
      "profile": profile,
      "route_kind": route_kind,
      "task_id": TASK_ID,
      "checkpoint": str(CHECKPOINT),
      "checkpoint_sha256": CHECKPOINT_SHA256,
      "contract": str(CONTRACT),
      "contract_sha256": _sha256(CONTRACT),
      "collector": str(Path(__file__).resolve()),
      "collector_sha256": _sha256(Path(__file__).resolve()),
      "timing_reference": str(TIMING_REFERENCE),
      "timing_reference_sha256": _sha256(TIMING_REFERENCE),
      "filter_module": str(FILTER_MODULE),
      "causal_filter_sha256": _sha256(FILTER_MODULE),
      "git_head": _git_head(),
      "profile_settings": profile_settings,
      "episode_settings": episode_settings,
      "native_foot_names": NATIVE_FOOT_NAMES,
      "native_geom_names": NATIVE_GEOM_NAMES,
      "canonical_foot_names": CANONICAL_FOOT_NAMES,
      "native_to_canonical_permutation": NATIVE_TO_CANONICAL,
      "runtime_sensor_names": sensor_names,
      "actor_term_slices": actor_slices,
      "critic_term_slices": critic_slices,
      "actor_observation_corruption": actor_corruption,
      "critic_observation_corruption": critic_corruption,
      "scenarios": scenarios,
      "steps_requested": steps,
      "smoke_timeout_steps": smoke_timeout_steps,
      "steps_executed": executed_steps,
      "num_envs": num_envs,
      "reset_count": reset_count.cpu(),
      "termination_code_names": termination_names,
      "causal_filter_audit": oracle_audit,
      "invariants": {
        "training_changed": False,
        "learn_called": False,
        "checkpoint_sha256_verified": True,
        "actor_critic_action_dims_verified": True,
        "runtime_contact_order_exact": True,
        "critic_contact_equals_native_sensor": critic_sensor_contact_equal,
        "command_cache_patch_exact": command_cache_identity,
        "initial_post_placement_observation_refreshes": 1,
        "external_observation_recompute_inside_loop": not cached_observation_only,
        "terminal_state_captured_inside_reset_hook": True,
        "post_reset_rows_excluded": True,
        "confirmed_online_replay_bitwise_equal": oracle_equal,
        "confirmed_early_or_backfill_count": int(
          oracle_audit["early_flip_or_future_backfill_count"]
        ),
        "confirmed_update_before_policy": True,
        "confirmed_reset_isolated": True,
        "recursive_finite": recursive_finite,
      },
      "arrays": {key: value.cpu() for key, value in arrays.items()},
    }
  finally:
    env.close()


def main() -> None:
  configure_torch_backends()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--profile", choices=PROFILES, required=True)
  parser.add_argument("--route-kind", choices=ROUTE_KINDS, required=True)
  parser.add_argument("--steps", type=int, default=2400)
  parser.add_argument("--matched-slots", type=int, nargs="*")
  parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
  parser.add_argument("--smoke-timeout-steps", type=int)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite raw chunk: {output}")
  payload = collect(
    seed=args.seed,
    profile=args.profile,
    route_kind=args.route_kind,
    steps=args.steps,
    matched_slots=(
      None if args.matched_slots is None else tuple(args.matched_slots)
    ),
    mode=args.mode,
    device=args.device,
    smoke_timeout_steps=args.smoke_timeout_steps,
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb") as stream:
    torch.save(payload, stream)
  print(f"WROTE {output}")
  print(f"SHA256 {_sha256(output)}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
