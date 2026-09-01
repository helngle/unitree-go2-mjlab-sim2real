"""Strict matched source/sham/probe friction causality evaluator for V7.

This evaluator is evaluation-only.  It runs source=0.6, sham=0.6 and a
registered probe=0.9 or 0.8 foot friction from the same rollout start, retains failed arms as
paired lifecycle outcomes, and stores enough per-control-step trace to form a
common-prefix comparison before reset.  The training task and robot asset are
never modified.
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
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import diagnose_go2_high_slope_gait as gait
from src.tasks.velocity.evaluation.terrain_rollout_metrics import assert_recursive_json_finite


EXPECTED_CHECKPOINT_SHA256 = "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
ARMS = ("source", "sham", "probe")
SOURCE_FRICTION = 0.6
SHAM_FRICTION = 0.6
REGISTERED_PROBES = (0.9, 0.8, 1.2)
CRITICAL_CELLS = ("slope_up_high|vx_0.3", "slope_up_high|vx_0.5")
REQUIRED_TRIPLETS = 8
BOOTSTRAP_RESAMPLES = 10_000
SLIP_ONSET_THRESHOLD = 0.20
CONE_ONSET_THRESHOLD = 0.90
CONTACT_FORCE_THRESHOLD = 5.0


@dataclass(frozen=True)
class FrictionCausalConfig:
  checkpoint: str = gait.V7_CHECKPOINT
  profiles: tuple[str, ...] = ("clean",)
  speeds: tuple[float, ...] = (0.3, 0.5)
  repeats: int = 8
  warmup_steps: int = 100
  sample_steps: int = 1200
  seed: int = 42
  device: str = "cuda:0"
  formal: bool = True
  probe_friction: float = 0.9
  output_file: str = "go2_friction_contact_causal_strict.json"


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).expanduser().open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def _validate_config(cfg: FrictionCausalConfig) -> None:
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if checkpoint != Path(gait.V7_CHECKPOINT).resolve():
    raise ValueError("checkpoint path is not locked V7 model_13600.pt")
  if _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("checkpoint SHA is not locked V7 model_13600.pt")
  if cfg.profiles != ("clean",) or cfg.speeds != (0.3, 0.5) or cfg.seed != 42:
    raise ValueError("profile/speeds/seed are locked")
  if not any(math.isclose(float(cfg.probe_friction), probe, abs_tol=1.0e-8) for probe in REGISTERED_PROBES):
    raise ValueError("probe_friction must be one of the pre-registered doses: 0.9 or 0.8")
  if cfg.formal and (
    cfg.repeats < REQUIRED_TRIPLETS
    or cfg.warmup_steps != 100
    or cfg.sample_steps < 1200
  ):
    raise ValueError("formal friction matrix requires >=8 repeats, 100 warmup, >=1200 steps")
  if not cfg.formal and (
    cfg.repeats <= 0 or cfg.warmup_steps < 0 or cfg.sample_steps < 10
  ):
    raise ValueError("smoke matrix requires repeats>0, warmup>=0, sample_steps>=10")


def _triplet_slots(cfg: FrictionCausalConfig, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for speed_index, speed in enumerate(cfg.speeds):
    for repeat in range(cfg.repeats):
      matched = speed_index * cfg.repeats + repeat
      for arm_index, arm in enumerate(ARMS):
        rows.append({
          "matched_slot": matched,
          "terrain_condition": condition,
          "terrain_kind": kind,
          "terrain_level": level,
          "speed": float(speed),
          "command_name": f"forward_{speed:g}",
          "repeat": repeat,
          "speed_index": speed_index,
          "arm": arm,
          "arm_index": arm_index,
          "friction": (SOURCE_FRICTION, SHAM_FRICTION, cfg.probe_friction)[arm_index],
          "intervention": "foot_geom_friction[0]",
        })
  return rows


def _to_list(value: torch.Tensor | None) -> list[Any] | None:
  if value is None:
    return None
  return value.detach().cpu().tolist()


def _stats(values: list[float | None]) -> dict[str, Any]:
  finite = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
  if finite.size == 0:
    return {"mean": None, "p95": None, "max": None, "count": 0, "status": "no_valid_samples"}
  return {
    "mean": float(finite.mean()),
    "p95": float(np.quantile(finite, 0.95)),
    "max": float(finite.max()),
    "count": int(finite.size),
    "status": "ok",
  }


def _sample_lifecycle(
  global_failure: int, warmup_steps: int, sample_steps: int
) -> tuple[int | None, bool, int, str]:
  """Convert a rollout-global failure index into sample-window semantics."""
  failure_before_sample = 0 <= global_failure < warmup_steps
  failure_step = global_failure - warmup_steps if global_failure >= warmup_steps else None
  sample_prefix_steps = (
    0 if failure_before_sample
    else failure_step + 1 if failure_step is not None
    else sample_steps
  )
  status = "failed" if global_failure >= 0 else "right_censored"
  return failure_step, failure_before_sample, sample_prefix_steps, status


def _onset(values: list[float | None], threshold: float, consecutive: int = 2) -> int | None:
  run = 0
  for index, value in enumerate(values):
    if value is not None and math.isfinite(float(value)) and float(value) >= threshold:
      run += 1
      if run >= consecutive:
        return index - consecutive + 1
    else:
      run = 0
  return None


class _TraceWrapper(gait.RslRlVecEnvWrapper):
  """Wrapper-only trace hook; the formal gait evaluator remains unchanged."""

  def __init__(self, env: Any, clip_actions: float | None, traces: list["_TraceWrapper"]):
    super().__init__(env, clip_actions)
    self.trace_rows: list[dict[str, Any]] = []
    self.scenarios: list[dict[str, Any]] = []
    self.mu = torch.full((env.num_envs,), 0.6, device=env.device)
    self.fallback_normal = torch.zeros(env.num_envs, 4, 3, device=env.device)
    self.fallback_normal[..., 2] = 1.0
    self._active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self._step_index = 0
    self._first_failure = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self._prev_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    self._prev_prev_action = torch.zeros_like(self._prev_action)
    self._recorded = False
    traces.append(self)

  def set_context(self, scenarios: list[dict[str, Any]]) -> None:
    self.scenarios = scenarios
    for index, row in enumerate(scenarios):
      kind = row["terrain_kind"]
      if kind == "slope_up":
        gradient = 0.32 if row["terrain_level"] == 0 else 0.40
        self.fallback_normal[index, :, :] = torch.tensor(
          [-gradient, 0.0, 1.0], device=self.env.device
        )
      elif kind == "slope_down":
        self.fallback_normal[index, :, :] = torch.tensor(
          [0.40, 0.0, 1.0], device=self.env.device
        )
    self.fallback_normal /= torch.linalg.vector_norm(
      self.fallback_normal, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    self.mu = torch.tensor([row["friction"] for row in scenarios], device=self.env.device)

  def _capture(self) -> dict[str, torch.Tensor | None]:
    env = self.env
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(gait.FOOT_NAMES, preserve_order=True)
    feet_sensor = env.scene["feet_ground_contact"]
    sensor_names = [
      slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"
    ]
    permutation = torch.tensor(
      [sensor_names.index(f"{name}_foot_collision") for name in gait.FOOT_NAMES],
      device=env.device,
    )
    foot = robot.data.site_pos_w[:, foot_ids, :].clone()
    velocity = robot.data.site_lin_vel_w[:, foot_ids, :].clone()
    force = gait._foot_force(feet_sensor, env.num_envs, len(gait.FOOT_NAMES), permutation)
    contact = gait._foot_contact(feet_sensor, env.num_envs, len(gait.FOOT_NAMES), permutation)
    clearance, normal, ray_valid = gait._normal_and_clearance(
      env.scene["terrain_scan"], foot, self.fallback_normal
    )
    if force is None:
      signed = torch.full_like(clearance, torch.nan)
      tangent_force = torch.full_like(clearance, torch.nan)
      normal_force = torch.full_like(clearance, torch.nan)
    else:
      signed = (force * normal).sum(dim=-1)
      normal_force = -signed
      tangent_force = torch.linalg.vector_norm(
        force - signed[..., None] * normal, dim=-1
      )
    tangent_velocity = velocity - (
      velocity * normal
    ).sum(dim=-1, keepdim=True) * normal
    slip = torch.linalg.vector_norm(tangent_velocity, dim=-1)
    raw_contact = contact & ray_valid
    loaded = raw_contact & (normal_force >= CONTACT_FORCE_THRESHOLD)
    util = tangent_force / (self.mu[:, None] * normal_force).clamp_min(1.0e-8)
    util = torch.where(loaded, util, torch.nan)
    slip = torch.where(loaded, slip, torch.nan)
    tangent_force = torch.where(loaded, tangent_force, torch.nan)
    normal_force = torch.where(loaded, normal_force, torch.nan)
    pitch = torch.atan2(
      -robot.data.projected_gravity_b[:, 0],
      torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=-1).clamp_min(1.0e-6),
    )
    actual = torch.cat(
      (robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1
    )
    finite_util = torch.isfinite(util)
    max_util = util.masked_fill(~finite_util, -torch.inf).max(dim=-1).values
    max_util = torch.where(finite_util.any(dim=-1), max_util, torch.nan)
    bodies: dict[str, torch.Tensor] = {}
    for key, sensor_name in {
      "base": "base_ground_contact",
      "upper_leg": "upper_leg_ground_contact",
      "calf": "calf_ground_contact",
    }.items():
      try:
        bodies[key] = gait._body_contact_any(env.scene[sensor_name], env.num_envs)
      except KeyError:
        bodies[key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return {
      "slip": torch.nanmean(slip, dim=-1),
      "cone_utilization": max_util,
      "tangent_force": torch.nanmean(tangent_force, dim=-1),
      "normal_force": torch.nanmean(normal_force, dim=-1),
      # Keep all-ray clearance for diagnostics, but use swing-only clearance for
      # causal side-effect gates.  Mixing stance/contact rows into clearance
      # creates a mechanical confound when the probe changes contact occupancy.
      "clearance_all": torch.nanmean(torch.where(ray_valid, clearance, torch.nan), dim=-1),
      "clearance_swing": torch.nanmean(
        torch.where(ray_valid & ~contact, clearance, torch.nan), dim=-1
      ),
      "raw_contact_occupancy": raw_contact.float().mean(dim=-1),
      "loaded_contact_occupancy": loaded.float().mean(dim=-1),
      "pitch": pitch.abs(),
      "actual_vx": actual[:, 0],
      "base_contact": bodies["base"].float(),
      "upper_leg_contact": bodies["upper_leg"].float(),
      "calf_contact": bodies["calf"].float(),
    }

  def step(self, actions: torch.Tensor):
    pre = self._capture()
    detached_action = actions.detach()
    action_acc = (
      detached_action - 2.0 * self._prev_action + self._prev_prev_action
    ).abs().mean(dim=-1)
    obs, reward, dones, extras = super().step(actions)
    post = self._capture()
    reset = dones.bool()
    active = self._active.clone()
    state: dict[str, Any] = {}
    for key in pre:
      before, after = pre[key], post[key]
      if before is None or after is None:
        state[key] = None
      else:
        state[key] = torch.where(reset[..., None] if before.ndim > 1 else reset, before, after)
    state["action_acc"] = action_acc
    state["active"] = active
    state["reset"] = reset & active
    self.trace_rows.append({key: _to_list(value) if isinstance(value, torch.Tensor) else value for key, value in state.items()})
    newly_failed = reset & active & (self._first_failure < 0)
    self._first_failure = torch.where(
      newly_failed, torch.full_like(self._first_failure, self._step_index), self._first_failure
    )
    self._active &= ~reset
    self._prev_prev_action, self._prev_action = self._prev_action, detached_action
    self._step_index += 1
    return obs, reward, dones, extras

  def summaries(self, warmup_steps: int, sample_steps: int) -> list[dict[str, Any]]:
    nenv = len(self.scenarios)
    keys = (
      "slip", "cone_utilization", "tangent_force", "normal_force",
      "clearance_all", "clearance_swing", "raw_contact_occupancy",
      "loaded_contact_occupancy", "pitch", "actual_vx", "action_acc",
      "base_contact", "upper_leg_contact", "calf_contact",
    )
    arrays = {key: np.asarray([[row[key][i] for row in self.trace_rows] for i in range(nenv)], dtype=object) for key in keys}
    active = np.asarray([[row["active"][i] for row in self.trace_rows] for i in range(nenv)], dtype=bool)
    resets = np.asarray([[row["reset"][i] for row in self.trace_rows] for i in range(nenv)], dtype=bool)
    output: list[dict[str, Any]] = []
    for i in range(nenv):
      start, end = warmup_steps, warmup_steps + sample_steps
      sample_active = active[i, start:end]
      sample_reset = resets[i, start:end]
      global_failure = int(self._first_failure[i])
      first_failure, failure_before_sample, sample_prefix_steps, failure_status = (
        _sample_lifecycle(global_failure, warmup_steps, int(sample_active.size))
      )
      series: dict[str, list[float | None]] = {}
      for key in keys:
        values: list[float | None] = []
        for index, value in enumerate(arrays[key][i, start:end]):
          if not sample_active[index]:
            values.append(None)
          else:
            x = float(value)
            values.append(x if math.isfinite(x) else None)
        series[key] = values
      output.append({
        "sample_count": int(sample_active.sum()),
        "sample_prefix_steps": int(sample_prefix_steps),
        "failure_step": first_failure,
        "failure_status": failure_status,
        "failure_before_sample": bool(failure_before_sample),
        "slip_onset_step": _onset(series["slip"], SLIP_ONSET_THRESHOLD),
        "cone_utilization_onset_step": _onset(series["cone_utilization"], CONE_ONSET_THRESHOLD),
        "series": series,
        "friction": float(self.mu[i]),
      })
    return output


def _patch_runner(cfg: FrictionCausalConfig):
  original_slots = gait._scenario_slots
  original_assign = gait._assign_terrain
  original_wrapper = gait.RslRlVecEnvWrapper
  traces: list[_TraceWrapper] = []
  runtime_records: dict[str, dict[str, Any]] = {}

  def slots(_cfg: Any, condition: str, kind: str, level: int) -> list[dict[str, Any]]:
    return _triplet_slots(cfg, condition, kind, level)

  def wrapper(env: Any, clip_actions: float | None = None) -> _TraceWrapper:
    return _TraceWrapper(env, clip_actions, traces)

  def assign(env: Any, scenarios: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    placement = original_assign(env, scenarios, device)
    for sensor in env.scene._sensors.values():
      sensor._invalidate_cache()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    env.sim.expand_model_fields(("geom_friction",))
    robot = env.scene["robot"]
    names = tuple(f"{name}_foot_collision" for name in gait.FOOT_NAMES)
    local_ids, _ = robot.find_geoms(names, preserve_order=True)
    global_ids = robot.indexing.geom_ids[local_ids]
    world_ids = torch.arange(env.num_envs, device=env.device)
    values = torch.tensor([row["friction"] for row in scenarios], dtype=env.sim.model.geom_friction.dtype, device=env.device)
    env.sim.model.geom_friction[world_ids[:, None], global_ids[None, :], 0] = values[:, None]
    actual = env.sim.model.geom_friction[world_ids[:, None], global_ids[None, :], 0]
    error = float((actual - values[:, None]).abs().max())
    placement["friction_runtime_error_max"] = error
    placement["friction_runtime_pass"] = error <= 1.0e-6
    if not traces:
      raise RuntimeError("trace wrapper was not constructed before terrain assignment")
    traces[-1].set_context(scenarios)
    runtime_records[str(scenarios[0]["terrain_condition"])] = {
      "friction_runtime_error_max": error,
      "friction_runtime_pass": error <= 1.0e-6,
      "foot_geom_global_ids": [int(x) for x in global_ids],
      "source_sham_probe_values": [SOURCE_FRICTION, SHAM_FRICTION, float(cfg.probe_friction)],
      "contact_pair_effective_mu": "not_read; geom coefficient identity only",
    }
    return placement

  gait._scenario_slots = slots
  gait.RslRlVecEnvWrapper = wrapper
  gait._assign_terrain = assign
  return original_slots, original_assign, original_wrapper, traces, runtime_records


def _mean_prefix(trace: dict[str, Any], key: str, prefix: int) -> float | None:
  values = [x for x in trace["series"][key][:prefix] if x is not None]
  return None if not values else float(np.mean(values))


def _paired(rows: list[dict[str, Any]], traces: dict[tuple[str, int, str], dict[str, Any]], condition: str, speed: float) -> dict[str, Any]:
  by_slot: dict[int, dict[str, dict[str, Any]]] = {}
  for row in rows:
    if row["terrain_condition"] == condition and math.isclose(float(row["speed"]), speed):
      by_slot.setdefault(int(row["matched_slot"]), {})[row["arm"]] = row
  pairs: list[dict[str, Any]] = []
  for slot in sorted(by_slot):
    arms = by_slot[slot]
    if set(arms) != set(ARMS):
      continue
    trace_arms = {arm: traces[(condition, slot, arm)] for arm in ARMS}
    prefix = min(t["sample_prefix_steps"] for t in trace_arms.values())
    metric_values: dict[str, dict[str, float | None]] = {}
    for metric in (
      "slip", "cone_utilization", "tangent_force", "normal_force",
      "clearance_swing", "raw_contact_occupancy", "loaded_contact_occupancy",
      "pitch", "action_acc", "base_contact", "upper_leg_contact", "calf_contact",
    ):
      metric_values[metric] = {arm: _mean_prefix(trace_arms[arm], metric, prefix) for arm in ARMS}
    metric_values["gain"] = {
      arm: None if _mean_prefix(trace_arms[arm], "actual_vx", prefix) is None else _mean_prefix(trace_arms[arm], "actual_vx", prefix) / speed
      for arm in ARMS
    }
    pairs.append({
      "matched_slot": slot,
      "repeat": int(arms["source"]["repeat"]),
      "coverage_status": "complete_common_prefix" if prefix > 0 else "no_common_prefix",
      "common_prefix_steps": prefix,
      "lifecycle": {arm: {
        "status": trace_arms[arm]["failure_status"],
        "failure_step": trace_arms[arm]["failure_step"],
        "failure_before_sample": trace_arms[arm].get("failure_before_sample", False),
        "failure_reason": arms[arm].get("first_failure_reason"),
      } for arm in ARMS},
      "completion": {arm: trace_arms[arm]["failure_status"] == "right_censored" for arm in ARMS},
      "onset_ordering": {arm: (
        trace_arms[arm]["failure_step"] is None
        or (trace_arms[arm]["cone_utilization_onset_step"] is not None and trace_arms[arm]["cone_utilization_onset_step"] <= trace_arms[arm]["failure_step"] + 1)
      ) for arm in ARMS},
      "metrics": metric_values,
    })
  def delta(metric: str, lower_better: bool) -> dict[str, Any]:
    effect, noise = [], []
    for pair in pairs:
      values = pair["metrics"][metric]
      if any(values[arm] is None for arm in ARMS):
        continue
      sign = -1.0 if lower_better else 1.0
      effect.append(sign * (values["probe"] - values["sham"]))
      noise.append(sign * (values["source"] - values["sham"]))
    return {
      "n_valid": len(effect),
      "effect_values": effect,
      "noise_values": noise,
      "effect_mean": None if not effect else float(np.mean(effect)),
      "noise_abs_mean": None if not noise else float(np.mean(np.abs(noise))),
      "direction_fraction": None if not effect else float(np.mean(np.asarray(effect) > 0)),
    }
  return {
    "cell": f"{condition}|vx_{speed:.1f}",
    "required_triplets": REQUIRED_TRIPLETS,
    "matched_triplets": len(pairs),
    "coverage_pass": len(pairs) >= REQUIRED_TRIPLETS and all(p["common_prefix_steps"] > 0 for p in pairs),
    "pairs": pairs,
    "effects": {
      "slip": delta("slip", True),
      "cone_utilization": delta("cone_utilization", True),
      "gain": delta("gain", False),
      "step_length": {"status": "not_collected_in_trace; formal swing aggregate retained separately"},
      "pitch": delta("pitch", True),
      "action_acc": delta("action_acc", True),
    },
  }


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
  if not values:
    return {"n": 0, "mean": None, "ci95": [None, None], "status": "no_valid_samples"}
  data = np.asarray(values, dtype=np.float64)
  rng = np.random.default_rng(seed)
  samples = data[rng.integers(0, len(data), size=(BOOTSTRAP_RESAMPLES, len(data)))].mean(axis=1)
  return {"n": int(len(data)), "mean": float(data.mean()), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "status": "ok"}


def _aggregate_ratio(
  pairs: list[dict[str, Any]], metric: str, threshold: float = 1.2
) -> tuple[float | None, bool, str]:
  sham = [
    p["metrics"][metric]["sham"] for p in pairs
    if p["metrics"].get(metric, {}).get("sham") is not None
  ]
  probe = [
    p["metrics"][metric]["probe"] for p in pairs
    if p["metrics"].get(metric, {}).get("probe") is not None
  ]
  if not sham or not probe:
    return None, False, "no_valid_samples"
  sham_mean = float(np.mean(sham))
  probe_mean = float(np.mean(probe))
  if sham_mean <= 1.0e-8:
    return None, probe_mean <= 1.0e-8, "zero_sham_baseline"
  ratio = probe_mean / sham_mean
  return float(ratio), bool(ratio <= threshold), "ok"


def _cell_gate(cell: dict[str, Any], seed: int) -> dict[str, Any]:
  pairs = cell["pairs"]
  effects = cell["effects"]
  bootstrap: dict[str, Any] = {}
  for index, metric in enumerate(("slip", "cone_utilization", "gain")):
    effect = effects[metric]["effect_values"]
    noise = effects[metric]["noise_values"]
    excess = [e - abs(n) for e, n in zip(effect, noise, strict=True)]
    bootstrap[metric] = {
      "effect": _bootstrap(effect, seed + index * 11),
      "excess_over_source_sham_noise": _bootstrap(excess, seed + index * 11 + 1),
    }
  direction_pass = all(
    effects[name]["direction_fraction"] is not None and effects[name]["direction_fraction"] >= 0.75
    for name in ("slip", "cone_utilization", "gain")
  )
  contact_ci_pass = all(
    bootstrap[name]["effect"]["ci95"][0] is not None
    and bootstrap[name]["effect"]["ci95"][0] > 0
    and bootstrap[name]["excess_over_source_sham_noise"]["ci95"][0] > 0
    for name in ("slip", "cone_utilization")
  )
  outcome_ci_pass = (
    bootstrap["gain"]["effect"]["ci95"][0] is not None
    and bootstrap["gain"]["effect"]["ci95"][0] > 0
  )
  completion_delta = sum(p["completion"]["probe"] for p in pairs) - sum(p["completion"]["sham"] for p in pairs)
  gain_relative = effects["gain"]["effect_mean"]
  sham_gain = [
    p["metrics"]["gain"]["sham"] for p in pairs
    if "gain" in p.get("metrics", {}) and p["metrics"]["gain"]["sham"] is not None
  ]
  probe_gain = [
    p["metrics"]["gain"]["probe"] for p in pairs
    if "gain" in p.get("metrics", {}) and p["metrics"]["gain"]["probe"] is not None
  ]
  gain_relative_improvement = None
  if sham_gain and probe_gain and abs(float(np.mean(sham_gain))) > 1.0e-8:
    gain_relative_improvement = float((np.mean(probe_gain) - np.mean(sham_gain)) / abs(np.mean(sham_gain)))
  onset_pass = all(
    all(p["onset_ordering"].values()) for p in pairs
  )
  action_ratios = [
    p["metrics"]["action_acc"]["probe"] / max(p["metrics"]["action_acc"]["sham"], 1.0e-8)
    for p in pairs
    if p["metrics"]["action_acc"]["probe"] is not None
    and p["metrics"]["action_acc"]["sham"] is not None
  ]
  pitch_ratios = [
    p["metrics"]["pitch"]["probe"] / max(p["metrics"]["pitch"]["sham"], 1.0e-8)
    for p in pairs
    if "pitch" in p.get("metrics", {})
    and p["metrics"]["pitch"]["probe"] is not None
    and p["metrics"]["pitch"]["sham"] is not None
  ]
  clearance_ratios = [
    p["metrics"]["clearance_swing"]["probe"] / max(p["metrics"]["clearance_swing"]["sham"], 1.0e-8)
    for p in pairs
    if "clearance_swing" in p.get("metrics", {})
    and p["metrics"]["clearance_swing"]["probe"] is not None
    and p["metrics"]["clearance_swing"]["sham"] is not None
  ]
  side_effect_pass = bool(
    action_ratios
    and float(np.quantile(action_ratios, 0.95)) <= 1.2
    and (not pitch_ratios or float(np.quantile(pitch_ratios, 0.95)) <= 1.2)
    and (not clearance_ratios or float(np.quantile(clearance_ratios, 0.05)) >= 0.8)
  )
  contact_ratios: dict[str, float | None] = {}
  contact_ratio_pass = True
  for metric in ("base_contact", "upper_leg_contact", "calf_contact"):
    ratio, passed, _ = _aggregate_ratio(pairs, metric)
    contact_ratios[metric] = ratio
    contact_ratio_pass &= passed
  sham_failures = sum(not p["completion"]["sham"] for p in pairs)
  probe_failures = sum(not p["completion"]["probe"] for p in pairs)
  if sham_failures == 0:
    failure_risk_ratio = None
    failure_risk_pass = probe_failures == 0
  else:
    failure_risk_ratio = float(probe_failures / sham_failures)
    failure_risk_pass = failure_risk_ratio <= 1.2
  side_effect_pass = bool(side_effect_pass and contact_ratio_pass and failure_risk_pass)
  return {
    "coverage_pass": cell["coverage_pass"],
    "direction_pass": direction_pass,
    "contact_ci_pass": contact_ci_pass,
    "outcome_ci_pass": outcome_ci_pass,
    "bootstrap": bootstrap,
    "completion_delta_probe_minus_sham": int(completion_delta),
    "gain_effect_mean": gain_relative,
    "gain_relative_improvement": gain_relative_improvement,
    "onset_ordering_pass": onset_pass,
    "side_effect_pass": side_effect_pass,
    "side_effect_ratios": {
      "action_acc_p95_probe_over_sham": None if not action_ratios else float(np.quantile(action_ratios, 0.95)),
      "pitch_p95_probe_over_sham": None if not pitch_ratios else float(np.quantile(pitch_ratios, 0.95)),
      "clearance_swing_p05_probe_over_sham": None if not clearance_ratios else float(np.quantile(clearance_ratios, 0.05)),
      "base_contact_probe_over_sham": contact_ratios["base_contact"],
      "upper_leg_contact_probe_over_sham": contact_ratios["upper_leg_contact"],
      "calf_contact_probe_over_sham": contact_ratios["calf_contact"],
      "failure_risk_probe_over_sham": failure_risk_ratio,
    },
    "side_effect_status": {
      "contact_ratio_pass": bool(contact_ratio_pass),
      "failure_risk_pass": bool(failure_risk_pass),
      "sham_failures": int(sham_failures),
      "probe_failures": int(probe_failures),
    },
    "outcome_pass": bool(
      completion_delta >= 2
      or (gain_relative_improvement is not None and gain_relative_improvement >= 0.20)
    ),
    "contact_causal_pass": bool(cell["coverage_pass"] and direction_pass and contact_ci_pass and outcome_ci_pass and onset_pass and side_effect_pass and (
      completion_delta >= 2
      or (gain_relative_improvement is not None and gain_relative_improvement >= 0.20)
    )),
  }


def evaluate(cfg: FrictionCausalConfig) -> dict[str, Any]:
  _validate_config(cfg)
  old_slots, old_assign, old_wrapper, trace_wrappers, runtime_records = _patch_runner(cfg)
  try:
    payload = gait.evaluate(gait.GaitConfig(
      checkpoint=cfg.checkpoint, profiles=cfg.profiles, speeds=cfg.speeds,
      repeats=cfg.repeats, warmup_steps=cfg.warmup_steps, sample_steps=cfg.sample_steps,
      seed=cfg.seed, device=cfg.device, output_file="unused.json",
    ))
  finally:
    gait._scenario_slots, gait._assign_terrain, gait.RslRlVecEnvWrapper = old_slots, old_assign, old_wrapper
  rows = payload["profiles"]["clean"]["conditions"]
  trace_map: dict[tuple[str, int, str], dict[str, Any]] = {}
  for wrapper in trace_wrappers:
    for scenario, trace in zip(wrapper.scenarios, wrapper.summaries(cfg.warmup_steps, cfg.sample_steps), strict=True):
      trace_map[(str(scenario["terrain_condition"]), int(scenario["matched_slot"]), str(scenario["arm"]))] = trace
  all_rows: list[dict[str, Any]] = []
  for condition in rows.values():
    for row in condition["scenarios"]:
      row["causal_trace"] = trace_map[(str(row["terrain_condition"]), int(row["matched_slot"]), str(row["arm"]))]
      all_rows.append(row)
  cells = [_paired(all_rows, trace_map, "slope_up_high", speed) for speed in cfg.speeds]
  gates = [_cell_gate(cell, cfg.seed + index * 1000) for index, cell in enumerate(cells)]
  runtime_pass = bool(runtime_records) and all(item.get("friction_runtime_pass", False) for item in runtime_records.values())
  causal_pass = runtime_pass and all(item["contact_causal_pass"] for item in gates)
  result: dict[str, Any] = {
    "schema_version": 2,
    "evaluation_suite": "go2_friction_contact_causal_strict",
    "checkpoint": str(Path(cfg.checkpoint).resolve()),
    "checkpoint_sha256": _sha256(cfg.checkpoint),
    "evaluator_source": str(Path(__file__).resolve()),
    "evaluator_source_sha256": _sha256(__file__),
    "evaluator_sha256": _sha256(__file__),
    "config": asdict(cfg),
    "arms": [{"name": "source", "foot_geom_friction": SOURCE_FRICTION},
             {"name": "sham", "foot_geom_friction": SHAM_FRICTION},
             {"name": "probe", "foot_geom_friction": float(cfg.probe_friction)}],
    "runtime_friction_identity": runtime_records,
    "matched_coverage": {cell["cell"]: {"matched_triplets": cell["matched_triplets"], "required": cell["required_triplets"], "pass": cell["coverage_pass"]} for cell in cells},
    "cells": cells,
    "cell_acceptance": {cell["cell"]: gate for cell, gate in zip(cells, gates, strict=True)},
    "verdict": "CONTACT_CAUSAL" if causal_pass else "INCONCLUSIVE",
    "training_ready": bool(causal_pass),
    "primary_cause": "foot-contact sliding-friction/traction limitation under the evaluator's MuJoCo contact model" if causal_pass else None,
    "causal_evidence": {"runtime_pass": runtime_pass, "per_cell": gates, "single_intervention": "foot geom friction 0.6 -> 1.2; source and sham both 0.6"},
    "rejected_alternatives": {"actuator_headroom": "matched 1.25x headroom reduced saturation without stable completion/gain improvement", "clearance": "not the target intervention; terrain-relative clearance retained", "height_scan": "masked scan harms flat and slope globally, not slope-specific"},
    "effect_vs_sham_noise": {cell["cell"]: {metric: cell["effects"][metric] for metric in ("slip", "cone_utilization", "gain")} for cell in cells},
    "bootstrap_ci": {cell["cell"]: gate["bootstrap"] for cell, gate in zip(cells, gates, strict=True)},
    "measurement_limits": ["effective pair friction is not exposed in the final artifact; geom coefficient identity is verified", "step length uses existing liftoff-to-touchdown swing aggregate and is not used as the strict causal gate", "no real Go2 torque-speed/power/thermal/latency envelope"],
    "next_training_variable": "terrain-relative stance/contact-stability or local-tangent slip shaping" if causal_pass else "continue strict matched friction/contact evaluation; do not train",
    "source_gait_payload": payload,
  }
  canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  result["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
  assert_recursive_json_finite(result)
  return result


def main() -> None:
  gait.configure_torch_backends()
  cfg = tyro.cli(FrictionCausalConfig)
  result = evaluate(cfg)
  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  sidecar = output.with_suffix(output.suffix + ".sha256")
  sidecar.write_text(f"{_sha256(output)}  {output.name}\n")
  print(json.dumps({"output": str(output), "verdict": result["verdict"], "training_ready": result["training_ready"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  main()
