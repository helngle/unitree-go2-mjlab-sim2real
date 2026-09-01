"""No-learning runtime preflight for the V7 stance-slip shaping task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401


TASK_ID = "Unitree-Go2-Rough-V7-StanceSlip"
REWARD_NAME = "terrain_tangent_stance_slip"
V7_CHECKPOINT = Path(
  "logs/rsl_rl/go2_velocity/"
  "2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/"
  "model_13600.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
  "73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff"
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def run_preflight(
  *,
  checkpoint: Path,
  num_envs: int,
  steps: int,
  seed: int,
  device: str,
  full_resume: bool = False,
) -> dict[str, Any]:
  """Load the locked V7 actor and roll out without calling ``learn``."""
  checkpoint = checkpoint.expanduser().resolve()
  expected = V7_CHECKPOINT.resolve()
  if checkpoint != expected:
    raise ValueError(f"checkpoint must be locked V7 model_13600.pt: {expected}")
  checkpoint_sha256 = _sha256(checkpoint)
  if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("locked V7 checkpoint SHA256 mismatch")
  if num_envs <= 0 or steps <= 0:
    raise ValueError("num_envs and steps must be positive")

  torch.manual_seed(seed)
  env_cfg = load_env_cfg(TASK_ID)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    if full_resume:
      runner.load(str(checkpoint), strict=True, map_location=device)
    else:
      runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
    policy = runner.get_inference_policy(device=device)
    observation = wrapped.get_observations()
    term_index = env.reward_manager.active_terms.index(REWARD_NAME)
    weight = env.reward_manager.get_term_cfg(REWARD_NAME).weight
    if weight == 0.0:
      raise RuntimeError("preflight requires a non-zero stance-slip reward weight")

    raw_cost_rows: list[torch.Tensor] = []
    action_finite = True
    reward_finite = True
    term_finite = True
    reset_count = 0
    for _ in range(steps):
      with torch.inference_mode():
        action = policy(observation)
      observation, reward, dones, _ = wrapped.step(action)
      weighted_rate = env.reward_manager._step_reward[:, term_index]
      raw_cost_rows.append((weighted_rate / weight).detach().clone())
      action_finite &= bool(torch.isfinite(action).all())
      reward_finite &= bool(torch.isfinite(reward).all())
      term_finite &= bool(torch.isfinite(weighted_rate).all())
      reset_count += int(dones.sum())

    raw_cost = torch.stack(raw_cost_rows)
    runtime_metrics = {
      key: float(value)
      for key, value in env.extras["log"].items()
      if "terrain_tangent" in key
    }
    result = {
      "schema_version": 1,
      "task_id": TASK_ID,
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": checkpoint_sha256,
      "num_envs": num_envs,
      "steps": steps,
      "seed": seed,
      "device": device,
      "learn_called": False,
      "load_mode": "full_resume" if full_resume else "actor_only",
      "strict_load": True,
      "restored_iteration": int(runner.current_learning_iteration),
      "common_step_counter": int(env.common_step_counter),
      "reward_name": REWARD_NAME,
      "reward_weight": weight,
      "finite": {
        "actions": action_finite,
        "total_reward": reward_finite,
        "stance_slip_term": term_finite,
      },
      "raw_cost": {
        "mean": float(raw_cost.mean()),
        "p95": float(torch.quantile(raw_cost.flatten(), 0.95)),
        "max": float(raw_cost.max()),
      },
      "weighted_reward_rate_mean": float((raw_cost * weight).mean()),
      "reset_count": reset_count,
      "runtime_metrics_last_step": runtime_metrics,
    }
    if not all(result["finite"].values()):
      raise RuntimeError("non-finite value observed during stance-slip preflight")
    return result
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=V7_CHECKPOINT)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--full-resume", action="store_true")
  parser.add_argument("--output-file", type=Path)
  args = parser.parse_args()
  result = run_preflight(
    checkpoint=args.checkpoint,
    num_envs=args.num_envs,
    steps=args.steps,
    seed=args.seed,
    device=args.device,
    full_resume=args.full_resume,
  )
  payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
  print(payload)
  if args.output_file is not None:
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
