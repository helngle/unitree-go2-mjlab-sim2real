"""Run one 32-env optimizer update as excluded technical teacher evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.velocity.config.go2  # noqa: F401
from src.tasks.velocity.privileged_teacher_schema import MATCHED_RNG_SEED, TASK_IDS
from src.tasks.velocity.rl.privileged_teacher_transfer import (
  ALLOW_ENV_STATE_MISMATCH_ENV,
  SOURCE_CHECKPOINT,
  SOURCE_MANIFEST_PATH_ENV,
  SOURCE_MANIFEST_SHA256_ENV,
  sha256_file,
)


def _finite(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(_finite(item) for item in value.values())
  if isinstance(value, (tuple, list)):
    return all(_finite(item) for item in value)
  return True


def run_smoke(
  *, arm: str, manifest: Path, manifest_sha: str, run_dir: Path,
  device: str,
) -> dict[str, Any]:
  if sha256_file(manifest) != manifest_sha:
    raise ValueError("source manifest SHA256 mismatch")
  if run_dir.exists():
    raise FileExistsError(run_dir)
  run_dir.mkdir(parents=True)
  os.environ[SOURCE_MANIFEST_PATH_ENV] = str(manifest.resolve())
  os.environ[SOURCE_MANIFEST_SHA256_ENV] = manifest_sha
  os.environ[ALLOW_ENV_STATE_MISMATCH_ENV] = "1"
  task_id = TASK_IDS[arm]
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 32
  env_cfg.seed = MATCHED_RNG_SEED
  agent_cfg.seed = MATCHED_RNG_SEED
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      raise RuntimeError("privileged teacher task has no runner")
    runner = runner_cls(wrapped, asdict(agent_cfg), str(run_dir), device=device)
    transfer = runner.load(str(SOURCE_CHECKPOINT), strict=True, map_location=device)
    if transfer["technical_smoke_env_state_override"] is not True:
      raise RuntimeError("32-env smoke is not marked as technical evidence")
    actor_before = {
      key: value.detach().cpu().clone()
      for key, value in runner.alg.actor.state_dict().items()
    }
    critic_before = {
      key: value.detach().cpu().clone()
      for key, value in runner.alg.critic.state_dict().items()
    }
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    actor_after = runner.alg.actor.state_dict()
    critic_after = runner.alg.critic.state_dict()
    actor_changed = any(
      not torch.equal(actor_before[key], value.detach().cpu())
      for key, value in actor_after.items()
    )
    critic_changed = any(
      not torch.equal(critic_before[key], value.detach().cpu())
      for key, value in critic_after.items()
    )
    optimizer_state = runner.alg.optimizer.state_dict()
    if not actor_changed or not critic_changed or not optimizer_state.get("state"):
      raise RuntimeError("optimizer smoke did not update actor/critic/state")
    if not all(_finite(item) for item in (actor_after, critic_after, optimizer_state)):
      raise RuntimeError("optimizer smoke produced NaN/Inf")
    return {
      "schema_version": 1,
      "evidence_class": "excluded_technical_optimizer_smoke",
      "arm": arm,
      "task_id": task_id,
      "num_envs": 32,
      "updates": 1,
      "seed": MATCHED_RNG_SEED,
      "device": device,
      "source_manifest": str(manifest.resolve()),
      "source_manifest_sha256": manifest_sha,
      "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
      "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
      "transfer": transfer,
      "actor_changed": actor_changed,
      "critic_changed": critic_changed,
      "optimizer_state_present": True,
      "all_finite": True,
      "run_dir": str(run_dir.resolve()),
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", choices=tuple(TASK_IDS), required=True)
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--source-manifest-sha256", required=True)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  result = run_smoke(
    arm=args.arm, manifest=args.source_manifest.expanduser().resolve(),
    manifest_sha=args.source_manifest_sha256,
    run_dir=args.run_dir.expanduser().resolve(), device=args.device,
  )
  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
  output.write_text(encoded, encoding="utf-8")
  print(encoded, end="")


if __name__ == "__main__":
  main()
