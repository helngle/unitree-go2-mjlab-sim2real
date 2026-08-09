"""Run one 32-env optimizer update for the contact-force Teacher candidate."""

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
from src.tasks.velocity.contact_force_teacher_schema import (
  CANDIDATE_ACTOR_DIM,
  CONTACT_FORCE_ACTOR_SLICE,
  CONTACT_FORCE_CRITIC_SLICE,
  CONTACT_FORCE_NATIVE_ORDER,
  MATCHED_RNG_SEED,
  SOURCE_ACTOR_DIM,
  TASK_IDS,
  schema_sha256,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (
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


def _runtime_foot_order(env: ManagerBasedRlEnv) -> tuple[str, ...]:
  sensor = env.scene["feet_ground_contact"]
  # ContactSensor stores one private slot per (primary, requested field), so the
  # configured ("found", "force") fields repeat every foot twice.  The force
  # tensor itself contains one vector per primary because num_slots=1.
  names = tuple(
    slot.primary_name for slot in sensor._slots if slot.field_name == "force"
  )
  suffix = "_foot_collision"
  if any(not name.endswith(suffix) for name in names):
    raise RuntimeError(f"unexpected foot contact slot names: {names}")
  return tuple(name.removesuffix(suffix) for name in names)


def run_smoke(
  *, manifest: Path, manifest_sha: str, run_dir: Path, device: str,
) -> dict[str, Any]:
  if sha256_file(manifest) != manifest_sha:
    raise ValueError("source manifest SHA256 mismatch")
  if run_dir.exists():
    raise FileExistsError(run_dir)
  run_dir.mkdir(parents=True)
  os.environ[SOURCE_MANIFEST_PATH_ENV] = str(manifest.resolve())
  os.environ[SOURCE_MANIFEST_SHA256_ENV] = manifest_sha
  os.environ[ALLOW_ENV_STATE_MISMATCH_ENV] = "1"

  arm = "candidate_246"
  task_id = TASK_IDS[arm]
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 32
  env_cfg.seed = MATCHED_RNG_SEED
  agent_cfg.seed = MATCHED_RNG_SEED
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    native_order = _runtime_foot_order(env)
    if native_order != CONTACT_FORCE_NATIVE_ORDER:
      raise RuntimeError(
        f"runtime foot order differs: {native_order} != {CONTACT_FORCE_NATIVE_ORDER}"
      )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      raise RuntimeError("contact-force Teacher task has no runner")
    runner = runner_cls(wrapped, asdict(agent_cfg), str(run_dir), device=device)
    transfer = runner.load(str(SOURCE_CHECKPOINT), strict=True, map_location=device)
    if transfer["technical_smoke_env_state_override"] is not True:
      raise RuntimeError("32-env smoke is not marked as technical evidence")

    observation = wrapped.get_observations()
    actor_obs = observation["actor"]
    critic_obs = observation["critic"]
    if actor_obs.shape != (32, CANDIDATE_ACTOR_DIM):
      raise RuntimeError(f"unexpected actor observation shape: {actor_obs.shape}")
    a_start, a_end = CONTACT_FORCE_ACTOR_SLICE
    c_start, c_end = CONTACT_FORCE_CRITIC_SLICE
    contact_force_identity = torch.equal(
      actor_obs[:, a_start:a_end], critic_obs[:, c_start:c_end]
    )
    if not contact_force_identity:
      raise RuntimeError("actor/critic contact-force observations differ")

    actor_before = {
      key: value.detach().cpu().clone()
      for key, value in runner.alg.actor.state_dict().items()
    }
    critic_before = {
      key: value.detach().cpu().clone()
      for key, value in runner.alg.critic.state_dict().items()
    }
    new_columns_before = actor_before["mlp.0.weight"][:, SOURCE_ACTOR_DIM:]
    if not torch.equal(new_columns_before, torch.zeros_like(new_columns_before)):
      raise RuntimeError("candidate contact-force columns are not zero initialized")

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
    new_columns_after = actor_after["mlp.0.weight"][
      :, SOURCE_ACTOR_DIM:
    ].detach().cpu()
    new_columns_learned = bool((new_columns_after != 0).any())
    optimizer_state = runner.alg.optimizer.state_dict()
    if not actor_changed or not critic_changed or not optimizer_state.get("state"):
      raise RuntimeError("optimizer smoke did not update actor/critic/state")
    if not new_columns_learned:
      raise RuntimeError("contact-force actor columns did not learn")
    if not all(_finite(item) for item in (actor_after, critic_after, optimizer_state)):
      raise RuntimeError("optimizer smoke produced NaN/Inf")
    return {
      "schema_version": 1,
      "evidence_class": "excluded_technical_optimizer_smoke",
      "experiment": "go2_contact_force_teacher_v1",
      "schema_sha256": schema_sha256(),
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
      "dimensions": {"actor": 246, "critic": 261, "action": 12},
      "runtime_foot_order": native_order,
      "actor_critic_contact_force_identity": contact_force_identity,
      "transfer": transfer,
      "actor_changed": actor_changed,
      "critic_changed": critic_changed,
      "new_contact_force_columns_learned": new_columns_learned,
      "new_contact_force_weight_norm": float(torch.linalg.vector_norm(new_columns_after)),
      "optimizer_state_present": True,
      "all_finite": True,
      "run_dir": str(run_dir.resolve()),
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--source-manifest-sha256", required=True)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  result = run_smoke(
    manifest=args.source_manifest.expanduser().resolve(),
    manifest_sha=args.source_manifest_sha256,
    run_dir=args.run_dir.expanduser().resolve(),
    device=args.device,
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
