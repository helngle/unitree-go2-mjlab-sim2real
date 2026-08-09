"""No-learning 2048-env preflight for one contact-force Teacher arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
  CRITIC_DIM,
  MATCHED_RNG_SEED,
  SOURCE_ACTOR_DIM,
  SOURCE_ENVIRONMENT_STATE_SHA256,
  TASK_IDS,
)
from src.tasks.velocity.rl.contact_force_teacher_transfer import (
  ALLOW_ENV_STATE_MISMATCH_ENV,
  SOURCE_CHECKPOINT,
  SOURCE_MANIFEST_PATH_ENV,
  SOURCE_MANIFEST_SHA256_ENV,
  sha256_file,
)
from src.tasks.velocity.rl.privileged_teacher_transfer import (
  environment_state_sha256,
)


def _tensor_sha(value: torch.Tensor) -> str:
  tensor = value.detach().cpu().contiguous()
  digest = hashlib.sha256()
  digest.update(str(tuple(tensor.shape)).encode("ascii"))
  digest.update(str(tensor.dtype).encode("ascii"))
  digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def _equal_state(
  actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> bool:
  return actual.keys() == expected.keys() and all(
    torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu())
    for key in actual
  )


def _runtime_force_order(env: ManagerBasedRlEnv) -> tuple[str, ...]:
  sensor = env.scene["feet_ground_contact"]
  suffix = "_foot_collision"
  names = tuple(
    slot.primary_name for slot in sensor._slots if slot.field_name == "force"
  )
  if any(not name.endswith(suffix) for name in names):
    raise RuntimeError(f"unexpected contact-force slot names: {names}")
  return tuple(name.removesuffix(suffix) for name in names)


def run_preflight(
  *, arm: str, manifest: Path, manifest_sha256: str, steps: int, device: str,
) -> dict[str, Any]:
  if arm not in TASK_IDS or steps <= 0:
    raise ValueError("invalid arm or steps")
  manifest = manifest.expanduser().resolve()
  if sha256_file(manifest) != manifest_sha256:
    raise ValueError("source manifest SHA256 mismatch")
  os.environ[SOURCE_MANIFEST_PATH_ENV] = str(manifest)
  os.environ[SOURCE_MANIFEST_SHA256_ENV] = manifest_sha256
  os.environ.pop(ALLOW_ENV_STATE_MISMATCH_ENV, None)

  task_id = TASK_IDS[arm]
  actor_dim = SOURCE_ACTOR_DIM if arm == "control_234" else CANDIDATE_ACTOR_DIM
  torch.manual_seed(MATCHED_RNG_SEED)
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 2048
  env_cfg.seed = MATCHED_RNG_SEED
  agent_cfg.seed = MATCHED_RNG_SEED
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    runtime_order = _runtime_force_order(env)
    if runtime_order != CONTACT_FORCE_NATIVE_ORDER:
      raise RuntimeError(
        f"runtime foot order differs: {runtime_order} != {CONTACT_FORCE_NATIVE_ORDER}"
      )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      raise RuntimeError("contact-force Teacher task has no registered runner")
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    transfer = runner.load(str(SOURCE_CHECKPOINT), strict=True, map_location=device)
    if not transfer["environment_state_restored_exact"]:
      raise RuntimeError("formal environment state was not restored exactly")
    if runner.alg.optimizer.state_dict().get("state"):
      raise RuntimeError("fresh optimizer unexpectedly contains source state")
    state = {
      "common_step_counter": env.common_step_counter,
      "terrain_levels": env.scene.terrain.terrain_levels,
      "terrain_types": env.scene.terrain.terrain_types,
    }
    if environment_state_sha256(state) != SOURCE_ENVIRONMENT_STATE_SHA256:
      raise RuntimeError("runtime environment state SHA256 mismatch")

    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    actor_state = runner.alg.actor.state_dict()
    critic_state = runner.alg.critic.state_dict()
    prefix_exact = all(
      torch.equal(
        actor_state[key].detach().cpu()[..., :SOURCE_ACTOR_DIM],
        value.detach().cpu(),
      ) if value.ndim > 0 and value.shape[-1] == SOURCE_ACTOR_DIM else
      torch.equal(actor_state[key].detach().cpu(), value.detach().cpu())
      for key, value in source["actor_state_dict"].items()
    )
    critic_exact = _equal_state(critic_state, source["critic_state_dict"])
    if not prefix_exact or not critic_exact:
      raise RuntimeError("actor/critic transfer parity failed")
    candidate = actor_dim == CANDIDATE_ACTOR_DIM
    if candidate:
      new_columns_zero = bool(
        (actor_state["mlp.0.weight"][:, SOURCE_ACTOR_DIM:] == 0).all()
      )
      c_start, c_end = CONTACT_FORCE_CRITIC_SLICE
      normalizer_exact = all(
        torch.equal(
          actor_state[key][..., SOURCE_ACTOR_DIM:].detach().cpu(),
          source["critic_state_dict"][key][..., c_start:c_end].detach().cpu(),
        )
        for key in (
          "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"
        )
      )
    else:
      new_columns_zero = True
      normalizer_exact = True
    if not new_columns_zero or not normalizer_exact:
      raise RuntimeError("candidate expansion mapping failed")

    observation = wrapped.get_observations()
    actor_obs = observation["actor"]
    critic_obs = observation["critic"]
    if actor_obs.shape != (2048, actor_dim) or critic_obs.shape != (2048, CRITIC_DIM):
      raise RuntimeError(
        f"unexpected observation shapes: {actor_obs.shape}/{critic_obs.shape}"
      )
    if candidate:
      a_start, a_end = CONTACT_FORCE_ACTOR_SLICE
      c_start, c_end = CONTACT_FORCE_CRITIC_SLICE
      contact_force_identity = torch.equal(
        actor_obs[:, a_start:a_end], critic_obs[:, c_start:c_end]
      )
    else:
      contact_force_identity = True
    if not contact_force_identity:
      raise RuntimeError("actor/critic contact-force observations differ")

    policy = runner.get_inference_policy(device=device)
    with torch.inference_mode():
      initial_action = policy(observation)
      initial_critic = runner.alg.critic(observation)
    finite = bool(
      torch.isfinite(actor_obs).all()
      and torch.isfinite(critic_obs).all()
      and torch.isfinite(initial_action).all()
      and torch.isfinite(initial_critic).all()
    )
    resets = 0
    rewards_finite = True
    for _ in range(steps):
      with torch.inference_mode():
        action = policy(observation)
      observation, reward, dones, _ = wrapped.step(action)
      finite &= bool(
        torch.isfinite(action).all()
        and torch.isfinite(observation["actor"]).all()
        and torch.isfinite(observation["critic"]).all()
      )
      rewards_finite &= bool(torch.isfinite(reward).all())
      resets += int(dones.sum())
    if not finite or not rewards_finite:
      raise RuntimeError("non-finite no-learning rollout")
    return {
      "schema_version": 1,
      "arm": arm,
      "task_id": task_id,
      "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
      "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
      "source_manifest": str(manifest),
      "source_manifest_sha256": manifest_sha256,
      "num_envs": 2048,
      "steps": steps,
      "device": device,
      "seed": MATCHED_RNG_SEED,
      "dimensions": {"actor": actor_dim, "critic": CRITIC_DIM, "action": 12},
      "learn_called": False,
      "optimizer_step_called": False,
      "fresh_optimizer_state_empty": True,
      "transfer": transfer,
      "runtime_foot_order": runtime_order,
      "actor_prefix_exact": prefix_exact,
      "critic_exact": critic_exact,
      "candidate_new_columns_zero": new_columns_zero,
      "candidate_normalizer_exact": normalizer_exact,
      "actor_critic_contact_force_identity": contact_force_identity,
      "initial_actor_prefix_sha256": _tensor_sha(actor_obs[:, :SOURCE_ACTOR_DIM]),
      "initial_deterministic_action_sha256": _tensor_sha(initial_action),
      "finite": {"observations_actions": finite, "rewards": rewards_finite},
      "reset_count": resets,
      "formal_environment_state_exact": True,
      "technical_smoke_override": False,
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", choices=tuple(TASK_IDS), required=True)
  parser.add_argument("--source-manifest", type=Path, required=True)
  parser.add_argument("--source-manifest-sha256", required=True)
  parser.add_argument("--steps", type=int, default=8)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  payload = run_preflight(
    arm=args.arm, manifest=args.source_manifest,
    manifest_sha256=args.source_manifest_sha256, steps=args.steps,
    device=args.device,
  )
  output = args.output_file.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite preflight artifact: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
  output.write_text(encoded, encoding="utf-8")
  print(encoded, end="")


if __name__ == "__main__":
  main()
