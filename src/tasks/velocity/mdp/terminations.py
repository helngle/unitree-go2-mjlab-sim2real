from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def sustained_illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 35.0,
  min_substeps: int = 2,
  min_orientation_angle: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate after non-foot contact persists across simulation substeps."""
  if min_substeps < 1:
    raise ValueError("min_substeps must be at least one")

  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit_per_substep = (force_mag > force_threshold).any(dim=1)  # [B, H]
    required = min(min_substeps, hit_per_substep.shape[-1])
    consecutive_hits = hit_per_substep.unfold(-1, required, 1).all(dim=-1)
    illegal = consecutive_hits.any(dim=-1)
  else:
    assert data.force is not None
    force_mag = torch.norm(data.force, dim=-1)
    illegal = (force_mag > force_threshold).any(dim=-1)

  if min_orientation_angle is not None:
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs()
    illegal &= angle > min_orientation_angle
  return illegal
