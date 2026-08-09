"""Teacher-rollout behavior cloning for the Go2 student initialization."""

from __future__ import annotations

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class TeacherRolloutDistillation(Distillation):
  """Collect retained-terrain rollouts with deterministic teacher actions."""

  def act(self, obs: TensorDict) -> torch.Tensor:
    with torch.inference_mode():
      teacher_actions = self.teacher(obs).detach()
    self.transition.actions = teacher_actions
    self.transition.privileged_actions = teacher_actions
    self.transition.observations = obs
    return teacher_actions

  def update(self) -> dict[str, float]:
    """Apply one mean Huber update over every stored rollout element."""
    self.num_updates += 1
    self.student.reset(hidden_state=self.last_hidden_states[0])
    self.teacher.reset(hidden_state=self.last_hidden_states[1])
    self.student.detach_hidden_state()

    total_loss = torch.zeros((), device=self.device)
    total_elements = 0
    batch_count = 0
    for batch in self.storage.generator():
      actions = self.student(batch.observations)
      total_loss = total_loss + self.loss_fn(
        actions, batch.privileged_actions, reduction="sum"
      )
      total_elements += actions.numel()
      batch_count += 1
      self.student.reset(batch.dones.view(-1))
      self.teacher.reset(batch.dones.view(-1))

    if batch_count != self.storage.num_transitions_per_env or total_elements == 0:
      raise RuntimeError("distillation update did not consume the complete rollout")
    behavior_loss = total_loss / total_elements
    self.optimizer.zero_grad()
    behavior_loss.backward()
    if self.is_multi_gpu:
      self.reduce_parameters()
    if self.max_grad_norm:
      nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.storage.clear()
    self.last_hidden_states = (
      self.student.get_hidden_state(),
      self.teacher.get_hidden_state(),
    )
    self.student.detach_hidden_state()
    self.last_update_batch_count = batch_count
    self.last_optimizer_step_count = 1
    return {"behavior": float(behavior_loss.detach())}


class BoundedTeacherRolloutDistillation(TeacherRolloutDistillation):
  """Distill old Gaussian teacher actions into the bounded student contract."""

  def act(self, obs: TensorDict) -> torch.Tensor:
    with torch.inference_mode():
      teacher_raw_actions = self.teacher(obs).detach()
      distribution = self.student.distribution
      if distribution is None or not hasattr(distribution, "transform"):
        raise TypeError("bounded distillation student requires an action transform")
      teacher_actions = distribution.transform(teacher_raw_actions).detach()
    self.last_teacher_raw_actions = teacher_raw_actions
    self.transition.actions = teacher_actions
    self.transition.privileged_actions = teacher_actions
    self.transition.observations = obs
    return teacher_actions
