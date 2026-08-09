"""Bounded Gaussian policy output for Go2 residual joint actions."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules.distribution import Distribution


class _AsymmetricActionTransform(nn.Module):
  """Map deterministic policy outputs into asymmetric per-joint bounds."""

  def __init__(
    self,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    mean_bound: float,
  ) -> None:
    super().__init__()
    self.register_buffer("action_low", action_low.detach().clone())
    self.register_buffer("action_high", action_high.detach().clone())
    self.mean_bound = float(mean_bound)

  def forward(self, raw_mean: torch.Tensor) -> torch.Tensor:
    latent = self.mean_bound * torch.tanh(raw_mean / self.mean_bound)
    unit = torch.tanh(latent)
    scale = torch.where(unit >= 0.0, self.action_high, -self.action_low)
    return unit * scale


class AsymmetricBoundedGaussianDistribution(Distribution):
  """Diagonal Gaussian transformed into fixed asymmetric action bounds.

  The Gaussian lives in latent space. Samples and deterministic inference are
  exposed in applied-action space so PPO storage, environment action history,
  rewards, metrics, and exported policies all share one action definition.
  """

  def __init__(
    self,
    output_dim: int,
    *,
    action_low: Sequence[float],
    action_high: Sequence[float],
    init_std: float = 1.0,
    std_type: str = "scalar",
    inverse_epsilon: float = 1.0e-6,
    mean_bound: float = 5.0,
  ) -> None:
    super().__init__(output_dim)
    low = torch.as_tensor(action_low, dtype=torch.float32)
    high = torch.as_tensor(action_high, dtype=torch.float32)
    if low.shape != (output_dim,) or high.shape != (output_dim,):
      raise ValueError("action bounds must have shape (output_dim,)")
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
      raise ValueError("action bounds must be finite")
    if not torch.all(low < 0.0) or not torch.all(high > 0.0):
      raise ValueError("each asymmetric action interval must contain zero")
    if not math.isfinite(inverse_epsilon) or not 0.0 < inverse_epsilon < 0.5:
      raise ValueError("inverse_epsilon must be finite and in (0, 0.5)")
    if not math.isfinite(init_std) or init_std <= 0.0:
      raise ValueError("init_std must be finite and positive")
    if not math.isfinite(mean_bound) or mean_bound <= 0.0:
      raise ValueError("mean_bound must be finite and positive")

    # Non-persistent bounds preserve strict state-dict compatibility with the
    # previous Gaussian actor, whose only distribution parameter is std_param.
    self.register_buffer("action_low", low, persistent=False)
    self.register_buffer("action_high", high, persistent=False)
    self.inverse_epsilon = float(inverse_epsilon)
    self.mean_bound = float(mean_bound)
    self.std_type = std_type
    if std_type == "scalar":
      self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
    elif std_type == "log":
      self.log_std_param = nn.Parameter(
        torch.log(init_std * torch.ones(output_dim))
      )
    else:
      raise ValueError(f"unknown std_type: {std_type}")

    self._distribution: Normal | None = None
    self._last_latent: torch.Tensor | None = None
    Normal.set_default_validate_args(False)

  @property
  def input_dim(self) -> int:
    return self.output_dim

  def update(self, mlp_output: torch.Tensor) -> None:
    if self.std_type == "scalar":
      std = self.std_param.expand_as(mlp_output)
    else:
      std = torch.exp(self.log_std_param).expand_as(mlp_output)
    torch._assert_async(torch.isfinite(mlp_output).all(), "raw mean contains NaN/Inf")
    torch._assert_async(
      torch.isfinite(std).all() & (std > 0.0).all(),
      "latent std must be finite and positive",
    )
    mean = self.mean_bound * torch.tanh(mlp_output / self.mean_bound)
    torch._assert_async(torch.isfinite(mean).all(), "latent mean contains NaN/Inf")
    self._distribution = Normal(mean, std)
    self._last_latent = None

  def transform(self, latent: torch.Tensor) -> torch.Tensor:
    """Transform latent actions into the applied normalized action space."""
    torch._assert_async(torch.isfinite(latent).all(), "latent action contains NaN/Inf")
    unit = torch.tanh(latent)
    scale = torch.where(unit >= 0.0, self.action_high, -self.action_low)
    return unit * scale

  def _inverse(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.where(action >= 0.0, self.action_high, -self.action_low)
    unit = action / scale
    unit = unit.clamp(
      min=-1.0 + self.inverse_epsilon,
      max=1.0 - self.inverse_epsilon,
    )
    latent = 0.5 * (torch.log1p(unit) - torch.log1p(-unit))
    return latent, scale

  def sample(self) -> torch.Tensor:
    if self._distribution is None:
      raise RuntimeError("distribution must be updated before sampling")
    self._last_latent = self._distribution.rsample()
    return self.transform(self._last_latent)

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    mean = self.mean_bound * torch.tanh(mlp_output / self.mean_bound)
    return self.transform(mean)

  def as_deterministic_output_module(self) -> nn.Module:
    return _AsymmetricActionTransform(
      self.action_low, self.action_high, self.mean_bound
    )

  @property
  def mean(self) -> torch.Tensor:
    if self._distribution is None:
      raise RuntimeError("distribution must be updated before reading mean")
    return self.transform(self._distribution.mean)

  @property
  def std(self) -> torch.Tensor:
    """Return latent Gaussian std; transformed std has no closed form."""
    if self._distribution is None:
      raise RuntimeError("distribution must be updated before reading std")
    return self._distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    """Return a pathwise Monte Carlo estimate of transformed entropy."""
    if self._distribution is None or self._last_latent is None:
      raise RuntimeError("distribution must be sampled before reading entropy")
    unit = torch.tanh(self._last_latent)
    scale = torch.where(unit >= 0.0, self.action_high, -self.action_low)
    log_tanh_jacobian = 2.0 * (
      math.log(2.0)
      - self._last_latent
      - torch.nn.functional.softplus(-2.0 * self._last_latent)
    )
    return self._distribution.entropy().sum(dim=-1) + (
      torch.log(scale) + log_tanh_jacobian
    ).sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    if self._distribution is None:
      raise RuntimeError("distribution must be updated before reading params")
    return self._distribution.mean, self._distribution.stddev

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    if self._distribution is None:
      raise RuntimeError("distribution must be updated before log_prob")
    torch._assert_async(torch.isfinite(outputs).all(), "applied action contains NaN/Inf")
    torch._assert_async(
      ((outputs >= self.action_low) & (outputs <= self.action_high)).all(),
      "applied action is outside the registered bounds",
    )
    latent, scale = self._inverse(outputs)
    unit = torch.tanh(latent)
    log_abs_det = torch.log(scale) + torch.log1p(-(unit * unit))
    return (self._distribution.log_prob(latent) - log_abs_det).sum(dim=-1)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    """Use exact latent KL; a shared bijection leaves KL unchanged."""
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    return torch.distributions.kl_divergence(
      Normal(old_mean, old_std), Normal(new_mean, new_std)
    ).sum(dim=-1)

  def init_mlp_weights(self, mlp: nn.Module) -> None:
    del mlp
