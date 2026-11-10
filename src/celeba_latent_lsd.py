"""CelebA-latent backbone for flow-map self-distillation.

This module is intentionally separate from :mod:`src.celeba_latent`.  Running
CelebA jobs snapshot that historical module, whose transition network uses the
normalized endpoint coordinate ``(t-s)/(1-s)``.  LSD differentiates with
respect to ``t`` and instead requires the released, nonsingular ``(s, t-s)``
conditioning convention.

The surrounding flow-map wrapper in :mod:`src.flow_map_self_distillation`
constructs ``X(x,s,t) = x + (t-s) phi(x,s,t)``.  Consequently ``endpoint`` in
this backbone returns *phi itself*; it never returns a mapped endpoint.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.celeba_latent import (
    CELEBA_LATENT_DIM,
    CELEBA_LATENT_SHAPE,
    flatten_latents,
    unflatten_latents,
)
from src.flow_map_self_distillation import (
    MagnitudePreservingTimeLogVariance,
    ResidualTwoTimeFlowMap,
    mixed_lsd_loss,
)


TIME_CONDITIONING = "start_delta_v1"


def _time_vector(value: Tensor | float, like: Tensor, *, name: str) -> Tensor:
    result = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if result.ndim == 0 or result.numel() == 1:
        return result.reshape(1).expand(like.shape[0])
    if result.numel() != like.shape[0]:
        raise ValueError(
            f"{name} must be scalar or contain {like.shape[0]} values, "
            f"got shape {tuple(result.shape)}"
        )
    return result.reshape(like.shape[0])


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if channels < 4 or channels % 4:
            raise ValueError("time_embedding_dim must be a positive multiple of four")
        half = channels // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.channels = channels
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: Tensor) -> Tensor:
        angles = time.float()[:, None] * self.frequencies.float()[None, :]
        result = math.sqrt(2.0) * torch.cat((angles.cos(), angles.sin()), dim=1)
        if result.shape[1] < self.channels:
            result = F.pad(result, (0, self.channels - result.shape[1]))
        return result.to(dtype=time.dtype)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class CelebALatentLSDBackbone(nn.Module):
    """Compact 4x8x8 U-Net predicting the absolute LSD velocity ``phi``.

    The temporal inputs are ``s`` and ``delta=t-s`` exactly as in the released
    EDM2 flow-map implementation.  ``sigma_data`` also implements its
    preconditioning, namely ``x_in=x/sigma_data`` and
    ``phi=sigma_data*UNet(x_in, s, delta)``.  The convolutional architecture is
    deliberately the repository's compact latent U-Net, not the paper's
    124M-parameter pixel-space EDM2 model.
    """

    endpoint_parameterization = "absolute"
    time_conditioning = TIME_CONDITIONING

    def __init__(
        self,
        *,
        base_channels: int = 256,
        time_embedding_dim: int = 32,
        condition_dim: int | None = None,
        sigma_data: float = 0.5,
        zero_init_velocity: bool = True,
    ) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least four")
        if not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError("sigma_data must be finite and strictly positive")
        condition_dim = 4 * base_channels if condition_dim is None else condition_dim
        if condition_dim < 4:
            raise ValueError("condition_dim must be at least four")

        self.base_channels = int(base_channels)
        self.time_embedding_dim = int(time_embedding_dim)
        self.condition_dim = int(condition_dim)
        self.sigma_data = float(sigma_data)
        self.zero_init_velocity = bool(zero_init_velocity)

        self.time_embedding = _SinusoidalEmbedding(time_embedding_dim)
        self.transition_mlp = nn.Sequential(
            nn.Linear(2 * time_embedding_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        high_channels = 2 * base_channels
        self.input_conv = nn.Conv2d(
            CELEBA_LATENT_SHAPE[0], base_channels, kernel_size=3, padding=1
        )
        self.down_block = _ConditionedResidualBlock(
            base_channels, base_channels, condition_dim
        )
        self.downsample = nn.Conv2d(
            base_channels, high_channels, kernel_size=4, stride=2, padding=1
        )
        self.middle_block1 = _ConditionedResidualBlock(
            high_channels, high_channels, condition_dim
        )
        self.middle_block2 = _ConditionedResidualBlock(
            high_channels, high_channels, condition_dim
        )
        self.upsample = nn.ConvTranspose2d(
            high_channels, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.up_block = _ConditionedResidualBlock(
            2 * base_channels, base_channels, condition_dim
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv2d(
            base_channels, CELEBA_LATENT_SHAPE[0], kernel_size=3, padding=1
        )
        # The official EDM2 initializes a scalar ``out_gain`` to zero.  This
        # compact adaptation zeroes the final convolution instead: both make
        # phi the exact zero function at initialization, though their first
        # optimizer step is not parameter-for-parameter identical.
        if zero_init_velocity:
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "base_channels": self.base_channels,
            "time_embedding_dim": self.time_embedding_dim,
            "condition_dim": self.condition_dim,
            "sigma_data": self.sigma_data,
            "zero_init_velocity": self.zero_init_velocity,
            "endpoint_parameterization": self.endpoint_parameterization,
            "time_conditioning": self.time_conditioning,
            "latent_shape": list(CELEBA_LATENT_SHAPE),
            "latent_dim": CELEBA_LATENT_DIM,
        }

    @staticmethod
    def _spatial_input(x: Tensor) -> tuple[Tensor, bool]:
        if x.ndim == 2:
            if x.shape[1] != CELEBA_LATENT_DIM:
                raise ValueError(
                    f"flat latent width must be {CELEBA_LATENT_DIM}, got {x.shape[1]}"
                )
            return unflatten_latents(x), True
        if x.ndim != 4 or tuple(x.shape[1:]) != CELEBA_LATENT_SHAPE:
            raise ValueError(
                f"expected [B,{CELEBA_LATENT_DIM}] or [B,{','.join(map(str, CELEBA_LATENT_SHAPE))}]"
            )
        return x, False

    def transition_condition(
        self,
        start_time: Tensor | float,
        end_time: Tensor | float,
        like: Tensor,
    ) -> Tensor:
        start = _time_vector(start_time, like, name="start_time")
        end = _time_vector(end_time, like, name="end_time")
        delta = end - start
        return self.transition_mlp(
            torch.cat(
                (self.time_embedding(start), self.time_embedding(delta)), dim=1
            )
        )

    def _raw_network(self, spatial: Tensor, condition: Tensor) -> Tensor:
        first_skip = self.down_block(self.input_conv(spatial), condition)
        hidden = self.downsample(first_skip)
        hidden = self.middle_block1(hidden, condition)
        hidden = self.middle_block2(hidden, condition)
        hidden = self.upsample(hidden)
        if hidden.shape[-2:] != first_skip.shape[-2:]:
            hidden = F.interpolate(hidden, size=first_skip.shape[-2:], mode="nearest")
        hidden = self.up_block(torch.cat((hidden, first_skip), dim=1), condition)
        return self.output_conv(F.silu(self.output_norm(hidden)))

    def endpoint(
        self,
        x: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor:
        spatial, was_flat = self._spatial_input(x)
        condition = self.transition_condition(start_time, end_time, spatial)
        velocity = self.sigma_data * self._raw_network(
            spatial / self.sigma_data, condition
        )
        return flatten_latents(velocity) if was_flat else velocity

    def forward(
        self,
        x: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor:
        """Return ``phi``; the residual flow-map wrapper constructs ``X``."""

        return self.endpoint(x, start_time, end_time)


class CelebALatentLSDTrainingModel(nn.Module):
    """One-forward DDP boundary around the complete mixed LSD objective.

    The objective evaluates the flow map several times before one backward
    (velocity JVP, stopped current-parameter target, and log-variance head).
    Keeping those evaluations inside one outer DDP ``forward`` avoids arming
    the reducer multiple times.  The stopped target is passed as the unwrapped
    ``flow_map`` object and is evaluated under ``no_grad`` by the core loss.
    """

    def __init__(self, flow_map: ResidualTwoTimeFlowMap) -> None:
        super().__init__()
        self.flow_map = flow_map

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        teacher_flow_map: nn.Module,
    ) -> tuple[Tensor, dict[str, Tensor | float | int]]:
        return mixed_lsd_loss(
            self.flow_map,
            x0,
            x1,
            diagonal_fraction=0.75,
            teacher_flow_map=teacher_flow_map,
        )


@torch.no_grad()
def project_log_variance_weight_(module: nn.Module, eps: float = 1e-4) -> int:
    """Apply the release's EDM2 unit-magnitude projection to the logvar head.

    The compact latent U-Net uses ordinary convolutions, so only the one
    genuinely magnitude-preserving weight in this adaptation is projected.
    A return value of one makes omissions detectable in the trainer/tests.
    """

    heads = [
        child
        for child in module.modules()
        if isinstance(child, MagnitudePreservingTimeLogVariance)
    ]
    if len(heads) != 1:
        raise ValueError(
            "expected exactly one MagnitudePreservingTimeLogVariance head, "
            f"found {len(heads)}"
        )
    weight = heads[0].weight
    dimensions = tuple(range(1, weight.ndim))
    norm = weight.float().square().sum(dim=dimensions, keepdim=True).sqrt()
    element_count = math.prod(weight.shape[1:])
    denominator = eps + norm / math.sqrt(element_count)
    weight.div_(denominator.to(dtype=weight.dtype))
    return 1


__all__ = [
    "CelebALatentLSDBackbone",
    "CelebALatentLSDTrainingModel",
    "TIME_CONDITIONING",
    "project_log_variance_weight_",
]
