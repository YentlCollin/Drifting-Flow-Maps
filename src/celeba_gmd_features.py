"""Small learned feature space for Drifting on ``4 x 8 x 8`` CelebA latents.

The encoder is intentionally independent from the generator and the drifting
loss.  It can first be trained as a masked autoencoder, then loaded frozen while
still propagating gradients from its features to generated latents.

Every value returned by :meth:`CelebALatentMaskedAutoencoder.encode_features`
uses the layout ``[batch, tokens, dimensions]`` expected by
``src.official_drifting.summed_feature_drifting_loss``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


CELEBA_LATENT_SHAPE = (4, 8, 8)
CHECKPOINT_FORMAT = "celeba-latent-masked-autoencoder-v1"
ARCHITECTURE_NAME = "CelebALatentMaskedAutoencoder"
ARCHITECTURE_VERSION = 1


def _group_count(channels: int, maximum: int) -> int:
    """Choose the largest valid GroupNorm divisor up to ``maximum``."""

    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    """Pre-activation residual block without batch-dependent statistics."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _group_count(in_channels, norm_groups),
            in_channels,
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            _group_count(out_channels, norm_groups),
            out_channels,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        # The fixed scale keeps activation variance controlled as blocks are
        # stacked, without introducing a learned or batch-dependent statistic.
        return (self.skip(x) + hidden) * math.sqrt(0.5)


class _Stage(nn.Sequential):
    def __init__(self, channels: int, blocks: int, norm_groups: int) -> None:
        super().__init__(
            *(_ResidualBlock(channels, channels, norm_groups) for _ in range(blocks))
        )


class _Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__(
            nn.GroupNorm(_group_count(in_channels, norm_groups), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )


class _Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        blocks: int,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.fusion = _ResidualBlock(
            out_channels + skip_channels,
            out_channels,
            norm_groups,
        )
        self.refinement = _Stage(out_channels, max(blocks - 1, 0), norm_groups)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.projection(x)
        x = self.fusion(torch.cat((x, skip), dim=1))
        return self.refinement(x)


def _tokens(feature_map: Tensor) -> Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def _spatial_statistics(feature_map: Tensor) -> tuple[Tensor, Tensor]:
    """Return stable population mean/std as one token per sample.

    Accumulating in float32 avoids half-precision cancellation.  The result is
    converted back to the activation dtype so mixed-precision callers keep a
    consistent feature dictionary.
    """

    values = feature_map.float()
    mean = values.mean(dim=(-2, -1))
    centered = values - mean[:, :, None, None]
    variance = centered.square().mean(dim=(-2, -1))
    std = variance.clamp_min(1e-8).sqrt()
    dtype = feature_map.dtype
    return mean.to(dtype=dtype).unsqueeze(1), std.to(dtype=dtype).unsqueeze(1)


def _patch_statistics(feature_map: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Return non-overlapping local mean/std tokens, as in the release."""

    height, width = feature_map.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch size must divide both feature-map dimensions")
    values = feature_map.float().unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    # [B,C,H/P,W/P,P,P] -> statistics over the two within-patch axes.
    mean = values.mean(dim=(-2, -1))
    centered = values - mean[..., None, None]
    std = centered.square().mean(dim=(-2, -1)).clamp_min(1e-8).sqrt()
    dtype = feature_map.dtype
    return _tokens(mean.to(dtype=dtype)), _tokens(std.to(dtype=dtype))


class CelebALatentMaskedAutoencoder(nn.Module):
    """ResNet-GN masked autoencoder for scaled SD-VAE CelebA latents.

    ``mask`` has shape ``[B, 1, 8, 8]`` (or ``[B, 8, 8]``) and uses one for a
    hidden spatial position.  Hidden values are forcibly zeroed before the
    encoder, even if ``masked_x`` still contains them, and the mask itself is
    supplied as a fifth input channel.  Reconstruction loss and mask sampling
    deliberately stay outside this module so a trainer can choose its own
    masking ratio and weighting.
    """

    def __init__(
        self,
        base_channels: int = 32,
        *,
        blocks_per_stage: int = 2,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int):
            raise TypeError("base_channels must be an integer")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        if isinstance(blocks_per_stage, bool) or not isinstance(blocks_per_stage, int):
            raise TypeError("blocks_per_stage must be an integer")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1")
        if isinstance(norm_groups, bool) or not isinstance(norm_groups, int):
            raise TypeError("norm_groups must be an integer")
        if norm_groups < 1:
            raise ValueError("norm_groups must be positive")

        self.base_channels = base_channels
        self.blocks_per_stage = blocks_per_stage
        self.norm_groups = norm_groups
        channels = (base_channels, 2 * base_channels, 4 * base_channels, 4 * base_channels)
        self.stage_channels = channels

        # Four latent channels plus a spatial missingness indicator.
        self.input_projection = nn.Conv2d(5, channels[0], kernel_size=3, padding=1)
        self.stage8 = _Stage(channels[0], blocks_per_stage, norm_groups)
        self.down8_to_4 = _Downsample(channels[0], channels[1], norm_groups)
        self.stage4 = _Stage(channels[1], blocks_per_stage, norm_groups)
        self.down4_to_2 = _Downsample(channels[1], channels[2], norm_groups)
        self.stage2 = _Stage(channels[2], blocks_per_stage, norm_groups)
        self.down2_to_1 = _Downsample(channels[2], channels[3], norm_groups)
        self.stage1 = _Stage(channels[3], blocks_per_stage, norm_groups)

        self.up1_to_2 = _Upsample(
            channels[3],
            channels[2],
            channels[2],
            blocks=blocks_per_stage,
            norm_groups=norm_groups,
        )
        self.up2_to_4 = _Upsample(
            channels[2],
            channels[1],
            channels[1],
            blocks=blocks_per_stage,
            norm_groups=norm_groups,
        )
        self.up4_to_8 = _Upsample(
            channels[1],
            channels[0],
            channels[0],
            blocks=blocks_per_stage,
            norm_groups=norm_groups,
        )
        self.output_norm = nn.GroupNorm(
            _group_count(channels[0], norm_groups),
            channels[0],
        )
        self.output_projection = nn.Conv2d(
            channels[0],
            CELEBA_LATENT_SHAPE[0],
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def _validate_latents(x: Tensor) -> None:
        if x.ndim != 4 or tuple(x.shape[1:]) != CELEBA_LATENT_SHAPE:
            raise ValueError("CelebA latents must have shape [B, 4, 8, 8]")
        if not x.is_floating_point():
            raise TypeError("CelebA latents must be floating point")

    @staticmethod
    def _prepare_mask(mask: Tensor, like: Tensor) -> Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        expected = (like.shape[0], 1, 8, 8)
        if tuple(mask.shape) != expected:
            raise ValueError(f"mask must have shape {expected} or {(like.shape[0], 8, 8)}")
        if mask.dtype == torch.bool:
            return mask.to(device=like.device, dtype=like.dtype)
        if mask.is_floating_point() or mask.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            if not bool(torch.isfinite(mask).all()):
                raise ValueError("mask values must be finite")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask values must be binary (zero or one)")
            return mask.to(device=like.device, dtype=like.dtype)
        raise TypeError("mask must be boolean or numeric")

    def _encode_maps(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.input_projection(torch.cat((x, mask), dim=1))
        feature8 = self.stage8(hidden)
        feature4 = self.stage4(self.down8_to_4(feature8))
        feature2 = self.stage2(self.down4_to_2(feature4))
        feature1 = self.stage1(self.down2_to_1(feature2))
        return feature8, feature4, feature2, feature1

    def forward(self, masked_x: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct a full latent tensor from spatially masked input."""

        self._validate_latents(masked_x)
        mask = self._prepare_mask(mask, masked_x)
        visible_x = masked_x * (1.0 - mask)
        feature8, feature4, feature2, feature1 = self._encode_maps(visible_x, mask)
        hidden = self.up1_to_2(feature1, feature2)
        hidden = self.up2_to_4(hidden, feature4)
        hidden = self.up4_to_8(hidden, feature8)
        return self.output_projection(F.silu(self.output_norm(hidden)))

    def encode_features(self, x: Tensor) -> dict[str, Tensor]:
        """Extract raw and learned multi-scale features.

        No ``torch.no_grad`` context is used here.  A frozen encoder therefore
        has no parameter gradients but remains differentiable with respect to
        ``x``, which is required when it scores generated samples.
        """

        self._validate_latents(x)
        no_mask = x.new_zeros((x.shape[0], 1, 8, 8))
        feature8, feature4, feature2, feature1 = self._encode_maps(x, no_mask)
        maps = {
            "stage8": feature8,
            "stage4": feature4,
            "stage2": feature2,
            "stage1": feature1,
        }
        result = {
            # Added unconditionally by the release's activation wrapper.
            "global": x.flatten(1).unsqueeze(1),
            # Official ``norm_x``: per-channel RMS across spatial positions.
            "norm_x": x.float()
            .square()
            .mean(dim=(-2, -1))
            .add(1e-6)
            .sqrt()
            .to(dtype=x.dtype)
            .unsqueeze(1),
            "stage8": _tokens(feature8),
            "stage4": _tokens(feature4),
            "stage2": _tokens(feature2),
            "stage1": _tokens(feature1),
        }
        for name, feature_map in maps.items():
            spatial_count = math.prod(feature_map.shape[-2:])
            if spatial_count > 1:
                mean, std = _spatial_statistics(feature_map)
                result[f"{name}_mean"] = mean
                result[f"{name}_std"] = std
            for patch_size in (2, 4):
                divides = all(
                    size % patch_size == 0 for size in feature_map.shape[-2:]
                )
                patch_count = (
                    math.prod(size // patch_size for size in feature_map.shape[-2:])
                    if divides
                    else 0
                )
                if divides and patch_count > 1:
                    patch_mean, patch_std = _patch_statistics(feature_map, patch_size)
                    result[f"{name}_mean_{patch_size}"] = patch_mean
                    result[f"{name}_std_{patch_size}"] = patch_std
        return result

    def architecture_config(self) -> dict[str, Any]:
        """Return every constructor choice needed to reproduce this network."""

        return {
            "name": ARCHITECTURE_NAME,
            "version": ARCHITECTURE_VERSION,
            "base_channels": self.base_channels,
            "blocks_per_stage": self.blocks_per_stage,
            "norm_groups": self.norm_groups,
        }

    def checkpoint_dict(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a self-describing checkpoint payload."""

        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "architecture": self.architecture_config(),
            "state_dict": self.state_dict(),
        }
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return payload

    def save_checkpoint(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save weights together with the architecture configuration."""

        torch.save(self.checkpoint_dict(metadata=metadata), Path(checkpoint))


def load_frozen_feature_encoder(
    checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    device: torch.device | str,
) -> CelebALatentMaskedAutoencoder:
    """Load a self-describing checkpoint as an eval-mode frozen encoder.

    Freezing parameters does *not* disable autograd through the module, so
    gradients can still flow to generated latent inputs.
    """

    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        payload = torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format must be {CHECKPOINT_FORMAT!r}")

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint is missing its architecture configuration")
    if architecture.get("name") != ARCHITECTURE_NAME:
        raise ValueError(f"checkpoint architecture must be {ARCHITECTURE_NAME!r}")
    if architecture.get("version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported architecture version {architecture.get('version')!r}"
        )

    required = ("base_channels", "blocks_per_stage", "norm_groups")
    missing = [key for key in required if key not in architecture]
    if missing:
        raise ValueError(f"checkpoint architecture is missing: {', '.join(missing)}")
    model = CelebALatentMaskedAutoencoder(
        base_channels=architecture["base_channels"],
        blocks_per_stage=architecture["blocks_per_stage"],
        norm_groups=architecture["norm_groups"],
    )

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model


__all__ = [
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_VERSION",
    "CELEBA_LATENT_SHAPE",
    "CHECKPOINT_FORMAT",
    "CelebALatentMaskedAutoencoder",
    "load_frozen_feature_encoder",
]
