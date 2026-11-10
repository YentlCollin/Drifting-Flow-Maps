"""Release-faithful latent MAE feature encoder adapted to ``4 x 8 x 8`` CelebA.

This module is deliberately versioned separately from
``src.celeba_gmd_features``.  It ports the latent ResNet-MAE architecture and
activation extraction in ``models/mae_model.py`` from the official Drifting
Models JAX release (commit
``accd0cf09c33b70892d33941d2a287ca86cb92e1``).  The reconstruction/feature
path changes its input spatial size from ``32 x 32`` to ``8 x 8``; the unused
ImageNet classification head is omitted for unlabeled CelebA.

Tensors use PyTorch's ``NCHW`` layout.  Feature dictionaries use
``[batch, tokens, dimensions]``, ready for
``src.official_drifting.summed_feature_drifting_loss``.  ``official_70`` is
the default release-faithful extraction profile.  ``small_grid_52`` keeps the
same maps and taps but removes statistics that become exact duplicates or
constants only because the CelebA latent hierarchy reaches ``2 x 2`` and
``1 x 1``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


CELEBA_LATENT_SHAPE = (4, 8, 8)
CHECKPOINT_FORMAT = "celeba-latent-masked-autoencoder-v2"
ARCHITECTURE_NAME = "CelebALatentMaskedAutoencoderV2"
ARCHITECTURE_VERSION = 2
DEFAULT_LAYERS = (3, 4, 6, 3)
DEFAULT_PATCH_SIZE = 2
DEFAULT_FEATURE_PATCH_SIZES = (2, 4)
DEFAULT_EVERY_K_BLOCK = 2
FEATURE_PROFILE_OFFICIAL_70 = "official_70"
FEATURE_PROFILE_SMALL_GRID_52 = "small_grid_52"
FEATURE_PROFILES = (
    FEATURE_PROFILE_OFFICIAL_70,
    FEATURE_PROFILE_SMALL_GRID_52,
)
DEFAULT_FEATURE_PROFILE = FEATURE_PROFILE_OFFICIAL_70
MASK_SAMPLING_OFFICIAL_BERNOULLI = "official_bernoulli"
MASK_SAMPLING_EXACT_COUNT = "exact_count"
MASK_SAMPLING_MODES = (
    MASK_SAMPLING_OFFICIAL_BERNOULLI,
    MASK_SAMPLING_EXACT_COUNT,
)
DEFAULT_MASK_SAMPLING = MASK_SAMPLING_OFFICIAL_BERNOULLI
DEFAULT_EXACT_MASKED_PATCHES = 8
SAFE_STD_EPS = 1e-6
_TRUNCATED_NORMAL_STDDEV_FACTOR = 0.8796256610342398


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """Choose the largest GroupNorm divisor no greater than ``max_groups``."""

    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups:
        groups -= 1
    return max(groups, 1)


def _group_norm(channels: int, max_groups: int) -> nn.GroupNorm:
    # Flax ``linen.GroupNorm`` defaults to epsilon=1e-6, whereas PyTorch's
    # default is 1e-5.  Keep the release value explicit in this port.
    return nn.GroupNorm(
        _choose_gn_groups(channels, max_groups),
        channels,
        eps=1e-6,
    )


def _init_like_flax(module: nn.Module) -> None:
    """Match Flax Conv's default LeCun truncated-normal initialization."""

    if not isinstance(module, nn.Conv2d):
        return
    fan_in = (module.in_channels // module.groups) * math.prod(module.kernel_size)
    std = math.sqrt(1.0 / fan_in) / _TRUNCATED_NORMAL_STDDEV_FACTOR
    nn.init.trunc_normal_(
        module.weight,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _validate_latents(x: Tensor) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != CELEBA_LATENT_SHAPE:
        raise ValueError("CelebA latents must have shape [B, 4, 8, 8]")
    if not x.is_floating_point():
        raise TypeError("CelebA latents must be floating point")


def _binary_mask(mask: Tensor, like: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    expected = (like.shape[0], 1, like.shape[-2], like.shape[-1])
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected} or {expected[:1] + expected[2:]}")
    if mask.dtype != torch.bool and not (
        mask.is_floating_point()
        or mask.dtype
        in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    ):
        raise TypeError("mask must be boolean or numeric")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("mask values must be finite")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask values must be binary (zero or one)")
    return mask.to(device=like.device, dtype=like.dtype)


def make_patch_mask(
    latents: Tensor,
    mask_ratio: float | Tensor = 0.5,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the official Bernoulli coarse-patch mask in ``NCHW`` layout.

    ``mask_ratio`` may be a scalar or one value per sample.  As in the release,
    the requested ratio is a Bernoulli probability rather than an exact count.
    A value of one means hidden.  One coarse decision is repeated over every
    ``patch_size x patch_size`` latent patch.
    """

    _validate_latents(latents)
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    height, width = latents.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide both latent spatial dimensions")

    ratio = torch.as_tensor(mask_ratio, device=latents.device, dtype=torch.float32)
    if ratio.ndim == 0:
        ratio = ratio.expand(latents.shape[0])
    elif ratio.numel() == latents.shape[0]:
        ratio = ratio.reshape(latents.shape[0])
    else:
        raise ValueError("mask_ratio must be scalar or contain one value per sample")
    if not bool(torch.isfinite(ratio).all()) or not bool(((ratio >= 0) & (ratio <= 1)).all()):
        raise ValueError("mask_ratio values must be finite and lie in [0, 1]")

    noise = torch.rand(
        (latents.shape[0], 1, height // patch_size, width // patch_size),
        device=latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    coarse = noise < ratio[:, None, None, None]
    mask = coarse.repeat_interleave(patch_size, dim=-2).repeat_interleave(
        patch_size, dim=-1
    )
    return mask.to(dtype=latents.dtype)


def make_exact_patch_mask(
    latents: Tensor,
    masked_patches: int | Tensor = DEFAULT_EXACT_MASKED_PATCHES,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniformly select an exact number of coarse patches in each sample.

    This is the explicit small-grid alternative to :func:`make_patch_mask`.
    The official function samples independent Bernoulli decisions; this one
    samples a random ordering without replacement and hides exactly
    ``masked_patches`` entries of the coarse grid.  With ``8 x 8`` latents and
    ``patch_size=2``, the default therefore hides exactly 8 of 16 patches.
    """

    _validate_latents(latents)
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    height, width = latents.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide both latent spatial dimensions")

    total_patches = (height // patch_size) * (width // patch_size)
    if isinstance(masked_patches, bool):
        raise TypeError("masked_patches must be an integer or integer tensor")
    counts = torch.as_tensor(masked_patches, device=latents.device)
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if counts.dtype not in integer_dtypes:
        raise TypeError("masked_patches must be an integer or integer tensor")
    if counts.ndim == 0:
        counts = counts.expand(latents.shape[0])
    elif counts.numel() == latents.shape[0]:
        counts = counts.reshape(latents.shape[0])
    else:
        raise ValueError("masked_patches must be scalar or contain one value per sample")
    counts = counts.to(dtype=torch.int64)
    if not bool(((counts >= 0) & (counts <= total_patches)).all()):
        raise ValueError(f"masked_patches values must lie in [0, {total_patches}]")

    # Sorting i.i.d. continuous scores gives a uniform random permutation.
    # Convert that permutation to ranks so a different exact count may be used
    # for each sample without a Python loop.
    scores = torch.rand(
        (latents.shape[0], total_patches),
        device=latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    ordering = scores.argsort(dim=1)
    ranks = torch.empty_like(ordering)
    ranks.scatter_(
        1,
        ordering,
        torch.arange(total_patches, device=latents.device).expand_as(ordering),
    )
    coarse = (ranks < counts[:, None]).reshape(
        latents.shape[0],
        1,
        height // patch_size,
        width // patch_size,
    )
    mask = coarse.repeat_interleave(patch_size, dim=-2).repeat_interleave(
        patch_size, dim=-1
    )
    return mask.to(dtype=latents.dtype)


def masked_reconstruction_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return the release's masked-only reconstruction loss, one value per sample.

    The denominator counts hidden spatial positions, not channels.  Therefore
    the channel errors are summed exactly as in the official JAX implementation.
    """

    _validate_latents(prediction)
    _validate_latents(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prepared_mask = _binary_mask(mask, target).float()
    squared = (prediction.float() - target.float()).square()
    numerator = (squared * prepared_mask).sum(dim=(1, 2, 3))
    denominator = prepared_mask.sum(dim=(1, 2, 3)) + 1e-8
    return numerator / denominator


def safe_std(
    x: Tensor,
    dim: int | tuple[int, ...],
    *,
    keepdim: bool = False,
    eps: float = SAFE_STD_EPS,
) -> Tensor:
    """Population standard deviation used by the release, accumulated in FP32."""

    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    values = x.float()
    mean = values.mean(dim=dim, keepdim=True)
    variance = (values - mean).square().mean(dim=dim, keepdim=keepdim)
    return (variance.clamp_min(0.0) + eps).sqrt()


class _BasicBlock(nn.Module):
    """Official post-activation ResNet basic block in ``NCHW`` layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        gn_max_groups: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = _group_norm(out_channels, gn_max_groups)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn2 = _group_norm(out_channels, gn_max_groups)
        self.projection = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _group_norm(out_channels, gn_max_groups),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, *, training: bool | None = None) -> Tensor:
        residual = self.projection(x)
        hidden = self.relu(self.gn1(self.conv1(x)))
        hidden = F.dropout(
            hidden,
            p=self.dropout.p,
            training=self.training if training is None else training,
        )
        hidden = self.gn2(self.conv2(hidden))
        return self.relu(residual + hidden)


class _EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
        gn_max_groups: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        modules = [
            _BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                gn_max_groups=gn_max_groups,
                dropout_prob=dropout_prob,
            )
        ]
        modules.extend(
            _BasicBlock(
                out_channels,
                out_channels,
                stride=1,
                gn_max_groups=gn_max_groups,
                dropout_prob=dropout_prob,
            )
            for _ in range(1, blocks)
        )
        self.blocks = nn.ModuleList(modules)

    def forward(
        self,
        x: Tensor,
        *,
        training: bool | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs: list[Tensor] = []
        for block in self.blocks:
            x = block(x, training=training)
            outputs.append(x)
        return x, outputs


class _ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int,
        layers: tuple[int, int, int, int],
        dropout_prob: float,
        gn_max_groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, base_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = _group_norm(base_channels, gn_max_groups)
        self.relu = nn.ReLU()

        stage_channels = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
        )
        stages: list[_EncoderStage] = []
        norms: list[nn.GroupNorm] = []
        in_channels = base_channels
        for stage_index, (out_channels, blocks) in enumerate(zip(stage_channels, layers)):
            stages.append(
                _EncoderStage(
                    in_channels,
                    out_channels,
                    blocks=blocks,
                    stride=1 if stage_index == 0 else 2,
                    gn_max_groups=gn_max_groups,
                    dropout_prob=dropout_prob,
                )
            )
            norms.append(_group_norm(out_channels, gn_max_groups))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(norms)
        self.stage_channels = stage_channels

    def forward(
        self,
        x: Tensor,
        *,
        return_block_outputs: bool = False,
        training: bool | None = None,
    ) -> (
        dict[str, Tensor]
        | tuple[dict[str, Tensor], dict[str, list[Tensor]]]
    ):
        features: dict[str, Tensor] = {}
        block_outputs: dict[str, list[Tensor]] = {}

        x = self.relu(self.gn1(self.conv1(x)))
        features["conv1"] = x
        for stage_index, (stage, stage_norm) in enumerate(
            zip(self.stages, self.stage_norms), start=1
        ):
            x, outputs = stage(x, training=training)
            name = f"layer{stage_index}"
            block_outputs[name] = outputs
            x = stage_norm(x)
            features[name] = x
        if return_block_outputs:
            return features, block_outputs
        return features


class _ConvGNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        gn_max_groups: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn = _group_norm(out_channels, gn_max_groups)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        gn_max_groups: int,
    ) -> None:
        super().__init__()
        concatenated = in_channels + skip_channels
        self.concat_norm = _group_norm(concatenated, gn_max_groups)
        self.projection = _ConvGNReLU(
            concatenated,
            out_channels,
            gn_max_groups=gn_max_groups,
        )
        self.refinement = _ConvGNReLU(
            out_channels,
            out_channels,
            gn_max_groups=gn_max_groups,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.concat_norm(torch.cat((x, skip), dim=1))
        x = self.projection(x)
        return self.refinement(x)


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, *, gn_max_groups: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = 2 * base_channels
        c4 = 4 * base_channels
        c5 = 8 * base_channels
        self.bridge = _ConvGNReLU(c5, c5, gn_max_groups=gn_max_groups)
        self.up43 = _UpBlock(c5, c4, c4, gn_max_groups=gn_max_groups)
        self.up32 = _UpBlock(c4, c3, c3, gn_max_groups=gn_max_groups)
        self.up21 = _UpBlock(c3, c2, c2, gn_max_groups=gn_max_groups)
        self.up10 = _UpBlock(c2, c1, c1, gn_max_groups=gn_max_groups)
        self.head = nn.Conv2d(c1, CELEBA_LATENT_SHAPE[0], kernel_size=1)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        x = self.bridge(features["layer4"])
        x = self.up43(x, features["layer3"])
        x = self.up32(x, features["layer2"])
        x = self.up21(x, features["layer1"])
        x = self.up10(x, features["conv1"])
        return self.head(x)


def _tokens(feature_map: Tensor) -> Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def _patch_values(feature_map: Tensor, patch_size: int) -> Tensor:
    height, width = feature_map.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch size must divide both feature-map dimensions")
    return feature_map.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )


def _process_feature(
    output: dict[str, Tensor],
    name: str,
    feature_map: Tensor,
    *,
    patch_sizes: Sequence[int],
    feature_profile: str,
) -> None:
    output[name] = _tokens(feature_map)
    height, width = feature_map.shape[-2:]
    keep_degenerate = feature_profile == FEATURE_PROFILE_OFFICIAL_70
    if keep_degenerate or height * width > 1:
        output[f"{name}_mean"] = feature_map.mean(dim=(-2, -1)).unsqueeze(1)
        output[f"{name}_std"] = safe_std(feature_map, (-2, -1)).unsqueeze(1)
    for patch_size in patch_sizes:
        if height % patch_size or width % patch_size:
            continue
        patch_count = (height // patch_size) * (width // patch_size)
        # On the 8x8 hierarchy, a single output patch exactly duplicates the
        # global mean/std pair.  Patch size one would likewise duplicate raw
        # tokens and add a constant std.  The official profile intentionally
        # retains both behaviours; small_grid_52 removes only these cases.
        if not keep_degenerate and (patch_size == 1 or patch_count <= 1):
            continue
        patches = _patch_values(feature_map, patch_size)
        patch_mean = patches.mean(dim=(-2, -1))
        patch_std = safe_std(patches, (-2, -1))
        output[f"{name}_mean_{patch_size}"] = _tokens(patch_mean)
        output[f"{name}_std_{patch_size}"] = _tokens(patch_std)


class CelebALatentMaskedAutoencoderV2(nn.Module):
    """PyTorch port of the released latent ResNet-MAE at CelebA resolution."""

    def __init__(
        self,
        base_channels: int = 64,
        *,
        layers: Sequence[int] = DEFAULT_LAYERS,
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
        patch_size: int = DEFAULT_PATCH_SIZE,
        feature_patch_sizes: Sequence[int] = DEFAULT_FEATURE_PATCH_SIZES,
        every_k_block: int = DEFAULT_EVERY_K_BLOCK,
        feature_profile: str = DEFAULT_FEATURE_PROFILE,
        mask_sampling: str = DEFAULT_MASK_SAMPLING,
        exact_masked_patches: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int):
            raise TypeError("base_channels must be an integer")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        layers_tuple = tuple(layers)
        if len(layers_tuple) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in layers_tuple
        ):
            raise ValueError("layers must contain four positive integers")
        if not math.isfinite(float(dropout_prob)) or not 0 <= dropout_prob < 1:
            raise ValueError("dropout_prob must lie in [0, 1)")
        if isinstance(gn_max_groups, bool) or not isinstance(gn_max_groups, int) or gn_max_groups < 1:
            raise ValueError("gn_max_groups must be a positive integer")
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        if any(size % patch_size for size in CELEBA_LATENT_SHAPE[-2:]):
            raise ValueError("patch_size must divide the latent spatial dimensions")
        feature_patch_sizes_tuple = tuple(feature_patch_sizes)
        if not feature_patch_sizes_tuple or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in feature_patch_sizes_tuple
        ):
            raise ValueError("feature_patch_sizes must contain positive integers")
        if len(set(feature_patch_sizes_tuple)) != len(feature_patch_sizes_tuple):
            raise ValueError("feature_patch_sizes must not contain duplicates")
        if isinstance(every_k_block, bool) or not isinstance(every_k_block, int) or every_k_block < 1:
            raise ValueError("every_k_block must be a positive integer")
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError(
                f"feature_profile must be one of {FEATURE_PROFILES}, got {feature_profile!r}"
            )
        if mask_sampling not in MASK_SAMPLING_MODES:
            raise ValueError(
                f"mask_sampling must be one of {MASK_SAMPLING_MODES}, got {mask_sampling!r}"
            )
        total_patches = math.prod(
            size // patch_size for size in CELEBA_LATENT_SHAPE[-2:]
        )
        if exact_masked_patches is None:
            exact_masked_patches = total_patches // 2
        if (
            isinstance(exact_masked_patches, bool)
            or not isinstance(exact_masked_patches, int)
            or not 0 <= exact_masked_patches <= total_patches
        ):
            raise ValueError(
                f"exact_masked_patches must be an integer in [0, {total_patches}]"
            )

        self.base_channels = base_channels
        self.layers_config = layers_tuple
        self.dropout_prob = float(dropout_prob)
        self.gn_max_groups = gn_max_groups
        self.patch_size = patch_size
        self.feature_patch_sizes = feature_patch_sizes_tuple
        self.every_k_block = every_k_block
        self.feature_profile = feature_profile
        self.mask_sampling = mask_sampling
        self.exact_masked_patches = exact_masked_patches
        self.encoder = _ResNetEncoder(
            base_channels=base_channels,
            layers=layers_tuple,
            dropout_prob=self.dropout_prob,
            gn_max_groups=gn_max_groups,
        )
        self.decoder = _UNetDecoder(base_channels, gn_max_groups=gn_max_groups)
        self.apply(_init_like_flax)

    def sample_mask(
        self,
        latents: Tensor,
        mask_ratio: float | Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample according to the checkpointed masking strategy.

        ``mask_ratio`` belongs exclusively to ``official_bernoulli``.  Exact
        count models use their checkpointed ``exact_masked_patches`` value, so
        accidentally supplying a Bernoulli ratio is rejected rather than
        silently ignored.
        """

        if self.mask_sampling == MASK_SAMPLING_OFFICIAL_BERNOULLI:
            return make_patch_mask(
                latents,
                0.5 if mask_ratio is None else mask_ratio,
                patch_size=self.patch_size,
                generator=generator,
            )
        if mask_ratio is not None:
            raise ValueError("mask_ratio is only valid with official_bernoulli masking")
        return make_exact_patch_mask(
            latents,
            self.exact_masked_patches,
            patch_size=self.patch_size,
            generator=generator,
        )

    def _prepare_mask(self, mask: Tensor, like: Tensor) -> Tensor:
        prepared = _binary_mask(mask, like)
        coarse = prepared[:, :, :: self.patch_size, :: self.patch_size]
        reconstructed = coarse.repeat_interleave(self.patch_size, dim=-2).repeat_interleave(
            self.patch_size, dim=-1
        )
        if not torch.equal(prepared, reconstructed):
            raise ValueError(
                f"mask must be constant inside each {self.patch_size}x{self.patch_size} patch"
            )
        return prepared

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct the latent after zeroing all coarse masked patches."""

        _validate_latents(x)
        prepared_mask = self._prepare_mask(mask, x)
        visible = x * (1.0 - prepared_mask)
        features = self.encoder(visible)
        assert isinstance(features, dict)
        return self.decoder(features)

    def encode_features(self, x: Tensor) -> dict[str, Tensor]:
        """Extract the checkpointed feature profile without disabling autograd."""

        _validate_latents(x)
        encoded = self.encoder(
            x,
            return_block_outputs=True,
            # The released feature function always uses train=False. Keep
            # extraction deterministic even if a direct caller leaves the
            # enclosing module in training mode.
            training=False,
        )
        assert isinstance(encoded, tuple)
        features, block_outputs = encoded
        output: dict[str, Tensor] = {
            "global": x.reshape(x.shape[0], 1, -1),
            "norm_x": (
                x.float().square().mean(dim=(-2, -1)).add(SAFE_STD_EPS).sqrt().unsqueeze(1)
            ),
        }
        for name, feature_map in features.items():
            _process_feature(
                output,
                name,
                feature_map,
                patch_sizes=self.feature_patch_sizes,
                feature_profile=self.feature_profile,
            )
        for stage_index in range(1, 5):
            stage_name = f"layer{stage_index}"
            for block_index, feature_map in enumerate(block_outputs[stage_name], start=1):
                if block_index % self.every_k_block == 0:
                    _process_feature(
                        output,
                        f"{stage_name}_blk{block_index}",
                        feature_map,
                        patch_sizes=self.feature_patch_sizes,
                        feature_profile=self.feature_profile,
                    )
        return output

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "version": ARCHITECTURE_VERSION,
            "base_channels": self.base_channels,
            "layers": list(self.layers_config),
            "dropout_prob": self.dropout_prob,
            "gn_max_groups": self.gn_max_groups,
            "patch_size": self.patch_size,
            "feature_patch_sizes": list(self.feature_patch_sizes),
            "every_k_block": self.every_k_block,
            "feature_profile": self.feature_profile,
            "mask_sampling": self.mask_sampling,
            "exact_masked_patches": self.exact_masked_patches,
        }

    def checkpoint_dict(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        torch.save(self.checkpoint_dict(metadata=metadata), Path(checkpoint))


def load_frozen_feature_encoder_v2(
    checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    device: torch.device | str,
) -> CelebALatentMaskedAutoencoderV2:
    """Load a v2 checkpoint frozen in eval mode while preserving input autograd."""

    payload = (
        checkpoint
        if isinstance(checkpoint, Mapping)
        else torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
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
    required = (
        "base_channels",
        "layers",
        "dropout_prob",
        "gn_max_groups",
        "patch_size",
        "feature_patch_sizes",
        "every_k_block",
    )
    missing = [name for name in required if name not in architecture]
    if missing:
        raise ValueError(f"checkpoint architecture is missing: {', '.join(missing)}")
    optional_with_legacy_defaults = {
        "feature_profile": DEFAULT_FEATURE_PROFILE,
        "mask_sampling": DEFAULT_MASK_SAMPLING,
        # ``None`` derives half of the checkpoint's coarse grid.  It restores
        # 8/16 for the normal patch_size=2 while remaining compatible with an
        # older v2 checkpoint that used another valid patch size.
        "exact_masked_patches": None,
    }
    allowed = {"name", "version", *required, *optional_with_legacy_defaults}
    unexpected = sorted(set(architecture) - allowed)
    if unexpected:
        raise ValueError(
            f"checkpoint architecture has unexpected fields: {', '.join(unexpected)}"
        )
    model = CelebALatentMaskedAutoencoderV2(
        base_channels=architecture["base_channels"],
        layers=architecture["layers"],
        dropout_prob=architecture["dropout_prob"],
        gn_max_groups=architecture["gn_max_groups"],
        patch_size=architecture["patch_size"],
        feature_patch_sizes=architecture["feature_patch_sizes"],
        every_k_block=architecture["every_k_block"],
        feature_profile=architecture.get(
            "feature_profile", optional_with_legacy_defaults["feature_profile"]
        ),
        mask_sampling=architecture.get(
            "mask_sampling", optional_with_legacy_defaults["mask_sampling"]
        ),
        exact_masked_patches=architecture.get(
            "exact_masked_patches",
            optional_with_legacy_defaults["exact_masked_patches"],
        ),
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
    "DEFAULT_EVERY_K_BLOCK",
    "DEFAULT_EXACT_MASKED_PATCHES",
    "DEFAULT_FEATURE_PROFILE",
    "DEFAULT_FEATURE_PATCH_SIZES",
    "DEFAULT_LAYERS",
    "DEFAULT_MASK_SAMPLING",
    "DEFAULT_PATCH_SIZE",
    "FEATURE_PROFILE_OFFICIAL_70",
    "FEATURE_PROFILE_SMALL_GRID_52",
    "FEATURE_PROFILES",
    "MASK_SAMPLING_EXACT_COUNT",
    "MASK_SAMPLING_MODES",
    "MASK_SAMPLING_OFFICIAL_BERNOULLI",
    "SAFE_STD_EPS",
    "CelebALatentMaskedAutoencoderV2",
    "load_frozen_feature_encoder_v2",
    "make_exact_patch_mask",
    "make_patch_mask",
    "masked_reconstruction_loss",
    "safe_std",
]
