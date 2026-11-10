"""CelebA-64 pixel-space MAE features for the released GMD objective.

This is a PyTorch port/adaptation of the pixel MAE in the official Drifting
release (``models/mae_model.py`` at commit
``accd0cf09c33b70892d33941d2a287ca86cb92e1``).  It intentionally does not
use a VAE: public inputs and outputs are RGB images with shape
``[batch, 3, 64, 64]``.

The ImageNet-256 pixel MAE packs each ``8 x 8`` RGB patch into channels so its
ResNet sees a ``32 x 32`` grid.  This CelebA adaptation packs ``2 x 2`` RGB
patches, preserving that same encoder-grid geometry at 64px.  The released
post-activation ResNet, GroupNorm, U-Net decoder, Bernoulli mask geometry,
masked-only reconstruction loss, activation taps, and feature statistics are
otherwise retained.

Every feature has layout ``[samples, tokens, dimensions]`` and is directly
consumable by :func:`flowmaps.objectives.drifting.summed_feature_drifting_loss`.
Freezing model parameters does not disable the input Jacobian: generated RGB
pixels still receive gradients through the frozen encoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flowmaps.objectives.drifting import (
    OFFICIAL_TEMPERATURES,
    summed_feature_drifting_loss,
)

CELEBA_PIXEL_SHAPE = (3, 64, 64)
MODEL_SPACE = "RGB NCHW, nominal range [-1,1]"
OFFICIAL_RELEASE_COMMIT = "accd0cf09c33b70892d33941d2a287ca86cb92e1"
OFFICIAL_REFERENCE = {
    "repository": "lambertae/drifting",
    "commit": OFFICIAL_RELEASE_COMMIT,
    "source": "models/mae_model.py",
    "reference_recipe": "configs/mae/pixel_640.yaml",
}

CHECKPOINT_FORMAT = "driftmodel.celeba64_pixel_gmd_mae"
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_NAME = "CelebA64PixelGMDMaskedAutoencoder"
ARCHITECTURE_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
STATE_DICT_HASH_VERSION = 1

DEFAULT_BASE_CHANNELS = 64
DEFAULT_LAYERS = (3, 4, 6, 3)
DEFAULT_INPUT_PATCH_SIZE = 2
DEFAULT_MASK_PATCH_SIZE = 2
DEFAULT_FEATURE_PATCH_SIZES = (2, 4)
DEFAULT_EVERY_K_BLOCK = 2
DEFAULT_MASK_RATIO = 0.5
SAFE_STD_EPS = 1e-6
_TRUNCATED_NORMAL_STDDEV_FACTOR = 0.8796256610342398

FEATURE_PROFILE_RELEASE_ALL = "release_all"
FEATURE_PROFILE_NONDEGENERATE = "nondegenerate"
FEATURE_PROFILE_NONDEGENERATE_BASE = "nondegenerate_base"
FEATURE_PROFILES = (
    FEATURE_PROFILE_RELEASE_ALL,
    FEATURE_PROFILE_NONDEGENERATE,
    FEATURE_PROFILE_NONDEGENERATE_BASE,
)
DEFAULT_FEATURE_PROFILE = FEATURE_PROFILE_NONDEGENERATE


def _validate_images(images: Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_PIXEL_SHAPE:
        raise ValueError("CelebA pixels must have shape [B, 3, 64, 64]")
    if not images.is_floating_point():
        raise TypeError("CelebA pixels must be floating point")


def pack_image_patches(
    images: Tensor,
    patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
) -> Tensor:
    """Apply the release's HWC ``space-to-depth`` ordering in NCHW.

    The packed channel index is ordered as ``(patch_y, patch_x, RGB)`` just as
    the official einops expression ``(h2 w2 c)``; this is not PyTorch's native
    ``pixel_unshuffle`` channel order.
    """

    _validate_images(images)
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, int)
        or patch_size < 1
    ):
        raise ValueError("patch_size must be a positive integer")
    height, width = images.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide both image dimensions")
    batch, channels = images.shape[:2]
    packed = images.permute(0, 2, 3, 1).reshape(
        batch,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    packed = packed.permute(0, 1, 3, 2, 4, 5).reshape(
        batch,
        height // patch_size,
        width // patch_size,
        patch_size * patch_size * channels,
    )
    return packed.permute(0, 3, 1, 2).contiguous()


def unpack_image_patches(
    packed: Tensor,
    patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
) -> Tensor:
    """Invert :func:`pack_image_patches` back to ``[B,3,64,64]``."""

    if packed.ndim != 4 or not packed.is_floating_point():
        raise ValueError("packed pixels must be a floating [B,C,H,W] tensor")
    channels = CELEBA_PIXEL_SHAPE[0]
    expected_channels = channels * patch_size * patch_size
    expected_grid = CELEBA_PIXEL_SHAPE[-1] // patch_size
    if tuple(packed.shape[1:]) != (
        expected_channels,
        expected_grid,
        expected_grid,
    ):
        raise ValueError(
            "packed pixels have incompatible channels or spatial dimensions"
        )
    batch = packed.shape[0]
    unpacked = packed.permute(0, 2, 3, 1).reshape(
        batch,
        expected_grid,
        expected_grid,
        patch_size,
        patch_size,
        channels,
    )
    unpacked = unpacked.permute(0, 1, 3, 2, 4, 5).reshape(
        batch,
        CELEBA_PIXEL_SHAPE[1],
        CELEBA_PIXEL_SHAPE[2],
        channels,
    )
    return unpacked.permute(0, 3, 1, 2).contiguous()


def _binary_mask(mask: Tensor, like: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    expected = (like.shape[0], 1, like.shape[-2], like.shape[-1])
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"packed mask must have shape {expected} or {expected[:1] + expected[2:]}"
        )
    valid_dtype = (
        mask.dtype == torch.bool
        or mask.is_floating_point()
        or mask.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    )
    if not valid_dtype:
        raise TypeError("mask must be boolean or numeric")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("mask values must be finite")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask values must be binary (zero or one)")
    return mask.to(device=like.device, dtype=like.dtype)


def make_patch_mask(
    images: Tensor,
    mask_ratio: float | Tensor = DEFAULT_MASK_RATIO,
    *,
    input_patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
    mask_patch_size: int = DEFAULT_MASK_PATCH_SIZE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the released Bernoulli mask on the packed encoder grid.

    A value of one is hidden.  For the defaults, each Bernoulli decision masks
    a ``2 x 2`` region of the 32px packed grid, corresponding to a ``4 x 4``
    region in the original 64px image.
    """

    packed = pack_image_patches(images, input_patch_size)
    if (
        isinstance(mask_patch_size, bool)
        or not isinstance(mask_patch_size, int)
        or mask_patch_size < 1
    ):
        raise ValueError("mask_patch_size must be a positive integer")
    height, width = packed.shape[-2:]
    if height % mask_patch_size or width % mask_patch_size:
        raise ValueError("mask_patch_size must divide the packed spatial grid")
    ratio = torch.as_tensor(mask_ratio, device=images.device, dtype=torch.float32)
    if ratio.ndim == 0:
        ratio = ratio.expand(images.shape[0])
    elif ratio.numel() == images.shape[0]:
        ratio = ratio.reshape(images.shape[0])
    else:
        raise ValueError("mask_ratio must be scalar or contain one value per sample")
    if not bool(torch.isfinite(ratio).all()) or not bool(
        ((ratio >= 0) & (ratio <= 1)).all()
    ):
        raise ValueError("mask_ratio values must be finite and lie in [0, 1]")
    noise = torch.rand(
        (
            images.shape[0],
            1,
            height // mask_patch_size,
            width // mask_patch_size,
        ),
        device=images.device,
        dtype=torch.float32,
        generator=generator,
    )
    coarse = noise < ratio[:, None, None, None]
    return (
        coarse.repeat_interleave(mask_patch_size, -2)
        .repeat_interleave(mask_patch_size, -1)
        .to(dtype=images.dtype)
    )


def packed_mask_to_pixels(
    mask: Tensor,
    *,
    like: Tensor,
    input_patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
) -> Tensor:
    """Expand a one-channel packed mask to its RGB pixel-space support."""

    _validate_images(like)
    packed = pack_image_patches(like, input_patch_size)
    prepared = _binary_mask(mask, packed)
    expanded = prepared.expand(-1, packed.shape[1], -1, -1)
    return unpack_image_patches(expanded, input_patch_size)


def masked_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    input_patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
) -> Tensor:
    """Return the released masked-only MSE, one scalar per image.

    Errors are summed over the packed RGB channels and divided by the number
    of hidden packed positions, matching the official JAX implementation.
    """

    _validate_images(prediction)
    _validate_images(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction_packed = pack_image_patches(prediction, input_patch_size).float()
    target_packed = pack_image_patches(target, input_patch_size).float()
    prepared = _binary_mask(mask, target_packed).float()
    squared = (prediction_packed - target_packed).square()
    numerator = (squared * prepared).sum(dim=(1, 2, 3))
    denominator = prepared.sum(dim=(1, 2, 3)) + 1e-8
    return numerator / denominator


def safe_std(
    values: Tensor,
    dim: int | tuple[int, ...],
    *,
    keepdim: bool = False,
    eps: float = SAFE_STD_EPS,
) -> Tensor:
    """Population standard deviation accumulated in FP32, as in the release."""

    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    values_f32 = values.float()
    mean = values_f32.mean(dim=dim, keepdim=True)
    variance = (values_f32 - mean).square().mean(dim=dim, keepdim=keepdim)
    return (variance.clamp_min(0.0) + eps).sqrt()


def _choose_gn_groups(channels: int, maximum: int) -> int:
    groups = min(channels, maximum)
    while groups > 1 and channels % groups:
        groups -= 1
    return max(groups, 1)


def _group_norm(channels: int, maximum: int) -> nn.GroupNorm:
    return nn.GroupNorm(_choose_gn_groups(channels, maximum), channels, eps=1e-6)


def _init_like_flax(module: nn.Module) -> None:
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


class _BasicBlock(nn.Module):
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
        self.dropout_prob = dropout_prob
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
            p=self.dropout_prob,
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
        in_channels: int,
        *,
        base_channels: int,
        layers: tuple[int, int, int, int],
        dropout_prob: float,
        gn_max_groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1, bias=False
        )
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
        previous = base_channels
        for stage_index, (channels, blocks) in enumerate(zip(stage_channels, layers)):
            stages.append(
                _EncoderStage(
                    previous,
                    channels,
                    blocks=blocks,
                    stride=1 if stage_index == 0 else 2,
                    gn_max_groups=gn_max_groups,
                    dropout_prob=dropout_prob,
                )
            )
            norms.append(_group_norm(channels, gn_max_groups))
            previous = channels
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(norms)
        self.stage_channels = stage_channels

    def forward(
        self,
        x: Tensor,
        *,
        return_block_outputs: bool = False,
        training: bool | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[Tensor]]]:
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
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
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
            concatenated, out_channels, gn_max_groups=gn_max_groups
        )
        self.refinement = _ConvGNReLU(
            out_channels, out_channels, gn_max_groups=gn_max_groups
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.concat_norm(torch.cat((x, skip), dim=1))
        return self.refinement(self.projection(x))


class _UNetDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int,
        out_channels: int,
        *,
        gn_max_groups: int,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
        )
        self.bridge = _ConvGNReLU(c5, c5, gn_max_groups=gn_max_groups)
        self.up43 = _UpBlock(c5, c4, c4, gn_max_groups=gn_max_groups)
        self.up32 = _UpBlock(c4, c3, c3, gn_max_groups=gn_max_groups)
        self.up21 = _UpBlock(c3, c2, c2, gn_max_groups=gn_max_groups)
        self.up10 = _UpBlock(c2, c1, c1, gn_max_groups=gn_max_groups)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

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


def _include_patch_family(profile: str, patch_count: int) -> bool:
    return profile == FEATURE_PROFILE_RELEASE_ALL or patch_count > 1


def _process_feature(
    output: dict[str, Tensor],
    name: str,
    feature_map: Tensor,
    *,
    patch_sizes: Sequence[int],
    profile: str,
) -> None:
    output[name] = _tokens(feature_map)
    output[f"{name}_mean"] = feature_map.mean(dim=(-2, -1)).unsqueeze(1)
    output[f"{name}_std"] = safe_std(feature_map, (-2, -1)).unsqueeze(1)
    height, width = feature_map.shape[-2:]
    for patch_size in patch_sizes:
        if height % patch_size or width % patch_size:
            continue
        patch_count = (height // patch_size) * (width // patch_size)
        if not _include_patch_family(profile, patch_count):
            continue
        patches = _patch_values(feature_map, patch_size)
        output[f"{name}_mean_{patch_size}"] = _tokens(patches.mean(dim=(-2, -1)))
        output[f"{name}_std_{patch_size}"] = _tokens(safe_std(patches, (-2, -1)))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "checkpoint metadata must be finite canonical JSON data"
        ) from error
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash names, dtypes, shapes, and raw tensor payloads deterministically."""

    digest = hashlib.sha256(f"state-dict-v{STATE_DICT_HASH_VERSION}\0".encode())
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("state_dict must map string names to tensors")
        if value.layout != torch.strided:
            raise TypeError("only strided tensors are supported in checkpoints")
        tensor = value.detach().to(device="cpu").contiguous()
        descriptor = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        header = _canonical_json_bytes(descriptor)
        raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CelebA64PixelGMDMaskedAutoencoder(nn.Module):
    """Release-mechanical pixel MAE adapted from ImageNet-256 to CelebA-64."""

    def __init__(
        self,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        *,
        layers: Sequence[int] = DEFAULT_LAYERS,
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
        input_patch_size: int = DEFAULT_INPUT_PATCH_SIZE,
        mask_patch_size: int = DEFAULT_MASK_PATCH_SIZE,
        feature_patch_sizes: Sequence[int] = DEFAULT_FEATURE_PATCH_SIZES,
        every_k_block: int = DEFAULT_EVERY_K_BLOCK,
        feature_profile: str = DEFAULT_FEATURE_PROFILE,
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
        if (
            isinstance(gn_max_groups, bool)
            or not isinstance(gn_max_groups, int)
            or gn_max_groups < 1
        ):
            raise ValueError("gn_max_groups must be a positive integer")
        if input_patch_size != DEFAULT_INPUT_PATCH_SIZE:
            raise ValueError(
                "this versioned CelebA-64 architecture requires input_patch_size=2"
            )
        packed_grid = CELEBA_PIXEL_SHAPE[-1] // input_patch_size
        if (
            isinstance(mask_patch_size, bool)
            or not isinstance(mask_patch_size, int)
            or mask_patch_size < 1
            or packed_grid % mask_patch_size
        ):
            raise ValueError(
                "mask_patch_size must be positive and divide the packed grid"
            )
        feature_patch_sizes_tuple = tuple(feature_patch_sizes)
        if not feature_patch_sizes_tuple or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in feature_patch_sizes_tuple
        ):
            raise ValueError("feature_patch_sizes must contain positive integers")
        if len(set(feature_patch_sizes_tuple)) != len(feature_patch_sizes_tuple):
            raise ValueError("feature_patch_sizes must not contain duplicates")
        if (
            isinstance(every_k_block, bool)
            or not isinstance(every_k_block, int)
            or every_k_block < 1
        ):
            raise ValueError("every_k_block must be a positive integer")
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError(f"feature_profile must be one of {FEATURE_PROFILES}")

        self.base_channels = base_channels
        self.layers_config = layers_tuple
        self.dropout_prob = float(dropout_prob)
        self.gn_max_groups = gn_max_groups
        self.input_patch_size = input_patch_size
        self.mask_patch_size = mask_patch_size
        self.feature_patch_sizes = feature_patch_sizes_tuple
        self.every_k_block = every_k_block
        self.feature_profile = feature_profile
        self.packed_channels = CELEBA_PIXEL_SHAPE[0] * input_patch_size**2
        self.packed_grid = packed_grid

        self.encoder = _ResNetEncoder(
            self.packed_channels,
            base_channels=base_channels,
            layers=layers_tuple,
            dropout_prob=self.dropout_prob,
            gn_max_groups=gn_max_groups,
        )
        self.decoder = _UNetDecoder(
            base_channels,
            self.packed_channels,
            gn_max_groups=gn_max_groups,
        )
        self.apply(_init_like_flax)

    @property
    def include_block_taps(self) -> bool:
        return self.feature_profile != FEATURE_PROFILE_NONDEGENERATE_BASE

    def sample_mask(
        self,
        images: Tensor,
        mask_ratio: float | Tensor = DEFAULT_MASK_RATIO,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return make_patch_mask(
            images,
            mask_ratio,
            input_patch_size=self.input_patch_size,
            mask_patch_size=self.mask_patch_size,
            generator=generator,
        )

    def _prepare_mask(self, mask: Tensor, packed: Tensor) -> Tensor:
        prepared = _binary_mask(mask, packed)
        coarse = prepared[:, :, :: self.mask_patch_size, :: self.mask_patch_size]
        rebuilt = coarse.repeat_interleave(self.mask_patch_size, -2).repeat_interleave(
            self.mask_patch_size, -1
        )
        if not torch.equal(prepared, rebuilt):
            raise ValueError("mask must be constant inside each packed mask patch")
        return prepared

    def forward(self, images: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct RGB pixels after zeroing hidden packed-grid positions."""

        packed = pack_image_patches(images, self.input_patch_size)
        prepared = self._prepare_mask(mask, packed)
        encoded = self.encoder(packed * (1.0 - prepared))
        assert isinstance(encoded, dict)
        reconstructed = self.decoder(encoded)
        return unpack_image_patches(reconstructed, self.input_patch_size)

    def encode_features(self, images: Tensor) -> dict[str, Tensor]:
        """Extract deterministic features while preserving RGB input autograd."""

        _validate_images(images)
        packed = pack_image_patches(images, self.input_patch_size)
        encoded = self.encoder(
            packed,
            return_block_outputs=True,
            training=False,
        )
        assert isinstance(encoded, tuple)
        features, block_outputs = encoded
        output: dict[str, Tensor] = {
            # The release activation wrapper adds raw input globally even when
            # learned MAE/ConvNeXt features are active.  This is one GMD family,
            # not a standalone RGB KDE estimator.
            "global": images.reshape(images.shape[0], 1, -1),
            "norm_x": packed.float()
            .square()
            .mean(dim=(-2, -1))
            .add(SAFE_STD_EPS)
            .sqrt()
            .unsqueeze(1),
        }
        for name, feature_map in features.items():
            _process_feature(
                output,
                name,
                feature_map,
                patch_sizes=self.feature_patch_sizes,
                profile=self.feature_profile,
            )
        if self.include_block_taps:
            for stage_index in range(1, 5):
                stage_name = f"layer{stage_index}"
                for block_index, feature_map in enumerate(
                    block_outputs[stage_name], start=1
                ):
                    if block_index % self.every_k_block == 0:
                        _process_feature(
                            output,
                            f"{stage_name}_blk{block_index}",
                            feature_map,
                            patch_sizes=self.feature_patch_sizes,
                            profile=self.feature_profile,
                        )
        return output

    def _map_specs(self) -> list[tuple[str, int, int]]:
        sizes = (
            self.packed_grid,
            self.packed_grid,
            self.packed_grid // 2,
            self.packed_grid // 4,
            self.packed_grid // 8,
        )
        channels = (
            self.base_channels,
            self.base_channels,
            2 * self.base_channels,
            4 * self.base_channels,
            8 * self.base_channels,
        )
        specs = list(
            zip(("conv1", "layer1", "layer2", "layer3", "layer4"), sizes, channels)
        )
        if self.include_block_taps:
            for stage_index, (depth, size, dimension) in enumerate(
                zip(self.layers_config, sizes[1:], channels[1:]), start=1
            ):
                for block_index in range(
                    self.every_k_block, depth + 1, self.every_k_block
                ):
                    specs.append(
                        (f"layer{stage_index}_blk{block_index}", size, dimension)
                    )
        return specs

    def feature_schema(self) -> dict[str, Any]:
        families: list[dict[str, Any]] = [
            {
                "name": "global",
                "tokens": 1,
                "dimensions": math.prod(CELEBA_PIXEL_SHAPE),
            },
            {"name": "norm_x", "tokens": 1, "dimensions": self.packed_channels},
        ]
        for name, size, dimensions in self._map_specs():
            families.extend(
                (
                    {"name": name, "tokens": size * size, "dimensions": dimensions},
                    {"name": f"{name}_mean", "tokens": 1, "dimensions": dimensions},
                    {"name": f"{name}_std", "tokens": 1, "dimensions": dimensions},
                )
            )
            for patch_size in self.feature_patch_sizes:
                if size % patch_size:
                    continue
                patch_count = (size // patch_size) ** 2
                if not _include_patch_family(self.feature_profile, patch_count):
                    continue
                families.extend(
                    (
                        {
                            "name": f"{name}_mean_{patch_size}",
                            "tokens": patch_count,
                            "dimensions": dimensions,
                        },
                        {
                            "name": f"{name}_std_{patch_size}",
                            "tokens": patch_count,
                            "dimensions": dimensions,
                        },
                    )
                )
        return {
            "version": FEATURE_SCHEMA_VERSION,
            "input_layout": "NCHW",
            "input_shape": list(CELEBA_PIXEL_SHAPE),
            "model_space": MODEL_SPACE,
            "packed_shape": [self.packed_channels, self.packed_grid, self.packed_grid],
            "profile": self.feature_profile,
            "family_count": len(families),
            "families": families,
        }

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "version": ARCHITECTURE_VERSION,
            "image_shape": list(CELEBA_PIXEL_SHAPE),
            "base_channels": self.base_channels,
            "layers": list(self.layers_config),
            "dropout_prob": self.dropout_prob,
            "gn_max_groups": self.gn_max_groups,
            "input_patch_size": self.input_patch_size,
            "mask_patch_size": self.mask_patch_size,
            "feature_patch_sizes": list(self.feature_patch_sizes),
            "every_k_block": self.every_k_block,
            "feature_profile": self.feature_profile,
        }

    def checkpoint_dict(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata_copy = json.loads(
            _canonical_json_bytes(dict(metadata or {})).decode("utf-8")
        )
        architecture = self.architecture_config()
        feature_schema = self.feature_schema()
        state_dict = self.state_dict()
        integrity = {
            "algorithm": "sha256",
            "reference_sha256": canonical_json_sha256(OFFICIAL_REFERENCE),
            "architecture_sha256": canonical_json_sha256(architecture),
            "feature_schema_sha256": canonical_json_sha256(feature_schema),
            "metadata_sha256": canonical_json_sha256(metadata_copy),
            "state_dict_sha256": state_dict_sha256(state_dict),
        }
        return {
            "format": CHECKPOINT_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "reference": dict(OFFICIAL_REFERENCE),
            "architecture": architecture,
            "feature_schema": feature_schema,
            "metadata": metadata_copy,
            "state_dict": state_dict,
            "integrity": integrity,
        }

    def save_checkpoint(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        path = Path(checkpoint)
        torch.save(self.checkpoint_dict(metadata=metadata), path)
        return sha256_file(path)


_TOP_LEVEL_KEYS = {
    "format",
    "schema_version",
    "reference",
    "architecture",
    "feature_schema",
    "metadata",
    "state_dict",
    "integrity",
}
_ARCHITECTURE_KEYS = {
    "name",
    "version",
    "image_shape",
    "base_channels",
    "layers",
    "dropout_prob",
    "gn_max_groups",
    "input_patch_size",
    "mask_patch_size",
    "feature_patch_sizes",
    "every_k_block",
    "feature_profile",
}
_INTEGRITY_KEYS = {
    "algorithm",
    "reference_sha256",
    "architecture_sha256",
    "feature_schema_sha256",
    "metadata_sha256",
    "state_dict_sha256",
}


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], name: str
) -> None:
    missing = sorted(expected - set(mapping))
    unexpected = sorted(set(mapping) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"{name} fields are not exact: {', '.join(details)}")


def _checked_hash(actual: Any, value: Any, label: str) -> None:
    expected = canonical_json_sha256(value)
    if actual != expected:
        raise ValueError(f"checkpoint {label} SHA-256 mismatch")


def load_frozen_pixel_feature_encoder(
    checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    device: torch.device | str,
) -> CelebA64PixelGMDMaskedAutoencoder:
    """Strictly verify, restore, freeze, and return a pixel feature encoder."""

    payload = (
        checkpoint
        if isinstance(checkpoint, Mapping)
        else torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "checkpoint")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format must be {CHECKPOINT_FORMAT!r}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")

    reference = payload["reference"]
    architecture = payload["architecture"]
    feature_schema = payload["feature_schema"]
    metadata = payload["metadata"]
    state_dict = payload["state_dict"]
    integrity = payload["integrity"]
    for value, label in (
        (reference, "reference"),
        (architecture, "architecture"),
        (feature_schema, "feature_schema"),
        (metadata, "metadata"),
        (state_dict, "state_dict"),
        (integrity, "integrity"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint {label} must be a mapping")
    if dict(reference) != OFFICIAL_REFERENCE:
        raise ValueError(
            "checkpoint official reference does not match this implementation"
        )
    _require_exact_keys(architecture, _ARCHITECTURE_KEYS, "architecture")
    _require_exact_keys(integrity, _INTEGRITY_KEYS, "integrity")
    if integrity["algorithm"] != "sha256":
        raise ValueError("checkpoint integrity algorithm must be sha256")
    _checked_hash(integrity["reference_sha256"], reference, "reference")
    _checked_hash(integrity["architecture_sha256"], architecture, "architecture")
    _checked_hash(integrity["feature_schema_sha256"], feature_schema, "feature schema")
    _checked_hash(integrity["metadata_sha256"], metadata, "metadata")
    if integrity["state_dict_sha256"] != state_dict_sha256(state_dict):
        raise ValueError("checkpoint state_dict SHA-256 mismatch")

    if architecture["name"] != ARCHITECTURE_NAME:
        raise ValueError(f"checkpoint architecture must be {ARCHITECTURE_NAME!r}")
    if architecture["version"] != ARCHITECTURE_VERSION:
        raise ValueError("unsupported architecture version")
    if architecture["image_shape"] != list(CELEBA_PIXEL_SHAPE):
        raise ValueError("checkpoint is not a CelebA-64 RGB architecture")
    model = CelebA64PixelGMDMaskedAutoencoder(
        base_channels=architecture["base_channels"],
        layers=architecture["layers"],
        dropout_prob=architecture["dropout_prob"],
        gn_max_groups=architecture["gn_max_groups"],
        input_patch_size=architecture["input_patch_size"],
        mask_patch_size=architecture["mask_patch_size"],
        feature_patch_sizes=architecture["feature_patch_sizes"],
        every_k_block=architecture["every_k_block"],
        feature_profile=architecture["feature_profile"],
    )
    if model.architecture_config() != dict(architecture):
        raise ValueError("checkpoint architecture is not canonical")
    if model.feature_schema() != dict(feature_schema):
        raise ValueError("checkpoint feature schema does not match its architecture")
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model


def pixel_feature_drifting_loss(
    feature_encoder: CelebA64PixelGMDMaskedAutoencoder,
    generated_images: Tensor,
    positive_images: Tensor,
    *,
    fixed_negative_images: Tensor | None = None,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    distributed_statistics: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Evaluate release GMD over the frozen CelebA pixel feature pyramid.

    This wrapper deliberately enforces a frozen/eval encoder.  Targets are
    extracted without autograd; generated images retain their full pixel-to-
    feature Jacobian.  The underlying objective is the released normalized
    multi-temperature drifting loss, not the repository's historical RGB KDE.
    """

    if feature_encoder.training:
        raise ValueError("feature_encoder must be in eval mode")
    if any(parameter.requires_grad for parameter in feature_encoder.parameters()):
        raise ValueError("feature_encoder parameters must be frozen")
    _validate_images(generated_images)
    _validate_images(positive_images)
    if fixed_negative_images is not None:
        _validate_images(fixed_negative_images)

    generated_features = feature_encoder.encode_features(generated_images)
    with torch.no_grad():
        positive_features = feature_encoder.encode_features(positive_images.detach())
        negative_features = (
            None
            if fixed_negative_images is None
            else feature_encoder.encode_features(fixed_negative_images.detach())
        )
    return summed_feature_drifting_loss(
        generated_features,
        positive_features,
        fixed_negative_features=negative_features,
        temperatures=temperatures,
        distributed_statistics=distributed_statistics,
    )


__all__ = [
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_VERSION",
    "CELEBA_PIXEL_SHAPE",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_BASE_CHANNELS",
    "DEFAULT_EVERY_K_BLOCK",
    "DEFAULT_FEATURE_PATCH_SIZES",
    "DEFAULT_FEATURE_PROFILE",
    "DEFAULT_INPUT_PATCH_SIZE",
    "DEFAULT_LAYERS",
    "DEFAULT_MASK_PATCH_SIZE",
    "DEFAULT_MASK_RATIO",
    "FEATURE_PROFILE_NONDEGENERATE",
    "FEATURE_PROFILE_NONDEGENERATE_BASE",
    "FEATURE_PROFILE_RELEASE_ALL",
    "FEATURE_PROFILES",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_SPACE",
    "OFFICIAL_REFERENCE",
    "OFFICIAL_RELEASE_COMMIT",
    "SAFE_STD_EPS",
    "CelebA64PixelGMDMaskedAutoencoder",
    "canonical_json_sha256",
    "load_frozen_pixel_feature_encoder",
    "make_patch_mask",
    "masked_reconstruction_loss",
    "pack_image_patches",
    "packed_mask_to_pixels",
    "pixel_feature_drifting_loss",
    "safe_std",
    "sha256_file",
    "state_dict_sha256",
    "unpack_image_patches",
]
