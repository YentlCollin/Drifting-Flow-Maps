"""Direct RGB CelebA-64 generator for the released GMD objective.

The scientific equations are not reimplemented here.
:func:`global_pixel_gmd_loss` maps frozen MAE features to the released layout
and calls ``flowmaps.objectives.drifting.official_drifting_loss`` directly.  This is
needed to retain the official finite-batch global-negative support at CFG=1:
its weight is exactly zero, but its entries still participate in affinity
softmax denominators before weighting.  Consequently this module is a
direct-pixel GMD model, not the repository's historical RGB KDE and not a VAE
model.

The generator is a PyTorch adaptation of the release's ``DitGen``/
``LightningDiT`` (``lambertae/drifting`` commit
``accd0cf09c33b70892d33941d2a287ca86cb92e1``): Gaussian image-shaped input,
discrete noise-coordinate embeddings, trainable 2-D sine/cosine positions,
AdaLN-zero transformer blocks, q/k RMS normalization, RoPE, SwiGLU, RMSNorm,
the release's constant-1 CFG timestep branch, and a zero-initialized RGB head.
CelebA is unconditional, so ImageNet class lookup and CFG extrapolation are
absent, but the parameterized CFG embedder is retained exactly rather than
absorbed into the unconditional vector.  At 64px a 4x4 patch retains the
release's 16x16 token grid used by its 256px/16x16 pixel configuration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from flowmaps.features.mae import (
    CELEBA_PIXEL_SHAPE,
    CelebA64PixelGMDMaskedAutoencoder,
    canonical_json_sha256,
    state_dict_sha256,
)
from flowmaps.objectives.drifting import (
    OFFICIAL_TEMPERATURES,
    official_drifting_loss,
    token_features_to_groups,
)

OFFICIAL_RELEASE_COMMIT = "accd0cf09c33b70892d33941d2a287ca86cb92e1"
ARCHITECTURE_NAME = "CelebA64DirectPixelGMDDiT"
ARCHITECTURE_VERSION = 2
PRIOR_NAME = "standard_normal_rgb_nchw"
PRIOR_VERSION = 1
MODEL_SPACE = "RGB NCHW, unbounded generator output; real data in [-1,1]"
ARTIFACT_FORMAT = "driftmodel.celeba64_direct_pixel_gmd_generator"
ARTIFACT_VERSION = 2

DEFAULT_PATCH_SIZE = 4
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_DEPTH = 12
DEFAULT_HEADS = 12
DEFAULT_MLP_RATIO = 4.0
DEFAULT_NOISE_CLASSES = 64
DEFAULT_NOISE_COORDS = 32
DEFAULT_CONDITION_TOKENS = 16
INFERENCE_PRECISIONS = ("float32", "bfloat16_autocast")

OFFICIAL_PIXEL_GENERATOR_CONFIG = {
    "resolution": 256,
    "input_layout": "BHWC",
    "in_channels": 3,
    "out_channels": 3,
    "patch_size": 16,
    "token_grid": [16, 16],
    "hidden_size": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "noise_classes": 64,
    "noise_coords": 32,
    "n_cls_tokens": 16,
    "cfg_embedding": {
        "frequency_embedding_size": 256,
        "normalization": "RMSNorm",
        "output_multiplier": 0.02,
    },
    "qk_norm": True,
    "swiglu": True,
    "rope": True,
    "rmsnorm": True,
    "attention_fp32": True,
    "prior": "standard normal with image shape",
    "generator_updates": 100_000,
    "learning_rate": 2e-4,
    "warmup_updates": 10_000,
    "weight_decay": 0.01,
    "adam_betas": [0.9, 0.95],
    "ema_decay": 0.999,
    "generated_per_label": 64,
    "positive_per_label": 128,
    "positive_bank_size": 128,
    "negative_per_label": 32,
    "negative_bank_size": 1000,
    "temperatures": [0.2, 0.05, 0.02],
}


def _xavier_linear(module: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class RMSNorm(nn.Module):
    """Release-style RMSNorm with its variance accumulated in FP32."""

    def __init__(self, dimensions: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dimensions))

    def forward(self, values: Tensor) -> Tensor:
        input_dtype = values.dtype
        values_f32 = values.float()
        inverse_rms = torch.rsqrt(
            values_f32.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = values_f32 * inverse_rms * self.weight.float()
        return normalized.to(dtype=input_dtype)


class TimestepEmbedder(nn.Module):
    """Released sinusoidal CFG scalar embedder with two normal-init linears."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or frequency_embedding_size < 1:
            raise ValueError("timestep embedding dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.first = nn.Linear(self.frequency_embedding_size, self.hidden_size)
        self.second = nn.Linear(self.hidden_size, self.hidden_size)
        for layer in (self.first, self.second):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1 or not timesteps.is_floating_point():
            raise ValueError("CFG timesteps must be a floating [B] tensor")
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / float(half)
        )
        arguments = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return self.second(F.silu(self.first(embedding)))


def _rotate_half(values: Tensor) -> Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def apply_rope(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the release's 1-D RoPE over flattened image tokens."""

    sequence = query.shape[-2]
    head_dimension = query.shape[-1]
    if head_dimension % 2:
        raise ValueError("RoPE head dimension must be even")
    half = head_dimension // 2
    frequencies = 1.0 / (
        10000.0
        ** (torch.arange(half, device=query.device, dtype=torch.float32) / float(half))
    )
    positions = torch.arange(sequence, device=query.device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)[None, None]
    cosine = angles.cos()
    sine = angles.sin()
    query_f32 = query.float()
    key_f32 = key.float()
    return (
        query_f32 * cosine + _rotate_half(query_f32) * sine,
        key_f32 * cosine + _rotate_half(key_f32) * sine,
    )


def _sincos_1d(dimensions: int, positions: Tensor) -> Tensor:
    if dimensions % 2:
        raise ValueError("sine/cosine dimension must be even")
    omega = torch.arange(dimensions // 2, dtype=torch.float64)
    omega = 1.0 / (10000.0 ** (omega / (dimensions / 2.0)))
    angles = torch.outer(positions.reshape(-1).double(), omega)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def sincos_position_embedding(dimensions: int, grid_size: int) -> Tensor:
    """Return the release-compatible 2-D embedding ``[1, grid^2, D]``."""

    if dimensions % 4:
        raise ValueError("2-D sine/cosine dimensions must be divisible by four")
    coordinates = torch.arange(grid_size, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    # The release's ``np.meshgrid(grid_w, grid_h)`` concatenates x before y.
    embedding_x = _sincos_1d(dimensions // 2, grid_x.reshape(-1))
    embedding_y = _sincos_1d(dimensions // 2, grid_y.reshape(-1))
    return torch.cat((embedding_x, embedding_y), dim=1).float().unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        qk_norm: bool,
        rope: bool,
        attention_fp32: bool,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden size must be divisible by head count")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dimension = hidden_size // num_heads
        self.qk_norm = bool(qk_norm)
        self.rope = bool(rope)
        self.attention_fp32 = bool(attention_fp32)
        self.qkv = _xavier_linear(nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.projection = _xavier_linear(nn.Linear(hidden_size, hidden_size, bias=True))
        self.query_norm = RMSNorm(self.head_dimension) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.head_dimension) if qk_norm else nn.Identity()

    def forward(self, values: Tensor) -> Tensor:
        batch, tokens, _ = values.shape
        qkv = self.qkv(values).reshape(
            batch, tokens, 3, self.num_heads, self.head_dimension
        )
        query, key, value = qkv.unbind(dim=2)
        # [B,T,H,D] -> [B,H,T,D], matching PyTorch attention convention.
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        if self.attention_fp32:
            # Merely calling .float() is insufficient inside an outer autocast
            # region: matmul would be cast back to BF16.  The released
            # attn_fp32 contract covers RoPE, logits, softmax, and A@V.
            with torch.autocast(device_type=values.device.type, enabled=False):
                query, key, value = query.float(), key.float(), value.float()
                if self.rope:
                    query, key = apply_rope(query, key)
                scale = self.head_dimension**-0.5
                attention = torch.matmul(query * scale, key.transpose(-1, -2)).softmax(
                    dim=-1
                )
                output = torch.matmul(attention, value)
        else:
            if self.rope:
                query, key = apply_rope(query, key)
            scale = self.head_dimension**-0.5
            attention = torch.matmul(query * scale, key.transpose(-1, -2)).softmax(
                dim=-1
            )
            output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        # Under autocast the projection returns BF16 while its parameters stay FP32.
        return self.projection(output.to(dtype=values.dtype))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = int((2.0 / 3.0) * hidden_size * mlp_ratio)
        intermediate = ((intermediate + 31) // 32) * 32
        self.first = _xavier_linear(nn.Linear(hidden_size, intermediate))
        self.gate = _xavier_linear(nn.Linear(hidden_size, intermediate))
        self.output = _xavier_linear(nn.Linear(intermediate, hidden_size))

    def forward(self, values: Tensor) -> Tensor:
        return self.output(F.silu(self.first(values)) * self.gate(values))


def _modulate(values: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return values * (1.0 + scale[:, None]) + shift[:, None]


class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        qk_norm: bool,
        rope: bool,
        attention_fp32: bool,
    ) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(hidden_size)
        self.norm_mlp = RMSNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(
            hidden_size,
            num_heads,
            qk_norm=qk_norm,
            rope=rope,
            attention_fp32=attention_fp32,
        )
        self.mlp = SwiGLU(hidden_size, mlp_ratio)
        self.modulation = nn.Linear(hidden_size, 6 * hidden_size)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        # The release forces AdaLN scale/shift/gate computation to FP32 even
        # when the rest of the DiT is BF16.
        with torch.autocast(device_type=condition.device.type, enabled=False):
            modulation = self.modulation(F.silu(condition.float()))
        modulation = modulation.to(values.dtype)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )
        values = values + gate_attn[:, None] * self.attention(
            _modulate(self.norm_attention(values), shift_attn, scale_attn)
        )
        values = values + gate_mlp[:, None] * self.mlp(
            _modulate(self.norm_mlp(values), shift_mlp, scale_mlp)
        )
        return values


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dimensions: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.modulation = nn.Linear(hidden_size, 2 * hidden_size)
        self.output = nn.Linear(hidden_size, output_dimensions)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        with torch.autocast(device_type=condition.device.type, enabled=False):
            modulation = self.modulation(F.silu(condition.float()))
        shift, scale = modulation.chunk(2, dim=-1)
        values = _modulate(
            self.norm(values), shift.to(values.dtype), scale.to(values.dtype)
        )
        return self.output(values)


class CelebA64DirectPixelGMDDiT(nn.Module):
    """One-pass, unconditional RGB generator driven by an explicit RGB prior."""

    def __init__(
        self,
        *,
        patch_size: int = DEFAULT_PATCH_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        depth: int = DEFAULT_DEPTH,
        num_heads: int = DEFAULT_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        noise_classes: int = DEFAULT_NOISE_CLASSES,
        noise_coords: int = DEFAULT_NOISE_COORDS,
        condition_tokens: int = DEFAULT_CONDITION_TOKENS,
        qk_norm: bool = True,
        swiglu: bool = True,
        rope: bool = True,
        rmsnorm: bool = True,
        attention_fp32: bool = True,
        gradient_checkpointing: bool = False,
        intended_inference_precision: str = "float32",
    ) -> None:
        super().__init__()
        if not swiglu or not rmsnorm:
            raise ValueError("architecture version 2 requires SwiGLU and RMSNorm")
        integer_values = {
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "noise_classes": noise_classes,
            "noise_coords": noise_coords,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_values.values()
        ):
            raise ValueError("integer architecture settings must be positive integers")
        if 64 % patch_size:
            raise ValueError("patch size must divide 64")
        if hidden_size % num_heads or (hidden_size // num_heads) % 2:
            raise ValueError("head dimensions must be integral and even")
        if hidden_size % 4:
            raise ValueError("hidden size must be divisible by four")
        if not math.isfinite(float(mlp_ratio)) or mlp_ratio <= 0:
            raise ValueError("MLP ratio must be finite and positive")
        if (
            isinstance(condition_tokens, bool)
            or not isinstance(condition_tokens, int)
            or condition_tokens < 0
        ):
            raise ValueError("condition token count must be a non-negative integer")
        if intended_inference_precision not in INFERENCE_PRECISIONS:
            raise ValueError(
                f"intended inference precision must be one of {INFERENCE_PRECISIONS}"
            )

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.condition_tokens = condition_tokens
        self.qk_norm = bool(qk_norm)
        self.swiglu = True
        self.rope = bool(rope)
        self.rmsnorm = True
        self.attention_fp32 = bool(attention_fp32)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.intended_inference_precision = intended_inference_precision
        self.token_grid = 64 // patch_size
        patch_dimensions = 3 * patch_size * patch_size

        self.patch_projection = _xavier_linear(nn.Linear(patch_dimensions, hidden_size))
        self.position_embedding = nn.Parameter(
            sincos_position_embedding(hidden_size, self.token_grid)
        )
        self.unconditional_embedding = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.unconditional_embedding, std=0.02)
        self.noise_embeddings = nn.ModuleList(
            nn.Embedding(noise_classes, hidden_size) for _ in range(noise_coords)
        )
        for embedding in self.noise_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        self.cfg_embedder = TimestepEmbedder(hidden_size)
        self.cfg_norm = RMSNorm(hidden_size)
        self.condition_projection = _xavier_linear(nn.Linear(hidden_size, hidden_size))
        self.condition_token_embedding = nn.Parameter(
            torch.empty(1, condition_tokens, hidden_size)
        )
        nn.init.normal_(self.condition_token_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(
                hidden_size,
                num_heads,
                self.mlp_ratio,
                qk_norm=self.qk_norm,
                rope=self.rope,
                attention_fp32=self.attention_fp32,
            )
            for _ in range(depth)
        )
        self.final = FinalLayer(hidden_size, patch_dimensions)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "version": ARCHITECTURE_VERSION,
            "image_shape": list(CELEBA_PIXEL_SHAPE),
            "model_space": MODEL_SPACE,
            "patch_size": self.patch_size,
            "token_grid": [self.token_grid, self.token_grid],
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "noise_classes": self.noise_classes,
            "noise_coords": self.noise_coords,
            "condition_tokens": self.condition_tokens,
            "qk_norm": self.qk_norm,
            "swiglu": self.swiglu,
            "rope": self.rope,
            "rmsnorm": self.rmsnorm,
            "attention_fp32": self.attention_fp32,
            "gradient_checkpointing": self.gradient_checkpointing,
            "intended_inference_precision": self.intended_inference_precision,
            "conditioning": (
                "one unconditional embedding plus discrete noise-coordinate embeddings "
                "plus released constant-1 CFG timestep branch"
            ),
            "cfg_embedding": {
                "input_scale": 1.0,
                "output_multiplier": 0.02,
                "frequency_embedding_size": 256,
                "normalized": True,
                "trainable": True,
                "guidance_extrapolation": False,
            },
            "class_conditioning": False,
            "classifier_free_guidance": False,
            "prior": {
                "name": PRIOR_NAME,
                "version": PRIOR_VERSION,
                "shape": list(CELEBA_PIXEL_SHAPE),
                "mean": 0.0,
                "standard_deviation": 1.0,
            },
        }

    def sample_prior(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch size must be a positive integer")
        prior = torch.randn(
            (batch_size, *CELEBA_PIXEL_SHAPE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        labels = torch.randint(
            self.noise_classes,
            (batch_size, self.noise_coords),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
        return prior, labels

    def _patchify(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_PIXEL_SHAPE:
            raise ValueError("generator prior must have shape [B,3,64,64]")
        batch = images.shape[0]
        patch = self.patch_size
        grid = self.token_grid
        return (
            images.permute(0, 2, 3, 1)
            .reshape(batch, grid, patch, grid, patch, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, grid * grid, patch * patch * 3)
        )

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        patch = self.patch_size
        grid = self.token_grid
        return (
            tokens.reshape(batch, grid, grid, patch, patch, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, 64, 64, 3)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _condition(self, noise_labels: Tensor, *, dtype: torch.dtype) -> Tensor:
        if noise_labels.ndim != 2 or noise_labels.shape[1] != self.noise_coords:
            raise ValueError(f"noise labels must have shape [B,{self.noise_coords}]")
        if noise_labels.dtype != torch.int64:
            raise TypeError("noise labels must be int64")
        condition = self.unconditional_embedding.to(dtype=dtype)[None].expand(
            noise_labels.shape[0], -1
        )
        for coordinate, embedding in enumerate(self.noise_embeddings):
            condition = condition + embedding(noise_labels[:, coordinate]).to(
                dtype=dtype
            )
        cfg_scale = torch.ones(
            noise_labels.shape[0], device=noise_labels.device, dtype=torch.float32
        )
        cfg_condition = self.cfg_norm(self.cfg_embedder(cfg_scale))
        condition = condition + cfg_condition.to(dtype=dtype) * 0.02
        return condition

    def forward(self, prior: Tensor, noise_labels: Tensor) -> Tensor:
        if not self.training:
            _validate_inference_autocast(self, prior.device)
        values = self.patch_projection(self._patchify(prior))
        values_dtype = values.dtype
        # The release keeps pos_embed in FP32 for the addition and only then
        # casts the sum to its compute dtype.
        values = (values.float() + self.position_embedding.float()).to(
            dtype=values_dtype
        )
        condition = self._condition(noise_labels, dtype=values.dtype)
        if self.condition_tokens:
            condition_values = self.condition_projection(condition)[:, None].expand(
                -1, self.condition_tokens, -1
            )
            condition_values_dtype = condition_values.dtype
            # Same FP32-add-then-cast order as the release's cls_embed branch.
            condition_values = (
                condition_values.float() + self.condition_token_embedding.float()
            ).to(dtype=condition_values_dtype)
            values = torch.cat((condition_values, values), dim=1)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                values = activation_checkpoint(
                    block, values, condition, use_reentrant=False
                )
            else:
                values = block(values, condition)
        values = self.final(values, condition)
        if self.condition_tokens:
            values = values[:, self.condition_tokens :]
        return self._unpatchify(values)


class PositiveImageBank:
    """Single-class ring buffer matching unconditional CelebA support semantics."""

    FORMAT = "driftmodel.celeba64_pixel_gmd_positive_bank"
    VERSION = 1

    def __init__(self, maximum_size: int) -> None:
        if (
            isinstance(maximum_size, bool)
            or not isinstance(maximum_size, int)
            or maximum_size < 1
        ):
            raise ValueError("positive bank size must be a positive integer")
        self.maximum_size = maximum_size
        self.storage = torch.zeros(
            (maximum_size, *CELEBA_PIXEL_SHAPE), dtype=torch.float32
        )
        self.pointer = 0
        self.count = 0
        self.total_inserted = 0

    def add(self, images: Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_PIXEL_SHAPE:
            raise ValueError("bank images must have shape [B,3,64,64]")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("bank images must be finite floating point")
        if images.shape[0] > self.maximum_size:
            raise ValueError("one insertion cannot exceed positive bank capacity")
        images_cpu = images.detach().to(device="cpu", dtype=torch.float32)
        slots = (torch.arange(images_cpu.shape[0]) + self.pointer) % self.maximum_size
        self.storage.index_copy_(0, slots, images_cpu)
        self.pointer = int((self.pointer + images_cpu.shape[0]) % self.maximum_size)
        self.count = min(self.maximum_size, self.count + images_cpu.shape[0])
        self.total_inserted += images_cpu.shape[0]

    def sample(self, count: int, *, generator: torch.Generator) -> Tensor:
        if self.count < 1:
            raise RuntimeError("positive bank is empty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("sample count must be a positive integer")
        if count <= self.count:
            indices = torch.randperm(self.count, generator=generator)[:count]
        else:
            indices = torch.randint(self.count, (count,), generator=generator)
        return self.storage.index_select(0, indices).clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "maximum_size": self.maximum_size,
            "pointer": self.pointer,
            "count": self.count,
            "total_inserted": self.total_inserted,
            "storage": self.storage.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "format",
            "version",
            "maximum_size",
            "pointer",
            "count",
            "total_inserted",
            "storage",
        }
        if set(state) != expected:
            raise ValueError("positive bank checkpoint fields are not exact")
        if state["format"] != self.FORMAT or state["version"] != self.VERSION:
            raise ValueError("unsupported positive bank checkpoint")
        if state["maximum_size"] != self.maximum_size:
            raise ValueError("positive bank capacity changed at resume")
        storage = state["storage"]
        if not isinstance(storage, Tensor) or storage.shape != self.storage.shape:
            raise ValueError("positive bank storage shape is invalid")
        pointer, count, inserted = (
            int(state["pointer"]),
            int(state["count"]),
            int(state["total_inserted"]),
        )
        if not 0 <= pointer < self.maximum_size or not 0 <= count <= self.maximum_size:
            raise ValueError("positive bank cursor is invalid")
        if inserted < count:
            raise ValueError("positive bank insertion counter is invalid")
        self.storage.copy_(storage.to(dtype=torch.float32, device="cpu"))
        self.pointer, self.count, self.total_inserted = pointer, count, inserted


class NegativeImageBank(PositiveImageBank):
    """Global real-example bank retained in affinities with zero CFG=1 weight."""

    FORMAT = "driftmodel.celeba64_pixel_gmd_negative_bank"


def gather_generated_images(images: Tensor) -> Tensor:
    """Differentiably gather the generated support in deterministic rank order."""

    if not dist.is_available() or not dist.is_initialized():
        return images
    gathered = differentiable_all_gather(images)
    return torch.cat(tuple(gathered), dim=0)


def global_pixel_gmd_loss(
    feature_encoder: CelebA64PixelGMDMaskedAutoencoder,
    local_generated_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    temperatures: tuple[float, ...] = OFFICIAL_TEMPERATURES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply official GMD to the complete cross-rank generated support.

    The released objective places detached generated samples and fixed global
    real examples in its negative-side target support.  At CFG=1 the latter
    have weight exactly zero *after* row/column softmax construction, so they
    affect finite-batch affinity denominators even though their force
    coefficients vanish after weighting.  Every rank evaluates the same
    global support; differentiable all-gather plus DDP produces the average
    over all generated queries.
    """

    if feature_encoder.training:
        raise ValueError("feature encoder must be in eval mode")
    if any(parameter.requires_grad for parameter in feature_encoder.parameters()):
        raise ValueError("feature encoder parameters must be frozen")
    for name, images in (
        ("local generated", local_generated_images),
        ("positive", positive_images),
        ("fixed negative", fixed_negative_images),
    ):
        if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_PIXEL_SHAPE:
            raise ValueError(f"{name} images must have shape [B,3,64,64]")
        if images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError(f"{name} images must be non-empty floating point")
    generated = gather_generated_images(local_generated_images)
    generated_features = feature_encoder.encode_features(generated)
    with torch.no_grad():
        positive_features = feature_encoder.encode_features(positive_images.detach())
        negative_features = feature_encoder.encode_features(
            fixed_negative_images.detach()
        )
    if not (
        generated_features.keys()
        == positive_features.keys()
        == negative_features.keys()
    ):
        raise RuntimeError("frozen feature families differ across GMD supports")

    total: Tensor | None = None
    statistics: dict[str, Tensor] = {}
    zero_weight_sum = generated.new_zeros((), dtype=torch.float32)
    for family_name in generated_features:
        # Feature extraction may stay under the outer BF16 autocast.  The
        # released distance/affinity/force geometry must not: autocast would
        # otherwise silently demote float() einsums and matmuls back to BF16.
        with torch.autocast(device_type=generated.device.type, enabled=False):
            generated_groups = token_features_to_groups(generated_features[family_name])
            positive_groups = token_features_to_groups(positive_features[family_name])
            negative_groups = token_features_to_groups(negative_features[family_name])
            negative_weight = torch.zeros(
                (negative_groups.shape[0], negative_groups.shape[1]),
                device=generated.device,
                dtype=torch.float32,
            )
            family_loss, family_info = official_drifting_loss(
                generated_groups,
                positive_groups,
                negative_groups,
                negative_weight=negative_weight,
                temperatures=temperatures,
                distributed_statistics=False,
            )
            mean_loss = family_loss.mean()
        total = mean_loss if total is None else total + mean_loss
        statistics[f"loss/{family_name}"] = mean_loss.detach()
        statistics[f"scale/{family_name}"] = family_info["scale"]
        for key, value in family_info.items():
            if key.startswith("force_energy_"):
                statistics[f"{key}/{family_name}"] = value
        zero_weight_sum = zero_weight_sum + negative_weight.sum()

    if total is None:
        raise RuntimeError("frozen feature encoder returned no GMD family")
    statistics["loss"] = total.detach()
    statistics["support/generated_global"] = torch.tensor(
        generated.shape[0], device=total.device, dtype=torch.float32
    )
    statistics["support/positive"] = torch.tensor(
        positive_images.shape[0], device=total.device, dtype=torch.float32
    )
    statistics["support/fixed_negative"] = torch.tensor(
        fixed_negative_images.shape[0], device=total.device, dtype=torch.float32
    )
    statistics["support/fixed_negative_weight_sum"] = zero_weight_sum.detach()
    return total, statistics


_ARCHITECTURE_FIELDS = {
    "name",
    "version",
    "image_shape",
    "model_space",
    "patch_size",
    "token_grid",
    "hidden_size",
    "depth",
    "num_heads",
    "mlp_ratio",
    "noise_classes",
    "noise_coords",
    "condition_tokens",
    "qk_norm",
    "swiglu",
    "rope",
    "rmsnorm",
    "attention_fp32",
    "gradient_checkpointing",
    "intended_inference_precision",
    "conditioning",
    "cfg_embedding",
    "class_conditioning",
    "classifier_free_guidance",
    "prior",
}


def generator_from_architecture(
    architecture: Mapping[str, Any],
) -> CelebA64DirectPixelGMDDiT:
    """Reconstruct version 2 only after exact semantic validation."""

    if set(architecture) != _ARCHITECTURE_FIELDS:
        raise ValueError("generator architecture fields are not exact")
    if (
        architecture["name"] != ARCHITECTURE_NAME
        or architecture["version"] != ARCHITECTURE_VERSION
    ):
        raise ValueError("unsupported generator architecture")
    if architecture["image_shape"] != list(CELEBA_PIXEL_SHAPE):
        raise ValueError("artifact is not a CelebA-64 RGB generator")
    if architecture["model_space"] != MODEL_SPACE:
        raise ValueError("generator model-space semantics differ")
    expected_prior = {
        "name": PRIOR_NAME,
        "version": PRIOR_VERSION,
        "shape": list(CELEBA_PIXEL_SHAPE),
        "mean": 0.0,
        "standard_deviation": 1.0,
    }
    if architecture["prior"] != expected_prior:
        raise ValueError("generator prior semantics differ")
    if (
        architecture["class_conditioning"] is not False
        or architecture["classifier_free_guidance"] is not False
    ):
        raise ValueError("CelebA generator must remain unconditional without CFG")
    if architecture["conditioning"] != (
        "one unconditional embedding plus discrete noise-coordinate embeddings "
        "plus released constant-1 CFG timestep branch"
    ):
        raise ValueError("generator conditioning semantics differ")
    expected_cfg_embedding = {
        "input_scale": 1.0,
        "output_multiplier": 0.02,
        "frequency_embedding_size": 256,
        "normalized": True,
        "trainable": True,
        "guidance_extrapolation": False,
    }
    if architecture["cfg_embedding"] != expected_cfg_embedding:
        raise ValueError("generator constant CFG embedding semantics differ")
    model = CelebA64DirectPixelGMDDiT(
        patch_size=architecture["patch_size"],
        hidden_size=architecture["hidden_size"],
        depth=architecture["depth"],
        num_heads=architecture["num_heads"],
        mlp_ratio=architecture["mlp_ratio"],
        noise_classes=architecture["noise_classes"],
        noise_coords=architecture["noise_coords"],
        condition_tokens=architecture["condition_tokens"],
        qk_norm=architecture["qk_norm"],
        swiglu=architecture["swiglu"],
        rope=architecture["rope"],
        rmsnorm=architecture["rmsnorm"],
        attention_fp32=architecture["attention_fp32"],
        gradient_checkpointing=architecture["gradient_checkpointing"],
        intended_inference_precision=architecture["intended_inference_precision"],
    )
    if model.architecture_config() != dict(architecture):
        raise ValueError("generator architecture is not canonical")
    return model


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        if device_type == "cpu":
            return bool(torch.is_autocast_cpu_enabled())
        return bool(torch.is_autocast_enabled())


def _autocast_dtype(device_type: str) -> torch.dtype:
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        if device_type == "cpu":
            return torch.get_autocast_cpu_dtype()
        return torch.get_autocast_gpu_dtype()


def _validate_inference_autocast(
    model: CelebA64DirectPixelGMDDiT, device: torch.device
) -> None:
    if model.intended_inference_precision == "bfloat16_autocast":
        if (
            not _is_autocast_enabled(device.type)
            or _autocast_dtype(device.type) != torch.bfloat16
        ):
            raise RuntimeError(
                "this artifact was trained for BF16 autocast inference; use "
                "pixel_gmd_inference_autocast instead of evaluating silently in FP32"
            )
    elif _is_autocast_enabled(device.type):
        raise RuntimeError(
            "this artifact records float32 inference but autocast is enabled"
        )


@contextmanager
def pixel_gmd_inference_autocast(
    model: CelebA64DirectPixelGMDDiT,
    device: torch.device | str,
):
    """Enforce the precision serialized in a phase-3 generator artifact."""

    resolved = torch.device(device)
    if model.training:
        raise ValueError("inference precision context requires model.eval()")
    enabled = model.intended_inference_precision == "bfloat16_autocast"
    with torch.autocast(
        device_type=resolved.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    ):
        yield


def generator_artifact_dict(
    model: CelebA64DirectPixelGMDDiT,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a weights-only-safe, internally hashed inference artifact."""

    metadata_copy = dict(metadata)
    # Validate canonical JSON and detach every tensor onto CPU before hashing.
    canonical_json_sha256(metadata_copy)
    architecture = model.architecture_config()
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    reference = {
        "repository": "lambertae/drifting",
        "commit": OFFICIAL_RELEASE_COMMIT,
        "sources": ["models/generator.py", "train.py", "drift_loss.py"],
        "reference_recipe": "configs/gen/pixel_sota_B.yaml",
    }
    integrity = {
        "algorithm": "sha256",
        "reference_sha256": canonical_json_sha256(reference),
        "architecture_sha256": canonical_json_sha256(architecture),
        "metadata_sha256": canonical_json_sha256(metadata_copy),
        "state_dict_sha256": state_dict_sha256(state),
    }
    return {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "reference": reference,
        "architecture": architecture,
        "metadata": metadata_copy,
        "state_dict": state,
        "integrity": integrity,
    }


def load_pixel_gmd_generator_artifact(
    artifact: str | Path | Mapping[str, Any],
    device: torch.device | str,
) -> tuple[CelebA64DirectPixelGMDDiT, dict[str, Any]]:
    """Strictly verify and load a raw or EMA generator artifact."""

    payload = (
        artifact
        if isinstance(artifact, Mapping)
        else torch.load(Path(artifact), map_location="cpu", weights_only=True)
    )
    expected_top = {
        "format",
        "version",
        "reference",
        "architecture",
        "metadata",
        "state_dict",
        "integrity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("generator artifact fields are not exact")
    if payload["format"] != ARTIFACT_FORMAT or payload["version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported generator artifact")
    reference = payload["reference"]
    architecture = payload["architecture"]
    metadata = payload["metadata"]
    state = payload["state_dict"]
    integrity = payload["integrity"]
    if not all(
        isinstance(value, Mapping)
        for value in (reference, architecture, metadata, state, integrity)
    ):
        raise ValueError("generator artifact contains malformed mappings")
    expected_reference = {
        "repository": "lambertae/drifting",
        "commit": OFFICIAL_RELEASE_COMMIT,
        "sources": ["models/generator.py", "train.py", "drift_loss.py"],
        "reference_recipe": "configs/gen/pixel_sota_B.yaml",
    }
    if dict(reference) != expected_reference:
        raise ValueError("generator artifact official reference differs")
    expected_integrity = {
        "algorithm": "sha256",
        "reference_sha256": canonical_json_sha256(reference),
        "architecture_sha256": canonical_json_sha256(architecture),
        "metadata_sha256": canonical_json_sha256(metadata),
        "state_dict_sha256": state_dict_sha256(state),
    }
    if dict(integrity) != expected_integrity:
        raise ValueError("generator artifact internal SHA-256 mismatch")
    model = generator_from_architecture(architecture)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, dict(metadata)


__all__ = [
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_VERSION",
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "DEFAULT_DEPTH",
    "DEFAULT_CONDITION_TOKENS",
    "DEFAULT_HEADS",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_MLP_RATIO",
    "DEFAULT_NOISE_CLASSES",
    "DEFAULT_NOISE_COORDS",
    "DEFAULT_PATCH_SIZE",
    "MODEL_SPACE",
    "INFERENCE_PRECISIONS",
    "MultiHeadSelfAttention",
    "NegativeImageBank",
    "OFFICIAL_PIXEL_GENERATOR_CONFIG",
    "OFFICIAL_RELEASE_COMMIT",
    "PRIOR_NAME",
    "PRIOR_VERSION",
    "CelebA64DirectPixelGMDDiT",
    "PositiveImageBank",
    "RMSNorm",
    "TimestepEmbedder",
    "apply_rope",
    "gather_generated_images",
    "generator_artifact_dict",
    "generator_from_architecture",
    "global_pixel_gmd_loss",
    "load_pixel_gmd_generator_artifact",
    "pixel_gmd_inference_autocast",
    "sincos_position_embedding",
]
