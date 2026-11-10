"""Trainable MAE-82 + ConvNeXt-13 feature path for CelebA-64 GMD.

This module is a new branch layered on top of the audited direct-RGB trainer.
It deliberately leaves that trainer and its active runs untouched.  The
generated support is gathered differentiably before either encoder runs.  The
MAE keeps the surrounding trainer autocast (BF16 on MI210), whereas the
ConvNeXt adapter enforces FP32 internally and activation-checkpoints generated
chunks non-reentrantly.  Fixed supports are encoded under ``torch.no_grad``.

Chunking ends at feature extraction: all chunks are concatenated along the
sample axis before exactly one official drifting-loss call per family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from flowmaps.features.convnext import (
    CONVNEXT_FAMILY_NAMES,
    CONVNEXT_GRN_COUNT,
    CelebA64ConvNextV2Features,
    extract_convnext_supports,
    extract_fixed_convnext_features,
    extract_generated_convnext_features,
    merge_feature_families,
)
from flowmaps.features.mae import CELEBA_PIXEL_SHAPE, CelebA64PixelGMDMaskedAutoencoder
from flowmaps.models.backbone import gather_generated_images
from flowmaps.objectives.drifting import (
    OFFICIAL_TEMPERATURES,
    official_drifting_loss,
    token_features_to_groups,
)

MAE_FAMILY_COUNT = 82
CONVNEXT_FAMILY_COUNT = 13
COMPOSITE_FAMILY_COUNT = MAE_FAMILY_COUNT + CONVNEXT_FAMILY_COUNT
GENERATED_CONVNEXT_CHUNK_SIZE = 8
FIXED_CONVNEXT_CHUNK_SIZE = 32
CHECKPOINT_GENERATED_CONVNEXT = True
CHECKPOINT_IMPLEMENTATION = "torch.utils.checkpoint.use_reentrant_false"
FIXED_SUPPORT_AUTOGRAD = "torch.no_grad"
NEGATIVE_WEIGHT = 0.0


def _validate_images(images: Tensor, label: str) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_PIXEL_SHAPE:
        raise ValueError(f"{label} must have shape [B,3,64,64]")
    if images.shape[0] < 1 or not images.is_floating_point():
        raise ValueError(f"{label} must be a non-empty floating tensor")


def _assert_feature_dictionary(
    features: Mapping[str, Tensor],
    *,
    expected_count: int,
    label: str,
) -> None:
    if len(features) != expected_count:
        raise RuntimeError(
            f"{label} exposed {len(features)} families, expected {expected_count}"
        )
    for name, value in features.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{label} contains an invalid family name")
        if value.ndim != 3 or value.shape[0] < 1:
            raise RuntimeError(
                f"{label} family {name!r} must have shape [samples,tokens,dimensions]"
            )
        if value.shape[1] < 1 or value.shape[2] < 1 or not value.is_floating_point():
            raise RuntimeError(f"{label} family {name!r} has invalid geometry")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{label} family {name!r} is non-finite")


class FrozenMAEConvNextFeatures(nn.Module):
    """Permanently frozen pair of the production MAE and pinned ConvNeXt."""

    def __init__(
        self,
        mae: CelebA64PixelGMDMaskedAutoencoder,
        convnext: CelebA64ConvNextV2Features,
    ) -> None:
        super().__init__()
        if not isinstance(mae, CelebA64PixelGMDMaskedAutoencoder):
            raise TypeError("mae must be CelebA64PixelGMDMaskedAutoencoder")
        if not isinstance(convnext, CelebA64ConvNextV2Features):
            raise TypeError("convnext must be CelebA64ConvNextV2Features")
        schema = mae.feature_schema()
        if (
            schema.get("profile") != "nondegenerate"
            or schema.get("family_count") != MAE_FAMILY_COUNT
        ):
            raise ValueError(
                "MAE must expose the production nondegenerate 82-family schema"
            )
        if tuple(convnext.model_identity.get("feature_family_names", ())) != (
            CONVNEXT_FAMILY_NAMES
        ):
            raise ValueError(
                "ConvNeXt identity lacks the exact ordered 13-family schema"
            )
        if convnext.model_identity.get("official_jax_grn_replacement_count") != (
            CONVNEXT_GRN_COUNT
        ):
            raise ValueError("ConvNeXt identity lacks exactly 36 corrected GRNs")

        self.mae = mae
        self.convnext = convnext
        self.requires_grad_(False)
        super().train(False)
        self._last_local_generated: Tensor | None = None

    def train(self, mode: bool = True) -> "FrozenMAEConvNextFeatures":
        del mode
        super().train(False)
        self.mae.train(False)
        self.convnext.train(False)
        return self

    def assert_frozen_eval(self) -> None:
        if self.training or self.mae.training or self.convnext.training:
            raise RuntimeError("MAE and ConvNeXt must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("MAE and ConvNeXt parameters must remain frozen")

    @property
    def convnext_identity(self) -> dict[str, Any]:
        return self.convnext.model_identity

    @property
    def last_local_generated(self) -> Tensor | None:
        return self._last_local_generated

    def retain_generated_gradient(self, images: Tensor) -> None:
        if not images.requires_grad:
            raise RuntimeError("generated images must require gradients")
        images.retain_grad()
        self._last_local_generated = images

    def clear_step_state(self) -> None:
        self._last_local_generated = None


def _extract_mae_supports(
    encoder: CelebA64PixelGMDMaskedAutoencoder,
    global_generated_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Keep the existing MAE autocast path and detach only fixed supports."""

    generated = encoder.encode_features(global_generated_images)
    with torch.no_grad():
        positive = encoder.encode_features(positive_images.detach())
        negative = encoder.encode_features(fixed_negative_images.detach())
    expected_names = tuple(
        family["name"] for family in encoder.feature_schema()["families"]
    )
    for label, features in (
        ("generated MAE", generated),
        ("positive MAE", positive),
        ("negative MAE", negative),
    ):
        _assert_feature_dictionary(
            features, expected_count=MAE_FAMILY_COUNT, label=label
        )
    if not (tuple(generated) == tuple(positive) == tuple(negative) == expected_names):
        raise RuntimeError(
            "MAE family order differs from its schema or across supports"
        )
    return generated, positive, negative


def extract_composite_supports(
    feature_encoder: FrozenMAEConvNextFeatures,
    local_generated_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    generated_convnext_chunk_size: int = GENERATED_CONVNEXT_CHUNK_SIZE,
    fixed_convnext_chunk_size: int = FIXED_CONVNEXT_CHUNK_SIZE,
    checkpoint_generated_convnext: bool = CHECKPOINT_GENERATED_CONVNEXT,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor]:
    """Gather once, encode both frozen representations, then merge 82+13."""

    feature_encoder.assert_frozen_eval()
    for label, images in (
        ("local generated", local_generated_images),
        ("positive", positive_images),
        ("fixed negative", fixed_negative_images),
    ):
        _validate_images(images, label)
    if generated_convnext_chunk_size != GENERATED_CONVNEXT_CHUNK_SIZE:
        raise ValueError("generated ConvNeXt chunk size must remain exactly 8")
    if fixed_convnext_chunk_size != FIXED_CONVNEXT_CHUNK_SIZE:
        raise ValueError("fixed ConvNeXt chunk size must remain exactly 32")
    if checkpoint_generated_convnext is not True:
        raise ValueError(
            "generated ConvNeXt chunks require non-reentrant checkpointing"
        )

    # This is intentionally before *both* feature extractors.  The custom
    # all-gather backward plus DDP then propagates the global-support objective
    # to each rank's local generator output.
    feature_encoder.retain_generated_gradient(local_generated_images)
    global_generated = gather_generated_images(local_generated_images)

    mae_generated, mae_positive, mae_negative = _extract_mae_supports(
        feature_encoder.mae,
        global_generated,
        positive_images,
        fixed_negative_images,
    )
    conv_generated, conv_positive, conv_negative = extract_convnext_supports(
        feature_encoder.convnext,
        global_generated,
        positive_images,
        fixed_negative_images,
        generated_chunk_size=generated_convnext_chunk_size,
        fixed_chunk_size=fixed_convnext_chunk_size,
        checkpoint_generated=checkpoint_generated_convnext,
    )
    for label, features in (
        ("generated ConvNeXt", conv_generated),
        ("positive ConvNeXt", conv_positive),
        ("negative ConvNeXt", conv_negative),
    ):
        _assert_feature_dictionary(
            features, expected_count=CONVNEXT_FAMILY_COUNT, label=label
        )
        if tuple(features) != CONVNEXT_FAMILY_NAMES:
            raise RuntimeError(f"{label} order differs from the pinned schema")

    generated = merge_feature_families(mae_generated, conv_generated)
    positive = merge_feature_families(mae_positive, conv_positive)
    negative = merge_feature_families(mae_negative, conv_negative)
    for label, features in (
        ("generated composite", generated),
        ("positive composite", positive),
        ("negative composite", negative),
    ):
        _assert_feature_dictionary(
            features, expected_count=COMPOSITE_FAMILY_COUNT, label=label
        )
    if not (tuple(generated) == tuple(positive) == tuple(negative)):
        raise RuntimeError("composite family order differs across supports")
    return generated, positive, negative, global_generated


def _drifting_loss_from_complete_features(
    generated_features: Mapping[str, Tensor],
    positive_features: Mapping[str, Tensor],
    fixed_negative_features: Mapping[str, Tensor],
    *,
    expected_family_count: int,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Internal complete-support loss shared with the chunk-equivalence gate."""

    if len(generated_features) != expected_family_count:
        raise ValueError(
            f"the objective requires exactly {expected_family_count} feature families"
        )
    if not (
        tuple(generated_features)
        == tuple(positive_features)
        == tuple(fixed_negative_features)
    ):
        raise ValueError("generated/positive/negative family order must match exactly")

    total: Tensor | None = None
    statistics: dict[str, Tensor] = {}
    negative_weight_sum: Tensor | None = None
    for family_name in generated_features:
        generated_groups = token_features_to_groups(generated_features[family_name])
        positive_groups = token_features_to_groups(positive_features[family_name])
        negative_groups = token_features_to_groups(fixed_negative_features[family_name])
        negative_weight = torch.zeros(
            (negative_groups.shape[0], negative_groups.shape[1]),
            device=generated_groups.device,
            dtype=torch.float32,
        )
        # Feature extraction may be BF16 for the MAE, but the official distance,
        # affinity, scale and force geometry is always evaluated in FP32.
        with torch.autocast(device_type=generated_groups.device.type, enabled=False):
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
        family_zero = negative_weight.sum()
        negative_weight_sum = (
            family_zero
            if negative_weight_sum is None
            else negative_weight_sum + family_zero
        )

    if total is None or negative_weight_sum is None:
        raise RuntimeError("composite feature objective is empty")
    if bool(negative_weight_sum.detach() != 0):
        raise RuntimeError("fixed-negative weight must be exactly zero")
    statistics["loss"] = total.detach()
    statistics["support/fixed_negative_weight_sum"] = negative_weight_sum.detach()
    return total, statistics


def drifting_loss_from_complete_features(
    generated_features: Mapping[str, Tensor],
    positive_features: Mapping[str, Tensor],
    fixed_negative_features: Mapping[str, Tensor],
    *,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Call the official loss once per family after all 95 supports are complete."""

    if tuple(generated_features)[-CONVNEXT_FAMILY_COUNT:] != CONVNEXT_FAMILY_NAMES:
        raise ValueError(
            "the final 13 composite families must be the pinned ConvNeXt order"
        )
    return _drifting_loss_from_complete_features(
        generated_features,
        positive_features,
        fixed_negative_features,
        expected_family_count=COMPOSITE_FAMILY_COUNT,
        temperatures=temperatures,
    )


def run_convnext_chunk_equivalence_gate(
    feature_encoder: CelebA64ConvNextV2Features,
    *,
    device: torch.device | str,
    temperatures: Sequence[float] = (0.2, 0.05, 0.02),
) -> dict[str, float | int | bool]:
    """Compare full and checkpointed/chunked ConvNeXt features, loss and pixels.

    This deterministic two-image diagnostic performs no optimizer update and
    consumes no RNG state.  It is intended to run on every preflight rank after
    the exact weights load and before the generator is constructed.
    """

    feature_encoder._assert_frozen_eval()
    resolved_device = torch.device(device)
    values = (
        torch.arange(
            2 * 3 * 64 * 64,
            device=resolved_device,
            dtype=torch.float32,
        )
        .mul_(0.0017)
        .sin_()
        .reshape(2, 3, 64, 64)
    )
    generated_full = values.to(dtype=torch.bfloat16).requires_grad_(True)
    generated_chunked = values.to(dtype=torch.bfloat16).requires_grad_(True)
    positive = (
        torch.arange(
            3 * 3 * 64 * 64,
            device=resolved_device,
            dtype=torch.float32,
        )
        .mul_(0.0013)
        .cos_()
        .reshape(3, 3, 64, 64)
    )
    negative = (
        torch.arange(
            2 * 3 * 64 * 64,
            device=resolved_device,
            dtype=torch.float32,
        )
        .sub_(8_000)
        .mul_(0.0004)
        .tanh_()
        .reshape(2, 3, 64, 64)
    )

    full_generated_features = feature_encoder(generated_full)
    with torch.no_grad():
        full_positive_features = feature_encoder(positive)
        full_negative_features = feature_encoder(negative)
    chunked_generated_features = extract_generated_convnext_features(
        feature_encoder,
        generated_chunked,
        chunk_size=1,
        checkpoint_chunks=True,
    )
    chunked_positive_features = extract_fixed_convnext_features(
        feature_encoder,
        positive,
        chunk_size=1,
    )
    chunked_negative_features = extract_fixed_convnext_features(
        feature_encoder,
        negative,
        chunk_size=1,
    )

    maximum_feature_error = 0.0
    features_close = True
    for name in CONVNEXT_FAMILY_NAMES:
        full_value = full_generated_features[name]
        chunked_value = chunked_generated_features[name]
        maximum_feature_error = max(
            maximum_feature_error,
            float((full_value - chunked_value).abs().max().detach()),
            float(
                (full_positive_features[name] - chunked_positive_features[name])
                .abs()
                .max()
                .detach()
            ),
            float(
                (full_negative_features[name] - chunked_negative_features[name])
                .abs()
                .max()
                .detach()
            ),
        )
        features_close = features_close and torch.allclose(
            full_value, chunked_value, rtol=2e-5, atol=1e-5
        )
        features_close = features_close and torch.allclose(
            full_positive_features[name],
            chunked_positive_features[name],
            rtol=2e-5,
            atol=1e-5,
        )
        features_close = features_close and torch.allclose(
            full_negative_features[name],
            chunked_negative_features[name],
            rtol=2e-5,
            atol=1e-5,
        )

    full_loss, full_statistics = _drifting_loss_from_complete_features(
        full_generated_features,
        full_positive_features,
        full_negative_features,
        expected_family_count=CONVNEXT_FAMILY_COUNT,
        temperatures=temperatures,
    )
    chunked_loss, chunked_statistics = _drifting_loss_from_complete_features(
        chunked_generated_features,
        chunked_positive_features,
        chunked_negative_features,
        expected_family_count=CONVNEXT_FAMILY_COUNT,
        temperatures=temperatures,
    )
    full_gradient = torch.autograd.grad(full_loss, generated_full)[0]
    chunked_gradient = torch.autograd.grad(chunked_loss, generated_chunked)[0]
    loss_error = float((full_loss - chunked_loss).abs().detach())
    gradient_error = float((full_gradient - chunked_gradient).abs().max().detach())
    gradients_valid = (
        bool(torch.isfinite(full_gradient).all())
        and bool(torch.isfinite(chunked_gradient).all())
        and bool(full_gradient.float().abs().sum() > 0)
        and bool(chunked_gradient.float().abs().sum() > 0)
    )
    gradient_close = torch.allclose(
        full_gradient,
        chunked_gradient,
        rtol=1e-4,
        atol=1e-6,
    )
    zero_weights = (
        float(full_statistics["support/fixed_negative_weight_sum"]) == 0.0
        and float(chunked_statistics["support/fixed_negative_weight_sum"]) == 0.0
    )
    passed = bool(
        features_close
        and loss_error <= 1e-6
        and gradients_valid
        and gradient_close
        and zero_weights
    )
    result: dict[str, float | int | bool] = {
        "passed": passed,
        "family_count": CONVNEXT_FAMILY_COUNT,
        "generated_count": 2,
        "positive_count": 3,
        "negative_count": 2,
        "chunk_size": 1,
        "checkpoint_use_reentrant": False,
        "features_rtol": 2e-5,
        "features_atol": 1e-5,
        "maximum_absolute_feature_error": maximum_feature_error,
        "maximum_absolute_loss_error": loss_error,
        "gradient_rtol": 1e-4,
        "gradient_atol": 1e-6,
        "maximum_absolute_pixel_gradient_error": gradient_error,
        "full_pixel_gradient_l1": float(full_gradient.float().abs().sum().detach()),
        "chunked_pixel_gradient_l1": float(
            chunked_gradient.float().abs().sum().detach()
        ),
        "fixed_negative_weight_sum": float(
            chunked_statistics["support/fixed_negative_weight_sum"]
        ),
    }
    return result


def global_mae_convnext_gmd_loss(
    feature_encoder: FrozenMAEConvNextFeatures,
    local_generated_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Evaluate MAE-82 + ConvNeXt-13 GMD on complete global supports."""

    generated, positive, negative, global_generated = extract_composite_supports(
        feature_encoder,
        local_generated_images,
        positive_images,
        fixed_negative_images,
    )
    loss, statistics = drifting_loss_from_complete_features(
        generated,
        positive,
        negative,
        temperatures=temperatures,
    )
    statistics["support/generated_global"] = torch.tensor(
        global_generated.shape[0], device=loss.device, dtype=torch.float32
    )
    statistics["support/positive"] = torch.tensor(
        positive_images.shape[0], device=loss.device, dtype=torch.float32
    )
    statistics["support/fixed_negative"] = torch.tensor(
        fixed_negative_images.shape[0], device=loss.device, dtype=torch.float32
    )
    return loss, statistics


def composite_representation_signature(
    feature_encoder: FrozenMAEConvNextFeatures,
) -> dict[str, Any]:
    """Return the closed representation/chunk/runtime identity for checkpoints."""

    feature_encoder.assert_frozen_eval()
    return {
        "name": "celeba64_pixel_mae82_plus_convnextv2_base13",
        "family_count": COMPOSITE_FAMILY_COUNT,
        "family_order": [
            *(
                family["name"]
                for family in feature_encoder.mae.feature_schema()["families"]
            ),
            *CONVNEXT_FAMILY_NAMES,
        ],
        "mae": {
            "family_count": MAE_FAMILY_COUNT,
            "feature_profile": "nondegenerate",
            "compute": "surrounding CUDA/ROCm BF16 autocast",
            "fixed_support_autograd": FIXED_SUPPORT_AUTOGRAD,
        },
        "convnext": {
            "family_count": CONVNEXT_FAMILY_COUNT,
            "compute_dtype": "float32",
            "generated_chunk_size": GENERATED_CONVNEXT_CHUNK_SIZE,
            "fixed_chunk_size": FIXED_CONVNEXT_CHUNK_SIZE,
            "generated_checkpoint": CHECKPOINT_IMPLEMENTATION,
            "fixed_support_autograd": FIXED_SUPPORT_AUTOGRAD,
            "official_jax_grn_replacement_count": CONVNEXT_GRN_COUNT,
            "model_identity": feature_encoder.convnext_identity,
        },
        "cross_rank_generated": (
            "differentiable all_gather before MAE and ConvNeXt extraction"
        ),
        "chunk_aggregation": (
            "concatenate the complete sample support per family before loss"
        ),
        "loss_evaluation": "exactly one official_drifting_loss call per family",
        "fixed_negative_weight": NEGATIVE_WEIGHT,
    }


__all__ = [
    "CHECKPOINT_GENERATED_CONVNEXT",
    "CHECKPOINT_IMPLEMENTATION",
    "COMPOSITE_FAMILY_COUNT",
    "CONVNEXT_FAMILY_COUNT",
    "FIXED_CONVNEXT_CHUNK_SIZE",
    "FIXED_SUPPORT_AUTOGRAD",
    "FrozenMAEConvNextFeatures",
    "GENERATED_CONVNEXT_CHUNK_SIZE",
    "MAE_FAMILY_COUNT",
    "NEGATIVE_WEIGHT",
    "composite_representation_signature",
    "drifting_loss_from_complete_features",
    "extract_composite_supports",
    "global_mae_convnext_gmd_loss",
    "run_convnext_chunk_equivalence_gate",
]
