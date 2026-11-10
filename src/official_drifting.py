"""PyTorch implementation of the released Drifting Models image loss.

The implementation follows ``drift_loss.py`` from the official JAX release
(``lambertae/drifting``, commit ``accd0cf09c33b70892d33941d2a287ca86cb92e1``).
It deliberately lives beside the historical Gaussian-KDE/WGF estimator in
``src.wgf``: the two objectives are related, but they are not the same finite
batch algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


OFFICIAL_TEMPERATURES = (0.02, 0.05, 0.2)


def _distributed_average(values: Tensor, enabled: bool) -> Tensor:
    if not enabled:
        return values
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed drifting statistics require an initialized process group")
    reduced = values.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()


def _validate_feature_batch(name: str, value: Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [groups, samples, dimensions]")
    if value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError(f"{name} sample and feature dimensions must be non-empty")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _weights(
    value: Tensor | None,
    *,
    groups: int,
    samples: int,
    like: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return torch.ones((groups, samples), device=like.device, dtype=torch.float32)
    if value.shape != (groups, samples):
        raise ValueError(f"{name} must have shape {(groups, samples)}")
    return value.to(device=like.device, dtype=torch.float32)


def official_drifting_loss(
    generated: Tensor,
    positive: Tensor,
    fixed_negative: Tensor | None = None,
    *,
    generated_weight: Tensor | None = None,
    positive_weight: Tensor | None = None,
    negative_weight: Tensor | None = None,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    distance_eps: float = 1e-8,
    scale_floor: float = 1e-3,
    affinity_floor: float = 1e-6,
    force_floor: float = 1e-8,
    distributed_statistics: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the official normalized drifting loss for one feature family.

    Parameters use the official layout ``[groups, samples, dimensions]``.
    A group is normally a class/feature-location pair.  In unconditional
    CelebA training, the groups are feature locations and the sample axis is
    the minibatch.  ``generated`` retains gradients; every target and every
    statistic used to construct the drifting goal is stop-gradient.

    The returned loss has one value per group, matching the JAX release.  A
    caller combining feature families should average this vector, then sum the
    family losses.
    """

    _validate_feature_batch("generated", generated)
    _validate_feature_batch("positive", positive)
    if generated.shape[0] != positive.shape[0] or generated.shape[2] != positive.shape[2]:
        raise ValueError("generated and positive must share groups and feature dimensions")

    groups, generated_count, dimensions = generated.shape
    positive_count = positive.shape[1]
    if fixed_negative is None:
        fixed_negative = generated.new_empty((groups, 0, dimensions))
    else:
        _validate_feature_batch("fixed_negative", fixed_negative)
        if (
            fixed_negative.shape[0] != groups
            or fixed_negative.shape[2] != dimensions
        ):
            raise ValueError(
                "fixed_negative must share generated groups and feature dimensions"
            )
    negative_count = fixed_negative.shape[1]

    temperatures = tuple(float(value) for value in temperatures)
    if not temperatures or any(not math.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError("temperatures must be a non-empty sequence of positive finite values")
    if min(distance_eps, scale_floor, affinity_floor, force_floor) <= 0:
        raise ValueError("all numerical floors must be positive")

    # The released implementation evaluates all geometry in float32 even when
    # the generator or feature encoder uses bf16.
    generated_f32 = generated.float()
    old_generated = generated_f32.detach()
    positive_f32 = positive.detach().to(device=generated.device, dtype=torch.float32)
    negative_f32 = fixed_negative.detach().to(device=generated.device, dtype=torch.float32)

    gen_weight = _weights(
        generated_weight,
        groups=groups,
        samples=generated_count,
        like=generated,
        name="generated_weight",
    )
    neg_weight = _weights(
        negative_weight,
        groups=groups,
        samples=negative_count,
        like=generated,
        name="negative_weight",
    )
    pos_weight = _weights(
        positive_weight,
        groups=groups,
        samples=positive_count,
        like=generated,
        name="positive_weight",
    )

    with torch.no_grad():
        targets = torch.cat((old_generated, negative_f32, positive_f32), dim=1)
        target_weight = torch.cat((gen_weight, neg_weight, pos_weight), dim=1)

        # ``torch.cdist`` and the explicit sqrt formulation in the JAX code
        # agree away from zero.  Clamp squared distances first to reproduce the
        # JAX epsilon for exact duplicates before applying the self mask.
        dot = torch.einsum("gnd,gmd->gnm", old_generated, targets)
        old_norm = old_generated.square().sum(dim=-1)
        target_norm = targets.square().sum(dim=-1)
        squared_distance = (
            old_norm[:, :, None] + target_norm[:, None, :] - 2.0 * dot
        )
        distance = squared_distance.clamp_min(distance_eps).sqrt()

        weighted_distance = distance * target_weight[:, None, :]
        scale_terms = _distributed_average(
            torch.stack((weighted_distance.mean(), target_weight.mean())),
            distributed_statistics,
        )
        scale = scale_terms[0] / scale_terms[1].clamp_min(force_floor)
        scale_inputs = (scale / math.sqrt(dimensions)).clamp_min(scale_floor)
        old_scaled = old_generated / scale_inputs
        targets_scaled = targets / scale_inputs
        normalized_distance = distance / scale.clamp_min(scale_floor)

        self_mask = torch.eye(
            generated_count,
            device=generated.device,
            dtype=torch.bool,
        )
        normalized_distance[:, :, :generated_count].add_(
            self_mask.to(normalized_distance.dtype) * 100.0
        )

        forces: list[Tensor] = []
        local_force_energies: list[Tensor] = []
        info: dict[str, Tensor] = {"scale": scale}
        split = generated_count + negative_count
        for temperature in temperatures:
            logits = -normalized_distance / temperature
            row_affinity = logits.softmax(dim=-1)
            column_affinity = logits.softmax(dim=-2)
            affinity = (row_affinity * column_affinity).clamp_min(affinity_floor).sqrt()
            affinity = affinity * target_weight[:, None, :]

            negative_affinity = affinity[:, :, :split]
            positive_affinity = affinity[:, :, split:]
            negative_coeff = -negative_affinity * positive_affinity.sum(
                dim=-1, keepdim=True
            )
            positive_coeff = positive_affinity * negative_affinity.sum(
                dim=-1, keepdim=True
            )
            coefficients = torch.cat((negative_coeff, positive_coeff), dim=-1)
            force_at_temperature = torch.einsum(
                "gnm,gmd->gnd", coefficients, targets_scaled
            )
            force_at_temperature = force_at_temperature - coefficients.sum(
                dim=-1, keepdim=True
            ) * old_scaled

            forces.append(force_at_temperature)
            local_force_energies.append(force_at_temperature.square().mean())

        force_energies = _distributed_average(
            torch.stack(local_force_energies),
            distributed_statistics,
        )
        force = torch.zeros_like(old_scaled)
        for temperature, force_at_temperature, force_energy in zip(
            temperatures,
            forces,
            force_energies,
            strict=True,
        ):
            info[f"force_energy_{temperature:g}"] = force_energy
            force.add_(
                force_at_temperature / force_energy.clamp_min(force_floor).sqrt()
            )

        goal_scaled = old_scaled + force

    difference = generated_f32 / scale_inputs - goal_scaled
    loss = difference.square().mean(dim=(-1, -2))
    return loss, {key: value.detach() for key, value in info.items()}


def feature_map_to_groups(feature: Tensor) -> Tensor:
    """Convert ``[samples, channels, height, width]`` to official groups."""

    if feature.ndim != 4:
        raise ValueError("feature map must have shape [samples, channels, height, width]")
    return feature.flatten(2).permute(2, 0, 1).contiguous()


def token_features_to_groups(feature: Tensor) -> Tensor:
    """Convert ``[samples, tokens, dimensions]`` to official groups."""

    if feature.ndim != 3:
        raise ValueError("token features must have shape [samples, tokens, dimensions]")
    return feature.permute(1, 0, 2).contiguous()


def summed_feature_drifting_loss(
    generated_features: Mapping[str, Tensor],
    positive_features: Mapping[str, Tensor],
    *,
    fixed_negative_features: Mapping[str, Tensor] | None = None,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    distributed_statistics: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum official drifting losses over matching token feature dictionaries."""

    if not generated_features:
        raise ValueError("at least one generated feature family is required")
    if generated_features.keys() != positive_features.keys():
        raise ValueError("generated and positive feature dictionaries must have identical keys")
    if (
        fixed_negative_features is not None
        and generated_features.keys() != fixed_negative_features.keys()
    ):
        raise ValueError(
            "generated and fixed-negative feature dictionaries must have identical keys"
        )

    total: Tensor | None = None
    statistics: dict[str, Tensor] = {}
    for name in generated_features:
        generated = token_features_to_groups(generated_features[name])
        positive = token_features_to_groups(positive_features[name])
        fixed_negative = (
            None
            if fixed_negative_features is None
            else token_features_to_groups(fixed_negative_features[name])
        )
        family_loss, family_info = official_drifting_loss(
            generated,
            positive,
            fixed_negative,
            temperatures=temperatures,
            distributed_statistics=distributed_statistics,
        )
        mean_loss = family_loss.mean()
        total = mean_loss if total is None else total + mean_loss
        statistics[f"loss/{name}"] = mean_loss.detach()
        statistics[f"scale/{name}"] = family_info["scale"]
        for key, value in family_info.items():
            if key.startswith("force_energy_"):
                statistics[f"{key}/{name}"] = value

    assert total is not None
    statistics["loss"] = total.detach()
    return total, statistics


def raw_latent_pyramid_features(latents: Tensor) -> dict[str, Tensor]:
    """Build a training-free spatial baseline for ``4x8x8`` VAE latents.

    This is useful as a controlled bridge between the repository's historical
    single 256-D KDE and the released multi-location loss.  It is not a
    replacement for the frozen learned encoder used by the full image recipe.
    Every dictionary value follows ``[samples, tokens, dimensions]``.
    """

    if latents.ndim == 2:
        if latents.shape[1] != 4 * 8 * 8:
            raise ValueError("flattened CelebA latents must contain 256 values")
        spatial = latents.reshape(latents.shape[0], 4, 8, 8)
    elif latents.ndim == 4 and latents.shape[1:] == (4, 8, 8):
        spatial = latents
    else:
        raise ValueError("CelebA latents must have shape [B,256] or [B,4,8,8]")

    result = {
        "global": spatial.flatten(1).unsqueeze(1),
        "latent_spatial": spatial.flatten(2).transpose(1, 2),
        "latent_channel_rms": spatial.square()
        .mean(dim=(-2, -1))
        .add(1e-6)
        .sqrt()
        .unsqueeze(1),
    }
    for patch in (2, 4):
        mean = F.avg_pool2d(spatial, kernel_size=patch, stride=patch)
        second_moment = F.avg_pool2d(spatial.square(), kernel_size=patch, stride=patch)
        std = (second_moment - mean.square()).clamp_min(1e-6).sqrt()
        result[f"latent_mean_{patch}"] = mean.flatten(2).transpose(1, 2)
        result[f"latent_std_{patch}"] = std.flatten(2).transpose(1, 2)
    return result


__all__ = [
    "OFFICIAL_TEMPERATURES",
    "feature_map_to_groups",
    "official_drifting_loss",
    "raw_latent_pyramid_features",
    "summed_feature_drifting_loss",
    "token_features_to_groups",
]
