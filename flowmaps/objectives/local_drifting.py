"""Local two-time feature GMD for the CelebA MAE-82 + ConvNeXt-13 flow.

This experimental module is deliberately separate from the active CelebA
trainers.  It adapts the two finite-difference views in :mod:`flowmaps.objectives.wgf` to the
released normalized multi-temperature Drifting loss:

* a single scalar ``start_time = a`` is shared by the whole support;
* ``source = stopgrad(Phi(z, 0, a))`` defines one GMD teacher geometry;
* ``origin_student = Phi(z, 0, u)`` and
  ``local_student = Phi(source, a, u)`` are the only live branches;
* ``u = a + time_fraction * (1 - a)``.

For one feature family, let ``q`` be the official input scale and ``F`` the
dimensionless force produced by :func:`flowmaps.objectives.drifting.official_drifting_loss`
at the detached source features ``X``.  Each student feature view ``Y`` uses

    mean((((Y / q - X / q) / force_step) - F) ** 2).

Thus ``time_fraction`` has units of remaining normalized flow time, whereas
``force_step`` is the dimensionless GMD pseudo-step.  Experiments may couple
both values to one ``rho`` but this module keeps them explicit so that this
scientific choice cannot happen accidentally.  At ``force_step == 1`` and
``Y == X``, the value and semi-gradient are exactly those of the official
Drifting loss evaluated at ``X``.  No claim of endpoint equivalence is made
away from that local identity.

The feature-level API consumes complete dictionaries in the production
``[samples, tokens, dimensions]`` layout.  The optional image-level adapter
uses the existing sharded MAE-82 + ConvNeXt-13 extraction path, retains the
official zero weight for the fixed-negative bank, and computes the teacher
geometry exactly once per feature family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from flowmaps.objectives.drifting import (
    OFFICIAL_TEMPERATURES,
    official_drifting_loss,
    token_features_to_groups,
)

OFFICIAL_SCALE_FLOOR = 1e-3


class FixedLabelTwoTimeFlow(Protocol):
    def __call__(
        self,
        state: Tensor,
        noise_labels: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor: ...


@dataclass(frozen=True)
class HomogeneousTwoViewStates:
    """Detached source and two live students at one homogeneous time pair."""

    start_time: Tensor
    time_fraction: Tensor
    end_time: Tensor
    source: Tensor
    origin_student: Tensor
    local_student: Tensor


@dataclass(frozen=True)
class OfficialFeatureGMDTeacher:
    """Detached per-family geometry recovered from the official loss API."""

    forces: dict[str, Tensor]
    input_scales: dict[str, Tensor]
    group_losses: dict[str, Tensor]
    information: dict[str, dict[str, Tensor]]


def _broadcast_rank_zero(value: Tensor, *, enabled: bool) -> Tensor:
    if (
        enabled
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        value = value.clone()
        dist.broadcast(value, src=0)
    return value


def _unit_fraction(value: float | Tensor, like: Tensor, *, name: str) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if result.numel() != 1:
        raise ValueError(f"{name} must be one scalar shared by the complete batch")
    result = result.reshape(())
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if not bool((result > 0.0) & (result <= 1.0)):
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


def _homogeneous_start_time(value: float | Tensor, like: Tensor) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError("start_time must be a real scalar, not bool")
    result = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if result.numel() != 1:
        raise ValueError("start_time must be one scalar shared by the complete batch")
    result = result.reshape(())
    if not bool(torch.isfinite(result)):
        raise ValueError("start_time must be finite")
    if not bool((result >= 0.0) & (result < 1.0)):
        raise ValueError("start_time must lie in [0, 1)")
    return result


def _validate_flow_inputs(prior: Tensor, noise_labels: Tensor) -> None:
    if not isinstance(prior, Tensor):
        raise TypeError("prior must be a torch.Tensor")
    if (
        prior.ndim < 2
        or prior.shape[0] < 1
        or any(size < 1 for size in prior.shape[1:])
    ):
        raise ValueError("prior must have a non-empty batch and feature dimensions")
    if not prior.is_floating_point():
        raise TypeError("prior must be floating point")
    if not bool(torch.isfinite(prior).all()):
        raise ValueError("prior must contain only finite values")
    if not isinstance(noise_labels, Tensor):
        raise TypeError("noise_labels must be a torch.Tensor")
    if noise_labels.ndim != 2 or noise_labels.shape[0] != prior.shape[0]:
        raise ValueError("noise_labels must have shape [B, noise_coords]")
    if noise_labels.shape[1] < 1 or noise_labels.dtype != torch.int64:
        raise TypeError("noise_labels must contain int64 noise coordinates")
    if noise_labels.device != prior.device:
        raise ValueError("prior and noise_labels must be on the same device")


def _checked_flow_output(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError(f"{name} must preserve the prior shape and device")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def homogeneous_two_view_states(
    flow: FixedLabelTwoTimeFlow,
    prior: Tensor,
    noise_labels: Tensor,
    *,
    start_time: float | Tensor,
    time_fraction: float | Tensor,
    synchronize_distributed: bool = True,
) -> HomogeneousTwoViewStates:
    """Build the stopped source plus origin/local student views.

    ``start_time`` and ``time_fraction`` must each be scalar.  In particular,
    per-sample times are rejected because they would mix different
    distributions along the sample axis of one GMD support.  The prior is
    detached as well: gradients are intended only for the two student calls.
    In DDP, rank zero's scalars overwrite other-rank draws.  A trainer must
    therefore maintain and checkpoint one authoritative time RNG stream.
    """

    _validate_flow_inputs(prior, noise_labels)
    start = _broadcast_rank_zero(
        _homogeneous_start_time(start_time, prior),
        enabled=synchronize_distributed,
    )
    fraction = _broadcast_rank_zero(
        _unit_fraction(time_fraction, prior, name="time_fraction"),
        enabled=synchronize_distributed,
    )
    end = start + fraction * (1.0 - start)
    initial = prior.detach()

    # Do not let a stopped teacher forward seed autocast's weight cache with
    # detached BF16 parameter casts.  Reusing those casts in the live,
    # activation-checkpointed students makes checkpoint recomputation observe
    # different saved tensors at backward time.
    device_type = prior.device.type
    with (
        torch.no_grad(),
        torch.autocast(
            device_type,
            enabled=torch.is_autocast_enabled(device_type),
            dtype=torch.get_autocast_dtype(device_type),
            cache_enabled=False,
        ),
    ):
        source = _checked_flow_output(
            flow(initial, noise_labels, 0.0, start),
            initial,
            name="source",
        ).detach()
    origin = _checked_flow_output(
        flow(initial, noise_labels, 0.0, end),
        initial,
        name="origin_student",
    )
    local = _checked_flow_output(
        flow(source, noise_labels, start, end),
        initial,
        name="local_student",
    )
    return HomogeneousTwoViewStates(
        start_time=start.detach(),
        time_fraction=fraction.detach(),
        end_time=end.detach(),
        source=source,
        origin_student=origin,
        local_student=local,
    )


def _validate_feature_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [samples, tokens, dimensions]")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty token and feature dimensions")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_teacher_feature_dictionaries(
    source: Mapping[str, Tensor],
    positive: Mapping[str, Tensor],
    fixed_negative: Mapping[str, Tensor] | None,
) -> tuple[str, ...]:
    names = tuple(source)
    if not names:
        raise ValueError("at least one feature family is required")
    if tuple(positive) != names:
        raise ValueError(
            "positive_features must have the same ordered keys as source_features"
        )
    if fixed_negative is not None and tuple(fixed_negative) != names:
        raise ValueError(
            "fixed_negative_features must have the same ordered keys as source_features"
        )

    for family in names:
        tensors = (
            ("source_features", source[family]),
            ("positive_features", positive[family]),
        )
        for label, value in tensors:
            _validate_feature_tensor(f"{label}[{family!r}]", value)
        source_shape = source[family].shape
        for label, value in tensors[1:]:
            if value.shape[1:] != source_shape[1:]:
                raise ValueError(
                    f"{label}[{family!r}] must share source tokens and dimensions"
                )
            if value.device != source[family].device:
                raise ValueError(f"all {family!r} supports must be on one device")
        if fixed_negative is not None:
            negative = fixed_negative[family]
            _validate_feature_tensor(f"fixed_negative_features[{family!r}]", negative)
            if negative.shape[1:] != source_shape[1:]:
                raise ValueError(
                    f"fixed_negative_features[{family!r}] must share source tokens and dimensions"
                )
            if negative.device != source[family].device:
                raise ValueError(f"all {family!r} supports must be on one device")
    return names


def _validate_student_feature_dictionaries(
    source: Mapping[str, Tensor],
    origin: Mapping[str, Tensor],
    local: Mapping[str, Tensor],
    names: tuple[str, ...],
    source_sample_range: tuple[int, int] | None,
) -> tuple[int, int]:
    for label, features in (
        ("origin_student_features", origin),
        ("local_student_features", local),
    ):
        if tuple(features) != names:
            raise ValueError(
                f"{label} must have the same ordered keys as source_features"
            )

    first_source_count = source[names[0]].shape[0]
    if source_sample_range is None:
        start, stop = 0, first_source_count
    else:
        if (
            not isinstance(source_sample_range, tuple)
            or len(source_sample_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in source_sample_range
            )
        ):
            raise TypeError("source_sample_range must be a pair of Python integers")
        start, stop = source_sample_range
        if start < 0 or stop <= start or stop > first_source_count:
            raise ValueError(
                "source_sample_range must select a non-empty in-bounds interval"
            )
    selected_count = stop - start

    for family in names:
        source_value = source[family]
        if source_value.shape[0] != first_source_count:
            raise ValueError("all source feature families must share one sample count")
        expected_shape = (selected_count, *source_value.shape[1:])
        for label, features in (
            ("origin_student_features", origin),
            ("local_student_features", local),
        ):
            value = features[family]
            _validate_feature_tensor(f"{label}[{family!r}]", value)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{label}[{family!r}] must match selected source shape {expected_shape}"
                )
            if value.device != source_value.device:
                raise ValueError(f"all {family!r} supports must be on one device")
    return start, stop


def _official_source_force(
    source_groups: Tensor,
    positive_groups: Tensor,
    fixed_negative_groups: Tensor | None,
    *,
    temperatures: Sequence[float],
    scale_floor: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Recover the released dimensionless force through its public loss API.

    If ``L_g = mean((X/q - goal)**2)`` is the official per-group loss, then
    ``F = goal - X/q = -(N*D*q/2) * dL_g/dX``.  Taking the gradient of the
    sum of group losses therefore exposes exactly the stopped force without
    copying the release's distance/affinity equations into this module.
    """

    source_proxy = source_groups.detach().float().clone().requires_grad_(True)
    negative_weight = None
    if fixed_negative_groups is not None:
        negative_weight = torch.zeros(
            fixed_negative_groups.shape[:2],
            device=source_proxy.device,
            dtype=torch.float32,
        )
    with (
        torch.enable_grad(),
        torch.autocast(
            device_type=source_proxy.device.type,
            enabled=False,
        ),
    ):
        teacher_group_loss, information = official_drifting_loss(
            source_proxy,
            positive_groups.detach(),
            None if fixed_negative_groups is None else fixed_negative_groups.detach(),
            negative_weight=negative_weight,
            temperatures=temperatures,
            scale_floor=scale_floor,
            distributed_statistics=False,
        )
        (teacher_gradient,) = torch.autograd.grad(
            teacher_group_loss.sum(),
            source_proxy,
            create_graph=False,
            retain_graph=False,
        )

    dimensions = source_proxy.shape[2]
    samples = source_proxy.shape[1]
    input_scale = (information["scale"] / math.sqrt(dimensions)).clamp_min(scale_floor)
    force = (-0.5 * samples * dimensions * input_scale * teacher_gradient).detach()
    return (
        force,
        input_scale.detach(),
        teacher_group_loss.detach(),
        {name: value.detach() for name, value in information.items()},
    )


def official_feature_gmd_teacher(
    source_features: Mapping[str, Tensor],
    positive_features: Mapping[str, Tensor],
    *,
    fixed_negative_features: Mapping[str, Tensor] | None = None,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    scale_floor: float = OFFICIAL_SCALE_FLOOR,
) -> OfficialFeatureGMDTeacher:
    """Build one detached official GMD force field on a complete source support."""

    names = _validate_teacher_feature_dictionaries(
        source_features,
        positive_features,
        fixed_negative_features,
    )
    if isinstance(scale_floor, bool):
        raise TypeError("scale_floor must be a finite positive real number")
    try:
        scale_floor_value = float(scale_floor)
    except (TypeError, ValueError) as error:
        raise TypeError("scale_floor must be a finite positive real number") from error
    if not math.isfinite(scale_floor_value) or scale_floor_value <= 0.0:
        raise ValueError("scale_floor must be finite and positive")

    forces: dict[str, Tensor] = {}
    input_scales: dict[str, Tensor] = {}
    group_losses: dict[str, Tensor] = {}
    all_information: dict[str, dict[str, Tensor]] = {}
    for family in names:
        source_groups = (
            token_features_to_groups(source_features[family]).detach().float()
        )
        positive_groups = token_features_to_groups(positive_features[family]).detach()
        negative_groups = (
            None
            if fixed_negative_features is None
            else token_features_to_groups(fixed_negative_features[family]).detach()
        )
        force, input_scale, teacher_group_loss, information = _official_source_force(
            source_groups,
            positive_groups,
            negative_groups,
            temperatures=temperatures,
            scale_floor=scale_floor_value,
        )
        forces[family] = force
        input_scales[family] = input_scale
        group_losses[family] = teacher_group_loss
        all_information[family] = information
    return OfficialFeatureGMDTeacher(
        forces=forces,
        input_scales=input_scales,
        group_losses=group_losses,
        information=all_information,
    )


def two_view_local_feature_gmd_loss(
    source_features: Mapping[str, Tensor],
    origin_student_features: Mapping[str, Tensor],
    local_student_features: Mapping[str, Tensor],
    positive_features: Mapping[str, Tensor],
    *,
    fixed_negative_features: Mapping[str, Tensor] | None = None,
    force_step: float | Tensor,
    origin_weight: float = 0.5,
    local_weight: float = 0.5,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    scale_floor: float = OFFICIAL_SCALE_FLOOR,
    source_sample_range: tuple[int, int] | None = None,
    synchronize_distributed: bool = True,
    teacher: OfficialFeatureGMDTeacher | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a weighted origin/local semi-gradient feature-GMD loss.

    All dictionaries use ``[samples, tokens, dimensions]`` values.  Source,
    positive, and fixed-negative supports are forcibly detached.  Fixed
    negatives retain the production MAE+ConvNeXt convention: they participate
    in the official softmax support but have exactly zero post-affinity weight.
    The historical/default objective averages the two student losses through
    ``origin_weight=local_weight=0.5``.  Explicit weights allow causal
    origin-only, local-only, and graph-matched zero controls without changing
    either student's forward topology.  ``force_step`` never changes their
    relative weight.
    """

    weights = (origin_weight, local_weight)
    if any(isinstance(weight, bool) for weight in weights):
        raise ValueError("origin/local weights must be finite and non-negative")
    try:
        origin_weight = float(origin_weight)
        local_weight = float(local_weight)
    except (TypeError, ValueError) as error:
        raise TypeError("origin/local weights must be real numbers") from error
    weights = (origin_weight, local_weight)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("origin/local weights must be finite and non-negative")

    names = _validate_teacher_feature_dictionaries(
        source_features,
        positive_features,
        fixed_negative_features,
    )
    sample_start, sample_stop = _validate_student_feature_dictionaries(
        source_features,
        origin_student_features,
        local_student_features,
        names,
        source_sample_range,
    )
    first = source_features[names[0]]
    step = _broadcast_rank_zero(
        _unit_fraction(force_step, first, name="force_step"),
        enabled=synchronize_distributed,
    )
    if teacher is None:
        teacher = official_feature_gmd_teacher(
            source_features,
            positive_features,
            fixed_negative_features=fixed_negative_features,
            temperatures=temperatures,
            scale_floor=scale_floor,
        )
    elif not isinstance(teacher, OfficialFeatureGMDTeacher):
        raise TypeError("teacher must be an OfficialFeatureGMDTeacher")
    teacher_key_sets = (
        tuple(teacher.forces),
        tuple(teacher.input_scales),
        tuple(teacher.group_losses),
        tuple(teacher.information),
    )
    if any(keys != names for keys in teacher_key_sets):
        raise ValueError("teacher feature-family order must match source_features")

    origin_total: Tensor | None = None
    local_total: Tensor | None = None
    statistics: dict[str, Tensor] = {}
    for family in names:
        source_groups = (
            token_features_to_groups(source_features[family])
            .detach()
            .float()[:, sample_start:sample_stop]
        )
        origin_groups = token_features_to_groups(
            origin_student_features[family]
        ).float()
        local_groups = token_features_to_groups(local_student_features[family]).float()
        force = teacher.forces[family][:, sample_start:sample_stop]
        input_scale = teacher.input_scales[family]
        complete_source_shape = token_features_to_groups(source_features[family]).shape
        if teacher.forces[family].shape != complete_source_shape:
            raise ValueError(f"teacher force shape differs for family {family!r}")
        if input_scale.numel() != 1 or not bool(torch.isfinite(input_scale)):
            raise ValueError(f"teacher input scale is invalid for family {family!r}")
        if not bool(torch.isfinite(force).all()):
            raise ValueError(f"teacher force is non-finite for family {family!r}")

        origin_residual = (origin_groups - source_groups) / input_scale / step - force
        local_residual = (local_groups - source_groups) / input_scale / step - force
        origin_family = origin_residual.square().mean(dim=(-1, -2)).mean()
        local_family = local_residual.square().mean(dim=(-1, -2)).mean()
        origin_total = (
            origin_family if origin_total is None else origin_total + origin_family
        )
        local_total = (
            local_family if local_total is None else local_total + local_family
        )

        statistics[f"loss/source_geometry/{family}"] = teacher.group_losses[
            family
        ].mean()
        statistics[f"loss/origin/{family}"] = origin_family.detach()
        statistics[f"loss/local/{family}"] = local_family.detach()
        statistics[f"scale/input/{family}"] = input_scale
        statistics[f"force/rms_global/{family}"] = (
            teacher.forces[family].square().mean().sqrt()
        )
        statistics[f"support/fixed_negative_weight_sum/{family}"] = torch.zeros(
            (), device=source_groups.device, dtype=torch.float32
        )
        for key, value in teacher.information[family].items():
            statistics[f"official/{key}/{family}"] = value

    if origin_total is None or local_total is None:
        raise RuntimeError("feature objective is empty")
    total = origin_weight * origin_total + local_weight * local_total
    statistics["loss"] = total.detach()
    statistics["loss/origin"] = origin_total.detach()
    statistics["loss/local"] = local_total.detach()
    statistics["weight/origin"] = total.detach().new_tensor(origin_weight)
    statistics["weight/local"] = total.detach().new_tensor(local_weight)
    statistics["force_step"] = step.detach()
    statistics["support/source_sample_start"] = torch.tensor(
        sample_start, device=first.device, dtype=torch.float32
    )
    statistics["support/source_sample_stop"] = torch.tensor(
        sample_stop, device=first.device, dtype=torch.float32
    )
    return total, statistics


def global_sharded_mae_convnext_two_view_loss(
    feature_encoder,
    source_images: Tensor,
    origin_student_images: Tensor,
    local_student_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    force_step: float | Tensor,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Efficient MAE-82 + ConvNeXt-13 adapter for a future DDP trainer.

    The stopped source and fixed supports are encoded locally under
    ``no_grad`` and then gathered non-differentiably.  Consequently the
    official row/column softmax sees the complete global source support.  Only
    the two rank-local students are concatenated into a live feature-encoder
    call.  DDP's gradient average of their rank-local mean losses is therefore
    the same update as a monolithic global student loss when local batch sizes
    agree.  Positive/negative banks must be complete and identical on every
    rank, exactly as in the existing sharded GMD adapter.
    """

    # Keep heavyweight/private production imports out of the pure CPU module
    # import path.  They are reused here rather than reimplemented.
    from flowmaps.features.combined import _extract_mae_supports
    from flowmaps.features.convnext import (
        extract_convnext_supports,
        extract_generated_convnext_features,
        merge_feature_families,
    )
    from flowmaps.features.distributed import gather_features, world

    feature_encoder.assert_frozen_eval()
    if not (
        source_images.shape == origin_student_images.shape == local_student_images.shape
    ):
        raise ValueError(
            "source, origin, and local image views must have identical shapes"
        )
    if source_images.ndim != 4 or tuple(source_images.shape[1:]) != (3, 64, 64):
        raise ValueError("all generated image views must have shape [B,3,64,64]")
    if source_images.shape[0] < 1:
        raise ValueError("generated image views must have a non-empty batch")
    if (
        source_images.device != origin_student_images.device
        or source_images.device != local_student_images.device
    ):
        raise ValueError("all generated image views must be on one device")
    if (
        not origin_student_images.requires_grad
        or not local_student_images.requires_grad
    ):
        raise RuntimeError("origin and local student images must require gradients")

    local_samples = source_images.shape[0]
    size = world()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if positive_images.shape[0] % size or fixed_negative_images.shape[0] % size:
        raise ValueError("fixed support counts must be divisible by world size")

    def local_fixed(images: Tensor) -> Tensor:
        count = images.shape[0] // size
        if count < 1:
            raise ValueError("each rank must receive a non-empty fixed support")
        return images[rank * count : (rank + 1) * count].detach()

    local_positive = local_fixed(positive_images)
    local_negative = local_fixed(fixed_negative_images)
    with torch.no_grad():
        source_mae, positive_mae, negative_mae = _extract_mae_supports(
            feature_encoder.mae,
            source_images.detach(),
            local_positive,
            local_negative,
        )
        source_convnext, positive_convnext, negative_convnext = (
            extract_convnext_supports(
                feature_encoder.convnext,
                source_images.detach(),
                local_positive,
                local_negative,
                generated_chunk_size=8,
                fixed_chunk_size=32,
                checkpoint_generated=False,
            )
        )
        source_local, positive_local, negative_local = (
            merge_feature_families(mae_support, convnext_support)
            for mae_support, convnext_support in zip(
                (source_mae, positive_mae, negative_mae),
                (source_convnext, positive_convnext, negative_convnext),
                strict=True,
            )
        )
    source = gather_features(source_local, differentiable=False)
    positive = gather_features(positive_local, differentiable=False)
    negative = gather_features(negative_local, differentiable=False)

    # Finish all 95 source-geometry softmax/force computations before keeping
    # either live student encoder graph in memory.
    teacher = official_feature_gmd_teacher(
        source,
        positive,
        fixed_negative_features=negative,
        temperatures=temperatures,
    )

    combined_students = torch.cat(
        (origin_student_images, local_student_images),
        dim=0,
    )
    feature_encoder.retain_generated_gradient(combined_students)
    student_mae = feature_encoder.mae.encode_features(combined_students)
    student_convnext = extract_generated_convnext_features(
        feature_encoder.convnext,
        combined_students,
        chunk_size=8,
        checkpoint_chunks=True,
    )
    students = merge_feature_families(student_mae, student_convnext)
    origin = {family: value[:local_samples] for family, value in students.items()}
    local = {family: value[local_samples:] for family, value in students.items()}

    loss, statistics = two_view_local_feature_gmd_loss(
        source,
        origin,
        local,
        positive,
        fixed_negative_features=negative,
        force_step=force_step,
        temperatures=temperatures,
        source_sample_range=(rank * local_samples, (rank + 1) * local_samples),
        teacher=teacher,
    )
    global_samples = local_samples * size
    for key, count in (
        ("support/source_global", global_samples),
        ("support/origin_global", global_samples),
        ("support/local_global", global_samples),
        ("support/positive", positive_images.shape[0]),
        ("support/fixed_negative", fixed_negative_images.shape[0]),
    ):
        statistics[key] = torch.tensor(count, device=loss.device, dtype=torch.float32)
    return loss, statistics


__all__ = [
    "HomogeneousTwoViewStates",
    "OFFICIAL_SCALE_FLOOR",
    "OfficialFeatureGMDTeacher",
    "global_sharded_mae_convnext_two_view_loss",
    "homogeneous_two_view_states",
    "official_feature_gmd_teacher",
    "two_view_local_feature_gmd_loss",
]
