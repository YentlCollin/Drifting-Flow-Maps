"""Training mechanics for the calibrated local CelebA Flow-GMD pilot.

This module is intentionally independent of the active one-step and structural
trainers.  It contains only the trainable flow boundary, time-pair sampling,
loss composition, and inference composition needed by the local feature-GMD
experiment.  The heavyweight MAE-82 + ConvNeXt-13 objective remains in
``flowmaps.objectives.local_drifting``.

For one pair ``(a, rho)``, the physical endpoint is

    u = a + rho * (1 - a),

but the calibrated feature-space pseudo-secant uses ``force_step = rho``.
These quantities are deliberately represented by different fields.  The
source state and every official-GMD force reconstructed from it are detached.
All learned flow calls made during one update live inside
``LocalTwoViewTrainingWrapper.forward`` so this wrapper can be the sole DDP
boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SUPPORTED_EVALUATION_NFES = (1, 2, 4, 8, 16)
DEFAULT_E10_DELTA = 0.05


@dataclass(frozen=True)
class TimePairGrid:
    """An ordered categorical distribution over homogeneous ``(a, rho)`` pairs."""

    pairs: tuple[tuple[float, float], ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("time-pair grid must not be empty")
        if len(self.pairs) != len(self.probabilities):
            raise ValueError("time pairs and probabilities must have equal lengths")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("time-pair grid must not contain duplicates")
        for pair in self.pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("every time pair must be a tuple (a, rho)")
            a, rho = pair
            if any(isinstance(value, bool) for value in pair):
                raise TypeError("time pairs must be real-valued, not bool")
            if not math.isfinite(float(a)) or not 0.0 <= float(a) < 1.0:
                raise ValueError("every start time a must lie in [0, 1)")
            if not math.isfinite(float(rho)) or not 0.0 < float(rho) <= 1.0:
                raise ValueError("every pseudo-step rho must lie in (0, 1]")
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in self.probabilities
        ):
            raise ValueError("pair probabilities must be finite and non-negative")
        if not any(float(value) > 0.0 for value in self.probabilities):
            raise ValueError("at least one pair probability must be positive")
        if not math.isclose(
            math.fsum(float(value) for value in self.probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("pair probabilities must sum to one")

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs": [[a, rho] for a, rho in self.pairs],
            "probabilities": list(self.probabilities),
            "sampling": "one categorical pair shared by the global support",
        }


def cartesian_time_pair_grid(
    start_times: Sequence[float],
    rhos: Sequence[float],
    probabilities: Sequence[float] | None = None,
) -> TimePairGrid:
    """Build a row-major Cartesian grid, or a uniform distribution by default."""

    starts = tuple(float(value) for value in start_times)
    steps = tuple(float(value) for value in rhos)
    if not starts or not steps:
        raise ValueError("start-time and rho grids must both be non-empty")
    pairs = tuple((a, rho) for a in starts for rho in steps)
    if probabilities is None:
        probability = 1.0 / len(pairs)
        weights = tuple(probability for _ in pairs)
        # Make the finite representation sum exactly as accepted by fsum.
        weights = (*weights[:-1], 1.0 - math.fsum(weights[:-1]))
    else:
        weights = tuple(float(value) for value in probabilities)
    return TimePairGrid(pairs=pairs, probabilities=weights)


def factorized_time_pair_grid(
    start_times: Sequence[float],
    rhos: Sequence[float],
    start_probabilities: Sequence[float],
    rho_probabilities: Sequence[float],
) -> TimePairGrid:
    """Build ``p(a,rho)=p(a)*p(rho)`` in row-major Cartesian order."""

    starts = tuple(float(value) for value in start_times)
    steps = tuple(float(value) for value in rhos)
    start_weights = tuple(float(value) for value in start_probabilities)
    step_weights = tuple(float(value) for value in rho_probabilities)
    if len(starts) != len(start_weights) or len(steps) != len(step_weights):
        raise ValueError("factor probabilities must match their respective grids")
    # Reuse TimePairGrid's strict probability/domain validation for each
    # marginal before multiplying them.
    if not starts or not steps:
        raise ValueError("factorized grids must both be non-empty")
    if any(value < 0.0 or not math.isfinite(value) for value in start_weights):
        raise ValueError("start probabilities must be finite and non-negative")
    if any(value < 0.0 or not math.isfinite(value) for value in step_weights):
        raise ValueError("rho probabilities must be finite and non-negative")
    if not math.isclose(math.fsum(start_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("start probabilities must sum to one")
    if not math.isclose(math.fsum(step_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rho probabilities must sum to one")
    probabilities = tuple(
        start_weight * step_weight
        for start_weight in start_weights
        for step_weight in step_weights
    )
    # Absorb the final floating-point sum residue without changing ordering.
    probabilities = (
        *probabilities[:-1],
        1.0 - math.fsum(probabilities[:-1]),
    )
    return TimePairGrid(
        pairs=tuple((a, rho) for a in starts for rho in steps),
        probabilities=probabilities,
    )


def explicit_time_pair_grid(
    pairs: Sequence[tuple[float, float]],
    probabilities: Sequence[float] | None = None,
) -> TimePairGrid:
    """Build an explicitly ordered grid, retaining that order in checkpoints."""

    canonical = tuple((float(a), float(rho)) for a, rho in pairs)
    if not canonical:
        raise ValueError("explicit time-pair grid must not be empty")
    if probabilities is None:
        probability = 1.0 / len(canonical)
        weights = tuple(probability for _ in canonical)
        weights = (*weights[:-1], 1.0 - math.fsum(weights[:-1]))
    else:
        weights = tuple(float(value) for value in probabilities)
    return TimePairGrid(canonical, weights)


def sample_time_pair(
    grid: TimePairGrid,
    generator: torch.Generator,
) -> tuple[int, float, float]:
    """Consume exactly one categorical draw from a checkpointable CPU RNG."""

    if not isinstance(grid, TimePairGrid):
        raise TypeError("grid must be a TimePairGrid")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if generator.device.type != "cpu":
        raise ValueError("time-pair RNG must be a CPU generator")
    probabilities = torch.tensor(grid.probabilities, dtype=torch.float64)
    index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    a, rho = grid.pairs[index]
    return index, a, rho


class LocalTwoViewOutputs(NamedTuple):
    """Outputs of one DDP-wrapped local Flow-GMD training graph."""

    terminal: Tensor
    source: Tensor
    origin_student: Tensor
    local_student: Tensor
    e10: Tensor
    end_time: Tensor


class LocalTwoViewTrainingWrapper(nn.Module):
    """Sole DDP boundary containing every learned flow call for one update.

    ``enable_e10`` controls graph topology for the entire run.  If enabled,
    ``e10`` is the unweighted Eulerian consistency MSE; a trainer coefficient
    of 10 therefore reproduces the established ``E10`` convention.  If it is
    disabled, the returned scalar is an exact graph-connected zero and no
    additional flow evaluations are performed.
    """

    def __init__(
        self,
        flow: nn.Module,
        *,
        enable_e10: bool = False,
        e10_delta: float = DEFAULT_E10_DELTA,
    ) -> None:
        super().__init__()
        if not isinstance(flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        if not isinstance(enable_e10, bool):
            raise TypeError("enable_e10 must be bool")
        if (
            isinstance(e10_delta, bool)
            or not math.isfinite(float(e10_delta))
            or not 0.0 < float(e10_delta) < 1.0
        ):
            raise ValueError("e10_delta must lie strictly between zero and one")
        self.flow = flow
        self.enable_e10 = enable_e10
        self.e10_delta = float(e10_delta)

    @staticmethod
    def _pair(
        start_time: float,
        rho: float,
        like: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        start = torch.as_tensor(start_time, device=like.device, dtype=torch.float32)
        step = torch.as_tensor(rho, device=like.device, dtype=torch.float32)
        if start.numel() != 1 or step.numel() != 1:
            raise ValueError("one homogeneous scalar (a, rho) pair is required")
        if not bool(torch.isfinite(start)) or not bool(torch.isfinite(step)):
            raise ValueError("a and rho must be finite")
        if not bool((start >= 0.0) & (start < 1.0)):
            raise ValueError("a must lie in [0, 1)")
        if not bool((step > 0.0) & (step <= 1.0)):
            raise ValueError("rho must lie in (0, 1]")
        end = start + step * (1.0 - start)
        return start.reshape(()), step.reshape(()), end.reshape(())

    def _stopped_flow(
        self,
        state: Tensor,
        labels: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor:
        # Avoid populating autocast's weight cache from a no-grad forward.  A
        # cached detached BF16 cast can conflict with activation-checkpoint
        # recomputation in the live branches.
        device_type = state.device.type
        with (
            torch.no_grad(),
            torch.autocast(
                device_type,
                enabled=torch.is_autocast_enabled(device_type),
                dtype=torch.get_autocast_dtype(device_type),
                cache_enabled=False,
            ),
        ):
            return self.flow(state, labels, start_time, end_time).detach()

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        start_time: float,
        rho: float,
    ) -> LocalTwoViewOutputs:
        if not isinstance(prior, Tensor) or not prior.is_floating_point():
            raise TypeError("prior must be a floating-point Tensor")
        if prior.ndim < 2 or prior.shape[0] < 1:
            raise ValueError("prior must contain a non-empty batch")
        if (
            not isinstance(noise_labels, Tensor)
            or noise_labels.dtype != torch.int64
            or noise_labels.ndim != 2
            or noise_labels.shape[0] != prior.shape[0]
            or noise_labels.device != prior.device
        ):
            raise ValueError("noise_labels must be int64 [B,C] beside prior")
        start, step, end = self._pair(start_time, rho, prior)
        if self.enable_e10 and not bool(start + self.e10_delta < 1.0):
            raise ValueError("E10 requires a + e10_delta < 1")

        initial = prior.detach()
        # Complete the detached branch before any live branch.  Besides making
        # the semigradient boundary explicit, this keeps a no-grad autocast
        # weight cast from interfering with live checkpoint recomputation.
        source = self._stopped_flow(initial, noise_labels, 0.0, start)
        terminal = self.flow(initial, noise_labels, 0.0, 1.0)

        # Keep this live call explicit even when rho=1 makes it numerically
        # equal to ``terminal``.  A constant forward/autograd topology lets
        # DDP use static buckets and preserves exact continuation on restart.
        origin = self.flow(initial, noise_labels, 0.0, end)
        local = self.flow(source, noise_labels, start, end)

        if self.enable_e10:
            local_end = start + self.e10_delta
            local_state = self.flow(source, noise_labels, start, local_end)
            # This stopped call also remains explicit at rho=1 so every update
            # in an E10-enabled run executes one invariant forward topology.
            direct_teacher = self._stopped_flow(source, noise_labels, start, 1.0)
            shifted = self.flow(
                local_state,
                noise_labels,
                local_end,
                1.0,
            )
            e10 = F.mse_loss(shifted.float(), direct_teacher.float())
        else:
            e10 = terminal.float().sum() * 0.0

        return LocalTwoViewOutputs(
            terminal=terminal,
            source=source.detach(),
            origin_student=origin,
            local_student=local,
            e10=e10,
            end_time=end.detach(),
        )


def unwrap_local_training(module: nn.Module) -> LocalTwoViewTrainingWrapper:
    """Return the wrapper directly or from exactly one DDP layer."""

    if isinstance(module, LocalTwoViewTrainingWrapper):
        return module
    if isinstance(module, nn.parallel.DistributedDataParallel) and isinstance(
        module.module, LocalTwoViewTrainingWrapper
    ):
        return module.module
    raise TypeError(
        "module must be LocalTwoViewTrainingWrapper or DDP[LocalTwoViewTrainingWrapper]"
    )


def combine_training_losses(
    boundary_gmd: Tensor,
    local_two_view_gmd: Tensor,
    e10: Tensor,
    *,
    lambda_local: float,
    lambda_e10: float,
) -> Tensor:
    """Return ``boundary + lambda_local*local + lambda_e10*E`` exactly."""

    terms = (boundary_gmd, local_two_view_gmd, e10)
    if any(not isinstance(term, Tensor) or term.numel() != 1 for term in terms):
        raise ValueError("all objective terms must be scalar tensors")
    if any(term.device != boundary_gmd.device for term in terms[1:]):
        raise ValueError("all objective terms must be on one device")
    weights = (lambda_local, lambda_e10)
    if any(
        isinstance(weight, bool)
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
        for weight in weights
    ):
        raise ValueError("objective weights must be finite and non-negative")
    result = (
        boundary_gmd
        + float(lambda_local) * local_two_view_gmd
        + float(lambda_e10) * e10
    )
    if not bool(torch.isfinite(result.detach())):
        raise FloatingPointError("combined Flow-GMD objective is non-finite")
    return result


def global_sharded_boundary_and_local_two_view_loss(
    feature_encoder,
    terminal_images: Tensor,
    source_images: Tensor,
    origin_student_images: Tensor,
    local_student_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    force_step: float | Tensor,
    origin_weight: float = 0.5,
    local_weight: float = 0.5,
    temperatures: Sequence[float],
) -> tuple[Tensor, dict[str, Tensor], Tensor, dict[str, Tensor]]:
    """Compute terminal and local losses while encoding fixed supports once.

    The positive and fixed-negative shards, plus the detached source shard,
    share one MAE call group and one ConvNeXt call group.  Their complete
    features and the detached official-GMD source teacher are then reused by
    both objectives.  The terminal features use differentiable all-gather;
    origin/local student features stay rank-local, whose mean gradients become
    the monolithic global update after DDP averaging.

    This function contains no learned-flow call.  It therefore does not widen
    the sole trainable DDP boundary established by
    :class:`LocalTwoViewTrainingWrapper`.
    """

    view_weights = (origin_weight, local_weight)
    if any(isinstance(weight, bool) for weight in view_weights):
        raise ValueError("origin/local weights must be finite and non-negative")
    try:
        view_weights = tuple(float(weight) for weight in view_weights)
    except (TypeError, ValueError) as error:
        raise TypeError("origin/local weights must be real numbers") from error
    if any(not math.isfinite(weight) or weight < 0.0 for weight in view_weights):
        raise ValueError("origin/local weights must be finite and non-negative")
    origin_weight, local_weight = view_weights

    # Heavy production dependencies remain lazy so pair/sampler/checkpoint CPU
    # tests do not import the ConvNeXt runtime.
    import torch.distributed as dist

    from flowmaps.features.combined import (
        _extract_mae_supports,
        drifting_loss_from_complete_features,
    )
    from flowmaps.features.convnext import (
        extract_convnext_supports,
        extract_generated_convnext_features,
        merge_feature_families,
    )
    from flowmaps.features.distributed import gather_features, world
    from flowmaps.objectives.local_drifting import (
        official_feature_gmd_teacher,
        two_view_local_feature_gmd_loss,
    )

    feature_encoder.assert_frozen_eval()
    generated_views = (
        terminal_images,
        source_images,
        origin_student_images,
        local_student_images,
    )
    if any(
        not isinstance(images, Tensor)
        or images.ndim != 4
        or tuple(images.shape[1:]) != (3, 64, 64)
        for images in (*generated_views, positive_images, fixed_negative_images)
    ):
        raise ValueError("all image supports must use NCHW CelebA-64 RGB layout")
    if not all(images.shape == source_images.shape for images in generated_views):
        raise ValueError(
            "terminal/source/origin/local views must have identical shapes"
        )
    if source_images.shape[0] < 1:
        raise ValueError("generated views must contain a non-empty local shard")
    if source_images.requires_grad or source_images.grad_fn is not None:
        raise RuntimeError("local source images must be fully detached")
    if any(
        not images.requires_grad
        for images in (terminal_images, origin_student_images, local_student_images)
    ):
        raise RuntimeError("terminal and both student views must require gradients")
    if any(images.device != source_images.device for images in generated_views):
        raise ValueError("all generated views must share one device")

    size = world()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if positive_images.shape[0] % size or fixed_negative_images.shape[0] % size:
        raise ValueError("fixed support counts must divide the distributed world size")

    def local_fixed(images: Tensor) -> Tensor:
        count = images.shape[0] // size
        if count < 1:
            raise ValueError("every rank must receive a non-empty fixed support")
        return images[rank * count : (rank + 1) * count].detach()

    local_positive = local_fixed(positive_images)
    local_negative = local_fixed(fixed_negative_images)
    # Finish all fixed/source feature work and teacher-force reconstruction
    # before retaining any live frozen-encoder activation graph.
    with torch.no_grad():
        source_mae, positive_mae, negative_mae = _extract_mae_supports(
            feature_encoder.mae,
            source_images,
            local_positive,
            local_negative,
        )
        source_convnext, positive_convnext, negative_convnext = (
            extract_convnext_supports(
                feature_encoder.convnext,
                source_images,
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
    teacher = official_feature_gmd_teacher(
        source,
        positive,
        fixed_negative_features=negative,
        temperatures=temperatures,
    )

    def live_features(images: Tensor) -> dict[str, Tensor]:
        mae_features = feature_encoder.mae.encode_features(images)
        convnext_features = extract_generated_convnext_features(
            feature_encoder.convnext,
            images,
            chunk_size=8,
            checkpoint_chunks=True,
        )
        return merge_feature_families(mae_features, convnext_features)

    feature_encoder.retain_generated_gradient(terminal_images)
    terminal_local = live_features(terminal_images)
    terminal = gather_features(terminal_local, differentiable=True)
    boundary_loss, boundary_statistics = drifting_loss_from_complete_features(
        terminal,
        positive,
        negative,
        temperatures=temperatures,
    )

    local_samples = source_images.shape[0]
    combined_students = torch.cat((origin_student_images, local_student_images), dim=0)
    feature_encoder.retain_generated_gradient(combined_students)
    combined_features = live_features(combined_students)
    origin = {
        family: value[:local_samples] for family, value in combined_features.items()
    }
    local = {
        family: value[local_samples:] for family, value in combined_features.items()
    }
    local_loss, local_statistics = two_view_local_feature_gmd_loss(
        source,
        origin,
        local,
        positive,
        fixed_negative_features=negative,
        force_step=force_step,
        origin_weight=origin_weight,
        local_weight=local_weight,
        temperatures=temperatures,
        source_sample_range=(rank * local_samples, (rank + 1) * local_samples),
        teacher=teacher,
    )

    global_samples = local_samples * size
    for statistics in (boundary_statistics, local_statistics):
        statistics["support/positive"] = torch.tensor(
            positive_images.shape[0], device=boundary_loss.device, dtype=torch.float32
        )
        statistics["support/fixed_negative"] = torch.tensor(
            fixed_negative_images.shape[0],
            device=boundary_loss.device,
            dtype=torch.float32,
        )
    boundary_statistics["support/generated_global"] = torch.tensor(
        global_samples, device=boundary_loss.device, dtype=torch.float32
    )
    for key in ("source_global", "origin_global", "local_global"):
        local_statistics[f"support/{key}"] = torch.tensor(
            global_samples, device=boundary_loss.device, dtype=torch.float32
        )
    return boundary_loss, boundary_statistics, local_loss, local_statistics


def sample_supported_nfe(
    flow: nn.Module,
    prior: Tensor,
    noise_labels: Tensor,
    *,
    nfe: int,
    time_power: float = 1.0,
) -> Tensor:
    """Compose one artifact at supported NFE 1/2/4/8/16 with fixed labels."""

    if type(nfe) is not int or nfe not in SUPPORTED_EVALUATION_NFES:
        raise ValueError(
            f"nfe must be one of the supported values {SUPPORTED_EVALUATION_NFES}"
        )
    if (
        isinstance(time_power, bool)
        or not math.isfinite(float(time_power))
        or float(time_power) <= 0.0
    ):
        raise ValueError("time_power must be finite and positive")
    times = torch.linspace(
        0.0,
        1.0,
        nfe + 1,
        dtype=torch.float32,
        device=prior.device,
    ).pow(float(time_power))
    state = prior
    for index in range(nfe):
        state = flow(
            state,
            noise_labels,
            times[index],
            times[index + 1],
        )
    return state


__all__ = [
    "DEFAULT_E10_DELTA",
    "SUPPORTED_EVALUATION_NFES",
    "LocalTwoViewOutputs",
    "LocalTwoViewTrainingWrapper",
    "TimePairGrid",
    "cartesian_time_pair_grid",
    "combine_training_losses",
    "explicit_time_pair_grid",
    "factorized_time_pair_grid",
    "global_sharded_boundary_and_local_two_view_loss",
    "sample_supported_nfe",
    "sample_time_pair",
    "unwrap_local_training",
]
