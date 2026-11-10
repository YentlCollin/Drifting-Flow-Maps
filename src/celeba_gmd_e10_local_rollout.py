"""One-call CelebA structural E10 rollout with detached local two-view states.

``E10LocalComposedRollout`` is the sole module that should be wrapped by
DistributedDataParallel.  It first delegates the complete terminal rollout
and every enabled structural branch to the audited
:class:`~src.celeba_gmd_structural_rollout.StructuralComposedRollout`.  Only
after that graph is complete does it build a detached local source and the two
live local students.  Consequently, adding the local views cannot replace or
detach the live rollout state used by the established E10 objective.

The wrapper deliberately contains no feature extraction, local loss, loss
weight, or random-number generation.  The caller supplies both homogeneous
local times, and all learned flow calls happen below one outer forward.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from src.celeba_gmd_local_two_view import homogeneous_two_view_states
from src.celeba_gmd_structural_rollout import (
    StructuralComposedRollout,
    StructuralRolloutResult,
)


class E10LocalRolloutResult(NamedTuple):
    """Unchanged structural result plus one homogeneous local two-view graph."""

    structural: StructuralRolloutResult
    local_start_time: Tensor
    local_time_fraction: Tensor
    local_end_time: Tensor
    local_source: Tensor
    local_origin_student: Tensor
    local_student: Tensor


def _validate_local_inputs(prior: Tensor, noise_labels: Tensor) -> None:
    """Reject malformed local inputs before executing the structural graph."""

    if not isinstance(prior, Tensor):
        raise TypeError("prior must be a torch.Tensor")
    if prior.ndim < 2 or prior.shape[0] < 1 or any(
        size < 1 for size in prior.shape[1:]
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


def _local_times(
    prior: Tensor,
    *,
    start_time: Tensor | float,
    time_fraction: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached scalar ``(a, alpha, a + alpha * (1 - a))`` tensors."""

    values: list[Tensor] = []
    for name, value in (
        ("local_start_time", start_time),
        ("local_time_fraction", time_fraction),
    ):
        if isinstance(value, bool) or (
            isinstance(value, Tensor) and value.dtype == torch.bool
        ):
            raise TypeError(f"{name} must be a real scalar, not bool")
        tensor = torch.as_tensor(value, device=prior.device, dtype=torch.float32)
        if tensor.numel() != 1:
            raise ValueError(f"{name} must be one scalar shared by the batch")
        tensor = tensor.reshape(()).detach()
        if not bool(torch.isfinite(tensor)):
            raise ValueError(f"{name} must be finite")
        values.append(tensor)

    start, fraction = values
    if not bool((start >= 0.0) & (start < 1.0)):
        raise ValueError("local_start_time must lie in [0, 1)")
    if not bool((fraction > 0.0) & (fraction <= 1.0)):
        raise ValueError("local_time_fraction must lie in (0, 1]")

    end = start + fraction * (1.0 - start)
    if not bool(torch.isfinite(end)) or not bool((end > start) & (end <= 1.0)):
        raise ValueError(
            "local times must produce a finite end strictly after the start "
            "and no later than one"
        )
    return start, fraction, end.detach()


class E10LocalComposedRollout(nn.Module):
    """Preserve the structural K/E10 graph and append two live local views.

    Wrap this object itself in DDP.  ``flow`` is registered exactly once,
    beneath ``structural_rollout``; the read-only :attr:`flow` property is
    provided for checkpoint and optimizer construction.
    """

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        if not isinstance(flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        if isinstance(flow, DistributedDataParallel):
            raise TypeError("flow must be unwrapped; wrap E10LocalComposedRollout in DDP")
        self.structural_rollout = StructuralComposedRollout(flow)

    @property
    def flow(self) -> nn.Module:
        """Return the single flow owned by the nested structural wrapper."""

        return self.structural_rollout.flow

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        k: int,
        rollout_time_grid: Tensor,
        *,
        local_start_time: Tensor | float,
        local_time_fraction: Tensor | float,
        semigroup_middle_time: Tensor | float | None = None,
        semigroup_end_time: Tensor | float | None = None,
        eulerian_state: Tensor | None = None,
        eulerian_rollout_index: int | None = None,
        eulerian_start_time: Tensor | float | None = None,
        eulerian_local_end_time: Tensor | float | None = None,
        eulerian_end_time: Tensor | float | None = None,
    ) -> E10LocalRolloutResult:
        _validate_local_inputs(prior, noise_labels)
        local_start, local_fraction, expected_local_end = _local_times(
            prior,
            start_time=local_start_time,
            time_fraction=local_time_fraction,
        )

        # This call, including its live rollout-selected E10 source, remains
        # byte-for-byte delegated to the audited structural implementation.
        structural = self.structural_rollout(
            prior,
            noise_labels,
            k,
            rollout_time_grid,
            semigroup_middle_time=semigroup_middle_time,
            semigroup_end_time=semigroup_end_time,
            eulerian_state=eulerian_state,
            eulerian_rollout_index=eulerian_rollout_index,
            eulerian_start_time=eulerian_start_time,
            eulerian_local_end_time=eulerian_local_end_time,
            eulerian_end_time=eulerian_end_time,
        )

        # The helper completes the stopped source under no_grad and autocast
        # cache_enabled=False before executing either live student.  DDP time
        # synchronization belongs to the caller, alongside its checkpointed
        # local-time RNG, so it is intentionally disabled here.
        local = homogeneous_two_view_states(
            self.flow,
            prior,
            noise_labels,
            start_time=local_start,
            time_fraction=local_fraction,
            synchronize_distributed=False,
        )
        if not torch.equal(local.end_time, expected_local_end):
            raise RuntimeError("local two-view helper changed the end-time equation")

        return E10LocalRolloutResult(
            structural=structural,
            local_start_time=local.start_time,
            local_time_fraction=local.time_fraction,
            local_end_time=local.end_time,
            local_source=local.source,
            local_origin_student=local.origin_student,
            local_student=local.local_student,
        )


def unwrap_e10_local_composed_rollout(module: nn.Module) -> E10LocalComposedRollout:
    """Return this wrapper directly or through exactly one outer DDP layer."""

    if isinstance(module, E10LocalComposedRollout):
        return module
    if isinstance(module, DistributedDataParallel) and isinstance(
        module.module, E10LocalComposedRollout
    ):
        return module.module
    raise TypeError(
        "module must be E10LocalComposedRollout or "
        "DistributedDataParallel[E10LocalComposedRollout]"
    )


__all__ = [
    "E10LocalComposedRollout",
    "E10LocalRolloutResult",
    "unwrap_e10_local_composed_rollout",
]
