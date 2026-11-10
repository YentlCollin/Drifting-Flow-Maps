"""One-call structural rollout objective for the CelebA two-time GMD map.

This module is deliberately separate from the frozen training/evaluation
sources used by running jobs.  ``StructuralComposedRollout`` is intended to be
the *single* module wrapped by DistributedDataParallel.  One external forward
can produce all of the following without calling ``DDP.module`` directly:

* a differentiable terminal state after exactly ``K`` transitions;
* the direct-teacher and composed-student branches of the canonical
  semigroup check; and
* the direct-teacher and composed-student branches of a local Eulerian
  composition check.

There is no random-number generation in this file.  The caller supplies the
complete rollout grid and every structural-loss time explicitly, so a DDP
trainer can broadcast them before this single forward call.  The exact same
``noise_labels`` tensor is reused for every transition.  Teacher branches are
always evaluated under ``torch.no_grad`` and returned detached; student
branches and the rollout retain their complete autograd graphs.

The terminal rollout reuses :mod:`src.celeba_gmd_flow_objectives`.  The two
small branch builders mirror its audited consistency equations because the
existing public helpers intentionally return only a scalar and statistics,
not the branch tensors required for diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from src.celeba_gmd_flow_objectives import (
    DifferentiableRollout,
    differentiable_fixed_label_rollout,
)


StructuralRolloutResult: TypeAlias = dict[str, Tensor | None]

_RESULT_KEYS = (
    "terminal",
    "rollout_times",
    "semigroup_loss",
    "semigroup_rmse",
    "semigroup_middle_state",
    "semigroup_student",
    "semigroup_teacher",
    "eulerian_loss",
    "eulerian_rmse",
    "eulerian_source_state",
    "eulerian_local_state",
    "eulerian_student",
    "eulerian_teacher",
)


def _optional_group(name: str, values: tuple[object | None, ...]) -> bool:
    """Return whether an all-or-none optional argument group is enabled."""

    supplied = tuple(value is not None for value in values)
    if any(supplied) and not all(supplied):
        raise ValueError(f"{name} arguments must be supplied together")
    return all(supplied)


def _validate_flow_result(result: Tensor, state: Tensor, *, branch: str) -> Tensor:
    if not isinstance(result, Tensor):
        raise TypeError(f"{branch} flow result must be a torch.Tensor")
    if result.shape != state.shape:
        raise ValueError(
            f"{branch} flow result must preserve state shape: "
            f"{tuple(state.shape)} -> {tuple(result.shape)}"
        )
    if result.device != state.device:
        raise ValueError(f"{branch} flow result changed device")
    if not result.is_floating_point():
        raise TypeError(f"{branch} flow result must be floating point")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{branch} flow result is non-finite")
    return result


def _batch_time(value: Tensor | float, state: Tensor, *, name: str) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be real-valued, not bool")
    result = torch.as_tensor(value, device=state.device, dtype=torch.float32)
    if result.numel() == 1:
        result = result.reshape(1).expand(state.shape[0])
    elif result.numel() == state.shape[0]:
        result = result.reshape(state.shape[0])
    else:
        raise ValueError(f"{name} must be scalar or batch-sized")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _ordered_times(
    state: Tensor,
    values: tuple[Tensor | float, ...],
    names: tuple[str, ...],
) -> tuple[Tensor, ...]:
    times = tuple(
        _batch_time(value, state, name=name)
        for value, name in zip(values, names, strict=True)
    )
    if any(bool((right < left).any()) for left, right in zip(times[:-1], times[1:])):
        raise ValueError(f"times must satisfy {' <= '.join(names)}")
    return times


def _loss_and_rmse(student: Tensor, teacher: Tensor, *, branch: str) -> tuple[Tensor, Tensor]:
    if student.shape != teacher.shape:
        raise ValueError(f"{branch} student and teacher shapes differ")
    loss = (student - teacher).square().mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"{branch} loss is non-finite")
    return loss, loss.detach().sqrt()


def _canonical_semigroup_branches(
    flow: nn.Module,
    initial_state: Tensor,
    noise_labels: Tensor,
    middle_time: Tensor | float,
    end_time: Tensor | float,
) -> dict[str, Tensor]:
    """Build ``Phi(Phi(z,0,a),a,b)`` and stopped ``Phi(z,0,b)``."""

    middle = _validate_flow_result(
        flow(initial_state, noise_labels, 0.0, middle_time),
        initial_state,
        branch="canonical middle",
    )
    with torch.no_grad():
        teacher = _validate_flow_result(
            flow(initial_state, noise_labels, 0.0, end_time),
            initial_state,
            branch="canonical teacher",
        ).detach()
    student = _validate_flow_result(
        flow(middle, noise_labels, middle_time, end_time),
        middle,
        branch="canonical student",
    )
    loss, rmse = _loss_and_rmse(student, teacher, branch="canonical semigroup")
    return {
        "loss": loss,
        "rmse": rmse,
        "middle_state": middle,
        "student": student,
        "teacher": teacher,
    }


def _eulerian_composition_branches(
    flow: nn.Module,
    state: Tensor,
    noise_labels: Tensor,
    start_time: Tensor | float,
    local_end_time: Tensor | float,
    end_time: Tensor | float,
) -> dict[str, Tensor]:
    """Build ``Phi(Phi(x,s,u),u,t)`` and stopped ``Phi(x,s,t)``."""

    local = _validate_flow_result(
        flow(state, noise_labels, start_time, local_end_time),
        state,
        branch="Eulerian local",
    )
    with torch.no_grad():
        teacher = _validate_flow_result(
            flow(state, noise_labels, start_time, end_time),
            state,
            branch="Eulerian teacher",
        ).detach()
    student = _validate_flow_result(
        flow(local, noise_labels, local_end_time, end_time),
        local,
        branch="Eulerian student",
    )
    loss, rmse = _loss_and_rmse(student, teacher, branch="Eulerian composition")
    return {
        "loss": loss,
        "rmse": rmse,
        "local_state": local,
        "student": student,
        "teacher": teacher,
    }


class StructuralComposedRollout(nn.Module):
    """Return terminal and optional stopped-teacher consistency branches.

    Wrap this object itself in DDP.  Do not wrap ``flow`` separately and do
    not invoke ``DDP.module.flow`` during a training update.

    ``rollout_time_grid`` is mandatory and has shape ``[K + 1]``.  Canonical
    semigroup evaluation is disabled when both canonical times are ``None``;
    otherwise both must be supplied.  Eulerian evaluation is similarly
    all-or-none for its three times.  Its source is either an explicit
    ``eulerian_state`` or one state selected from the live terminal rollout by
    ``eulerian_rollout_index``.  Selecting a rollout state preserves its
    upstream graph.
    """

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        if not isinstance(flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        self.flow = flow

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        k: int,
        rollout_time_grid: Tensor,
        *,
        semigroup_middle_time: Tensor | float | None = None,
        semigroup_end_time: Tensor | float | None = None,
        eulerian_state: Tensor | None = None,
        eulerian_rollout_index: int | None = None,
        eulerian_start_time: Tensor | float | None = None,
        eulerian_local_end_time: Tensor | float | None = None,
        eulerian_end_time: Tensor | float | None = None,
    ) -> StructuralRolloutResult:
        if not isinstance(rollout_time_grid, Tensor):
            raise TypeError("rollout_time_grid must be a caller-supplied Tensor")

        canonical_enabled = _optional_group(
            "canonical semigroup",
            (semigroup_middle_time, semigroup_end_time),
        )
        eulerian_enabled = _optional_group(
            "Eulerian composition",
            (
                eulerian_start_time,
                eulerian_local_end_time,
                eulerian_end_time,
            ),
        )
        source_options = int(eulerian_state is not None) + int(
            eulerian_rollout_index is not None
        )
        if eulerian_enabled and source_options != 1:
            raise ValueError(
                "Eulerian composition requires exactly one of eulerian_state "
                "or eulerian_rollout_index"
            )
        if not eulerian_enabled and source_options:
            raise ValueError("Eulerian source was supplied without Eulerian times")

        rollout: DifferentiableRollout = differentiable_fixed_label_rollout(
            self.flow,
            prior,
            noise_labels,
            k=k,
            time_grid=rollout_time_grid,
        )
        result: StructuralRolloutResult = {key: None for key in _RESULT_KEYS}
        result["terminal"] = rollout.terminal
        result["rollout_times"] = rollout.times

        if canonical_enabled:
            assert semigroup_middle_time is not None
            assert semigroup_end_time is not None
            canonical_middle, canonical_end = _ordered_times(
                prior,
                (semigroup_middle_time, semigroup_end_time),
                ("semigroup_middle_time", "semigroup_end_time"),
            )
            canonical = _canonical_semigroup_branches(
                self.flow,
                prior,
                noise_labels,
                canonical_middle,
                canonical_end,
            )
            result.update(
                semigroup_loss=canonical["loss"],
                semigroup_rmse=canonical["rmse"],
                semigroup_middle_state=canonical["middle_state"],
                semigroup_student=canonical["student"],
                semigroup_teacher=canonical["teacher"],
            )

        if eulerian_enabled:
            if eulerian_state is not None:
                source = eulerian_state
                if source.shape != prior.shape or source.device != prior.device:
                    raise ValueError(
                        "explicit eulerian_state must match prior shape and device"
                    )
                if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
                    raise ValueError(
                        "explicit eulerian_state must be finite floating point"
                    )
            else:
                if (
                    isinstance(eulerian_rollout_index, bool)
                    or not isinstance(eulerian_rollout_index, int)
                    or not 0 <= eulerian_rollout_index <= rollout.k
                ):
                    raise ValueError("eulerian_rollout_index must lie in [0, K]")
                source = rollout.states[eulerian_rollout_index]
            assert eulerian_start_time is not None
            assert eulerian_local_end_time is not None
            assert eulerian_end_time is not None
            eulerian_start, eulerian_local_end, eulerian_end = _ordered_times(
                source,
                (
                    eulerian_start_time,
                    eulerian_local_end_time,
                    eulerian_end_time,
                ),
                (
                    "eulerian_start_time",
                    "eulerian_local_end_time",
                    "eulerian_end_time",
                ),
            )
            if eulerian_state is None:
                expected_start = rollout.times[eulerian_rollout_index]
                if bool((eulerian_start != expected_start).any()):
                    raise ValueError(
                        "eulerian_start_time must equal the selected rollout "
                        "state's grid time"
                    )
            eulerian = _eulerian_composition_branches(
                self.flow,
                source,
                noise_labels,
                eulerian_start,
                eulerian_local_end,
                eulerian_end,
            )
            result.update(
                eulerian_loss=eulerian["loss"],
                eulerian_rmse=eulerian["rmse"],
                eulerian_source_state=source,
                eulerian_local_state=eulerian["local_state"],
                eulerian_student=eulerian["student"],
                eulerian_teacher=eulerian["teacher"],
            )

        if tuple(result) != _RESULT_KEYS:
            raise RuntimeError("structural result schema changed unexpectedly")
        return result


def unwrap_structural_composed_rollout(module: nn.Module) -> StructuralComposedRollout:
    """Return this wrapper directly or through exactly one DDP layer."""

    if isinstance(module, StructuralComposedRollout):
        return module
    if isinstance(module, DistributedDataParallel) and isinstance(
        module.module, StructuralComposedRollout
    ):
        return module.module
    raise TypeError(
        "module must be StructuralComposedRollout or "
        "DistributedDataParallel[StructuralComposedRollout]"
    )


def structural_losses(result: Mapping[str, Tensor | None]) -> dict[str, Tensor]:
    """Extract enabled scalar losses from an exact wrapper result schema."""

    if tuple(result) != _RESULT_KEYS:
        raise ValueError("structural rollout result schema differs")
    losses: dict[str, Tensor] = {}
    for name in ("semigroup_loss", "eulerian_loss"):
        value = result[name]
        if value is not None:
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError(f"{name} must be a scalar Tensor")
            losses[name] = value
    return losses


__all__ = [
    "StructuralComposedRollout",
    "StructuralRolloutResult",
    "structural_losses",
    "unwrap_structural_composed_rollout",
]
