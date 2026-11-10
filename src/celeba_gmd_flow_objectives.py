"""Pure flow-map rollout and consistency objectives for CelebA-64 GMD.

The functions here use the public four-argument contract of
``CelebA64TwoTimeGMDDiT``::

    next_state = model(state, noise_labels, start_time, end_time)

They deliberately know nothing about GMD features or feature-space losses.
This keeps the transport mechanics independently testable and lets a trainer
apply its feature objective to :func:`differentiable_terminal_output` outside
this module.

Two related, but intentionally distinct, consistency losses are provided.
The weak canonical loss checks the semigroup identity only on paths reached
from the time-zero input,

    Phi(Phi(z, 0, a), a, b) = Phi(z, 0, b).

The Eulerian/local loss starts from an arbitrary state at an arbitrary time
and makes the first leg a short local transition,

    Phi(Phi(x_s, s, s + delta), s + delta, b) = Phi(x_s, s, b).

The latter is the finite-composition analogue of the local Eulerian term used
by the MNIST flow-map objective.  It needs only three map evaluations per
batch (two sequential student evaluations and one direct teacher evaluation),
not a Jacobian/JVP or an O(K**2) collection of grid pairs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor


GridKind = Literal["uniform", "power"]


class FixedLabelTwoTimeFlowMap(Protocol):
    """Structural type implemented by ``CelebA64TwoTimeGMDDiT``."""

    def __call__(
        self,
        state: Tensor,
        noise_labels: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor: ...


@dataclass(frozen=True)
class DifferentiableRollout:
    """A fixed-label trajectory whose state tensors retain their autograd graph.

    ``states[0]`` is the supplied initial state and ``states[-1]`` is the
    terminal output.  A tuple is used instead of eagerly stacking full-size
    images, avoiding an otherwise unnecessary trajectory-sized copy.
    """

    times: Tensor
    states: tuple[Tensor, ...]

    @property
    def k(self) -> int:
        return len(self.states) - 1

    @property
    def initial(self) -> Tensor:
        return self.states[0]

    @property
    def terminal(self) -> Tensor:
        return self.states[-1]

    def stacked_states(self) -> Tensor:
        """Materialize the path as ``[K + 1, B, ...]`` when it is needed."""

        return torch.stack(self.states, dim=0)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive Python integer shared by the batch")
    return k


def _validate_state(state: Tensor, *, name: str = "state") -> None:
    if not isinstance(state, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim < 2 or state.shape[0] < 1 or any(size < 1 for size in state.shape[1:]):
        raise ValueError(f"{name} must have a non-empty batch and feature dimensions")
    if not state.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(state).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_noise_labels(state: Tensor, noise_labels: Tensor) -> None:
    if not isinstance(noise_labels, Tensor):
        raise TypeError("noise_labels must be a torch.Tensor")
    if noise_labels.ndim != 2 or noise_labels.shape[0] != state.shape[0]:
        raise ValueError(
            "noise_labels must have shape [B, noise_coords] with the same "
            "batch size as state"
        )
    if noise_labels.shape[1] < 1:
        raise ValueError("noise_labels must contain at least one noise coordinate")
    if noise_labels.dtype != torch.int64:
        raise TypeError("noise_labels must have dtype torch.int64")
    if noise_labels.device != state.device:
        raise ValueError("state and noise_labels must be on the same device")


def _validate_inputs(state: Tensor, noise_labels: Tensor) -> None:
    _validate_state(state)
    _validate_noise_labels(state, noise_labels)


def _scalar_time(value: Tensor | float, like: Tensor, *, name: str) -> Tensor:
    if isinstance(value, bool) or (isinstance(value, Tensor) and value.dtype == torch.bool):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if result.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    result = result.reshape(())
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _batch_time(value: Tensor | float, like: Tensor, *, name: str) -> Tensor:
    if isinstance(value, bool) or (isinstance(value, Tensor) and value.dtype == torch.bool):
        raise TypeError(f"{name} must be real-valued, not bool")
    result = torch.as_tensor(value, device=like.device, dtype=torch.float32)
    if result.ndim == 0 or result.numel() == 1:
        result = result.reshape(1).expand(like.shape[0])
    elif result.numel() == like.shape[0]:
        result = result.reshape(like.shape[0])
    else:
        raise ValueError(
            f"{name} must be scalar or contain {like.shape[0]} values, "
            f"got shape {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _check_unit_interval(*named_times: tuple[str, Tensor]) -> None:
    for name, value in named_times:
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"{name} must lie in [0, 1]")


def make_rollout_time_grid(
    k: int,
    like: Tensor,
    *,
    grid: GridKind = "uniform",
    power: float = 1.0,
    start_time: Tensor | float = 0.0,
    end_time: Tensor | float = 1.0,
) -> Tensor:
    """Build a strictly increasing, scalar time partition of length ``K + 1``.

    ``grid="uniform"`` uses equally spaced physical times.  ``grid="power"``
    maps a uniform support ``u`` to ``u**power`` before scaling it to the
    requested interval.  One Python integer ``k`` is used for the complete
    minibatch; per-sample/heterogeneous step counts are intentionally rejected.
    Physical times use float32, matching ``CelebA64TwoTimeGMDDiT``'s time path
    even when image computation is autocast.
    """

    _validate_k(k)
    _validate_state(like, name="like")
    if grid not in {"uniform", "power"}:
        raise ValueError("grid must be 'uniform' or 'power'")
    if isinstance(power, bool):
        raise TypeError("power must be a finite positive real number")
    try:
        power_value = float(power)
    except (TypeError, ValueError) as error:
        raise TypeError("power must be a finite positive real number") from error
    if not math.isfinite(power_value) or power_value <= 0.0:
        raise ValueError("power must be finite and positive")

    start = _scalar_time(start_time, like, name="start_time")
    end = _scalar_time(end_time, like, name="end_time")
    _check_unit_interval(("start_time", start), ("end_time", end))
    if bool(start >= end):
        raise ValueError("rollout requires start_time < end_time")

    support = torch.linspace(
        0.0,
        1.0,
        k + 1,
        device=like.device,
        dtype=torch.float32,
    )
    if grid == "power":
        support = support.pow(power_value)
    result = start + (end - start) * support
    return validate_rollout_time_grid(result, k, like)


def validate_rollout_time_grid(
    time_grid: Tensor | Sequence[float],
    k: int,
    like: Tensor,
) -> Tensor:
    """Move an explicit grid to ``like`` and validate its physical ordering."""

    _validate_k(k)
    _validate_state(like, name="like")
    if isinstance(time_grid, Tensor) and time_grid.dtype == torch.bool:
        raise TypeError("time_grid must be real-valued, not bool")
    values = torch.as_tensor(time_grid, device=like.device, dtype=torch.float32)
    if values.ndim != 1 or values.shape[0] != k + 1:
        raise ValueError(f"time_grid must have shape ({k + 1},)")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("time_grid must contain only finite values")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("time_grid values must lie in [0, 1]")
    if bool((values[1:] <= values[:-1]).any()):
        raise ValueError("time_grid must be strictly increasing")
    return values


def _checked_transition(
    model: FixedLabelTwoTimeFlowMap,
    state: Tensor,
    noise_labels: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
) -> Tensor:
    result = model(state, noise_labels, start_time, end_time)
    if not isinstance(result, Tensor):
        raise TypeError("flow map must return a torch.Tensor")
    if result.shape != state.shape:
        raise ValueError(
            "flow map must preserve state shape, "
            f"got {tuple(state.shape)} -> {tuple(result.shape)}"
        )
    if result.device != state.device:
        raise ValueError("flow map output must stay on the state device")
    if not result.is_floating_point():
        raise TypeError("flow map output must have a floating-point dtype")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("flow map produced non-finite values")
    return result


def differentiable_fixed_label_rollout(
    model: FixedLabelTwoTimeFlowMap,
    initial_state: Tensor,
    noise_labels: Tensor,
    *,
    k: int,
    grid: GridKind = "uniform",
    power: float = 1.0,
    start_time: Tensor | float = 0.0,
    end_time: Tensor | float = 1.0,
    time_grid: Tensor | Sequence[float] | None = None,
) -> DifferentiableRollout:
    """Compose exactly ``K`` flow-map calls without disabling or cutting autograd.

    The exact same ``noise_labels`` tensor is supplied to every transition;
    labels are never resampled, repeated, cast, or detached.  ``time_grid`` may
    provide an explicit shared partition.  Otherwise a uniform or power grid
    is generated from ``start_time`` to ``end_time``.
    """

    _validate_inputs(initial_state, noise_labels)
    _validate_k(k)
    if time_grid is None:
        times = make_rollout_time_grid(
            k,
            initial_state,
            grid=grid,
            power=power,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        times = validate_rollout_time_grid(time_grid, k, initial_state)

    states = [initial_state]
    state = initial_state
    for index in range(k):
        state = _checked_transition(
            model,
            state,
            noise_labels,
            times[index],
            times[index + 1],
        )
        states.append(state)
    return DifferentiableRollout(times=times, states=tuple(states))


def terminal_output(rollout: DifferentiableRollout) -> Tensor:
    """Return a rollout's terminal tensor without copying or detaching it."""

    if not isinstance(rollout, DifferentiableRollout):
        raise TypeError("rollout must be a DifferentiableRollout")
    return rollout.terminal


def differentiable_terminal_output(
    model: FixedLabelTwoTimeFlowMap,
    initial_state: Tensor,
    noise_labels: Tensor,
    *,
    k: int,
    grid: GridKind = "uniform",
    power: float = 1.0,
    start_time: Tensor | float = 0.0,
    end_time: Tensor | float = 1.0,
    time_grid: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Convenience wrapper returning the differentiable final state only."""

    rollout = differentiable_fixed_label_rollout(
        model,
        initial_state,
        noise_labels,
        k=k,
        grid=grid,
        power=power,
        start_time=start_time,
        end_time=end_time,
        time_grid=time_grid,
    )
    return terminal_output(rollout)


def _validate_ordered_times(
    state: Tensor,
    start_time: Tensor | float,
    middle_time: Tensor | float,
    end_time: Tensor | float,
    *,
    middle_name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    start = _batch_time(start_time, state, name="start_time")
    middle = _batch_time(middle_time, state, name=middle_name)
    end = _batch_time(end_time, state, name="end_time")
    _check_unit_interval(
        ("start_time", start),
        (middle_name, middle),
        ("end_time", end),
    )
    if bool((start > middle).any()) or bool((middle > end).any()):
        raise ValueError(f"times must satisfy start_time <= {middle_name} <= end_time")
    return start, middle, end


def _squared_error_loss(prediction: Tensor, target: Tensor, *, name: str) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} prediction and target shapes differ: "
            f"{tuple(prediction.shape)} versus {tuple(target.shape)}"
        )
    loss = (prediction - target).square().mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"{name} loss is non-finite")
    return loss


def weak_canonical_semigroup_loss(
    model: FixedLabelTwoTimeFlowMap,
    initial_state: Tensor,
    noise_labels: Tensor,
    middle_time: Tensor | float,
    end_time: Tensor | float = 1.0,
    *,
    detach_teacher: bool = True,
) -> tuple[Tensor, dict[str, Tensor | bool]]:
    """Check semigroup consistency on a canonical path from physical time zero.

    The composed path is always the student and remains differentiable through
    *both* map calls.  When ``detach_teacher=True``, the direct ``Phi(z,0,b)``
    target is evaluated under ``no_grad``.  Setting it to ``False`` gives a
    symmetric objective with gradients through both direct and composed
    branches; there is no implicit detach in that mode.
    """

    _validate_inputs(initial_state, noise_labels)
    if not isinstance(detach_teacher, bool):
        raise TypeError("detach_teacher must be bool")
    start, middle, end = _validate_ordered_times(
        initial_state,
        0.0,
        middle_time,
        end_time,
        middle_name="middle_time",
    )

    middle_state = _checked_transition(
        model, initial_state, noise_labels, start, middle
    )
    teacher_context = torch.no_grad() if detach_teacher else nullcontext()
    with teacher_context:
        direct = _checked_transition(model, initial_state, noise_labels, start, end)
    if detach_teacher:
        direct = direct.detach()
    composed = _checked_transition(
        model, middle_state, noise_labels, middle, end
    )
    loss = _squared_error_loss(composed, direct, name="canonical semigroup")
    error = composed.detach() - direct.detach()
    return loss, {
        "loss_semigroup": loss.detach(),
        "semigroup_rmse": error.square().mean().sqrt(),
        "semigroup_error_rms": error.square().mean().sqrt(),
        "middle_time_mean": middle.detach().mean(),
        "end_time_mean": end.detach().mean(),
        "teacher_detached": detach_teacher,
    }


def eulerian_local_composition_loss(
    model: FixedLabelTwoTimeFlowMap,
    state: Tensor,
    noise_labels: Tensor,
    start_time: Tensor | float,
    local_end_time: Tensor | float,
    end_time: Tensor | float,
    *,
    detach_teacher: bool = True,
) -> tuple[Tensor, dict[str, Tensor | bool]]:
    """Enforce a short-hop composition identity from an arbitrary state.

    ``state`` may itself be a differentiable state from a longer rollout; this
    function does not detach it.  For the intended local/Eulerian use,
    ``local_end_time - start_time`` should be small.  The mathematical check is

    ``Phi(Phi(state,s,u),u,t) == Phi(state,s,t)``, where ``u`` is
    ``local_end_time``.  The direct right-hand side is the optional stopped
    teacher.  Only three map evaluations are required, irrespective of the
    rollout length ``K``.
    """

    _validate_inputs(state, noise_labels)
    if not isinstance(detach_teacher, bool):
        raise TypeError("detach_teacher must be bool")
    start, local_end, end = _validate_ordered_times(
        state,
        start_time,
        local_end_time,
        end_time,
        middle_name="local_end_time",
    )

    local_state = _checked_transition(
        model, state, noise_labels, start, local_end
    )
    teacher_context = torch.no_grad() if detach_teacher else nullcontext()
    with teacher_context:
        direct = _checked_transition(model, state, noise_labels, start, end)
    if detach_teacher:
        direct = direct.detach()
    composed = _checked_transition(
        model, local_state, noise_labels, local_end, end
    )
    loss = _squared_error_loss(composed, direct, name="local composition")
    error = composed.detach() - direct.detach()
    return loss, {
        "loss_eulerian": loss.detach(),
        "loss_local_composition": loss.detach(),
        "local_composition_rmse": error.square().mean().sqrt(),
        "local_span_mean": (local_end - start).detach().mean(),
        "remaining_span_mean": (end - local_end).detach().mean(),
        "teacher_detached": detach_teacher,
    }


def diagonal_identity_loss(
    model: FixedLabelTwoTimeFlowMap,
    state: Tensor,
    noise_labels: Tensor,
    time: Tensor | float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the diagonal identity error ``||Phi(x,t,t)-x||^2``.

    ``CelebA64TwoTimeGMDDiT`` satisfies this identity by construction, so this
    is primarily an invariant check rather than a necessary training
    penalty.  The target state is stopped, while the map evaluation remains
    differentiable.
    """

    _validate_inputs(state, noise_labels)
    diagonal_time = _batch_time(time, state, name="time")
    _check_unit_interval(("time", diagonal_time))
    prediction = _checked_transition(
        model,
        state,
        noise_labels,
        diagonal_time,
        diagonal_time,
    )
    loss = _squared_error_loss(prediction, state.detach(), name="diagonal identity")
    error = prediction.detach() - state.detach()
    return loss, {
        "loss_diagonal_identity": loss.detach(),
        "diagonal_identity_rmse": error.square().mean().sqrt(),
        "diagonal_time_mean": diagonal_time.detach().mean(),
    }


__all__ = [
    "DifferentiableRollout",
    "FixedLabelTwoTimeFlowMap",
    "diagonal_identity_loss",
    "differentiable_fixed_label_rollout",
    "differentiable_terminal_output",
    "eulerian_local_composition_loss",
    "make_rollout_time_grid",
    "terminal_output",
    "validate_rollout_time_grid",
    "weak_canonical_semigroup_loss",
]
