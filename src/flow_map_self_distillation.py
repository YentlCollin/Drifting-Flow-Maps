"""Faithful PyTorch building blocks for flow-map self-distillation.

The definitions in this module follow the released implementation of
``How to build a consistency model: Learning flow maps via self-distillation``
(Boffi, Albergo, and Vanden-Eijnden, 2025), official repository commit
``2f115a07fa9073553193e4b265dfc303827af2b0``.

The central parameterization is

    X_theta(x, s, t) = x + (t - s) * phi_theta(x, s, t).

It gives an exact identity on the diagonal and makes the associated diagonal
velocity simply ``b_theta(x, t) = phi_theta(x, t, t)``.  Lagrangian
self-distillation (LSD) then enforces

    partial_t X_theta(x, s, t)
        = stopgrad(b_theta(stopgrad(X_theta(x, s, t)), t)).

This file deliberately contains no CelebA training loop.  It is independent
of the sources snapshotted by running jobs, and can therefore be tested before
the experiment driver is switched to the new objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

import torch
from torch import Tensor, nn


class VelocityNetwork(Protocol):
    """Protocol implemented by a two-time velocity/residual network."""

    def __call__(self, x: Tensor, start_time: Tensor, end_time: Tensor) -> Tensor: ...


def _time_vector(value: Tensor | float, like: Tensor, *, name: str) -> Tensor:
    """Return one scalar time per sample without silently changing values."""

    result = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if result.ndim == 0 or result.numel() == 1:
        return result.reshape(1).expand(like.shape[0])
    if result.numel() != like.shape[0]:
        raise ValueError(
            f"{name} must be scalar or contain {like.shape[0]} values, "
            f"got shape {tuple(result.shape)}"
        )
    return result.reshape(like.shape[0])


def _broadcast_time(value: Tensor, like: Tensor) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))


def _per_sample_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError("flow-map samples must contain a batch and feature dimensions")
    return (prediction - target).square().flatten(1).sum(dim=1)


class MagnitudePreservingTimeLogVariance(nn.Module):
    """PyTorch port of the release's learned two-time loss weighting head.

    The head uses two deterministic Fourier embeddings, their
    magnitude-preserving average, and a normalized bias-free linear map.  Its
    scalar output enters every term as ``exp(-logvar) * squared_error +
    logvar``.  The official EDM2 optimizer additionally projects raw weights
    to a sphere after updates; normalizing in ``forward`` makes the represented
    function identical and a trainer may still apply an explicit projection.
    """

    def __init__(self, channels: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        if channels < 4 or channels % 4:
            raise ValueError("channels must be a positive multiple of four")
        if max_period <= 0:
            raise ValueError("max_period must be positive")
        half = channels // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.channels = channels
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.weight = nn.Parameter(torch.randn(1, channels))

    @torch.no_grad()
    def project_weights_(self) -> None:
        """Apply the released EDM2 post-update MPConv projection in place.

        The release normalizes every MPConv filter to approximately unit RMS
        after each optimizer update.  The same normalization also appears in
        :meth:`forward`, but projecting the stored parameter is still relevant
        to optimizer state and weight decay, so experiment drivers should call
        this method immediately after ``optimizer.step()``.
        """

        weight = self.weight.float()
        denominator = 1e-4 + weight.square().sum(
            dim=1,
            keepdim=True,
        ).sqrt() / math.sqrt(self.channels)
        self.weight.copy_((weight / denominator).to(dtype=self.weight.dtype))

    def _embedding(self, time: Tensor) -> Tensor:
        arguments = time.float()[:, None] * self.frequencies.float()[None, :]
        return math.sqrt(2.0) * torch.cat(
            (arguments.cos(), arguments.sin()),
            dim=1,
        )

    def forward(self, start_time: Tensor, end_time: Tensor) -> Tensor:
        if start_time.shape != end_time.shape or start_time.ndim != 1:
            raise ValueError("loss-weight times must be matching one-dimensional tensors")
        start_embedding = self._embedding(start_time)
        end_embedding = self._embedding(end_time)
        embedding = (start_embedding + end_embedding) / math.sqrt(2.0)
        weight = self.weight.float()
        weight = weight / (
            1e-4 + weight.square().sum(dim=1, keepdim=True).sqrt() / math.sqrt(self.channels)
        )
        weight = weight / math.sqrt(self.channels)
        return torch.nn.functional.linear(embedding, weight).reshape(-1)


class ResidualTwoTimeFlowMap(nn.Module):
    """Turn a two-time velocity network into an identity-grounded flow map.

    ``velocity_network`` may be a fresh network or a CelebA latent U-Net whose
    ``endpoint`` method predicts an unconstrained tensor.  Using ``endpoint``
    lets the existing U-Net serve as :math:`phi_theta` without inheriting its
    historical normalized endpoint interpolation.
    """

    def __init__(
        self,
        velocity_network: nn.Module,
        log_variance_network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        endpoint = getattr(velocity_network, "endpoint", None)
        endpoint_parameterization = getattr(
            velocity_network,
            "endpoint_parameterization",
            None,
        )
        if callable(endpoint) and endpoint_parameterization not in {None, "absolute"}:
            raise ValueError(
                "a backbone exposing endpoint() must use "
                "endpoint_parameterization='absolute' so endpoint() represents phi; "
                f"got {endpoint_parameterization!r}"
            )
        self.velocity_network = velocity_network
        self.log_variance_network = log_variance_network

    def velocity(
        self,
        x: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor:
        start = _time_vector(start_time, x, name="start_time")
        end = _time_vector(end_time, x, name="end_time")
        endpoint = getattr(self.velocity_network, "endpoint", None)
        if callable(endpoint):
            result = endpoint(x, start, end)
        else:
            result = self.velocity_network(x, start, end)
        if result.shape != x.shape:
            raise ValueError(
                "velocity network must preserve the sample shape, "
                f"got {tuple(x.shape)} -> {tuple(result.shape)}"
            )
        return result

    def diagonal_velocity(self, x: Tensor, time: Tensor | float) -> Tensor:
        time = _time_vector(time, x, name="time")
        return self.velocity(x, time, time)

    def loss_log_variance(
        self,
        start_time: Tensor | float,
        end_time: Tensor | float,
        like: Tensor,
    ) -> Tensor:
        start = _time_vector(start_time, like, name="start_time")
        end = _time_vector(end_time, like, name="end_time")
        if self.log_variance_network is None:
            return torch.zeros_like(start)
        result = self.log_variance_network(start, end)
        if result.shape != start.shape:
            raise ValueError(
                "log-variance network must return one scalar per sample, "
                f"got {tuple(result.shape)}"
            )
        return result.float()

    def forward(
        self,
        x: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
        *,
        output_kind: str = "map",
    ) -> Tensor:
        start = _time_vector(start_time, x, name="start_time")
        end = _time_vector(end_time, x, name="end_time")
        displacement = self.velocity(x, start, end)
        if output_kind == "velocity":
            return displacement
        if output_kind == "log_variance":
            return self.loss_log_variance(start, end, x)
        if output_kind != "map":
            raise ValueError("output_kind must be 'map', 'velocity', or 'log_variance'")
        return x + _broadcast_time(end - start, x) * displacement


def flow_map_velocity(
    flow_map: nn.Module,
    x: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
) -> Tensor:
    """Evaluate ``phi`` through ``forward`` so DDP hooks remain active."""

    result = flow_map(x, start_time, end_time, output_kind="velocity")
    if result.shape != x.shape:
        raise ValueError("flow-map velocity has the wrong sample shape")
    return result


def flow_map_log_variance(
    flow_map: nn.Module,
    x: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
) -> Tensor:
    """Evaluate the learned loss weight through the DDP-visible forward path."""

    result = flow_map(x, start_time, end_time, output_kind="log_variance")
    if result.shape != (x.shape[0],):
        raise ValueError("flow-map log variance has the wrong batch shape")
    return result.float()


def linear_interpolant(
    x0: Tensor,
    x1: Tensor,
    time: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Return ``I_t=(1-t)x0+t*x1`` and its exact velocity ``x1-x0``."""

    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    time = _time_vector(time, x0, name="time")
    velocity = x1 - x0
    return x0 + _broadcast_time(time, x0) * velocity, velocity


def sample_diagonal_times(
    batch_size: int,
    like: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return torch.rand(
        batch_size,
        device=like.device,
        dtype=like.dtype,
        generator=generator,
    )


def sample_upper_triangle_times(
    batch_size: int,
    like: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample ``(s,t)`` uniformly from the triangle ``0 <= s <= t <= 1``.

    Taking the ordered statistics of two independent uniforms reproduces the
    official release's sampling rule and its constant density over the upper
    triangle.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = torch.rand(
        2,
        batch_size,
        device=like.device,
        dtype=like.dtype,
        generator=generator,
    )
    return values.min(dim=0).values, values.max(dim=0).values


def endpoint_time_jvp(
    flow_map: nn.Module,
    x: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
    *,
    create_graph: bool,
) -> tuple[Tensor, Tensor]:
    """Evaluate a flow map and its directional derivative in endpoint time.

    Endpoint times contain one independent scalar per sample.  The all-ones
    tangent therefore returns each sample's own ``partial_t X`` in one JVP.
    """

    start = _time_vector(start_time, x, name="start_time")
    end = _time_vector(end_time, x, name="end_time")

    def transition(candidate_end: Tensor) -> Tensor:
        return flow_map(x, start, candidate_end)

    return torch.autograd.functional.jvp(
        transition,
        end,
        torch.ones_like(end),
        create_graph=create_graph,
        strict=False,
    )


def uncertainty_weighted_loss(
    flow_map: nn.Module,
    per_sample_squared_error: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
    like: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the release's learned ``exp(-w) E + w`` weighting."""

    if per_sample_squared_error.shape != (like.shape[0],):
        raise ValueError("per-sample squared error has the wrong batch shape")
    log_variance = flow_map_log_variance(
        flow_map,
        like,
        start_time,
        end_time,
    )
    weighted = torch.exp(-log_variance) * per_sample_squared_error.float() + log_variance
    return weighted.mean(), log_variance


def diagonal_flow_matching_term(
    flow_map: nn.Module,
    x0: Tensor,
    x1: Tensor,
    time: Tensor | float,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Official diagonal grounding term for a linear interpolant."""

    interpolated, target_velocity = linear_interpolant(x0, x1, time)
    prediction = flow_map_velocity(flow_map, interpolated, time, time)
    per_sample = _per_sample_squared_error(prediction, target_velocity)
    loss, log_variance = uncertainty_weighted_loss(
        flow_map,
        per_sample,
        time,
        time,
        prediction,
    )
    return loss, {
        "loss_diagonal": loss.detach(),
        "diagonal_squared_error": per_sample.detach().mean(),
        "diagonal_log_variance_mean": log_variance.detach().mean(),
        "diagonal_velocity_rms": prediction.detach().square().mean().sqrt(),
        "diagonal_target_rms": target_velocity.detach().square().mean().sqrt(),
    }


def lagrangian_self_distillation_term(
    flow_map: nn.Module,
    x0: Tensor,
    x1: Tensor,
    start_time: Tensor | float,
    end_time: Tensor | float,
    *,
    teacher_flow_map: nn.Module | None = None,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Official convex-stop-gradient Lagrangian self-distillation term.

    By default the stopped target uses the current parameters, matching the
    official implementation.  A separate frozen/EMA teacher can be supplied
    as an explicit ablation, but is not silently enabled.
    """

    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    start = _time_vector(start_time, x0, name="start_time")
    end = _time_vector(end_time, x0, name="end_time")
    if bool((start > end).any()):
        raise ValueError("LSD requires start_time <= end_time for every sample")

    x_start, _ = linear_interpolant(x0, x1, start)
    prediction, endpoint_derivative = endpoint_time_jvp(
        flow_map,
        x_start,
        start,
        end,
        create_graph=True,
    )
    teacher = flow_map if teacher_flow_map is None else teacher_flow_map
    with torch.no_grad():
        target_velocity = flow_map_velocity(
            teacher,
            prediction.detach(),
            end,
            end,
        ).detach()
    per_sample = _per_sample_squared_error(endpoint_derivative, target_velocity)
    loss, log_variance = uncertainty_weighted_loss(
        flow_map,
        per_sample,
        start,
        end,
        endpoint_derivative,
    )
    return loss, {
        "loss_lsd": loss.detach(),
        "lsd_squared_error": per_sample.detach().mean(),
        "lsd_log_variance_mean": log_variance.detach().mean(),
        "lsd_endpoint_derivative_rms": endpoint_derivative.detach().square().mean().sqrt(),
        "lsd_target_velocity_rms": target_velocity.square().mean().sqrt(),
        "lsd_span_mean": (end - start).mean().detach(),
    }


def mixed_lsd_loss(
    flow_map: nn.Module,
    x0: Tensor,
    x1: Tensor,
    *,
    diagonal_fraction: float = 0.75,
    teacher_flow_map: nn.Module | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor | float | int]]:
    """Mix diagonal grounding and off-diagonal LSD in one minibatch.

    The default 75/25 split is the exact CelebA-64 recipe in the official
    release.  Losses are sums over sample coordinates followed by a mean over
    the full minibatch, again matching that code rather than PyTorch's default
    coordinate-wise MSE scaling.
    """

    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    if x0.ndim < 2 or x0.shape[0] < 2:
        raise ValueError("mixed LSD requires at least two batched samples")
    if not 0.0 < diagonal_fraction < 1.0:
        raise ValueError("diagonal_fraction must lie strictly between zero and one")

    batch_size = x0.shape[0]
    diagonal_count = max(1, int(batch_size * diagonal_fraction))
    diagonal_count = min(diagonal_count, batch_size - 1)
    offdiagonal_count = batch_size - diagonal_count

    diagonal_times = sample_diagonal_times(
        diagonal_count,
        x0,
        generator=generator,
    )
    diagonal_loss, diagonal_stats = diagonal_flow_matching_term(
        flow_map,
        x0[:diagonal_count],
        x1[:diagonal_count],
        diagonal_times,
    )
    start, end = sample_upper_triangle_times(
        offdiagonal_count,
        x0,
        generator=generator,
    )
    lsd_loss, lsd_stats = lagrangian_self_distillation_term(
        flow_map,
        x0[diagonal_count:],
        x1[diagonal_count:],
        start,
        end,
        teacher_flow_map=teacher_flow_map,
    )
    diagonal_weight = diagonal_count / batch_size
    offdiagonal_weight = offdiagonal_count / batch_size
    total = diagonal_weight * diagonal_loss + offdiagonal_weight * lsd_loss
    return total, {
        "loss": total.detach(),
        **diagonal_stats,
        **lsd_stats,
        "diagonal_fraction_requested": float(diagonal_fraction),
        "diagonal_count": diagonal_count,
        "offdiagonal_count": offdiagonal_count,
        "weighted_diagonal": (diagonal_weight * diagonal_loss).detach(),
        "weighted_lsd": (offdiagonal_weight * lsd_loss).detach(),
    }


__all__ = [
    "MagnitudePreservingTimeLogVariance",
    "ResidualTwoTimeFlowMap",
    "diagonal_flow_matching_term",
    "endpoint_time_jvp",
    "flow_map_log_variance",
    "flow_map_velocity",
    "lagrangian_self_distillation_term",
    "linear_interpolant",
    "mixed_lsd_loss",
    "sample_diagonal_times",
    "sample_upper_triangle_times",
    "uncertainty_weighted_loss",
]
