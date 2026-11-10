"""Two-time flow-map adapters for the validated CelebA-64 GMD generator.

The one-step GMD generator is a strong deterministic transport from its
Gaussian RGB prior to CelebA.  This module embeds that transport in the
identity-grounded two-time parameterisation

    Phi(x, s, t) = x + alpha(s,t) * (G_theta(x, labels, s, t) - x),

where ``alpha(s,t)=(t-s)/(1-s)``.  The endpoint network therefore predicts
the destination at physical time one from the current state, while ``alpha``
interpolates consistently over the remaining interval.

Consequently ``Phi(x, t, t) == x`` by construction and a model initialised
from a one-step checkpoint satisfies ``Phi(z, 0, 1) == G(z)`` exactly (up to
the normal floating-point execution of the shared backbone).  The additional
time branch is zero-initialised, so diagnostics can measure the behaviour of
composition before any two-time fine-tuning without sacrificing the validated
NFE=1 baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.celeba_pixel_gmd_generator import (
    CelebA64DirectPixelGMDDiT,
    TimestepEmbedder,
    _validate_inference_autocast,
)


FLOW_ARCHITECTURE_NAME = "CelebA64TwoTimeGMDDiT"
FLOW_ARCHITECTURE_VERSION = 2


def _time_vector(value: Tensor | float, like: Tensor, *, name: str) -> Tensor:
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
        raise ValueError(f"{name} must be finite")
    return result


def _validated_times(
    start_time: Tensor | float,
    end_time: Tensor | float,
    like: Tensor,
) -> tuple[Tensor, Tensor]:
    start = _time_vector(start_time, like, name="start_time")
    end = _time_vector(end_time, like, name="end_time")
    if bool((start < 0).any()) or bool((end > 1).any()):
        raise ValueError("flow-map times must lie in [0, 1]")
    if bool((start > end).any()):
        raise ValueError("flow-map start_time must not exceed end_time")
    return start, end


def _broadcast_time(value: Tensor, like: Tensor) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))


def _endpoint_fraction(start: Tensor, end: Tensor) -> Tensor:
    """Return ``(end-start)/(1-start)`` with an exact diagonal at ``s=t=1``."""

    span = end - start
    denominator = 1.0 - start
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    fraction = span / safe_denominator
    return torch.where(span == 0, torch.zeros_like(fraction), fraction)


class CelebA64TwoTimeGMDDiT(CelebA64DirectPixelGMDDiT):
    """Time-conditioned flow map warm-started from the one-step GMD DiT.

    The complete one-step backbone remains trainable.  A dedicated Fourier
    embedding of ``(s, t, t-s)`` is injected into its existing AdaLN condition.
    The final projection is initialised to zero, which makes the initial
    endpoint network independent of time and preserves the parent generator.
    """

    def __init__(
        self,
        *,
        flow_time_frequency_embedding_size: int = 256,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__(**backbone_kwargs)
        if (
            isinstance(flow_time_frequency_embedding_size, bool)
            or not isinstance(flow_time_frequency_embedding_size, int)
            or flow_time_frequency_embedding_size < 2
        ):
            raise ValueError("flow-time frequency embedding size must be an integer >= 2")
        self.flow_time_frequency_embedding_size = flow_time_frequency_embedding_size
        self.flow_time_embedder = TimestepEmbedder(
            self.hidden_size,
            frequency_embedding_size=flow_time_frequency_embedding_size,
        )
        self.flow_time_projection = nn.Linear(3 * self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.flow_time_projection.weight)
        nn.init.zeros_(self.flow_time_projection.bias)

    @classmethod
    def from_one_step(
        cls,
        generator: CelebA64DirectPixelGMDDiT,
        *,
        flow_time_frequency_embedding_size: int = 256,
    ) -> "CelebA64TwoTimeGMDDiT":
        """Create an exact NFE=1 warm start from a validated generator."""

        if not isinstance(generator, CelebA64DirectPixelGMDDiT):
            raise TypeError("generator must be CelebA64DirectPixelGMDDiT")
        architecture = generator.architecture_config()
        result = cls(
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
            intended_inference_precision=architecture[
                "intended_inference_precision"
            ],
            flow_time_frequency_embedding_size=(
                flow_time_frequency_embedding_size
            ),
        )
        incompatible = result.load_state_dict(generator.state_dict(), strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"one-step warm start has unexpected keys: {incompatible.unexpected_keys}"
            )
        if not incompatible.missing_keys or any(
            not key.startswith(("flow_time_embedder.", "flow_time_projection."))
            for key in incompatible.missing_keys
        ):
            raise RuntimeError(
                "one-step warm start must leave only the new flow-time branch uninitialised"
            )
        if bool(result.flow_time_projection.weight.detach().count_nonzero()) or bool(
            result.flow_time_projection.bias.detach().count_nonzero()
        ):
            raise RuntimeError("flow-time projection did not remain exactly zero")
        return result

    def flow_architecture_config(self) -> dict[str, Any]:
        return {
            "name": FLOW_ARCHITECTURE_NAME,
            "version": FLOW_ARCHITECTURE_VERSION,
            "parameterization": "x_plus_(t-s)/(1-s)_times_endpoint_minus_x",
            "time_domain": [0.0, 1.0],
            "time_conditioning": "fourier_s_t_delta_then_zero_initialized_projection",
            "flow_time_frequency_embedding_size": (
                self.flow_time_frequency_embedding_size
            ),
            "one_step_backbone": self.architecture_config(),
        }

    def _flow_condition(
        self,
        noise_labels: Tensor,
        start: Tensor,
        end: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        condition = self._condition(noise_labels, dtype=dtype)
        time_features = torch.cat(
            (
                self.flow_time_embedder(start),
                self.flow_time_embedder(end),
                self.flow_time_embedder(end - start),
            ),
            dim=-1,
        )
        time_condition = self.flow_time_projection(time_features.float())
        return condition + time_condition.to(dtype=dtype)

    def endpoint_candidate(
        self,
        state: Tensor,
        noise_labels: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
    ) -> Tensor:
        """Return the unconstrained endpoint candidate ``G_theta``."""

        start, end = _validated_times(start_time, end_time, state)
        if not self.training:
            _validate_inference_autocast(self, state.device)
        values = self.patch_projection(self._patchify(state))
        values_dtype = values.dtype
        values = (values.float() + self.position_embedding.float()).to(
            dtype=values_dtype
        )
        condition = self._flow_condition(
            noise_labels,
            start,
            end,
            dtype=values.dtype,
        )
        if self.condition_tokens:
            condition_values = self.condition_projection(condition)[:, None].expand(
                -1, self.condition_tokens, -1
            )
            condition_values_dtype = condition_values.dtype
            condition_values = (
                condition_values.float() + self.condition_token_embedding.float()
            ).to(dtype=condition_values_dtype)
            values = torch.cat((condition_values, values), dim=1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                values = activation_checkpoint(
                    block, values, condition, use_reentrant=False
                )
            else:
                values = block(values, condition)
        values = self.final(values, condition)
        if self.condition_tokens:
            values = values[:, self.condition_tokens :]
        return self._unpatchify(values)

    def forward(
        self,
        state: Tensor,
        noise_labels: Tensor,
        start_time: Tensor | float,
        end_time: Tensor | float,
        *,
        output_kind: str = "map",
    ) -> Tensor:
        start, end = _validated_times(start_time, end_time, state)
        candidate = self.endpoint_candidate(state, noise_labels, start, end)
        velocity = candidate - state
        if output_kind == "endpoint":
            return candidate
        if output_kind == "velocity":
            return velocity
        if output_kind != "map":
            raise ValueError("output_kind must be 'map', 'velocity', or 'endpoint'")
        fraction = _broadcast_time(_endpoint_fraction(start, end), state)
        mapped = state + fraction * velocity
        # ``x + (g - x)`` can differ from ``g`` by one FP32 rounding bit even
        # when alpha is exactly one.  Select the endpoint explicitly so that
        # the converted NFE=1 model preserves the parent pixels bit-for-bit.
        return torch.where(fraction == 1, candidate.to(mapped.dtype), mapped)


def sample_two_time_flow(
    model: CelebA64TwoTimeGMDDiT,
    prior: Tensor,
    noise_labels: Tensor,
    *,
    nfe: int,
    time_power: float = 1.0,
) -> Tensor:
    """Compose ``Phi`` on a deterministic partition of ``[0,1]``."""

    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 1:
        raise ValueError("nfe must be a positive integer")
    if not math.isfinite(float(time_power)) or time_power <= 0:
        raise ValueError("time_power must be finite and positive")
    times = torch.linspace(
        0.0,
        1.0,
        nfe + 1,
        device=prior.device,
        dtype=torch.float32,
    ).pow(float(time_power))
    state = prior
    for start, end in zip(times[:-1], times[1:], strict=True):
        state = model(state, noise_labels, start, end)
    return state


def load_one_step_state(
    checkpoint: Mapping[str, Any],
    *,
    weights: str,
) -> CelebA64TwoTimeGMDDiT:
    """Load a baseline training checkpoint as an untrained two-time map."""

    if weights not in {"model", "ema"}:
        raise ValueError("weights must be 'model' or 'ema'")
    required = {"training_signature", "completed_step", weights}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint lacks the baseline GMD training fields")
    signature = checkpoint["training_signature"]
    if not isinstance(signature, Mapping) or not isinstance(
        signature.get("generator"), Mapping
    ):
        raise ValueError("checkpoint lacks a generator architecture signature")
    from src.celeba_pixel_gmd_generator import generator_from_architecture

    one_step = generator_from_architecture(signature["generator"])
    one_step.load_state_dict(checkpoint[weights], strict=True)
    return CelebA64TwoTimeGMDDiT.from_one_step(one_step)


__all__ = [
    "FLOW_ARCHITECTURE_NAME",
    "FLOW_ARCHITECTURE_VERSION",
    "CelebA64TwoTimeGMDDiT",
    "load_one_step_state",
    "sample_two_time_flow",
]
