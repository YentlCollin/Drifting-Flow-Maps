"""Terminal GMD anchoring for the CelebA latent LSD flow map.

Variant D keeps the released 75/25 flow-matching/Lagrangian-
self-distillation objective intact and adds one endpoint objective

    L_D = L_LSD + lambda_GMD L_GMD(X_theta(x_0, 0, 1), x_data).

The GMD branch uses the frozen v2 MAE geometry and the exact released
``summed_feature_drifting_loss`` implementation.  Its geometry is evaluated
in float32 even when the LSD branch is autocast to bfloat16.  The feature
encoder is held by weak reference: it is frozen experiment state, not an
optimized/checkpointed part of the flow map.  The exported flow weights retain
the pure-LSD state-dict and architecture layout, but their hybrid signature is
intentionally rejected by the pure evaluator and requires the dedicated
variant-D evaluation pipeline.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from src.celeba_latent import unflatten_latents
from src.celeba_latent_lsd import CelebALatentLSDTrainingModel
from src.flow_map_self_distillation import mixed_lsd_loss
from src.official_drifting import OFFICIAL_TEMPERATURES, summed_feature_drifting_loss


@dataclass(frozen=True)
class HybridObjectiveTerms:
    """Differentiable terms from one outer hybrid forward."""

    total: Tensor
    lsd: Tensor
    gmd: Tensor
    statistics: Mapping[str, Tensor | float | int]


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def terminal_mae_gmd_loss(
    flow_map: nn.Module,
    x0: Tensor,
    data: Tensor,
    feature_encoder: nn.Module,
    *,
    temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
    distributed_statistics: bool = False,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Apply exact official GMD to ``X_theta(x0, 0, 1)``.

    Only the positive-data feature branch is stopped.  The frozen encoder is
    still differentiable with respect to the terminal generated sample, as
    required for the GMD force to reach ``theta``.  Particle support remains
    rank-local, matching the official-v2 CelebA baseline; only scalar scale
    and force-energy statistics are globally averaged under DDP.
    """

    if x0.shape != data.shape:
        raise ValueError("x0 and data must have identical shapes")
    if x0.ndim != 2 or x0.shape[1] != 4 * 8 * 8:
        raise ValueError("hybrid CelebA latents must have shape [B, 256]")
    if not callable(getattr(feature_encoder, "encode_features", None)):
        raise TypeError("feature_encoder must expose encode_features()")
    if distributed_statistics and (
        not dist.is_available() or not dist.is_initialized()
    ):
        raise RuntimeError("distributed GMD statistics require an initialized process group")

    # The official GMD image recipe evaluates both feature geometry and the
    # drifting force in FP32.  Disable an enclosing LSD BF16 autocast without
    # breaking the graph back to the flow-map parameters.
    with torch.autocast(device_type=x0.device.type, enabled=False):
        source = x0.float()
        positive = data.detach().float()
        zeros = torch.zeros(source.shape[0], device=source.device, dtype=source.dtype)
        terminal = flow_map(source, zeros, torch.ones_like(zeros))
        generated_features = feature_encoder.encode_features(
            unflatten_latents(terminal.float())
        )
        with torch.no_grad():
            positive_features = feature_encoder.encode_features(
                unflatten_latents(positive)
            )
        loss, raw_statistics = summed_feature_drifting_loss(
            generated_features,
            positive_features,
            temperatures=temperatures,
            distributed_statistics=distributed_statistics,
        )

    statistics: dict[str, Tensor | float] = {
        f"gmd/{key}": value for key, value in raw_statistics.items()
    }
    statistics.update(
        {
            "gmd/terminal_mean": terminal.detach().float().mean(),
            "gmd/terminal_rms": terminal.detach().float().square().mean().sqrt(),
            "gmd/feature_family_count": float(len(generated_features)),
            "gmd/positive_support_count": float(data.shape[0]),
            "gmd/generated_support_count": float(x0.shape[0]),
        }
    )
    return loss, statistics


class DriftLSDHybridTrainingModel(CelebALatentLSDTrainingModel):
    """Single DDP boundary around exact mixed LSD and terminal MAE-GMD.

    Subclassing the pure-LSD wrapper deliberately preserves its state-dict
    layout.  The weak reference also prevents ``AveragedModel`` from cloning
    the large frozen MAE or exporting it with the generator.
    """

    def __init__(
        self,
        flow_map: nn.Module,
        feature_encoder: nn.Module,
        *,
        gmd_lambda: float,
        temperatures: Sequence[float] = OFFICIAL_TEMPERATURES,
        distributed_statistics: bool = False,
    ) -> None:
        super().__init__(flow_map)
        if any(parameter.requires_grad for parameter in feature_encoder.parameters()):
            raise ValueError("the MAE feature encoder must be frozen")
        if feature_encoder.training:
            raise ValueError("the MAE feature encoder must be in eval mode")
        self.gmd_lambda = _positive_finite(gmd_lambda, name="gmd_lambda")
        parsed_temperatures = tuple(
            _positive_finite(value, name="GMD temperature") for value in temperatures
        )
        if not parsed_temperatures:
            raise ValueError("at least one GMD temperature is required")
        self.gmd_temperatures = parsed_temperatures
        self.distributed_gmd_statistics = bool(distributed_statistics)
        # nn.Module.__setattr__ would register the encoder and duplicate it in
        # checkpoints/EMA.  A weak reference keeps experiment geometry alive
        # without making it part of theta.
        object.__setattr__(self, "_feature_encoder_ref", weakref.ref(feature_encoder))

    @property
    def feature_encoder(self) -> nn.Module:
        encoder = self._feature_encoder_ref()
        if encoder is None:
            raise RuntimeError("the external frozen MAE feature encoder was released")
        return encoder

    def objective_terms(
        self,
        x0: Tensor,
        x1: Tensor,
        teacher_flow_map: nn.Module,
    ) -> HybridObjectiveTerms:
        lsd_loss, lsd_statistics = mixed_lsd_loss(
            self.flow_map,
            x0,
            x1,
            diagonal_fraction=0.75,
            teacher_flow_map=teacher_flow_map,
        )
        gmd_loss, gmd_statistics = terminal_mae_gmd_loss(
            self.flow_map,
            x0,
            x1,
            self.feature_encoder,
            temperatures=self.gmd_temperatures,
            distributed_statistics=self.distributed_gmd_statistics,
        )
        total = lsd_loss + self.gmd_lambda * gmd_loss
        statistics: dict[str, Tensor | float | int] = {
            **lsd_statistics,
            **gmd_statistics,
            "loss": total.detach(),
            "loss_mixed_lsd": lsd_loss.detach(),
            "loss_terminal_gmd": gmd_loss.detach(),
            "gmd_lambda": self.gmd_lambda,
            "weighted_terminal_gmd": (self.gmd_lambda * gmd_loss).detach(),
        }
        return HybridObjectiveTerms(total, lsd_loss, gmd_loss, statistics)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        teacher_flow_map: nn.Module,
    ) -> tuple[Tensor, Mapping[str, Tensor | float | int]]:
        terms = self.objective_terms(x0, x1, teacher_flow_map)
        return terms.total, terms.statistics


def distributed_component_gradient_moments(
    lsd_loss: Tensor,
    gmd_loss: Tensor,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    distributed: bool,
) -> dict[str, float | int]:
    """Measure component gradients after the same averaging as DDP.

    ``autograd.grad`` leaves ``parameter.grad`` untouched.  Each component
    gradient is explicitly averaged across ranks before norms and the cosine
    are accumulated, so the result describes the two vectors that DDP would
    contribute to an optimizer update—not a mean of unrelated local norms.
    """

    parameters = tuple(parameter for _, parameter in named_parameters if parameter.requires_grad)
    if not parameters:
        raise ValueError("at least one trainable parameter is required")
    if distributed and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError("distributed gradient calibration requires an initialized process group")

    lsd_gradients = torch.autograd.grad(
        lsd_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gmd_gradients = torch.autograd.grad(
        gmd_loss,
        parameters,
        allow_unused=True,
    )

    accumulator = torch.zeros(3, device=lsd_loss.device, dtype=torch.float64)
    shared_accumulator = torch.zeros(3, device=lsd_loss.device, dtype=torch.float64)
    lsd_active = 0
    gmd_active = 0
    shared_tensors = 0
    world_size = dist.get_world_size() if distributed else 1
    for parameter, lsd_gradient, gmd_gradient in zip(
        parameters, lsd_gradients, gmd_gradients, strict=True
    ):
        lsd_active += int(lsd_gradient is not None)
        gmd_active += int(gmd_gradient is not None)
        left = (
            torch.zeros_like(parameter, dtype=torch.float32)
            if lsd_gradient is None
            else lsd_gradient.detach().float().clone()
        )
        right = (
            torch.zeros_like(parameter, dtype=torch.float32)
            if gmd_gradient is None
            else gmd_gradient.detach().float().clone()
        )
        if distributed:
            dist.all_reduce(left, op=dist.ReduceOp.SUM)
            dist.all_reduce(right, op=dist.ReduceOp.SUM)
            left.div_(world_size)
            right.div_(world_size)
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise FloatingPointError("gradient calibration produced non-finite values")
        accumulator[0].add_(left.double().square().sum())
        accumulator[1].add_(right.double().square().sum())
        accumulator[2].add_((left.double() * right.double()).sum())
        # GMD intentionally cannot reach the independent log-variance head.
        # Calibrating against that disjoint LSD-only gradient would inflate
        # lambda.  Structural reachability (not a possibly zero value at the
        # zero-output initialization) defines the shared parameter subspace.
        if gmd_gradient is not None:
            shared_tensors += 1
            shared_accumulator[0].add_(left.double().square().sum())
            shared_accumulator[1].add_(right.double().square().sum())
            shared_accumulator[2].add_((left.double() * right.double()).sum())

    lsd_squared, gmd_squared, dot = (float(value) for value in accumulator.cpu())
    shared_lsd_squared, shared_gmd_squared, shared_dot = (
        float(value) for value in shared_accumulator.cpu()
    )
    lsd_norm = math.sqrt(max(lsd_squared, 0.0))
    gmd_norm = math.sqrt(max(gmd_squared, 0.0))
    shared_lsd_norm = math.sqrt(max(shared_lsd_squared, 0.0))
    shared_gmd_norm = math.sqrt(max(shared_gmd_squared, 0.0))
    cosine = dot / (lsd_norm * gmd_norm) if lsd_norm > 0 and gmd_norm > 0 else None
    shared_cosine = (
        shared_dot / (shared_lsd_norm * shared_gmd_norm)
        if shared_lsd_norm > 0 and shared_gmd_norm > 0
        else None
    )
    if distributed:
        counts = torch.tensor(
            [lsd_active, gmd_active, len(parameters), shared_tensors],
            device=lsd_loss.device,
            dtype=torch.int64,
        )
        minima = counts.clone()
        maxima = counts.clone()
        dist.all_reduce(minima, op=dist.ReduceOp.MIN)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        if not bool(torch.equal(minima, maxima)):
            raise RuntimeError(
                "component-gradient parameter reachability differs across DDP ranks"
            )
    return {
        "lsd_gradient_squared_norm": lsd_squared,
        "gmd_gradient_squared_norm": gmd_squared,
        "gradient_dot_product": dot,
        "lsd_gradient_norm": lsd_norm,
        "gmd_gradient_norm": gmd_norm,
        "gradient_cosine": cosine,
        "lsd_active_parameter_tensors": lsd_active,
        "gmd_active_parameter_tensors": gmd_active,
        "trainable_parameter_tensors": len(parameters),
        "shared_parameter_tensors": shared_tensors,
        "shared_lsd_gradient_squared_norm": shared_lsd_squared,
        "shared_gmd_gradient_squared_norm": shared_gmd_squared,
        "shared_gradient_dot_product": shared_dot,
        "shared_lsd_gradient_norm": shared_lsd_norm,
        "shared_gmd_gradient_norm": shared_gmd_norm,
        "shared_gradient_cosine": shared_cosine,
        "world_size": world_size,
    }


def equal_rms_gradient_calibration(
    batch_moments: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int | list[Mapping[str, float | int]]]:
    """Choose the unique lambda equating RMS component-gradient norms."""

    if not batch_moments:
        raise ValueError("gradient calibration requires at least one batch")
    all_lsd_squared = sum(
        float(item["lsd_gradient_squared_norm"]) for item in batch_moments
    )
    all_gmd_squared = sum(
        float(item["gmd_gradient_squared_norm"]) for item in batch_moments
    )
    lsd_squared = sum(
        float(item["shared_lsd_gradient_squared_norm"]) for item in batch_moments
    )
    gmd_squared = sum(
        float(item["shared_gmd_gradient_squared_norm"]) for item in batch_moments
    )
    dot = sum(float(item["shared_gradient_dot_product"]) for item in batch_moments)
    count = len(batch_moments)
    lsd_rms = math.sqrt(max(lsd_squared / count, 0.0))
    gmd_rms = math.sqrt(max(gmd_squared / count, 0.0))
    if not math.isfinite(lsd_rms) or lsd_rms <= 0:
        raise RuntimeError("LSD calibration gradient has zero or non-finite RMS norm")
    if not math.isfinite(gmd_rms) or gmd_rms <= 0:
        raise RuntimeError("GMD calibration gradient has zero or non-finite RMS norm")
    recommended = lsd_rms / gmd_rms
    cosine = dot / math.sqrt(lsd_squared * gmd_squared)
    if not math.isfinite(recommended) or recommended <= 0:
        raise RuntimeError("equal-RMS calibration produced an invalid positive lambda")
    if not math.isfinite(cosine):
        raise RuntimeError("aggregate component-gradient cosine is non-finite")
    return {
        "criterion": (
            "equal_RMS_norm_of_DDP_averaged_component_gradients_on_shared_parameters"
        ),
        "parameter_scope": "parameters_structurally_reached_by_both_LSD_and_GMD",
        "calibration_batches": count,
        "lsd_gradient_rms_norm": lsd_rms,
        "gmd_gradient_rms_norm": gmd_rms,
        "all_parameter_lsd_gradient_rms_norm": math.sqrt(all_lsd_squared / count),
        "all_parameter_gmd_gradient_rms_norm": math.sqrt(all_gmd_squared / count),
        "aggregate_gradient_cosine": cosine,
        "lambda_equal_rms": recommended,
        "scaled_gmd_to_lsd_rms_ratio": recommended * gmd_rms / lsd_rms,
        "batches": list(batch_moments),
    }


__all__ = [
    "DriftLSDHybridTrainingModel",
    "HybridObjectiveTerms",
    "distributed_component_gradient_moments",
    "equal_rms_gradient_calibration",
    "terminal_mae_gmd_loss",
]
