from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.celeba_latent import CELEBA_LATENT_DIM, TransitionConditionedLatentUNet
from src.flow_map_self_distillation import (
    MagnitudePreservingTimeLogVariance,
    ResidualTwoTimeFlowMap,
    diagonal_flow_matching_term,
    endpoint_time_jvp,
    lagrangian_self_distillation_term,
    linear_interpolant,
    mixed_lsd_loss,
    sample_upper_triangle_times,
    uncertainty_weighted_loss,
)


class ConstantVelocity(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.velocity_parameter = nn.Parameter(velocity.clone())

    def forward(self, x, start_time, end_time):
        del start_time, end_time
        return self.velocity_parameter.expand_as(x)


class AffineVelocity(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension + 2, dimension, bias=False)

    def forward(self, x, start_time, end_time):
        return self.linear(torch.cat((x, start_time[:, None], end_time[:, None]), dim=1))


class ForwardOnlyProxy(nn.Module):
    """Exercise the same public surface exposed by DistributedDataParallel."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_residual_parameterization_is_identity_on_diagonal_and_composes_translation():
    velocity = torch.tensor([[0.5, -1.0, 2.0]])
    flow = ResidualTwoTimeFlowMap(ConstantVelocity(velocity))
    x = torch.randn(5, 3)
    time = torch.linspace(0, 1, 5)

    assert torch.equal(flow(x, time, time), x)
    direct = flow(x, 0.0, 1.0)
    halfway = flow(x, 0.0, 0.4)
    composed = flow(halfway, 0.4, 1.0)
    torch.testing.assert_close(direct, composed)


def test_endpoint_jvp_equals_constant_diagonal_velocity():
    velocity = torch.tensor([[0.25, -0.75]])
    flow = ResidualTwoTimeFlowMap(ConstantVelocity(velocity))
    x = torch.randn(4, 2)
    start = torch.tensor([0.0, 0.1, 0.3, 0.8])
    end = torch.tensor([0.2, 0.4, 0.9, 1.0])

    prediction, derivative = endpoint_time_jvp(
        flow, x, start, end, create_graph=True
    )
    torch.testing.assert_close(prediction, flow(x, start, end))
    torch.testing.assert_close(derivative, velocity.expand_as(x))


def test_linear_interpolant_and_diagonal_term_are_exact_for_translation_coupling():
    velocity = torch.tensor([[1.0, -2.0, 0.5]])
    flow = ResidualTwoTimeFlowMap(ConstantVelocity(velocity))
    x0 = torch.randn(8, 3)
    x1 = x0 + velocity
    times = torch.linspace(0.0, 1.0, 8)

    interpolated, target = linear_interpolant(x0, x1, times)
    torch.testing.assert_close(
        interpolated,
        x0 + times[:, None] * velocity,
    )
    torch.testing.assert_close(target, velocity.expand_as(x0))
    loss, stats = diagonal_flow_matching_term(flow, x0, x1, times)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert float(stats["loss_diagonal"]) == pytest.approx(0.0, abs=1e-12)


def test_lsd_term_is_exact_for_constant_translation_flow():
    velocity = torch.tensor([[0.5, -1.0]])
    flow = ResidualTwoTimeFlowMap(ConstantVelocity(velocity))
    x0 = torch.randn(6, 2)
    x1 = x0 + velocity
    start = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    end = torch.tensor([0.2, 0.4, 0.8, 0.9, 0.75, 1.0])

    loss, stats = lagrangian_self_distillation_term(
        flow, x0, x1, start, end
    )
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert float(stats["loss_lsd"]) == pytest.approx(0.0, abs=1e-12)


def test_lsd_stop_gradient_still_trains_student_derivative():
    torch.manual_seed(4)
    flow = ResidualTwoTimeFlowMap(AffineVelocity(3))
    x0 = torch.randn(7, 3)
    x1 = torch.randn(7, 3)
    start = torch.linspace(0.0, 0.5, 7)
    end = torch.linspace(0.4, 1.0, 7)

    loss, _ = lagrangian_self_distillation_term(flow, x0, x1, start, end)
    loss.backward()
    gradient = flow.velocity_network.linear.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 0


def test_upper_triangle_sampler_matches_ordering_and_range():
    generator = torch.Generator().manual_seed(12)
    like = torch.empty(10_000, 2)
    start, end = sample_upper_triangle_times(10_000, like, generator=generator)
    assert bool(((0 <= start) & (start <= end) & (end <= 1)).all())
    # Ordered uniforms have E[min]=1/3 and E[max]=2/3.
    assert float(start.mean()) == pytest.approx(1 / 3, abs=0.015)
    assert float(end.mean()) == pytest.approx(2 / 3, abs=0.015)


def test_uncertainty_weighting_matches_exp_minus_w_error_plus_w():
    class FixedLogVariance(nn.Module):
        def forward(self, start_time, end_time):
            del end_time
            return torch.full_like(start_time, 0.25)

    flow = ResidualTwoTimeFlowMap(
        ConstantVelocity(torch.zeros(1, 2)),
        FixedLogVariance(),
    )
    errors = torch.tensor([1.0, 2.0, 4.0])
    like = torch.zeros(3, 2)
    loss, log_variance = uncertainty_weighted_loss(flow, errors, 0.1, 0.7, like)
    expected = (torch.exp(torch.tensor(-0.25)) * errors + 0.25).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(log_variance, torch.full((3,), 0.25))


def test_magnitude_preserving_log_variance_head_is_finite_and_trainable():
    torch.manual_seed(2)
    head = MagnitudePreservingTimeLogVariance(channels=16)
    start = torch.linspace(0.0, 0.5, 9)
    end = torch.linspace(0.4, 1.0, 9)
    values = head(start, end)
    assert values.shape == (9,)
    assert bool(torch.isfinite(values).all())
    values.square().mean().backward()
    assert head.weight.grad is not None
    assert bool(torch.isfinite(head.weight.grad).all())


def test_magnitude_preserving_log_variance_projection_matches_edm2_rule():
    head = MagnitudePreservingTimeLogVariance(channels=16)
    with torch.no_grad():
        head.weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(1, 16))
        before = head.weight.clone()
    expected = before / (
        1e-4 + before.square().sum(dim=1, keepdim=True).sqrt() / math.sqrt(16)
    )
    head.project_weights_()
    torch.testing.assert_close(head.weight, expected)


def test_mixed_loss_uses_exact_75_25_split_and_backpropagates():
    torch.manual_seed(7)
    flow = ResidualTwoTimeFlowMap(AffineVelocity(4))
    x0 = torch.randn(64, 4)
    x1 = torch.randn(64, 4)

    loss, stats = mixed_lsd_loss(
        flow,
        x0,
        x1,
        generator=torch.Generator().manual_seed(8),
    )
    assert stats["diagonal_count"] == 48
    assert stats["offdiagonal_count"] == 16
    torch.testing.assert_close(
        loss.detach(),
        stats["weighted_diagonal"] + stats["weighted_lsd"],
    )
    loss.backward()
    gradient = flow.velocity_network.linear.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 0


def test_mixed_loss_runs_through_the_existing_celeba_latent_unet_endpoint():
    torch.manual_seed(11)
    backbone = TransitionConditionedLatentUNet(
        base_channels=8,
        time_embedding_dim=8,
        condition_dim=16,
        zero_init_endpoint=True,
        endpoint_parameterization="absolute",
    )
    flow = ResidualTwoTimeFlowMap(backbone)
    x0 = torch.randn(4, CELEBA_LATENT_DIM)
    x1 = torch.randn_like(x0)

    loss, stats = mixed_lsd_loss(
        flow,
        x0,
        x1,
        generator=torch.Generator().manual_seed(13),
    )
    assert bool(torch.isfinite(loss))
    assert stats["diagonal_count"] == 3
    assert stats["offdiagonal_count"] == 1
    loss.backward()
    gradient = backbone.output_conv.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 0


def test_flow_map_rejects_celeba_backbone_with_residual_endpoint_semantics():
    backbone = TransitionConditionedLatentUNet(
        base_channels=8,
        time_embedding_dim=8,
        condition_dim=16,
        endpoint_parameterization="residual",
    )
    with pytest.raises(ValueError, match="endpoint_parameterization='absolute'"):
        ResidualTwoTimeFlowMap(backbone)


def test_mixed_loss_uses_only_forward_methods_and_is_ddp_proxy_safe():
    torch.manual_seed(17)
    wrapped = ResidualTwoTimeFlowMap(AffineVelocity(3))
    proxy = ForwardOnlyProxy(wrapped)
    x0 = torch.randn(8, 3)
    x1 = torch.randn_like(x0)

    loss, stats = mixed_lsd_loss(
        proxy,
        x0,
        x1,
        generator=torch.Generator().manual_seed(18),
    )
    assert bool(torch.isfinite(loss))
    assert stats["diagonal_count"] == 6
    loss.backward()
    assert wrapped.velocity_network.linear.weight.grad is not None


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_mixed_loss_rejects_invalid_fraction(fraction):
    flow = ResidualTwoTimeFlowMap(ConstantVelocity(torch.zeros(1, 2)))
    with pytest.raises(ValueError, match="diagonal_fraction"):
        mixed_lsd_loss(flow, torch.zeros(4, 2), torch.ones(4, 2), diagonal_fraction=fraction)
