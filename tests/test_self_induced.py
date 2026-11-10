import copy
from dataclasses import replace

import pytest
import torch
from flowmaps.models.rgb import CelebA64TwoTimeGMDDiT
from flowmaps.objectives.drifting import official_drifting_loss
from flowmaps.objectives.local_drifting import two_view_local_feature_gmd_loss
from flowmaps.objectives.self_induced import (
    FACTORS,
    SelfInducedSignals,
    SignalWeights,
    factorial_variants,
    masked_weights,
    sample_times,
    signal_objective,
)
from torch import nn


def tiny_flow():
    return CelebA64TwoTimeGMDDiT(
        patch_size=32,
        hidden_size=16,
        depth=1,
        num_heads=4,
        noise_coords=2,
        noise_classes=8,
        condition_tokens=0,
        gradient_checkpointing=False,
        intended_inference_precision="float32",
    )


class PolynomialFlow(nn.Module):
    def __init__(self, affine=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.affine = affine
        self.calls = []

    def forward(self, x, labels, a, b):
        y = x + (b - a) * self.weight * (
            torch.ones_like(x) if self.affine else (1 + x.square())
        )
        self.calls.append((a, b, x.requires_grad, torch.is_grad_enabled(), labels))
        return y


def wrapper(weights, affine=False):
    result = SelfInducedSignals(tiny_flow(), weights)
    # Replace only in the mathematical fixture, preserving production's strict
    # serializable DiT requirement in SelfInducedSignals.__init__.
    result.flow = PolynomialFlow(affine)
    return result


def mini_features(x):
    return {"f": x[:, None, :]}


@pytest.fixture
def feature_losses(monkeypatch):
    def path(
        encoder,
        source,
        origin,
        local,
        positives,
        negatives,
        *,
        force_step,
        temperatures,
    ):
        return two_view_local_feature_gmd_loss(
            mini_features(source),
            mini_features(origin),
            mini_features(local),
            mini_features(positives),
            fixed_negative_features=mini_features(negatives),
            force_step=force_step,
            temperatures=temperatures,
            synchronize_distributed=False,
        )

    def boundary(encoder, generated, positives, negatives, *, temperatures):
        return official_drifting_loss(
            generated[None], positives[None], temperatures=temperatures
        )

    monkeypatch.setattr(
        "flowmaps.objectives.local_drifting.global_sharded_mae_convnext_two_view_loss",
        path,
    )
    monkeypatch.setattr(
        "flowmaps.features.distributed.global_sharded_mae_convnext_loss", boundary
    )


@pytest.mark.parametrize("mask", list(factorial_variants()))
def test_all_combinations_have_exact_active_components_and_finite_gradients(
    mask, feature_losses
):
    weights = masked_weights(mask)
    model = wrapper(weights)
    z = torch.tensor([[0.2, -0.5], [0.6, 0.1], [-0.3, 0.8], [0.9, -0.1]])
    labels = torch.zeros(4, 2, dtype=torch.int64)
    outputs = model(z, labels, 0.2, 0.6, 0.4)
    loss, parts = signal_objective(
        outputs,
        weights,
        None,
        z + 1,
        z - 1,
        force_step=0.5,
        temperatures=(0.2, 0.05, 0.02),
    )
    assert set(parts) == {k for k, bit in factorial_variants()[mask].items() if bit}
    assert torch.isfinite(loss)
    loss.backward()
    assert model.flow.weight.grad is not None and torch.isfinite(model.flow.weight.grad)
    if "source" in outputs:
        assert not outputs["source"].requires_grad
    assert all(call[-1] is labels for call in model.flow.calls)


def test_exact_flow_annuls_each_structural_loss():
    outputs = wrapper(masked_weights("011100"), affine=True)(
        torch.randn(5, 2), torch.zeros(5, 2, dtype=torch.int64), 0.2, 0.6, 0.4
    )
    for key in ("lagrangian", "eulerian", "composition"):
        assert outputs[key].item() < 1e-12


def test_arbitrary_boundary_detaches_its_source_and_never_trains_rollout():
    model = wrapper(masked_weights("000001"))
    model(
        torch.randn(4, 2, requires_grad=True),
        torch.zeros(4, 2, dtype=torch.int64),
        0.2,
        0.6,
        0.4,
    )
    calls = model.flow.calls
    assert [(a, b, grad) for a, b, _, grad, _ in calls] == [
        (0.0, 0.4, False),
        (0.4, 1.0, True),
    ]
    assert calls[1][2] is False


def test_full_objective_has_no_four_transition_training_chain():
    model = wrapper(SignalWeights())
    z = torch.randn(4, 2)
    labels = torch.zeros(4, 2, dtype=torch.int64)
    model(z, labels, 0.2, 0.6, 0.4)
    # Only E composes two live transitions. Every other live call starts
    # from an original prior or a detached state; neither boundary chains.
    live_chained = [
        (a, b)
        for a, b, input_grad, enabled, _ in model.flow.calls
        if input_grad and enabled
    ]
    assert live_chained == [(0.25, 0.6)]


def test_times_are_valid_homogeneous_and_resume_exactly():
    g = torch.Generator().manual_seed(42)
    for _ in range(100):
        a, b, c = sample_times(g)
        assert 0 <= a < a + 0.05 <= b < b + 0.05 <= 1
        assert 0 <= c < 1
    state = g.get_state()
    draws = [sample_times(g) for _ in range(10)]
    g.set_state(state)
    assert draws == [sample_times(g) for _ in range(10)]


def test_factorial_covers_mnist_knockouts_and_rejects_no_objective():
    cells = factorial_variants()
    assert len(cells) == 63
    assert set(cells) == {f"{i:06b}" for i in range(1, 64)}
    for k in FACTORS:
        assert sum(cell[k] for cell in cells.values()) == 32
    with pytest.raises(ValueError):
        masked_weights("000000")


def test_feature_path_has_source_marginal_force_not_terminal_force(feature_losses):
    model = wrapper(masked_weights("100000"))
    z = torch.randn(4, 3)
    out = model(z, torch.zeros(4, 2, dtype=torch.int64), 0.2, 0.6, 0.4)
    assert "boundary_origin" not in out and "boundary_arbitrary" not in out
    reference, _ = two_view_local_feature_gmd_loss(
        mini_features(out["source"]),
        mini_features(out["origin"]),
        mini_features(out["local"]),
        mini_features(z + 1),
        fixed_negative_features=mini_features(z - 1),
        force_step=0.5,
        synchronize_distributed=False,
    )
    _, components = signal_objective(
        out,
        masked_weights("100000"),
        None,
        z + 1,
        z - 1,
        force_step=0.5,
        temperatures=(0.2, 0.05, 0.02),
    )
    torch.testing.assert_close(components["path"], 2 * reference)
