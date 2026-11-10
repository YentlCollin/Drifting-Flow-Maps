from __future__ import annotations

import pytest
import torch
from flowmaps.models.backbone import CelebA64DirectPixelGMDDiT
from flowmaps.models.rgb import CelebA64TwoTimeGMDDiT
from flowmaps.objectives.drifting import (
    OFFICIAL_TEMPERATURES,
    official_drifting_loss,
    token_features_to_groups,
)
from flowmaps.objectives.local_drifting import (
    homogeneous_two_view_states,
    official_feature_gmd_teacher,
    two_view_local_feature_gmd_loss,
)
from torch import nn


class CallSeparatedFlow(nn.Module):
    """A separate parameter per call makes stopped-teacher gradients visible."""

    def __init__(self) -> None:
        super().__init__()
        self.gains = nn.ParameterList(
            nn.Parameter(torch.tensor(value)) for value in (0.2, 0.3, 0.4)
        )
        self.calls: list[tuple[object, object, bool]] = []

    def forward(self, state, labels, start, end):
        del labels
        index = len(self.calls)
        start_tensor = torch.as_tensor(start, device=state.device, dtype=state.dtype)
        end_tensor = torch.as_tensor(end, device=state.device, dtype=state.dtype)
        output = state + (end_tensor - start_tensor) * self.gains[index]
        self.calls.append((start, end, output.requires_grad))
        return output


def _feature_batches(*, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(9107)
    source = torch.randn(5, 3, 4, generator=generator, requires_grad=requires_grad)
    origin = source.detach().clone().requires_grad_(True)
    local = source.detach().clone().requires_grad_(True)
    positive = torch.randn(7, 3, 4, generator=generator, requires_grad=requires_grad)
    negative = torch.randn(2, 3, 4, generator=generator, requires_grad=requires_grad)
    return (
        {"tokens": source},
        {"tokens": origin},
        {"tokens": local},
        {"tokens": positive},
        {"tokens": negative},
    )


def test_homogeneous_state_builder_stops_teacher_and_prior_gradients() -> None:
    flow = CallSeparatedFlow()
    prior = torch.randn(4, 2, requires_grad=True)
    labels = torch.zeros(4, 1, dtype=torch.int64)

    views = homogeneous_two_view_states(
        flow,
        prior,
        labels,
        start_time=0.25,
        time_fraction=0.4,
    )

    assert float(views.start_time) == pytest.approx(0.25)
    assert float(views.end_time) == pytest.approx(0.55)
    assert not views.source.requires_grad
    assert views.origin_student.requires_grad
    assert views.local_student.requires_grad
    assert [call[2] for call in flow.calls] == [False, True, True]

    (views.origin_student.mean() + views.local_student.mean()).backward()
    assert prior.grad is None
    assert flow.gains[0].grad is None
    assert flow.gains[1].grad is not None and float(flow.gains[1].grad.abs()) > 0
    assert flow.gains[2].grad is not None and float(flow.gains[2].grad.abs()) > 0


def test_state_builder_preserves_autocast_checkpoint_recomputation() -> None:
    one_step = CelebA64DirectPixelGMDDiT(
        patch_size=16,
        hidden_size=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        noise_classes=4,
        noise_coords=2,
        condition_tokens=2,
        gradient_checkpointing=True,
        intended_inference_precision="float32",
    )
    flow = CelebA64TwoTimeGMDDiT.from_one_step(one_step).train()
    prior, labels = flow.sample_prior(
        2,
        device="cpu",
        generator=torch.Generator().manual_seed(11),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        views = homogeneous_two_view_states(
            flow,
            prior,
            labels,
            start_time=0.3,
            time_fraction=0.4,
        )
        loss = views.origin_student.float().square().mean()
        loss = loss + views.local_student.float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in flow.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize(
    ("start_time", "time_fraction", "message"),
    [
        (torch.tensor([0.1, 0.2]), 0.5, "start_time"),
        (1.0, 0.5, "start_time"),
        (0.2, torch.tensor([0.3, 0.4]), "time_fraction"),
        (0.2, 0.0, "time_fraction"),
        (0.2, 1.01, "time_fraction"),
    ],
)
def test_state_builder_rejects_heterogeneous_or_invalid_times(
    start_time, time_fraction, message
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        homogeneous_two_view_states(
            CallSeparatedFlow(),
            torch.zeros(2, 3),
            torch.zeros(2, 1, dtype=torch.int64),
            start_time=start_time,
            time_fraction=time_fraction,
        )


def test_force_step_one_matches_official_loss_and_semigradient_at_source() -> None:
    source, _, _, positive, negative = _feature_batches()
    shared_student = source["tokens"].detach().clone().requires_grad_(True)

    actual, statistics = two_view_local_feature_gmd_loss(
        source,
        {"tokens": shared_student},
        {"tokens": shared_student},
        positive,
        fixed_negative_features=negative,
        force_step=1.0,
        temperatures=OFFICIAL_TEMPERATURES,
    )
    actual.backward()
    actual_gradient = shared_student.grad.detach().clone()

    direct_student = source["tokens"].detach().clone().requires_grad_(True)
    direct_groups = token_features_to_groups(direct_student)
    negative_groups = token_features_to_groups(negative["tokens"])
    direct_group_loss, _ = official_drifting_loss(
        direct_groups,
        token_features_to_groups(positive["tokens"]),
        negative_groups,
        negative_weight=torch.zeros(negative_groups.shape[:2]),
        temperatures=OFFICIAL_TEMPERATURES,
    )
    expected = direct_group_loss.mean()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        actual_gradient, direct_student.grad, rtol=3e-6, atol=3e-7
    )
    torch.testing.assert_close(
        statistics["loss/source_geometry/tokens"], expected.detach()
    )


def test_only_origin_and_local_students_receive_feature_gradients() -> None:
    source, origin, local, positive, negative = _feature_batches(requires_grad=True)
    loss, statistics = two_view_local_feature_gmd_loss(
        source,
        origin,
        local,
        positive,
        fixed_negative_features=negative,
        force_step=0.4,
        temperatures=(0.05, 0.2),
    )
    loss.backward()

    assert origin["tokens"].grad is not None
    assert local["tokens"].grad is not None
    assert bool(torch.isfinite(origin["tokens"].grad).all())
    assert bool(torch.isfinite(local["tokens"].grad).all())
    assert float(origin["tokens"].grad.abs().sum()) > 0
    assert float(local["tokens"].grad.abs().sum()) > 0
    assert source["tokens"].grad is None
    assert positive["tokens"].grad is None
    assert negative["tokens"].grad is None
    assert all(not value.requires_grad for value in statistics.values())
    assert float(statistics["support/fixed_negative_weight_sum/tokens"]) == 0.0


def test_local_identity_gradient_scales_as_inverse_force_step() -> None:
    source, _, _, positive, negative = _feature_batches()
    gradients = []
    losses = []
    for force_step in (1.0, 0.25):
        shared = source["tokens"].detach().clone().requires_grad_(True)
        loss, _ = two_view_local_feature_gmd_loss(
            source,
            {"tokens": shared},
            {"tokens": shared},
            positive,
            fixed_negative_features=negative,
            force_step=force_step,
            temperatures=(0.2,),
        )
        loss.backward()
        losses.append(loss.detach())
        gradients.append(shared.grad.detach())

    # At zero displacement the residual is -F for every step, while
    # d[(Y-X)/(q*step)]/dY contributes exactly 1/step.
    torch.testing.assert_close(losses[0], losses[1], rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradients[1], 4.0 * gradients[0], rtol=4e-6, atol=4e-7)


def test_exact_teacher_pseudostep_is_zero_residual_for_both_views() -> None:
    source, _, _, positive, negative = _feature_batches()
    teacher = official_feature_gmd_teacher(
        source,
        positive,
        fixed_negative_features=negative,
        temperatures=(0.02, 0.05, 0.2),
    )
    step = 0.3
    source_groups = token_features_to_groups(source["tokens"]).float()
    target_groups = (
        source_groups + step * teacher.input_scales["tokens"] * teacher.forces["tokens"]
    )
    target = target_groups.permute(1, 0, 2).contiguous().requires_grad_(True)

    loss, _ = two_view_local_feature_gmd_loss(
        source,
        {"tokens": target},
        {"tokens": target},
        positive,
        fixed_negative_features=negative,
        force_step=step,
        temperatures=(0.02, 0.05, 0.2),
        teacher=teacher,
    )
    assert float(loss.detach()) < 2e-12
    loss.backward()
    assert target.grad is not None
    assert float(target.grad.abs().max()) < 2e-6


def test_zero_weight_fixed_negative_still_changes_official_softmax_force() -> None:
    source, _, _, positive, negative = _feature_batches()
    with_negative = official_feature_gmd_teacher(
        source,
        positive,
        fixed_negative_features=negative,
        temperatures=(0.02, 0.05, 0.2),
    )
    without_negative = official_feature_gmd_teacher(
        source,
        positive,
        temperatures=(0.02, 0.05, 0.2),
    )

    # The zero weight is applied after row/column softmax in the release, so
    # these candidates still change the normalization and hence the force.
    assert not torch.allclose(
        with_negative.forces["tokens"],
        without_negative.forces["tokens"],
        rtol=1e-5,
        atol=1e-7,
    )
    assert all(not value.requires_grad for value in with_negative.forces.values())
    assert all(
        bool(torch.isfinite(value).all()) for value in with_negative.forces.values()
    )


@pytest.mark.parametrize("force_step", [0.0, -0.1, 1.01, float("nan"), True])
def test_invalid_force_steps_are_rejected(force_step) -> None:
    source, origin, local, positive, negative = _feature_batches()
    with pytest.raises((TypeError, ValueError), match="force_step"):
        two_view_local_feature_gmd_loss(
            source,
            origin,
            local,
            positive,
            fixed_negative_features=negative,
            force_step=force_step,
        )


def test_feature_dictionary_keys_and_geometry_are_validated() -> None:
    source, origin, local, positive, negative = _feature_batches()
    with pytest.raises(ValueError, match="ordered keys"):
        two_view_local_feature_gmd_loss(
            source,
            {"wrong": origin["tokens"]},
            local,
            positive,
            fixed_negative_features=negative,
            force_step=0.5,
        )
    with pytest.raises(ValueError, match="match selected source shape"):
        two_view_local_feature_gmd_loss(
            source,
            {"tokens": origin["tokens"][:-1]},
            local,
            positive,
            fixed_negative_features=negative,
            force_step=0.5,
        )
    with pytest.raises(ValueError, match="tokens and dimensions"):
        two_view_local_feature_gmd_loss(
            source,
            origin,
            local,
            {"tokens": positive["tokens"][:, :2]},
            fixed_negative_features=negative,
            force_step=0.5,
        )
