from __future__ import annotations

import pytest
import torch
from flowmaps.models.backbone import CelebA64DirectPixelGMDDiT
from flowmaps.models.rgb import (
    CelebA64TwoTimeGMDDiT,
    load_one_step_state,
    sample_two_time_flow,
)


def tiny_one_step() -> CelebA64DirectPixelGMDDiT:
    return CelebA64DirectPixelGMDDiT(
        patch_size=16,
        hidden_size=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        noise_classes=4,
        noise_coords=2,
        condition_tokens=2,
        gradient_checkpointing=False,
        intended_inference_precision="float32",
    )


def batch(model: CelebA64DirectPixelGMDDiT, size: int = 2):
    generator = torch.Generator().manual_seed(7)
    return model.sample_prior(size, device="cpu", generator=generator)


def test_warm_start_preserves_the_one_step_generator_exactly() -> None:
    one_step = tiny_one_step().eval()
    prior, labels = batch(one_step)
    with torch.inference_mode():
        expected = one_step(prior, labels)
    flow = CelebA64TwoTimeGMDDiT.from_one_step(one_step).eval()
    with torch.inference_mode():
        actual = flow(prior, labels, 0.0, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert flow.flow_time_projection.weight.count_nonzero() == 0
    assert flow.flow_time_projection.bias.count_nonzero() == 0


def test_diagonal_is_an_exact_identity() -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    with torch.inference_mode():
        result = flow(prior, labels, torch.tensor([0.2, 0.8]), torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(result, prior, rtol=0, atol=0)


def test_transition_to_one_is_exactly_the_endpoint_candidate() -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    start = torch.tensor([0.2, 0.8])
    with torch.inference_mode():
        endpoint = flow(prior, labels, start, 1.0, output_kind="endpoint")
        mapped = flow(prior, labels, start, 1.0)
    torch.testing.assert_close(mapped, endpoint, rtol=0, atol=0)


def test_endpoint_selection_avoids_residual_roundoff(monkeypatch) -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    candidate = torch.full_like(prior, 0.333251953125, dtype=torch.bfloat16)

    def fixed_endpoint(*_args, **_kwargs):
        return candidate

    monkeypatch.setattr(flow, "endpoint_candidate", fixed_endpoint)
    with torch.inference_mode():
        mapped = flow(prior, labels, 0.0, 1.0)
    torch.testing.assert_close(mapped, candidate.float(), rtol=0, atol=0)


def test_composition_uses_the_requested_number_of_evaluations(monkeypatch) -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    calls = 0
    original = flow.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(flow, "forward", counted)
    with torch.inference_mode():
        result = sample_two_time_flow(flow, prior, labels, nfe=4)
    assert result.shape == prior.shape
    assert calls == 4


def test_checkpoint_adapter_loads_raw_or_ema_state() -> None:
    one_step = tiny_one_step()
    payload = {
        "completed_step": 12,
        "training_signature": {"generator": one_step.architecture_config()},
        "model": one_step.state_dict(),
        "ema": one_step.state_dict(),
    }
    flow = load_one_step_state(payload, weights="ema")
    assert (
        flow.flow_architecture_config()["one_step_backbone"]
        == one_step.architecture_config()
    )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-0.1, 0.5, "lie in"), (0.2, 1.1, "lie in"), (0.8, 0.2, "must not exceed")],
)
def test_invalid_times_are_rejected(start: float, end: float, message: str) -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    with pytest.raises(ValueError, match=message):
        flow(prior, labels, start, end)


@pytest.mark.parametrize(("nfe", "time_power"), [(0, 1.0), (1, 0.0)])
def test_invalid_partition_is_rejected(nfe: int, time_power: float) -> None:
    flow = CelebA64TwoTimeGMDDiT.from_one_step(tiny_one_step()).eval()
    prior, labels = batch(flow)
    with pytest.raises(ValueError):
        sample_two_time_flow(flow, prior, labels, nfe=nfe, time_power=time_power)
