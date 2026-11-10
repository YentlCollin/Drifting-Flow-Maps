from __future__ import annotations

import copy

import pytest
import torch
from flowmaps.features.mae import canonical_json_sha256, state_dict_sha256
from flowmaps.models.backbone import CelebA64DirectPixelGMDDiT
from flowmaps.models.rgb import CelebA64TwoTimeGMDDiT
from flowmaps.models.rollout import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    ComposedRollout,
    composed_rollout_artifact_dict,
    flow_from_pinned_parent_payload,
    load_composed_rollout_artifact,
    unwrap_composed_rollout,
)


def _tiny_flow() -> CelebA64TwoTimeGMDDiT:
    torch.manual_seed(101)
    flow = CelebA64TwoTimeGMDDiT(
        patch_size=64,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        noise_classes=4,
        noise_coords=2,
        condition_tokens=0,
        gradient_checkpointing=False,
        intended_inference_precision="float32",
        flow_time_frequency_embedding_size=4,
    )
    # Avoid the production zero head masking state/parameter gradients in this
    # small graph-connectivity test.
    with torch.no_grad():
        flow.final.output.weight.normal_(std=0.02)
        flow.final.output.bias.normal_(std=0.02)
    return flow


def _tiny_parent() -> CelebA64DirectPixelGMDDiT:
    torch.manual_seed(202)
    model = CelebA64DirectPixelGMDDiT(
        patch_size=64,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        noise_classes=4,
        noise_coords=2,
        condition_tokens=0,
        gradient_checkpointing=False,
        intended_inference_precision="float32",
    )
    with torch.no_grad():
        model.final.output.weight.normal_(std=0.02)
        model.final.output.bias.normal_(std=0.02)
    return model


def _inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    prior = torch.randn(1, 3, 64, 64, requires_grad=requires_grad)
    labels = torch.tensor([[1, 3]], dtype=torch.int64)
    return prior, labels


def test_outer_forward_uses_exactly_k_calls_fixed_labels_and_one_live_graph() -> None:
    flow = _tiny_flow().train()
    model = ComposedRollout(flow)
    prior, labels = _inputs(requires_grad=True)
    seen_labels: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    def record_call(module, arguments, output):
        del module
        seen_labels.append(arguments[1])
        output.retain_grad()
        outputs.append(output)

    handle = flow.register_forward_hook(record_call)
    try:
        terminal = model(prior, labels, 3)
        terminal.square().mean().backward()
    finally:
        handle.remove()

    assert model.flow is flow
    assert terminal.shape == prior.shape
    assert len(outputs) == 3
    assert all(seen is labels for seen in seen_labels)
    assert all(output.grad is not None for output in outputs)
    assert all(float(output.grad.abs().sum()) > 0.0 for output in outputs)
    assert prior.grad is not None and float(prior.grad.abs().sum()) > 0.0
    gradient = flow.final.output.weight.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0
    assert unwrap_composed_rollout(model) is model


def test_outer_forward_rejects_non_homogeneous_python_integer_k() -> None:
    model = ComposedRollout(_tiny_flow())
    prior, labels = _inputs()
    for invalid in (0, -1, True, 1.5, torch.tensor(2)):
        with pytest.raises(ValueError, match="positive Python integer"):
            model(prior, labels, invalid)  # type: ignore[arg-type]


def test_strict_artifact_roundtrip_is_exact_and_runtime_k_is_not_serialized(
    tmp_path,
) -> None:
    model = ComposedRollout(_tiny_flow()).eval()
    metadata = {
        "completed_step": 7,
        "objective": "synthetic_composed_GMD",
        "weight_kind": "ema",
    }
    payload = composed_rollout_artifact_dict(model, metadata=metadata)

    assert payload["format"] == ARTIFACT_FORMAT
    assert payload["version"] == ARTIFACT_VERSION
    assert "k" not in payload and "schedule" not in payload
    assert all(not name.startswith("flow.") for name in payload["state_dict"])
    assert payload["integrity"]["state_dict_sha256"] == state_dict_sha256(
        payload["state_dict"]
    )

    path = tmp_path / "composed_rollout.pt"
    torch.save(payload, path)
    restored, restored_metadata = load_composed_rollout_artifact(path, "cpu")
    assert restored_metadata == metadata
    assert restored.flow.flow_architecture_config() == (
        model.flow.flow_architecture_config()
    )
    assert state_dict_sha256(restored.flow.state_dict()) == state_dict_sha256(
        model.flow.state_dict()
    )
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    prior, labels = _inputs()
    with torch.no_grad():
        for k in (1, 3):
            torch.testing.assert_close(
                restored(prior, labels, k),
                model(prior, labels, k),
                rtol=0,
                atol=0,
            )

    differentiable_prior = prior.clone().requires_grad_(True)
    restored(differentiable_prior, labels, 2).sum().backward()
    assert differentiable_prior.grad is not None
    assert float(differentiable_prior.grad.abs().sum()) > 0.0


def test_artifact_loader_rejects_invalid_schema_hashes_and_semantics() -> None:
    payload = composed_rollout_artifact_dict(
        ComposedRollout(_tiny_flow()),
        metadata={"objective": "test"},
    )

    extra = copy.deepcopy(payload)
    extra["k"] = 2
    with pytest.raises(ValueError, match="fields are not exact"):
        load_composed_rollout_artifact(extra, "cpu")

    bad_state = copy.deepcopy(payload)
    first = next(iter(bad_state["state_dict"].values()))
    first.view(-1)[0].add_(1)
    with pytest.raises(ValueError, match="SHA-256"):
        load_composed_rollout_artifact(bad_state, "cpu")

    bad_architecture = copy.deepcopy(payload)
    bad_architecture["architecture"]["parameterization"] = "different"
    bad_architecture["integrity"]["architecture_sha256"] = canonical_json_sha256(
        bad_architecture["architecture"]
    )
    with pytest.raises(ValueError, match="semantics differ"):
        load_composed_rollout_artifact(bad_architecture, "cpu")


def test_parent_adapter_selects_ema_and_checks_step_objective_and_hashes() -> None:
    parent = _tiny_parent().eval()
    signature = {
        "objective": "pinned_test_objective",
        "generator": parent.architecture_config(),
    }
    ema = {
        name: value.detach().cpu().clone()
        for name, value in parent.state_dict().items()
    }
    ema_sha = state_dict_sha256(ema)
    signature_sha = canonical_json_sha256(signature)
    payload = {
        "completed_step": 60_000,
        "training_signature": signature,
        "ema": ema,
        "integrity": {
            "algorithm": "sha256",
            "ema_state_sha256": ema_sha,
            "training_signature_sha256": signature_sha,
        },
    }

    flow = flow_from_pinned_parent_payload(
        payload,
        expected_step=60_000,
        expected_objective="pinned_test_objective",
        expected_ema_state_sha256=ema_sha,
        expected_training_signature_sha256=signature_sha,
        flow_time_frequency_embedding_size=4,
    )
    assert flow.training
    assert all(parameter.requires_grad for parameter in flow.parameters())
    prior, labels = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            flow(prior, labels, 0.0, 1.0),
            parent(prior, labels),
            rtol=0,
            atol=0,
        )

    with pytest.raises(ValueError, match="completed_step differs"):
        flow_from_pinned_parent_payload(
            payload,
            expected_step=59_999,
            expected_objective="pinned_test_objective",
        )
    with pytest.raises(ValueError, match="objective differs"):
        flow_from_pinned_parent_payload(
            payload,
            expected_step=60_000,
            expected_objective="wrong",
        )
    with pytest.raises(ValueError, match="EMA state SHA-256 differs"):
        flow_from_pinned_parent_payload(
            payload,
            expected_step=60_000,
            expected_objective="pinned_test_objective",
            expected_ema_state_sha256="0" * 64,
        )
