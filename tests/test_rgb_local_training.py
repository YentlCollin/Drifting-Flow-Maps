"""CPU contracts for the isolated calibrated local Flow-GMD trainer."""

from __future__ import annotations

import copy
import math
import os
import pickle
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flowmaps.features.distributed import global_sharded_mae_convnext_loss
from flowmaps.features.mae import canonical_json_sha256
from flowmaps.models.backbone import NegativeImageBank, PositiveImageBank
from flowmaps.objectives.drifting import token_features_to_groups
from flowmaps.objectives.local_drifting import (
    OfficialFeatureGMDTeacher,
    two_view_local_feature_gmd_loss,
)
from flowmaps.training.local_steps import (
    SUPPORTED_EVALUATION_NFES,
    LocalTwoViewTrainingWrapper,
    cartesian_time_pair_grid,
    combine_training_losses,
    explicit_time_pair_grid,
    factorized_time_pair_grid,
    global_sharded_boundary_and_local_two_view_loss,
    sample_supported_nfe,
    sample_time_pair,
)
from torch.nn.parallel import DistributedDataParallel


class _ToyFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The repository's historical state-dict hash expects tensors with at
        # least one dimension, as do all real generator parameters.
        self.scale = torch.nn.Parameter(torch.tensor([0.2]))
        self.bias = torch.nn.Parameter(torch.tensor([-0.03]))
        self.calls: list[tuple[float, float, int]] = []

    def forward(self, state, labels, start, end):
        start_value = float(torch.as_tensor(start))
        end_value = float(torch.as_tensor(end))
        self.calls.append((start_value, end_value, labels.data_ptr()))
        span = torch.as_tensor(
            end, device=state.device, dtype=state.dtype
        ) - torch.as_tensor(start, device=state.device, dtype=state.dtype)
        return state + span * (self.scale * state + self.bias)


def _features(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"toy": value.unsqueeze(1)}


def test_loss_algebra_is_exact_half_sum_and_force_is_detached() -> None:
    source = torch.tensor(
        [[0.2, -0.4], [0.7, 0.1]], dtype=torch.float32, requires_grad=True
    )
    origin = torch.tensor(
        [[0.5, -0.1], [0.9, 0.0]], dtype=torch.float32, requires_grad=True
    )
    local = torch.tensor(
        [[0.1, -0.2], [0.4, 0.6]], dtype=torch.float32, requires_grad=True
    )
    positive = torch.zeros(3, 2)
    source_groups = token_features_to_groups(_features(source)["toy"])
    force = torch.tensor([[[0.25, -0.5], [0.75, 0.125]]], dtype=torch.float32)
    teacher = OfficialFeatureGMDTeacher(
        forces={"toy": force},
        input_scales={"toy": torch.tensor(2.0)},
        group_losses={"toy": torch.tensor([3.0])},
        information={"toy": {}},
    )
    assert teacher.forces["toy"].shape == source_groups.shape

    loss, statistics = two_view_local_feature_gmd_loss(
        _features(source),
        _features(origin),
        _features(local),
        _features(positive),
        force_step=0.5,
        teacher=teacher,
        synchronize_distributed=False,
    )
    expected_origin = (
        (
            (
                token_features_to_groups(_features(origin)["toy"])
                - source_groups.detach()
            )
            / 2.0
            / 0.5
            - force
        )
        .square()
        .mean()
    )
    expected_local = (
        (
            (token_features_to_groups(_features(local)["toy"]) - source_groups.detach())
            / 2.0
            / 0.5
            - force
        )
        .square()
        .mean()
    )
    expected = 0.5 * (expected_origin + expected_local)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(statistics["loss/origin"], expected_origin.detach())
    torch.testing.assert_close(statistics["loss/local"], expected_local.detach())

    boundary = torch.tensor(2.0, requires_grad=True)
    e10 = torch.tensor(5.0, requires_grad=True)
    total = combine_training_losses(
        boundary,
        loss,
        e10,
        lambda_local=1.5,
        lambda_e10=0.0,
    )
    torch.testing.assert_close(total, boundary + 1.5 * expected, rtol=0, atol=0)
    total.backward()
    assert source.grad is None
    assert origin.grad is not None and bool(torch.isfinite(origin.grad).all())
    assert local.grad is not None and bool(torch.isfinite(local.grad).all())
    torch.testing.assert_close(boundary.grad, torch.tensor(1.0))
    torch.testing.assert_close(e10.grad, torch.tensor(0.0))


def test_wrapper_contains_all_calls_detaches_source_and_has_no_optimizer_step() -> None:
    flow = _ToyFlow()
    wrapper = LocalTwoViewTrainingWrapper(flow, enable_e10=False)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
    initial_parameters = [
        parameter.detach().clone() for parameter in wrapper.parameters()
    ]
    prior = torch.arange(12, dtype=torch.float32).reshape(4, 3).div(10)
    labels = torch.zeros(4, 2, dtype=torch.int64)
    outputs = wrapper(prior, labels, 0.5, 0.25)
    torch.testing.assert_close(outputs.end_time, torch.tensor(0.625))
    assert outputs.source.grad_fn is None and not outputs.source.requires_grad
    assert len(flow.calls) == 4
    assert [(start, end) for start, end, _ in flow.calls] == [
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, 0.625),
        (0.5, 0.625),
    ]
    assert len({label_pointer for _, _, label_pointer in flow.calls}) == 1
    loss = (
        outputs.terminal.square().mean()
        + 0.5
        * (
            (outputs.origin_student - outputs.source).square().mean()
            + (outputs.local_student - outputs.source).square().mean()
        )
        + outputs.e10
    )
    loss.backward()
    # Objective construction/backward has no hidden optimizer mutation.
    assert not optimizer.state
    for actual, expected in zip(wrapper.parameters(), initial_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    flow.calls.clear()
    outputs_rho_one = wrapper(prior, labels, 0.5, 1.0)
    torch.testing.assert_close(
        outputs_rho_one.origin_student,
        outputs_rho_one.terminal,
        rtol=0,
        atol=0,
    )
    assert outputs_rho_one.origin_student is not outputs_rho_one.terminal
    assert len(flow.calls) == 4


def test_e10_and_nfe_sampler_call_contracts() -> None:
    flow = _ToyFlow()
    wrapper = LocalTwoViewTrainingWrapper(flow, enable_e10=True, e10_delta=0.05)
    prior = torch.ones(2, 3)
    labels = torch.zeros(2, 1, dtype=torch.int64)
    outputs = wrapper(prior, labels, 0.4, 0.5)
    assert outputs.e10.ndim == 0 and outputs.e10.requires_grad
    assert len(flow.calls) == 7

    assert SUPPORTED_EVALUATION_NFES == (1, 2, 4, 8, 16)
    for nfe in SUPPORTED_EVALUATION_NFES:
        flow.calls.clear()
        sampled = sample_supported_nfe(flow, prior, labels, nfe=nfe, time_power=1.5)
        assert sampled.shape == prior.shape
        assert len(flow.calls) == nfe
        assert len({label_pointer for _, _, label_pointer in flow.calls}) == 1
    with pytest.raises(ValueError, match="supported"):
        sample_supported_nfe(flow, prior, labels, nfe=32)


def test_pair_grid_probabilities_and_rng_are_exactly_replayable() -> None:
    grid = cartesian_time_pair_grid((0.1, 0.5), (0.25, 0.5))
    assert grid.pairs == ((0.1, 0.25), (0.1, 0.5), (0.5, 0.25), (0.5, 0.5))
    assert math.fsum(grid.probabilities) == pytest.approx(1.0, abs=1e-15)
    weighted = explicit_time_pair_grid(((0.1, 0.25), (0.9, 0.5)), (0.0, 1.0))
    generator = torch.Generator().manual_seed(90210)
    before = generator.get_state().clone()
    assert sample_time_pair(weighted, generator) == (1, 0.9, 0.5)
    after = generator.get_state().clone()
    assert not torch.equal(before, after)
    generator.set_state(before)
    assert sample_time_pair(weighted, generator) == (1, 0.9, 0.5)
    torch.testing.assert_close(generator.get_state(), after, rtol=0, atol=0)
    with pytest.raises(ValueError, match="sum to one"):
        explicit_time_pair_grid(((0.1, 0.5), (0.5, 0.5)), (0.2, 0.2))


def test_combined_feature_adapter_encodes_fixed_supports_once(monkeypatch) -> None:
    import flowmaps.features.combined as composite_module
    import flowmaps.features.convnext as convnext_module
    import flowmaps.objectives.local_drifting as local_module

    calls = {"fixed": 0, "source_convnext": 0, "live_convnext": 0}

    def features(images):
        return {"family": images.mean(dim=(-1, -2)).unsqueeze(1)}

    def fake_fixed(mae, source, positive, negative):
        del mae
        calls["fixed"] += 1
        return features(source), features(positive), features(negative)

    def fake_convnext_supports(convnext, source, positive, negative, **kwargs):
        del convnext, kwargs
        calls["source_convnext"] += 1
        return features(source), features(positive), features(negative)

    def fake_live(convnext, images, **kwargs):
        del convnext, kwargs
        calls["live_convnext"] += 1
        return features(images)

    def fake_merge(mae, convnext):
        assert tuple(mae) == tuple(convnext) == ("family",)
        return dict(mae)

    force_teacher = object()

    def fake_teacher(source, positive, **kwargs):
        del source, positive, kwargs
        return force_teacher

    def fake_boundary(generated, positive, negative, **kwargs):
        del positive, negative, kwargs
        value = generated["family"].square().mean()
        return value, {"loss": value.detach()}

    def fake_local(source, origin, local, positive, **kwargs):
        del positive
        assert kwargs["teacher"] is force_teacher
        assert kwargs["force_step"] == 0.5
        origin_loss = (origin["family"] - source["family"]).square().mean()
        local_loss = (local["family"] - source["family"]).square().mean()
        value = 0.5 * (origin_loss + local_loss)
        return value, {"loss": value.detach()}

    monkeypatch.setattr(composite_module, "_extract_mae_supports", fake_fixed)
    monkeypatch.setattr(
        composite_module, "drifting_loss_from_complete_features", fake_boundary
    )
    monkeypatch.setattr(
        convnext_module, "extract_convnext_supports", fake_convnext_supports
    )
    monkeypatch.setattr(
        convnext_module, "extract_generated_convnext_features", fake_live
    )
    monkeypatch.setattr(convnext_module, "merge_feature_families", fake_merge)
    monkeypatch.setattr(local_module, "official_feature_gmd_teacher", fake_teacher)
    monkeypatch.setattr(local_module, "two_view_local_feature_gmd_loss", fake_local)

    class FakeMAE:
        def encode_features(self, images):
            return features(images)

    class FakeEncoder:
        def __init__(self):
            self.mae = FakeMAE()
            self.convnext = object()
            self.last_local_generated = None

        def assert_frozen_eval(self):
            return None

        def retain_generated_gradient(self, images):
            images.retain_grad()
            self.last_local_generated = images

    source = torch.zeros(2, 3, 64, 64)
    terminal = torch.ones_like(source, requires_grad=True)
    origin = torch.full_like(source, 0.5, requires_grad=True)
    local = torch.full_like(source, -0.25, requires_grad=True)
    positive = torch.ones(4, 3, 64, 64)
    negative = torch.zeros(2, 3, 64, 64)
    encoder = FakeEncoder()
    boundary, _, local_loss, _ = global_sharded_boundary_and_local_two_view_loss(
        encoder,
        terminal,
        source,
        origin,
        local,
        positive,
        negative,
        force_step=0.5,
        temperatures=(0.2, 0.05, 0.02),
    )
    (boundary + local_loss).backward()
    assert calls == {"fixed": 1, "source_convnext": 1, "live_convnext": 2}
    assert terminal.grad is not None
    assert origin.grad is not None
    assert local.grad is not None

    zero_encoder = FakeEncoder()
    zero_terminal = torch.ones_like(source, requires_grad=True)
    zero_origin = torch.full_like(source, 0.5, requires_grad=True)
    zero_local = torch.full_like(source, -0.25, requires_grad=True)
    zero_boundary, _, computed_local, _ = (
        global_sharded_boundary_and_local_two_view_loss(
            zero_encoder,
            zero_terminal,
            source,
            zero_origin,
            zero_local,
            positive,
            negative,
            force_step=0.5,
            temperatures=(0.2, 0.05, 0.02),
        )
    )
    graph_matched_control = combine_training_losses(
        zero_boundary,
        computed_local,
        zero_terminal.sum() * 0.0,
        lambda_local=0.0,
        lambda_e10=0.0,
    )
    graph_matched_control.backward()
    assert calls == {"fixed": 2, "source_convnext": 2, "live_convnext": 4}
    assert zero_terminal.grad is not None and zero_terminal.grad.abs().sum() > 0
    assert zero_origin.grad is not None and zero_origin.grad.count_nonzero() == 0
    assert zero_local.grad is not None and zero_local.grad.count_nonzero() == 0
    assert zero_encoder.last_local_generated.grad is not None
    assert zero_encoder.last_local_generated.grad.count_nonzero() == 0


def test_combined_adapter_zero_local_is_exact_terminal_loss_and_gradient(
    monkeypatch,
) -> None:
    """The graph-matched local control must preserve historical terminal GMD."""

    import flowmaps.features.combined as composite_module
    import flowmaps.features.convnext as convnext_module
    import flowmaps.objectives.local_drifting as local_module

    def mae_features(images):
        return {"mae": images.mean(dim=(-1, -2)).unsqueeze(1)}

    def convnext_features(images):
        return {"convnext": images.square().mean(dim=(-1, -2)).unsqueeze(1)}

    def fake_fixed(mae, generated, positive, negative):
        del mae
        return tuple(mae_features(images) for images in (generated, positive, negative))

    def fake_convnext_supports(convnext, generated, positive, negative, **kwargs):
        del convnext, kwargs
        return tuple(
            convnext_features(images) for images in (generated, positive, negative)
        )

    def fake_live(convnext, images, **kwargs):
        del convnext, kwargs
        return convnext_features(images)

    def fake_merge(mae, convnext):
        assert not set(mae).intersection(convnext)
        return {**mae, **convnext}

    def fake_boundary(generated, positive, negative, **kwargs):
        del positive, negative, kwargs
        loss = sum(value.square().mean() for value in generated.values())
        return loss, {"loss/boundary": loss.detach()}

    teacher = object()

    def fake_teacher(source, positive, **kwargs):
        del source, positive, kwargs
        return teacher

    def fake_local(source, origin, local, positive, **kwargs):
        del positive
        assert kwargs["teacher"] is teacher
        origin_loss = sum((origin[key] - source[key]).square().mean() for key in source)
        transition_loss = sum(
            (local[key] - source[key]).square().mean() for key in source
        )
        loss = (
            kwargs["origin_weight"] * origin_loss
            + kwargs["local_weight"] * transition_loss
        )
        return loss, {"loss/local": loss.detach()}

    monkeypatch.setattr(composite_module, "_extract_mae_supports", fake_fixed)
    monkeypatch.setattr(
        composite_module, "drifting_loss_from_complete_features", fake_boundary
    )
    monkeypatch.setattr(
        convnext_module, "extract_convnext_supports", fake_convnext_supports
    )
    monkeypatch.setattr(
        convnext_module, "extract_generated_convnext_features", fake_live
    )
    monkeypatch.setattr(convnext_module, "merge_feature_families", fake_merge)
    monkeypatch.setattr(local_module, "official_feature_gmd_teacher", fake_teacher)
    monkeypatch.setattr(local_module, "two_view_local_feature_gmd_loss", fake_local)

    class FakeMAE:
        def encode_features(self, images):
            return mae_features(images)

    class FakeEncoder:
        def __init__(self):
            self.mae = FakeMAE()
            self.convnext = object()

        def assert_frozen_eval(self):
            return None

        def retain_generated_gradient(self, images):
            images.retain_grad()

    torch.manual_seed(20260915)
    terminal_value = torch.randn(2, 3, 64, 64)
    historical_terminal = terminal_value.clone().requires_grad_()
    combined_terminal = terminal_value.clone().requires_grad_()
    source = torch.randn_like(terminal_value)
    origin = torch.randn_like(terminal_value, requires_grad=True)
    transition = torch.randn_like(terminal_value, requires_grad=True)
    positive = torch.randn(4, 3, 64, 64)
    negative = torch.randn(2, 3, 64, 64)
    temperatures = (0.2, 0.05, 0.02)

    historical_loss, historical_statistics = global_sharded_mae_convnext_loss(
        FakeEncoder(),
        historical_terminal,
        positive,
        negative,
        temperatures=temperatures,
    )
    combined_loss, combined_statistics, local_loss, _ = (
        global_sharded_boundary_and_local_two_view_loss(
            FakeEncoder(),
            combined_terminal,
            source,
            origin,
            transition,
            positive,
            negative,
            force_step=0.5,
            temperatures=temperatures,
        )
    )
    graph_matched_total = combined_loss + 0.0 * local_loss

    torch.testing.assert_close(combined_loss, historical_loss, rtol=0, atol=0)
    assert combined_statistics.keys() == historical_statistics.keys()
    for key in historical_statistics:
        torch.testing.assert_close(
            combined_statistics[key], historical_statistics[key], rtol=0, atol=0
        )

    (historical_gradient,) = torch.autograd.grad(historical_loss, historical_terminal)
    combined_gradient, origin_gradient, transition_gradient = torch.autograd.grad(
        graph_matched_total, (combined_terminal, origin, transition)
    )
    torch.testing.assert_close(combined_gradient, historical_gradient, rtol=0, atol=0)
    assert historical_gradient.abs().sum() > 0
    torch.testing.assert_close(
        origin_gradient, torch.zeros_like(origin_gradient), rtol=0, atol=0
    )
    torch.testing.assert_close(
        transition_gradient, torch.zeros_like(transition_gradient), rtol=0, atol=0
    )


class _TinyStream:
    def __init__(self) -> None:
        self.epoch = 3
        self.offset = 9

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state):
        if set(state) != {"epoch", "offset"}:
            raise ValueError("bad stream state")
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])


def _cpu_args() -> SimpleNamespace:
    return SimpleNamespace(
        distributed=False,
        world_size=1,
        rank=0,
        local_rank=0,
        device=torch.device("cpu"),
        time_grid=explicit_time_pair_grid(((0.1, 0.25), (0.5, 0.5))),
    )
