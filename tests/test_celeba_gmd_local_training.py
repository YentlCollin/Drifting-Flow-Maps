"""CPU contracts for the isolated calibrated local Flow-GMD trainer."""

from __future__ import annotations

import copy
from datetime import timedelta
import math
import os
import pickle
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from experiments.celeba_flow_gmd_20260914 import train_local_two_view_gmd as trainer
from src.celeba_gmd_local_training import (
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
from src.celeba_gmd_local_two_view import (
    OfficialFeatureGMDTeacher,
    two_view_local_feature_gmd_loss,
)
from src.celeba_pixel_gmd_features import canonical_json_sha256
from src.celeba_pixel_gmd_generator import NegativeImageBank, PositiveImageBank
from src.celeba_gmd_sharded_features import global_sharded_mae_convnext_loss
from src.official_drifting import token_features_to_groups


def test_runtime_rng_state_is_object_collective_safe_and_exact() -> None:
    torch.manual_seed(901)
    random.seed(902)
    np.random.seed(903)
    state = trainer.capture_runtime_rng(torch.device("cpu"))
    assert isinstance(state["numpy"]["keys"], list)
    assert len(state["numpy"]["keys"]) == 624
    round_tripped = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    expected_torch = torch.rand(8)
    expected_python = [random.random() for _ in range(8)]
    expected_numpy = np.random.random(8)
    trainer.restore_runtime_rng(round_tripped, torch.device("cpu"))
    torch.testing.assert_close(torch.rand(8), expected_torch, rtol=0, atol=0)
    assert [random.random() for _ in range(8)] == expected_python
    np.testing.assert_array_equal(np.random.random(8), expected_numpy)


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
    force = torch.tensor(
        [[[0.25, -0.5], [0.75, 0.125]]], dtype=torch.float32
    )
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
        (token_features_to_groups(_features(origin)["toy"]) - source_groups.detach())
        / 2.0
        / 0.5
        - force
    ).square().mean()
    expected_local = (
        (token_features_to_groups(_features(local)["toy"]) - source_groups.detach())
        / 2.0
        / 0.5
        - force
    ).square().mean()
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
    initial_parameters = [parameter.detach().clone() for parameter in wrapper.parameters()]
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
        sampled = sample_supported_nfe(
            flow, prior, labels, nfe=nfe, time_power=1.5
        )
        assert sampled.shape == prior.shape
        assert len(flow.calls) == nfe
        assert len({label_pointer for _, _, label_pointer in flow.calls}) == 1
    with pytest.raises(ValueError, match="supported"):
        sample_supported_nfe(flow, prior, labels, nfe=32)


def test_zero_local_weight_is_a_valid_graph_matched_control() -> None:
    trainer.validate_objective_weights(0.0, 0.0)
    trainer.validate_objective_weights(0.05, 10.0)
    for invalid in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="non-negative"):
            trainer.validate_objective_weights(invalid, 0.0)

    boundary = torch.tensor(2.0, requires_grad=True)
    local = torch.tensor(7.0, requires_grad=True)
    e10 = torch.tensor(11.0, requires_grad=True)
    total = combine_training_losses(
        boundary,
        local,
        e10,
        lambda_local=0.0,
        lambda_e10=0.0,
    )
    torch.testing.assert_close(total, boundary, rtol=0, atol=0)
    total.backward()
    torch.testing.assert_close(boundary.grad, torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(local.grad, torch.tensor(0.0), rtol=0, atol=0)
    torch.testing.assert_close(e10.grad, torch.tensor(0.0), rtol=0, atol=0)


def test_pair_grid_probabilities_and_rng_are_exactly_replayable() -> None:
    grid = cartesian_time_pair_grid((0.1, 0.5), (0.25, 0.5))
    assert grid.pairs == ((0.1, 0.25), (0.1, 0.5), (0.5, 0.25), (0.5, 0.5))
    assert math.fsum(grid.probabilities) == pytest.approx(1.0, abs=1e-15)
    weighted = explicit_time_pair_grid(
        ((0.1, 0.25), (0.9, 0.5)), (0.0, 1.0)
    )
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


def test_factorized_and_default_main_time_distributions_are_exact() -> None:
    factorized = factorized_time_pair_grid(
        (0.1, 0.8), (1.0, 0.25), (0.25, 0.75), (0.2, 0.8)
    )
    assert factorized.pairs == (
        (0.1, 1.0),
        (0.1, 0.25),
        (0.8, 1.0),
        (0.8, 0.25),
    )
    assert factorized.probabilities == pytest.approx((0.05, 0.2, 0.15, 0.6))

    args = trainer.parse_args([])
    main = trainer.resolve_time_grid(args)
    assert len(main.pairs) == 12
    assert main.pairs[:3] == ((0.1, 1.0), (0.1, 0.5), (0.1, 0.25))
    assert main.probabilities == pytest.approx((0.05, 0.1, 0.1) * 4)
    for rho, expected in ((1.0, 0.2), (0.5, 0.4), (0.25, 0.4)):
        assert math.fsum(
            probability
            for (_, observed_rho), probability in zip(
                main.pairs, main.probabilities, strict=True
            )
            if observed_rho == rho
        ) == pytest.approx(expected)
    assert args.lambda_local == 0.05
    assert tuple(args.sampler_nfes) == (1, 2, 4, 8, 16)

    # The documented short-only ablation remains a one-flag uniform preset.
    args.time_rhos = (0.5, 0.25)
    short_only = trainer.resolve_time_grid(args)
    assert len(short_only.pairs) == 8
    assert short_only.probabilities == pytest.approx((0.125,) * 8)


def test_combined_feature_adapter_encodes_fixed_supports_once(monkeypatch) -> None:
    import src.celeba_gmd_local_two_view as local_module
    import src.celeba_pixel_gmd_convnext as convnext_module
    import src.celeba_pixel_gmd_mae_convnext as composite_module

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

    import src.celeba_gmd_local_two_view as local_module
    import src.celeba_pixel_gmd_convnext as convnext_module
    import src.celeba_pixel_gmd_mae_convnext as composite_module

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
        origin_loss = sum(
            (origin[key] - source[key]).square().mean() for key in source
        )
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

    (historical_gradient,) = torch.autograd.grad(
        historical_loss, historical_terminal
    )
    combined_gradient, origin_gradient, transition_gradient = torch.autograd.grad(
        graph_matched_total, (combined_terminal, origin, transition)
    )
    torch.testing.assert_close(
        combined_gradient, historical_gradient, rtol=0, atol=0
    )
    assert historical_gradient.abs().sum() > 0
    torch.testing.assert_close(
        origin_gradient, torch.zeros_like(origin_gradient), rtol=0, atol=0
    )
    torch.testing.assert_close(
        transition_gradient, torch.zeros_like(transition_gradient), rtol=0, atol=0
    )


def _ddp_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(71)
        distributed_flow = _ToyFlow()
        reference_flow = copy.deepcopy(distributed_flow)
        distributed = DistributedDataParallel(
            LocalTwoViewTrainingWrapper(distributed_flow, enable_e10=False),
            static_graph=True,
        )
        reference = LocalTwoViewTrainingWrapper(reference_flow, enable_e10=False)
        complete_prior = torch.arange(24, dtype=torch.float32).reshape(8, 3).div(17)
        complete_labels = torch.zeros(8, 2, dtype=torch.int64)
        local_prior = complete_prior[rank * 4 : (rank + 1) * 4]
        local_labels = complete_labels[rank * 4 : (rank + 1) * 4]

        expected = reference(complete_prior, complete_labels, 0.5, 0.25)
        expected_boundary = expected.terminal.square().mean()
        expected_origin = (expected.origin_student - expected.source).square().mean()
        expected_local = (expected.local_student - expected.source).square().mean()
        expected_loss = expected_boundary + 0.7 * 0.5 * (
            expected_origin + expected_local
        )
        expected_loss.backward()

        actual = distributed(local_prior, local_labels, 0.5, 0.25)
        assert actual.source.grad_fn is None and not actual.source.requires_grad
        actual_boundary = actual.terminal.square().mean()
        actual_origin = (actual.origin_student - actual.source).square().mean()
        actual_local = (actual.local_student - actual.source).square().mean()
        actual_loss = actual_boundary + 0.7 * 0.5 * (actual_origin + actual_local)
        actual_loss.backward()

        for actual_parameter, expected_parameter in zip(
            distributed_flow.parameters(), reference_flow.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=2e-6,
                atol=2e-7,
            )
        actual_optimizer = torch.optim.AdamW(distributed.parameters(), lr=1e-3)
        expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual_parameter, expected_parameter in zip(
            distributed_flow.parameters(), reference_flow.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_parameter, expected_parameter, rtol=2e-6, atol=2e-7
            )

        # Compare the second update of a continuing static DDP reducer with
        # the first update after reconstructing DDP from the exact model and
        # optimizer state.  This catches restart-time bucket-layout drift.
        restart_model_state = copy.deepcopy(distributed_flow.state_dict())
        restart_optimizer_state = copy.deepcopy(actual_optimizer.state_dict())
        next_complete_prior = complete_prior.flip(0).add(0.125)
        next_local_prior = next_complete_prior[rank * 4 : (rank + 1) * 4]

        actual_optimizer.zero_grad(set_to_none=True)
        continued_output = distributed(next_local_prior, local_labels, 0.8, 0.5)
        continued_loss = (
            continued_output.terminal.square().mean()
            + 0.7
            * 0.5
            * (
                (continued_output.origin_student - continued_output.source)
                .square()
                .mean()
                + (continued_output.local_student - continued_output.source)
                .square()
                .mean()
            )
        )
        continued_loss.backward()
        actual_optimizer.step()

        restarted_flow = _ToyFlow()
        restarted_flow.load_state_dict(restart_model_state, strict=True)
        restarted = DistributedDataParallel(
            LocalTwoViewTrainingWrapper(restarted_flow, enable_e10=False),
            static_graph=True,
        )
        restarted_optimizer = torch.optim.AdamW(restarted.parameters(), lr=1e-3)
        restarted_optimizer.load_state_dict(restart_optimizer_state)
        restarted_optimizer.zero_grad(set_to_none=True)
        restarted_output = restarted(next_local_prior, local_labels, 0.8, 0.5)
        restarted_loss = (
            restarted_output.terminal.square().mean()
            + 0.7
            * 0.5
            * (
                (restarted_output.origin_student - restarted_output.source)
                .square()
                .mean()
                + (restarted_output.local_student - restarted_output.source)
                .square()
                .mean()
            )
        )
        restarted_loss.backward()
        restarted_optimizer.step()
        for continued_parameter, restarted_parameter in zip(
            distributed_flow.parameters(), restarted_flow.parameters(), strict=True
        ):
            assert torch.equal(continued_parameter, restarted_parameter)
        assert trainer.torch_tree_sha256(actual_optimizer.state_dict()) == (
            trainer.torch_tree_sha256(restarted_optimizer.state_dict())
        )

        # Parent stream import must yield one identical signature fragment on
        # every rank even though each rank intentionally receives a different
        # prior RNG state.
        source_stream = _TinyStream()
        source_stream.epoch, source_stream.offset = 12, 5
        source_positive = PositiveImageBank(2)
        source_negative = NegativeImageBank(3)
        bank_images = torch.linspace(-1, 1, 2 * 3 * 64 * 64).reshape(
            2, 3, 64, 64
        )
        source_positive.add(bank_images)
        source_negative.add(bank_images)
        source_bank_rng = torch.Generator().manual_seed(801)
        torch.rand(3, generator=source_bank_rng)
        parent_prior_states = []
        for prior_rank in range(2):
            parent_prior_rng = torch.Generator().manual_seed(900 + prior_rank)
            torch.rand(prior_rank + 1, generator=parent_prior_rng)
            parent_prior_states.append(parent_prior_rng.get_state())
        payload = {
            "data_stream": source_stream.state_dict(),
            "positive_bank": source_positive.state_dict(),
            "negative_bank": source_negative.state_dict(),
            "bank_rng_state": source_bank_rng.get_state(),
            "prior_rng_by_rank": parent_prior_states,
        }
        target_stream = _TinyStream()
        target_positive = PositiveImageBank(2)
        target_negative = NegativeImageBank(3)
        target_bank_rng = torch.Generator().manual_seed(1)
        target_prior_rng = torch.Generator().manual_seed(2)
        distributed_args = SimpleNamespace(
            distributed=True, world_size=2, rank=rank
        )
        identity = trainer.import_parent_streams(
            payload,
            stream=target_stream,
            positive_bank=target_positive,
            negative_bank=target_negative,
            bank_generator=target_bank_rng,
            prior_generator=target_prior_rng,
            args=distributed_args,
        )
        identities = [None, None]
        dist.all_gather_object(identities, identity)
        assert identities[0] == identities[1]
        assert identity["parent_training_seed"] == 42
        assert identity["data_stream_seed"] == 143
        assert len(set(identity["prior_rng_sha256_by_rank"])) == 2
    finally:
        dist.destroy_process_group()


def test_two_rank_wrapper_update_matches_monolithic_update(tmp_path: Path) -> None:
    context = mp.spawn(
        _ddp_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 60
        while not context.join(timeout=2):
            if time.monotonic() > deadline:
                raise TimeoutError("DDP parity workers did not finish")
        assert all(process.exitcode == 0 for process in context.processes)
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


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


def test_attempt_tokens_and_kernel_lock_recover_after_process_exit(
    tmp_path: Path,
) -> None:
    args = _cpu_args()
    args.attempt_token = None
    automatic = trainer.synchronize_attempt_token(args)
    assert trainer.ATTEMPT_TOKEN_PATTERN.fullmatch(automatic)

    invalid = _cpu_args()
    invalid.attempt_token = "../../unsafe"
    with pytest.raises(ValueError, match="attempt token"):
        trainer.synchronize_attempt_token(invalid)

    run_dir = tmp_path / "run-lock"
    run_dir.mkdir()
    first = trainer.reserve_run(run_dir, args)
    contender = _cpu_args()
    contender.attempt_token = "second_attempt"
    with pytest.raises(RuntimeError, match="active invocation"):
        trainer.reserve_run(run_dir, contender)

    # Closing the owning process' descriptor models the kernel cleanup that
    # also happens on SIGKILL/node-process loss: no stale directory remains.
    os.close(first["descriptor"])
    recovered = trainer.reserve_run(run_dir, contender)
    trainer.release_run(recovered, contender)
    assert (run_dir / ".local_two_view_training.lock").is_file()
    assert (run_dir / "lock_owners" / f"attempt_{automatic}.json").is_file()
    assert (run_dir / "lock_owners" / "attempt_second_attempt.json").is_file()


@pytest.mark.parametrize(
    ("counts", "completed_step"),
    [
        ({"0": 1, "1": 1}, 3),
        ({"0": -1, "1": 2}, 1),
        ({"0": True, "1": 0}, 1),
        ({"0": 1.0, "1": 0}, 1),
    ],
)
def test_pair_count_accounting_rejects_non_exact_resume_state(
    counts, completed_step
) -> None:
    with pytest.raises(ValueError, match="pair-count|pair counts"):
        trainer.validated_pair_counts(
            counts,
            expected_keys={"0", "1"},
            completed_step=completed_step,
        )


def test_json_export_recovery_is_idempotent_and_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    payload = {"format": "test", "version": 1, "values": [1, 2, 3]}
    first = trainer.publish_or_verify_json(path, payload)
    assert len(first) == 64
    assert trainer.publish_or_verify_json(path, payload) == first

    sidecar = Path(f"{path}.sha256")
    sidecar.unlink()
    assert trainer.publish_or_verify_json(path, payload) == first
    assert sidecar.read_text(encoding="utf-8").strip() == f"{first}  {path.name}"
    with pytest.raises(ValueError, match="differs"):
        trainer.publish_or_verify_json(path, {**payload, "version": 2})


def test_checkpoint_resume_restores_exact_state_rng_and_signature(tmp_path: Path) -> None:
    torch.manual_seed(123)
    random.seed(124)
    np.random.seed(125)
    flow = _ToyFlow()
    wrapper = LocalTwoViewTrainingWrapper(flow, enable_e10=False)
    ema = copy.deepcopy(flow).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=2e-3)
    prior = torch.arange(12, dtype=torch.float32).reshape(4, 3).div(11)
    labels = torch.zeros(4, 1, dtype=torch.int64)
    outputs = wrapper(prior, labels, 0.5, 0.25)
    outputs.terminal.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    stream = _TinyStream()
    positive = PositiveImageBank(2)
    negative = NegativeImageBank(3)
    images = torch.linspace(-1, 1, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64)
    positive.add(images)
    negative.add(images)
    bank_rng = torch.Generator().manual_seed(201)
    prior_rng = torch.Generator().manual_seed(202)
    time_rng = torch.Generator().manual_seed(203)
    counts = {"0": 4, "1": 7}
    signature = {
        "version": 1,
        "lambda_local": 1.0,
        "pairs": [[0.1, 0.25], [0.5, 0.5]],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoints" / "checkpoint_step_00000011.pt"
    args = _cpu_args()
    digest = trainer.save_checkpoint(
        checkpoint,
        completed_step=11,
        model=wrapper,
        ema=ema,
        optimizer=optimizer,
        stream=stream,
        positive_bank=positive,
        negative_bank=negative,
        bank_generator=bank_rng,
        prior_generator=prior_rng,
        time_generator=time_rng,
        pair_counts=counts,
        signature=signature,
        run_dir=run_dir,
        args=args,
    )
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer.exclusive_torch_save(checkpoint, {"different": True})

    expected = {
        "torch": torch.rand(4),
        "python": random.random(),
        "numpy": np.random.random(4),
        "bank": torch.rand(4, generator=bank_rng),
        "prior": torch.rand(4, generator=prior_rng),
        "time": torch.rand(4, generator=time_rng),
        "stream": copy.deepcopy(stream.state_dict()),
        "positive": copy.deepcopy(positive.state_dict()),
        "negative": copy.deepcopy(negative.state_dict()),
        "model": copy.deepcopy(flow.state_dict()),
        "ema": copy.deepcopy(ema.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }

    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(10)
        for parameter in ema.parameters():
            parameter.sub_(10)
    stream.epoch = 99
    stream.offset = 98
    positive.add(-images)
    negative.add(-images)
    counts["0"] = 999
    torch.rand(20)
    random.random()
    np.random.random(20)
    torch.rand(20, generator=bank_rng)
    torch.rand(20, generator=prior_rng)
    torch.rand(20, generator=time_rng)

    load_kwargs = dict(
        path=checkpoint,
        expected_sha256=digest,
        model=wrapper,
        ema=ema,
        optimizer=optimizer,
        stream=stream,
        positive_bank=positive,
        negative_bank=negative,
        bank_generator=bank_rng,
        prior_generator=prior_rng,
        time_generator=time_rng,
        pair_counts=counts,
        run_dir=run_dir,
        args=args,
    )
    with pytest.raises(ValueError, match="signature"):
        trainer.load_checkpoint(
            **load_kwargs, signature={**signature, "lambda_local": 2.0}
        )

    completed = trainer.load_checkpoint(**load_kwargs, signature=signature)
    assert completed == 11
    assert canonical_json_sha256(signature) == canonical_json_sha256(
        torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "training_signature"
        ]
    )
    torch.testing.assert_close(torch.rand(4), expected["torch"], rtol=0, atol=0)
    assert random.random() == expected["python"]
    np.testing.assert_array_equal(np.random.random(4), expected["numpy"])
    torch.testing.assert_close(
        torch.rand(4, generator=bank_rng), expected["bank"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.rand(4, generator=prior_rng), expected["prior"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.rand(4, generator=time_rng), expected["time"], rtol=0, atol=0
    )
    assert stream.state_dict() == expected["stream"]
    assert counts == {"0": 4, "1": 7}
    for observed_state, expected_state in (
        (positive.state_dict(), expected["positive"]),
        (negative.state_dict(), expected["negative"]),
    ):
        for key, actual in observed_state.items():
            wanted = expected_state[key]
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            else:
                assert actual == wanted
    for actual, wanted in zip(
        flow.state_dict().values(), expected["model"].values(), strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for actual, wanted in zip(
        ema.state_dict().values(), expected["ema"].values(), strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert trainer.torch_tree_sha256(optimizer.state_dict()) == trainer.torch_tree_sha256(
        expected["optimizer"]
    )
