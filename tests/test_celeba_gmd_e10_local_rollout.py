"""Contracts for the isolated structural-E10 plus local-view rollout."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.celeba_gmd_e10_local_rollout import (
    E10LocalComposedRollout,
    E10LocalRolloutResult,
    unwrap_e10_local_composed_rollout,
)
from src.celeba_gmd_structural_rollout import StructuralComposedRollout


class RecordingToyFlow(nn.Module):
    """Nonlinear deterministic flow recording every time/label/grad contract."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.2))
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _batch_time(value, state: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, device=state.device, dtype=state.dtype)
        if result.numel() == 1:
            result = result.reshape(1).expand(state.shape[0])
        elif result.numel() == state.shape[0]:
            result = result.reshape(state.shape[0])
        else:
            raise ValueError("time must be scalar or batch-sized")
        return result.reshape(state.shape[0], *([1] * (state.ndim - 1)))

    def forward(self, state, noise_labels, start_time, end_time):
        start = self._batch_time(start_time, state)
        end = self._batch_time(end_time, state)
        self.calls.append(
            {
                "labels": noise_labels,
                "labels_pointer": noise_labels.data_ptr(),
                "grad_enabled": torch.is_grad_enabled(),
                "start": start.detach().clone(),
                "end": end.detach().clone(),
            }
        )
        label_term = noise_labels.float().sum(dim=1).reshape(
            state.shape[0], *([1] * (state.ndim - 1))
        )
        span = end - start
        return state + span * (self.gain * state.square() + 0.01 * label_term)


def _inputs(*, requires_grad: bool = False):
    prior = torch.tensor(
        [[[-0.7, 0.2], [0.4, 0.9]], [[0.1, -0.3], [0.8, -0.2]]],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    labels = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64)
    return prior, labels


def _winner_e10_arguments(k: int):
    grid = torch.linspace(0.0, 1.0, k + 1, dtype=torch.float32)
    source_index = min(1, k - 1)
    start = grid[source_index]
    arguments = {
        "eulerian_rollout_index": source_index,
        "eulerian_start_time": start,
        "eulerian_local_end_time": start + torch.tensor(0.05),
        "eulerian_end_time": torch.tensor(1.0),
    }
    return grid, arguments


def _call_times(calls):
    return [
        (float(call["start"].reshape(-1)[0]), float(call["end"].reshape(-1)[0]))
        for call in calls
    ]


@pytest.mark.parametrize("k", [1, 2])
def test_structural_terminal_e10_and_gradients_match_exactly_for_k1_k2(k) -> None:
    combined_flow = RecordingToyFlow()
    reference_flow = copy.deepcopy(combined_flow)
    combined = E10LocalComposedRollout(combined_flow)
    reference = StructuralComposedRollout(reference_flow)
    prior, labels = _inputs(requires_grad=True)
    reference_prior = prior.detach().clone().requires_grad_(True)
    reference_labels = labels.clone()
    grid, structural_kwargs = _winner_e10_arguments(k)

    output = combined(
        prior,
        labels,
        k,
        grid,
        local_start_time=0.8,
        local_time_fraction=0.25,
        **structural_kwargs,
    )
    expected = reference(
        reference_prior,
        reference_labels,
        k,
        grid.clone(),
        **structural_kwargs,
    )

    assert isinstance(output, E10LocalRolloutResult)
    assert tuple(output.structural) == tuple(expected)
    for name, expected_value in expected.items():
        actual_value = output.structural[name]
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    output.local_origin_student.retain_grad()
    output.local_student.retain_grad()
    actual_objective = (
        output.structural["terminal"].square().mean()
        + 10.0 * output.structural["eulerian_loss"]
        + 0.0 * output.local_origin_student.float().sum()
        + 0.0 * output.local_student.float().sum()
    )
    expected_objective = (
        expected["terminal"].square().mean() + 10.0 * expected["eulerian_loss"]
    )
    torch.testing.assert_close(actual_objective, expected_objective, rtol=0, atol=0)
    actual_objective.backward()
    expected_objective.backward()
    torch.testing.assert_close(
        combined_flow.gain.grad, reference_flow.gain.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(prior.grad, reference_prior.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        output.local_origin_student.grad,
        torch.zeros_like(output.local_origin_student),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.local_student.grad,
        torch.zeros_like(output.local_student),
        rtol=0,
        atol=0,
    )

    # Structural calls are exactly the reference prefix; the three local
    # calls are appended afterward without replacing any endpoint call.
    assert len(reference_flow.calls) == k + 3
    assert len(combined_flow.calls) == k + 6
    assert _call_times(combined_flow.calls[: k + 3]) == _call_times(reference_flow.calls)
    torch.testing.assert_close(
        torch.tensor(_call_times(combined_flow.calls[-3:])),
        torch.tensor([(0.0, 0.8), (0.0, 0.85), (0.8, 0.85)]),
        rtol=0,
        atol=1e-7,
    )
    assert all(call["labels"] is labels for call in combined_flow.calls)
    assert len({call["labels_pointer"] for call in combined_flow.calls}) == 1
    assert [
        index
        for index, call in enumerate(combined_flow.calls)
        if not call["grad_enabled"]
    ] == [k + 1, k + 3]


def test_local_source_is_stopped_but_both_students_are_live() -> None:
    flow = RecordingToyFlow()
    model = E10LocalComposedRollout(flow)
    prior, labels = _inputs(requires_grad=True)
    grid, structural_kwargs = _winner_e10_arguments(2)

    output = model(
        prior,
        labels,
        2,
        grid,
        local_start_time=0.5,
        local_time_fraction=0.1,
        **structural_kwargs,
    )

    assert not output.local_source.requires_grad
    assert output.local_source.grad_fn is None
    assert output.local_origin_student.requires_grad
    assert output.local_origin_student.grad_fn is not None
    assert output.local_student.requires_grad
    assert output.local_student.grad_fn is not None
    output.local_origin_student.retain_grad()
    output.local_student.retain_grad()

    local_objective = (
        (output.local_origin_student - output.local_source).square().mean()
        + (output.local_student - output.local_source).square().mean()
    )
    local_objective.backward()
    assert output.local_origin_student.grad is not None
    assert float(output.local_origin_student.grad.abs().sum()) > 0.0
    assert output.local_student.grad is not None
    assert float(output.local_student.grad.abs().sum()) > 0.0
    assert prior.grad is None
    assert flow.gain.grad is not None and float(flow.gain.grad.abs()) > 0.0


@pytest.mark.parametrize(
    ("start", "fraction", "expected_end"),
    [
        (0.1, 0.05 / 0.9, 0.15),
        (0.5, 0.1, 0.55),
        (0.8, 0.25, 0.85),
        (0.9, 0.5, 0.95),
    ],
)
def test_true_short_fraction_produces_a_plus_point_zero_five(
    start, fraction, expected_end
) -> None:
    flow = RecordingToyFlow()
    model = E10LocalComposedRollout(flow)
    prior, labels = _inputs()
    grid, structural_kwargs = _winner_e10_arguments(1)

    output = model(
        prior,
        labels,
        1,
        grid,
        local_start_time=start,
        local_time_fraction=fraction,
        **structural_kwargs,
    )

    torch.testing.assert_close(
        output.local_start_time, torch.tensor(start), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        output.local_time_fraction, torch.tensor(fraction), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        output.local_end_time, torch.tensor(expected_end), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        output.local_end_time - output.local_start_time,
        torch.tensor(0.05),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.tensor(_call_times(flow.calls[-3:])),
        torch.tensor(
            [(0.0, start), (0.0, expected_end), (start, expected_end)]
        ),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"local_start_time": True, "local_time_fraction": 0.1}, TypeError, "not bool"),
        ({"local_start_time": -0.1, "local_time_fraction": 0.1}, ValueError, r"\[0, 1\)"),
        ({"local_start_time": 1.0, "local_time_fraction": 0.1}, ValueError, r"\[0, 1\)"),
        ({"local_start_time": 0.5, "local_time_fraction": False}, TypeError, "not bool"),
        ({"local_start_time": 0.5, "local_time_fraction": 0.0}, ValueError, r"\(0, 1\]"),
        ({"local_start_time": 0.5, "local_time_fraction": 1.1}, ValueError, r"\(0, 1\]"),
        (
            {"local_start_time": 0.5, "local_time_fraction": float("nan")},
            ValueError,
            "finite",
        ),
        (
            {
                "local_start_time": torch.tensor([0.2, 0.3]),
                "local_time_fraction": 0.1,
            },
            ValueError,
            "one scalar",
        ),
    ],
)
def test_invalid_local_times_are_rejected_before_any_flow_call(
    kwargs, error_type, message
) -> None:
    flow = RecordingToyFlow()
    model = E10LocalComposedRollout(flow)
    prior, labels = _inputs()
    grid, structural_kwargs = _winner_e10_arguments(1)
    with pytest.raises(error_type, match=message):
        model(prior, labels, 1, grid, **kwargs, **structural_kwargs)
    assert flow.calls == []


def test_invalid_prior_and_labels_are_rejected_before_any_flow_call() -> None:
    flow = RecordingToyFlow()
    model = E10LocalComposedRollout(flow)
    prior, labels = _inputs()
    grid, structural_kwargs = _winner_e10_arguments(1)

    bad_prior = prior.clone()
    bad_prior[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(
            bad_prior,
            labels,
            1,
            grid,
            local_start_time=0.5,
            local_time_fraction=0.1,
            **structural_kwargs,
        )
    with pytest.raises(TypeError, match="int64"):
        model(
            prior,
            labels.float(),
            1,
            grid,
            local_start_time=0.5,
            local_time_fraction=0.1,
            **structural_kwargs,
        )
    assert flow.calls == []


def test_structural_result_is_returned_intact_and_flow_is_registered_once() -> None:
    flow = RecordingToyFlow()
    model = E10LocalComposedRollout(flow)
    prior, labels = _inputs()
    grid, structural_kwargs = _winner_e10_arguments(1)
    captured: list[object] = []
    hook = model.structural_rollout.register_forward_hook(
        lambda _module, _inputs, result: captured.append(result)
    )
    try:
        output = model(
            prior,
            labels,
            1,
            grid,
            local_start_time=0.5,
            local_time_fraction=0.1,
            **structural_kwargs,
        )
    finally:
        hook.remove()

    assert output.structural is captured[0]
    assert model.flow is flow
    assert sum(
        isinstance(module, StructuralComposedRollout) for module in model.modules()
    ) == 1
    assert tuple(name for name, _ in model.named_parameters()) == (
        "structural_rollout.flow.gain",
    )
    assert unwrap_e10_local_composed_rollout(model) is model
    with pytest.raises(TypeError, match="E10LocalComposedRollout"):
        unwrap_e10_local_composed_rollout(nn.Linear(2, 2))


def test_wrapper_is_the_single_ddp_boundary_and_unwraps(tmp_path) -> None:
    if dist.is_initialized():
        pytest.skip("test requires ownership of the process group")
    init_file = tmp_path / "gloo_init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        outer = E10LocalComposedRollout(RecordingToyFlow())
        ddp = DistributedDataParallel(
            outer,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=False,
        )
        prior, labels = _inputs()
        grid, structural_kwargs = _winner_e10_arguments(2)
        output = ddp(
            prior,
            labels,
            2,
            grid,
            local_start_time=0.8,
            local_time_fraction=0.25,
            **structural_kwargs,
        )
        loss = (
            output.structural["terminal"].square().mean()
            + 10.0 * output.structural["eulerian_loss"]
            + output.local_origin_student.square().mean()
            + output.local_student.square().mean()
        )
        loss.backward()
        assert unwrap_e10_local_composed_rollout(ddp) is outer
        assert outer.flow.gain.grad is not None
        assert float(outer.flow.gain.grad.abs()) > 0.0
    finally:
        dist.destroy_process_group()
