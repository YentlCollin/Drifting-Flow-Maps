from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
from flowmaps.models.rgb import CelebA64TwoTimeGMDDiT
from flowmaps.models.rollout import (
    StructuralComposedRollout,
    structural_losses,
    unwrap_structural_composed_rollout,
)
from flowmaps.objectives.structure import (
    eulerian_local_composition_loss,
    weak_canonical_semigroup_loss,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class DeterministicToyFlow(nn.Module):
    """Small nonlinear map that records label identity and grad mode."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.2))
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _time(value, state: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, device=state.device, dtype=state.dtype)
        if result.numel() == 1:
            result = result.reshape(1).expand(state.shape[0])
        else:
            result = result.reshape(state.shape[0])
        return result.reshape(state.shape[0], *([1] * (state.ndim - 1)))

    def forward(self, state, noise_labels, start_time, end_time):
        start = self._time(start_time, state)
        end = self._time(end_time, state)
        if (
            bool((start > end).any())
            or bool((start < 0).any())
            or bool((end > 1).any())
        ):
            raise ValueError("invalid ordered times")
        self.calls.append(
            {
                "labels": noise_labels,
                "grad_enabled": torch.is_grad_enabled(),
                "start": start.detach().clone(),
                "end": end.detach().clone(),
            }
        )
        label_term = (
            noise_labels.float()
            .sum(dim=1)
            .reshape(state.shape[0], *([1] * (state.ndim - 1)))
        )
        span = end - start
        # Nonlinearity ensures the direct and composed branches are generally
        # distinct and therefore exercise nonzero consistency gradients.
        return state + span * (self.gain * state.square() + 0.01 * label_term)


def _inputs(*, requires_grad: bool = False):
    prior = torch.tensor(
        [[[-0.7, 0.2], [0.4, 0.9]], [[0.1, -0.3], [0.8, -0.2]]],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    labels = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64)
    grid = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32)
    return prior, labels, grid


def _full_call(model, prior, labels, grid, **overrides):
    arguments = {
        "semigroup_middle_time": torch.tensor(0.4),
        "semigroup_end_time": torch.tensor(0.9),
        "eulerian_rollout_index": 1,
        "eulerian_start_time": torch.tensor(0.25),
        "eulerian_local_end_time": torch.tensor(0.35),
        "eulerian_end_time": torch.tensor(0.8),
    }
    arguments.update(overrides)
    return model(prior, labels, 2, grid, **arguments)


def test_one_outer_call_returns_terminal_and_both_complete_branch_pairs() -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs(requires_grad=True)

    result = _full_call(model, prior, labels, grid)

    assert result["terminal"].shape == prior.shape
    torch.testing.assert_close(result["rollout_times"], grid)
    for prefix in ("semigroup", "eulerian"):
        assert result[f"{prefix}_loss"].ndim == 0
        assert result[f"{prefix}_rmse"].ndim == 0
        assert result[f"{prefix}_student"].requires_grad
        assert not result[f"{prefix}_teacher"].requires_grad
        torch.testing.assert_close(
            result[f"{prefix}_rmse"], result[f"{prefix}_loss"].detach().sqrt()
        )

    # K terminal calls plus three calls for each consistency equation.
    assert len(flow.calls) == 2 + 3 + 3
    assert all(call["labels"] is labels for call in flow.calls)
    assert sum(not call["grad_enabled"] for call in flow.calls) == 2

    objective = (
        result["terminal"].square().mean()
        + result["semigroup_loss"]
        + result["eulerian_loss"]
    )
    objective.backward()
    assert prior.grad is not None and float(prior.grad.abs().sum()) > 0
    assert flow.gain.grad is not None and float(flow.gain.grad.abs()) > 0


def test_canonical_branch_matches_existing_audited_helper_exactly() -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs()
    result = model(
        prior,
        labels,
        2,
        grid,
        semigroup_middle_time=0.4,
        semigroup_end_time=0.9,
    )
    expected_loss, expected_stats = weak_canonical_semigroup_loss(
        flow,
        prior,
        labels,
        0.4,
        0.9,
        detach_teacher=True,
    )
    torch.testing.assert_close(result["semigroup_loss"], expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        result["semigroup_rmse"], expected_stats["semigroup_rmse"], rtol=0, atol=0
    )


def test_eulerian_branch_matches_existing_audited_helper_exactly() -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs()
    state = prior + 0.125
    result = model(
        prior,
        labels,
        2,
        grid,
        eulerian_state=state,
        eulerian_start_time=0.2,
        eulerian_local_end_time=0.3,
        eulerian_end_time=0.85,
    )
    expected_loss, expected_stats = eulerian_local_composition_loss(
        flow,
        state,
        labels,
        0.2,
        0.3,
        0.85,
        detach_teacher=True,
    )
    torch.testing.assert_close(result["eulerian_loss"], expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        result["eulerian_rmse"],
        expected_stats["local_composition_rmse"],
        rtol=0,
        atol=0,
    )


def test_rollout_selected_eulerian_source_keeps_upstream_autograd() -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs(requires_grad=True)
    result = model(
        prior,
        labels,
        2,
        grid,
        eulerian_rollout_index=1,
        eulerian_start_time=0.25,
        eulerian_local_end_time=0.4,
        eulerian_end_time=0.75,
    )
    source = result["eulerian_source_state"]
    assert source.grad_fn is not None
    source.retain_grad()
    result["eulerian_loss"].backward()
    assert source.grad is not None and float(source.grad.abs().sum()) > 0
    assert prior.grad is not None and float(prior.grad.abs().sum()) > 0


def test_explicit_arbitrary_eulerian_state_receives_student_gradient_only() -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs()
    state = (prior + 0.31).detach().requires_grad_(True)
    result = model(
        prior,
        labels,
        2,
        grid,
        eulerian_state=state,
        eulerian_start_time=0.1,
        eulerian_local_end_time=0.2,
        eulerian_end_time=0.7,
    )
    assert result["eulerian_source_state"] is state
    assert not result["eulerian_teacher"].requires_grad
    result["eulerian_loss"].backward()
    assert state.grad is not None and float(state.grad.abs().sum()) > 0


def test_disabled_structural_branches_are_none_and_extract_to_empty() -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    result = model(prior, labels, 2, grid)
    assert result["terminal"] is not None
    assert all(
        value is None
        for name, value in result.items()
        if name not in {"terminal", "rollout_times"}
    )
    assert structural_losses(result) == {}


def test_loss_extractor_returns_only_enabled_live_scalar_losses() -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    result = _full_call(model, prior, labels, grid)
    losses = structural_losses(result)
    assert tuple(losses) == ("semigroup_loss", "eulerian_loss")
    assert losses["semigroup_loss"] is result["semigroup_loss"]
    assert losses["eulerian_loss"] is result["eulerian_loss"]


def test_forward_consumes_no_rng_and_repeated_call_is_exact() -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    before = torch.random.get_rng_state().clone()
    first = _full_call(model, prior, labels, grid)
    middle = torch.random.get_rng_state().clone()
    second = _full_call(model, prior, labels, grid)
    after = torch.random.get_rng_state().clone()
    torch.testing.assert_close(before, middle, rtol=0, atol=0)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for name in first:
        if first[name] is not None:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semigroup_middle_time": 0.5},
        {"semigroup_end_time": 1.0},
        {"eulerian_start_time": 0.1, "eulerian_local_end_time": 0.2},
        {"eulerian_end_time": 0.8},
    ],
)
def test_partial_time_groups_are_rejected_before_any_flow_call(kwargs) -> None:
    flow = DeterministicToyFlow()
    model = StructuralComposedRollout(flow)
    prior, labels, grid = _inputs()
    with pytest.raises(ValueError, match="supplied together"):
        model(prior, labels, 2, grid, **kwargs)
    assert flow.calls == []


def test_eulerian_source_selection_is_explicit_and_exclusive() -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    times = dict(
        eulerian_start_time=0.1,
        eulerian_local_end_time=0.2,
        eulerian_end_time=0.8,
    )
    with pytest.raises(ValueError, match="exactly one"):
        model(prior, labels, 2, grid, **times)
    with pytest.raises(ValueError, match="exactly one"):
        model(
            prior,
            labels,
            2,
            grid,
            eulerian_state=prior,
            eulerian_rollout_index=0,
            **times,
        )
    with pytest.raises(ValueError, match=r"\[0, K\]"):
        model(prior, labels, 2, grid, eulerian_rollout_index=3, **times)
    mismatched = dict(
        times,
        eulerian_start_time=0.24,
        eulerian_local_end_time=0.3,
    )
    with pytest.raises(ValueError, match="selected rollout state's grid time"):
        model(prior, labels, 2, grid, eulerian_rollout_index=1, **mismatched)
    with pytest.raises(ValueError, match="without Eulerian times"):
        model(prior, labels, 2, grid, eulerian_state=prior)


def test_grid_must_be_an_explicit_tensor_and_underlying_grid_checks_remain_active() -> (
    None
):
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    with pytest.raises(TypeError, match="caller-supplied Tensor"):
        model(prior, labels, 2, [0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="shape"):
        model(prior, labels, 2, grid[:2])
    with pytest.raises(ValueError, match="strictly increasing"):
        model(prior, labels, 2, torch.tensor([0.0, 0.7, 0.6]))


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"semigroup_middle_time": True, "semigroup_end_time": 1.0},
            TypeError,
            "not bool",
        ),
        (
            {"semigroup_middle_time": 0.7, "semigroup_end_time": 0.6},
            ValueError,
            "must satisfy",
        ),
        (
            {"semigroup_middle_time": 0.5, "semigroup_end_time": 1.1},
            ValueError,
            r"\[0, 1\]",
        ),
    ],
)
def test_canonical_times_are_strictly_validated(kwargs, error_type, message) -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    with pytest.raises(error_type, match=message):
        model(prior, labels, 2, grid, **kwargs)


@pytest.mark.parametrize("mutation", ["shape", "integer", "nan"])
def test_explicit_eulerian_state_must_match_the_live_state_contract(mutation) -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    if mutation == "shape":
        state = prior[:1]
    elif mutation == "integer":
        state = torch.zeros_like(prior, dtype=torch.int64)
    else:
        state = prior.clone()
        state[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="eulerian_state"):
        model(
            prior,
            labels,
            2,
            grid,
            eulerian_state=state,
            eulerian_start_time=0.1,
            eulerian_local_end_time=0.2,
            eulerian_end_time=0.8,
        )


def test_result_schema_is_strict() -> None:
    model = StructuralComposedRollout(DeterministicToyFlow())
    prior, labels, grid = _inputs()
    result = model(prior, labels, 2, grid)
    malformed = copy.copy(result)
    malformed["unexpected"] = None
    with pytest.raises(ValueError, match="schema differs"):
        structural_losses(malformed)


def test_wrapper_is_the_single_ddp_boundary(tmp_path) -> None:
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
        outer = StructuralComposedRollout(DeterministicToyFlow())
        ddp = DistributedDataParallel(outer, broadcast_buffers=False)
        prior, labels, grid = _inputs(requires_grad=True)
        result = _full_call(ddp, prior, labels, grid)
        loss = result["terminal"].mean() + sum(structural_losses(result).values())
        loss.backward()
        assert unwrap_structural_composed_rollout(ddp) is outer
        assert outer.flow.gain.grad is not None
        assert float(outer.flow.gain.grad.abs()) > 0
    finally:
        dist.destroy_process_group()


def test_unwrap_rejects_unrelated_or_nested_modules() -> None:
    with pytest.raises(TypeError, match="StructuralComposedRollout"):
        unwrap_structural_composed_rollout(nn.Linear(2, 2))


def test_real_two_time_flow_contract_runs_all_branches_and_backpropagates() -> None:
    torch.manual_seed(91)
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
    with torch.no_grad():
        flow.final.output.weight.normal_(std=0.02)
        flow.final.output.bias.normal_(std=0.02)
    model = StructuralComposedRollout(flow).train()
    prior, labels = flow.sample_prior(
        1,
        device="cpu",
        generator=torch.Generator().manual_seed(17),
    )
    grid = torch.tensor([0.0, 0.5, 1.0])
    result = model(
        prior,
        labels,
        2,
        grid,
        semigroup_middle_time=0.5,
        semigroup_end_time=1.0,
        eulerian_rollout_index=1,
        eulerian_start_time=0.5,
        eulerian_local_end_time=0.6,
        eulerian_end_time=0.9,
    )
    assert result["terminal"].shape == (1, 3, 64, 64)
    loss = result["terminal"].square().mean() + sum(structural_losses(result).values())
    loss.backward()
    assert flow.final.output.weight.grad is not None
    assert bool(torch.isfinite(flow.final.output.weight.grad).all())
