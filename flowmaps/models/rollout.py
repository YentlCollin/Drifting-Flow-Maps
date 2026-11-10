"""Single-boundary differentiable composition for the CelebA GMD flow map.

``ComposedRollout`` deliberately keeps the rollout length out of module state
and serialized artifacts.  A positive Python integer is supplied to each
forward call and is shared by the whole minibatch, while the same noise labels
are reused at every transition.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from flowmaps.features.mae import canonical_json_sha256, state_dict_sha256
from flowmaps.models.backbone import generator_from_architecture
from flowmaps.models.rgb import (
    FLOW_ARCHITECTURE_NAME,
    FLOW_ARCHITECTURE_VERSION,
    CelebA64TwoTimeGMDDiT,
)
from flowmaps.objectives.structure import differentiable_terminal_output

ARTIFACT_FORMAT = "driftmodel.celeba64_two_time_gmd_composed_rollout"
ARTIFACT_VERSION = 1

_FLOW_ARCHITECTURE_FIELDS = {
    "name",
    "version",
    "parameterization",
    "time_domain",
    "time_conditioning",
    "flow_time_frequency_embedding_size",
    "one_step_backbone",
}
_FLOW_PARAMETERIZATION = "x_plus_(t-s)/(1-s)_times_endpoint_minus_x"
_FLOW_TIME_CONDITIONING = "fourier_s_t_delta_then_zero_initialized_projection"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ComposedRollout(nn.Module):
    """Expose a complete fixed-label rollout as one outer module call."""

    flow: CelebA64TwoTimeGMDDiT

    def __init__(self, flow: CelebA64TwoTimeGMDDiT) -> None:
        super().__init__()
        # A subclass may override forward while retaining the canonical state
        # and architecture config, which could not be reconstructed exactly.
        if type(flow) is not CelebA64TwoTimeGMDDiT:
            raise TypeError("flow must be exactly CelebA64TwoTimeGMDDiT")
        self.flow = flow

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        k: int,
    ) -> Tensor:
        """Return terminal RGB after exactly ``k`` uniform transitions."""

        return differentiable_terminal_output(
            self.flow,
            prior,
            noise_labels,
            k=k,
        )


def unwrap_composed_rollout(module: nn.Module) -> ComposedRollout:
    """Return a rollout directly or from one DistributedDataParallel layer."""

    if isinstance(module, ComposedRollout):
        return module
    if isinstance(module, nn.parallel.DistributedDataParallel) and isinstance(
        module.module, ComposedRollout
    ):
        return module.module
    raise TypeError("module must be ComposedRollout or DDP[ComposedRollout]")


def _flow_from_architecture(
    architecture: Mapping[str, Any],
) -> CelebA64TwoTimeGMDDiT:
    if not isinstance(architecture, Mapping) or set(architecture) != (
        _FLOW_ARCHITECTURE_FIELDS
    ):
        raise ValueError("flow architecture fields are not exact")
    if (
        type(architecture["name"]) is not str
        or architecture["name"] != FLOW_ARCHITECTURE_NAME
        or type(architecture["version"]) is not int
        or architecture["version"] != FLOW_ARCHITECTURE_VERSION
    ):
        raise ValueError("unsupported flow architecture")
    if (
        type(architecture["parameterization"]) is not str
        or architecture["parameterization"] != _FLOW_PARAMETERIZATION
        or type(architecture["time_conditioning"]) is not str
        or architecture["time_conditioning"] != _FLOW_TIME_CONDITIONING
    ):
        raise ValueError("flow-map semantics differ")
    time_domain = architecture["time_domain"]
    if (
        type(time_domain) is not list
        or len(time_domain) != 2
        or any(type(value) is not float for value in time_domain)
        or time_domain != [0.0, 1.0]
    ):
        raise ValueError("flow time-domain semantics differ")
    frequency_size = architecture["flow_time_frequency_embedding_size"]
    if type(frequency_size) is not int or frequency_size < 2:
        raise ValueError("flow-time frequency embedding size is malformed")
    backbone_architecture = architecture["one_step_backbone"]
    if not isinstance(backbone_architecture, Mapping):
        raise ValueError("flow architecture lacks a one-step backbone mapping")

    backbone = generator_from_architecture(backbone_architecture)
    flow = CelebA64TwoTimeGMDDiT.from_one_step(
        backbone,
        flow_time_frequency_embedding_size=frequency_size,
    )
    # Compare canonical encodings as well as values so bool/int aliases or
    # other JSON type substitutions cannot pass through Python equality.
    if canonical_json_sha256(flow.flow_architecture_config()) != (
        canonical_json_sha256(dict(architecture))
    ):
        raise ValueError("flow architecture is not canonical")
    return flow


def composed_rollout_artifact_dict(
    model: nn.Module,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a strict inference artifact containing only the trained flow.

    The rollout length is intentionally absent: ``k`` remains a runtime input
    so one artifact can be evaluated under any homogeneous composition count.
    """

    rollout = unwrap_composed_rollout(model)
    if not isinstance(metadata, Mapping):
        raise TypeError("artifact metadata must be a mapping")
    metadata_copy = copy.deepcopy(dict(metadata))
    canonical_json_sha256(metadata_copy)
    architecture = copy.deepcopy(rollout.flow.flow_architecture_config())
    state = {
        name: value.detach().to(device="cpu").clone()
        for name, value in rollout.flow.state_dict().items()
    }
    integrity = {
        "algorithm": "sha256",
        "architecture_sha256": canonical_json_sha256(architecture),
        "metadata_sha256": canonical_json_sha256(metadata_copy),
        "state_dict_sha256": state_dict_sha256(state),
    }
    return {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "architecture": architecture,
        "metadata": metadata_copy,
        "state_dict": state,
        "integrity": integrity,
    }


def load_composed_rollout_artifact(
    artifact: str | Path | Mapping[str, Any],
    device: torch.device | str,
) -> tuple[ComposedRollout, dict[str, Any]]:
    """Verify, reconstruct, freeze, and return a composed-rollout artifact."""

    payload = (
        artifact
        if isinstance(artifact, Mapping)
        else torch.load(Path(artifact), map_location="cpu", weights_only=True)
    )
    expected_top = {
        "format",
        "version",
        "architecture",
        "metadata",
        "state_dict",
        "integrity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("composed-rollout artifact fields are not exact")
    if (
        type(payload["format"]) is not str
        or payload["format"] != ARTIFACT_FORMAT
        or type(payload["version"]) is not int
        or payload["version"] != ARTIFACT_VERSION
    ):
        raise ValueError("unsupported composed-rollout artifact")

    architecture = payload["architecture"]
    metadata = payload["metadata"]
    state = payload["state_dict"]
    integrity = payload["integrity"]
    if not all(
        isinstance(value, Mapping)
        for value in (architecture, metadata, state, integrity)
    ):
        raise ValueError("composed-rollout artifact contains malformed mappings")
    expected_integrity = {
        "algorithm": "sha256",
        "architecture_sha256": canonical_json_sha256(architecture),
        "metadata_sha256": canonical_json_sha256(metadata),
        "state_dict_sha256": state_dict_sha256(state),
    }
    if dict(integrity) != expected_integrity:
        raise ValueError("composed-rollout artifact internal SHA-256 mismatch")

    flow = _flow_from_architecture(architecture)
    flow.load_state_dict(state, strict=True)
    rollout = ComposedRollout(flow).to(device).eval().requires_grad_(False)
    return rollout, copy.deepcopy(dict(metadata))


def _validate_expected_sha256(value: str | None, *, name: str) -> None:
    if value is not None and (
        type(value) is not str or _SHA256.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def flow_from_pinned_parent_payload(
    payload: Mapping[str, Any],
    *,
    expected_step: int,
    expected_objective: str,
    expected_ema_state_sha256: str | None = None,
    expected_training_signature_sha256: str | None = None,
    flow_time_frequency_embedding_size: int = 256,
) -> CelebA64TwoTimeGMDDiT:
    """Create a trainable two-time flow from an exactly checked parent EMA."""

    if not isinstance(payload, Mapping):
        raise TypeError("parent payload must be a mapping")
    if type(expected_step) is not int or expected_step < 0:
        raise ValueError("expected_step must be a non-negative Python integer")
    if type(expected_objective) is not str:
        raise TypeError("expected_objective must be a string")
    if type(payload.get("completed_step")) is not int:
        raise ValueError("parent completed_step must be a Python integer")
    if payload["completed_step"] != expected_step:
        raise ValueError("parent completed_step differs from the pinned step")

    signature = payload.get("training_signature")
    ema_state = payload.get("ema")
    if not isinstance(signature, Mapping) or not isinstance(ema_state, Mapping):
        raise ValueError("parent payload lacks training_signature or EMA state")
    if signature.get("objective") != expected_objective:
        raise ValueError("parent objective differs from the pinned objective")

    _validate_expected_sha256(
        expected_ema_state_sha256,
        name="expected_ema_state_sha256",
    )
    _validate_expected_sha256(
        expected_training_signature_sha256,
        name="expected_training_signature_sha256",
    )
    ema_sha256 = state_dict_sha256(ema_state)
    signature_sha256 = canonical_json_sha256(dict(signature))
    if (
        expected_ema_state_sha256 is not None
        and ema_sha256 != expected_ema_state_sha256
    ):
        raise ValueError("parent EMA state SHA-256 differs")
    if (
        expected_training_signature_sha256 is not None
        and signature_sha256 != expected_training_signature_sha256
    ):
        raise ValueError("parent training-signature SHA-256 differs")

    # Full training checkpoints carry hashes for these two consumed pieces.
    # Validate them when present without coupling this adapter to optimizer,
    # data-stream, or RNG checkpoint schemas.
    integrity = payload.get("integrity")
    if integrity is not None:
        if not isinstance(integrity, Mapping) or integrity.get("algorithm") != "sha256":
            raise ValueError("parent integrity metadata is malformed")
        if integrity.get("ema_state_sha256") != ema_sha256:
            raise ValueError("parent internal EMA SHA-256 differs")
        if integrity.get("training_signature_sha256") != signature_sha256:
            raise ValueError("parent internal training-signature SHA-256 differs")

    architecture = signature.get("generator")
    if not isinstance(architecture, Mapping):
        raise ValueError("parent training signature lacks generator architecture")
    one_step = generator_from_architecture(architecture)
    one_step.load_state_dict(ema_state, strict=True)
    return CelebA64TwoTimeGMDDiT.from_one_step(
        one_step,
        flow_time_frequency_embedding_size=flow_time_frequency_embedding_size,
    )


from collections.abc import Mapping
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from flowmaps.objectives.structure import (
    DifferentiableRollout,
    differentiable_fixed_label_rollout,
)

StructuralRolloutResult: TypeAlias = dict[str, Tensor | None]

_RESULT_KEYS = (
    "terminal",
    "rollout_times",
    "semigroup_loss",
    "semigroup_rmse",
    "semigroup_middle_state",
    "semigroup_student",
    "semigroup_teacher",
    "eulerian_loss",
    "eulerian_rmse",
    "eulerian_source_state",
    "eulerian_local_state",
    "eulerian_student",
    "eulerian_teacher",
)


def _optional_group(name: str, values: tuple[object | None, ...]) -> bool:
    """Return whether an all-or-none optional argument group is enabled."""

    supplied = tuple(value is not None for value in values)
    if any(supplied) and not all(supplied):
        raise ValueError(f"{name} arguments must be supplied together")
    return all(supplied)


def _validate_flow_result(result: Tensor, state: Tensor, *, branch: str) -> Tensor:
    if not isinstance(result, Tensor):
        raise TypeError(f"{branch} flow result must be a torch.Tensor")
    if result.shape != state.shape:
        raise ValueError(
            f"{branch} flow result must preserve state shape: "
            f"{tuple(state.shape)} -> {tuple(result.shape)}"
        )
    if result.device != state.device:
        raise ValueError(f"{branch} flow result changed device")
    if not result.is_floating_point():
        raise TypeError(f"{branch} flow result must be floating point")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{branch} flow result is non-finite")
    return result


def _batch_time(value: Tensor | float, state: Tensor, *, name: str) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be real-valued, not bool")
    result = torch.as_tensor(value, device=state.device, dtype=torch.float32)
    if result.numel() == 1:
        result = result.reshape(1).expand(state.shape[0])
    elif result.numel() == state.shape[0]:
        result = result.reshape(state.shape[0])
    else:
        raise ValueError(f"{name} must be scalar or batch-sized")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _ordered_times(
    state: Tensor,
    values: tuple[Tensor | float, ...],
    names: tuple[str, ...],
) -> tuple[Tensor, ...]:
    times = tuple(
        _batch_time(value, state, name=name)
        for value, name in zip(values, names, strict=True)
    )
    if any(bool((right < left).any()) for left, right in zip(times[:-1], times[1:])):
        raise ValueError(f"times must satisfy {' <= '.join(names)}")
    return times


def _loss_and_rmse(
    student: Tensor, teacher: Tensor, *, branch: str
) -> tuple[Tensor, Tensor]:
    if student.shape != teacher.shape:
        raise ValueError(f"{branch} student and teacher shapes differ")
    loss = (student - teacher).square().mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"{branch} loss is non-finite")
    return loss, loss.detach().sqrt()


def _canonical_semigroup_branches(
    flow: nn.Module,
    initial_state: Tensor,
    noise_labels: Tensor,
    middle_time: Tensor | float,
    end_time: Tensor | float,
) -> dict[str, Tensor]:
    """Build ``Phi(Phi(z,0,a),a,b)`` and stopped ``Phi(z,0,b)``."""

    middle = _validate_flow_result(
        flow(initial_state, noise_labels, 0.0, middle_time),
        initial_state,
        branch="canonical middle",
    )
    with torch.no_grad():
        teacher = _validate_flow_result(
            flow(initial_state, noise_labels, 0.0, end_time),
            initial_state,
            branch="canonical teacher",
        ).detach()
    student = _validate_flow_result(
        flow(middle, noise_labels, middle_time, end_time),
        middle,
        branch="canonical student",
    )
    loss, rmse = _loss_and_rmse(student, teacher, branch="canonical semigroup")
    return {
        "loss": loss,
        "rmse": rmse,
        "middle_state": middle,
        "student": student,
        "teacher": teacher,
    }


def _eulerian_composition_branches(
    flow: nn.Module,
    state: Tensor,
    noise_labels: Tensor,
    start_time: Tensor | float,
    local_end_time: Tensor | float,
    end_time: Tensor | float,
) -> dict[str, Tensor]:
    """Build ``Phi(Phi(x,s,u),u,t)`` and stopped ``Phi(x,s,t)``."""

    local = _validate_flow_result(
        flow(state, noise_labels, start_time, local_end_time),
        state,
        branch="Eulerian local",
    )
    with torch.no_grad():
        teacher = _validate_flow_result(
            flow(state, noise_labels, start_time, end_time),
            state,
            branch="Eulerian teacher",
        ).detach()
    student = _validate_flow_result(
        flow(local, noise_labels, local_end_time, end_time),
        local,
        branch="Eulerian student",
    )
    loss, rmse = _loss_and_rmse(student, teacher, branch="Eulerian composition")
    return {
        "loss": loss,
        "rmse": rmse,
        "local_state": local,
        "student": student,
        "teacher": teacher,
    }


class StructuralComposedRollout(nn.Module):
    """Return terminal and optional stopped-teacher consistency branches.

    Wrap this object itself in DDP.  Do not wrap ``flow`` separately and do
    not invoke ``DDP.module.flow`` during a training update.

    ``rollout_time_grid`` is mandatory and has shape ``[K + 1]``.  Canonical
    semigroup evaluation is disabled when both canonical times are ``None``;
    otherwise both must be supplied.  Eulerian evaluation is similarly
    all-or-none for its three times.  Its source is either an explicit
    ``eulerian_state`` or one state selected from the live terminal rollout by
    ``eulerian_rollout_index``.  Selecting a rollout state preserves its
    upstream graph.
    """

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        if not isinstance(flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        self.flow = flow

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        k: int,
        rollout_time_grid: Tensor,
        *,
        semigroup_middle_time: Tensor | float | None = None,
        semigroup_end_time: Tensor | float | None = None,
        eulerian_state: Tensor | None = None,
        eulerian_rollout_index: int | None = None,
        eulerian_start_time: Tensor | float | None = None,
        eulerian_local_end_time: Tensor | float | None = None,
        eulerian_end_time: Tensor | float | None = None,
    ) -> StructuralRolloutResult:
        if not isinstance(rollout_time_grid, Tensor):
            raise TypeError("rollout_time_grid must be a caller-supplied Tensor")

        canonical_enabled = _optional_group(
            "canonical semigroup",
            (semigroup_middle_time, semigroup_end_time),
        )
        eulerian_enabled = _optional_group(
            "Eulerian composition",
            (
                eulerian_start_time,
                eulerian_local_end_time,
                eulerian_end_time,
            ),
        )
        source_options = int(eulerian_state is not None) + int(
            eulerian_rollout_index is not None
        )
        if eulerian_enabled and source_options != 1:
            raise ValueError(
                "Eulerian composition requires exactly one of eulerian_state "
                "or eulerian_rollout_index"
            )
        if not eulerian_enabled and source_options:
            raise ValueError("Eulerian source was supplied without Eulerian times")

        rollout: DifferentiableRollout = differentiable_fixed_label_rollout(
            self.flow,
            prior,
            noise_labels,
            k=k,
            time_grid=rollout_time_grid,
        )
        result: StructuralRolloutResult = {key: None for key in _RESULT_KEYS}
        result["terminal"] = rollout.terminal
        result["rollout_times"] = rollout.times

        if canonical_enabled:
            assert semigroup_middle_time is not None
            assert semigroup_end_time is not None
            canonical_middle, canonical_end = _ordered_times(
                prior,
                (semigroup_middle_time, semigroup_end_time),
                ("semigroup_middle_time", "semigroup_end_time"),
            )
            canonical = _canonical_semigroup_branches(
                self.flow,
                prior,
                noise_labels,
                canonical_middle,
                canonical_end,
            )
            result.update(
                semigroup_loss=canonical["loss"],
                semigroup_rmse=canonical["rmse"],
                semigroup_middle_state=canonical["middle_state"],
                semigroup_student=canonical["student"],
                semigroup_teacher=canonical["teacher"],
            )

        if eulerian_enabled:
            if eulerian_state is not None:
                source = eulerian_state
                if source.shape != prior.shape or source.device != prior.device:
                    raise ValueError(
                        "explicit eulerian_state must match prior shape and device"
                    )
                if not source.is_floating_point() or not bool(
                    torch.isfinite(source).all()
                ):
                    raise ValueError(
                        "explicit eulerian_state must be finite floating point"
                    )
            else:
                if (
                    isinstance(eulerian_rollout_index, bool)
                    or not isinstance(eulerian_rollout_index, int)
                    or not 0 <= eulerian_rollout_index <= rollout.k
                ):
                    raise ValueError("eulerian_rollout_index must lie in [0, K]")
                source = rollout.states[eulerian_rollout_index]
            assert eulerian_start_time is not None
            assert eulerian_local_end_time is not None
            assert eulerian_end_time is not None
            eulerian_start, eulerian_local_end, eulerian_end = _ordered_times(
                source,
                (
                    eulerian_start_time,
                    eulerian_local_end_time,
                    eulerian_end_time,
                ),
                (
                    "eulerian_start_time",
                    "eulerian_local_end_time",
                    "eulerian_end_time",
                ),
            )
            if eulerian_state is None:
                expected_start = rollout.times[eulerian_rollout_index]
                if bool((eulerian_start != expected_start).any()):
                    raise ValueError(
                        "eulerian_start_time must equal the selected rollout "
                        "state's grid time"
                    )
            eulerian = _eulerian_composition_branches(
                self.flow,
                source,
                noise_labels,
                eulerian_start,
                eulerian_local_end,
                eulerian_end,
            )
            result.update(
                eulerian_loss=eulerian["loss"],
                eulerian_rmse=eulerian["rmse"],
                eulerian_source_state=source,
                eulerian_local_state=eulerian["local_state"],
                eulerian_student=eulerian["student"],
                eulerian_teacher=eulerian["teacher"],
            )

        if tuple(result) != _RESULT_KEYS:
            raise RuntimeError("structural result schema changed unexpectedly")
        return result


def unwrap_structural_composed_rollout(module: nn.Module) -> StructuralComposedRollout:
    """Return this wrapper directly or through exactly one DDP layer."""

    if isinstance(module, StructuralComposedRollout):
        return module
    if isinstance(module, DistributedDataParallel) and isinstance(
        module.module, StructuralComposedRollout
    ):
        return module.module
    raise TypeError(
        "module must be StructuralComposedRollout or "
        "DistributedDataParallel[StructuralComposedRollout]"
    )


def structural_losses(result: Mapping[str, Tensor | None]) -> dict[str, Tensor]:
    """Extract enabled scalar losses from an exact wrapper result schema."""

    if tuple(result) != _RESULT_KEYS:
        raise ValueError("structural rollout result schema differs")
    losses: dict[str, Tensor] = {}
    for name in ("semigroup_loss", "eulerian_loss"):
        value = result[name]
        if value is not None:
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError(f"{name} must be a scalar Tensor")
            losses[name] = value
    return losses


from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from flowmaps.objectives.local_drifting import homogeneous_two_view_states


class E10LocalRolloutResult(NamedTuple):
    """Unchanged structural result plus one homogeneous local two-view graph."""

    structural: StructuralRolloutResult
    local_start_time: Tensor
    local_time_fraction: Tensor
    local_end_time: Tensor
    local_source: Tensor
    local_origin_student: Tensor
    local_student: Tensor


def _validate_local_inputs(prior: Tensor, noise_labels: Tensor) -> None:
    """Reject malformed local inputs before executing the structural graph."""

    if not isinstance(prior, Tensor):
        raise TypeError("prior must be a torch.Tensor")
    if (
        prior.ndim < 2
        or prior.shape[0] < 1
        or any(size < 1 for size in prior.shape[1:])
    ):
        raise ValueError("prior must have a non-empty batch and feature dimensions")
    if not prior.is_floating_point():
        raise TypeError("prior must be floating point")
    if not bool(torch.isfinite(prior).all()):
        raise ValueError("prior must contain only finite values")

    if not isinstance(noise_labels, Tensor):
        raise TypeError("noise_labels must be a torch.Tensor")
    if noise_labels.ndim != 2 or noise_labels.shape[0] != prior.shape[0]:
        raise ValueError("noise_labels must have shape [B, noise_coords]")
    if noise_labels.shape[1] < 1 or noise_labels.dtype != torch.int64:
        raise TypeError("noise_labels must contain int64 noise coordinates")
    if noise_labels.device != prior.device:
        raise ValueError("prior and noise_labels must be on the same device")


def _local_times(
    prior: Tensor,
    *,
    start_time: Tensor | float,
    time_fraction: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached scalar ``(a, alpha, a + alpha * (1 - a))`` tensors."""

    values: list[Tensor] = []
    for name, value in (
        ("local_start_time", start_time),
        ("local_time_fraction", time_fraction),
    ):
        if isinstance(value, bool) or (
            isinstance(value, Tensor) and value.dtype == torch.bool
        ):
            raise TypeError(f"{name} must be a real scalar, not bool")
        tensor = torch.as_tensor(value, device=prior.device, dtype=torch.float32)
        if tensor.numel() != 1:
            raise ValueError(f"{name} must be one scalar shared by the batch")
        tensor = tensor.reshape(()).detach()
        if not bool(torch.isfinite(tensor)):
            raise ValueError(f"{name} must be finite")
        values.append(tensor)

    start, fraction = values
    if not bool((start >= 0.0) & (start < 1.0)):
        raise ValueError("local_start_time must lie in [0, 1)")
    if not bool((fraction > 0.0) & (fraction <= 1.0)):
        raise ValueError("local_time_fraction must lie in (0, 1]")

    end = start + fraction * (1.0 - start)
    if not bool(torch.isfinite(end)) or not bool((end > start) & (end <= 1.0)):
        raise ValueError(
            "local times must produce a finite end strictly after the start "
            "and no later than one"
        )
    return start, fraction, end.detach()


class E10LocalComposedRollout(nn.Module):
    """Preserve the structural K/E10 graph and append two live local views.

    Wrap this object itself in DDP.  ``flow`` is registered exactly once,
    beneath ``structural_rollout``; the read-only :attr:`flow` property is
    provided for checkpoint and optimizer construction.
    """

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        if not isinstance(flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        if isinstance(flow, DistributedDataParallel):
            raise TypeError(
                "flow must be unwrapped; wrap E10LocalComposedRollout in DDP"
            )
        self.structural_rollout = StructuralComposedRollout(flow)

    @property
    def flow(self) -> nn.Module:
        """Return the single flow owned by the nested structural wrapper."""

        return self.structural_rollout.flow

    def forward(
        self,
        prior: Tensor,
        noise_labels: Tensor,
        k: int,
        rollout_time_grid: Tensor,
        *,
        local_start_time: Tensor | float,
        local_time_fraction: Tensor | float,
        semigroup_middle_time: Tensor | float | None = None,
        semigroup_end_time: Tensor | float | None = None,
        eulerian_state: Tensor | None = None,
        eulerian_rollout_index: int | None = None,
        eulerian_start_time: Tensor | float | None = None,
        eulerian_local_end_time: Tensor | float | None = None,
        eulerian_end_time: Tensor | float | None = None,
    ) -> E10LocalRolloutResult:
        _validate_local_inputs(prior, noise_labels)
        local_start, local_fraction, expected_local_end = _local_times(
            prior,
            start_time=local_start_time,
            time_fraction=local_time_fraction,
        )

        # This call, including its live rollout-selected E10 source, remains
        # byte-for-byte delegated to the audited structural implementation.
        structural = self.structural_rollout(
            prior,
            noise_labels,
            k,
            rollout_time_grid,
            semigroup_middle_time=semigroup_middle_time,
            semigroup_end_time=semigroup_end_time,
            eulerian_state=eulerian_state,
            eulerian_rollout_index=eulerian_rollout_index,
            eulerian_start_time=eulerian_start_time,
            eulerian_local_end_time=eulerian_local_end_time,
            eulerian_end_time=eulerian_end_time,
        )

        # The helper completes the stopped source under no_grad and autocast
        # cache_enabled=False before executing either live student.  DDP time
        # synchronization belongs to the caller, alongside its checkpointed
        # local-time RNG, so it is intentionally disabled here.
        local = homogeneous_two_view_states(
            self.flow,
            prior,
            noise_labels,
            start_time=local_start,
            time_fraction=local_fraction,
            synchronize_distributed=False,
        )
        if not torch.equal(local.end_time, expected_local_end):
            raise RuntimeError("local two-view helper changed the end-time equation")

        return E10LocalRolloutResult(
            structural=structural,
            local_start_time=local.start_time,
            local_time_fraction=local.time_fraction,
            local_end_time=local.end_time,
            local_source=local.source,
            local_origin_student=local.origin_student,
            local_student=local.local_student,
        )


def unwrap_e10_local_composed_rollout(module: nn.Module) -> E10LocalComposedRollout:
    """Return this wrapper directly or through exactly one outer DDP layer."""

    if isinstance(module, E10LocalComposedRollout):
        return module
    if isinstance(module, DistributedDataParallel) and isinstance(
        module.module, E10LocalComposedRollout
    ):
        return module.module
    raise TypeError(
        "module must be E10LocalComposedRollout or "
        "DistributedDataParallel[E10LocalComposedRollout]"
    )


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "ComposedRollout",
    "E10LocalComposedRollout",
    "E10LocalRolloutResult",
    "StructuralComposedRollout",
    "StructuralRolloutResult",
    "composed_rollout_artifact_dict",
    "flow_from_pinned_parent_payload",
    "load_composed_rollout_artifact",
    "structural_losses",
    "unwrap_composed_rollout",
    "unwrap_e10_local_composed_rollout",
    "unwrap_structural_composed_rollout",
]
