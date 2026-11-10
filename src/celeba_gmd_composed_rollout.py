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

from src.celeba_gmd_flow_map import (
    FLOW_ARCHITECTURE_NAME,
    FLOW_ARCHITECTURE_VERSION,
    CelebA64TwoTimeGMDDiT,
)
from src.celeba_gmd_flow_objectives import differentiable_terminal_output
from src.celeba_pixel_gmd_features import (
    canonical_json_sha256,
    state_dict_sha256,
)
from src.celeba_pixel_gmd_generator import generator_from_architecture


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
_FLOW_TIME_CONDITIONING = (
    "fourier_s_t_delta_then_zero_initialized_projection"
)
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


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "ComposedRollout",
    "composed_rollout_artifact_dict",
    "flow_from_pinned_parent_payload",
    "load_composed_rollout_artifact",
    "unwrap_composed_rollout",
]
