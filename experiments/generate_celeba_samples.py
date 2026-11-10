#!/usr/bin/env python3
"""Generate deterministic individual CelebA PNGs from pixel or latent runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.celeba_latent import (
    LATENT_PRIOR_FILENAME,
    LatentAffineNormalization,
    LatentGaussianPrior,
)
PIXEL_METHODS = ("drift", "fast_wgf_flow_map")
LATENT_METHODS = ("drift", "flow")
WEIGHT_KINDS = ("auto", "raw", "ema")
LATENT_CHECKPOINT_FORMAT = "driftmodel.celeba_latent_flow"
LATENT_DIM = 4 * 8 * 8
PIXEL_DIM = 3 * 64 * 64


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    space: str
    method: str
    config_path: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    requested_weights: str
    resolved_weights: str
    container: str
    completed_step: int | None
    reason: str


@dataclass(frozen=True)
class GenerationPlan:
    run: RunSpec
    checkpoint: CheckpointSelection
    output_dir: Path
    num_samples: int
    batch_size: int
    decode_batch_size: int
    nfe: int
    seed: int
    time_power: float
    device: torch.device
    vae: dict[str, Any] | None
    latent_normalization: LatentAffineNormalization
    latent_prior: LatentGaussianPrior
    latent_prior_path: Path | None
    latent_prior_sha256: str | None
    training_checkpoint_path: Path | None
    training_checkpoint_sha256: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible CelebA PNG directories from trained runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True, help="Pixel or latent run directory.")
    parser.add_argument("--space", choices=("auto", "pixel", "latent"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint; relative paths are first resolved below --run.",
    )
    parser.add_argument(
        "--weights",
        choices=WEIGHT_KINDS,
        default="auto",
        help="Requested raw/EMA weights; auto prefers a final EMA then the newest usable weights.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument(
        "--repeat-drift",
        action="store_true",
        help="Permit NFE > 1 for drift by repeatedly applying its endpoint model (off-distribution diagnostic).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-power",
        type=float,
        default=None,
        help="Override flow partition power; otherwise use the run config.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--vae-model", type=Path, default=None, help="Override the latent run's local VAE.")
    parser.add_argument(
        "--vae-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate inputs without loading a model or writing files.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="End-to-end checkpoint smoke: cap generation to 2 images, batch 2, NFE 1.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.num_samples = min(args.num_samples, 2)
        args.batch_size = min(args.batch_size, 2)
        args.decode_batch_size = min(args.decode_batch_size, 2)
        args.nfe = 1
    if args.num_samples < 1 or args.batch_size < 1 or args.decode_batch_size < 1:
        raise ValueError("sample and batch sizes must be positive")
    if args.nfe < 1:
        raise ValueError("--nfe must be positive")
    if args.time_power is not None and (not math.isfinite(args.time_power) or args.time_power <= 0):
        raise ValueError("--time-power must be finite and positive")
    return args


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        specification = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA/ROCm is unavailable")
    return device


def parse_config_value(value: str) -> Any:
    stripped = value.strip()
    if stripped in {"None", "null"}:
        return None
    if stripped in {"True", "true"}:
        return True
    if stripped in {"False", "false"}:
        return False
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return stripped


def load_key_value_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed config line {line_number} in {path}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in config:
            raise ValueError(f"empty or duplicate config key on line {line_number} in {path}")
        config[key] = parse_config_value(value)
    return config


def load_run_spec(run_dir: Path, requested_space: str = "auto") -> RunSpec:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    pixel_config = run_dir / "config.txt"
    latent_config = run_dir / "config.json"
    if requested_space == "auto":
        if latent_config.is_file() and not pixel_config.is_file():
            space = "latent"
        elif pixel_config.is_file() and not latent_config.is_file():
            space = "pixel"
        elif latent_config.is_file() and pixel_config.is_file():
            raise ValueError("run contains both config.json and config.txt; specify --space")
        else:
            raise FileNotFoundError(f"no config.json or config.txt found in {run_dir}")
    else:
        space = requested_space

    config_path = latent_config if space == "latent" else pixel_config
    if not config_path.is_file():
        raise FileNotFoundError(f"{space} run config is missing: {config_path}")
    if space == "latent":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("latent config.json must contain a JSON object")
        allowed_methods = LATENT_METHODS
    else:
        config = load_key_value_config(config_path)
        allowed_methods = PIXEL_METHODS
    method = str(config.get("method", ""))
    if method not in allowed_methods:
        raise ValueError(f"unsupported {space} method {method!r}; expected one of {allowed_methods}")
    expected_dim = LATENT_DIM if space == "latent" else PIXEL_DIM
    configured_dim = int(config.get("latent_dim", expected_dim))
    if configured_dim != expected_dim:
        raise ValueError(f"{space} config has latent_dim={configured_dim}, expected {expected_dim}")
    return RunSpec(run_dir, space, method, config_path, config)


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_step(path: Path, fallback: int | None = None) -> int | None:
    match = re.search(r"(?:step_|model_step_)(\d+)", path.name)
    return int(match.group(1)) if match else fallback


def pixel_checkpoint_weight_kind(path: Path, config: Mapping[str, Any]) -> tuple[str | None, int | None]:
    if path.name == "model_raw_final.pth":
        return "raw", int(config.get("steps", 0))
    step = checkpoint_step(path, int(config.get("steps", 0)) if path.name == "model_final.pth" else None)
    if path.name == "model_final.pth" or step is not None:
        ema_decay = float(config.get("ema_decay", 0.0))
        warmup = int(config.get("ema_warmup_steps", 0))
        return ("ema" if ema_decay > 0 and step is not None and step >= warmup else "raw"), step
    return None, step


def latent_checkpoint_capabilities(path: Path) -> tuple[set[str], str, int | None]:
    if path.name == "model_raw_final.pt":
        return {"raw"}, "direct", None
    if path.name == "model_ema_final.pt":
        return {"ema"}, "direct", None
    payload = torch_load(path)
    if isinstance(payload, Mapping) and payload.get("format") == LATENT_CHECKPOINT_FORMAT:
        capabilities = {"raw"}
        if payload.get("ema") is not None and bool(payload.get("ema_started", False)):
            capabilities.add("ema")
        return capabilities, "full", int(payload.get("completed_step", 0))
    return set(), "direct", checkpoint_step(path)


def resolve_explicit_checkpoint(spec: RunSpec, path: Path, requested_weights: str) -> CheckpointSelection:
    candidate = path if path.is_absolute() else spec.run_dir / path
    if not candidate.is_file() and not path.is_absolute():
        candidate = path.resolve()
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    if spec.space == "pixel":
        inferred, step = pixel_checkpoint_weight_kind(candidate, spec.config)
        if inferred is None:
            if requested_weights == "auto":
                raise ValueError("cannot infer raw/EMA semantics for this pixel checkpoint; specify --weights")
            inferred = requested_weights
        if requested_weights != "auto" and inferred != requested_weights:
            raise ValueError(
                f"checkpoint {candidate.name} contains {inferred} weights, not requested {requested_weights}"
            )
        return CheckpointSelection(
            candidate,
            requested_weights,
            inferred,
            "direct",
            step,
            "explicit pixel checkpoint",
        )

    capabilities, container, step = latent_checkpoint_capabilities(candidate)
    if not capabilities:
        if requested_weights == "auto":
            raise ValueError("cannot infer raw/EMA semantics for this latent state dict; specify --weights")
        capabilities = {requested_weights}
    if requested_weights == "auto":
        resolved = "ema" if "ema" in capabilities else "raw"
    elif requested_weights in capabilities:
        resolved = requested_weights
    else:
        raise ValueError(
            f"checkpoint {candidate.name} supports {sorted(capabilities)}, not requested {requested_weights}"
        )
    return CheckpointSelection(
        candidate,
        requested_weights,
        resolved,
        f"{container}:{resolved}" if container == "full" else "direct",
        step,
        "explicit latent checkpoint",
    )


def periodic_checkpoints(spec: RunSpec) -> list[Path]:
    suffix = "*.pt" if spec.space == "latent" else "*.pth"
    directory = spec.run_dir / "checkpoints"
    return sorted(
        directory.glob(suffix),
        key=lambda path: (checkpoint_step(path, -1) or -1, path.name),
        reverse=True,
    ) if directory.is_dir() else []


def resolve_default_checkpoint(spec: RunSpec, requested_weights: str) -> CheckpointSelection:
    if spec.space == "pixel":
        ordered = [spec.run_dir / "model_final.pth", spec.run_dir / "model_raw_final.pth"]
        ordered.extend(periodic_checkpoints(spec))
        available: list[tuple[Path, str, int | None]] = []
        for path in ordered:
            if path.is_file():
                kind, step = pixel_checkpoint_weight_kind(path, spec.config)
                if kind is not None:
                    available.append((path.resolve(), kind, step))
        if requested_weights == "auto":
            selected = next((item for item in available if item[0].name == "model_final.pth"), None)
            selected = selected or next((item for item in available if item[1] == "ema"), None)
            selected = selected or (available[0] if available else None)
        else:
            selected = next((item for item in available if item[1] == requested_weights), None)
        if selected is None:
            raise FileNotFoundError(
                f"no {requested_weights} pixel checkpoint is available in {spec.run_dir}"
            )
        path, resolved, step = selected
        return CheckpointSelection(
            path,
            requested_weights,
            resolved,
            "direct",
            step,
            "selected from pixel trainer filenames and EMA warmup",
        )

    final_candidates = {
        "ema": spec.run_dir / "model_ema_final.pt",
        "raw": spec.run_dir / "model_raw_final.pt",
    }
    final_full = spec.run_dir / "checkpoint_final.pt"
    final_capabilities = (
        latent_checkpoint_capabilities(final_full)[0] if final_full.is_file() else set()
    )
    if (
        requested_weights in {"ema", "raw"}
        and final_candidates[requested_weights].is_file()
        and (requested_weights != "ema" or "ema" in final_capabilities)
    ):
        return CheckpointSelection(
            final_candidates[requested_weights].resolve(),
            requested_weights,
            requested_weights,
            "direct",
            None,
            "selected latent final state dict",
        )
    if requested_weights == "auto":
        for kind in ("ema", "raw"):
            if final_candidates[kind].is_file() and (
                kind != "ema" or "ema" in final_capabilities
            ):
                return CheckpointSelection(
                    final_candidates[kind].resolve(),
                    requested_weights,
                    kind,
                    "direct",
                    None,
                    "auto-selected latent final state dict",
                )

    full_candidates = [spec.run_dir / "checkpoint_final.pt", *periodic_checkpoints(spec)]
    for path in full_candidates:
        if not path.is_file():
            continue
        capabilities, container, step = latent_checkpoint_capabilities(path)
        resolved = (
            "ema"
            if requested_weights == "auto" and "ema" in capabilities
            else "raw"
            if requested_weights == "auto" and "raw" in capabilities
            else requested_weights
        )
        if resolved in capabilities:
            return CheckpointSelection(
                path.resolve(),
                requested_weights,
                resolved,
                f"{container}:{resolved}",
                step,
                "selected latent full checkpoint",
            )
    raise FileNotFoundError(f"no {requested_weights} latent checkpoint is available in {spec.run_dir}")


def resolve_checkpoint(
    spec: RunSpec,
    checkpoint: Path | None,
    requested_weights: str,
) -> CheckpointSelection:
    return (
        resolve_explicit_checkpoint(spec, checkpoint, requested_weights)
        if checkpoint is not None
        else resolve_default_checkpoint(spec, requested_weights)
    )


def strip_ema_wrapper(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    module_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.startswith("module.") and torch.is_tensor(value)
    }
    if not module_state:
        raise ValueError("EMA checkpoint does not contain AveragedModel module.* weights")
    return module_state


def state_dict_from_checkpoint(selection: CheckpointSelection) -> dict[str, torch.Tensor]:
    payload = torch_load(selection.path)
    if selection.container == "direct":
        state = payload
    elif selection.container == "full:raw":
        state = payload.get("model") if isinstance(payload, Mapping) else None
    elif selection.container == "full:ema":
        ema_state = payload.get("ema") if isinstance(payload, Mapping) else None
        if not isinstance(ema_state, Mapping):
            raise ValueError("full checkpoint does not contain EMA weights")
        return strip_ema_wrapper(ema_state)
    else:
        raise ValueError(f"unknown checkpoint container: {selection.container}")
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError(f"checkpoint does not contain a direct model state_dict: {selection.path}")
    return dict(state)


def state_dicts_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def resolve_training_checkpoint(
    spec: RunSpec,
    selection: CheckpointSelection,
) -> tuple[Path | None, str | None]:
    """Resolve the full checkpoint attesting a latent inference state dict."""

    if spec.space != "latent":
        return None, None
    path = (
        selection.path
        if selection.container.startswith("full:")
        else spec.run_dir / "checkpoint_final.pt"
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "latent generation requires the full training checkpoint that attests "
            f"the selected state dict: {path}"
        )
    return path, sha256_file(path)


def validate_checkpoint_latent_run(
    plan: GenerationPlan,
    selected_state: Mapping[str, torch.Tensor],
) -> None:
    """Tie latent weights, training signature, normalization, and prior to a run."""

    if plan.run.space != "latent":
        return
    if plan.training_checkpoint_path is None:
        raise ValueError("latent generation plan lacks a full training-checkpoint attestation")
    payload = torch_load(plan.training_checkpoint_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != LATENT_CHECKPOINT_FORMAT
        or payload.get("version") != 1
    ):
        raise ValueError("full latent training checkpoint is malformed")
    saved_run_dir = payload.get("run_dir")
    if not isinstance(saved_run_dir, str) or Path(saved_run_dir).resolve() != plan.run.run_dir:
        raise ValueError("full checkpoint is not tied to the selected run directory")
    signature = payload.get("training_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("full checkpoint lacks a training signature")

    configured_endpoint = str(plan.run.config.get("endpoint_parameterization", "absolute"))
    saved_endpoint = str(signature.get("endpoint_parameterization", "absolute"))
    expected_prior_name = (
        "N(0,I)" if plan.latent_prior.mode == "standard" else plan.latent_prior.mode
    )
    structural_pairs = {
        "method": (signature.get("method"), plan.run.method),
        "endpoint_parameterization": (saved_endpoint, configured_endpoint),
        "prior": (signature.get("prior", "N(0,I)"), expected_prior_name),
    }
    structural_mismatches = [
        name for name, (saved, configured) in structural_pairs.items() if saved != configured
    ]
    if structural_mismatches:
        raise ValueError(
            "full checkpoint structural signature disagrees with run config for "
            f"{structural_mismatches}"
        )

    # Every training-signature field also present in config must agree.  Older
    # checkpoints omit later provenance additions, but cannot contradict them.
    recipe_mismatches = sorted(
        key
        for key, saved in signature.items()
        if key in plan.run.config
        and key not in {"prior", "endpoint_parameterization", "latent_prior"}
        and saved != plan.run.config[key]
    )
    resolved_indices = {
        "train_start_index": plan.run.config.get("train_start_resolved"),
        "train_end_index": plan.run.config.get("train_end_resolved"),
    }
    recipe_mismatches.extend(
        key
        for key, configured in resolved_indices.items()
        if configured is not None and signature.get(key) != configured
    )
    if recipe_mismatches:
        raise ValueError(
            "full checkpoint training signature disagrees with run config for "
            f"{sorted(set(recipe_mismatches))}"
        )

    saved_normalization = LatentAffineNormalization.from_metadata(
        payload.get("latent_normalization")
    )
    if saved_normalization.metadata() != plan.latent_normalization.metadata():
        raise ValueError("checkpoint latent normalization disagrees with run config")

    saved_payload = payload.get("latent_prior")
    if plan.latent_prior.mode == "standard":
        if saved_payload is not None:
            saved_prior = LatentGaussianPrior.from_payload(saved_payload)
            if saved_prior.mode != "standard":
                raise ValueError("checkpoint embeds a nonstandard prior but config requests standard")
    else:
        if not isinstance(saved_payload, Mapping):
            raise ValueError("full checkpoint lacks fitted latent-prior state")
        saved_prior = LatentGaussianPrior.from_payload(saved_payload)
        if not saved_prior.same_definition(plan.latent_prior):
            raise ValueError("checkpoint fitted latent prior disagrees with the run sidecar")
        if not plan.latent_prior.matches_metadata(signature.get("latent_prior", {})):
            raise ValueError("checkpoint training signature disagrees with its fitted prior")

    if plan.checkpoint.container == "direct":
        if plan.checkpoint.resolved_weights == "raw":
            attested_state = payload.get("model")
        elif plan.checkpoint.resolved_weights == "ema":
            if not bool(payload.get("ema_started", False)):
                raise ValueError("selected direct EMA file was saved before EMA became active")
            ema_state = payload.get("ema")
            attested_state = strip_ema_wrapper(ema_state) if isinstance(ema_state, Mapping) else None
        else:
            attested_state = None
        if not isinstance(attested_state, Mapping) or not state_dicts_equal(
            selected_state, attested_state
        ):
            raise ValueError(
                "selected direct latent state dict does not match its full training checkpoint"
            )


class LatentEndpointModel(nn.Module):
    """Inference-only counterpart of celeba_latent_flow.LatentEndpointModel."""

    def __init__(self, flow_map: nn.Module) -> None:
        super().__init__()
        self.flow_map = flow_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self.flow_map(x, start, torch.ones_like(start))


def build_model(spec: RunSpec, device: torch.device) -> nn.Module:
    config = spec.config
    if spec.space == "pixel":
        from src.model import CelebaUNetGenerator, TransitionConditionedCelebaUNet

        constructor = CelebaUNetGenerator if spec.method == "drift" else TransitionConditionedCelebaUNet
        return constructor(
            int(config.get("latent_dim", PIXEL_DIM)),
            int(config["base_channels"]),
        ).to(device)

    from src.celeba_latent import TransitionConditionedLatentUNet

    flow_map = TransitionConditionedLatentUNet(
        base_channels=int(config["base_channels"]),
        time_embedding_dim=int(config["time_embedding_dim"]),
        condition_dim=(None if config.get("condition_dim") is None else int(config["condition_dim"])),
        zero_init_endpoint=bool(config.get("zero_init_endpoint", True)),
        endpoint_parameterization=str(config.get("endpoint_parameterization", "absolute")),
    )
    model: nn.Module = flow_map if spec.method == "flow" else LatentEndpointModel(flow_map)
    return model.to(device)


def resolve_path_from_config(value: Any, run_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    return repo_candidate if repo_candidate.exists() else (run_dir / path).resolve()


def cache_scaling_factor(spec: RunSpec) -> float | None:
    configured = spec.config.get("vae_scaling_factor")
    if configured is not None:
        return float(configured)
    provenance = spec.run_dir / "provenance.json"
    if provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        value = payload.get("cache", {}).get("metadata", {}).get("scaling_factor")
        if value is not None:
            return float(value)
    cache_value = spec.config.get("cache")
    if cache_value is not None:
        cache_path = resolve_path_from_config(cache_value, spec.run_dir)
        if cache_path.is_file():
            payload = torch_load(cache_path)
            if isinstance(payload, Mapping):
                value = payload.get("metadata", {}).get("scaling_factor")
                if value is not None:
                    return float(value)
    return None


def resolve_latent_normalization(spec: RunSpec) -> LatentAffineNormalization:
    """Load the training coordinate transform, defaulting old runs to identity."""

    if spec.space != "latent":
        return LatentAffineNormalization.identity()
    configured_mode = spec.config.get("latent_normalization", "none")
    if isinstance(configured_mode, Mapping):
        metadata = dict(configured_mode)
        configured_mode = metadata.get("mode", "none")
    else:
        metadata = spec.config.get("latent_normalization_stats")
    mode = str(configured_mode)
    if mode == "none":
        if isinstance(metadata, Mapping) and str(metadata.get("mode", "none")) != "none":
            raise ValueError("latent normalization mode and saved statistics disagree")
        return LatentAffineNormalization.identity()
    if mode != "channel":
        raise ValueError(f"unsupported latent normalization mode in config: {mode!r}")
    if not isinstance(metadata, Mapping):
        raise ValueError("channel-normalized latent run lacks latent_normalization_stats")
    metadata = {**metadata, "mode": mode}
    return LatentAffineNormalization.from_metadata(metadata)


def resolve_latent_prior(
    spec: RunSpec,
) -> tuple[LatentGaussianPrior, Path | None, str | None]:
    """Load a fitted prior sidecar, while defaulting every old run to N(0,I)."""

    if spec.space != "latent":
        return LatentGaussianPrior.standard(), None, None
    configured = str(spec.config.get("latent_prior", "standard"))
    if configured == "standard":
        return LatentGaussianPrior.standard(), None, None
    if configured != "train-full-gaussian":
        raise ValueError(f"unsupported latent prior in config: {configured!r}")
    if str(spec.config.get("latent_normalization", "none")) != "none":
        raise ValueError("train-full-gaussian prior requires raw latent coordinates")
    sidecar_value = spec.config.get("latent_prior_sidecar", LATENT_PRIOR_FILENAME)
    sidecar = Path(str(sidecar_value))
    sidecar = sidecar.resolve() if sidecar.is_absolute() else (spec.run_dir / sidecar).resolve()
    if not sidecar.is_file():
        raise FileNotFoundError(f"fitted latent-prior sidecar is missing: {sidecar}")
    expected_hash = spec.config.get("latent_prior_sidecar_sha256")
    if not isinstance(expected_hash, str):
        raise ValueError("fitted latent-prior config lacks sidecar SHA-256")
    actual_hash = sha256_file(sidecar)
    if actual_hash != expected_hash:
        raise ValueError("fitted latent-prior sidecar SHA-256 disagrees with config")
    prior = LatentGaussianPrior.from_payload(torch_load(sidecar))
    if prior.mode != "train_full_gaussian":
        raise ValueError("fitted latent-prior sidecar has the wrong mode")
    configured_stats = spec.config.get("latent_prior_stats")
    if not isinstance(configured_stats, Mapping) or not prior.matches_metadata(configured_stats):
        raise ValueError("fitted latent-prior sidecar disagrees with config metadata")
    if (
        int(spec.config.get("train_start_resolved", prior.train_start_index))
        != prior.train_start_index
        or int(spec.config.get("train_end_resolved", prior.train_end_index))
        != prior.train_end_index
    ):
        raise ValueError("fitted latent-prior training split disagrees with config")
    return prior, sidecar, actual_hash


def resolve_vae(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> dict[str, Any] | None:
    if spec.space != "latent":
        return None
    configured_path = args.vae_model or spec.config.get("vae_model")
    if configured_path is None:
        raise ValueError("latent config does not specify vae_model; pass --vae-model")
    path = (
        Path(configured_path).resolve()
        if Path(configured_path).is_absolute()
        else resolve_path_from_config(configured_path, spec.run_dir)
    )
    if not path.is_dir():
        raise FileNotFoundError(f"local VAE directory does not exist: {path}")
    dtype_name = args.vae_dtype
    if dtype_name == "auto":
        configured_dtype = str(spec.config.get("vae_dtype", "auto"))
        dtype_name = configured_dtype if configured_dtype != "auto" else (
            "float16" if device.type == "cuda" else "float32"
        )
    if dtype_name in {"float16", "bfloat16"} and device.type == "cpu":
        raise ValueError(f"{dtype_name} VAE decoding on CPU is unsupported; use --vae-dtype float32")
    expected_hashes = spec.config.get("cache_vae_files_sha256")
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping) or not expected_hashes:
            raise ValueError("run config has malformed cache VAE file hashes")
        actual_hashes = directory_hashes(path)
        if dict(expected_hashes) != actual_hashes:
            raise ValueError("decoder VAE files differ from the encoder attested by the cache")
    return {
        "path": path,
        "scaling_factor": cache_scaling_factor(spec),
        "dtype": dtype_name,
        "cache_encoder_files_sha256": (
            None if expected_hashes is None else dict(expected_hashes)
        ),
        "cache_attestation": (
            "legacy_cache_unavailable" if expected_hashes is None else "verified"
        ),
    }


def build_decoder(
    spec: RunSpec,
    args: argparse.Namespace,
    device: torch.device,
    vae: dict[str, Any] | None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    if spec.space == "pixel":
        def decode_pixels(flat: torch.Tensor) -> torch.Tensor:
            return flat.reshape(flat.shape[0], 3, 64, 64)

        return decode_pixels, {"kind": "pixel_reshape"}

    if vae is None:
        raise ValueError("latent decoding requires a resolved VAE")
    from src.celeba_latent import CelebALatentCodec

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[vae["dtype"]]
    codec = CelebALatentCodec.from_pretrained(
        str(vae["path"]),
        scaling_factor=vae["scaling_factor"],
        device=device,
        torch_dtype=dtype,
        local_files_only=True,
    )
    normalization = resolve_latent_normalization(spec)

    def decode_latents(flat: torch.Tensor) -> torch.Tensor:
        flat = normalization.denormalize(flat)
        chunks = [
            codec.decode_flat(chunk.to(device), clamp=False)
            for chunk in flat.split(args.decode_batch_size)
        ]
        return torch.cat(chunks)

    return decode_latents, {
        "kind": "local_sd_vae",
        "path": str(vae["path"]),
        "scaling_factor": codec.scaling_factor,
        "dtype": vae["dtype"],
        "latent_normalization": normalization.metadata(),
    }


def sample_model(
    model: nn.Module,
    spec: RunSpec,
    prior: torch.Tensor,
    *,
    nfe: int,
    time_power: float,
) -> torch.Tensor:
    if spec.method == "drift":
        samples = prior
        for _ in range(nfe):
            samples = model(samples)
        return samples
    times = torch.linspace(0, 1, nfe + 1, device=prior.device, dtype=prior.dtype).pow(time_power)
    samples = prior
    for start, end in zip(times[:-1], times[1:]):
        samples = model(samples, start, end)
    return samples


def seeded_prior(start: int, count: int, dimension: int, seed: int) -> torch.Tensor:
    """Create sample-index-seeded priors, invariant to generation batch size."""

    modulus = 2**63 - 1
    rows = []
    for sample_index in range(start, start + count):
        generator = torch.Generator(device="cpu").manual_seed((seed + sample_index) % modulus)
        rows.append(torch.randn(dimension, generator=generator, dtype=torch.float32))
    return torch.stack(rows)


def seeded_latent_prior(
    start: int,
    count: int,
    seed: int,
    prior: LatentGaussianPrior,
) -> torch.Tensor:
    """Create index-seeded latent draws with a rowwise fitted-prior transform."""

    standard = seeded_prior(start, count, LATENT_DIM, seed)
    sampler = prior.sampler(device="cpu", dtype=torch.float32)
    return sampler.transform_standard(standard, rowwise=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes(spec: RunSpec) -> dict[str, str]:
    paths = [Path(__file__).resolve(), REPO_ROOT / "src" / "wgf.py"]
    if spec.space == "pixel":
        paths.extend((REPO_ROOT / "celeba_lagrangian_flow.py", REPO_ROOT / "src" / "model.py"))
    else:
        paths.extend((REPO_ROOT / "celeba_latent_flow.py", REPO_ROOT / "src" / "celeba_latent.py"))
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        # ``is_file`` follows file symlinks.  Hash their target bytes as well:
        # Hugging Face model snapshots commonly expose weights this way.
        if path.is_file()
    }


def default_output_dir(
    spec: RunSpec,
    checkpoint: CheckpointSelection,
    *,
    nfe: int,
    seed: int,
    count: int,
    batch_size: int,
    decode_batch_size: int,
) -> Path:
    name = (
        f"{checkpoint.path.stem}__{checkpoint.resolved_weights}__"
        f"nfe{nfe:03d}__seed{seed}__n{count}__b{batch_size}__db{decode_batch_size}"
    )
    return spec.run_dir / "generated" / name


def validate_output_target(output_dir: Path, run_dir: Path, overwrite: bool) -> None:
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir.is_symlink():
        raise ValueError(f"refusing symlink output directory: {output_dir}")
    forbidden = {Path("/").resolve(), REPO_ROOT.resolve(), run_dir.resolve()}
    if output_dir.resolve() in forbidden:
        raise ValueError(f"refusing unsafe output directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}; pass --overwrite")


def make_plan(args: argparse.Namespace) -> GenerationPlan:
    spec = load_run_spec(args.run, args.space)
    if spec.method == "drift" and args.nfe != 1 and not args.repeat_drift:
        raise ValueError(
            "drift is an endpoint model and normally requires --nfe 1; "
            "pass --repeat-drift only for the explicit off-distribution diagnostic"
        )
    selection = resolve_checkpoint(spec, args.checkpoint, args.weights)
    time_power = float(
        args.time_power
        if args.time_power is not None
        else spec.config.get("sample_time_power", 1.0)
    )
    if not math.isfinite(time_power) or time_power <= 0:
        raise ValueError(f"invalid sample_time_power in config: {time_power}")
    device = resolve_device(args.device)
    output_dir = (
        Path(os.path.abspath(args.output_dir))
        if args.output_dir is not None
        else default_output_dir(
            spec,
            selection,
            nfe=args.nfe,
            seed=args.seed,
            count=args.num_samples,
            batch_size=args.batch_size,
            decode_batch_size=args.decode_batch_size,
        ).resolve()
    )
    validate_output_target(output_dir, spec.run_dir, args.overwrite)
    vae = resolve_vae(spec, args, device)
    latent_normalization = resolve_latent_normalization(spec)
    latent_prior, latent_prior_path, latent_prior_sha256 = resolve_latent_prior(spec)
    training_checkpoint_path, training_checkpoint_sha256 = resolve_training_checkpoint(
        spec, selection
    )
    return GenerationPlan(
        spec,
        selection,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.decode_batch_size,
        args.nfe,
        args.seed,
        time_power,
        device,
        vae,
        latent_normalization,
        latent_prior,
        latent_prior_path,
        latent_prior_sha256,
        training_checkpoint_path,
        training_checkpoint_sha256,
    )


def plan_payload(plan: GenerationPlan) -> dict[str, Any]:
    payload = {
        "run_dir": str(plan.run.run_dir),
        "space": plan.run.space,
        "method": plan.run.method,
        "config": str(plan.run.config_path),
        "config_sha256": sha256_file(plan.run.config_path),
        "checkpoint": str(plan.checkpoint.path),
        "checkpoint_sha256": sha256_file(plan.checkpoint.path),
        "requested_weights": plan.checkpoint.requested_weights,
        "resolved_weights": plan.checkpoint.resolved_weights,
        "checkpoint_container": plan.checkpoint.container,
        "checkpoint_step": plan.checkpoint.completed_step,
        "checkpoint_reason": plan.checkpoint.reason,
        "output_dir": str(plan.output_dir),
        "num_samples": plan.num_samples,
        "batch_size": plan.batch_size,
        "decode_batch_size": plan.decode_batch_size,
        "nfe": plan.nfe,
        "time_power": plan.time_power,
        "seed": plan.seed,
        "device": str(plan.device),
        "vae": (
            None
            if plan.vae is None
            else {
                "path": str(plan.vae["path"]),
                "scaling_factor": plan.vae["scaling_factor"],
                "dtype": plan.vae["dtype"],
                "cache_attestation": plan.vae["cache_attestation"],
                "cache_encoder_files_sha256": plan.vae[
                    "cache_encoder_files_sha256"
                ],
            }
        ),
        "latent_normalization": plan.latent_normalization.metadata(),
    }
    if plan.training_checkpoint_path is not None:
        payload["training_checkpoint_attestation"] = str(plan.training_checkpoint_path)
        payload["training_checkpoint_attestation_sha256"] = plan.training_checkpoint_sha256
    if plan.latent_prior.mode != "standard":
        payload["latent_prior"] = plan.latent_prior.metadata()
        payload["latent_prior_sidecar"] = str(plan.latent_prior_path)
        payload["latent_prior_sidecar_sha256"] = plan.latent_prior_sha256
    return payload


def remove_exact_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def commit_staging_directory(staging: Path, target: Path, overwrite: bool) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"output already exists: {target}")
            backup = target.parent / f".{target.name}.backup-{os.getpid()}"
            if backup.exists():
                raise FileExistsError(f"stale output backup exists: {backup}")
            os.replace(target, backup)
        os.replace(staging, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        remove_exact_path(backup)


def atomic_json(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_png(image: torch.Tensor, path: Path) -> str:
    from PIL import Image

    if image.shape != (3, 64, 64):
        raise ValueError(f"decoder returned image shape {tuple(image.shape)}, expected (3, 64, 64)")
    if not bool(torch.isfinite(image).all()):
        raise FloatingPointError("decoder returned non-finite image values")
    pixels = image.detach().float().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8)
    array = pixels.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(array, mode="RGB").save(path, format="PNG", optimize=False, compress_level=6)
    return sha256_file(path)


@torch.inference_mode()
def run_generation(
    args: argparse.Namespace,
    *,
    model_builder: Callable[[RunSpec, torch.device], nn.Module] = build_model,
    decoder_builder: Callable[
        [RunSpec, argparse.Namespace, torch.device, dict[str, Any] | None],
        tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]],
    ] = build_decoder,
) -> dict[str, Any]:
    plan = make_plan(args)
    resolved_plan_payload = plan_payload(plan)
    source_hashes = code_hashes(plan.run)
    vae_hashes = directory_hashes(plan.vae["path"]) if plan.vae is not None else None
    model = model_builder(plan.run, plan.device)
    state_dict = state_dict_from_checkpoint(plan.checkpoint)
    validate_checkpoint_latent_run(plan, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    decoder, decoder_metadata = decoder_builder(plan.run, args, plan.device, plan.vae)

    plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=plan.output_dir.parent,
            prefix=f".{plan.output_dir.name}.staging-",
        )
    )
    file_records = []
    image_set_digest = hashlib.sha256()
    output_min = math.inf
    output_max = -math.inf
    clipped_low = 0
    clipped_high = 0
    value_count = 0
    filename_width = max(6, len(str(plan.num_samples - 1)))
    dimension = LATENT_DIM if plan.run.space == "latent" else PIXEL_DIM
    started_at = datetime.now(timezone.utc)
    try:
        for start in range(0, plan.num_samples, plan.batch_size):
            count = min(plan.batch_size, plan.num_samples - start)
            prior = (
                seeded_latent_prior(start, count, plan.seed, plan.latent_prior)
                if plan.run.space == "latent"
                else seeded_prior(start, count, dimension, plan.seed)
            )
            prior = prior.to(plan.device)
            generated = sample_model(
                model,
                plan.run,
                prior,
                nfe=plan.nfe,
                time_power=plan.time_power,
            )
            images = decoder(generated).detach().float().cpu()
            if images.shape != (count, 3, 64, 64):
                raise ValueError(
                    f"decoder returned batch shape {tuple(images.shape)}, expected {(count, 3, 64, 64)}"
                )
            output_min = min(output_min, float(images.min()))
            output_max = max(output_max, float(images.max()))
            clipped_low += int((images < 0).sum())
            clipped_high += int((images > 1).sum())
            value_count += images.numel()
            for offset, image in enumerate(images):
                index = start + offset
                filename = f"sample_{index:0{filename_width}d}.png"
                image_hash = save_png(image, staging / filename)
                sample_seed = (plan.seed + index) % (2**63 - 1)
                file_records.append(
                    {"index": index, "filename": filename, "seed": sample_seed, "sha256": image_hash}
                )
                image_set_digest.update(f"{filename}\0{image_hash}\n".encode("utf-8"))

        if resolved_plan_payload["config_sha256"] != sha256_file(plan.run.config_path):
            raise RuntimeError("run config changed during generation; refusing an ambiguous manifest")
        if resolved_plan_payload["checkpoint_sha256"] != sha256_file(plan.checkpoint.path):
            raise RuntimeError("checkpoint changed during generation; refusing an ambiguous manifest")
        if plan.training_checkpoint_path is not None and (
            plan.training_checkpoint_sha256 != sha256_file(plan.training_checkpoint_path)
        ):
            raise RuntimeError(
                "full training-checkpoint attestation changed during generation; refusing manifest"
            )
        if plan.latent_prior_path is not None and (
            plan.latent_prior_sha256 != sha256_file(plan.latent_prior_path)
        ):
            raise RuntimeError("latent-prior sidecar changed during generation; refusing manifest")
        if source_hashes != code_hashes(plan.run):
            raise RuntimeError("inference source changed during generation; refusing an ambiguous manifest")
        if vae_hashes is not None and vae_hashes != directory_hashes(plan.vae["path"]):
            raise RuntimeError(
                "VAE files changed during generation; refusing an ambiguous manifest"
            )
        manifest = {
            "format": "driftmodel.celeba_sample_directory",
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed": True,
            "plan": resolved_plan_payload,
            "config": plan.run.config,
            "provenance": {
                "python": sys.version,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": str(plan.device),
                "device_name": (
                    torch.cuda.get_device_name(plan.device)
                    if plan.device.type == "cuda"
                    else platform.processor() or "cpu"
                ),
                "source_sha256": source_hashes,
                "vae_files_sha256": vae_hashes,
            },
            "sampling": {
                "distribution": plan.latent_prior.metadata()["distribution"],
                "seed_scheme": "torch CPU generator seeded by (base_seed + sample_index) mod (2^63-1)",
                "base_seed": plan.seed,
                "nfe": plan.nfe,
                "time_power": plan.time_power,
                "batch_size": plan.batch_size,
                "decode_batch_size": plan.decode_batch_size,
                "prior_draws_batch_partition_invariant": True,
                "model_and_decoder_batch_size_invariant": False,
                "reproducibility_scope": (
                    "sample-index seeds are invariant; exact model/decoder outputs require the "
                    "recorded model and decode batch sizes, software stack, and device"
                ),
                "drift_reapplication_warning": (
                    "NFE > 1 repeatedly applies an endpoint model off its trained input distribution"
                    if plan.run.method == "drift" and plan.nfe > 1
                    else None
                ),
            },
            "decoder": decoder_metadata,
            "outputs": {
                "count": len(file_records),
                "naming": f"sample_{{index:0{filename_width}d}}.png",
                "image_set_sha256": image_set_digest.hexdigest(),
                "pre_clamp_min": output_min,
                "pre_clamp_max": output_max,
                "clipped_low_fraction": clipped_low / value_count,
                "clipped_high_fraction": clipped_high / value_count,
                "files": file_records,
            },
            "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        }
        if plan.latent_prior.mode != "standard":
            manifest["sampling"]["latent_prior"] = plan.latent_prior.metadata()
            manifest["sampling"]["latent_prior_sidecar_sha256"] = plan.latent_prior_sha256
        atomic_json(staging / "manifest.json", manifest)
        commit_staging_directory(staging, plan.output_dir, args.overwrite)
        return manifest
    except BaseException:
        if staging.exists():
            remove_exact_path(staging)
        raise


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.dry_run:
        plan = make_plan(args)
        state_dict = state_dict_from_checkpoint(plan.checkpoint)
        validate_checkpoint_latent_run(plan, state_dict)
        print(json.dumps({"dry_run": True, **plan_payload(plan)}, indent=2, sort_keys=True))
        return None
    manifest = run_generation(args)
    print(
        f"wrote {manifest['outputs']['count']} PNGs to {manifest['plan']['output_dir']}; "
        f"image_set_sha256={manifest['outputs']['image_set_sha256']}"
    )
    return manifest


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ImportError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
