#!/usr/bin/env python3
"""Continue the attested CelebA LSD Flow Map with a weak ramped GMD anchor.

This is the pre-registered D2 pilot from
``CELEBA_LSD_GMD_RAMPED_ANCHOR_PROTOCOL.md``.  It forks the exact raw
LR=0.003 pure-LSD state at update 30,000, including RAdam, EMA, data-stream
and rank-local RNG states, then performs updates 30,001--40,000 with

    L = L_mixed_LSD + lambda(k) L_terminal_GMD,

where lambda grows linearly from zero to ten percent of the post-hoc
equal-RMS value.  The implementation reuses the attested pure-LSD training
engine; process-local hooks change only the model wrapper, signature,
fork-loader, metadata, and per-update lambda schedule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# Import before torch so the baseline runner creates rank-local MIOpen caches.
import celeba_latent_lsd as lsd_runner
import torch
from torch import nn

from celeba_drift_lsd_hybrid import load_feature_encoder, parse_temperatures
from local_paths import REPO_ROOT
from src.celeba_drift_lsd_hybrid import DriftLSDHybridTrainingModel
from src.celeba_gmd_features_v2 import FEATURE_PROFILES, FEATURE_PROFILE_OFFICIAL_70


OBJECTIVE = "mixed_lsd_plus_weak_linearly_ramped_terminal_official_gmd_mae_v2"
RAMP_START_STEP = 30_000
RAMP_END_STEP = 40_000
LAMBDA_EQUAL_RMS_RAW_30K = 0.0024191577475160164
LAMBDA_TARGET = 0.00024191577475160165

PURE_RUN_DIR = (
    REPO_ROOT
    / "runs/celeba_latent_lsd/lsd_independent_std1_base256_seed42_lr003_screen10k_v1"
)
PURE_CHECKPOINT = PURE_RUN_DIR / "checkpoints/checkpoint_step_00030000.pt"
PURE_CHECKPOINT_SHA256 = (
    "70177ce81139684a7e06d2ceff575d0ea77ec16431e6a4b3ca198e9002f2a24a"
)
FEATURE_CHECKPOINT = (
    REPO_ROOT
    / "runs/celeba_official_drifting/"
    "feature_mae_v2_official_port_base64_seed42_30k_v1/feature_encoder_final.pt"
)
FEATURE_CHECKPOINT_SHA256 = (
    "c63d687e087ef2824eba35398949a1cdf96fe6a6e267dc34ae97e2a9952d98ed"
)
EXPECTED_RUN_DIR = (
    REPO_ROOT
    / "runs/celeba_latent_lsd/"
    "lsd_independent_std1_base256_seed42_lr003_ramped_gmd10pct_40k_v1"
)
PROTOCOL_PATH = REPO_ROOT / "CELEBA_LSD_GMD_RAMPED_ANCHOR_PROTOCOL.md"
PROTOCOL_SHA256 = "086f25df01344cc180c34dd521ffbb251733ffa9e882304660fc24623bd2ea28"
DIAGNOSTIC_REPORT = (
    REPO_ROOT
    / "runs/celeba_diagnostics/lsd_gmd_posthoc_gradients/"
    "raw_ema_30k_batches8_job756806_v1/report.json"
)
DIAGNOSTIC_REPORT_SHA256 = (
    "ee471370a9b38c4c8eaa24eb8e18412c8b462519b90694100031e42d667a710d"
)

ANCHOR_SOURCE_FILES = (
    'celeba_lsd_gmd_ramped_anchor.py',
    'celeba_latent_lsd.py',
    'celeba_drift_lsd_hybrid.py',
    'src/celeba_drift_lsd_hybrid.py',
    'src/celeba_gmd_features_v2.py',
    'src/official_drifting.py',
    'src/celeba_latent_lsd.py',
    'src/flow_map_self_distillation.py',
    'src/celeba_latent.py',
    'local_paths.py',
    'pyproject.toml',
)


@dataclass(frozen=True)
class LinearLambdaSchedule:
    """Lambda indexed by the completed update being optimized."""

    start_step: int
    end_step: int
    target: float

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("lambda ramp requires 0 <= start_step < end_step")
        if not math.isfinite(self.target) or self.target <= 0:
            raise ValueError("lambda target must be finite and positive")

    def value(self, completed_update: int) -> float:
        if completed_update <= self.start_step:
            return 0.0
        if completed_update > self.end_step:
            raise ValueError("completed update lies beyond the registered lambda ramp")
        fraction = (completed_update - self.start_step) / (
            self.end_step - self.start_step
        )
        return self.target * fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "linear_by_absolute_completed_update",
            "start_step": self.start_step,
            "end_step": self.end_step,
            "target": self.target,
            "formula": "target * (completed_update - start_step) / (end_step - start_step)",
        }


def sha256_file(path: str | Path) -> str:
    return lsd_runner.sha256_file(path)


def anchor_source_hashes() -> dict[str, str]:
    missing = [name for name in ANCHOR_SOURCE_FILES if not (REPO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ramped-anchor sources are missing: {missing}")
    return {name: sha256_file(REPO_ROOT / name) for name in ANCHOR_SOURCE_FILES}


def verify_regular_file(path: Path, expected_sha256: str | None, *, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be an existing regular non-symlink file: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def load_verified_ramped_resume(
    path: Path,
    expected_sha256: str,
    loader: Callable[[], int],
) -> int:
    """Load one D2 resume while binding both sides of the load to its pin."""

    verify_regular_file(
        path,
        expected_sha256,
        label="ramped-anchor resume checkpoint pre-load",
    )
    completed = loader()
    verify_regular_file(
        path,
        expected_sha256,
        label="ramped-anchor resume checkpoint post-load",
    )
    return completed


def load_verified_feature_encoder(
    path: Path,
    expected_sha256: str,
    loader: Callable[[], tuple[nn.Module, dict[str, Any]]],
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the frozen MAE while binding the loaded payload to its exact pin."""

    verify_regular_file(
        path,
        expected_sha256,
        label="MAE-v2 feature checkpoint pre-load",
    )
    encoder, metadata = loader()
    verify_regular_file(
        path,
        expected_sha256,
        label="MAE-v2 feature checkpoint post-load",
    )
    if metadata.get("sha256") != expected_sha256:
        raise ValueError("loaded MAE-v2 metadata SHA-256 differs from its exact pin")
    return encoder, metadata


def verify_source_manifest(path: Path, expected_sha256: str) -> dict[str, str]:
    verify_regular_file(path, expected_sha256, label="source manifest")
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split(None, 1)
        if len(pieces) != 2:
            raise ValueError("malformed source-manifest line")
        digest, raw_path = pieces
        raw_path = raw_path.strip()
        source = Path(raw_path)
        if source.is_absolute():
            try:
                name = str(source.resolve().relative_to(REPO_ROOT.resolve()))
            except ValueError as error:
                raise ValueError(f"source manifest escapes repository: {source}") from error
        else:
            name = str(source)
            source = REPO_ROOT / source
        if name in rows:
            raise ValueError(f"duplicate source-manifest entry: {name}")
        rows[name] = digest
        verify_regular_file(source, digest, label=f"source {name}")
    expected_names = set(ANCHOR_SOURCE_FILES)
    if set(rows) != expected_names:
        raise ValueError(
            "source-manifest inventory differs: "
            f"missing={sorted(expected_names-set(rows))}, extra={sorted(set(rows)-expected_names)}"
        )
    live = anchor_source_hashes()
    if rows != live:
        raise ValueError("source-manifest hashes differ from live ramped-anchor sources")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = lsd_runner.build_parser()
    parser.description = "CelebA LR=.003 LSD continuation with weak ramped terminal GMD"
    parser.set_defaults(
        output_root=REPO_ROOT / "runs/celeba_latent_lsd",
        run_label="lsd_lr003_ramped_gmd10pct_40k",
        steps=RAMP_END_STEP,
        lr=0.003,
        checkpoint_every=2_500,
        log_every=10,
    )
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-checkpoint-sha256", required=True)
    parser.add_argument(
        "--feature-profile", choices=FEATURE_PROFILES, default=FEATURE_PROFILE_OFFICIAL_70
    )
    parser.add_argument(
        "--gmd-temperatures",
        type=parse_temperatures,
        default=parse_temperatures("0.2,0.05,0.02"),
    )
    parser.add_argument("--lambda-ramp-start-step", type=int, default=RAMP_START_STEP)
    parser.add_argument("--lambda-ramp-end-step", type=int, default=RAMP_END_STEP)
    parser.add_argument("--gmd-lambda-target", type=float, default=LAMBDA_TARGET)
    parser.add_argument("--fork-from-pure-checkpoint", type=Path, default=None)
    parser.add_argument("--fork-checkpoint-sha256", default=None)
    parser.add_argument("--resume-sha256", default=None)
    parser.add_argument("--expected-source-manifest", type=Path, default=None)
    parser.add_argument("--expected-source-manifest-sha256", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    if args.smoke:
        args.device = "cpu"
        args.distributed_backend = "gloo"
        args.steps = min(args.steps, 2)
        args.base_channels = min(args.base_channels, 8)
        args.time_embedding_dim = min(args.time_embedding_dim, 8)
        args.condition_dim = 16 if args.condition_dim is None else min(args.condition_dim, 16)
        args.logvar_channels = min(args.logvar_channels, 8)
        args.use_bfloat16 = False
        args.pin_memory = False
        args.train_start_index = 0
        args.train_end_index = None
        args.log_every = 1
        args.checkpoint_every = 1
    return args


def validate_anchor_args(args: argparse.Namespace) -> LinearLambdaSchedule:
    if args.fork_from_pure_checkpoint is not None and args.resume is not None:
        raise ValueError("choose exactly one of --fork-from-pure-checkpoint and --resume")
    if not args.smoke and args.fork_from_pure_checkpoint is None and args.resume is None:
        raise ValueError("production ramped-anchor training must fork or resume")
    if not args.smoke and (
        getattr(args.device, "type", None) != "cuda"
        or args.distributed_backend != "nccl"
    ):
        raise ValueError("production ramped-anchor training requires CUDA/ROCm with NCCL/RCCL")
    schedule = LinearLambdaSchedule(
        args.lambda_ramp_start_step,
        args.lambda_ramp_end_step,
        args.gmd_lambda_target,
    )
    verify_regular_file(
        args.feature_checkpoint,
        args.feature_checkpoint_sha256,
        label="MAE-v2 feature checkpoint",
    )
    if args.fork_from_pure_checkpoint is not None:
        if args.fork_checkpoint_sha256 is None:
            raise ValueError("a pure fork requires --fork-checkpoint-sha256")
        verify_regular_file(
            args.fork_from_pure_checkpoint,
            args.fork_checkpoint_sha256,
            label="pure-LSD fork checkpoint",
        )
    if args.resume is not None:
        if args.resume_sha256 is None:
            raise ValueError("resume requires --resume-sha256")
        verify_regular_file(
            args.resume,
            args.resume_sha256,
            label="ramped-anchor resume checkpoint",
        )

    if args.expected_source_manifest is None:
        if not args.smoke:
            raise ValueError("production run requires --expected-source-manifest")
    else:
        if args.expected_source_manifest_sha256 is None:
            raise ValueError("source manifest requires its SHA-256")
        verify_source_manifest(
            args.expected_source_manifest, args.expected_source_manifest_sha256
        )

    if args.smoke:
        return schedule

    exact = {
        "steps": RAMP_END_STEP,
        "train_start_index": 5_000,
        "train_end_index": 30_000,
        "global_batch": 256,
        "world_size": 8,
        "local_batch": 32,
        "source_std": 1.0,
        "coupling": "independent",
        "gmd_teacher_checkpoint": None,
        "lsd_target": "current",
        "base_channels": 256,
        "time_embedding_dim": 32,
        "condition_dim": None,
        "sigma_data": 0.5,
        "zero_init_velocity": True,
        "logvar_channels": 128,
        "use_bfloat16": True,
        "pin_memory": True,
        "lr": 0.003,
        "lr_decay_steps": 35_000,
        "radam_beta1": 0.9,
        "radam_beta2": 0.999,
        "radam_eps": 1e-8,
        "grad_clip": 1.0,
        "ema_decay": 0.9999,
        "seed": 42,
        "deterministic": True,
        "verify_gradients": True,
        "checkpoint_every": 2_500,
        "feature_profile": FEATURE_PROFILE_OFFICIAL_70,
        "gmd_temperatures": (0.2, 0.05, 0.02),
        "lambda_ramp_start_step": RAMP_START_STEP,
        "lambda_ramp_end_step": RAMP_END_STEP,
        "gmd_lambda_target": LAMBDA_TARGET,
    }
    mismatches = {
        key: {"expected": expected, "observed": getattr(args, key)}
        for key, expected in exact.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(f"production ramped-anchor contract differs: {mismatches}")
    if args.run_dir is None or args.run_dir.resolve() != EXPECTED_RUN_DIR.resolve():
        raise ValueError(f"production run directory must be {EXPECTED_RUN_DIR}")
    if args.feature_checkpoint.resolve() != FEATURE_CHECKPOINT.resolve():
        raise ValueError("production run requires the pinned MAE-v2 checkpoint path")
    if args.feature_checkpoint_sha256 != FEATURE_CHECKPOINT_SHA256:
        raise ValueError("production run requires the pinned MAE-v2 checkpoint hash")
    verify_regular_file(PROTOCOL_PATH, PROTOCOL_SHA256, label="pre-registered protocol")
    verify_regular_file(
        DIAGNOSTIC_REPORT,
        DIAGNOSTIC_REPORT_SHA256,
        label="post-hoc gradient diagnostic",
    )
    if args.fork_from_pure_checkpoint is not None:
        if args.fork_from_pure_checkpoint.resolve() != PURE_CHECKPOINT.resolve():
            raise ValueError("production fork path is not the pinned raw 30k checkpoint")
        if args.fork_checkpoint_sha256 != PURE_CHECKPOINT_SHA256:
            raise ValueError("production fork hash is not the pinned raw 30k checkpoint")
    if args.resume is not None:
        try:
            args.resume.resolve().relative_to(EXPECTED_RUN_DIR.resolve())
        except ValueError as error:
            raise ValueError("production resume must come from the registered D2 run") from error
    return schedule


def augment_signature(
    base: Mapping[str, Any],
    args: argparse.Namespace,
    feature_metadata: Mapping[str, Any],
    schedule: LinearLambdaSchedule,
) -> dict[str, Any]:
    result = dict(base)
    base_version = result.get("signature_version")
    result.update(
        {
            "signature_version": 3,
            "base_lsd_signature_version": base_version,
            "objective": OBJECTIVE,
            "base_lsd_objective": "exact_75_25_diagonal_fm_and_lagrangian_self_distillation",
            "terminal_gmd_map": "X_theta(x0,0,1)",
            "terminal_gmd_lambda_schedule": schedule.as_dict(),
            "terminal_gmd_lambda_equal_rms_raw_30k": LAMBDA_EQUAL_RMS_RAW_30K,
            "terminal_gmd_target_fraction_of_equal_rms": 0.1,
            "terminal_gmd_temperatures": list(args.gmd_temperatures),
            "terminal_gmd_feature_checkpoint_sha256": feature_metadata["sha256"],
            "terminal_gmd_feature_profile": args.feature_profile,
            "terminal_gmd_feature_dtype": "float32",
            "terminal_gmd_formula": "src.official_drifting.summed_feature_drifting_loss",
            "terminal_gmd_positive_stop_gradient": True,
            "terminal_gmd_generated_feature_input_gradient": True,
            "terminal_gmd_particle_support": "rank_local_32",
            "terminal_gmd_scalar_statistics": (
                "global_ddp_average" if args.distributed else "local"
            ),
            "fork_origin_completed_step": schedule.start_step,
            "fork_origin_checkpoint_sha256": (
                args.fork_checkpoint_sha256
                if args.fork_checkpoint_sha256 is not None
                else PURE_CHECKPOINT_SHA256
            ),
            "fork_restores": ["raw_model", "RAdam", "EMA", "data_stream", "rng_by_rank"],
            "control": "pure_LSD_LR003_continuation_from_same_raw_30k_state",
            "pre_registered_protocol_sha256": PROTOCOL_SHA256,
            "gradient_diagnostic_report_sha256": DIAGNOSTIC_REPORT_SHA256,
            "ramped_anchor_source_files_sha256": anchor_source_hashes(),
            "expected_source_manifest_sha256": args.expected_source_manifest_sha256,
            "ddp_boundary": (
                "one outer forward containing mixed LSD JVP/current-stopgrad/logvar "
                "and terminal MAE-v2 GMD"
            ),
        }
    )
    return result


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def load_pure_fork_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_completed_step: int,
    expected_pure_signature: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: torch.optim.swa_utils.AveragedModel,
    stream: lsd_runner.LatentBatchStream,
    run_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Strictly import the old state into a new run without mutating its signature."""

    verify_regular_file(path, expected_sha256, label="pure-LSD fork checkpoint")
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    verify_regular_file(path, expected_sha256, label="pure-LSD fork checkpoint post-load")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("pure fork checkpoint is not a mapping")
    if checkpoint.get("format") != lsd_runner.CHECKPOINT_FORMAT:
        raise ValueError("pure fork checkpoint has the wrong format")
    if checkpoint.get("version") != lsd_runner.CHECKPOINT_VERSION:
        raise ValueError("pure fork checkpoint has the wrong version")
    if checkpoint.get("completed_step") != expected_completed_step:
        raise ValueError("pure fork checkpoint is not at the registered start step")
    if checkpoint.get("training_signature") != dict(expected_pure_signature):
        raise ValueError("pure fork checkpoint signature differs from the reconstructed baseline")
    saved_run_dir = checkpoint.get("run_dir")
    if not isinstance(saved_run_dir, str):
        raise ValueError("pure fork checkpoint lacks source run provenance")
    checkpoint_parent = path.resolve().parent
    inferred_source = (
        checkpoint_parent.parent
        if checkpoint_parent.name == "checkpoints"
        else checkpoint_parent
    )
    if Path(saved_run_dir).resolve() != inferred_source:
        raise ValueError("pure fork checkpoint source run provenance differs")
    if run_dir.resolve() == inferred_source:
        raise ValueError("ramped anchor must fork into a new immutable run directory")
    required = ("model", "optimizer", "ema", "data_stream", "rng_by_rank")
    if any(key not in checkpoint for key in required):
        raise ValueError("pure fork checkpoint lacks a required resumable state")
    if (
        not isinstance(checkpoint["rng_by_rank"], list)
        or len(checkpoint["rng_by_rank"]) != args.world_size
    ):
        raise ValueError("pure fork checkpoint does not contain one RNG state per rank")
    if not _all_finite(checkpoint["model"]) or not _all_finite(checkpoint["ema"]):
        raise FloatingPointError("pure fork model or EMA contains non-finite tensors")
    if not _all_finite(checkpoint["optimizer"]):
        raise FloatingPointError("pure fork optimizer contains non-finite tensors")

    lsd_runner.unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load_state_dict(checkpoint["ema"])
    stream.load_state_dict(checkpoint["data_stream"])
    lsd_runner.restore_rng_state(checkpoint["rng_by_rank"], args)
    if not _all_finite(lsd_runner.unwrap_model(model).state_dict()):
        raise FloatingPointError("loaded ramped-anchor raw model is non-finite")
    if not _all_finite(ema.state_dict()) or not _all_finite(optimizer.state_dict()):
        raise FloatingPointError("loaded ramped-anchor EMA or RAdam state is non-finite")
    return expected_completed_step


def archive_anchor_sources(run_dir: Path, suffix: str) -> dict[str, Any]:
    destination = run_dir / f"ramped_anchor_source_snapshot{suffix}"
    destination.mkdir(parents=True, exist_ok=False)
    hashes = anchor_source_hashes()
    for name in ANCHOR_SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    payload = {"created_at_utc": lsd_runner.utc_now(), "source_files_sha256": hashes}
    lsd_runner.atomic_json_dump(destination / "manifest.json", payload)
    return {"path": str(destination), **payload}


def write_anchor_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    feature_metadata: Mapping[str, Any],
    schedule: LinearLambdaSchedule,
    *,
    initial_fork: bool,
) -> None:
    suffix = (
        "_fork_" if initial_fork else "_resume_"
    ) + lsd_runner.utc_now().replace(":", "-")
    snapshot = archive_anchor_sources(run_dir, suffix)
    payload = {
        "format": "driftmodel.celeba_lsd_gmd_ramped_anchor_provenance",
        "version": 1,
        "created_at_utc": lsd_runner.utc_now(),
        "objective": OBJECTIVE,
        "run_role": "initial_exact_fork" if initial_fork else "exact_resume",
        "schedule": schedule.as_dict(),
        "feature_encoder": dict(feature_metadata),
        "pure_fork": {
            "path": str(args.fork_from_pure_checkpoint.resolve())
            if args.fork_from_pure_checkpoint is not None
            else str(PURE_CHECKPOINT),
            "sha256": args.fork_checkpoint_sha256 or PURE_CHECKPOINT_SHA256,
            "completed_step": schedule.start_step,
            "restored_state": ["raw_model", "RAdam", "EMA", "data_stream", "rng_by_rank"],
            "optimizer_reset": False,
        },
        "pre_registered_protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "gradient_diagnostic": {
            "path": str(DIAGNOSTIC_REPORT),
            "sha256": DIAGNOSTIC_REPORT_SHA256,
            "lambda_equal_rms_raw_30k": LAMBDA_EQUAL_RMS_RAW_30K,
            "chosen_fraction": 0.1,
        },
        "source_snapshot": snapshot,
        "source_manifest": {
            "path": str(args.expected_source_manifest.resolve())
            if args.expected_source_manifest is not None
            else None,
            "sha256": args.expected_source_manifest_sha256,
        },
        "claim_limit": "single-seed pre-registered development pilot; not a paper result",
    }
    lsd_runner.atomic_json_dump(run_dir / f"ramped_anchor_provenance{suffix}.json", payload)


def train(args: argparse.Namespace, schedule: LinearLambdaSchedule) -> Path:
    cache_latents, cache_metadata = lsd_runner.load_latent_cache(args.cache)
    _, _, _, _ = lsd_runner._select_training_latents(cache_latents, args)
    cache_digest = sha256_file(args.cache)
    feature_encoder, feature_metadata = load_verified_feature_encoder(
        args.feature_checkpoint,
        args.feature_checkpoint_sha256,
        lambda: load_feature_encoder(args, cache_metadata, cache_digest),
    )

    initial_fork = args.fork_from_pure_checkpoint is not None
    original_resume = args.resume
    if initial_fork:
        args.resume = args.fork_from_pure_checkpoint

    original_make_model = lsd_runner.make_training_model
    original_signature = lsd_runner.training_signature
    original_write_metadata = lsd_runner.write_metadata
    original_learning_rate = lsd_runner.learning_rate_at_step
    original_resolve_run_directory = lsd_runner.resolve_run_directory
    original_load_checkpoint = lsd_runner.load_checkpoint
    model_holder: dict[str, DriftLSDHybridTrainingModel] = {}
    pure_signature_holder: dict[str, Mapping[str, Any]] = {}

    def make_model_hook(hook_args: argparse.Namespace) -> DriftLSDHybridTrainingModel:
        pure = original_make_model(hook_args)
        model = DriftLSDHybridTrainingModel(
            pure.flow_map,
            feature_encoder,
            gmd_lambda=schedule.target,
            temperatures=hook_args.gmd_temperatures,
            distributed_statistics=hook_args.distributed,
        )
        model_holder["raw"] = model
        return model

    def signature_hook(*positional: Any, **keywords: Any) -> dict[str, Any]:
        base = original_signature(*positional, **keywords)
        pure_signature_holder["value"] = dict(base)
        return augment_signature(base, args, feature_metadata, schedule)

    def learning_rate_hook(step_index: int, hook_args: argparse.Namespace) -> float:
        completed_update = step_index + 1
        if "raw" not in model_holder:
            raise RuntimeError("lambda schedule advanced before model construction")
        value = schedule.value(completed_update)
        if value <= 0:
            raise RuntimeError("ramped GMD lambda must be positive on every forked update")
        model_holder["raw"].gmd_lambda = value
        return original_learning_rate(step_index, hook_args)

    def resolve_fork_run_directory(hook_args: argparse.Namespace) -> Path:
        if not initial_fork:
            return original_resolve_run_directory(hook_args)
        if hook_args.run_dir is None:
            raise ValueError("an initial pure-LSD fork requires --run-dir")

        def create() -> str:
            path = hook_args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"new ramped-anchor run directory is not empty: {path}")
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        return Path(lsd_runner.run_on_rank_zero(hook_args, create))

    def load_checkpoint_hook(
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: torch.optim.swa_utils.AveragedModel,
        stream: lsd_runner.LatentBatchStream,
        signature: Mapping[str, Any],
        run_dir: Path,
        args: argparse.Namespace,
    ) -> int:
        if not initial_fork:
            if args.resume_sha256 is None:
                raise ValueError("ramped-anchor resume loader requires its exact SHA-256")
            return load_verified_ramped_resume(
                path,
                args.resume_sha256,
                lambda: original_load_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    stream=stream,
                    signature=signature,
                    run_dir=run_dir,
                    args=args,
                ),
            )
        if path.resolve() != args.fork_from_pure_checkpoint.resolve():
            raise ValueError("fork loader received an unexpected checkpoint path")
        if "value" not in pure_signature_holder:
            raise RuntimeError("pure signature was not reconstructed before fork loading")
        return load_pure_fork_checkpoint(
            path,
            expected_sha256=args.fork_checkpoint_sha256,
            expected_completed_step=schedule.start_step,
            expected_pure_signature=pure_signature_holder["value"],
            model=model,
            optimizer=optimizer,
            ema=ema,
            stream=stream,
            run_dir=run_dir,
            args=args,
        )

    def write_metadata_hook(
        hook_args: argparse.Namespace,
        run_dir: Path,
        *positional: Any,
        **keywords: Any,
    ) -> None:
        if initial_fork:
            # The pure checkpoint is a causal state import into a *new* D2 run,
            # not a continuation inside the pure run. Emit the canonical
            # config.json/provenance.json/source_snapshot names required by the
            # standard evaluators; the D2 provenance below records the exact
            # fork path and hash.
            saved_resume = hook_args.resume
            hook_args.resume = None
            try:
                original_write_metadata(hook_args, run_dir, *positional, **keywords)
            finally:
                hook_args.resume = saved_resume
        else:
            original_write_metadata(hook_args, run_dir, *positional, **keywords)
        write_anchor_metadata(
            hook_args,
            run_dir,
            feature_metadata,
            schedule,
            initial_fork=initial_fork,
        )

    lsd_runner.make_training_model = make_model_hook
    lsd_runner.training_signature = signature_hook
    lsd_runner.write_metadata = write_metadata_hook
    lsd_runner.learning_rate_at_step = learning_rate_hook
    lsd_runner.resolve_run_directory = resolve_fork_run_directory
    lsd_runner.load_checkpoint = load_checkpoint_hook
    try:
        return lsd_runner.train(args)
    finally:
        lsd_runner.make_training_model = original_make_model
        lsd_runner.training_signature = original_signature
        lsd_runner.write_metadata = original_write_metadata
        lsd_runner.learning_rate_at_step = original_learning_rate
        lsd_runner.resolve_run_directory = original_resolve_run_directory
        lsd_runner.load_checkpoint = original_load_checkpoint
        args.resume = original_resume


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    try:
        lsd_runner.setup_distributed(args)
        schedule = validate_anchor_args(args)
        return train(args, schedule)
    finally:
        lsd_runner.cleanup_distributed(args)


if __name__ == "__main__":
    main()
